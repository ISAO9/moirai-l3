#!/usr/bin/env python
"""
15_phase_confusion.py -- GJI major revision round 2: is the forge_19 collapse a
loss of P detection, or a confusion between the P and S phases?
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Reviewer 1 (round 2) looked at the diagnosis figure and observed that, at
forge_19, the array model appears to place S probability where the catalogue
puts P, and P probability where the catalogue puts S:

    "Before 200 ms, a strong S detection is also visible in the probability
     map and seems that the S phase matches the cataloged P picks better.
     Similarly, there is a visible P at ~250 ms, overlapping with true S
     picks. This implies that the picker fails to distinguish P and S phases.
     The conclusion that 'the collapse is therefore a loss of P detection
     confidence' does not fully reveal the underlying issue."

That is a testable claim, and the test needs no new inference pass: script 10
already cached, for every held-out site and configuration, the sufficient
statistics of the picking rule -- per event and station, the maximum of each
phase channel and the SAMPLE INDEX at which it occurs, alongside the
catalogued P and S arrival samples. A pick is the global argmax of its
channel, so asking "where did the P channel's pick land?" is answered exactly
by comparing pr_time[P] with BOTH gt_time[P] and gt_time[S].

WHAT THIS SCRIPT DOES
---------------------
  1. ASSIGNMENT -- for every declared pick (channel max >= threshold) on an
     earthquake event, classify where it landed:
         on-phase  : within +/-tol of that channel's own catalogued arrival
         cross      : within +/-tol of the OTHER phase's catalogued arrival
         elsewhere : neither
     Reported per site and configuration, with the cross rate among picks
     that missed their own phase -- the number that settles the reviewer's
     reading.
  2. RESIDUALS -- pooled histograms of (pick time - own-phase arrival) and
     (pick time - other-phase arrival). A detector that merely fails to fire
     produces few picks with a flat residual; a detector that confuses the
     phases produces a residual that peaks at zero against the OTHER phase.
  3. FIGURE -- the two residual distributions per channel, for the
     out-of-distribution site and an in-distribution contrast, with the
     per-trace model as the control.

Everything runs from the cached statistics: no GPU, no AMBER, no checkpoints.
Build the caches first with script 10 (--cache) if logs/ does not hold them.

OUTPUTS
  logs/15_phase_confusion_<ckpt>.json   per site/config assignment counts
  logs/15_phase_confusion_<ckpt>.md     the same as a table to paste
  PDF/15_a_phase_residuals.pdf          residual distributions (+ .png 300dpi)

All figures: white background, English labels, legends clear of the data.

RUN (no GPU needed; caches from script 10 must be in logs/)
  python src/15_phase_confusion.py --stats
  python src/15_phase_confusion.py --figure
  python src/15_phase_confusion.py --stats --figure --sites forge_19 mseel_5h
  # self-test (synthetic, no caches required):
  python src/15_phase_confusion.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(path: str):
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")
DATA, EVALU = cfg.DATA, cfg.EVALU

PHASES = ((0, "P", 1, "S"), (1, "S", 0, "P"))   # (ch, name, other_ch, other)
FIG_SITES = ("forge_19", "mseel_5h")            # OOD site + in-dist contrast


# ----------------------------------------------------------------------------
# cache loading (same convention as script 10)
# ----------------------------------------------------------------------------
def load_caches(ckpt: str = "ema") -> dict:
    """{(site, method): stats} from every 10_cache_*.npz in logs/."""
    caches = {}
    for f in sorted(cfg.LOGS_DIR.glob(f"10_cache_*_{ckpt}*loso_*.npz")):
        stem = f.stem
        method = ("pertrace" if "_pertrace" in stem else
                  "shuffle" if "_shuf" in stem else "array")
        site = stem.split("_loso_")[-1]
        caches[(site, method)] = dict(np.load(f))
    return caches


# ----------------------------------------------------------------------------
# 1. assignment of declared picks
# ----------------------------------------------------------------------------
def assign(stats: dict, thr: float, tol: int) -> dict:
    """Where does each declared pick land, relative to BOTH catalogued phases?

    A pick exists on (event, station, channel) when that channel's maximum
    reaches `thr`; its time is the channel's argmax. It is counted once, in
    the first category it satisfies:
        on_phase  |t - gt_own|   <= tol
        cross     |t - gt_other| <= tol
        elsewhere otherwise
    Only earthquake events are considered (noise windows carry no arrivals).
    A pick is only classifiable against a phase the catalogue actually has on
    that station, so counts are reported alongside the number of picks that
    had an `other` arrival available to be confused with.
    """
    out = {}
    eq = ~stats["is_noise"]                               # (E,)
    for ch, name, och, oname in PHASES:
        gt = stats["gt_time"][:, ch]                      # (E,S)
        ogt = stats["gt_time"][:, och]
        mx = stats["pr_max"][:, ch]
        tm = stats["pr_time"][:, ch]

        picked = (mx >= thr) & eq[:, None]
        has_own = gt >= 0
        has_oth = ogt >= 0

        on_phase = picked & has_own & (np.abs(tm - gt) <= tol)
        cross = picked & ~on_phase & has_oth & (np.abs(tm - ogt) <= tol)
        elsewhere = picked & ~on_phase & ~cross

        n_pick = int(picked.sum())
        n_on = int(on_phase.sum())
        n_cross = int(cross.sum())
        n_else = int(elsewhere.sum())
        missed = n_cross + n_else                         # picks off own phase
        # how many of those could in principle have been confused
        n_missed_with_other = int((picked & ~on_phase & has_oth).sum())

        out[name] = dict(
            n_picks=n_pick,
            on_phase=n_on,
            cross=n_cross,
            elsewhere=n_else,
            frac_on_phase=n_on / n_pick if n_pick else 0.0,
            frac_cross=n_cross / n_pick if n_pick else 0.0,
            frac_elsewhere=n_else / n_pick if n_pick else 0.0,
            # the reviewer's question: of the picks that are NOT on their own
            # phase, how many sit on the other phase?
            frac_cross_of_missed=n_cross / missed if missed else 0.0,
            n_missed=missed,
            n_missed_with_other_arrival=n_missed_with_other,
            other_phase=oname,
        )
    return out


# ----------------------------------------------------------------------------
# 2. residuals against both phases
# ----------------------------------------------------------------------------
def residual_summary(stats: dict, thr: float, fs: float, tol: int) -> dict:
    """Where the picks sit, in milliseconds, on stations carrying BOTH arrivals.

    Restricting to stations with a catalogued P and S makes the own-phase and
    other-phase residuals directly comparable, and lets us ask the question the
    assignment counts leave open: of the picks that land on neither arrival,
    how many sit in the interval BETWEEN them? A detector that has locked onto
    a late coherent ridge puts them there; one that is merely firing at random
    does not.
    """
    eq = ~stats["is_noise"]
    out = {}
    for ch, name, och, oname in PHASES:
        gt = stats["gt_time"][:, ch]
        ogt = stats["gt_time"][:, och]
        tm = stats["pr_time"][:, ch]
        both = (gt >= 0) & (ogt >= 0) & (stats["pr_max"][:, ch] >= thr) \
            & eq[:, None]
        if not both.any():
            out[name] = dict(n=0)
            continue
        d_own = (tm[both] - gt[both]) / fs * 1000.0
        d_oth = (tm[both] - ogt[both]) / fs * 1000.0
        # strictly between the two catalogued arrivals, outside tolerance of both
        lo = np.minimum(gt[both], ogt[both]) + tol
        hi = np.maximum(gt[both], ogt[both]) - tol
        between = (tm[both] > lo) & (tm[both] < hi)
        q = lambda a: [float(np.percentile(a, p)) for p in (25, 50, 75)]  # noqa
        out[name] = dict(
            n=int(both.sum()),
            own_ms_q25_med_q75=q(d_own),
            other_ms_q25_med_q75=q(d_oth),
            frac_between_phases=float(between.mean()),
            other_phase=oname,
        )
    return out


def residuals(stats: dict, thr: float, fs: float) -> dict:
    """Pick-time residuals in ms against the own and the other phase.

    Returned per channel as two arrays; only declared picks on earthquake
    events, and only where the relevant catalogued arrival exists.
    """
    eq = ~stats["is_noise"]
    res = {}
    for ch, name, och, oname in PHASES:
        gt = stats["gt_time"][:, ch]
        ogt = stats["gt_time"][:, och]
        mx = stats["pr_max"][:, ch]
        tm = stats["pr_time"][:, ch]
        picked = (mx >= thr) & eq[:, None]
        own = (tm - gt)[picked & (gt >= 0)] / fs * 1000.0
        oth = (tm - ogt)[picked & (ogt >= 0)] / fs * 1000.0
        res[name] = dict(own=own.astype(np.float32),
                         other=oth.astype(np.float32),
                         own_label=name, other_label=oname)
    return res


# ----------------------------------------------------------------------------
# 3. figure
# ----------------------------------------------------------------------------
def make_figure(caches: dict, sites, thr: float, fs: float, out_pdf: Path):
    """Residual distributions against both phases, OOD site vs contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    rows = [(s, m) for s in sites for m in ("array", "pertrace")
            if (s, m) in caches]
    if not rows:
        raise SystemExit("[figure] no caches for the requested sites")

    fig, axes = plt.subplots(len(rows), 2, figsize=(10.0, 2.45 * len(rows)),
                             squeeze=False)
    span = 200.0                                    # ms shown either side
    bins = np.linspace(-span, span, 81)
    h_own = h_oth = None

    for i, (site, method) in enumerate(rows):
        res = residuals(caches[(site, method)], thr, fs)
        for j, (_, name, _, oname) in enumerate(PHASES):
            ax = axes[i][j]
            d = res[name]
            h_own = ax.hist(np.clip(d["own"], -span, span), bins=bins,
                            density=True, histtype="stepfilled", alpha=0.55,
                            color="#1f77b4",
                            label=f"relative to catalogued {name}")[2]
            h_oth = ax.hist(np.clip(d["other"], -span, span), bins=bins,
                            density=True, histtype="step", lw=1.6,
                            color="#d62728",
                            label=f"relative to catalogued {oname}")[2]
            ax.axvline(0.0, color="k", lw=0.9, ls="--")
            ax.set_xlim(-span, span)
            ax.set_title(f"{site} — {method} — {name} channel",
                         fontsize=10)
            ax.set_xlabel("pick time minus catalogued arrival (ms)")
            ax.set_ylabel("density")
            ax.grid(alpha=0.25, lw=0.5)

    # one legend for the whole figure, in the margin under the panels
    handles = [h_own[0], h_oth[0]]
    labels = ["relative to this channel's own catalogued phase",
              "relative to the other catalogued phase"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True,
               framealpha=0.95, edgecolor="0.8", fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Where the declared picks land, against both catalogued "
                 "phases", fontsize=12)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf} (+ .png)")


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def write_report(results: dict, ckpt: str):
    """results: {(site, method): per-channel dict} -> JSON + markdown table."""
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    jpath = cfg.LOGS_DIR / f"15_phase_confusion_{ckpt}.json"
    # JSON keys must be strings; keep site and config as separate fields too
    serialisable = {f"{site}__{method}": dict(site=site, config=method, **per_ch)
                    for (site, method), per_ch in results.items()}
    jpath.write_text(json.dumps(serialisable, indent=2))

    md = ["| Site | Config | Channel | picks | on-phase | cross-phase | "
          "elsewhere | cross / missed | median offset (ms) | between phases |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (site, method), per_ch in sorted(results.items(), key=lambda kv: kv[0]):
        for name in ("P", "S"):
            d = per_ch[name]
            r = d.get("residuals", {})
            med = (f"{r['own_ms_q25_med_q75'][1]:+.1f}"
                   if r.get("n") else "-")
            btw = (f"{r['frac_between_phases']:.3f}"
                   if r.get("n") else "-")
            md.append(
                f"| {site} | {method} | {name} | {d['n_picks']} | "
                f"{d['frac_on_phase']:.3f} | {d['frac_cross']:.3f} | "
                f"{d['frac_elsewhere']:.3f} | "
                f"{d['frac_cross_of_missed']:.3f} | {med} | {btw} |")
    mpath = cfg.LOGS_DIR / f"15_phase_confusion_{ckpt}.md"
    mpath.write_text("\n".join(md) + "\n")
    print(f"[stats] wrote {jpath} and {mpath}")
    print("\n".join(md))


def run_stats(ckpt: str, thr: float, tol: int) -> dict:
    caches = load_caches(ckpt)
    if not caches:
        raise SystemExit(f"[stats] no 10_cache_*_{ckpt}*loso_*.npz in "
                         f"{cfg.LOGS_DIR} -- build them with script 10 --cache")
    fs = float(getattr(DATA, "FS", 2000.0))
    results = {}
    for key, stats in sorted(caches.items()):
        res = assign(stats, thr, tol)
        summ = residual_summary(stats, thr, fs, tol)
        for name in ("P", "S"):
            res[name]["residuals"] = summ.get(name, {})
        results[key] = res
        site, method = key
        p, r = res["P"], res["P"].get("residuals", {})
        med = r["own_ms_q25_med_q75"][1] if r.get("n") else float("nan")
        print(f"[stats] {site:<16s} {method:<9s} P: {p['n_picks']:>5d} picks, "
              f"on-phase {p['frac_on_phase']:.3f}, "
              f"cross {p['frac_cross']:.3f}, "
              f"median offset {med:+.1f} ms, "
              f"between phases {r.get('frac_between_phases', float('nan')):.3f}")
    write_report(results, ckpt)
    return results


# ----------------------------------------------------------------------------
# self-test (synthetic; no caches, no AMBER, no checkpoints)
# ----------------------------------------------------------------------------
def _synth(confused: bool, n_ev: int = 200, n_st: int = 12, T: int = 2048,
           seed: int = 0) -> dict:
    """Build cache-shaped statistics with a known answer.

    confused=False : the P channel picks the true P, the S channel the true S.
    confused=True  : the P channel picks the true S instead (phase confusion),
                     the S channel still picks the true S.
    """
    rng = np.random.default_rng(seed)
    gt = np.full((n_ev, 2, n_st), -1, np.int64)
    pr_max = np.zeros((n_ev, 2, n_st), np.float32)
    pr_time = np.zeros((n_ev, 2, n_st), np.int64)
    is_noise = np.zeros(n_ev, bool)
    is_noise[: n_ev // 10] = True
    for e in range(n_ev):
        if is_noise[e]:
            pr_max[e] = 0.05
            continue
        base = int(rng.integers(300, 600))
        for s in range(n_st):
            tp, ts = base + 10 * s, base + 260 + 18 * s
            gt[e, 0, s], gt[e, 1, s] = tp, ts
            jit = lambda: int(rng.integers(-8, 9))        # noqa: E731
            pr_max[e, :, s] = 0.9
            pr_time[e, 0, s] = (ts if confused else tp) + jit()
            pr_time[e, 1, s] = ts + jit()
    return dict(gt_time=gt, pr_max=pr_max, pr_time=pr_time, is_noise=is_noise)


def selftest():
    fs, tol, thr = 2000.0, 40, 0.30

    clean = assign(_synth(confused=False), thr, tol)
    assert clean["P"]["frac_on_phase"] > 0.99, clean["P"]
    assert clean["P"]["frac_cross"] < 0.01, clean["P"]
    assert clean["S"]["frac_on_phase"] > 0.99, clean["S"]

    conf = assign(_synth(confused=True), thr, tol)
    assert conf["P"]["frac_on_phase"] < 0.01, conf["P"]
    assert conf["P"]["frac_cross"] > 0.99, conf["P"]
    assert conf["P"]["frac_cross_of_missed"] > 0.99, conf["P"]
    assert conf["S"]["frac_on_phase"] > 0.99, conf["S"]

    # noise windows must contribute no picks to the assignment
    s = _synth(confused=False)
    s["pr_max"][s["is_noise"]] = 1.0
    a = assign(s, thr, tol)
    assert a["P"]["n_picks"] == clean["P"]["n_picks"], "noise leaked into picks"

    # residuals: the confused model's P residual against S peaks at zero
    r = residuals(_synth(confused=True), thr, fs)
    assert abs(float(np.median(r["P"]["other"]))) < 5.0, "cross residual off"
    assert abs(float(np.median(r["P"]["own"]))) > 100.0, "own residual too small"

    # figure path exercised on synthetic caches
    caches = {("synthA", "array"): _synth(confused=True),
              ("synthA", "pertrace"): _synth(confused=False)}
    out = cfg.PDF_DIR / "15_selftest_phase_residuals.pdf"
    make_figure(caches, ["synthA"], thr, fs, out)
    assert out.exists() and out.stat().st_size > 5_000

    # residual summary: the confused model's P picks sit ON the S arrival,
    # so the own-phase median is the P-S separation and none lie between
    rs = residual_summary(_synth(confused=True), thr, fs, tol)
    assert rs["P"]["n"] > 0
    assert rs["P"]["own_ms_q25_med_q75"][1] > 100.0, rs["P"]
    assert abs(rs["P"]["other_ms_q25_med_q75"][1]) < 5.0, rs["P"]
    assert rs["P"]["frac_between_phases"] < 0.05, rs["P"]
    rc = residual_summary(_synth(confused=False), thr, fs, tol)
    assert abs(rc["P"]["own_ms_q25_med_q75"][1]) < 5.0, rc["P"]

    # the report writer must survive tuple keys and round-trip through JSON
    rep = {}
    for k, v in caches.items():
        a = assign(v, thr, tol)
        sm = residual_summary(v, thr, fs, tol)
        for nm in ("P", "S"):
            a[nm]["residuals"] = sm.get(nm, {})
        rep[k] = a
    write_report(rep, "selftest")
    rt = json.loads((cfg.LOGS_DIR / "15_phase_confusion_selftest.json")
                    .read_text())
    assert rt["synthA__array"]["P"]["frac_cross"] > 0.99, rt["synthA__array"]
    assert rt["synthA__pertrace"]["P"]["frac_on_phase"] > 0.99

    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stats", action="store_true",
                    help="assignment counts from the script-10 caches")
    ap.add_argument("--figure", action="store_true",
                    help="residual distributions against both phases")
    ap.add_argument("--sites", nargs="*", default=list(FIG_SITES),
                    help="sites to draw (default: forge_19 mseel_5h)")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--threshold", type=float,
                    default=EVALU.PEAK_PROB_THRESHOLD)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    fs = float(getattr(DATA, "FS", 2000.0))
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * fs))
    if a.stats:
        run_stats(a.ckpt, a.threshold, tol)
    if a.figure:
        caches = load_caches(a.ckpt)
        if not caches:
            raise SystemExit(f"[figure] no caches in {cfg.LOGS_DIR}")
        make_figure(caches, a.sites, a.threshold, fs,
                    cfg.PDF_DIR / "15_a_phase_residuals.pdf")
    if not (a.stats or a.figure):
        ap.print_help()


if __name__ == "__main__":
    main()
