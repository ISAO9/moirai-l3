#!/usr/bin/env python
"""
24_moveout_ratio_check.py -- is each site's S/P moveout ratio consistent with
its own velocity model?
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Looking at the supplement panels, the P and S moveouts at aneth have visibly
different slopes: P is almost flat while S fans out across the string. The
geometry table makes the same point numerically -- the ratio of the median S
moveout to the median P moveout is

    pnr-1 1.69   mseel_3h 1.59   mseel_5h 1.62   clearfield_mw6 1.68
    pnr-2 1.60   clearfield_mw4 1.68   forge_19 1.84   ANETH 3.00

Seven sites sit in 1.59-1.84. aneth is 3.00. That is worth a test rather than
an assumption, because the ratio is PREDICTED by the velocity model:

    For one source and one receiver string, the P and S rays follow the same
    geometric path to first order, so the across-station arrival spread scales
    with the reciprocal of the phase velocity. The moveout ratio is therefore
    Vp/Vs over the depth interval the rays traverse -- a number AMBER ships for
    every site in Attr/<dataset>.Vmodel.csv.

So the observed ratio and the model Vp/Vs are two independent estimates of the
same quantity. If they agree, the geometry explains the figure. If a site's
observed ratio is far above its model Vp/Vs, the catalogued arrivals of that
site are not consistent with its own velocity model, and the likeliest cause is
a mis-identified phase (for example an S pick placed on a later converted or
reflected arrival). That bears directly on Section S1 (label quality, reviewer
2 point R2-24) and on the paper's reading of aneth in Section 5.2.

WHAT IT DOES
------------
  1. Per site, the PER-EVENT ratio of S moveout to P moveout, from the
     catalogued arrivals in the script-10 caches. Reported as a distribution,
     not a single number, so a heavy tail is distinguishable from a shifted
     bulk.
  2. Per site, Vp/Vs from Attr/<dataset>.Vmodel.csv, averaged over the depth
     interval spanned by the receiver string, and over the whole model.
  3. The comparison, with a verdict per site.

This neither changes nor re-derives any number in the paper. It is a
consistency check on the catalogue.

OUTPUTS
  logs/24_moveout_ratio.json   per-site distributions, Vp/Vs and verdicts
  logs/24_moveout_ratio.md     the same as a table to paste
  PDF/24_a_moveout_ratio.pdf   per-event ratio distributions with the model
                               Vp/Vs marked  (+ .png at 300 dpi)

All figures: white background, English labels, legend clear of the data.

RUN (no GPU; needs the script-10 caches, and the few-kB Attr members once)
  pip -q install remotezip
  python src/24_moveout_ratio_check.py
  python src/24_moveout_ratio_check.py --attr-dir /path/to/Attr
  # self-test (synthetic, no caches, no network):
  python src/24_moveout_ratio_check.py --selftest
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
    """Numbered files cannot be imported, so load them by path."""
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")
G17 = _load("17_geometry_and_distance.py")      # caches, Attr, depth averaging

DATA = cfg.DATA
SITE_ORDER = G17.SITE_ORDER
TOL = 0.25            # a site is flagged when |observed/predicted - 1| > TOL
MIN_BAND_EVENTS = 30  # below this the band's median is not reported as a verdict
COMMON_BAND = (10.0, 20.0)   # the P-moveout band every site has events in
MIN_P_MOVEOUT_MS = 1.0    # below this the ratio is dominated by pick rounding


# ----------------------------------------------------------------------------
def per_event_ratio(stats: dict, fs: float) -> np.ndarray:
    """S moveout / P moveout, per event, for events carrying both phases.

    Moveout is the across-station spread of the catalogued arrivals of one
    phase, the definition used in Table 1. Events whose P moveout is below
    MIN_P_MOVEOUT_MS are dropped: there the ratio is set by the sample grid
    rather than by the geometry.
    """
    eq = ~stats["is_noise"]
    gt = stats["gt_time"]                                   # (E,2,S)
    out = []
    for e in np.where(eq)[0]:
        p = gt[e, 0][gt[e, 0] >= 0]
        s = gt[e, 1][gt[e, 1] >= 0]
        if len(p) < 2 or len(s) < 2:
            continue
        mp = (p.max() - p.min()) / fs * 1000.0
        ms = (s.max() - s.min()) / fs * 1000.0
        if mp < MIN_P_MOVEOUT_MS:
            continue
        out.append(ms / mp)
    return np.asarray(out, float)



# the P-moveout bins the ratio is stratified over, in ms
P_BINS = ((1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 1e9))
WELL_RESOLVED_MS = 20.0        # >= 10 samples of spread at 2000 Hz


def per_event_pairs(stats: dict, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(P moveout, S moveout) per event, in ms, for events carrying both."""
    eq = ~stats["is_noise"]
    gt = stats["gt_time"]
    mp, ms = [], []
    for e in np.where(eq)[0]:
        p_ = gt[e, 0][gt[e, 0] >= 0]
        s_ = gt[e, 1][gt[e, 1] >= 0]
        if len(p_) < 2 or len(s_) < 2:
            continue
        a = (p_.max() - p_.min()) / fs * 1000.0
        b = (s_.max() - s_.min()) / fs * 1000.0
        if a < MIN_P_MOVEOUT_MS:
            continue
        mp.append(a); ms.append(b)
    return np.asarray(mp, float), np.asarray(ms, float)


def ratio_by_pmoveout(mp: np.ndarray, ms: np.ndarray) -> dict:
    """Median ratio within bins of P moveout, plus the well-resolved subset.

    A ratio inflated by a small denominator falls towards Vp/Vs as the
    denominator grows. One that is systematic does not. The bins make that
    visible instead of leaving it to be argued.
    """
    out = {"bins": []}
    for lo, hi in P_BINS:
        m = (mp >= lo) & (mp < hi)
        lab = f"{lo:.0f}-{hi:.0f} ms" if hi < 1e8 else f">={lo:.0f} ms"
        out["bins"].append(dict(
            bin=lab, n=int(m.sum()),
            median=(float(np.median(ms[m] / mp[m])) if m.sum() else None),
            p_moveout_median=(float(np.median(mp[m])) if m.sum() else None)))
    w = mp >= WELL_RESOLVED_MS
    out["well_resolved"] = dict(
        threshold_ms=WELL_RESOLVED_MS, n=int(w.sum()),
        median=(float(np.median(ms[w] / mp[w])) if w.sum() else None))
    return out

def model_vpvs(vmodel: np.ndarray, z0: float, z1: float) -> float:
    """Thickness-weighted Vp/Vs between two depths, from the site's model."""
    vp, vs = G17._depth_average(vmodel, z0, z1)
    return float(vp / vs)


def summarise(a: np.ndarray) -> dict:
    if a.size == 0:
        return dict(n=0)
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return dict(n=int(a.size), p05=float(q[0]), q25=float(q[1]),
                median=float(q[2]), q75=float(q[3]), p95=float(q[4]))


def verdict(obs: float | None, pred: float | None, n: int = 10 ** 9) -> str:
    """The verdict for ONE comparable estimate of the ratio.

    `obs` must come from a P-moveout band shared with the other sites, not from
    the site median: the stratification shows the estimator is biased high at
    small P moveout and low at large P moveout everywhere, so a site whose
    events sit mostly in one regime is not comparable with one that does not.
    With too few events in the band the function says so instead of reporting a
    median of a handful.
    """
    if obs is None or pred is None or not np.isfinite(obs) or not np.isfinite(pred):
        return "NO DATA"
    if n < MIN_BAND_EVENTS:
        return (f"NOT DECIDABLE: only {n} events in the common band "
                f"({COMMON_BAND[0]:.0f}-{COMMON_BAND[1]:.0f} ms); the ratio is "
                f"biased by the P-moveout regime, so a median of this few is "
                f"not comparable with the other sites")
    rel = obs / pred - 1.0
    if abs(rel) <= TOL:
        return f"CONSISTENT ({rel:+.0%} of the model Vp/Vs)"
    if rel > 0:
        return (f"S MOVEOUT TOO LARGE ({rel:+.0%}) in the common band: the "
                f"catalogued S spread exceeds what this site's velocity model "
                f"allows for the same ray geometry")
    return f"S MOVEOUT TOO SMALL ({rel:+.0%}) in the common band"


def common_band(st: dict) -> tuple[float | None, int]:
    """(median ratio, n) within COMMON_BAND, from a stratified result."""
    lab = f"{COMMON_BAND[0]:.0f}-{COMMON_BAND[1]:.0f} ms"
    for b in st.get("bins", []):
        if b["bin"] == lab:
            return b["median"], b["n"]
    return None, 0


# ----------------------------------------------------------------------------
def make_figure(res: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    sites = [s for s in SITE_ORDER if res.get(s, {}).get("ratios")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    data = [np.asarray(res[s]["ratios"], float) for s in sites]
    # matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9, so the
    # tick labels are set separately: that works on every version.
    bp = ax.boxplot(data, showfliers=False, widths=0.6,
                    medianprops=dict(color="#1f77b4", linewidth=1.6))
    ax.set_xticks(range(1, len(sites) + 1))
    ax.set_xticklabels(sites)
    hpred = None
    for i, s in enumerate(sites, 1):
        pred = res[s].get("vpvs_string")
        if pred:
            hpred, = ax.plot([i - 0.32, i + 0.32], [pred, pred], color="#d62728",
                             linewidth=2.0, solid_capstyle="butt")
    ax.set_ylabel("S moveout / P moveout, per event")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Observed moveout ratio against the velocity model's Vp/Vs")
    handles = [bp["medians"][0]]
    labels = ["observed ratio (box: IQR, line: median)"]
    if hpred is not None:
        handles.append(hpred)
        labels.append("Vp/Vs over the receiver string, from the site's model")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True,
               framealpha=0.95, edgecolor="0.8", bbox_to_anchor=(0.5, 0.005))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[24] wrote {out_pdf} (+ .png)")


def write_report(res: dict, tag: str = "") -> Path:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"24_moveout_ratio{tag}"
    slim = {s: {k: v for k, v in d.items() if k != "ratios"}
            for s, d in res.items()}
    (cfg.LOGS_DIR / f"{stem}.json").write_text(json.dumps(slim, indent=2))

    md = ["# S/P moveout ratio against the velocity model", "",
          "The moveout ratio is Vp/Vs for the same ray geometry, so the two "
          "columns are independent estimates of one quantity. The estimator is "
          "biased high where the P moveout is small and low where it is large, "
          "at every site, so the verdict is taken on a band every site shares "
          "(10-20 ms) rather than on the site median.", "",
          "| Site | events | site median (IQR) | P<10 ms share | "
          "common band 10-20 ms | Vp/Vs (string) | verdict |",
          "|---|---|---|---|---|---|---|"]
    for s in SITE_ORDER:
        d = res.get(s)
        if not d or d.get("n", 0) == 0:
            md.append(f"| {s} | 0 | — | — | — | — | NO DATA |")
            continue
        cb = d.get("common_band", {})
        ctxt = (f"{cb['median']:.2f} (n={cb['n']:,})"
                if cb.get("median") is not None else "—")
        md.append(
            f"| {s} | {d['n']:,} | {d['median']:.2f} "
            f"({d['q25']:.2f}–{d['q75']:.2f}) | "
            f"{100*d.get('frac_P_below_10ms', 0):.1f} % | {ctxt} | "
            f"{d.get('vpvs_string', float('nan')):.2f} | {d['verdict']} |")
    p = cfg.LOGS_DIR / f"{stem}.md"
    p.write_text("\n".join(md))
    return p


def run(attr_dir, url, tag: str = "") -> dict:
    fs = float(getattr(DATA, "SAMPLE_RATE", 2000.0))
    caches = G17.load_caches("ema")
    if not caches:
        raise SystemExit(f"[24] no 10_cache_*.npz in {cfg.LOGS_DIR} -- "
                         f"build them with script 10 --cache")
    attrs = G17.read_attributes(attr_dir, url, want_vmodel=True)

    res = {}
    for site in SITE_ORDER:
        if site not in caches:
            continue
        r = per_event_ratio(caches[site], fs)
        d = summarise(r)
        d["ratios"] = r.tolist()
        mp, ms_ = per_event_pairs(caches[site], fs)
        d["stratified"] = ratio_by_pmoveout(mp, ms_)
        a = attrs.get(site, {})
        if "vmodel" in a and "xyz" in a:
            z = np.asarray(a["xyz"])[:, 2]
            d["vpvs_string"] = model_vpvs(a["vmodel"], float(z.min()), float(z.max()))
            vm = a["vmodel"]
            d["vpvs_model"] = model_vpvs(vm, float(vm[:, 0].min()),
                                         float(vm[:, 0].max()))
            d["string_depth_m"] = [float(z.min()), float(z.max())]
        cb, cn = common_band(d["stratified"])
        d["common_band"] = dict(band=f"{COMMON_BAND[0]:.0f}-{COMMON_BAND[1]:.0f} ms",
                                median=cb, n=cn)
        # the share of events in the biased small-moveout regime, which is what
        # makes a site median incomparable
        small = sum(b["n"] for b in d["stratified"]["bins"]
                    if b["bin"] in ("1-5 ms", "5-10 ms"))
        tot = sum(b["n"] for b in d["stratified"]["bins"]) or 1
        d["frac_P_below_10ms"] = round(small / tot, 4)
        d["verdict"] = verdict(cb, d.get("vpvs_string"), cn)
        res[site] = d
        wr = d["stratified"]["well_resolved"]
        wrm = wr["median"]
        print(f"[24] {site:<16s} n={d.get('n', 0):>5} "
              f"observed {d.get('median', float('nan')):.2f} "
              f"vs model {d.get('vpvs_string', float('nan')):.2f}  "
              f"{d['verdict']}")
        print(f"     P moveout で層別（分母が小さいための見かけかを判定）:")
        for b in d["stratified"]["bins"]:
            if b["n"]:
                print(f"       P {b['bin']:>10s}  n={b['n']:>4d}  "
                      f"ratio median {b['median']:.2f}")
        print(f"       P >= {wr['threshold_ms']:.0f} ms のみ: n={wr['n']}, "
              f"ratio median " + (f"{wrm:.2f}" if wrm else "—"))
        print(f"       P < 10 ms の事象の割合: {100*d['frac_P_below_10ms']:.1f} %"
              f"  （この領域では比が高めに偏る)")

    p = write_report(res, tag)
    print(f"\n[24] wrote {p} and {p.with_suffix('.json')}")
    print(p.read_text())
    if any(d.get("n", 0) for d in res.values()):
        make_figure(res, cfg.PDF_DIR / f"24_a_moveout_ratio{tag}.pdf")
    return res


# ----------------------------------------------------------------------------
# self-test (synthetic; no caches, no network)
# ----------------------------------------------------------------------------
def _synth(ratio: float, n_ev=200, n_st=12, fs=2000.0, seed=0) -> dict:
    """Catalogue-shaped arrays whose S/P moveout ratio is known exactly."""
    rng = np.random.default_rng(seed)
    gt = np.full((n_ev, 2, n_st), -1, np.int64)
    for e in range(n_ev):
        step_p = rng.uniform(4, 10)                  # samples per station
        t0 = rng.integers(300, 900)
        for s in range(n_st):
            gt[e, 0, s] = int(t0 + step_p * s)
            gt[e, 1, s] = int(t0 + 400 + step_p * ratio * s)
    return dict(is_noise=np.zeros(n_ev, bool), gt_time=gt)


def selftest() -> None:
    fs = 2000.0
    # 1. a known ratio is recovered
    for want in (1.7, 3.0):
        r = per_event_ratio(_synth(want, seed=int(want * 10)), fs)
        med = float(np.median(r))
        assert abs(med - want) < 0.05, (want, med)

    # 2. events with too little P moveout are dropped, not turned into noise
    st = _synth(1.7, n_ev=50)
    st["gt_time"][:25, 0, :] = st["gt_time"][:25, 0, 0][:, None]   # flat P
    r = per_event_ratio(st, fs)
    assert len(r) == 25, len(r)

    # 3. noise windows never enter
    st2 = _synth(1.7, n_ev=40)
    st2["is_noise"][:10] = True
    assert len(per_event_ratio(st2, fs)) == 30

    # 4. Vp/Vs from a two-layer model, and the verdict thresholds
    vm = np.array([[0.0, 3400.0, 2000.0], [1000.0, 5100.0, 3000.0]])
    assert abs(model_vpvs(vm, 0.0, 1000.0) - 1.7) < 1e-9
    assert verdict(1.70, 1.70, 100).startswith("CONSISTENT")
    assert verdict(3.00, 1.70, 100).startswith("S MOVEOUT TOO LARGE")
    assert verdict(1.00, 1.70, 100).startswith("S MOVEOUT TOO SMALL")
    assert verdict(1.70 * (1 + TOL * 0.9), 1.70, 100).startswith("CONSISTENT")
    assert verdict(1.70 * (1 + TOL * 1.1), 1.70, 100).startswith("S MOVEOUT TOO LARGE")
    assert verdict(None, 1.7, 100) == "NO DATA"
    # a handful of events must NOT produce a verdict: this is the defect the
    # aneth run exposed, where n=11 in the >=20 ms bucket read as "systematic"
    assert verdict(0.50, 1.70, 11).startswith("NOT DECIDABLE"), verdict(0.50, 1.70, 11)
    assert verdict(3.00, 1.70, MIN_BAND_EVENTS - 1).startswith("NOT DECIDABLE")
    assert verdict(3.00, 1.70, MIN_BAND_EVENTS).startswith("S MOVEOUT TOO LARGE")
    # common_band picks the right bin, and reports nothing when it is absent
    stx = dict(bins=[dict(bin="10-20 ms", n=42, median=1.61),
                     dict(bin=">=40 ms", n=5, median=0.5)])
    assert common_band(stx) == (1.61, 42)
    assert common_band(dict(bins=[])) == (None, 0)

    # 5. the stratification must separate a real ratio from a floor artefact
    mp = np.concatenate([np.full(200, 3.0), np.full(200, 50.0)])
    ms_ = np.concatenate([np.full(200, 3.0 * 3.0),     # small P: ratio 3
                          np.full(200, 50.0 * 1.7)])   # large P: ratio 1.7
    st = ratio_by_pmoveout(mp, ms_)
    small = [b for b in st["bins"] if b["bin"] == "1-5 ms"][0]
    assert abs(small["median"] - 3.0) < 1e-6, small
    assert abs(st["well_resolved"]["median"] - 1.7) < 1e-6, st["well_resolved"]
    assert st["well_resolved"]["n"] == 200
    # and a ratio that is the same at every P moveout survives unchanged
    mp2 = np.concatenate([np.full(150, 3.0), np.full(150, 50.0)])
    st2 = ratio_by_pmoveout(mp2, mp2 * 2.7)
    assert abs(st2["well_resolved"]["median"] - 2.7) < 1e-6, st2
    # per_event_pairs must agree with per_event_ratio on the same input
    syn = _synth(1.9, n_ev=60, seed=7)
    a, b = per_event_pairs(syn, fs)
    assert np.allclose(np.sort(b / a), np.sort(per_event_ratio(syn, fs)))

    # 6. the figure builds and the report writes, on synthetic input
    res = {}
    for i, (site, ratio) in enumerate(zip(SITE_ORDER, (1.7, 1.6, 1.65, 1.7,
                                                       1.6, 1.68, 3.0, 1.84))):
        r = per_event_ratio(_synth(ratio, seed=i), fs)
        d = summarise(r); d["ratios"] = r.tolist()
        d["vpvs_string"] = 1.70; d["vpvs_model"] = 1.72
        d["common_band"] = dict(band="10-20 ms", median=d["median"], n=d["n"])
        d["frac_P_below_10ms"] = 0.0
        d["verdict"] = verdict(d["median"], d["vpvs_string"], d["n"])
        res[site] = d
    assert res["aneth"]["verdict"].startswith("S MOVEOUT TOO LARGE"), res["aneth"]
    assert res["pnr-1"]["verdict"].startswith("CONSISTENT")
    p = write_report(res, "_selftest")
    assert p.exists() and "| aneth |" in p.read_text()
    out = cfg.PDF_DIR / "24_selftest_moveout_ratio.pdf"
    make_figure(res, out)
    assert out.exists() and out.stat().st_size > 10_000

    print("[selftest] ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--attr-dir", default=None,
                    help="local copy of AMBER's Attr/ (otherwise read from Zenodo)")
    ap.add_argument("--zenodo-url", default=getattr(G17, "ZEN_URL", None))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(Path(a.attr_dir) if a.attr_dir else None, a.zenodo_url)


if __name__ == "__main__":
    main()
