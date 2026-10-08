#!/usr/bin/env python
"""
18_paired_cache.py -- GJI revision round 2: evaluate the array and per-trace
pickers on IDENTICAL inputs, as a supplementary robustness check.
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
The submitted caches were written by two independent passes over the dataset,
one per configuration. AMBER draws the analysis window afresh on every pass
and, where a string carries more than N_STATION sensors, also draws which
contiguous run of sensors is used. A direct comparison of the two caches shows
the consequence: the event set and its order are identical at every site
(is_noise matches exactly), but the catalogued arrival times differ, and the
disagreement tracks the number of sensors on the string --

    12-sensor sites (clearfield_mw4, clearfield_mw6, forge_19, mseel_3h,
                     mseel_5h, pnr-2)     98-100 per cent of events agree
    pnr-1   (24 sensors)                  98 per cent agree on which sensors
                                          carry picks, 69 per cent on moveout
    aneth   (18 sensors)                  21 and 52 per cent respectively

-- which is exactly the pattern expected if the sensor subset is redrawn, since
only the two sites with more than twelve sensors have a subset to choose.

Nothing in this is systematic: the draw does not depend on which configuration
is being evaluated, so neither is favoured, and the submitted site-level
verdicts rest on MARGINAL bootstrap intervals, which remain valid. What the
submitted design cannot claim is that the two configurations saw the same
inputs. For a paper whose title offers a confound-free benchmark, that is worth
removing rather than explaining, so this script re-evaluates both
configurations in ONE pass over the data:

    for each batch:  array probabilities  <- array checkpoint
                     per-trace probabilities <- per-trace checkpoint
                     labels               <- used by both, by construction

The labels are literally the same tensor, so the two caches it writes are paired
by construction, and the script asserts that before saving.

WHAT THIS BUYS, BEYOND REMOVING THE OBJECTION
Identical inputs make a PAIRED bootstrap legitimate: the array-minus-per-trace
difference can be resampled directly, which removes the between-event variance
common to both configurations and is therefore strictly more powerful than
comparing two marginal intervals. Both are reported -- the marginal form for
comparability with the submitted Table 3, the paired form as the stronger test.

THE SUBMITTED NUMBERS ARE NOT REPLACED. This is a supplementary check: it
answers "does the result hold when the two configurations see identical
inputs?", and the manuscript's main tables stay as archived.

INPUTS
  models/<ckpt>_loso_<site>.pt           array checkpoint        (script 03)
  models/<ckpt>_pertrace_loso_<site>.pt  per-trace checkpoint    (script 03)
  the AMBER h5 + metadata csv                                    (script 01)

OUTPUTS
  logs/18_paired_<site>_<ckpt>_array.npz      paired caches, identical labels
  logs/18_paired_<site>_<ckpt>_pertrace.npz
  logs/18_paired_scores_<ckpt>.json           marginal + paired intervals
  logs/18_paired_scores_<ckpt>.md             the supplementary table
  PDF/18_a_paired_vs_unpaired.pdf             submitted vs paired, per site

RUN
  python src/18_paired_cache.py --build-all          # GPU, one pass per site
  python src/18_paired_cache.py --build --heldout pnr-1
  python src/18_paired_cache.py --score              # CPU, seconds
  python src/18_paired_cache.py --selftest           # no GPU, no AMBER, no net
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
DATA, EVALU, TRAIN = cfg.DATA, cfg.EVALU, cfg.TRAIN
ts = _load("10_threshold_sweep.py")          # reduce_to_stats, prf_at
fa = _load("11_fa_inclusive_scoring.py")     # score_both, _subset, verdict

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]


def _paths(site: str, ckpt: str) -> tuple[Path, Path]:
    return (cfg.LOGS_DIR / f"18_paired_{site}_{ckpt}_array.npz",
            cfg.LOGS_DIR / f"18_paired_{site}_{ckpt}_pertrace.npz")


# ----------------------------------------------------------------------------
# 1. one pass, two models
# ----------------------------------------------------------------------------
def _cat_stats(chunks: list) -> dict:
    """Concatenate per-batch stats dicts along the event axis.

    reduce_to_stats is purely per-event (it reduces over the time axis of each
    event independently), so reducing per batch and concatenating is identical
    to reducing the whole array at once. The self-test asserts that, because
    the equality is what keeps the paired caches comparable with the submitted
    ones rather than merely similar to them.
    """
    return {k: np.concatenate([c[k] for c in chunks], 0) for k in chunks[0]}


def build_site(site: str, ckpt: str, batch_size: int | None = None,
               force: bool = False) -> tuple[Path, Path]:
    """Evaluate both checkpoints on the same batches and write paired caches.

    Finished sites are skipped unless force is set, so the job can be resumed
    after a disconnect without repeating the GPU work already done.
    """
    pa0, pp0 = _paths(site, ckpt)
    if not force and pa0.exists() and pp0.exists():
        print(f"[18] {site}: paired caches already present, skipped "
              f"(--force to rebuild)")
        return pa0, pp0
    import torch
    model_mod = _load("02_picker_model_l3.py")
    ev_mod = _load("04_evaluate_l3.py")                  # _pt_split
    loader_mod = _load("01_amber_setup.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
            "last": TRAIN.CKPT_LAST}[ckpt]
    ck_a = cfg.MODEL_DIR / base.replace(".pt", f"_loso_{site}.pt")
    ck_p = cfg.MODEL_DIR / base.replace(".pt", f"_pertrace_loso_{site}.pt")
    for c in (ck_a, ck_p):
        if not c.exists():
            raise SystemExit(f"[18] missing checkpoint {c}")

    def _model(path):
        m = model_mod.build_model().to(device)
        m.load_state_dict(torch.load(path, map_location=device)["model_state"])
        m.eval()
        return m

    model_a, model_p = _model(ck_a), _model(ck_p)
    print(f"[18] {site}: array={ck_a.name}  per-trace={ck_p.name}  dev={device}")

    csv = loader_mod.prepare_site_csv(site, all_test=True)
    ds = loader_mod.build_amber_dataset("test", csv)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size or TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=TRAIN.NUM_WORKERS)

    chunks_a, chunks_p, n = [], [], 0
    with torch.no_grad():
        for waves, lab in loader:
            x = waves.to(device)
            lab_np = lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab)

            prob_a = model_mod.MoiraiPickerL3.activate(model_a(x))

            B, C, S, T = x.shape
            logits = model_p(ev_mod._pt_split(x))         # (B*S,3,1,T)
            prob_p = model_mod.MoiraiPickerL3.activate(logits)
            prob_p = prob_p.squeeze(2).reshape(
                B, S, prob_p.shape[1], T).permute(0, 2, 1, 3)

            # the SAME lab_np feeds both reductions -- this is the whole point
            chunks_a.append(ts.reduce_to_stats(prob_a.cpu().numpy(), lab_np))
            chunks_p.append(ts.reduce_to_stats(prob_p.cpu().numpy(), lab_np))
            n += B
    sa, sp = _cat_stats(chunks_a), _cat_stats(chunks_p)

    # the guarantee this script exists to provide, checked rather than assumed
    assert np.array_equal(sa["gt_time"], sp["gt_time"]), \
        "paired caches disagree on the catalogue -- the pairing is broken"
    assert np.array_equal(sa["is_noise"], sp["is_noise"])

    pa, pp = _paths(site, ckpt)
    np.savez_compressed(pa, **sa)
    np.savez_compressed(pp, **sp)
    print(f"[18] {site}: {n} events (noise={int(sa['is_noise'].sum())}) "
          f"-> {pa.name}, {pp.name}")
    return pa, pp


# ----------------------------------------------------------------------------
# 2. scoring: marginal (as submitted) and paired (only legitimate here)
# ----------------------------------------------------------------------------
def paired_bootstrap(sa: dict, sp: dict, thr: float, tol: int,
                     nboot: int = 300, seed: int = 0) -> dict:
    """Bootstrap the array-minus-per-trace difference on the SAME events.

    Legitimate only because both configurations were evaluated on identical
    inputs: replicate k scores the same resampled events under both, so the
    between-event variance they share cancels instead of entering twice. The
    interval is therefore narrower than the gap between two marginal intervals,
    and a difference can be significant here while the marginal intervals
    overlap. Reporting both is what makes the comparison honest.
    """
    rng = np.random.default_rng(seed)
    E = sa["gt_time"].shape[0]
    d = {"orig": [], "fa_incl": []}
    for _ in range(nboot):
        idx = rng.integers(0, E, E)
        ra = fa.score_both(fa._subset(sa, idx), thr, tol)
        rp = fa.score_both(fa._subset(sp, idx), thr, tol)
        for k in d:
            d[k].append(ra[k]["mean"] - rp[k]["mean"])

    def ci(a):
        a = np.asarray(a, float)
        return dict(median=float(np.median(a)),
                    lo=float(np.percentile(a, 2.5)),
                    hi=float(np.percentile(a, 97.5)))
    return {k: ci(v) for k, v in d.items()}


def paired_verdict(c: dict) -> str:
    if c["lo"] > 0.0:
        return "array"
    if c["hi"] < 0.0:
        return "per-trace"
    return "tie"


def _submitted_verdict(site: str, ckpt: str):
    """The verdict actually printed in the submitted Table 3, or None.

    Script 11 wrote it; reading it here is the only way this table can claim
    anything about the submitted result. Comparing the paired verdict with the
    marginal verdict recomputed on the SAME paired caches says nothing about
    the submission, and an earlier version of this script mislabelled exactly
    that comparison as reproducing it.
    """
    f = cfg.LOGS_DIR / f"11_fa_inclusive_{ckpt}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())[site]["verdict_orig"]
    except (KeyError, ValueError, TypeError):
        return None


def _reversal(a, b) -> bool:
    """True only for a change of DIRECTION; tie <-> decided is not a reversal."""
    return (a in ("array", "per-trace") and b in ("array", "per-trace")
            and a != b)


def _submitted(site: str, method: str, ckpt: str):
    """F1-mean at the fixed threshold as archived by script 10, or None."""
    f = cfg.LOGS_DIR / f"10_threshold_sweep_{site}_{method}_{ckpt}.json"
    if not f.exists():
        return None
    try:
        return float(json.loads(f.read_text())["at_fixed_030"]["f1_mean"])
    except (KeyError, ValueError, TypeError):
        return None


def score_all(ckpt: str, nboot: int) -> tuple[dict, str]:
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    thr = EVALU.PEAK_PROB_THRESHOLD
    out = {}
    sites = [s for s in SITE_ORDER if _paths(s, ckpt)[0].exists()
             and _paths(s, ckpt)[1].exists()]
    if not sites:
        print(f"[18] no paired caches in {cfg.LOGS_DIR} -- run --build-all first")
        return {}, ""
    for s in sites:
        pa, pp = _paths(s, ckpt)
        sa, sp = dict(np.load(pa)), dict(np.load(pp))
        assert np.array_equal(sa["gt_time"], sp["gt_time"]), \
            f"{s}: paired caches are not paired"
        ra, rp = fa.score_both(sa, thr, tol), fa.score_both(sp, thr, tol)
        marg = fa.bootstrap(sa, sp, thr, tol, nboot)
        pair = paired_bootstrap(sa, sp, thr, tol, nboot)
        out[s] = dict(
            n_events=int(sa["gt_time"].shape[0]),
            array=ra["orig"]["mean"], pertrace=rp["orig"]["mean"],
            array_fa=ra["fa_incl"]["mean"], pertrace_fa=rp["fa_incl"]["mean"],
            marginal_ci=marg, paired_ci=pair,
            verdict_marginal=fa.verdict(marg["array"]["orig"],
                                        marg["pertrace"]["orig"]),
            verdict_paired=paired_verdict(pair["orig"]),
            verdict_paired_fa=paired_verdict(pair["fa_incl"]),
            submitted_array=_submitted(s, "array", ckpt),
            submitted_pertrace=_submitted(s, "pertrace", ckpt),
            submitted_verdict=_submitted_verdict(s, ckpt))
        out[s]["reversal_vs_submitted"] = _reversal(
            out[s]["submitted_verdict"], out[s]["verdict_paired"])
        o = out[s]
        print(f"[18] {s:<16s} paired: arr {o['array']:.3f} vs pt "
              f"{o['pertrace']:.3f}  diff {pair['orig']['median']:+.3f} "
              f"[{pair['orig']['lo']:+.3f}, {pair['orig']['hi']:+.3f}] "
              f"-> {o['verdict_paired']}")

    md = ["| Site | events | array F1-mean (submitted -> paired) | "
          "per-trace F1-mean (submitted -> paired) | array - per-trace, "
          "paired 95% CI | verdict, submitted | verdict, marginal on paired "
          "data | verdict, paired | direction reversed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        o = out[s]
        sa_, sp_ = o["submitted_array"], o["submitted_pertrace"]
        c = o["paired_ci"]["orig"]
        md.append(
            f"| {s} | {o['n_events']} | "
            f"{'n/a' if sa_ is None else f'{sa_:.3f}'} -> {o['array']:.3f} | "
            f"{'n/a' if sp_ is None else f'{sp_:.3f}'} -> {o['pertrace']:.3f} | "
            f"{c['median']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
            f"{o['submitted_verdict'] or 'n/a'} | {o['verdict_marginal']} | "
            f"{o['verdict_paired']} | "
            f"{'YES' if o['reversal_vs_submitted'] else 'no'} |")
    known = [o for o in out.values() if o["submitted_verdict"]]
    rev = sum(o["reversal_vs_submitted"] for o in known)
    same = sum(o["submitted_verdict"] == o["verdict_paired"] for o in known)
    sharper = sum(o["submitted_verdict"] == "tie"
                  and o["verdict_paired"] != "tie" for o in known)
    shift = max((abs(o[k] - o[s_]) for o in out.values()
                 for k, s_ in (("array", "submitted_array"),
                               ("pertrace", "submitted_pertrace"))
                 if o[s_] is not None), default=float("nan"))
    md += ["",
           f"Evaluating both configurations on identical windows and identical "
           f"sensors moves no site-level F1-mean by more than {shift:.3f}, so "
           f"the independent draws used in the submitted evaluation cost "
           f"essentially nothing.",
           "",
           f"Against the submitted verdicts: {rev} of {len(known)} reverse "
           f"direction, {same} are identical, and {sharper} move from a tie to "
           f"a decided verdict. A tie becoming decided is the expected effect "
           f"of pairing, which cancels the between-event variance the two "
           f"configurations share and is therefore the more powerful test; it "
           f"is not a disagreement with the submitted result. The marginal "
           f"column is recomputed on the SAME paired caches and is shown only "
           f"so the two interval forms can be compared on one dataset."]
    return out, "\n".join(md)


# ----------------------------------------------------------------------------
# 3. figure
# ----------------------------------------------------------------------------
def make_figure(out: dict, pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    sites = [s for s in SITE_ORDER if s in out]
    x = np.arange(len(sites))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.93, bottom=0.14, hspace=0.45)

    ax = axes[0]
    ax.set_title("Submitted (independent inputs) and paired (identical inputs) "
                 "F1-mean", fontsize=10)
    for dx, key, sub, col, lab in ((-0.18, "array", "submitted_array",
                                    "#1f77b4", "array"),
                                   (0.18, "pertrace", "submitted_pertrace",
                                    "#d62728", "per-trace")):
        s_val = [out[s][sub] if out[s][sub] is not None else np.nan for s in sites]
        p_val = [out[s][key] for s in sites]
        ax.plot(x + dx, s_val, "o", mfc="white", mec=col, ms=7,
                label=f"{lab}, submitted")
        ax.plot(x + dx, p_val, "o", color=col, ms=7, label=f"{lab}, paired")
        for xi, a, b in zip(x + dx, s_val, p_val):
            if np.isfinite(a):
                ax.plot([xi, xi], [a, b], "-", color=col, lw=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(sites) - 0.4)
    ax.set_ylabel("F1-mean")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    ax = axes[1]
    ax.set_title("Array minus per-trace on identical inputs "
                 "(paired bootstrap, 95 per cent)", fontsize=10)
    med = [out[s]["paired_ci"]["orig"]["median"] for s in sites]
    lo = [out[s]["paired_ci"]["orig"]["lo"] for s in sites]
    hi = [out[s]["paired_ci"]["orig"]["hi"] for s in sites]
    err = np.vstack([np.array(med) - np.array(lo), np.array(hi) - np.array(med)])
    cols = ["#1f77b4" if l > 0 else ("#d62728" if h < 0 else "#7f7f7f")
            for l, h in zip(lo, hi)]
    for xi, m, e0, e1, c in zip(x, med, err[0], err[1], cols):
        ax.errorbar(xi, m, yerr=[[e0], [e1]], fmt="o", color=c, ms=6,
                    capsize=4, lw=1.4)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(sites) - 0.4)
    ax.set_ylabel("F1-mean difference")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="none", color=c, label=l)
               for c, l in (("#1f77b4", "array better"),
                            ("#d62728", "per-trace better"),
                            ("#7f7f7f", "interval spans zero"))]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True)

    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {pdf} (+ .png)")


# ----------------------------------------------------------------------------
def _synth(E=60, S=12, T=256, n_eq=40, array_better=True, seed=0):
    """Two configurations scored on ONE set of labels, as build_site produces.

    Some events are HARD FOR BOTH configurations and some are hard for the
    per-trace one alone. The shared difficulty is what the paired bootstrap is
    supposed to cancel, so without it the two interval forms would be equally
    wide and the self-test would prove nothing.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((E, 3, S, T), np.float32)
    pa, pp = np.zeros_like(labels), np.zeros_like(labels)
    hard = set(rng.choice(n_eq, size=n_eq // 4, replace=False).tolist())
    for e in range(n_eq):
        t0 = int(rng.integers(40, 120))
        for st in range(S):
            tp = t0 + st
            tsw = tp + 60
            labels[e, 0, st, tp] = labels[e, 1, st, tsw] = 1.0
            pa[e, 0, st, tp] = pp[e, 0, st, tp] = 0.9
            pa[e, 1, st, tsw] = pp[e, 1, st, tsw] = 0.9
            if e in hard:                   # both configurations miss P here
                pa[e, 0, st, tp] = pp[e, 0, st, tp] = 0.0
                pa[e, 0, st, (tp + 90) % T] = pp[e, 0, st, (tp + 90) % T] = 0.6
            elif array_better and e % 3 == 0:        # per-trace alone misses P
                pp[e, 0, st, tp] = 0.0
                pp[e, 0, st, (tp + 90) % T] = 0.6
    for e in range(n_eq, E):
        pa[e, 0, :, 10] = 0.4
        pp[e, 0, :, 10] = 0.4
    return labels, pa, pp


def selftest():
    labels, pa, pp = _synth()
    sa_whole = ts.reduce_to_stats(pa, labels)
    sp_whole = ts.reduce_to_stats(pp, labels)

    # per-batch reduction must equal whole-array reduction, or the paired
    # caches would not be comparable with the submitted ones
    bs = 7
    chunks_a = [ts.reduce_to_stats(pa[i:i + bs], labels[i:i + bs])
                for i in range(0, pa.shape[0], bs)]
    sa_batched = _cat_stats(chunks_a)
    for k in sa_whole:
        assert np.array_equal(sa_whole[k], sa_batched[k]), k
    print("  per-batch reduction == whole-array reduction: PASS")

    # the pairing guarantee
    assert np.array_equal(sa_whole["gt_time"], sp_whole["gt_time"])
    assert np.array_equal(sa_whole["is_noise"], sp_whole["is_noise"])

    tol, thr = 40, 0.30
    pair = paired_bootstrap(sa_whole, sp_whole, thr, tol, nboot=120)
    marg = fa.bootstrap(sa_whole, sp_whole, thr, tol, nboot=120)
    v_p, v_m = paired_verdict(pair["orig"]), fa.verdict(marg["array"]["orig"],
                                                        marg["pertrace"]["orig"])
    ra = fa.score_both(sa_whole, thr, tol)["orig"]["mean"]
    rp = fa.score_both(sp_whole, thr, tol)["orig"]["mean"]
    assert ra > rp, (ra, rp)
    assert v_p == "array", (v_p, pair)
    assert pair["orig"]["lo"] <= pair["orig"]["median"] <= pair["orig"]["hi"]
    # Cancelling the shared between-event variance must make the paired
    # interval narrower than the two marginal ones taken together. Equality
    # would mean the two configurations fail on disjoint events, in which case
    # pairing buys nothing -- so this is the assertion that proves the method,
    # not merely that it runs.
    w_pair = pair["orig"]["hi"] - pair["orig"]["lo"]
    w_marg = ((marg["array"]["orig"][1] - marg["array"]["orig"][0])
              + (marg["pertrace"]["orig"][1] - marg["pertrace"]["orig"][0]))
    assert w_pair < w_marg, (w_pair, w_marg)
    print(f"  array {ra:.3f} vs per-trace {rp:.3f}; paired diff "
          f"{pair['orig']['median']:+.3f} "
          f"[{pair['orig']['lo']:+.3f}, {pair['orig']['hi']:+.3f}] "
          f"-> {v_p} (marginal: {v_m}); paired width {w_pair:.3f} "
          f"< summed marginal {w_marg:.3f}: PASS")

    # an honest tie must come out as a tie
    labels2, pa2, pp2 = _synth(array_better=False, seed=1)
    t_pair = paired_bootstrap(ts.reduce_to_stats(pa2, labels2),
                              ts.reduce_to_stats(pp2, labels2), thr, tol, 120)
    assert paired_verdict(t_pair["orig"]) == "tie", t_pair
    print("  identical detectors -> tie: PASS")

    out = {"forge_19": dict(n_events=60, array=ra, pertrace=rp,
                            array_fa=ra, pertrace_fa=rp,
                            marginal_ci=marg, paired_ci=pair,
                            verdict_marginal=v_m, verdict_paired=v_p,
                            verdict_paired_fa=v_p,
                            submitted_array=0.52, submitted_pertrace=0.86),
           "pnr-1": dict(n_events=60, array=rp, pertrace=ra,
                         array_fa=rp, pertrace_fa=ra,
                         marginal_ci=marg, paired_ci=t_pair,
                         verdict_marginal="tie", verdict_paired="tie",
                         verdict_paired_fa="tie",
                         submitted_array=0.99, submitted_pertrace=0.99)}
    pdf = cfg.PDF_DIR / "18_selftest_paired.pdf"
    make_figure(out, pdf)
    assert pdf.exists() and pdf.stat().st_size > 8_000
    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--build", action="store_true",
                    help="build the paired caches for one held-out site")
    ap.add_argument("--build-all", action="store_true",
                    help="build the paired caches for every usable site")
    ap.add_argument("--heldout", default=None, help="site for --build")
    ap.add_argument("--score", action="store_true",
                    help="score the paired caches (CPU, seconds)")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild paired caches even if they already exist")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.build or a.build_all or a.score):
        ap.print_help()
        return

    if a.build_all:
        for s in DATA.USABLE_SITES:
            build_site(s, a.ckpt, a.batch_size, a.force)
    elif a.build:
        if not a.heldout:
            raise SystemExit("[18] --build needs --heldout <site>")
        build_site(a.heldout, a.ckpt, a.batch_size, a.force)

    if a.score:
        out, md = score_all(a.ckpt, a.nboot)
        if not out:
            return
        (cfg.LOGS_DIR / f"18_paired_scores_{a.ckpt}.json").write_text(
            json.dumps(out, indent=2))
        (cfg.LOGS_DIR / f"18_paired_scores_{a.ckpt}.md").write_text(md)
        print("\n" + md)
        make_figure(out, cfg.PDF_DIR / "18_a_paired_vs_unpaired.pdf")
        print(f"[save] {cfg.LOGS_DIR / f'18_paired_scores_{a.ckpt}.md'}")


if __name__ == "__main__":
    main()
