#!/usr/bin/env python
"""
20_within_array.py -- GJI revision round 2: the within-array experiment
(Reviewer 2, point 2; Editor, point 4).
=============================================================================
WHAT THE REVIEWER ASKED FOR
---------------------------
"A plausible experiment would be to train with something like 90 per cent of
the data from each array, and then test on the remaining 10 per cent. I would
like to see the performance of a model that saw long moveout data in training,
versus a model that did not, like the shown in the manuscript."

The manuscript's leave-one-site-out protocol answers a different question: how
a model behaves on an array it has never seen. The reviewer's protocol answers
whether the array picker is capable of the moveout at all once that moveout is
in its training data. Both are worth having, and the contrast between them is
the paper's claim stated as an experiment rather than as an argument.

The benchmark already carries the split the reviewer describes: every site has
its own train, development and test split in the AMBER metadata, so no new
partition has to be invented. This script trains ONE pair of models -- array
and per-trace -- on the training splits of all eight sites together, then
evaluates each on each site's own test split.

THE PREDICTION THIS TESTS
At forge_19 the leave-one-site-out array picker collapses (F1-mean 0.52 against
0.86 per trace) while its moveout is reached by 0.32 per cent of its training
events. If that collapse is caused by the scarcity of such moveout in training,
then a model trained with forge_19's own events in the training split should
pick forge_19 normally, and the array-minus-per-trace gap there should close.
If instead forge_19 is intrinsically hard -- noisier, smaller events, worse
picks -- the gap should persist. The experiment therefore separates the two
explanations the editor asked to be separated, and it can fail: a persistent
gap would refute the paper's mechanism rather than confirm it.

HOW IT REUSES THE EXISTING CODE
The training loop, optimiser, EMA, early stopping and checkpointing come from
03_train_l3.py unchanged -- this script only substitutes the data splits and
redirects the outputs, so the comparison with the leave-one-site-out runs is
not confounded by a second implementation of training. Scoring comes from
10_threshold_sweep.py and 11_fa_inclusive_scoring.py for the same reason.

INPUTS
  the AMBER h5 + metadata csv                                   (script 01)
  logs/11_fa_inclusive_<ckpt>.json       optional; the submitted LOSO verdicts

OUTPUTS
  model/within_array/…                   the two checkpoints, kept apart from
                                         the leave-one-site-out models
  logs/20_within_array_history_*.json    per-epoch training history
  logs/20_within_array_<ckpt>.json/.md   per-site scores and the comparison
  PDF/20_a_within_vs_loso.pdf            the figure for the response

RUN
  python src/20_within_array.py --train                  # GPU, two runs
  python src/20_within_array.py --train --per-trace      # one of the two
  python src/20_within_array.py --eval                   # GPU, one pass
  python src/20_within_array.py --selftest               # no GPU, no AMBER
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
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
ts = _load("10_threshold_sweep.py")
fa = _load("11_fa_inclusive_scoring.py")

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]
OUT_MODEL = cfg.MODEL_DIR / "within_array"
TAG = "allsites"


def _ckpt(ckpt: str, per_trace: bool) -> Path:
    base = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
            "last": TRAIN.CKPT_LAST}[ckpt]
    return OUT_MODEL / base.replace(".pt", "_pertrace.pt" if per_trace else ".pt")


# ----------------------------------------------------------------------------
# 1. training: script 03's loop, this script's splits
# ----------------------------------------------------------------------------
def _all_sites_loaders(loader_mod):
    """Replacement for build_dataloaders: every split from all eight sites.

    The signature matches the original so script 03 calls it without knowing
    the difference; the arguments that select a site or a held-out fold are
    accepted and ignored, because here there is no held-out site.
    """
    import torch

    def build(site=None, loso=False, heldout=None, per_trace=False):
        csv = loader_mod.prepare_sites_csv(list(DATA.USABLE_SITES), tag=TAG)
        loaders = {}
        for mode in ("train", "dev", "test"):
            ds = loader_mod.build_amber_dataset(mode, csv)
            loaders[mode] = torch.utils.data.DataLoader(
                ds, batch_size=TRAIN.BATCH_SIZE, shuffle=(mode == "train"),
                num_workers=TRAIN.NUM_WORKERS, pin_memory=True,
                drop_last=(mode == "train"))
        return loaders
    return build


def train(per_trace: bool, epochs: int = 0, smoke: bool = False):
    """Run script 03's training with the all-sites splits and its own outputs."""
    t03 = _load("03_train_l3.py")

    # script 03 loads its modules inside run(), so the substitution has to be
    # made on its loader rather than on an instance held here
    original = t03._load_module

    def patched(filename: str):
        m = original(filename)
        if filename == "01_amber_setup.py":
            m.build_dataloaders = _all_sites_loaders(m)
        return m
    t03._load_module = patched

    # keep these checkpoints and this history away from the LOSO ones, which
    # carry no suffix in this configuration and would otherwise be overwritten
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    t03.cfg.MODEL_DIR = OUT_MODEL
    hist_dir = cfg.LOGS_DIR / "within_array"
    hist_dir.mkdir(parents=True, exist_ok=True)
    t03.cfg.LOGS_DIR = hist_dir

    args = argparse.Namespace(site=DATA.SITE, epochs=epochs, smoke=smoke,
                              loso=False, heldout=None, per_trace=per_trace,
                              shuffle_stations=False, seed=None, selftest=False)
    print(f"[20] training on the train splits of {len(DATA.USABLE_SITES)} sites, "
          f"config={'per-trace' if per_trace else 'array'}")
    t03.train(args)

    src = hist_dir / ("03_train_history_pertrace.json" if per_trace
                      else "03_train_history.json")
    if src.exists():
        dst = cfg.LOGS_DIR / (f"20_within_array_history_"
                              f"{'pertrace' if per_trace else 'array'}.json")
        shutil.copy(src, dst)
        print(f"[20] history -> {dst}")
    for c in ("best", "ema", "last"):
        p = _ckpt(c, per_trace)
        print(f"[20] checkpoint {'OK  ' if p.exists() else 'MISSING '}{p}")


# ----------------------------------------------------------------------------
# 2. evaluation: each site's own test split, both configurations, one pass
# ----------------------------------------------------------------------------
def evaluate(ckpt: str = "ema") -> dict:
    """Score both checkpoints on every site's test split, on identical inputs."""
    import torch
    model_mod = _load("02_picker_model_l3.py")
    ev_mod = _load("04_evaluate_l3.py")
    loader_mod = _load("01_amber_setup.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_a, ck_p = _ckpt(ckpt, False), _ckpt(ckpt, True)
    for c in (ck_a, ck_p):
        if not c.exists():
            raise SystemExit(f"[20] missing checkpoint {c} -- run --train first")

    def _model(path):
        m = model_mod.build_model().to(device)
        m.load_state_dict(torch.load(path, map_location=device)["model_state"])
        m.eval()
        return m

    model_a, model_p = _model(ck_a), _model(ck_p)
    out = {}
    for site in DATA.USABLE_SITES:
        csv = loader_mod.prepare_site_csv(site)          # native splits
        ds = loader_mod.build_amber_dataset("test", csv)  # that site's own test split
        if len(ds) == 0:
            print(f"[20] {site}: empty test split, skipped")
            continue
        loader = torch.utils.data.DataLoader(
            ds, batch_size=TRAIN.BATCH_SIZE, shuffle=False,
            num_workers=TRAIN.NUM_WORKERS)
        ca, cp = [], []
        with torch.no_grad():
            for waves, lab in loader:
                x = waves.to(device)
                lab_np = (lab.cpu().numpy() if hasattr(lab, "cpu")
                          else np.asarray(lab))
                pa = model_mod.MoiraiPickerL3.activate(model_a(x))
                B, C, S, T = x.shape
                pp = model_mod.MoiraiPickerL3.activate(
                    model_p(ev_mod._pt_split(x)))
                pp = pp.squeeze(2).reshape(B, S, pp.shape[1], T).permute(0, 2, 1, 3)
                ca.append(ts.reduce_to_stats(pa.cpu().numpy(), lab_np))
                cp.append(ts.reduce_to_stats(pp.cpu().numpy(), lab_np))
        sa = {k: np.concatenate([c[k] for c in ca], 0) for k in ca[0]}
        sp = {k: np.concatenate([c[k] for c in cp], 0) for k in cp[0]}
        assert np.array_equal(sa["gt_time"], sp["gt_time"]), \
            f"{site}: the two configurations were not scored on the same labels"
        np.savez_compressed(
            cfg.LOGS_DIR / f"20_cache_{site}_{ckpt}_array.npz", **sa)
        np.savez_compressed(
            cfg.LOGS_DIR / f"20_cache_{site}_{ckpt}_pertrace.npz", **sp)
        out[site] = (sa, sp)
        print(f"[20] {site:<16s} test events {sa['gt_time'].shape[0]}")
    return out


def score(ckpt: str, nboot: int) -> tuple[dict, str]:
    """Per-site scores, paired intervals, and the contrast with the LOSO run."""
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    thr = EVALU.PEAK_PROB_THRESHOLD
    loso = {}
    f = cfg.LOGS_DIR / f"11_fa_inclusive_{ckpt}.json"
    if f.exists():
        try:
            loso = json.loads(f.read_text())
        except ValueError:
            loso = {}

    rows = {}
    for site in SITE_ORDER:
        pa = cfg.LOGS_DIR / f"20_cache_{site}_{ckpt}_array.npz"
        pp = cfg.LOGS_DIR / f"20_cache_{site}_{ckpt}_pertrace.npz"
        if not (pa.exists() and pp.exists()):
            continue
        sa, sp = dict(np.load(pa)), dict(np.load(pp))
        ra, rp = fa.score_both(sa, thr, tol), fa.score_both(sp, thr, tol)
        # identical inputs by construction, so the paired difference is valid
        rng = np.random.default_rng(0)
        E = sa["gt_time"].shape[0]
        d = []
        for _ in range(nboot):
            idx = rng.integers(0, E, E)
            d.append(fa.score_both(fa._subset(sa, idx), thr, tol)["orig"]["mean"]
                     - fa.score_both(fa._subset(sp, idx), thr, tol)["orig"]["mean"])
        d = np.asarray(d, float)
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        verdict = ("array" if ci[0] > 0 else
                   "per-trace" if ci[1] < 0 else "tie")
        l = loso.get(site, {})
        rows[site] = dict(
            n_test_events=int(E),
            array=ra["orig"]["mean"], pertrace=rp["orig"]["mean"],
            diff_median=float(np.median(d)), diff_ci=ci, verdict=verdict,
            loso_array=(l.get("array", {}).get("orig", {}) or {}).get("mean"),
            loso_pertrace=(l.get("pertrace", {}).get("orig", {}) or {}).get("mean"),
            loso_verdict=l.get("verdict_orig"))
        r = rows[site]
        print(f"[20] {site:<16s} within-array: arr {r['array']:.3f} vs pt "
              f"{r['pertrace']:.3f}  diff {r['diff_median']:+.3f} "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}] -> {verdict}"
              + (f"   (LOSO: {r['loso_array']:.3f} / {r['loso_pertrace']:.3f} "
                 f"[{r['loso_verdict']}])" if r["loso_array"] is not None else ""))

    md = ["| Site | test events | LOSO array / per-trace | within-array array / "
          "per-trace | array - per-trace, paired 95% CI | LOSO verdict | "
          "within-array verdict |",
          "|---|---|---|---|---|---|---|"]
    for s in SITE_ORDER:
        if s not in rows:
            continue
        r = rows[s]
        lo = ("-" if r["loso_array"] is None else
              f"{r['loso_array']:.3f} / {r['loso_pertrace']:.3f}")
        md.append(f"| {s} | {r['n_test_events']} | {lo} | "
                  f"{r['array']:.3f} / {r['pertrace']:.3f} | "
                  f"{r['diff_median']:+.3f} [{r['diff_ci'][0]:+.3f}, "
                  f"{r['diff_ci'][1]:+.3f}] | {r['loso_verdict'] or '-'} | "
                  f"{r['verdict']} |")

    f19 = rows.get("forge_19")
    if f19 and f19["loso_array"] is not None:
        gap_loso = f19["loso_pertrace"] - f19["loso_array"]
        gap_within = f19["pertrace"] - f19["array"]
        md += ["", f"At forge_19 the per-trace advantage is {gap_loso:+.3f} "
                   f"under leave-one-site-out and {gap_within:+.3f} when the "
                   f"site's own events are in the training split."]
        if gap_within < 0.25 * gap_loso:
            md.append("The gap closes once the moveout is represented in "
                      "training, which is what the paper's mechanism predicts: "
                      "the collapse is a property of the training distribution, "
                      "not of the site.")
        elif gap_within > 0.75 * gap_loso:
            md.append("The gap persists even with the site's own events in "
                      "training, so the collapse is NOT explained by the "
                      "scarcity of such moveout and the manuscript's mechanism "
                      "must be revised.")
        else:
            md.append("The gap narrows but does not close, so scarcity of such "
                      "moveout in training explains part, but not all, of the "
                      "collapse.")
    return rows, "\n".join(md)


# ----------------------------------------------------------------------------
# 3. figure
# ----------------------------------------------------------------------------
def make_figure(rows: dict, pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    sites = [s for s in SITE_ORDER if s in rows]
    x = np.arange(len(sites))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.93, bottom=0.14, hspace=0.5)

    ax = axes[0]
    ax.set_title("F1-mean under the two protocols", fontsize=10)
    for dx, key, lkey, col, lab in ((-0.18, "array", "loso_array", "#1f77b4",
                                     "array"),
                                    (0.18, "pertrace", "loso_pertrace",
                                     "#d62728", "per-trace")):
        lo = [rows[s][lkey] if rows[s][lkey] is not None else np.nan
              for s in sites]
        wi = [rows[s][key] for s in sites]
        ax.plot(x + dx, lo, "o", mfc="white", mec=col, ms=7,
                label=f"{lab}, leave-one-site-out")
        ax.plot(x + dx, wi, "o", color=col, ms=7, label=f"{lab}, within-array")
        for xi, a, b in zip(x + dx, lo, wi):
            if np.isfinite(a):
                ax.plot([xi, xi], [a, b], "-", color=col, lw=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(sites) - 0.4)
    ax.set_ylabel("F1-mean")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    ax = axes[1]
    ax.set_title("Array minus per-trace within the array (paired bootstrap, "
                 "95 per cent)", fontsize=10)
    med = [rows[s]["diff_median"] for s in sites]
    lo = [rows[s]["diff_ci"][0] for s in sites]
    hi = [rows[s]["diff_ci"][1] for s in sites]
    cols = ["#1f77b4" if a > 0 else ("#d62728" if b < 0 else "#7f7f7f")
            for a, b in zip(lo, hi)]
    for xi, m, a, b, c in zip(x, med, lo, hi, cols):
        ax.errorbar(xi, m, yerr=[[m - a], [b - m]], fmt="o", color=c, ms=6,
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
def _check_script03_api() -> list:
    """Verify, without importing torch, that script 03 still offers what this
    script borrows from it.

    The training path cannot run in the self-test -- it needs a GPU, AMBER and
    the waveform archive -- so the names it depends on are checked statically
    instead. An earlier version of this script called a function that does not
    exist and the self-test passed anyway; this is the check that closes that
    gap, and it will fail loudly if script 03 is ever renamed or restructured.
    """
    import ast
    src = (HERE / "03_train_l3.py").read_text()
    tree = ast.parse(src)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    names = {t.id for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name)}
    missing = [n for n in ("train", "_load_module") if n not in funcs]
    missing += [n for n in ("cfg",) if n not in names]
    # train(args) must accept the attributes this script sets on its Namespace
    needed = {"site", "epochs", "smoke", "loso", "heldout", "per_trace",
              "shuffle_stations", "seed"}
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    missing += [f"args.{a}" for a in needed
                if a not in used and f'"{a}"' not in src and f"'{a}'" not in src]
    # script 01 must still offer the two builders the replacement loader uses
    s01 = ast.parse((HERE / "01_amber_setup.py").read_text())
    f01 = {n.name for n in s01.body if isinstance(n, ast.FunctionDef)}
    missing += [f"01.{n}" for n in ("prepare_sites_csv", "prepare_site_csv",
                                    "build_amber_dataset", "build_dataloaders")
                if n not in f01]
    return missing


def selftest():
    miss = _check_script03_api()
    assert not miss, f"scripts 01/03 no longer provide: {miss}"
    print("  scripts 01/03 still provide the names this script borrows: PASS")

    # the scoring path, on labels shared by the two configurations
    E, S, T, n_eq = 40, 12, 256, 28
    rng = np.random.default_rng(0)
    labels = np.zeros((E, 3, S, T), np.float32)
    pa, pp = np.zeros_like(labels), np.zeros_like(labels)
    for e in range(n_eq):
        t0 = int(rng.integers(40, 120))
        for st in range(S):
            tp, tsw = t0 + st, t0 + st + 60
            labels[e, 0, st, tp] = labels[e, 1, st, tsw] = 1.0
            pa[e, 0, st, tp] = 0.9
            pa[e, 1, st, tsw] = 0.9
            pp[e, 0, st, tp] = 0.9 if e % 2 else 0.0
            pp[e, 1, st, tsw] = 0.9
    sa = ts.reduce_to_stats(pa, labels)
    sp = ts.reduce_to_stats(pp, labels)
    assert np.array_equal(sa["gt_time"], sp["gt_time"])
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for site, st_ in (("forge_19", (sa, sp)), ("pnr-1", (sa, sa))):
        np.savez(cfg.LOGS_DIR / f"20_cache_{site}_selftest_array.npz", **st_[0])
        np.savez(cfg.LOGS_DIR / f"20_cache_{site}_selftest_pertrace.npz", **st_[1])
    rows, md = score("selftest", nboot=60)
    assert set(rows) == {"forge_19", "pnr-1"}, rows.keys()
    assert rows["forge_19"]["verdict"] == "array", rows["forge_19"]
    assert rows["pnr-1"]["verdict"] == "tie", rows["pnr-1"]
    assert rows["pnr-1"]["diff_ci"][0] <= 0 <= rows["pnr-1"]["diff_ci"][1]
    print(f"  scoring: forge_19 {rows['forge_19']['diff_median']:+.3f} -> array; "
          f"identical configs -> tie: PASS")

    # the narrative test must read the forge_19 gap in the right direction
    for gl, gw, want in ((0.34, 0.02, "gap closes"),
                         (0.34, 0.30, "gap persists"),
                         (0.34, 0.17, "narrows but does not close")):
        r = {"forge_19": dict(rows["forge_19"], loso_array=0.52,
                              loso_pertrace=0.52 + gl,
                              array=0.90, pertrace=0.90 + gw,
                              loso_verdict="per-trace")}
        f19 = r["forge_19"]
        gap_l = f19["loso_pertrace"] - f19["loso_array"]
        gap_w = f19["pertrace"] - f19["array"]
        got = ("gap closes" if gap_w < 0.25 * gap_l else
               "gap persists" if gap_w > 0.75 * gap_l else
               "narrows but does not close")
        assert got == want, (gl, gw, got, want)
    print("  forge_19 narrative: closes / narrows / persists all reachable: PASS")

    pdf = cfg.PDF_DIR / "20_selftest_within.pdf"
    for s in rows:
        rows[s]["loso_array"] = 0.52
        rows[s]["loso_pertrace"] = 0.86
        rows[s]["loso_verdict"] = "per-trace"
    make_figure(rows, pdf)
    assert pdf.exists() and pdf.stat().st_size > 8_000
    for p in cfg.LOGS_DIR.glob("20_cache_*_selftest_*.npz"):
        p.unlink()
    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--per-trace", dest="per_trace", action="store_true",
                    help="with --train, run only the per-trace configuration")
    ap.add_argument("--both", action="store_true",
                    help="with --train, run array then per-trace in one go")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.train:
        if a.both:
            train(False, a.epochs, a.smoke)
            train(True, a.epochs, a.smoke)
        else:
            train(a.per_trace, a.epochs, a.smoke)
    if a.eval:
        evaluate(a.ckpt)
        rows, md = score(a.ckpt, a.nboot)
        if not rows:
            return
        (cfg.LOGS_DIR / f"20_within_array_{a.ckpt}.json").write_text(
            json.dumps(rows, indent=2))
        (cfg.LOGS_DIR / f"20_within_array_{a.ckpt}.md").write_text(md)
        print("\n" + md)
        make_figure(rows, cfg.PDF_DIR / "20_a_within_vs_loso.pdf")
        print(f"[save] {cfg.LOGS_DIR / f'20_within_array_{a.ckpt}.md'}")
    if not (a.train or a.eval):
        ap.print_help()


if __name__ == "__main__":
    main()
