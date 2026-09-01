#!/usr/bin/env python
"""
10_threshold_sweep.py -- GJI major revision: threshold-free defence of the
forge_19 collapse (Reviewer 2, comment 1; Reviewer 1, lines 171/518).
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
Reviewer 2 asks whether the array P collapse at forge_19 is merely a
CALIBRATION artefact of the fixed 0.30 peak-probability threshold. This
script answers with three threshold-free / threshold-optimal analyses:

  1. CACHE      -- one inference pass per (held-out site, configuration)
                   reduced to the SUFFICIENT STATISTICS of the picking rule
                   (per station & phase: max probability + its time; ground
                   truth time; event noise flag). Because a pick is the
                   global argmax of a phase channel declared when its max
                   exceeds the threshold, these statistics reproduce the
                   paper's scoring EXACTLY at ANY threshold, so the sweep
                   needs no second forward pass.
  2. SWEEP      -- precision / recall / F1 per phase on a dense threshold
                   grid (0.02..0.98), plus noise false alarms per noise
                   event at every threshold.
  3. ORACLE     -- the best achievable F1 per phase and for F1-mean when
                   the threshold is tuned ON THE HELD-OUT SITE ITSELF
                   (an upper bound that maximally favours each model; if
                   the array P still loses to per-trace at forge_19 under
                   its own oracle threshold, the collapse cannot be a
                   calibration problem), and the area under the PR curve
                   (AUC-PR), which is threshold-free.

OUTPUTS
  logs/10_cache_<site>_<ckpt><suffix>.npz      sufficient statistics
  logs/10_threshold_sweep_<site>_<ckpt>.json   sweep + oracle numbers
  PDF/10_a_pr_curves.pdf                       PR curves (forge_19 + one
                                               in-distribution contrast site)
  PDF/10_b_f1_vs_threshold.pdf                 F1 vs threshold, all cached sites

All figures: white background, English labels, legends in the margin.

RUN (on the machine holding AMBER_H5 + model/ checkpoints)
  # 1) build caches for the decisive sites, both configurations:
  python src/10_threshold_sweep.py --cache --heldout forge_19
  python src/10_threshold_sweep.py --cache --heldout forge_19 --per-trace
  python src/10_threshold_sweep.py --cache --heldout mseel_5h
  python src/10_threshold_sweep.py --cache --heldout mseel_5h --per-trace
  #    (repeat for pnr-2, aneth, clearfield_mw4 ... all 8 sites recommended)
  # 2) sweep + figures from the caches (no GPU needed):
  python src/10_threshold_sweep.py --sweep
  # self-test (synthetic, no AMBER/ckpt required):
  python src/10_threshold_sweep.py --selftest
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

THR_GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)


# ----------------------------------------------------------------------------
# 1. sufficient statistics cache
# ----------------------------------------------------------------------------
def reduce_to_stats(preds: np.ndarray, labels: np.ndarray) -> dict:
    """(E,3,S,T) probs + labels -> the sufficient statistics of the pick rule.

    For each event e, station s, phase ch in {0:P, 1:S}:
      gt_time  : ground-truth sample (argmax of label channel) or -1
      pr_max   : maximum predicted probability over time
      pr_time  : sample index of that maximum
      is_noise : event has no ground-truth pick in either phase
    """
    E, _, S, _ = preds.shape
    gt_time = np.full((E, 2, S), -1, np.int64)
    pr_max = np.zeros((E, 2, S), np.float32)
    pr_time = np.zeros((E, 2, S), np.int64)
    for ch in (0, 1):
        lab = labels[:, ch]                       # (E,S,T)
        has = lab.max(-1) >= 0.5
        arg = lab.argmax(-1)
        gt_time[:, ch][has] = arg[has]
        pr = preds[:, ch]
        pr_max[:, ch] = pr.max(-1)
        pr_time[:, ch] = pr.argmax(-1)
    is_noise = (gt_time < 0).all(axis=(1, 2))
    return dict(gt_time=gt_time, pr_max=pr_max, pr_time=pr_time,
                is_noise=is_noise)


def build_cache(args) -> Path:
    """Run one inference pass (reusing 07's predict_grouped) and save stats."""
    import torch
    model_mod = _load("02_picker_model_l3.py")
    bc = _load("07_bootstrap_ci.py")              # predict_grouped lives here

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heldout = args.heldout
    shuf = args.shuffle_stations and not args.per_trace
    suffix = (("_pertrace" if args.per_trace else "")
              + ("_shuf" if shuf else "")
              + (f"_loso_{heldout}" if heldout else ""))
    site = heldout or DATA.SITE
    ckpt_name = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
                 "last": TRAIN.CKPT_LAST}[args.ckpt].replace(".pt", f"{suffix}.pt")
    ckpt = cfg.MODEL_DIR / ckpt_name
    print(f"[cache] {ckpt}  site={site}  device={device}")

    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

    loader_mod = _load("01_amber_setup.py")
    csv = loader_mod.prepare_site_csv(site, all_test=bool(heldout))
    ds = loader_mod.build_amber_dataset("test", csv)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=TRAIN.NUM_WORKERS)
    preds, labels = bc.predict_grouped(model, loader, device,
                                       args.per_trace, shuf)
    stats = reduce_to_stats(preds, labels)
    out = cfg.LOGS_DIR / f"10_cache_{site}_{args.ckpt}{suffix}.npz"
    np.savez_compressed(out, **stats)
    print(f"[save] {out}  events={stats['gt_time'].shape[0]} "
          f"(noise={int(stats['is_noise'].sum())})")
    return out


# ----------------------------------------------------------------------------
# 2. threshold sweep from the cache (exactly reproduces 04's scoring rule)
# ----------------------------------------------------------------------------
def prf_at(stats: dict, thr: float, tol_samples: int) -> dict:
    """Precision/recall/F1 per phase at one threshold + noise false alarms.

    Matches src/04_evaluate_l3.py: one pick per station per phase (global
    argmax, declared if max >= thr); TP if |pred - gt| <= tol; a declared pick
    with no ground truth on that station is an FP; a ground truth with no
    declared pick (or outside tolerance) is an FN. Noise-event picks are
    counted separately (fa / noise event) as in the submitted paper; the
    FA-INCLUSIVE variant (script 11) folds them into precision.
    """
    out = {}
    noise = stats["is_noise"]
    for ch, name in ((0, "P"), (1, "S")):
        gt = stats["gt_time"][:, ch]              # (E,S)
        mx = stats["pr_max"][:, ch]
        tm = stats["pr_time"][:, ch]
        picked = mx >= thr
        eq = ~noise[:, None]
        has_gt = gt >= 0
        hit = picked & has_gt & (np.abs(tm - gt) <= tol_samples)
        tp = int((hit & eq).sum())
        fp_eq = int((picked & eq & ~hit).sum())   # wrong-time or no-GT picks
        fn = int((has_gt & eq & ~hit).sum())
        fa_noise = int((picked & noise[:, None]).sum())
        prec = tp / (tp + fp_eq) if tp + fp_eq else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = dict(precision=prec, recall=rec, f1=f1,
                         tp=tp, fp=fp_eq, fn=fn, fa_noise=fa_noise)
    n_noise = max(int(noise.sum()), 1)
    out["fa_per_noise_event"] = ((out["P"]["fa_noise"]
                                  + out["S"]["fa_noise"]) / n_noise)
    out["f1_mean"] = 0.5 * (out["P"]["f1"] + out["S"]["f1"])
    return out


def sweep(stats: dict, tol_samples: int) -> dict:
    curves = {t: prf_at(stats, float(t), tol_samples) for t in THR_GRID}
    res = {"thresholds": THR_GRID.tolist()}
    for ph in ("P", "S"):
        res[ph] = {k: [curves[t][ph][k] for t in THR_GRID]
                   for k in ("precision", "recall", "f1")}
        # oracle threshold: tuned on the held-out site itself (upper bound)
        f1s = np.array(res[ph]["f1"])
        i = int(f1s.argmax())
        # AUC-PR by trapezoid over recall (threshold-free summary)
        r = np.array(res[ph]["recall"])[::-1]
        p = np.array(res[ph]["precision"])[::-1]
        order = np.argsort(r)
        _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1/2 compat
        auc = float(_trapz(p[order], r[order]))
        res[ph]["oracle"] = dict(thr=float(THR_GRID[i]), f1=float(f1s[i]),
                                 precision=res[ph]["precision"][i],
                                 recall=res[ph]["recall"][i], auc_pr=auc)
    fm = np.array([curves[t]["f1_mean"] for t in THR_GRID])
    i = int(fm.argmax())
    res["f1_mean"] = {"values": fm.tolist(),
                      "oracle": dict(thr=float(THR_GRID[i]), f1=float(fm[i]))}
    res["fa_per_noise_event"] = [curves[t]["fa_per_noise_event"]
                                 for t in THR_GRID]
    fixed = prf_at(stats, EVALU.PEAK_PROB_THRESHOLD, tol_samples)
    res["at_fixed_030"] = dict(P=fixed["P"]["f1"], S=fixed["S"]["f1"],
                               f1_mean=fixed["f1_mean"])
    return res


# ----------------------------------------------------------------------------
# 3. figures (white background, English, legend in the margin)
# ----------------------------------------------------------------------------
def _load_caches(ckpt: str):
    """{(site, method): stats} from every 10_cache_*.npz in logs/."""
    caches = {}
    for f in sorted(cfg.LOGS_DIR.glob(f"10_cache_*_{ckpt}*loso_*.npz")):
        stem = f.stem                               # 10_cache_<site>_<ckpt>...
        method = "pertrace" if "_pertrace" in stem else \
                 ("shuffle" if "_shuf" in stem else "array")
        site = stem.split("_loso_")[-1]
        caches[(site, method)] = dict(np.load(f))
    return caches


def make_figures(caches: dict, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white",
                         "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    col = {"array": "#1f6fd6", "pertrace": "#666666"}
    lab = {"array": "array-level", "pertrace": "per-trace"}

    # ---- 10_a: PR curves, OOD site vs in-distribution contrast -------------
    sites_pr = [s for s in ("forge_19", "mseel_5h")
                if (s, "array") in caches and (s, "pertrace") in caches]
    if sites_pr:
        fig, axes = plt.subplots(len(sites_pr), 2,
                                 figsize=(8.4, 3.4 * len(sites_pr)),
                                 squeeze=False)
        for r, site in enumerate(sites_pr):
            for c, ph in enumerate(("P", "S")):
                ax = axes[r][c]
                for m in ("array", "pertrace"):
                    d = results[(site, m)][ph]
                    ax.plot(d["recall"], d["precision"], "-o", ms=2.5,
                            color=col[m],
                            label=f"{lab[m]} (AUC-PR {d['oracle']['auc_pr']:.2f})")
                ax.set(xlabel="Recall", ylabel="Precision",
                       title=f"{site} — {ph} phase", xlim=(0, 1.02),
                       ylim=(0, 1.02))
                ax.grid(alpha=0.3)
        # legends in the right margin, clear of the axes
        for r in range(len(sites_pr)):
            axes[r][1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                              frameon=False, fontsize=8)
        fig.suptitle("Precision-recall curves over the full threshold range "
                     "(threshold-free comparison)", y=1.0)
        fig.tight_layout(rect=(0, 0, 0.84, 0.97))
        out = cfg.PDF_DIR / "10_a_pr_curves.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {out}")

    # ---- 10_b: F1 vs threshold grid ----------------------------------------
    sites = sorted({s for (s, _) in caches})
    if sites:
        ncol = 4
        nrow = int(np.ceil(len(sites) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.9 * nrow),
                                 squeeze=False)
        for i, site in enumerate(sites):
            ax = axes[i // ncol][i % ncol]
            for m in ("array", "pertrace"):
                if (site, m) not in results:
                    continue
                d = results[(site, m)]
                ax.plot(d["thresholds"], d["P"]["f1"], "-", color=col[m],
                        label=f"{lab[m]} P")
                ax.plot(d["thresholds"], d["S"]["f1"], "--", color=col[m],
                        alpha=0.6, label=f"{lab[m]} S")
            ax.axvline(EVALU.PEAK_PROB_THRESHOLD, color="k", lw=0.8, ls=":")
            ax.set(title=site, xlabel="Peak-probability threshold",
                   ylabel="F1", ylim=(0, 1.02))
            ax.grid(alpha=0.3)
        for j in range(len(sites), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        handles, labels_ = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="center left",
                   bbox_to_anchor=(0.86, 0.5), frameon=False, fontsize=8)
        fig.suptitle("F1 versus detection threshold (dotted line: fixed 0.30 "
                     "used in the paper)", y=1.0)
        fig.tight_layout(rect=(0, 0, 0.85, 0.97))
        out = cfg.PDF_DIR / "10_b_f1_vs_threshold.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {out}")


# ----------------------------------------------------------------------------
def run_sweep(args):
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    caches = _load_caches(args.ckpt)
    if not caches:
        print("[sweep] no logs/10_cache_*.npz found -- run --cache first.")
        return 1
    results = {}
    for key, stats in caches.items():
        results[key] = sweep(stats, tol)
        site, m = key
        o = results[key]
        print(f"{site:16s} {m:9s} fixed0.30 P {o['at_fixed_030']['P']:.3f} "
              f"S {o['at_fixed_030']['S']:.3f} | oracle P "
              f"{o['P']['oracle']['f1']:.3f}@{o['P']['oracle']['thr']:.2f} "
              f"S {o['S']['oracle']['f1']:.3f}@{o['S']['oracle']['thr']:.2f} "
              f"| AUC-PR P {o['P']['oracle']['auc_pr']:.3f}")
        jpath = cfg.LOGS_DIR / f"10_threshold_sweep_{site}_{m}_{args.ckpt}.json"
        json.dump(results[key], open(jpath, "w"), indent=2)
    make_figures(caches, results)
    return 0


# ----------------------------------------------------------------------------
def selftest():
    """Synthetic check: oracle >= fixed-threshold F1; perfect preds -> F1=1."""
    E, S, T = 8, 12, 512
    rng = np.random.default_rng(0)
    labels = np.zeros((E, 3, S, T), np.float32)
    preds = np.zeros_like(labels)
    for e in range(E - 2):                        # last 2 events = noise
        for st in range(S):
            p, s = 100 + st, 250 + st
            labels[e, 0, st, p] = labels[e, 1, st, s] = 1.0
            preds[e, 0, st, p] = 0.15 + 0.1 * rng.random()   # below 0.30!
            preds[e, 1, st, s] = 0.9
    stats = reduce_to_stats(preds, labels)
    tol = 40
    fixed = prf_at(stats, 0.30, tol)
    sw = sweep(stats, tol)
    ok = (fixed["P"]["f1"] == 0.0                  # sub-threshold at 0.30
          and sw["P"]["oracle"]["f1"] > 0.99       # recovered by oracle thr
          and sw["S"]["oracle"]["f1"] > 0.99
          and abs(sw["at_fixed_030"]["S"] - 1.0) < 1e-9
          and int(stats["is_noise"].sum()) == 2)
    print(f"  fixed-0.30 P F1 {fixed['P']['f1']:.2f} (expect 0), oracle P F1 "
          f"{sw['P']['oracle']['f1']:.2f}@{sw['P']['oracle']['thr']:.2f} "
          f"(expect ~1): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true",
                    help="run inference and store sufficient statistics")
    ap.add_argument("--sweep", action="store_true",
                    help="threshold sweep + figures from existing caches")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--per-trace", dest="per_trace", action="store_true")
    ap.add_argument("--shuffle-stations", dest="shuffle_stations",
                    action="store_true")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.cache:
        build_cache(args)
    if args.sweep:
        return run_sweep(args)
    if not (args.cache or args.sweep):
        print("nothing to do: pass --cache and/or --sweep (see docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
