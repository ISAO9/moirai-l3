#!/usr/bin/env python
"""
13_station_subset_eval.py -- GJI major revision: does the trained ARRAY
picker survive a different station count? (Reviewer 1, final comment.)
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
Reviewer 1 asks whether the pretrained array-based picker can be applied to
an array with a different number of stations, suggesting this as an
unstated limitation. Architecturally the answer is yes (the station axis
is never strided, so any 1 <= S <= 12 runs unchanged); whether PERFORMANCE
survives is the empirical question this script answers, zero-shot, with no
retraining:

  For each held-out site, the LOSO ARRAY checkpoint is evaluated on
  decimated arrays of S_sub in {4, 6, 8} stations (12 = published full
  array), in two decimation modes that separate two physical effects:

    * random     -- S_sub depth-ordered stations drawn at random from the
                    12: the array APERTURE is roughly preserved, spatial
                    sampling becomes sparser (station spacing grows).
    * contiguous -- a contiguous depth window of S_sub stations: the
                    aperture (and hence the available moveout) SHRINKS,
                    reproducing an aneth-like small-aperture geometry.

  Each repeat fixes ONE index set for the whole test set (a decimated
  array is a fixed acquisition geometry); results are mean +/- std over
  --repeats independent draws. The PER-TRACE model is the control: each
  station is processed independently, so its subset scores are computed
  EXACTLY by masking the script-10 cache to the same stations -- no new
  inference, and by construction it is insensitive to array geometry.

INTERPRETATION HOOKS (for the response letter / new supplementary figure)
  - array(random) stable, array(contiguous) degrades -> the array model
    depends on the aperture/moveout, consistent with Section 5;
  - both degrade while per-trace holds -> generic station-count
    sensitivity (train-time S=12 mismatch);
  - all stable -> station count is NOT a practical limitation in 4..12.

INPUTS
  model/  LOSO array checkpoints (as for 04/07/10)
  logs/10_cache_<site>_<ckpt>_pertrace_loso_<site>.npz   (per-trace control;
        run 10 --cache --per-trace first)

OUTPUTS
  logs/13_station_subset_<site>_<ckpt>.json
  PDF/13_a_station_subset.pdf     F1-mean vs station count, per site
                                  (white background, English, margin legend)

RUN (per site; GPU recommended, inference only)
  python src/13_station_subset_eval.py --heldout forge_19
  python src/13_station_subset_eval.py --heldout mseel_5h
  ...                                  # all 8 sites recommended
  python src/13_station_subset_eval.py --figure          # after all sites
  python src/13_station_subset_eval.py --selftest
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
ts = _load("10_threshold_sweep.py")     # reduce_to_stats, prf_at, cache loader

SUBSET_SIZES = (4, 6, 8)
FULL_S = DATA.N_STATION                 # 12


# ----------------------------------------------------------------------------
# subset drawing
# ----------------------------------------------------------------------------
def draw_subset(mode: str, s_sub: int, rng: np.random.Generator) -> np.ndarray:
    """Depth-ordered station indices for one decimated-array realisation."""
    if mode == "random":
        return np.sort(rng.choice(FULL_S, size=s_sub, replace=False))
    if mode == "contiguous":
        start = int(rng.integers(0, FULL_S - s_sub + 1))
        return np.arange(start, start + s_sub)
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# array-model inference on a fixed subset
# ----------------------------------------------------------------------------
def predict_subset(model, loader, device, idx: np.ndarray):
    """(E,3,S_sub,T) probabilities + labels with stations restricted to idx."""
    import torch
    model_mod = _load("02_picker_model_l3.py")
    model.eval()
    P, L = [], []
    sel = torch.as_tensor(idx, dtype=torch.long, device=device)
    with torch.no_grad():
        for waves, lab in loader:
            x = waves.to(device)[:, :, sel, :]
            prob = model_mod.MoiraiPickerL3.activate(model(x))
            P.append(prob.cpu().numpy())
            l = lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab)
            L.append(l[:, :, idx, :])
    return np.concatenate(P, 0), np.concatenate(L, 0)


def _mask_cache(stats: dict, idx: np.ndarray) -> dict:
    """Restrict per-trace script-10 sufficient statistics to stations idx.
    Exact for the per-trace model: every station is processed independently,
    so full-array per-station outputs equal subset per-station outputs."""
    out = {k: (v[:, :, idx] if v.ndim == 3 else v) for k, v in stats.items()}
    gt = out["gt_time"]
    out["is_noise"] = (gt < 0).all(axis=(1, 2))   # recompute on the subset
    return out


# ----------------------------------------------------------------------------
def run_site(args):
    import torch
    model_mod = _load("02_picker_model_l3.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    site = args.heldout
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    thr = EVALU.PEAK_PROB_THRESHOLD
    rng = np.random.default_rng(args.seed)

    # ---- array checkpoint + loader (mirrors 07) ----------------------------
    suffix = f"_loso_{site}"
    ckpt_name = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
                 "last": TRAIN.CKPT_LAST}[args.ckpt].replace(".pt",
                                                             f"{suffix}.pt")
    ckpt = cfg.MODEL_DIR / ckpt_name
    print(f"[13] array ckpt {ckpt}  site={site}  device={device}")
    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

    loader_mod = _load("01_amber_setup.py")
    csv = loader_mod.prepare_site_csv(site, all_test=True)
    ds = loader_mod.build_amber_dataset("test", csv)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=TRAIN.NUM_WORKERS)

    # ---- per-trace control from the script-10 cache ------------------------
    pt_cache_path = (cfg.LOGS_DIR
                     / f"10_cache_{site}_{args.ckpt}_pertrace_loso_{site}.npz")
    pt_stats = dict(np.load(pt_cache_path)) if pt_cache_path.exists() else None
    if pt_stats is None:
        print(f"[13] WARNING no per-trace cache {pt_cache_path.name} -- "
              "run 10 --cache --per-trace; per-trace control skipped.")

    results = {"site": site, "ckpt": args.ckpt, "repeats": args.repeats,
               "sizes": list(SUBSET_SIZES) + [FULL_S], "modes": {}}

    # full 12-station baseline (one pass, both models)
    idx_full = np.arange(FULL_S)
    preds, labels = predict_subset(model, loader, device, idx_full)
    base_stats = ts.reduce_to_stats(preds, labels)
    base = ts.prf_at(base_stats, thr, tol)
    results["full_array"] = dict(P=base["P"]["f1"], S=base["S"]["f1"],
                                 f1_mean=base["f1_mean"])
    print(f"    full 12-station array F1-mean {base['f1_mean']:.3f} "
          f"(P {base['P']['f1']:.3f} S {base['S']['f1']:.3f})")

    for mode in ("random", "contiguous"):
        results["modes"][mode] = {}
        for s_sub in SUBSET_SIZES:
            arr_scores, pt_scores, subsets = [], [], []
            for _ in range(args.repeats):
                idx = draw_subset(mode, s_sub, rng)
                subsets.append(idx.tolist())
                preds, labels = predict_subset(model, loader, device, idx)
                st = ts.reduce_to_stats(preds, labels)
                arr_scores.append(ts.prf_at(st, thr, tol))
                if pt_stats is not None:
                    pt_scores.append(
                        ts.prf_at(_mask_cache(pt_stats, idx), thr, tol))

            def agg(scores, key_ph):
                v = np.array([(s[key_ph]["f1"] if key_ph in ("P", "S")
                               else s["f1_mean"]) for s in scores])
                return dict(mean=float(v.mean()), std=float(v.std()))

            entry = {"subsets": subsets,
                     "array": {k: agg(arr_scores, k)
                               for k in ("P", "S", "f1_mean")}}
            if pt_scores:
                entry["pertrace"] = {k: agg(pt_scores, k)
                                     for k in ("P", "S", "f1_mean")}
            results["modes"][mode][str(s_sub)] = entry
            a = entry["array"]["f1_mean"]
            msg = (f"    {mode:10s} S={s_sub}: array "
                   f"{a['mean']:.3f}±{a['std']:.3f}")
            if pt_scores:
                p = entry["pertrace"]["f1_mean"]
                msg += f" | per-trace {p['mean']:.3f}±{p['std']:.3f}"
            print(msg)

    out = cfg.LOGS_DIR / f"13_station_subset_{site}_{args.ckpt}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"[save] {out}")
    return 0


# ----------------------------------------------------------------------------
def make_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white",
                         "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    files = sorted(cfg.LOGS_DIR.glob(f"13_station_subset_*_{args.ckpt}.json"))
    if not files:
        print("[13] no result jsons -- run per-site first.")
        return 1
    data = [json.load(open(f)) for f in files]
    ncol = 4
    nrow = int(np.ceil(len(data) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.0 * nrow),
                             squeeze=False)
    styles = {("array", "random"): ("#1f6fd6", "-", "o",
                                    "array — random decimation"),
              ("array", "contiguous"): ("#d62728", "--", "s",
                                        "array — contiguous window"),
              ("pertrace", "random"): ("#666666", "-", "^",
                                       "per-trace — random decimation")}
    for i, d in enumerate(data):
        ax = axes[i // ncol][i % ncol]
        sizes = SUBSET_SIZES
        for (mdl, mode), (c, ls, mk, lb) in styles.items():
            y, e = [], []
            for s in sizes:
                ent = d["modes"][mode][str(s)].get(mdl)
                if ent is None:
                    y = []
                    break
                y.append(ent["f1_mean"]["mean"])
                e.append(ent["f1_mean"]["std"])
            if not y:
                continue
            ax.errorbar(sizes, y, yerr=e, color=c, ls=ls, marker=mk, ms=4,
                        capsize=2, label=lb)
        ax.axhline(d["full_array"]["f1_mean"], color="#1f6fd6", lw=0.8,
                   ls=":", label="array — full 12 stations")
        ax.set(title=d["site"], xlabel="Number of stations",
               ylabel="F1-mean", xticks=list(sizes) + [FULL_S],
               ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    for j in range(len(data), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    fig.legend(H, L, loc="center left", bbox_to_anchor=(0.855, 0.5),
               frameon=False, fontsize=8)
    fig.suptitle("Zero-shot transfer of the trained array picker to "
                 "decimated arrays (mean ± std over repeats)", y=1.0)
    fig.tight_layout(rect=(0, 0, 0.85, 0.97))
    out = cfg.PDF_DIR / "13_a_station_subset.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")
    return 0


# ----------------------------------------------------------------------------
def selftest():
    """No model needed: (1) subset draws are depth-ordered and in range,
    contiguous draws are contiguous; (2) masking the per-trace cache to a
    subset gives EXACTLY the score of computing stats on that subset
    directly; (3) S_sub = 12 masking reproduces the full-array score."""
    rng = np.random.default_rng(0)
    ok = True
    for mode in ("random", "contiguous"):
        for s_sub in SUBSET_SIZES:
            for _ in range(50):
                idx = draw_subset(mode, s_sub, rng)
                ok &= len(idx) == s_sub and (np.diff(idx) > 0).all() \
                    and idx.min() >= 0 and idx.max() < FULL_S
                if mode == "contiguous":
                    ok &= (np.diff(idx) == 1).all()
    # synthetic predictions
    E, S, T = 10, 12, 256
    labels = np.zeros((E, 3, S, T), np.float32)
    preds = np.zeros_like(labels)
    for e in range(8):
        for st in range(S):
            labels[e, 0, st, 80 + st] = labels[e, 1, st, 150 + st] = 1.0
            good = st % 3 != 0                      # some stations mispick
            preds[e, 0, st, (80 + st) if good else 40] = 0.9
            preds[e, 1, st, 150 + st] = 0.9
    full = ts.reduce_to_stats(preds, labels)
    idx = np.array([0, 3, 7, 10])                  # includes mispicking sts
    direct = ts.reduce_to_stats(preds[:, :, idx, :], labels[:, :, idx, :])
    masked = _mask_cache(full, idx)
    a, b = (ts.prf_at(s, 0.30, 40) for s in (direct, masked))
    ok &= all(abs(a[ph]["f1"] - b[ph]["f1"]) < 1e-12 for ph in ("P", "S"))
    ok &= 0.0 < a["P"]["f1"] < 1.0                 # non-trivial score
    c = ts.prf_at(_mask_cache(full, np.arange(12)), 0.30, 40)
    d = ts.prf_at(full, 0.30, 40)
    ok &= abs(c["f1_mean"] - d["f1_mean"]) < 1e-12
    print(f"  subset draws valid; cache-mask == direct subset "
          f"(P {a['P']['f1']:.3f}) ; S12 mask == full: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=None, help="run one held-out site")
    ap.add_argument("--figure", action="store_true",
                    help="build summary figure from existing jsons")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent subset draws per (mode, size)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.heldout:
        return run_site(args)
    if args.figure:
        return make_figure(args)
    print("nothing to do: pass --heldout <site>, --figure, or --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
