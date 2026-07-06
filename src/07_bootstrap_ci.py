#!/usr/bin/env python
"""
07_bootstrap_ci.py  --  GJI revision Tier-1: finite-test uncertainty.

For one checkpoint on one held-out site, this script:
  1. produces EVENT-GROUPED P/S probability maps (works for array, per-trace and
     station-shuffle checkpoints by reshaping per-trace output back to (E,3,S,T));
  2. computes event-level BOOTSTRAP confidence intervals on P/S F1 and the
     F1-mean at the primary tolerance (resampling whole events with replacement),
     so the array-vs-per-trace differences carry honest error bars;
  3. dumps matched-pick signed residuals (ms) for the residual-distribution
     figure (05_h).

Outputs (logs/):
  07_bootci_<site>_<ckpt><suffix>.json     mean + 95% CI per phase
  07_resid_<site>_<ckpt><suffix>.npz       P/S residual arrays (ms)

This addresses the within-test-set uncertainty; training-seed variance is
handled separately by multi-seed runs of 03 (--seed) summarised in figure 05_j.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(path):
    spec = importlib.util.spec_from_file_location(path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


cfg = _load("00_config_l3.py")
DATA, EVALU, TRAIN = cfg.DATA, cfg.EVALU, cfg.TRAIN
model_mod = _load("02_picker_model_l3.py")
ev_mod = _load("04_evaluate_l3.py")     # reuse score() and _single_pick()


@torch.no_grad()
def predict_grouped(model, loader, device, per_trace=False, shuffle=False):
    """Return (E,3,S,T) probabilities + labels, event-grouped for all model types."""
    model.eval(); P, L = [], []
    for waves, lab in loader:
        x = waves.to(device)
        if per_trace:
            B, C, S, T = x.shape
            logits = model(ev_mod._pt_split(x))                    # (B*S,3,1,T)
            prob = model_mod.MoiraiPickerL3.activate(logits)
            prob = prob.squeeze(2).reshape(B, S, prob.shape[1], T).permute(0, 2, 1, 3)
        else:
            if shuffle:
                x, lab = ev_mod._shuffle_stations(x, lab)
            prob = model_mod.MoiraiPickerL3.activate(model(x))
        P.append(prob.cpu().numpy())
        L.append(lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab))
    return np.concatenate(P, 0), np.concatenate(L, 0)


def bootstrap_ci(preds, labels, peak_thr, nboot=300, seed=0):
    rng = np.random.default_rng(seed)
    E = preds.shape[0]
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    acc = {"P": [], "S": [], "mean": []}
    for _ in range(nboot):
        idx = rng.integers(0, E, E)
        r = ev_mod.score(preds[idx], labels[idx], peak_thr)
        acc["P"].append(r["by_phase"]["P"][pt]["f1"])
        acc["S"].append(r["by_phase"]["S"][pt]["f1"])
        acc["mean"].append(r["f1_mean_primary"])

    def ci(a):
        a = np.asarray(a)
        return dict(mean=float(a.mean()),
                    lo=float(np.percentile(a, 2.5)),
                    hi=float(np.percentile(a, 97.5)))
    return {k: ci(v) for k, v in acc.items()}


def residuals_ms(preds, labels, peak_thr):
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    out = {"P": [], "S": []}
    for b in range(preds.shape[0]):
        for ch, name in ((0, "P"), (1, "S")):
            gt = ev_mod._single_pick(labels[b, ch], 0.5)
            pr = ev_mod._single_pick(preds[b, ch], peak_thr)
            for st in range(gt.shape[0]):
                if gt[st] >= 0 and pr[st] >= 0 and abs(pr[st] - gt[st]) <= tol:
                    out[name].append((pr[st] - gt[st]) / DATA.FS * 1000.0)
    return {k: np.asarray(v, float) for k, v in out.items()}


def selftest():
    # perfect predictions -> F1 CI should be ~[1,1]; residuals ~0
    E, S, T = 6, 12, 256
    lab = np.zeros((E, 3, S, T), np.float32)
    for b in range(E):
        for st in range(S):
            lab[b, 0, st, 80 + st] = 1.0
            lab[b, 1, st, 150 + st] = 1.0
    pred = lab.copy()
    ci = bootstrap_ci(pred, lab, 0.3, nboot=50)
    res = residuals_ms(pred, lab, 0.3)
    ok = ci["P"]["mean"] > 0.99 and ci["S"]["mean"] > 0.99 and \
        abs(res["P"]).max() < 1e-6
    print(f"  bootstrap CI on perfect preds: P {ci['P']['mean']:.3f} "
          f"[{ci['P']['lo']:.3f},{ci['P']['hi']:.3f}], residual~0: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--site", default=DATA.SITE)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--per-trace", dest="per_trace", action="store_true")
    ap.add_argument("--shuffle-stations", dest="shuffle_stations", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--peak-thr", type=float, default=EVALU.PEAK_PROB_THRESHOLD)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heldout = args.heldout
    shuf = args.shuffle_stations and not args.per_trace
    suffix = (("_pertrace" if args.per_trace else "")
              + ("_shuf" if shuf else "")
              + (f"_loso_{heldout}" if heldout else "")
              + (f"_seed{args.seed}" if args.seed is not None else ""))
    site = heldout or args.site
    ckpt_name = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
                 "last": TRAIN.CKPT_LAST}[args.ckpt].replace(".pt", f"{suffix}.pt")
    ckpt = cfg.MODEL_DIR / ckpt_name
    print(f"[bootci] {ckpt}  site={site}  nboot={args.nboot}  device={device}")

    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

    loader_mod = _load("01_amber_setup.py")
    csv = loader_mod.prepare_site_csv(site, all_test=bool(heldout))
    ds = loader_mod.build_amber_dataset("test", csv)
    loader = torch.utils.data.DataLoader(ds, batch_size=TRAIN.BATCH_SIZE,
                                         shuffle=False, num_workers=TRAIN.NUM_WORKERS)
    preds, labels = predict_grouped(model, loader, device, args.per_trace, shuf)
    print(f"[bootci] grouped preds {preds.shape}  events={preds.shape[0]}")

    ci = bootstrap_ci(preds, labels, args.peak_thr, nboot=args.nboot)
    res = residuals_ms(preds, labels, args.peak_thr)
    method = "pertrace" if args.per_trace else ("shuf" if shuf else "array")
    print(f"  P  F1 {ci['P']['mean']:.3f} [{ci['P']['lo']:.3f}, {ci['P']['hi']:.3f}]")
    print(f"  S  F1 {ci['S']['mean']:.3f} [{ci['S']['lo']:.3f}, {ci['S']['hi']:.3f}]")
    print(f"  mean {ci['mean']['mean']:.3f} [{ci['mean']['lo']:.3f}, {ci['mean']['hi']:.3f}]")

    out = dict(site=site, ckpt=args.ckpt, method=method, seed=args.seed,
               n_events=int(preds.shape[0]), nboot=args.nboot,
               primary_tolerance_s=EVALU.PRIMARY_TOLERANCE_S, ci=ci)
    jpath = cfg.LOGS_DIR / f"07_bootci_{site}_{args.ckpt}{suffix}.json"
    json.dump(out, open(jpath, "w"), indent=2)
    npath = cfg.LOGS_DIR / f"07_resid_{site}_{args.ckpt}{suffix}.npz"
    np.savez(npath, P=res["P"], S=res["S"])
    print(f"[save] {jpath}\n[save] {npath}")


if __name__ == "__main__":
    raise SystemExit(main())
