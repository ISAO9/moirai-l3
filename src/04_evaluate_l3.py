"""
04_evaluate_l3.py — MOIRAI L3 evaluation (array P/S picking)
=============================================================
WHAT THIS SCRIPT DOES
---------------------
Scores a trained MoiraiPickerL3 checkpoint on a split with PUBLICATION-grade,
HONEST metrics (the central L2 lesson: report median and fail-rate, never just
the mean). For phases P and S it computes, at every tolerance in
EvalCfg.TOLERANCES_S:

    precision, recall, F1, and (over matched picks) mean AE, MEDIAN AE,
    plus fail-rate = fraction of ground-truth picks NOT recovered.

It also breaks results down by event type recovered from the labels:
    * earthquake events  (>=1 ground-truth pick) -> P/S precision/recall/F1/MAE
    * noise events       (no ground-truth pick)   -> FALSE-ALARM rate
mseel_3h is 684 earthquake + 1000 noise events, so the noise false-alarm rate is
a first-class metric (does the picker stay silent on noise?).

A pick is the argmax-over-time of a phase channel above PEAK_PROB_THRESHOLD (one
P and one S per station per window). Results are written to
logs/04_test_results_<site>_<ckpt>.json.

RUN
  python 04_evaluate_l3.py                      # test split, best ckpt
  python 04_evaluate_l3.py --ckpt ema           # EMA checkpoint
  python 04_evaluate_l3.py --site pnr-1 --split test   # unseen-site transfer
  python 04_evaluate_l3.py --selftest           # synthetic, no amber
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_module(filename: str):
    path = Path(__file__).with_name(filename)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cfg = _load_module("00_config_l3.py")
model_mod = _load_module("02_picker_model_l3.py")
DATA, MODEL, LOSS, TRAIN, EVALU = cfg.DATA, cfg.MODEL, cfg.LOSS, cfg.TRAIN, cfg.EVALU


# ----------------------------------------------------------------------------
# picking + rich scoring
# ----------------------------------------------------------------------------
def _single_pick(prob_2d: np.ndarray, thr: float):
    """prob_2d (n_station, T) -> per-station pick index or -1."""
    picks = np.full(prob_2d.shape[0], -1, dtype=np.int64)
    mx, am = prob_2d.max(1), prob_2d.argmax(1)
    picks[mx >= thr] = am[mx >= thr]
    return picks


def score(pred_prob: np.ndarray, label: np.ndarray, peak_thr: float,
          label_thr: float = 0.5):
    """pred_prob, label: (B, 3, n_station, T). Returns the full metric dict.

    Event-type split: an event is 'earthquake' if it has >=1 GT pick in P or S,
    else 'noise'. Picks/MAE accumulate per phase and per tolerance."""
    tols = [int(round(t * DATA.FS)) for t in EVALU.TOLERANCES_S]
    acc = {name: {tol: {"tp": 0, "fp": 0, "fn": 0, "ae": []} for tol in tols}
           for name in ("P", "S")}
    noise_events = noise_false_alarm = 0
    eq_events = 0

    for b in range(pred_prob.shape[0]):
        gt_p = _single_pick(label[b, 0], label_thr)
        gt_s = _single_pick(label[b, 1], label_thr)
        has_gt = (gt_p >= 0).any() or (gt_s >= 0).any()
        if has_gt:
            eq_events += 1
        else:
            noise_events += 1
        for ch, name, gt in ((0, "P", gt_p), (1, "S", gt_s)):
            pred = _single_pick(pred_prob[b, ch], peak_thr)
            for st in range(pred.shape[0]):
                hp, hg = pred[st] >= 0, gt[st] >= 0
                err = abs(pred[st] - gt[st]) if (hp and hg) else None
                if (not has_gt) and hp:
                    noise_false_alarm += 1
                for tol in tols:
                    a = acc[name][tol]
                    if hp and hg:
                        if err <= tol:
                            a["tp"] += 1
                            a["ae"].append(err)
                        else:
                            a["fp"] += 1
                            a["fn"] += 1
                    elif hp and not hg:
                        a["fp"] += 1
                    elif hg and not hp:
                        a["fn"] += 1

    out = {"by_phase": {}, "tolerances_s": list(EVALU.TOLERANCES_S),
           "n_events": int(pred_prob.shape[0]),
           "n_earthquake_events": int(eq_events),
           "n_noise_events": int(noise_events),
           "noise_false_alarm_picks": int(noise_false_alarm),
           "noise_false_alarm_per_event":
               float(noise_false_alarm / noise_events) if noise_events else 0.0}
    for name in ("P", "S"):
        out["by_phase"][name] = {}
        for tol, t_s in zip(tols, EVALU.TOLERANCES_S):
            a = acc[name][tol]
            tp, fp, fn = a["tp"], a["fp"], a["fn"]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            ae = np.asarray(a["ae"], float)
            out["by_phase"][name][f"tol_{t_s}s"] = dict(
                precision=prec, recall=rec, f1=f1,
                fail_rate=1.0 - rec,
                mae_s=float(ae.mean() / DATA.FS) if ae.size else float("nan"),
                median_ae_s=float(np.median(ae) / DATA.FS) if ae.size else float("nan"),
                n_matched=int(ae.size))
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    out["f1_mean_primary"] = 0.5 * (out["by_phase"]["P"][pt]["f1"]
                                    + out["by_phase"]["S"][pt]["f1"])
    return out


# ----------------------------------------------------------------------------
# evaluation driver
# ----------------------------------------------------------------------------
def _pt_split(t):
    """(B,C,S,T) -> (B*S,C,1,T): per-trace ablation split (see 01.pertrace_split)."""
    B, C, S, T = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * S, C, T).unsqueeze(2)


def _shuffle_stations(waves, lab):
    """Permute station axis identically for waves and labels (independent perm per
    sample): destroys across-station moveout, keeps array size. Array-mode control."""
    B, C, S, T = waves.shape
    w, l = waves.clone(), lab.clone()
    for b in range(B):
        perm = torch.randperm(S)
        w[b] = waves[b, :, perm.to(waves.device), :]
        l[b] = lab[b, :, perm.to(lab.device), :]
    return w, l


@torch.no_grad()
def run(model, loader, device, per_trace=False, shuffle_stations=False):
    model.eval()
    preds, labels = [], []
    for waves, lab in loader:
        waves = waves.to(device)
        if per_trace:
            waves, lab = _pt_split(waves), _pt_split(lab)
        elif shuffle_stations:
            waves, lab = _shuffle_stations(waves, lab)
        logits = model(waves)
        preds.append(model_mod.MoiraiPickerL3.activate(logits).cpu().numpy())
        labels.append(lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab))
    return np.concatenate(preds, 0), np.concatenate(labels, 0)


def _pretty(res):
    print(f"  events: {res['n_events']} "
          f"(eq {res['n_earthquake_events']}, noise {res['n_noise_events']})")
    print(f"  noise false-alarm: {res['noise_false_alarm_picks']} picks "
          f"({res['noise_false_alarm_per_event']:.3f}/noise-event)")
    for name in ("P", "S"):
        print(f"  --- {name} ---")
        for t_s in res["tolerances_s"]:
            m = res["by_phase"][name][f"tol_{t_s}s"]
            print(f"    +/-{t_s*1e3:4.0f} ms | F1 {m['f1']:.3f} "
                  f"P {m['precision']:.3f} R {m['recall']:.3f} "
                  f"fail {m['fail_rate']:.3f} | MAE {m['mae_s']*1e3:5.1f} ms "
                  f"med {m['median_ae_s']*1e3:5.1f} ms (n={m['n_matched']})")
    print(f"  F1-mean @ primary ({EVALU.PRIMARY_TOLERANCE_S}s): "
          f"{res['f1_mean_primary']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best", choices=["best", "ema", "last"])
    ap.add_argument("--site", default=DATA.SITE)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--peak-thr", type=float, default=EVALU.PEAK_PROB_THRESHOLD,
                    help="peak probability threshold for declaring a pick")
    ap.add_argument("--sweep", action="store_true",
                    help="re-score one inference pass at several thresholds "
                         "(find the recall/precision sweet spot; no retraining)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--loso", action="store_true",
                    help="evaluate the LOSO checkpoint on the default held-out site")
    ap.add_argument("--heldout", default=None,
                    help="evaluate the LOSO fold held out on this specific site")
    ap.add_argument("--per-trace", dest="per_trace", action="store_true",
                    help="evaluate the per-trace ablation checkpoint (single-station)")
    ap.add_argument("--shuffle-stations", dest="shuffle_stations", action="store_true",
                    help="evaluate the station-shuffle control checkpoint")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed suffix _seed<n> to select the matching checkpoint")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heldout = args.heldout or (DATA.HELDOUT_SITE if args.loso else None)
    shuf = getattr(args, "shuffle_stations", False) and not args.per_trace
    suffix = (("_pertrace" if args.per_trace else "")
              + ("_shuf" if shuf else "")
              + (f"_loso_{heldout}" if heldout else "")
              + (f"_seed{args.seed}" if getattr(args, "seed", None) is not None else ""))
    if heldout:
        args.site, args.split = heldout, "test"
    ckpt_name = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
                 "last": TRAIN.CKPT_LAST}[args.ckpt].replace(".pt", f"{suffix}.pt")
    ckpt_path = cfg.MODEL_DIR / ckpt_name
    print(f"[eval] {ckpt_path}  site={args.site} split={args.split} device={device}")

    model = model_mod.build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"[eval] loaded epoch {state.get('epoch')} "
          f"(dev metric at save: {state.get('metrics', {}).get('f1_mean')})")

    loader_mod = _load_module("01_amber_setup.py")
    site_csv = loader_mod.prepare_site_csv(args.site, all_test=bool(heldout))
    ds = loader_mod.build_amber_dataset(args.split, site_csv)
    loader = torch.utils.data.DataLoader(ds, batch_size=TRAIN.BATCH_SIZE,
                                         shuffle=False, num_workers=TRAIN.NUM_WORKERS)
    mult = DATA.N_STATION if args.per_trace else 1
    print(f"[eval] {args.split} {'traces' if args.per_trace else 'events'}: {len(ds) * mult}")

    preds, labels = run(model, loader, device, args.per_trace, shuf)   # one inference pass, reused below

    if args.sweep:
        pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
        print(f"\n[sweep] threshold scan @ +/-{EVALU.PRIMARY_TOLERANCE_S*1e3:.0f} ms "
              f"(one inference pass re-scored)")
        print(f"  {'thr':>5} | {'P_F1':>6} {'P_R':>6} | {'S_F1':>6} {'S_R':>6} | "
              f"{'F1mean':>7} | noiseFA/ev")
        for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            r = score(preds, labels, thr)
            P, S = r["by_phase"]["P"][pt], r["by_phase"]["S"][pt]
            print(f"  {thr:5.2f} | {P['f1']:6.3f} {P['recall']:6.3f} | "
                  f"{S['f1']:6.3f} {S['recall']:6.3f} | {r['f1_mean_primary']:7.3f} | "
                  f"{r['noise_false_alarm_per_event']:.3f}")
        print("[sweep] pick the thr that maximises F1-mean while noiseFA stays ~0,"
              " then set EvalCfg.PEAK_PROB_THRESHOLD (00) to it.")
        return

    res = score(preds, labels, args.peak_thr)
    res.update(dict(site=args.site, split=args.split, ckpt=args.ckpt,
                    peak_thr=args.peak_thr, heldout=heldout,
                    seed=getattr(args, "seed", None),
                    method=("pertrace" if args.per_trace else ("shuf" if shuf else "array"))))
    _pretty(res)
    out = cfg.LOGS_DIR / f"04_test_results_{args.site}_{args.ckpt}{suffix}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[save] {out}\n[next] run 05_figures_l3.py")


# ----------------------------------------------------------------------------
# synthetic self-test (no amber): verify scoring runs and is sane
# ----------------------------------------------------------------------------
def selftest():
    print("=" * 70)
    print("MOIRAI L3 — 04 evaluation scoring self-test (synthetic)")
    print("=" * 70)
    train_mod = _load_module("03_train_l3.py")
    waves, labels = train_mod._make_synthetic_batch(b=6, T=1024)
    labels = labels.numpy()
    # 'perfect' predictions = the labels themselves -> should give ~F1 1.0
    res = score(labels.copy(), labels, EVALU.PEAK_PROB_THRESHOLD)
    _pretty(res)
    p_f1 = res["by_phase"]["P"][f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"]["f1"]
    assert p_f1 > 0.99, "scoring failed on perfect predictions"
    print("  scoring on perfect predictions: PASS (F1~1.0)")


if __name__ == "__main__":
    main()
