#!/usr/bin/env python
"""06_baseline_pertrace.py -- per-trace deep-learning picker baseline.

Runs an off-the-shelf SeisBench model (PhaseNet or EQTransformer) INDEPENDENTLY
on every 3-C station trace of each AMBER event -- the per-trace paradigm of
Leung et al. (GJI, 2024) -- then assembles the per-station P/S probability into
an (n_phase, n_station, T) map at the AMBER sampling rate and scores it with the
SAME evaluator (04.score) and the SAME tolerance-window F1 used for the
array-level MOIRAI L3 model. Array-level and per-trace results are therefore
directly comparable on identical events and identical metrics.

Why this exists: it answers the reviewer question "is the array-level model
better than just running a per-trace picker?" on the AMBER held-out sites, and
quantifies the gain that exploiting depth-axis moveout provides.

Usage
-----
  python 06_baseline_pertrace.py --site forge_19 --model phasenet     --weights original
  python 06_baseline_pertrace.py --site pnr-1    --model eqtransformer --weights original
  python 06_baseline_pertrace.py --selftest          # plumbing test, no SeisBench needed

Notes
-----
* AMBER is 2000 Hz, components N,E,Z on the channel axis (index 0,1,2). SeisBench
  resamples to the model rate (100 Hz) internally via annotate(); we feed Z,N,E.
* AMBER event windows are short (~1 s = 2048 samples). PhaseNet/EQTransformer
  expect much longer inputs (30/60 s), so each trace is symmetrically zero-padded
  up to the model's required length before annotate(), and the prediction is then
  cropped back to the central window -- this gives the per-trace model its full
  receptive field and is the fair "apply off-the-shelf picker" condition.
* Output: logs/06_baseline_<model>_<site>.json in the 04 schema (+ model fields),
  consumed by 05.figure_baseline_comparison (05_f).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_module(filename: str):
    path = HERE / filename
    name = "m_" + filename.split(".")[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cfg = _load_module("00_config_l3.py")
DATA, EVALU = cfg.DATA, cfg.EVALU

# AMBER channel-axis order is N,E,Z (figure_example treats index 2 as Z).
COMP_AXIS = {"Z": 2, "N": 0, "E": 1}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _fit_length(d: np.ndarray, T: int) -> np.ndarray:
    """Center-crop or center-pad a 1-D array to length T."""
    d = np.asarray(d, np.float32)
    if len(d) == T:
        return d
    if len(d) > T:
        o = (len(d) - T) // 2
        return d[o:o + T]
    out = np.zeros(T, np.float32)
    o = (T - len(d)) // 2
    out[o:o + len(d)] = d
    return out


def _build_model(model_name: str, weights: str, device: str):
    import seisbench.models as sbm
    cls = {"phasenet": sbm.PhaseNet, "eqtransformer": sbm.EQTransformer}[model_name]
    model = cls.from_pretrained(weights)
    model.eval()
    model.to(device)
    return model


def _need_samples(model) -> int:
    """Input length (in AMBER samples @ DATA.FS) the model wants after resampling
    from DATA.FS to the model's native sampling_rate."""
    try:
        in_samp = int(model.in_samples)
        rate = float(model.sampling_rate)
    except Exception:
        in_samp, rate = 3001, 100.0
    return int(np.ceil(in_samp * DATA.FS / rate))


def _annotate_event(model, waves_event: np.ndarray, pad_to: int):
    """waves_event: (3, n_station, T) at DATA.FS, channel order N,E,Z.
    Returns (P_map, S_map), each (n_station, T) probability aligned to the
    original window. Each station's 3 components are symmetrically zero-padded
    to `pad_to` samples, annotated, then the prediction is cropped back to T."""
    import obspy
    _, nS, T = waves_event.shape
    pad = max(0, pad_to - T)
    left = pad // 2
    st = obspy.Stream()
    for s in range(nS):
        for comp in ("Z", "N", "E"):
            x = waves_event[COMP_AXIS[comp], s].astype(np.float32)
            if pad:
                x = np.pad(x, (left, pad - left))
            tr = obspy.Trace(data=np.ascontiguousarray(x))
            tr.stats.sampling_rate = DATA.FS
            tr.stats.network, tr.stats.station = "XX", f"S{s:03d}"
            tr.stats.channel = f"FX{comp}"
            st += tr
    ann = model.annotate(st)
    P_map = np.zeros((nS, T), np.float32)
    S_map = np.zeros((nS, T), np.float32)
    for s in range(nS):
        for ph, out in (("P", P_map), ("S", S_map)):
            sel = ann.select(station=f"S{s:03d}", channel=f"*_{ph}")
            if len(sel):
                d = _fit_length(sel[0].data, T + pad)
                out[s] = d[left:left + T] if pad else d
    return P_map, S_map


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run(args):
    import torch
    from tqdm import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    score_mod = _load_module("04_evaluate_l3.py")
    loader_mod = _load_module("01_amber_setup.py")

    model = _build_model(args.model, args.weights, device)
    pad_to = _need_samples(model)
    print(f"[baseline] {args.model}/{args.weights}  site={args.site}  device={device}\n"
          f"[baseline] pad each {DATA.WINDOW_SAMPLES}-sample window -> {pad_to} "
          f"(model needs {getattr(model, 'in_samples', '?')} @ "
          f"{getattr(model, 'sampling_rate', '?')} Hz)")

    ds = loader_mod.build_amber_dataset(
        "test", loader_mod.prepare_site_csv(args.site, all_test=True))
    N = len(ds) if args.max_ev is None else min(args.max_ev, len(ds))
    preds, labels = [], []
    for i in tqdm(range(N), desc=f"{args.model}:{args.site}"):
        waves, lab = ds[i]
        lab = lab.numpy()
        P_map, S_map = _annotate_event(model, waves.numpy(), pad_to)
        pm = np.zeros_like(lab)
        pm[0], pm[1] = P_map, S_map
        preds.append(pm)
        labels.append(lab)
    preds = np.stack(preds)
    labels = np.stack(labels)

    res = score_mod.score(preds, labels, args.peak_thr)
    res.update(dict(site=args.site, model=args.model, weights=args.weights,
                    ckpt=f"baseline-{args.model}", heldout=args.site))
    score_mod._pretty(res)
    out = cfg.LOGS_DIR / f"06_baseline_{args.model}_{args.site}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[save] {out}\n[next] run 05_figures_l3.py (figure 05_f compares array vs per-trace)")


def selftest():
    """Plumbing test (no SeisBench/obspy/network): a perfect per-trace picker
    placing Gaussian bumps at the ground truth must score F1~1 through 04.score,
    confirming the (n_phase, n_station, T) assembly and the scoring integration.
    Also checks _fit_length crop/pad."""
    assert np.array_equal(_fit_length(np.arange(10), 6), np.arange(2, 8))
    padded = _fit_length(np.arange(4, dtype=float), 8)
    assert padded[:2].sum() == 0 and padded[-2:].sum() == 0 and padded[2:6].sum() == 6
    score_mod = _load_module("04_evaluate_l3.py")
    train_mod = _load_module("03_train_l3.py")
    waves, labels = train_mod._make_synthetic_batch(b=6, T=1024)
    labels = labels.numpy()
    B, _, S, T = labels.shape
    sig = max(1, int(0.01 * DATA.FS))
    tt = np.arange(T)
    preds = np.zeros_like(labels)
    for b in range(B):
        for ch in (0, 1):
            for s in range(S):
                r = labels[b, ch, s]
                if r.max() >= 0.5:
                    c = int(r.argmax())
                    preds[b, ch, s] = np.exp(-0.5 * ((tt - c) / sig) ** 2)
    res = score_mod.score(preds, labels, EVALU.PEAK_PROB_THRESHOLD)
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    f1 = res["by_phase"]["P"][pt]["f1"]
    assert f1 > 0.95, f"plumbing F1 too low: {f1}"
    print(f"  _fit_length crop/pad: PASS")
    print(f"  assemble + 04.score on perfect per-trace picks: F1={f1:.3f} PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DATA.HELDOUT_SITE)
    ap.add_argument("--model", default="phasenet", choices=["phasenet", "eqtransformer"])
    ap.add_argument("--weights", default="original",
                    help="SeisBench pretrained weights name (original/instance/stead/...)")
    ap.add_argument("--peak-thr", type=float, default=EVALU.PEAK_PROB_THRESHOLD)
    ap.add_argument("--max-ev", type=int, default=None,
                    help="cap number of events (default: all, for a paper-grade run)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    run(args)


if __name__ == "__main__":
    main()
