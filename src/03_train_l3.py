"""
03_train_l3.py — MOIRAI L3 trainer (array P/S picking)
=======================================================
WHAT THIS SCRIPT DOES
---------------------
Trains MoiraiPickerL3 (02) on AMBER (01) to predict per-pixel [P, S, noise]
probability maps for a whole vertical array. Production machinery:

  * loss: focal (default; handles the P/S-vs-noise class imbalance), with BCE
    and softmax-CE alternatives for PhaseNet-comparable benchmarking. Targets
    are SOFT (tapered), handled correctly by all three.
  * mixed precision (AMP) on A100, gradient clipping, AdamW.
  * warmup + cosine LR schedule.
  * EMA of weights (decay 0.999); both raw and EMA models are scored on dev.
  * BEST-checkpoint saving every epoch (model/moirai_l3_best.pt and
    moirai_l3_ema_best.pt) keyed on dev F1-mean; plus moirai_l3_last.pt.
  * early stopping on dev F1-mean.

DEV METRIC (selection) — picking F1:
  Each station has at most one P and one S in the window, so a pick is the
  argmax-over-time of a phase channel when its peak exceeds a threshold. A
  predicted pick matches ground truth (recovered as the argmax of the tapered
  label channel) when within the tolerance window. We report precision / recall
  / F1 per phase and select on their mean. (04 produces the publication-grade
  version: multiple tolerances, MAE, median, fail-rate — the L2 honest-reporting
  lesson.)

PAPER-GRADE VOLUME: mseel_3h has train/dev/test = 1163/268/253 array EVENTS
(each event = 12 stations x 3 comp), 200 epochs with early stopping. Later:
transfer-test on the unseen site pnr-1.

RUN MODES
  python 03_train_l3.py                 # full training (needs amber + h5; Colab)
  python 03_train_l3.py --smoke         # 2-epoch gate on real data
  python 03_train_l3.py --selftest      # synthetic overfit, NO amber needed
                                        #   (verifies loss/EMA/scheduler wiring)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# version-robust AMP (new torch.amp API, fallback to torch.cuda.amp)
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    def make_scaler(enabled):
        return _GradScaler("cuda", enabled=enabled)

    def amp_ctx(enabled):
        return _autocast("cuda", enabled=enabled)
except Exception:  # noqa: BLE001  (older torch)
    def make_scaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def amp_ctx(enabled):
        return torch.cuda.amp.autocast(enabled=enabled)


# ----------------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# loss (soft targets supported by all three variants)
# ----------------------------------------------------------------------------
def make_loss():
    w = torch.tensor(LOSS.CLASS_WEIGHT, dtype=torch.float32).view(1, -1, 1, 1)
    gamma = LOSS.FOCAL_GAMMA
    kind = LOSS.TYPE

    def loss_fn(logits, targets):
        cw = w.to(logits.device)
        if kind == "bce":
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            return (cw * bce).mean()
        if kind == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)   # soft-target generalisation
            mod = (1.0 - p_t).clamp_min(1e-6) ** gamma
            return (cw * mod * bce).mean()
        if kind == "ce":  # soft cross-entropy over the 3 channels (PhaseNet style)
            logp = F.log_softmax(logits, dim=1)
            return -(cw * targets * logp).sum(1).mean()
        raise ValueError(f"unknown LOSS.TYPE {kind}")

    return loss_fn


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = TRAIN.EMA_DECAY):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m, alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


# ----------------------------------------------------------------------------
# scheduler (linear warmup -> cosine)
# ----------------------------------------------------------------------------
def build_scheduler(optimizer, epochs: int, warmup: int):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        prog = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------------
# picking + scoring (dev-selection metric)
# ----------------------------------------------------------------------------
def _single_pick(prob_2d: np.ndarray, thr: float):
    """prob_2d: (n_station, T) for one phase. Return per-station (idx or -1)."""
    picks = np.full(prob_2d.shape[0], -1, dtype=np.int64)
    mx = prob_2d.max(axis=1)
    am = prob_2d.argmax(axis=1)
    picks[mx >= thr] = am[mx >= thr]
    return picks


def score_picks(pred_prob: np.ndarray, label: np.ndarray, tol_samples: int,
                peak_thr: float, label_thr: float = 0.5):
    """pred_prob, label: (B, 3, n_station, T). Score P (ch0) and S (ch1)."""
    out = {}
    for ch, name in ((0, "P"), (1, "S")):
        tp = fp = fn = 0
        abs_err = []
        for b in range(pred_prob.shape[0]):
            pred = _single_pick(pred_prob[b, ch], peak_thr)
            gt = _single_pick(label[b, ch], label_thr)
            for st in range(pred.shape[0]):
                has_gt, has_pred = gt[st] >= 0, pred[st] >= 0
                if has_gt and has_pred:
                    if abs(pred[st] - gt[st]) <= tol_samples:
                        tp += 1
                        abs_err.append(abs(pred[st] - gt[st]))
                    else:
                        fp += 1
                        fn += 1
                elif has_pred and not has_gt:
                    fp += 1
                elif has_gt and not has_pred:
                    fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                         mae_s=(float(np.mean(abs_err)) / DATA.FS) if abs_err else float("nan"))
    out["f1_mean"] = 0.5 * (out["P"]["f1"] + out["S"]["f1"])
    return out


def _pt_split(t):
    """(B,C,S,T) -> (B*S,C,1,T): per-trace ablation split (see 01.pertrace_split)."""
    B, C, S, T = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * S, C, T).unsqueeze(2)


def _shuffle_stations(waves, lab):
    """Permute the station axis (dim 2) identically for waves AND labels, with an
    INDEPENDENT permutation per sample. This destroys across-station moveout
    coherence while preserving array size and the per-station waveform/label
    pairing -- the clean control isolating "moveout" from "array size" (Tier-1
    ablation requested by GJI review). Used in array mode only."""
    B, C, S, T = waves.shape
    w, l = waves.clone(), lab.clone()
    for b in range(B):
        perm = torch.randperm(S, device=waves.device)
        w[b] = waves[b, :, perm, :]
        l[b] = lab[b, :, perm, :]
    return w, l


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, per_trace=False, shuffle_stations=False):
    model.eval()
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    losses, preds, labels = [], [], []
    for waves, lab in loader:
        waves, lab = waves.to(device), lab.to(device)
        if per_trace:
            waves, lab = _pt_split(waves), _pt_split(lab)
        elif shuffle_stations:
            waves, lab = _shuffle_stations(waves, lab)
        logits = model(waves)
        losses.append(loss_fn(logits, lab).item())
        preds.append(model_mod.MoiraiPickerL3.activate(logits).cpu().numpy())
        labels.append(lab.cpu().numpy())
    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    metrics = score_picks(preds, labels, tol, EVALU.PEAK_PROB_THRESHOLD)
    metrics["loss"] = float(np.mean(losses))
    return metrics


# ----------------------------------------------------------------------------
# checkpoint
# ----------------------------------------------------------------------------
def save_ckpt(path: Path, model, epoch: int, metrics: dict):
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "metrics": metrics, "site": DATA.SITE,
                "config": {"window": DATA.WINDOW_SAMPLES, "loss": LOSS.TYPE}}, path)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def train(args):
    seed = getattr(args, "seed", None)
    set_seed(seed if seed is not None else cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heldout = getattr(args, "heldout", None)
    loso = getattr(args, "loso", False) or bool(heldout)
    if loso:
        heldout = heldout or DATA.HELDOUT_SITE
    per_trace = getattr(args, "per_trace", False)
    shuffle_stations = getattr(args, "shuffle_stations", False) and not per_trace
    suffix = (("_pertrace" if per_trace else "")
              + ("_shuf" if shuffle_stations else "")
              + (f"_loso_{heldout}" if loso else "")
              + (f"_seed{seed}" if seed is not None else ""))

    loader_mod = _load_module("01_amber_setup.py")
    tag = ("PER-TRACE ablation " if per_trace else ("STATION-SHUFFLE control " if shuffle_stations else ""))
    if loso:
        train_sites = loader_mod.loso_train_sites(heldout)
        print(f"[train] {tag}LOSO  device={device}  heldout={heldout}\n"
              f"        train_sites={list(train_sites)}")
        loaders = loader_mod.build_dataloaders(loso=True, heldout=heldout, per_trace=per_trace)
    else:
        print(f"[train] {tag}device={device}  site={args.site}")
        loaders = loader_mod.build_dataloaders(args.site, per_trace=per_trace)
    unit = "traces" if per_trace else "events"
    mult = DATA.N_STATION if per_trace else 1
    print(f"[train] {unit}: " + ", ".join(
        f"{m}={len(loaders[m].dataset) * mult}" for m in ("train", "dev", "test")))

    def ckpt_path(name):  # insert the LOSO/heldout suffix before ".pt"
        return cfg.MODEL_DIR / name.replace(".pt", f"{suffix}.pt")

    model = model_mod.build_model().to(device)
    ema = EMA(model)
    loss_fn = make_loss()
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN.LR,
                            weight_decay=TRAIN.WEIGHT_DECAY)
    epochs = args.epochs or TRAIN.EPOCHS
    sched = build_scheduler(opt, epochs, TRAIN.WARMUP_EPOCHS)
    scaler = make_scaler(TRAIN.AMP and device == "cuda")

    best_f1, best_ema_f1, bad = -1.0, -1.0, 0
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        n_step, n_skip = 0, 0
        for waves, lab in loaders["train"]:
            waves, lab = waves.to(device), lab.to(device)
            if per_trace:
                waves, lab = _pt_split(waves), _pt_split(lab)
            elif shuffle_stations:
                waves, lab = _shuffle_stations(waves, lab)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(TRAIN.AMP and device == "cuda"):
                logits = model(waves)
                loss = loss_fn(logits, lab)
            if not torch.isfinite(loss):     # guard: never let a NaN/Inf batch
                n_skip += 1                  # corrupt the weights (AMP + focal can
                continue                     # overflow on a degenerate trace batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run += loss.item()
            n_step += 1
        sched.step()

        dev_raw = evaluate(model, loaders["dev"], device, loss_fn, per_trace, shuffle_stations)
        dev_ema = evaluate(ema.shadow, loaders["dev"], device, loss_fn, per_trace, shuffle_stations)
        dt = time.time() - t0
        skip_note = f" | skipped {n_skip}" if n_skip else ""
        print(f"  ep {ep:3d} | train {run/max(1,n_step):.4f} "
              f"| dev F1 {dev_raw['f1_mean']:.3f} (P {dev_raw['P']['f1']:.3f} "
              f"S {dev_raw['S']['f1']:.3f}) | ema F1 {dev_ema['f1_mean']:.3f} "
              f"| {dt:.0f}s{skip_note}")
        history.append({"epoch": ep, "train_loss": run / max(1, len(loaders['train'])),
                        "dev": dev_raw, "dev_ema": dev_ema})

        improved = False
        if dev_raw["f1_mean"] > best_f1:
            best_f1 = dev_raw["f1_mean"]
            save_ckpt(ckpt_path(TRAIN.CKPT_BEST), model, ep, dev_raw)
            improved = True
        if dev_ema["f1_mean"] > best_ema_f1:
            best_ema_f1 = dev_ema["f1_mean"]
            save_ckpt(ckpt_path(TRAIN.CKPT_EMA), ema.shadow, ep, dev_ema)
            improved = True
        save_ckpt(ckpt_path(TRAIN.CKPT_LAST), model, ep, dev_raw)

        bad = 0 if improved else bad + 1
        if bad >= TRAIN.EARLY_STOP_PATIENCE:
            print(f"[train] early stop at epoch {ep} (no dev improvement for "
                  f"{TRAIN.EARLY_STOP_PATIENCE}).")
            break
        if args.smoke and ep >= 1:
            print("[train] smoke gate reached; stopping.")
            break

    hist_name = f"03_train_history{suffix}.json"
    json.dump(history, open(cfg.LOGS_DIR / hist_name, "w"), indent=2)
    print(f"[train] best dev F1 raw={best_f1:.3f} ema={best_ema_f1:.3f}")
    print(f"[train] checkpoints in {cfg.MODEL_DIR} (suffix '{suffix}')")
    print("[next] run 04_evaluate_l3.py"
          + (f" --heldout {heldout}" if loso else ""))


# ----------------------------------------------------------------------------
# synthetic self-test (NO amber): verify the training machinery overfits
# ----------------------------------------------------------------------------
def _make_synthetic_batch(b=4, T=1024, device="cpu"):
    """Random NEZ noise with planted P/S Gaussian bumps that move out across
    stations; tapered [P,S,noise] labels. Lets us confirm loss decreases."""
    S, fs = DATA.N_STATION, DATA.FS
    waves = 0.1 * torch.randn(b, 3, S, T)
    labels = torch.zeros(b, 3, S, T)
    drop = DATA.DROPOFF_SAMPLES
    tri = torch.clamp(1 - torch.arange(-drop, drop + 1).abs() / drop, min=0)
    for bi in range(b):
        p0 = np.random.randint(T // 4, T // 2)
        s0 = p0 + np.random.randint(150, 350)
        for st in range(S):
            tp = p0 + st * 6          # P moveout (steeper apparent velocity)
            ts = s0 + st * 14         # S moveout (slower)
            for t0, ch in ((tp, 0), (ts, 1)):
                lo, hi = max(0, t0 - drop), min(T, t0 + drop + 1)
                seg = tri[(lo - (t0 - drop)):(hi - (t0 - drop))]
                labels[bi, ch, st, lo:hi] = seg
                waves[bi, :, st, lo:hi] += seg.unsqueeze(0)  # bump in all comps
    labels[:, 2] = (1 - labels[:, 0] - labels[:, 1]).clamp(0, 1)
    return waves.to(device), labels.to(device)


def selftest():
    print("=" * 70)
    print("MOIRAI L3 — 03 trainer SYNTHETIC self-test (no amber)")
    print("=" * 70)
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_mod.build_model().to(device)
    ema = EMA(model)
    loss_fn = make_loss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    waves, labels = _make_synthetic_batch(b=4, T=1024, device=device)
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))

    first = last = None
    for step in range(40):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(waves)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN.GRAD_CLIP)
        opt.step()
        ema.update(model)
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0 or step == 39:
            model.eval()
            with torch.no_grad():
                prob = model_mod.MoiraiPickerL3.activate(model(waves)).cpu().numpy()
            m = score_picks(prob, labels.cpu().numpy(), tol, EVALU.PEAK_PROB_THRESHOLD)
            print(f"  step {step:2d} | loss {loss.item():.4f} "
                  f"| F1 P {m['P']['f1']:.3f} S {m['S']['f1']:.3f} "
                  f"mean {m['f1_mean']:.3f}")
    print(f"  loss {first:.4f} -> {last:.4f}")
    assert last < first, "loss did not decrease — training wiring broken"
    print("  EMA/scheduler/loss/backward wiring: PASS "
          "(overfit drives F1 up as expected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DATA.SITE)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--loso", action="store_true",
                    help="leave-one-site-out using the default DATA.HELDOUT_SITE")
    ap.add_argument("--heldout", default=None,
                    help="LOSO with this specific held-out site (full-matrix fold)")
    ap.add_argument("--per-trace", dest="per_trace", action="store_true",
                    help="ablation: train on single-station samples (no moveout)")
    ap.add_argument("--shuffle-stations", dest="shuffle_stations", action="store_true",
                    help="control: permute station order (destroy moveout, keep array size)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; appends _seed<n> to checkpoint/result names")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        train(args)


if __name__ == "__main__":
    main()
