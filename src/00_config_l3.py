"""
00_config_l3.py — MOIRAI L3 single source of truth (config)
============================================================
WHAT THIS SCRIPT DOES
---------------------
Defines every tunable constant for MOIRAI L3 in ONE place so that the loader
(01), model (02), trainer (03), evaluator (04) and figures (05) all import the
same numbers and never drift. Importing this module also resolves the runtime
environment (Google Colab + Drive vs. local sandbox) and creates the project
standard folder layout (src / data / model / PDF / logs).

PROJECT CONTEXT
---------------
MOIRAI is the author's main research line: a velocity-free, waveform-direct approach
to induced-seismicity monitoring. The ladder is
    L1 (done) 3-C geophone, synthetic, P/S separation, +21 dB SI-SNR
    L2 (done) full-scale synthetic DAS (2000 ch), direct-regression 2-D U-Net
    L3 (HERE)  real 3-C downhole vertical arrays (AMBER), P/S PICKING
    L4 (future) real DAS (Otway / FORGE / Curtin)

L3 task = "Plan B" PICKING (not waveform separation): predict per-pixel P / S /
noise probability for a WHOLE vertical array at once, exploiting the depth-axis
MOVEOUT (apparent-velocity difference of P vs S) the way L2 used dense-DAS
spatial coherence. This array-level, moveout-driven design is what distinguishes
MOIRAI L3 from the author's earlier per-trace picker.

DATA = AMBER (Leung et al. 2026, SRL, under review; Bristol). 10 downhole sites,
all 3-C (NEZ), 2000 Hz. We start with a SINGLE site, mseel_3h (most events:
1684; train/dev/test = 1163/268/253; 12 stations -> usable with nstation=12).
Later: train mseel_3h -> transfer-test pnr-1 (0/0/1258, all test) as a strong
unseen-site generalisation claim; and extend to forge_19/22 (the author's own operational setting).

PROJECT STANDARDS ENFORCED HERE
----------------------------
* uv virtual environment; project = MOIRAI_L3; folders src/data/model/PDF/logs.
* numbered scripts, header docstring stating purpose, full scripts (no excerpts).
* figures: English text, white background, legend in the margin, saved as PDF
  in the PDF/ folder (handled in 05).
* models saved in model/, best checkpoint kept every epoch.
* paper-grade data/training volume; honest reporting (median/fail-rate, not just
  mean) — the central L2 lesson.
* no personal addresses or email addresses are stored anywhere.

SITE REALITY (mseel_3h, confirmed via 01 self-test on metadata.csv)
-------------------------------------------------------------------
1684 events = 684 earthquake + 1000 noise; EVERY event has exactly 12 stations
(nstation=12 drops nothing). Earthquake events all have P and S (8-12 stations
with P, 9-12 with S) -> strong moveout. Noise events have no picks -> all-noise
labels = true-negative supervision. Components NEZ, 2000 Hz throughout. S-P is
tiny (median 110 / p95 169 samples), which drives WINDOW/DROPOFF/TOLERANCE.
Split (events): train 1163 / dev 268 / test 253.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ----------------------------------------------------------------------------
# 1. Runtime environment + paths (Colab/Drive vs. local sandbox, auto-detected)
# ----------------------------------------------------------------------------
def _detect_base() -> Path:
    """Return the project root. On Colab, persist under Drive; else use the
    repository folder so the sandbox can run syntax/logic checks unchanged."""
    drive_base = Path("/content/drive/MyDrive/MOIRAI_L3")
    if "google.colab" in sys.modules or Path("/content/drive/MyDrive").exists():
        drive_base.mkdir(parents=True, exist_ok=True)
        return drive_base
    # local / sandbox: project root is the parent of this src/ directory
    return Path(__file__).resolve().parents[1]


BASE_DIR: Path = _detect_base()
SRC_DIR: Path = BASE_DIR / "src"
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "model"
PDF_DIR: Path = BASE_DIR / "PDF"
LOGS_DIR: Path = BASE_DIR / "logs"
for _d in (SRC_DIR, DATA_DIR, MODEL_DIR, PDF_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# AMBER artefacts. On Colab the multi-GB waveforms.hdf5 lives on Drive; locally
# we only ever have metadata.csv (+ a mini extract) for verification.
AMBER_H5: Path = Path(os.environ.get("AMBER_H5", str(DATA_DIR / "waveforms.hdf5")))
AMBER_CSV: Path = Path(os.environ.get("AMBER_CSV", str(DATA_DIR / "metadata.csv")))


# ----------------------------------------------------------------------------
# 2. Reproducibility
# ----------------------------------------------------------------------------
SEED: int = 1234  # fixed for paper reproducibility; set torch/np/random in 03.


# ----------------------------------------------------------------------------
# 3. Dataset / site selection
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DataCfg:
    # --- site ---------------------------------------------------------------
    SITE: str = "mseel_3h"          # single-site mode (most events, 12 stations)
    TRANSFER_SITE: str = "pnr-1"    # unseen-site test for single-site transfer

    # --- LOSO (leave-one-site-out) mode -------------------------------------
    # Sites usable with the 12-station architecture (>=12 channels/event). For
    # the full LOSO matrix we hold out ONE site (--heldout) and train on the
    # rest; sites with >12 channels keep a sequential 12. Excluded:
    #   forge_22  -> only 8 channels/event (cannot fill the 12-station input)
    #   cottonvalley_stgb -> 40 short-trace events (noisy held-out eval + heavy
    #                        zero-padding + log flood); not worth a matrix row.
    USABLE_SITES: tuple = ("mseel_3h", "mseel_5h", "forge_19",
                           "clearfield_mw4", "clearfield_mw6", "pnr-2",
                           "pnr-1", "aneth")
    HELDOUT_SITE: str = "pnr-1"     # default held-out site (used by --loso)

    # --- geometry / sampling ------------------------------------------------
    N_STATION: int = 12             # mseel_3h vertical array; events with <12 dropped
    FS: int = 2000                  # Hz (AMBER native; resample if a site differs)
    N_COMPONENT: int = 3            # NEZ (component_order in metadata)

    # --- window (samples) ---------------------------------------------------
    # CONFIRMED from metadata.csv (01 self-test): mseel_3h S-P is TINY -- median
    # 110 / p95 169 samples (55-85 ms; source close to the array). P/S picks
    # range widely in absolute position (median ~4060, up to ~12000); AMBER's
    # random crop positions the window around the picks. 2048 (1.024 s) easily
    # spans P+S+context, keeps arrivals less sparse than 6144 did, cuts compute
    # ~3x, and is U-Net friendly (2048/32 = 64 at the bottleneck).
    WINDOW_SAMPLES: int = 2048

    # --- AMBER DatasetConfig knobs -----------------------------------------
    NORMALISATION: str = "stationwise"   # per-station normalisation
    SEQUENTIAL_STATIONS: bool = True     # keep depth order (moveout depends on it!)
    # NOTE: fullphasecoverage does NOT filter events; it makes the crop window
    # span BOTH P and S for events that have them (else phase-centric). Good for
    # array P/S picking. Events are dropped only when they have < nstation traces
    # (mseel_3h has exactly 12 per event, so none are dropped).
    FULL_PHASE_COVERAGE: bool = True

    # --- TaperedLabeller ----------------------------------------------------
    # Triangular taper half-width around each pick (samples). MUST stay well
    # below S-P (median 110) or the P and S labels merge -- the kind of label
    # collapse seen in earlier per-trace experiments. 30 = +/-15 ms keeps them separated for
    # the majority while still giving the picker a gradient.
    DROPOFF_SAMPLES: int = 30

    # --- label channels -----------------------------------------------------
    # AMBER TaperedLabeller emits [P, S, noise] with noise = clip(1 - P - S).
    N_PHASE: int = 3
    PHASE_NAMES: tuple = ("P", "S", "noise")


DATA = DataCfg()


# ----------------------------------------------------------------------------
# 4. Model (2-D U-Net, transplanted from L2 RegressionUNet2D; 3-ch prob output)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelCfg:
    IN_CH: int = 3                  # NEZ components -> conv input channels
    OUT_CH: int = 3                 # P / S / noise probabilities -> conv output
    BASE_CH: int = 32               # first-stage feature width
    DEPTH: int = 4                  # number of down/up stages
    CH_MULT: tuple = (1, 2, 4, 8)   # per-stage channel multiplier (len == DEPTH)

    # Image axes for the 2-D U-Net are (station, time):
    #   height = N_STATION (12, SMALL/sparse) ; width = WINDOW_SAMPLES (large).
    # L2 lesson: aggressive stem downsampling kills P/S arrival-time resolution.
    # With only 12 stations we must NOT downsample the station axis; downsample
    # TIME only and gently. STEM_TIME_DOWNSAMPLE=2 mirrors the L2 fix (stem=2).
    STEM_TIME_DOWNSAMPLE: int = 2   # time-axis stride at the stem (station: 1)
    DOWN_STATION: bool = False      # never pool the 12-station axis
    USE_ATTENTION: bool = True      # light self-attention at the bottleneck
    DROPOUT: float = 0.1


MODEL = ModelCfg()


# ----------------------------------------------------------------------------
# 5. Loss
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LossCfg:
    # P/S pixels are RARE vs noise -> class imbalance. focal handles it best;
    # bce and softmax-ce kept for PhaseNet-comparable benchmarking.
    TYPE: str = "focal"             # {"focal", "bce", "ce"}
    FOCAL_GAMMA: float = 2.0
    # per-channel positive weighting [P, S, noise]; lift the sparse phases.
    CLASS_WEIGHT: tuple = (1.0, 1.0, 0.2)


LOSS = LossCfg()


# ----------------------------------------------------------------------------
# 6. Training (paper-grade; A100 on Colab Pro+)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainCfg:
    EPOCHS: int = 200
    BATCH_SIZE: int = 16            # one item = one event (N_STATION,3,T)
    NUM_WORKERS: int = 4
    LR: float = 3.0e-4
    WEIGHT_DECAY: float = 1.0e-4
    GRAD_CLIP: float = 1.0
    WARMUP_EPOCHS: int = 5
    SCHEDULER: str = "cosine"
    AMP: bool = True                # mixed precision on A100
    EMA_DECAY: float = 0.999        # exponential moving average of weights
    EARLY_STOP_PATIENCE: int = 25   # epochs without dev-F1 improvement
    # checkpoints (best is overwritten whenever dev metric improves)
    CKPT_BEST: str = "moirai_l3_best.pt"
    CKPT_LAST: str = "moirai_l3_last.pt"
    CKPT_EMA: str = "moirai_l3_ema_best.pt"
    DEV_METRIC: str = "f1_mean"     # selection metric on dev split


TRAIN = TrainCfg()


# ----------------------------------------------------------------------------
# 7. Evaluation (picking metrics; honest reporting)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalCfg:
    # peak picking on the probability channels
    PEAK_PROB_THRESHOLD: float = 0.30   # min prob to declare a pick
    PEAK_MIN_DISTANCE_SAMPLES: int = 30   # suppress duplicate peaks within 15 ms
    # Detection tolerances (seconds). CONFIRMED from metadata: S-P median is 110
    # samples (55 ms), so tolerances MUST be below that or a "P" match could grab
    # the S arrival. Primary 0.02 s = 40 samples; 0.05 s kept as the loosest.
    TOLERANCES_S: tuple = (0.01, 0.02, 0.05)
    PRIMARY_TOLERANCE_S: float = 0.02
    # report median + fail-rate, not just mean (L2 lesson)
    REPORT_FIELDS: tuple = ("precision", "recall", "f1", "mae_s", "median_ae_s")


EVALU = EvalCfg()


# ----------------------------------------------------------------------------
# 8. Convenience
# ----------------------------------------------------------------------------
def summary() -> str:
    return (
        f"[MOIRAI L3 config]\n"
        f"  base dir         : {BASE_DIR}\n"
        f"  site             : {DATA.SITE}  (transfer -> {DATA.TRANSFER_SITE})\n"
        f"  array            : {DATA.N_STATION} stations x {DATA.N_COMPONENT} comp"
        f" @ {DATA.FS} Hz\n"
        f"  window           : {DATA.WINDOW_SAMPLES} samples"
        f" ({DATA.WINDOW_SAMPLES / DATA.FS:.3f} s)\n"
        f"  labeller dropoff : {DATA.DROPOFF_SAMPLES} samples"
        f" (+/-{DATA.DROPOFF_SAMPLES / DATA.FS * 1e3:.0f} ms)\n"
        f"  model            : 2-D U-Net  in{MODEL.IN_CH}->out{MODEL.OUT_CH}"
        f"  base{MODEL.BASE_CH} depth{MODEL.DEPTH}"
        f"  stem_time x{MODEL.STEM_TIME_DOWNSAMPLE}\n"
        f"  loss             : {LOSS.TYPE}\n"
        f"  train            : {TRAIN.EPOCHS} ep, bs {TRAIN.BATCH_SIZE},"
        f" lr {TRAIN.LR}, EMA {TRAIN.EMA_DECAY}\n"
        f"  H5 / CSV         : {AMBER_H5}\n"
        f"                     {AMBER_CSV}\n"
    )


if __name__ == "__main__":
    print(summary())
