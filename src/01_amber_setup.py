"""
01_amber_setup.py — MOIRAI L3 AMBER dataloader wrapper + verification self-test
===============================================================================
WHAT THIS SCRIPT DOES
---------------------
Wraps AMBER's event-centric PyTorch dataset (Leung et al. 2026) so the rest of
the pipeline sees a clean, config-driven interface that yields, per event:

    waves  : float32 tensor (N_COMPONENT, N_STATION, T)   # NEZ x stations x time
    labels : float32 tensor (N_PHASE,    N_STATION, T)   # [P, S, noise] probs

AMBER itself returns waves as (n_station, 3, n_datapoints) and TaperedLabeller
returns per-station [P, S, noise]; this wrapper PERMUTES both to channel-first
(C, station, T) so the 2-D U-Net (02) reads (components -> conv channels,
stations -> image height, time -> image width) and exploits depth-axis MOVEOUT.

Site selection is done WITHOUT relying on any unverified AMBER internal filter:
we write a site-filtered metadata CSV (data/metadata_<site>.csv) keeping the
`split` column and hand THAT to AMBER. waveforms.hdf5 is shared (trace_name
encodes the bucket), so a filtered CSV is sufficient and robust.

THE SELF-TEST (run `python 01_amber_setup.py`) IS HANDOVER STEP 2:
It does not assume the documented API is exactly right — it INTROSPECTS the real
amber signatures, prints them, then:
  (a) reads metadata.csv, filters to SITE, prints split counts, per-event station
      counts, and the P/S arrival-sample distribution  -> sets WINDOW_SAMPLES /
      DROPOFF_SAMPLES correctly;
  (b) builds the train dataset and pulls one item, printing real shapes / dtypes
      / value ranges and asserting the (C, station, T) layout.
Every stage is guarded so a partial environment (e.g. sandbox without amber or
without waveforms.hdf5) still yields a useful report instead of a hard crash.

NOTE: amber is imported LAZILY inside functions, so 02/03 can import the wrapper
helpers even where amber is not installed (sandbox). Real loading needs amber +
waveforms.hdf5 (Colab/Drive).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# import the numbered config module by path (leading digit -> not importable)
# ----------------------------------------------------------------------------
def _load_module(filename: str):
    path = Path(__file__).with_name(filename)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod  # register BEFORE exec so dataclasses can resolve types
    spec.loader.exec_module(mod)
    return mod


cfg = _load_module("00_config_l3.py")
DATA, MODEL, TRAIN = cfg.DATA, cfg.MODEL, cfg.TRAIN


# ----------------------------------------------------------------------------
# site-filtered metadata CSV
# ----------------------------------------------------------------------------
def prepare_site_csv(site: str = DATA.SITE,
                     full_csv: Path = cfg.AMBER_CSV,
                     out_dir: Path = cfg.DATA_DIR,
                     all_test: bool = False) -> Path:
    """Write data/metadata_<site>.csv keeping only rows of `site` (column
    'dataset'). If all_test, mark every row split='test' (used for a held-out
    LOSO site so the WHOLE site is evaluated, not just its ~15% test split --
    valid because a held-out site is entirely unseen in training)."""
    tag = f"{site}_alltest" if all_test else site
    return prepare_sites_csv([site], tag=tag, full_csv=full_csv,
                             out_dir=out_dir, all_test=all_test)


def prepare_sites_csv(sites, tag: str,
                      full_csv: Path = cfg.AMBER_CSV,
                      out_dir: Path = cfg.DATA_DIR,
                      all_test: bool = False) -> Path:
    """Write data/metadata_<tag>.csv keeping rows whose 'dataset' is in `sites`
    plus ALL columns. If all_test, overwrite the 'split' column to 'test'."""
    import pandas as pd
    out_dir = Path(out_dir)
    out_path = out_dir / f"metadata_{tag}.csv"
    df = pd.read_csv(full_csv)
    if "dataset" not in df.columns:
        raise KeyError(f"'dataset' column not in {full_csv}; columns={list(df.columns)}")
    sub = df[df["dataset"].isin(list(sites))].copy()
    if len(sub) == 0:
        raise ValueError(f"no rows for sites {sites} in {full_csv}")
    if all_test and "split" in sub.columns:
        sub["split"] = "test"
    sub.to_csv(out_path, index=False)
    return out_path


# ----------------------------------------------------------------------------
# channel-first permutation wrapper around the AMBER dataset
# ----------------------------------------------------------------------------
class PermutedAMBER(Dataset):
    """Wrap an AMBER dataset and return (waves[C,S,T], labels[Ph,S,T]) tensors.

    AMBER waves come as (n_station, n_component, T); TaperedLabeller labels as
    (n_station, n_phase, T). We move the per-station channel axis to the front
    so the 2-D U-Net sees (channel, station, time). Axis identification is by
    size (station == N_STATION == 12; channel == 3; time == largest), which is
    unambiguous because 12 != 3 != T. The self-test asserts the result."""

    def __init__(self, amber_dataset, n_station=DATA.N_STATION):
        self.ds = amber_dataset
        self.n_station = n_station

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _to_channel_first(x: torch.Tensor, n_station: int) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3-D (station, ch, time), got {tuple(x.shape)}")
        # find station axis (size == n_station) and time axis (largest)
        sizes = list(x.shape)
        t_axis = int(np.argmax(sizes))
        s_candidates = [i for i, s in enumerate(sizes) if s == n_station and i != t_axis]
        if not s_candidates:
            raise ValueError(
                f"no axis of size n_station={n_station} in {sizes}; "
                "check N_STATION / event station count")
        s_axis = s_candidates[0]
        c_axis = ({0, 1, 2} - {t_axis, s_axis}).pop()
        return x.permute(c_axis, s_axis, t_axis).contiguous()

    def __getitem__(self, idx):
        item = self.ds[idx]
        # AMBER returns either (waves, labels) or a dict; handle both.
        if isinstance(item, dict):
            waves = item.get("waves", item.get("X"))
            labels = item.get("labels", item.get("y"))
        else:
            waves, labels = item[0], item[1]
        waves = torch.as_tensor(np.asarray(waves), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        waves = self._to_channel_first(waves, self.n_station)
        labels = self._to_channel_first(labels, self.n_station)
        return waves, labels


# ----------------------------------------------------------------------------
# dataset / dataloader builders (documented AMBER API)
# ----------------------------------------------------------------------------
def build_amber_dataset(mode: str, site_csv: Path, h5_path: Path = cfg.AMBER_H5):
    """Build a PermutedAMBER for split `mode` in {'train','dev','test'} using the
    documented AMBER API. amber is imported lazily so importing this module does
    not require amber to be installed."""
    from amber.dataloaders import AMBER, DatasetConfig
    from amber.Labeller import create_labeller, LabellerConfig

    dataset_cfg = DatasetConfig(
        windowlength=DATA.WINDOW_SAMPLES,
        nstation=DATA.N_STATION,
        normalisation=DATA.NORMALISATION,
        sequentialstations=DATA.SEQUENTIAL_STATIONS,
        fullphasecoverage=DATA.FULL_PHASE_COVERAGE,
    )
    labeller = create_labeller(
        "tapered_labeller",
        LabellerConfig(dynamic_params={
            "dropoff": DATA.DROPOFF_SAMPLES,
            "windowlength": DATA.WINDOW_SAMPLES,
        }),
    )
    amber_ds = AMBER(
        dataset_cfg,
        str(h5_path),
        str(site_csv),
        labeller,
        augmentations=None,
        mode=mode,
    )
    return PermutedAMBER(amber_ds, n_station=DATA.N_STATION)


def loso_train_sites(heldout: str):
    """Training sites for a LOSO fold = USABLE_SITES minus the held-out site."""
    if heldout not in DATA.USABLE_SITES:
        raise ValueError(f"heldout '{heldout}' not in USABLE_SITES {DATA.USABLE_SITES}")
    return tuple(s for s in DATA.USABLE_SITES if s != heldout)


def pertrace_split(t):
    """(B, C, S, T) -> (B*S, C, 1, T): make every station its own sample.

    Used for the per-trace ABLATION. Applied IN THE TRAINING/EVAL STEP (after
    loading a normal array batch) so each event is read from HDF5 only ONCE,
    then split on-device into S single-station samples -- ~N_STATION x less I/O
    than expanding in the Dataset. With station axis = 1 the station-spanning
    convolutions see only zero-padding, so no cross-station moveout is used."""
    B, C, S, T = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * S, C, T).unsqueeze(2)


def build_dataloaders(site: str = DATA.SITE, loso: bool = False, heldout: str = None,
                      per_trace: bool = False):
    """Return train/dev/test DataLoaders (always array-shaped (C,S,T)).

    single-site (loso=False): all three splits come from `site`.
    LOSO (loso=True): train/dev come from USABLE_SITES minus `heldout`; test
    comes from the held-out unseen site `heldout` (default DATA.HELDOUT_SITE).
    per_trace is handled in the train/eval step via pertrace_split(), so the
    loaders themselves are identical -- this keeps HDF5 I/O efficient."""
    if loso:
        heldout = heldout or DATA.HELDOUT_SITE
        train_sites = loso_train_sites(heldout)
        train_csv = prepare_sites_csv(train_sites, tag=f"train_ex_{heldout}")
        test_csv = prepare_site_csv(heldout, all_test=True)  # full held-out site
        specs = {"train": train_csv, "dev": train_csv, "test": test_csv}
    else:
        site_csv = prepare_site_csv(site)
        specs = {"train": site_csv, "dev": site_csv, "test": site_csv}
    loaders = {}
    for mode, csv_path in specs.items():
        ds = build_amber_dataset(mode, csv_path)
        loaders[mode] = DataLoader(
            ds,
            batch_size=TRAIN.BATCH_SIZE,
            shuffle=(mode == "train"),
            num_workers=TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=(mode == "train"),
        )
    return loaders


# ----------------------------------------------------------------------------
# self-test == handover step 2
# ----------------------------------------------------------------------------
def _print_sig(name, obj):
    try:
        print(f"  {name:16s}: {name}{inspect.signature(obj)}")
    except (TypeError, ValueError):
        print(f"  {name:16s}: <signature unavailable>")


def _inspect_metadata(site: str):
    import pandas as pd
    print("\n[B] metadata.csv inspection")
    if not Path(cfg.AMBER_CSV).exists():
        print(f"    SKIP: {cfg.AMBER_CSV} not found (upload metadata.csv).")
        return
    df = pd.read_csv(cfg.AMBER_CSV)
    print(f"    full rows={len(df)}  columns={list(df.columns)[:12]}...")
    if "dataset" in df.columns:
        print("    events per site:")
        for s, n in df["dataset"].value_counts().items():
            mark = "  <-- start" if s == site else ""
            print(f"      {s:18s}: {n}{mark}")
        sub = df[df["dataset"] == site]
    else:
        sub = df
    if "split" in sub.columns:
        print(f"    {site} split counts: "
              f"{dict(sub['split'].value_counts())}")
    if "event_id" in sub.columns:
        per_event = sub.groupby("event_id").size()
        print(f"    {site} stations/event: min={per_event.min()} "
              f"median={int(per_event.median())} max={per_event.max()} "
              f"(events={per_event.size})")
        usable = (per_event >= DATA.N_STATION).sum()
        print(f"    events with >= N_STATION({DATA.N_STATION}): {usable}")
    for col in ("trace_p_arrival_sample", "trace_s_arrival_sample"):
        if col in sub.columns:
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(v):
                print(f"    {col}: min={v.min():.0f} median={v.median():.0f} "
                      f"p95={v.quantile(0.95):.0f} max={v.max():.0f}  "
                      f"-> WINDOW_SAMPLES must exceed max + S-coda")
    if {"trace_p_arrival_sample", "trace_s_arrival_sample"} <= set(sub.columns):
        p = pd.to_numeric(sub["trace_p_arrival_sample"], errors="coerce")
        s = pd.to_numeric(sub["trace_s_arrival_sample"], errors="coerce")
        sp = (s - p).dropna()
        if len(sp):
            print(f"    S-P (samples): median={sp.median():.0f} "
                  f"p95={sp.quantile(0.95):.0f}  "
                  f"-> DROPOFF_SAMPLES should stay below this")


def selftest():
    print("=" * 74)
    print("MOIRAI L3 — 01 AMBER loader verification (handover step 2)")
    print("=" * 74)
    print(cfg.summary())

    # [A] amber API introspection ------------------------------------------
    print("[A] amber API introspection")
    try:
        from amber.dataloaders import AMBER, DatasetConfig
        from amber.Labeller import create_labeller, LabellerConfig
        _print_sig("DatasetConfig", DatasetConfig)
        _print_sig("LabellerConfig", LabellerConfig)
        _print_sig("create_labeller", create_labeller)
        _print_sig("AMBER", AMBER)
        amber_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"    amber not importable: {type(e).__name__}: {e}")
        print("    (install with `uv pip install /path/to/AMBER_Public`)")
        amber_ok = False

    # [B] metadata inspection ----------------------------------------------
    try:
        _inspect_metadata(DATA.SITE)
    except Exception as e:  # noqa: BLE001
        print(f"    metadata inspection failed: {type(e).__name__}: {e}")

    # [C] one real item ----------------------------------------------------
    print("\n[C] one-item load + layout check")
    if not amber_ok:
        print("    SKIP: amber unavailable.")
    elif not Path(cfg.AMBER_H5).exists():
        print(f"    SKIP: {cfg.AMBER_H5} not found (waveforms.hdf5 lives on Drive).")
    else:
        try:
            site_csv = prepare_site_csv(DATA.SITE)
            print(f"    wrote site CSV: {site_csv}")
            ds = build_amber_dataset("train", site_csv)
            print(f"    train events (>= N_STATION): {len(ds)}")
            waves, labels = ds[0]
            print(f"    waves : shape={tuple(waves.shape)} dtype={waves.dtype} "
                  f"range=[{waves.min():.3f},{waves.max():.3f}]")
            print(f"    labels: shape={tuple(labels.shape)} dtype={labels.dtype} "
                  f"range=[{labels.min():.3f},{labels.max():.3f}]")
            C, S, T = waves.shape
            assert C == DATA.N_COMPONENT, f"component axis {C} != {DATA.N_COMPONENT}"
            assert S == DATA.N_STATION, f"station axis {S} != {DATA.N_STATION}"
            assert labels.shape[0] == DATA.N_PHASE, "phase axis mismatch"
            assert labels.shape[1] == DATA.N_STATION, "label station axis mismatch"
            ch_sum = labels.sum(0)  # [P,S,noise] should ~sum to 1 per pixel
            print(f"    per-pixel channel sum: mean={ch_sum.mean():.3f} "
                  f"(expect ~1.0 for tapered [P,S,noise])")
            print("    LAYOUT CHECK: PASS")
        except AssertionError as e:
            print(f"    LAYOUT CHECK: FAIL -> {e}")
        except Exception as e:  # noqa: BLE001
            print(f"    one-item load failed: {type(e).__name__}: {e}")

    print("\n[done] If [A]/[C] PASS and [B] confirms WINDOW/DROPOFF, proceed to 03.")


if __name__ == "__main__":
    selftest()
