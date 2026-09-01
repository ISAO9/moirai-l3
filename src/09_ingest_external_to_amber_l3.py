"""
09_ingest_external_to_amber_l3.py — convert ANY external borehole array waveforms
into the MOIRAI L3 picking input format (and, optionally, AMBER/SeisBench format)
=================================================================================
WHAT THIS SCRIPT DOES
---------------------
You have raw waveform data from a borehole / vertical microseismic array (e.g. a
new shale-gas monitoring well: 12 depth-ordered stations, 3 components). The
MOIRAI L3 picker expects, per event, a tensor

    waves : float32 (N_COMPONENT=3 [N,E,Z], N_STATION=12, T=2048) at 2000 Hz

This tool turns your data into exactly that, so the trained L3 model can pick P/S
on it ZERO-SHOT (no velocity model needed). It does the unglamorous-but-critical
work: read various formats, map/realise channels to N,E,Z, order stations by
depth, resample to 2000 Hz, cut 2048-sample windows around event triggers, and
write a clean, self-describing HDF5 + metadata CSV that the inference harness
(05x_infer) reads directly. It can ALSO export to SeisBench/AMBER format (so you
can plug into the AMBER training pipeline) when `seisbench` is installed.

It deliberately does NOT pre-normalise amplitudes by default: the L3 loader does
per-station normalisation at load time, so we store physical (de-meaned) traces
and let the model side normalise — avoiding double normalisation.

INPUTS IT UNDERSTANDS
---------------------
* miniSEED / SAC / SEG-Y / any ObsPy-readable file or glob  (needs `obspy`)
* a directory or glob of .npy / .npz arrays
* a single .h5/.hdf5 with a 3-D array (events|station|comp ordering configurable)
You provide a depth-ordered station list and a component mapping; for ObsPy
inputs the script matches SEED channel codes; for array inputs it trusts the axis
order you declare.

EVENT WINDOWING (pick ONE)
--------------------------
* --triggers CSV   : event_id, trigger_time (UTC) or trigger_sample [, per-station
                     p_sample_<STA> / s_sample_<STA> if you happen to have picks]
* --sliding        : tile continuous data into non-overlapping (or --stride) windows
* --sta-lta        : simple STA/LTA auto-trigger on the stack (scipy only; optional)

OUTPUTS
-------
* data/external_<site>_waveforms.h5   dataset 'waveforms' (Nev, 3, 12, 2048) f32
  + event_id, station_codes, station_depth, window_start_sample, fs;
  optional p_arrival_sample / s_arrival_sample (Nev, 12), NaN where unknown.
* data/external_<site>_metadata.csv   one row per event (split='test' for zero-shot)
* PDF/external_<site>_qc.pdf          (--qc) one converted event, English labels,
  white background, legend in the margin — to eyeball station order / window.
* (optional) --seisbench-out DIR      waveforms.hdf5 + metadata.csv in SeisBench
  grouping=event layout (column names AMBER reads): needs `seisbench`.

PROJECT STANDARDS
--------------
uv env; project MOIRAI_L3; folders src/data/model/PDF/logs; numbered script with a
header docstring; full script (no excerpts); figures English + white background +
legend in the margin, saved as PDF in PDF/; no personal addresses or email addresses.
Run `python 09_ingest_external_to_amber_l3.py --selftest` to verify end-to-end on
synthetic data WITHOUT any external dependency (no obspy/seisbench needed).

USAGE EXAMPLES
--------------
# miniSEED, 12 stations in depth order, channels DH[ZNE], triggers from a catalogue
python 09_ingest_external_to_amber_l3.py \
    --input "raw/*.mseed" --reader obspy --site shalegas_w1 \
    --stations ST01,ST02,ST03,ST04,ST05,ST06,ST07,ST08,ST09,ST10,ST11,ST12 \
    --depths  300,330,360,390,420,450,480,510,540,570,600,630 \
    --channel-order Z,N,E --triggers raw/catalogue.csv --qc

# already an array (Nev, station, comp, time) at 4000 Hz in one .npy
python 09_ingest_external_to_amber_l3.py \
    --input raw/events.npy --reader npy --array-axes event,station,comp,time \
    --src-fs 4000 --site shalegas_w1 --qc
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# stay in sync with 00_config_l3.py when present; otherwise use L3 defaults
# --------------------------------------------------------------------------
def _load_l3_config():
    cfg_path = Path(__file__).with_name("00_config_l3.py")
    try:
        spec = importlib.util.spec_from_file_location("cfg_l3", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cfg_l3"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return (int(mod.DATA.FS), int(mod.DATA.WINDOW_SAMPLES),
                int(mod.DATA.N_STATION), int(mod.MODEL.IN_CH))
    except Exception:
        return (2000, 2048, 12, 3)

TARGET_FS, WINDOW, N_STATION, N_COMPONENT = _load_l3_config()
COMPONENT_ORDER = ["N", "E", "Z"]  # L3 loader layout: (NEZ, station, time)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "PDF"
for _d in (DATA_DIR, PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# signal helpers
# ==========================================================================
def resample_to_target(x: np.ndarray, src_fs: float, axis: int = -1) -> np.ndarray:
    """Anti-aliased rational resample of `x` from src_fs to TARGET_FS along `axis`."""
    if abs(src_fs - TARGET_FS) < 1e-6:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = math.gcd(int(round(src_fs)), int(TARGET_FS))
    up, down = int(TARGET_FS) // g, int(round(src_fs)) // g
    return resample_poly(x, up, down, axis=axis).astype(np.float32)


def demean_detrend(x: np.ndarray, axis: int = -1, detrend: bool = True) -> np.ndarray:
    """Remove per-trace mean (and optional linear trend) — never amplitude-normalise."""
    x = x - x.mean(axis=axis, keepdims=True)
    if detrend:
        n = x.shape[axis]
        t = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        tt = (t * t).sum()
        xm = np.moveaxis(x, axis, -1)
        slope = (xm * t).sum(-1, keepdims=True) / (tt + 1e-12)
        xm = xm - slope * t
        x = np.moveaxis(xm, -1, axis)
    return x.astype(np.float32)


def bandpass(x: np.ndarray, fmin: float, fmax: float, fs: float, axis: int = -1):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [fmin, fmax], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis).astype(np.float32)


def sta_lta_trigger(stack: np.ndarray, fs: float, sta_s=0.05, lta_s=0.5,
                    thr_on=3.0, min_gap_s=0.5):
    """Very small classic STA/LTA on an energy stack -> trigger sample indices."""
    e = stack.astype(np.float64) ** 2
    nsta, nlta = max(1, int(sta_s * fs)), max(2, int(lta_s * fs))
    csum = np.cumsum(np.insert(e, 0, 0.0))
    sta = (csum[nsta:] - csum[:-nsta]) / nsta
    lta = (csum[nlta:] - csum[:-nlta]) / nlta
    m = min(len(sta), len(lta))
    ratio = sta[:m] / (lta[:m] + 1e-12)
    on = ratio > thr_on
    trigs, last, gap = [], -10**9, int(min_gap_s * fs)
    for i in np.flatnonzero(on):
        if i - last >= gap:
            trigs.append(int(i)); last = i
    return np.asarray(trigs, dtype=int)


# ==========================================================================
# readers  ->  every reader returns:
#   data   : dict station_code -> ndarray (n_component_raw, n_samples) at src_fs
#   ch_map : dict station_code -> list of raw-channel orientation letters
# (component realisation to N,E,Z happens later, centrally)
# ==========================================================================
def read_obspy(input_glob, station_codes, channel_order):
    """Read miniSEED/SAC/SEG-Y/etc via ObsPy and group traces by station."""
    try:
        from obspy import read as obspy_read
    except Exception as exc:  # pragma: no cover - depends on user env
        raise SystemExit("ObsPy is required for --reader obspy: `uv pip install obspy`") from exc
    st = None
    for fp in sorted(glob.glob(input_glob)):
        st = obspy_read(fp) if st is None else st + obspy_read(fp)
    if st is None or len(st) == 0:
        raise SystemExit(f"No traces read from {input_glob!r}")
    fs_set = {round(float(tr.stats.sampling_rate), 3) for tr in st}
    if len(fs_set) != 1:
        raise SystemExit(f"Mixed sampling rates {fs_set}; please homogenise first.")
    src_fs = fs_set.pop()
    data, ch_map = {}, {}
    for sta in station_codes:
        sub = st.select(station=sta)
        if len(sub) == 0:
            raise SystemExit(f"Station {sta!r} not found in input.")
        # last letter of channel code is the orientation (Z/N/E/1/2/3)
        traces, letters = [], []
        for tr in sub:
            traces.append(tr.data.astype(np.float32))
            letters.append(tr.stats.channel[-1].upper())
        n = min(len(t) for t in traces)
        data[sta] = np.stack([t[:n] for t in traces], axis=0)
        ch_map[sta] = letters
    return data, ch_map, src_fs


def read_arrays(input_path, reader, array_axes, station_codes, src_fs):
    """Read a generic .npy/.npz/.h5 holding one big array; declare its axis order."""
    p = Path(input_path)
    if reader == "npy":
        arr = np.load(p)
    elif reader == "npz":
        z = np.load(p); arr = z[z.files[0]]
    elif reader == "h5":
        import h5py
        with h5py.File(p, "r") as h:
            key = next(iter(h.keys()))
            arr = h[key][...]
    else:
        raise SystemExit(f"Unknown array reader {reader!r}")
    axes = [a.strip() for a in array_axes.split(",")]
    # move to canonical (event, station, comp, time)
    if "event" not in axes:
        arr = arr[None, ...]; axes = ["event"] + axes
    order = [axes.index(a) for a in ("event", "station", "comp", "time")]
    arr = np.transpose(arr, order).astype(np.float32)
    nev, nst, ncomp, _ = arr.shape
    if nst != len(station_codes):
        raise SystemExit(f"Array has {nst} stations but --stations lists {len(station_codes)}.")
    return arr, src_fs  # returned as a pre-grouped event array (special path)


# ==========================================================================
# component realisation -> always (N, E, Z)
# ==========================================================================
_ORIENT = {"N": "N", "E": "E", "Z": "Z", "1": "N", "2": "E", "3": "Z"}

def realise_components(traces: np.ndarray, letters, channel_order):
    """Return (3, T) ordered [N,E,Z]. `letters` are raw orientations; if absent,
    fall back to the user-declared `channel_order` (e.g. Z,N,E)."""
    T = traces.shape[1]
    out = np.zeros((3, T), dtype=np.float32)
    have = letters if letters else channel_order
    norm = [_ORIENT.get(str(l).upper(), str(l).upper()) for l in have]
    for i, comp in enumerate(COMPONENT_ORDER):  # N, E, Z
        if comp in norm:
            out[i] = traces[norm.index(comp)]
        else:
            raise SystemExit(f"Component {comp} missing (have {norm}); L3 needs 3-C N,E,Z.")
    return out


# ==========================================================================
# build event windows
# ==========================================================================
def cut_window(sig_3c: np.ndarray, start: int) -> np.ndarray:
    """Extract WINDOW samples from a (3, T) trace at TARGET_FS, zero-padding edges."""
    T = sig_3c.shape[1]
    out = np.zeros((3, WINDOW), dtype=np.float32)
    a, b = max(0, start), min(T, start + WINDOW)
    if b > a:
        out[:, (a - start):(b - start)] = sig_3c[:, a:b]
    return out


# ==========================================================================
# writers
# ==========================================================================
def write_clean_h5(out_h5, out_csv, waves, event_ids, station_codes, depths,
                   win_start, p_arr=None, s_arr=None, site=""):
    import h5py, pandas as pd
    waves = np.asarray(waves, dtype=np.float32)            # (Nev, 3, 12, T)
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("waveforms", data=waves, compression="gzip", compression_opts=4)
        h.create_dataset("event_id", data=np.array(event_ids, dtype="S64"))
        h.create_dataset("station_codes", data=np.array(station_codes, dtype="S32"))
        h.create_dataset("station_depth", data=np.asarray(depths, dtype=np.float32))
        h.create_dataset("window_start_sample", data=np.asarray(win_start, dtype=np.int64))
        if p_arr is not None:
            h.create_dataset("p_arrival_sample", data=np.asarray(p_arr, dtype=np.float32))
        if s_arr is not None:
            h.create_dataset("s_arrival_sample", data=np.asarray(s_arr, dtype=np.float32))
        h.attrs.update(dict(fs=TARGET_FS, window=WINDOW, n_station=N_STATION,
                            components="".join(COMPONENT_ORDER), site=site,
                            note="physical (de-meaned) traces; L3 loader normalises per station"))
    rows = []
    for i, ev in enumerate(event_ids):
        rows.append(dict(event_id=ev, split="test", site=site,
                         window_start_sample=int(win_start[i]),
                         trace_sampling_rate_hz=TARGET_FS, n_station=N_STATION,
                         station_codes=";".join(station_codes),
                         has_picks=bool(p_arr is not None)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return waves.shape


def write_seisbench(out_dir, waves, event_ids, station_codes, depths,
                    win_start, p_arr, s_arr, site):
    """Optional: SeisBench grouping=event layout (column names AMBER reads)."""
    try:
        import seisbench.data as sbd
    except Exception as exc:
        raise SystemExit("--seisbench-out needs seisbench: `uv pip install seisbench`") from exc
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with sbd.WaveformDataWriter(out / "metadata.csv", out / "waveforms.hdf5") as wr:
        wr.data_format = {"dimension_order": "CW", "component_order": "".join(COMPONENT_ORDER),
                          "sampling_rate": TARGET_FS, "measurement": "velocity", "unit": "counts"}
        for i, ev in enumerate(event_ids):
            for s, sta in enumerate(station_codes):
                meta = dict(source_id=str(ev), split="test", station_code=sta,
                            station_depth_m=float(depths[s]),
                            trace_name=f"{ev}.{sta}", trace_sampling_rate_hz=TARGET_FS,
                            trace_channel="DH")
                if p_arr is not None and not np.isnan(p_arr[i, s]):
                    meta["trace_p_arrival_sample"] = float(p_arr[i, s])
                if s_arr is not None and not np.isnan(s_arr[i, s]):
                    meta["trace_s_arrival_sample"] = float(s_arr[i, s])
                wr.add_trace(meta, waves[i, :, s, :])      # (3, T)
    return out


# ==========================================================================
# QC figure  (white background, English labels, legend in the margin)
# ==========================================================================
def qc_figure(waves, event_ids, station_codes, depths, win_start,
              p_arr, s_arr, site, idx=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = waves[idx]                                   # (3, 12, T)
    t = np.arange(WINDOW) / TARGET_FS * 1e3           # ms
    comp_colors = {"N": "#1f77b4", "E": "#2ca02c", "Z": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.2, 6.0), facecolor="white")
    ax.set_facecolor("white")
    span = np.nanmax(np.abs(ev)) + 1e-12
    for s in range(N_STATION):
        base = (N_STATION - 1 - s) * 2.2              # deepest at bottom
        for c, comp in enumerate(COMPONENT_ORDER):
            ax.plot(t, ev[c, s] / span + base, lw=0.7, color=comp_colors[comp],
                    label=comp if s == 0 else None)
        ax.text(-0.02 * t[-1], base, station_codes[s], ha="right", va="center", fontsize=7)
        if p_arr is not None and not np.isnan(p_arr[idx, s]):
            ax.plot(p_arr[idx, s] / TARGET_FS * 1e3, base, "v", ms=4, color="black")
        if s_arr is not None and not np.isnan(s_arr[idx, s]):
            ax.plot(s_arr[idx, s] / TARGET_FS * 1e3, base, "s", ms=4, color="gray")
    ax.set_xlabel("Time within window (ms)")
    ax.set_ylabel("Station (top \u2192 bottom = shallow \u2192 deep), offset traces")
    ax.set_yticks([])
    ax.set_title(f"MOIRAI L3 ingest QC \u2014 site '{site}', event {event_ids[idx]}\n"
                 f"{N_STATION} stations \u00d7 3-C (N,E,Z), {TARGET_FS} Hz, "
                 f"{WINDOW} samples ({WINDOW/TARGET_FS*1e3:.0f} ms)", fontsize=10)
    # legend OUTSIDE the axes (in the right margin) so it never overlaps the data
    handles = [plt.Line2D([0], [0], color=comp_colors[c], lw=1.4) for c in COMPONENT_ORDER]
    labels = list(COMPONENT_ORDER)
    if p_arr is not None:
        handles += [plt.Line2D([0], [0], marker="v", color="black", ls="none", ms=5)]
        labels += ["P pick"]
    if s_arr is not None:
        handles += [plt.Line2D([0], [0], marker="s", color="gray", ls="none", ms=5)]
        labels += ["S pick"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9, title="Component")
    fig.subplots_adjust(left=0.12, right=0.84, top=0.90, bottom=0.09)
    out = PDF_DIR / f"external_{site}_qc.pdf"
    fig.savefig(out, format="pdf", facecolor="white")
    plt.close(fig)
    return out


# ==========================================================================
# trigger / window planning
# ==========================================================================
def plan_events_from_triggers(csv_path, station_codes, src_fs_for_time, pre_frac):
    import pandas as pd
    df = pd.read_csv(csv_path)
    pre = int(pre_frac * WINDOW)
    plans = []
    for _, r in df.iterrows():
        if "trigger_sample" in df.columns and not pd.isna(r["trigger_sample"]):
            trig_resamp = int(round(int(r["trigger_sample"]) * TARGET_FS / src_fs_for_time))
        elif "trigger_time" in df.columns:
            # seconds-from-start float OR ISO time relative to record start
            try:
                trig_resamp = int(round(float(r["trigger_time"]) * TARGET_FS))
            except ValueError:
                raise SystemExit("trigger_time must be seconds-from-record-start (float).")
        else:
            raise SystemExit("triggers CSV needs 'trigger_sample' or 'trigger_time'.")
        start = max(0, trig_resamp - pre)
        ev = str(r.get("event_id", f"ev{len(plans):05d}"))
        p_row = np.full(len(station_codes), np.nan, dtype=np.float32)
        s_row = np.full(len(station_codes), np.nan, dtype=np.float32)
        for si, sta in enumerate(station_codes):
            for col, arr in ((f"p_sample_{sta}", p_row), (f"s_sample_{sta}", s_row)):
                if col in df.columns and not pd.isna(r[col]):
                    arr[si] = int(round(int(r[col]) * TARGET_FS / src_fs_for_time)) - start
        plans.append((ev, start, p_row, s_row))
    return plans


# ==========================================================================
# minimal Dataset for the inference harness (per-station normalisation here)
# ==========================================================================
class ExternalArrayDataset:
    """Torch-style dataset over the clean H5; yields (3, 12, T) float32, normalised
    per station (matching the L3 training-time normalisation)."""
    def __init__(self, h5_path):
        import h5py
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as h:
            self.n = h["waveforms"].shape[0]
            self.event_id = [e.decode() for e in h["event_id"][...]]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import h5py, torch
        with h5py.File(self.h5_path, "r") as h:
            w = h["waveforms"][i].astype(np.float32)      # (3, 12, T)
        # per-station normalisation: divide each station by its max |amp| over C,T
        scale = np.abs(w).max(axis=(0, 2), keepdims=True)
        scale[scale == 0] = 1.0
        w = w / scale
        return torch.from_numpy(w), self.event_id[i]


# ==========================================================================
# orchestration
# ==========================================================================
def run(args):
    station_codes = [s.strip() for s in args.stations.split(",")] if args.stations else \
        [f"ST{i+1:02d}" for i in range(N_STATION)]
    if len(station_codes) != N_STATION:
        raise SystemExit(f"L3 expects {N_STATION} stations; got {len(station_codes)}.")
    depths = ([float(x) for x in args.depths.split(",")] if args.depths
              else list(range(N_STATION)))
    channel_order = [c.strip() for c in args.channel_order.split(",")]

    # ---- gather per-station continuous 3-C signals at TARGET_FS ----
    if args.reader in ("npy", "npz", "h5"):
        arr, src_fs = read_arrays(args.input, args.reader, args.array_axes,
                                  station_codes, args.src_fs)
        arr = resample_to_target(arr, src_fs, axis=-1)              # (Nev, st, comp, T)
        arr = demean_detrend(arr, axis=-1, detrend=not args.no_detrend)
        if args.bandpass:
            fmin, fmax = (float(x) for x in args.bandpass.split(","))
            arr = bandpass(arr, fmin, fmax, TARGET_FS, axis=-1)
        # realise components per event/station and centre window
        nev = arr.shape[0]
        waves = np.zeros((nev, 3, N_STATION, WINDOW), dtype=np.float32)
        for e in range(nev):
            for s in range(N_STATION):
                sig = realise_components(arr[e, s], None, channel_order)
                start = max(0, sig.shape[1] // 2 - int(args.pre_frac * WINDOW))
                waves[e, :, s, :] = cut_window(sig, start)
        event_ids = [f"ev{e:05d}" for e in range(nev)]
        win_start = [0] * nev
        p_arr = s_arr = None
    else:  # obspy continuous -> window by triggers/sliding/sta-lta
        data, ch_map, src_fs = read_obspy(args.input, station_codes, channel_order)
        sig = {}
        for sta in station_codes:
            x = resample_to_target(data[sta], src_fs, axis=-1)
            x = demean_detrend(x, axis=-1, detrend=not args.no_detrend)
            if args.bandpass:
                fmin, fmax = (float(x_) for x_ in args.bandpass.split(","))
                x = bandpass(x, fmin, fmax, TARGET_FS, axis=-1)
            sig[sta] = realise_components(x, ch_map[sta], channel_order)   # (3, T)
        T = min(v.shape[1] for v in sig.values())
        for sta in sig:
            sig[sta] = sig[sta][:, :T]

        if args.triggers:
            plans = plan_events_from_triggers(args.triggers, station_codes, src_fs, args.pre_frac)
        elif args.sliding:
            stride = int(args.stride) if args.stride else WINDOW
            starts = list(range(0, max(1, T - WINDOW + 1), stride))
            plans = [(f"win{j:05d}", st, None, None) for j, st in enumerate(starts)]
        elif args.sta_lta:
            stack = np.sqrt(sum((sig[s] ** 2).sum(0) for s in station_codes))
            pre = int(args.pre_frac * WINDOW)
            trigs = sta_lta_trigger(stack, TARGET_FS)
            plans = [(f"ev{j:05d}", max(0, int(tg) - pre), None, None)
                     for j, tg in enumerate(trigs)]
        else:
            raise SystemExit("Choose --triggers / --sliding / --sta-lta for ObsPy input.")

        nev = len(plans)
        waves = np.zeros((nev, 3, N_STATION, WINDOW), dtype=np.float32)
        event_ids, win_start = [], []
        has_picks = any(p is not None for _, _, p, _ in plans)
        p_arr = np.full((nev, N_STATION), np.nan, np.float32) if has_picks else None
        s_arr = np.full((nev, N_STATION), np.nan, np.float32) if has_picks else None
        for e, (ev, start, prow, srow) in enumerate(plans):
            for s, sta in enumerate(station_codes):
                waves[e, :, s, :] = cut_window(sig[sta], start)
            event_ids.append(ev); win_start.append(start)
            if has_picks and prow is not None:
                p_arr[e] = prow; s_arr[e] = srow

    # ---- write clean L3 inference format ----
    site = args.site
    out_h5 = DATA_DIR / f"external_{site}_waveforms.h5"
    out_csv = DATA_DIR / f"external_{site}_metadata.csv"
    shape = write_clean_h5(out_h5, out_csv, waves, event_ids, station_codes, depths,
                           win_start, p_arr, s_arr, site)
    print(f"[ok] wrote {out_h5}  waveforms{shape}")
    print(f"[ok] wrote {out_csv}  ({len(event_ids)} events, split=test)")

    if args.qc and len(event_ids):
        qc = qc_figure(waves, event_ids, station_codes, depths, win_start, p_arr, s_arr, site)
        print(f"[ok] wrote QC figure {qc}")

    if args.seisbench_out:
        d = write_seisbench(args.seisbench_out, waves, event_ids, station_codes, depths,
                            win_start, p_arr, s_arr, site)
        print(f"[ok] wrote SeisBench/AMBER-format dataset to {d}")

    # sanity for the model interface
    ds = ExternalArrayDataset(out_h5)
    x, ev0 = ds[0]
    assert tuple(x.shape) == (3, N_STATION, WINDOW), x.shape
    print(f"[ok] dataset check: item shape {tuple(x.shape)} (event {ev0}); "
          f"ready for the L3 picker (zero-shot).")


# ==========================================================================
# self-test (synthetic; no obspy/seisbench needed)
# ==========================================================================
def selftest():
    print("[selftest] synthesising a 12-station 3-C shale-gas-like array ...")
    src_fs, nev, T = 4000, 5, 8000
    rng = np.random.default_rng(0)
    arr = (0.02 * rng.standard_normal((nev, N_STATION, 3, T))).astype(np.float32)
    # inject P (steep) and S (later) moveout across depth, on Z and N respectively
    for e in range(nev):
        p0, s0 = 2500 + 30 * e, 3200 + 30 * e
        for s in range(N_STATION):
            tp, ts = p0 + 45 * s, s0 + 78 * s          # moveout in samples @4000 Hz
            if tp < T:
                arr[e, s, 2, tp:tp + 40] += np.hanning(40).astype(np.float32)   # Z = P
            if ts < T:
                arr[e, s, 0, ts:ts + 60] += 1.3 * np.hanning(60).astype(np.float32)  # N = S
    tmp = DATA_DIR / "_selftest_events.npy"
    np.save(tmp, arr)

    from types import SimpleNamespace
    A = SimpleNamespace(
        input=str(tmp), reader="npy", array_axes="event,station,comp,time",
        src_fs=src_fs, site="selftest", stations=None, depths=None,
        channel_order="N,E,Z", triggers=None, sliding=False, stride=None,
        sta_lta=False, pre_frac=0.25, no_detrend=False, bandpass=None,
        qc=True, seisbench_out=None)
    run(A)

    # verify outputs
    import h5py
    with h5py.File(DATA_DIR / "external_selftest_waveforms.h5", "r") as h:
        w = h["waveforms"][...]
        assert w.shape == (nev, 3, N_STATION, WINDOW), w.shape
        assert h.attrs["fs"] == TARGET_FS and h.attrs["window"] == WINDOW
        assert np.isfinite(w).all()
    assert (PDF_DIR / "external_selftest_qc.pdf").exists()
    tmp.unlink(missing_ok=True)
    print(f"[selftest] PASS \u2014 shapes ({nev},3,{N_STATION},{WINDOW}) @ {TARGET_FS} Hz, "
          f"QC PDF written, dataset loads for the model.")


# ==========================================================================
def build_argparser():
    ap = argparse.ArgumentParser(description="Convert external borehole-array waveforms "
                                             "into MOIRAI L3 picking input (and optional AMBER format).")
    ap.add_argument("--selftest", action="store_true", help="run synthetic end-to-end check and exit")
    ap.add_argument("--input", help="file / glob (obspy) or array file (npy/npz/h5)")
    ap.add_argument("--reader", choices=["obspy", "npy", "npz", "h5"], default="obspy")
    ap.add_argument("--site", default="external", help="site tag used in output filenames")
    ap.add_argument("--stations", help="comma list of station codes, DEPTH ORDER (shallow->deep)")
    ap.add_argument("--depths", help="comma list of station depths (m), same order as --stations")
    ap.add_argument("--channel-order", default="Z,N,E",
                    help="raw channel orientation order if codes are ambiguous (array readers)")
    ap.add_argument("--src-fs", type=float, default=TARGET_FS, help="source sampling rate (array readers)")
    ap.add_argument("--array-axes", default="event,station,comp,time",
                    help="axis order of an array input (comma list of event,station,comp,time)")
    ap.add_argument("--triggers", help="CSV: event_id, trigger_time|trigger_sample [, p_sample_<STA>...]")
    ap.add_argument("--sliding", action="store_true", help="tile continuous data into windows")
    ap.add_argument("--stride", help="stride (samples@2000Hz) for --sliding (default = window)")
    ap.add_argument("--sta-lta", action="store_true", help="auto-trigger windows with STA/LTA")
    ap.add_argument("--pre-frac", type=float, default=0.25, help="trigger position within window (0-1)")
    ap.add_argument("--no-detrend", action="store_true", help="demean only, skip linear detrend")
    ap.add_argument("--bandpass", help="optional 'fmin,fmax' Hz band-pass before windowing")
    ap.add_argument("--qc", action="store_true", help="write a QC PDF of one converted event")
    ap.add_argument("--seisbench-out", help="also export SeisBench/AMBER-format dataset to this dir")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.selftest:
        selftest()
    elif not args.input:
        build_argparser().print_help()
        raise SystemExit("\nProvide --input, or run --selftest.")
    else:
        run(args)
