#!/usr/bin/env python
"""
14_fig4_and_tableS1.py -- GJI major revision: rebuild the diagnosis figure
(submitted Figure 4) and compute Supplementary Table S1 (array geometry).
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
(A) FIGURE 4 REBUILD (Reviewer 1, comment 36)
    The reviewer asked for (i) real wiggle waveform panels instead of the
    image-style Z panel, (ii) an explicit legend identifying the ground-truth
    circles, and (iii) an in-distribution contrast event so the failure is
    read against a success. This script runs ONE inference pass per site
    (array configuration, LOSO checkpoints) on forge_19 (out-of-distribution)
    and mseel_5h (in-distribution contrast), selects a representative
    earthquake event per site (>= MIN_PICKS picked P stations, across-station
    P moveout closest to the site median), and draws a 3x3 figure:

        row 1  forge_19 : Z-component wiggles | predicted P map | predicted S map
        row 2  mseel_5h : Z-component wiggles | predicted P map | predicted S map
        row 3  pooled max-score histograms at true arrivals (P | S)
               with the 0.30 detection threshold marked

    Ground-truth picks are overlaid on every panel with a legended marker.
    All text in English, white background, legends kept out of the data.

(B) SUPPLEMENTARY TABLE S1 (Reviewer 1, comments 4-5)
    Per-site array geometry (station count, vertical extent, 3D end-to-end
    aperture, median adjacent spacing) computed from the station coordinates
    that AMBER distributes as Attr/<Dataset>.Array.csv inside the raw-SEGY
    archive on Zenodo. The 20.7-GB zip is NOT downloaded: zip archives allow
    random access, so with `remotezip` only the few-kB Array.csv members are
    fetched over HTTP range requests. A local Attr/ directory can be used
    instead with --attr-dir.

OUTPUTS
  PDF/14_a_fig4_diagnosis.pdf     rebuilt Figure 4 (also .png at 300 dpi
                                  for direct insertion into the docx)
  logs/14_fig4_event_choice.json  which events were shown, and why
  logs/14_table_s1.csv            Supplementary Table S1 values
  logs/14_table_s1.md             the same, as a Markdown table for pasting

RUN (Colab A100, Drive mounted, AMBER_H5 set, model/ holding LOSO ckpts)
  python src/14_fig4_and_tableS1.py --fig4
  pip -q install remotezip && python src/14_fig4_and_tableS1.py --table-s1
  # or, with a local copy of the Attr/ directory:
  python src/14_fig4_and_tableS1.py --table-s1 --attr-dir /path/to/Attr
  # self-test (synthetic, no AMBER / checkpoints / network needed):
  python src/14_fig4_and_tableS1.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
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

SITES_FIG4 = ("forge_19", "mseel_5h")      # OOD failure + in-distribution contrast
MIN_PICKS = 9                                # representative event: >= this many P stations
THRESHOLD = 0.30                             # paper's fixed detection threshold
ZEN_URL = ("https://zenodo.org/records/18944111/files/"
           "amber_raw_segys.zip")

# aliases for matching AMBER dataset names in the zip to our site keys
SITE_ALIASES = {
    "pnr-1":          ("pnr1", "prestonnewroad1", "pnr-1"),
    "pnr-2":          ("pnr2", "prestonnewroad2", "pnr-2"),
    "mseel_3h":       ("mseel3h", "mseelstage3h", "mseel_3h"),
    "mseel_5h":       ("mseel5h", "mseelstage5h", "mseel_5h"),
    "clearfield_mw4": ("clearfieldmw4", "clearfield_mw4"),
    "clearfield_mw6": ("clearfieldmw6", "clearfield_mw6"),
    "aneth":          ("aneth",),
    "forge_19":       ("forge2019", "forgegeothermal2019", "forge19",
                       "forge_19", "utahforge2019"),
}


# ----------------------------------------------------------------------------
# (A) Figure 4 rebuild
# ----------------------------------------------------------------------------
def _moveout_ms(gt: np.ndarray, fs: float) -> np.ndarray:
    """Per-event across-station P arrival spread in ms from gt (E,S) samples."""
    out = np.full(gt.shape[0], np.nan)
    for e in range(gt.shape[0]):
        v = gt[e][gt[e] >= 0]
        if len(v) >= 2:
            out[e] = (v.max() - v.min()) / fs * 1000.0
    return out


def _pick_event(labels: np.ndarray, fs: float) -> tuple[int, dict]:
    """Representative earthquake event: >= MIN_PICKS P stations, all catalogued
    arrivals comfortably inside the window (so panels are readable), and P
    moveout closest to the site median (over events with >= 2 P picks)."""
    T = labels.shape[-1]
    gt_p = np.where(labels[:, 0].max(-1) >= 0.5, labels[:, 0].argmax(-1), -1)
    gt_s = np.where(labels[:, 1].max(-1) >= 0.5, labels[:, 1].argmax(-1), -1)
    npick = (gt_p >= 0).sum(1)
    mo = _moveout_ms(gt_p, fs)
    med = float(np.nanmedian(mo))
    lo, hi = int(0.08 * T), int(0.85 * T)     # arrivals not hugging the edges
    tmin = np.where((gt_p >= 0) | (gt_s >= 0),
                    np.where(gt_p >= 0, gt_p, T) , T).min(1)
    allts = np.where(gt_s >= 0, gt_s, np.where(gt_p >= 0, gt_p, 0))
    tmax = allts.max(1)
    central = (tmin >= lo) & (tmax <= hi)
    # Fallback order matters. Reviewer 1 (round 2) observed that the mseel_5h
    # panel of Figure 4 showed an event whose arrivals sat at the very end of
    # the window, which is unreadable. The earlier order dropped the CENTRED
    # test first and kept the pick-count test, which is backwards: the figure
    # exists to be read, so stations are given up before centring is. The
    # stage actually used is recorded, so the choice is auditable.
    stages = [(">=MIN_PICKS and centred", (npick >= MIN_PICKS) & central),
              (">=6 and centred", (npick >= 6) & central),
              (">=2 and centred", (npick >= 2) & central),
              (">=MIN_PICKS, not centred", npick >= MIN_PICKS),
              (">=2, not centred", npick >= 2)]
    cand, stage = np.array([], dtype=int), "none"
    for name, mask in stages:
        hit = np.where(mask)[0]
        if len(hit):
            cand, stage = hit, name
            break
    if not len(cand):
        raise SystemExit("[fig4] no event carries >= 2 catalogued P picks")
    order = cand[np.argsort(np.abs(mo[cand] - med))]
    info = dict(site_median_p_moveout_ms=med,
                arrivals_within_window_frac=[lo / T, hi / T],
                selection_stage=stage, n_candidates=int(len(cand)))
    return order, npick, mo, info


def _windowed_max(prob: np.ndarray, gt: np.ndarray,
                  half: int = 40) -> np.ndarray:
    """Max score within +/-half samples of the true arrival, for every
    (event, station) that carries a catalogue pick. `half`=40 samples matches
    the paper's +/-20 ms scoring tolerance, so this is the quantity the pick
    rule actually thresholds when detection at the arrival is at stake."""
    E, S, T = prob.shape
    out = []
    for e, s in zip(*np.where(gt >= 0)):
        g = int(gt[e, s])
        out.append(prob[e, s, max(0, g - half):min(T, g + half + 1)].max())
    return np.asarray(out, np.float32)


def _late_energy_ratio(waves_z: np.ndarray, gt: np.ndarray,
                       fs: float) -> float:
    """max |Z| after (last S + 150 samples) divided by max |Z| within the
    pick span (+/-100 samples). >1 means a later, larger burst (a second,
    unlabelled event) dominates the window and would visually swamp the
    primary event under per-station normalisation."""
    T = waves_z.shape[-1]
    ts = gt[gt >= 0]
    if len(ts) == 0:
        return np.inf
    lo = max(0, int(ts.min()) - 100)
    hi = min(T, int(ts.max()) + 100)
    amp_pick = np.abs(waves_z[:, lo:hi]).max()
    late0 = min(T, int(ts.max()) + 150)
    if late0 >= T - 10 or amp_pick == 0:
        return 0.0
    return float(np.abs(waves_z[:, late0:]).max() / amp_pick)


def _infer_site(site: str):
    """One array-configuration inference pass on the held-out site."""
    import torch
    model_mod = _load("02_picker_model_l3.py")
    bc = _load("07_bootstrap_ci.py")
    loader_mod = _load("01_amber_setup.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_name = TRAIN.CKPT_EMA.replace(".pt", f"_loso_{site}.pt")
    ckpt = cfg.MODEL_DIR / ckpt_name
    print(f"[fig4] site={site}  ckpt={ckpt}  device={device}")
    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])

    csv = loader_mod.prepare_site_csv(site, all_test=True)
    ds = loader_mod.build_amber_dataset("test", csv)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=TRAIN.NUM_WORKERS)
    preds, labels = bc.predict_grouped(model, loader, device, False, False)
    ctx = dict(model=model, device=device, model_mod=model_mod)
    return ds, np.asarray(preds), np.asarray(labels), ctx


def _gt_times(labels: np.ndarray) -> np.ndarray:
    """(E,2,S) ground-truth samples (or -1) from soft labels (E,3,S,T)."""
    E, _, S, _ = labels.shape
    gt = np.full((E, 2, S), -1, np.int64)
    for ch in (0, 1):
        lab = labels[:, ch]
        has = lab.max(-1) >= 0.5
        gt[:, ch][has] = lab.argmax(-1)[has]
    return gt


def _wiggle_panel(ax, waves_z: np.ndarray, gt: np.ndarray, fs: float):
    """Depth-ordered normalized Z-component wiggles + ground-truth markers."""
    S, T = waves_z.shape
    t_ms = np.arange(T) / fs * 1000.0
    for s in range(S):
        z = waves_z[s].astype(float)
        m = np.abs(z).max()
        if m > 0:
            z = z / m
        ax.plot(t_ms, s + 0.42 * z, color="k", lw=0.45)
    hP = hS = None
    for s in range(S):
        if gt[0, s] >= 0:
            hP, = ax.plot(gt[0, s] / fs * 1000.0, s, "o", ms=5.5, mfc="none",
                          mec="#1f77b4", mew=1.6)
        if gt[1, s] >= 0:
            hS, = ax.plot(gt[1, s] / fs * 1000.0, s, "^", ms=5.5, mfc="none",
                          mec="#d62728", mew=1.6)
    ax.set_ylim(S - 0.5, -0.5)
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylabel("Station (depth order)")
    ax.set_xlabel("Time (ms)")
    return hP, hS


def _prob_panel(ax, prob: np.ndarray, gt_ch: np.ndarray, fs: float, cmap: str):
    S, T = prob.shape
    im = ax.imshow(prob, aspect="auto", origin="upper", cmap=cmap,
                   vmin=0.0, vmax=1.0,
                   extent=[0, T / fs * 1000.0, S - 0.5, -0.5])
    hG = None
    for s in range(S):
        if gt_ch[s] >= 0:
            hG, = ax.plot(gt_ch[s] / fs * 1000.0, s, "o", ms=6, mfc="none",
                          mec="k", mew=1.5)
    ax.set_xlabel("Time (ms)")
    return im, hG


def _hist_panel(ax, vals: dict, title: str):
    bins = np.linspace(0, 1, 31)
    colors = {"forge_19": "#1f77b4", "mseel_5h": "0.55"}
    for site, v in vals.items():
        ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                alpha=0.55 if site == "forge_19" else 0.45,
                color=colors[site], label=f"{site} (held out)")
    ax.axvline(THRESHOLD, color="k", ls="--", lw=1.2,
               label=f"pick threshold = {THRESHOLD:.2f}")
    ax.set_xlabel(r"max score within $\pm$20 ms of the true arrival")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)



def consistent_draw(ctx, ds, j: int, fs: float, tries: int = 4):
    """Waveform, labels and prediction from ONE window draw of event j.

    WHY THIS EXISTS. AMBER re-crops the analysis window on every access: its
    __getitem__ builds `sample_rng` from (torch.initial_seed() in the main
    process, or the worker seed inside a DataLoader) plus the index, and passes
    it to extract_window. A waveform fetched with `ds[j][0]` AFTER a DataLoader
    pass therefore belongs to a different window than the labels and
    predictions that pass returned, and overlaying them puts the catalogued
    pick markers away from the arrivals they mark. Reviewer 1 reported exactly
    that for Figure 4 (R1-5).

    So the panel must be built from a single access, with the prediction
    computed on that access. Several draws are tried and the first one whose
    arrivals sit inside the window and whose later energy does not swamp the
    event is kept; otherwise the best seen is used. What was kept is reported.
    """
    import torch
    best, tried = None, 0
    for _ in range(tries):
        tried += 1
        x, y = ds[int(j)]
        xn = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        yn = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        gt = _gt_times(yn[None])[0]
        T = xn.shape[-1]
        ts = gt[gt >= 0]
        if len(ts) == 0:
            continue
        centred = bool(ts.min() >= 0.08 * T and ts.max() <= 0.85 * T)
        r = _late_energy_ratio(xn[2], gt.ravel(), fs)
        key = (0 if centred else 1, float(r))
        if best is None or key < best[0]:
            best = (key, xn, gt, centred, float(r))
        if centred and r < 0.8:
            break
    if best is None:
        raise SystemExit(f"[fig4] event {j}: no draw carried a catalogued arrival")
    _, xn, gt, centred, r = best
    with torch.no_grad():
        t = torch.as_tensor(xn)[None].to(ctx["device"])
        prob = ctx["model_mod"].MoiraiPickerL3.activate(ctx["model"](t))
    prob = prob[0].detach().cpu().numpy()
    info = dict(draws_tried=tried, draw_centred=centred,
                late_energy_ratio=float(r))
    return xn[2], prob, gt, info


def pick_panel(ctx, ds, order, fs, n_events: int = 40, draws: int = 4):
    """Choose the event AND the window draw that the panel will show.

    Two things must hold at once. The panel must be self-consistent — waveform,
    labels and prediction from ONE access (consistent_draw) — and it must be
    readable: the labelled arrivals inside the window, and no later arrival
    swamping them under per-station normalisation.

    Those are different failure modes and need different remedies. A badly
    placed window is cured by drawing again; a neighbouring event that is
    simply larger is not, and needs a different event. So this searches over
    candidate events, nearest the site median first, and over a few draws of
    each, and stops at the first that satisfies both. If none does, the best
    seen is used and the reason is recorded rather than hidden.
    """
    best = None
    for n, j in enumerate(order[:n_events], 1):
        wz, prob, gt, dinfo = consistent_draw(ctx, ds, int(j), fs, tries=draws)
        key = (0 if dinfo["draw_centred"] else 1, dinfo["late_energy_ratio"])
        if best is None or key < best[0]:
            best = (key, int(j), wz, prob, gt, dinfo, n)
        if dinfo["draw_centred"] and dinfo["late_energy_ratio"] < 0.8:
            dinfo["events_tried"] = n
            dinfo["panel_ok"] = True
            return int(j), wz, prob, gt, dinfo
    _, j, wz, prob, gt, dinfo, n = best
    dinfo["events_tried"] = min(len(order), n_events)
    dinfo["panel_ok"] = False
    return j, wz, prob, gt, dinfo

def make_fig4(payload: dict, out_pdf: Path):
    """payload[site] = dict(waves_z=(S,T), prob=(3,S,T), gt=(2,S),
                            hist_P=array, hist_S=array, fs=float)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.4))
    handles = {}
    for i, site in enumerate(SITES_FIG4):
        d = payload[site]
        hP, hS = _wiggle_panel(axes[i, 0], d["waves_z"], d["gt"], d["fs"])
        handles.setdefault("P pick (catalogue)", hP)
        handles.setdefault("S pick (catalogue)", hS)
        axes[i, 0].set_title(f"{site} — Z-component waveforms", fontsize=10)
        imP, hG = _prob_panel(axes[i, 1], d["prob"][0], d["gt"][0], d["fs"], "Blues")
        handles.setdefault("catalogued pick (ground truth)", hG)
        axes[i, 1].set_title(f"{site} — predicted P score", fontsize=10)
        imS, _ = _prob_panel(axes[i, 2], d["prob"][1], d["gt"][1], d["fs"], "Reds")
        axes[i, 2].set_title(f"{site} — predicted S score", fontsize=10)
        fig.colorbar(imP, ax=axes[i, 1], fraction=0.045, pad=0.02)
        fig.colorbar(imS, ax=axes[i, 2], fraction=0.045, pad=0.02)
    _hist_panel(axes[2, 0], {s: payload[s]["hist_P"] for s in SITES_FIG4},
                "Max P score where a true P exists")
    _hist_panel(axes[2, 1], {s: payload[s]["hist_S"] for s in SITES_FIG4},
                "Max S score where a true S exists")
    # dedicated legend axis (right cell of the bottom row): nothing overlaps
    axL = axes[2, 2]
    axL.axis("off")
    hh, ll = axes[2, 0].get_legend_handles_labels()
    marker_h = [h for h in (handles.get("P pick (catalogue)"),
                            handles.get("S pick (catalogue)"),
                            handles.get("catalogued pick (ground truth)")) if h]
    marker_l = [l for l, h in (("P pick (catalogue)", handles.get("P pick (catalogue)")),
                               ("S pick (catalogue)", handles.get("S pick (catalogue)")),
                               ("catalogued pick on score maps",
                                handles.get("catalogued pick (ground truth)"))) if h]
    axL.legend(marker_h + hh, marker_l + ll, loc="center left", frameon=True,
               framealpha=0.95, edgecolor="0.8", fontsize=9,
               title="Legend", title_fontsize=10)
    fig.suptitle("Diagnosis of the forge_19 P collapse against an "
                 "in-distribution contrast (mseel_5h)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[fig4] wrote {out_pdf} (+ .png)")


def run_fig4():
    fs = float(getattr(DATA, "SAMPLE_RATE", 2000.0))
    payload, choice = {}, {}
    for site in SITES_FIG4:
        ds, preds, labels, ctx = _infer_site(site)
        gt_all = _gt_times(labels)                       # (E,2,S)
        order, npick, mo, info = _pick_event(labels, fs)
        # Search candidate events AND window draws: one access per panel keeps
        # the markers on the arrivals (R1-5), and the event search keeps a
        # larger neighbouring arrival out of the panel.
        idx, waves_z, prob, gt, dinfo = pick_panel(ctx, ds, order, fs)
        info.update(event_index=idx, n_p_picks=int(npick[idx]),
                    p_moveout_ms=float(mo[idx]), **dinfo)
        choice[site] = info
        hist_P = _windowed_max(preds[:, 0], gt_all[:, 0])
        hist_S = _windowed_max(preds[:, 1], gt_all[:, 1])
        payload[site] = dict(waves_z=waves_z, prob=prob, gt=gt,
                             hist_P=hist_P, hist_S=hist_S, fs=fs)
        print(f"[fig4] {site}: event {idx} "
              f"(P picks {info['n_p_picks']}, moveout {info['p_moveout_ms']:.1f} ms; "
              f"site median {info['site_median_p_moveout_ms']:.1f} ms; "
              f"events {dinfo['events_tried']}, draws {dinfo['draws_tried']}, "
              f"centred {dinfo['draw_centred']}, "
              f"late-energy {dinfo['late_energy_ratio']:.2f}, "
              f"ok {dinfo['panel_ok']})")
    make_fig4(payload, cfg.PDF_DIR / "14_a_fig4_diagnosis.pdf")
    out = cfg.LOGS_DIR / "14_fig4_event_choice.json"
    out.write_text(json.dumps(choice, indent=2))
    print(f"[fig4] wrote {out}")


# ----------------------------------------------------------------------------
# (B) Supplementary Table S1 from Attr/<Dataset>.Array.csv
# ----------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_site(dataset_name: str) -> str | None:
    n = _norm_name(dataset_name)
    for site, aliases in SITE_ALIASES.items():
        for a in aliases:
            an = _norm_name(a)
            if an == n or an in n or n in an:
                return site
    return None


def _geometry_from_array_csv(text: str) -> dict:
    """Array.csv columns: Station ID, X, Y, Depth (metres)."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue                                   # header line
        rows.append((x, y, z))
    if len(rows) < 2:
        raise ValueError("Array.csv parsed fewer than 2 stations")
    xyz = np.asarray(rows, float)
    order = np.argsort(xyz[:, 2])                      # sort by depth
    xyz = xyz[order]
    d_adj = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return dict(
        n_stations=len(xyz),
        vertical_extent_m=float(xyz[:, 2].max() - xyz[:, 2].min()),
        aperture_3d_m=float(np.linalg.norm(xyz[-1] - xyz[0])),
        median_spacing_m=float(np.median(d_adj)),
    )


def _iter_array_csvs_remote(url: str):
    from remotezip import RemoteZip
    with RemoteZip(url) as z:
        names = [n for n in z.namelist() if n.endswith(".Array.csv")]
        print(f"[tableS1] found {len(names)} Array.csv members in the archive")
        for n in names:
            yield Path(n).name.replace(".Array.csv", ""), z.read(n).decode(
                "utf-8", errors="replace")


def _iter_array_csvs_local(attr_dir: Path):
    for f in sorted(attr_dir.glob("*.Array.csv")):
        yield f.name.replace(".Array.csv", ""), f.read_text(
            encoding="utf-8", errors="replace")


def run_table_s1(attr_dir: str | None, url: str):
    src = (_iter_array_csvs_local(Path(attr_dir)) if attr_dir
           else _iter_array_csvs_remote(url))
    per_site, unmatched = {}, []
    for dsname, text in src:
        site = _match_site(dsname)
        if site is None:
            unmatched.append(dsname)
            continue
        per_site[site] = dict(dataset=dsname, **_geometry_from_array_csv(text))
        g = per_site[site]
        print(f"[tableS1] {site:<15s} <- {dsname:<24s} "
              f"N={g['n_stations']:>3d}  vert={g['vertical_extent_m']:7.1f} m  "
              f"aper3D={g['aperture_3d_m']:7.1f} m  "
              f"spacing={g['median_spacing_m']:6.1f} m")
    if unmatched:
        print(f"[tableS1] unmatched datasets (excluded sites are expected): "
              f"{unmatched}")
    missing = [s for s in SITE_ALIASES if s not in per_site]
    if missing:
        print(f"[tableS1] WARNING: no Array.csv matched for: {missing}")

    order = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
             "clearfield_mw4", "aneth", "forge_19"]
    csv_lines = ["site,dataset,n_stations,vertical_extent_m,"
                 "aperture_3d_m,median_adjacent_spacing_m"]
    md = ["| Site | Stations (full string) | Vertical extent (m) | "
          "3-D aperture (m) | Median spacing (m) |",
          "|---|---|---|---|---|"]
    for s in order:
        if s not in per_site:
            continue
        g = per_site[s]
        csv_lines.append(f"{s},{g['dataset']},{g['n_stations']},"
                         f"{g['vertical_extent_m']:.1f},{g['aperture_3d_m']:.1f},"
                         f"{g['median_spacing_m']:.1f}")
        md.append(f"| {s} | {g['n_stations']} | {g['vertical_extent_m']:.1f} | "
                  f"{g['aperture_3d_m']:.1f} | {g['median_spacing_m']:.1f} |")
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR / "14_table_s1.csv").write_text("\n".join(csv_lines) + "\n")
    (cfg.LOGS_DIR / "14_table_s1.md").write_text("\n".join(md) + "\n")
    print(f"[tableS1] wrote {cfg.LOGS_DIR/'14_table_s1.csv'} and .md")
    print("\n".join(md))


# ----------------------------------------------------------------------------
# self-test (synthetic; no AMBER / checkpoints / network)
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    fs, S, T = 2000.0, 12, 2048
    payload = {}
    for site, lowP in (("forge_19", True), ("mseel_5h", False)):
        waves_z = rng.normal(0, 1, (S, T)).astype(np.float32)
        prob = np.zeros((3, S, T), np.float32)
        gt = np.full((2, S), -1, np.int64)
        for s in range(S):
            tp, ts = 400 + 12 * s, 700 + 20 * s
            gt[0, s], gt[1, s] = tp, ts
            aP = 0.15 if lowP else 0.95
            prob[0, s, max(0, tp - 20):tp + 20] = aP
            prob[1, s, max(0, ts - 20):ts + 20] = 0.9
            waves_z[s, tp:tp + 60] += 4 * np.hanning(60)
        hist_P = np.clip(rng.normal(0.2 if lowP else 0.95, 0.1, 400), 0, 1)
        hist_S = np.clip(rng.normal(0.8, 0.08, 400), 0, 1)
        payload[site] = dict(waves_z=waves_z, prob=prob, gt=gt,
                             hist_P=hist_P, hist_S=hist_S, fs=fs)
    out = cfg.PDF_DIR / "14_selftest_fig4.pdf"
    make_fig4(payload, out)
    assert out.exists() and out.stat().st_size > 10_000

    # late-energy ratio: a big later burst must be flagged, a clean event not
    z = np.zeros((12, 2048), np.float32)
    gtv = np.arange(300, 300 + 12 * 12, 12)
    for s, g in enumerate(gtv):
        z[s, g:g + 40] = 1.0                     # primary arrivals
    gt2 = np.concatenate([gtv, gtv + 200])       # P and S columns flattened
    assert _late_energy_ratio(z, gt2, fs) < 0.1
    z2 = z.copy(); z2[:, 1400:1600] = 5.0        # later, larger burst
    assert _late_energy_ratio(z2, gt2, fs) > 3.0

    # windowed max: score peak away from the arrival must NOT count
    pr = np.zeros((1, 1, 2048), np.float32)
    pr[0, 0, 1500] = 0.9                    # spurious late ridge
    pr[0, 0, 400] = 0.2                     # weak response at the arrival
    gt = np.array([[400]])
    wm = _windowed_max(pr, gt)
    assert wm.shape == (1,) and abs(wm[0] - 0.2) < 1e-6

    # geometry math on a known string: 12 stations, 30 m apart, vertical
    csv = "\n".join(f"{i+1},0.0,0.0,{1000 + 30*i:.1f}" for i in range(12))
    g = _geometry_from_array_csv("StationID,X,Y,Depth\n" + csv)
    assert g["n_stations"] == 12
    assert abs(g["vertical_extent_m"] - 330.0) < 1e-6
    assert abs(g["aperture_3d_m"] - 330.0) < 1e-6
    assert abs(g["median_spacing_m"] - 30.0) < 1e-6
    # consistent_draw: a dataset that re-crops on EVERY access must not be
    # able to desynchronise the waveform from the labels drawn over it. This is
    # the R1-5 defect: AMBER redraws the window per access, the waveform was
    # fetched separately from the labels, and the markers landed on quiet
    # traces. The fake dataset below reproduces that behaviour exactly.
    import sys as _sys, types as _types, importlib.util as _ilu

    class _RedrawDS:
        """Every access crops the same event at a NEW random offset."""
        def __init__(self, S=12, T=2048, seed=0):
            self.S, self.T, self.rng, self.calls = S, T, np.random.default_rng(seed), 0

        def __getitem__(self, i):
            self.calls += 1
            off = int(self.rng.integers(int(0.10 * self.T), int(0.55 * self.T)))
            x = np.zeros((3, self.S, self.T), np.float32)
            y = np.zeros((3, self.S, self.T), np.float32)
            for st in range(self.S):
                tp, ts = off + 6 * st, off + 300 + 10 * st
                x[2, st, tp:tp + 40] = 1.0            # burst exactly at the P time
                y[0, st, tp] = 1.0                    # label peaks ON the onset
                y[1, st, ts] = 1.0
            return x, y

    if _ilu.find_spec("torch") is None:               # sandbox: stand in for torch
        class _T(np.ndarray):
            def to(self, d): return self
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        class _NG:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        _fake = _types.ModuleType("torch")
        _fake.no_grad = lambda: _NG()
        _fake.as_tensor = lambda a: np.asarray(a).view(_T)
        _sys.modules["torch"] = _fake

    _ctx = dict(model=lambda t: t, device="cpu",
                model_mod=_types.SimpleNamespace(
                    MoiraiPickerL3=_types.SimpleNamespace(activate=lambda z: z)))
    _ds = _RedrawDS()
    wz, prob, gt, dinfo = consistent_draw(_ctx, _ds, 0, 2000.0)
    assert wz.shape == (12, 2048), wz.shape
    assert prob.shape == (3, 12, 2048), prob.shape
    # every catalogued P must sit on a burst of the waveform ACTUALLY returned
    for st in range(12):
        tp = int(gt[0, st])
        assert wz[st, tp] > 0.5, (
            f"station {st}: marker at {tp} is not on the waveform returned "
            f"with it — the draw is not self-consistent")
    assert dinfo["draw_centred"] is True, dinfo
    # and the prediction must belong to the SAME draw as the waveform
    assert np.allclose(prob[2], wz), "prediction is not from the returned draw"

    # pick_panel: when the nearest-median event is PERMANENTLY swamped by a
    # later arrival, no amount of redrawing helps and another event must be
    # taken. Dropping that search is what left clearfield_mw4 with a bigger
    # neighbouring arrival filling its panel.
    class _TwoEventDS:
        """Event 0 always carries a huge later burst; event 1 never does."""
        def __init__(self, S=12, T=2048, seed=1):
            self.S, self.T, self.rng = S, T, np.random.default_rng(seed)
            self.seen = []

        def __getitem__(self, i):
            self.seen.append(int(i))
            off = int(self.rng.integers(int(0.15 * self.T), int(0.40 * self.T)))
            x = np.zeros((3, self.S, self.T), np.float32)
            y = np.zeros((3, self.S, self.T), np.float32)
            for st in range(self.S):
                tp, ts = off + 6 * st, off + 200 + 8 * st
                x[2, st, tp:tp + 40] = 1.0
                y[0, st, tp] = 1.0
                y[1, st, ts] = 1.0
                if int(i) == 0:                       # a much larger later event
                    x[2, st, int(0.80 * self.T):int(0.80 * self.T) + 60] = 9.0
            return x, y

    _ds2 = _TwoEventDS()
    j, wz2, _, gt2, info2 = pick_panel(_ctx, _ds2, np.array([0, 1]), 2000.0)
    assert j == 1, f"swamped event {j} was kept instead of moving on"
    assert info2["panel_ok"] is True, info2
    assert info2["late_energy_ratio"] < 0.8, info2
    assert info2["events_tried"] == 2, info2
    assert 0 in _ds2.seen, "the first candidate was never tried"
    for st in range(12):                              # still self-consistent
        assert wz2[st, int(gt2[0, st])] > 0.5, st

    # and when the first candidate is already good, it is kept
    class _GoodDS(_TwoEventDS):
        def __getitem__(self, i):
            x, y = super().__getitem__(1)             # never swamped
            return x, y
    _ds3 = _GoodDS()
    j3, _, _, _, info3 = pick_panel(_ctx, _ds3, np.array([0, 1]), 2000.0)
    assert j3 == 0 and info3["events_tried"] == 1, (j3, info3)

    # a second, independent access must give a DIFFERENT window — i.e. the
    # fake dataset really does reproduce the behaviour this guards against
    x2, _ = _ds[0]
    assert not np.allclose(x2[2], wz), "the fake dataset did not redraw"

    # _pick_event: a centred event with FEWER picks must beat a full-pick
    # event jammed against the end of the window (reviewer 1, R1-5).
    T = 2048
    labels = np.zeros((2, 3, 12, T), np.float32)
    # event 0: all 12 stations, but arrivals at 0.90-0.96 T  (not centred)
    for st in range(12):
        labels[0, 0, st, int(0.90 * T) + st] = 1.0
        labels[0, 1, st, int(0.95 * T) + st] = 1.0
    # event 1: only 7 stations, arrivals near the middle       (centred)
    for st in range(7):
        labels[1, 0, st, int(0.35 * T) + st] = 1.0
        labels[1, 1, st, int(0.45 * T) + st] = 1.0
    order, npick, mo, info = _pick_event(labels, 2000.0)
    assert npick[0] == 12 and npick[1] == 7, (npick[0], npick[1])
    assert order[0] == 1, f"centred event not preferred: order={order}"
    assert info["selection_stage"] == ">=6 and centred", info["selection_stage"]
    # and when a centred event DOES carry enough picks, it wins at stage 1
    labels2 = labels.copy()
    for st in range(7, 12):
        labels2[1, 0, st, int(0.35 * T) + st] = 1.0
    _, _, _, info2 = _pick_event(labels2, 2000.0)
    assert info2["selection_stage"] == ">=MIN_PICKS and centred", info2

    # alias matching
    assert _match_site("FORGE_Geothermal_2019") == "forge_19"
    assert _match_site("PNR-1") == "pnr-1"
    assert _match_site("CottonValleyStageB") is None
    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--fig4", action="store_true")
    ap.add_argument("--table-s1", action="store_true")
    ap.add_argument("--attr-dir", default=None,
                    help="local directory holding <Dataset>.Array.csv files")
    ap.add_argument("--zenodo-url", default=ZEN_URL)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.fig4:
        run_fig4()
    if a.table_s1:
        run_table_s1(a.attr_dir, a.zenodo_url)
    if not (a.fig4 or a.table_s1):
        ap.print_help()


if __name__ == "__main__":
    main()
