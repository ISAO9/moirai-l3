#!/usr/bin/env python
"""
17_geometry_and_distance.py -- GJI major revision round 2: characterise the
array geometries and the source-array distances directly, rather than through
a pick-derived proxy.
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Reviewer 1 (round 2) made the sharpest methodological criticism of the paper:

    "The study relies heavily on phase moveouts ... and inter-quartile range
     (IQR) as a representation of the array geometry, but it is not clear how
     these two factors fully distinguish one array from another. For example,
     it is not clear to me why forge_19 has a different geometry as pnr-1
     based on the number in Table 1. Moreover, these moveouts are affected by
     the available picks, which are also affected by event magnitudes,
     cataloging methods, attenuation and noise level."

The reviewer is right that a descriptor computed FROM the catalogue inherits
the catalogue's biases. The answer is to show the geometry itself. Both
reviewers and the editor also asked for maps of sources and receivers, and
for the moveout ranges rather than a median and an IQR.

AMBER distributes, per site, the station coordinates (Attr/<ds>.Array.csv) and
a 1-D layered velocity model (Attr/<ds>.Vmodel.csv). The first is a pick-free
description of the receiver geometry. The second, combined with the catalogued
S-P times, places the sources well enough for a descriptive figure -- and S-P
time is itself the distance proxy Reviewer 2 asked to see in Table 1.

WHAT THIS SCRIPT DOES
---------------------
  1. GEOMETRY  -- per-site receiver strings from Array.csv, drawn to scale, so
     aperture, station spacing and well deviation are visible side by side.
  2. RANGES    -- the FULL moveout distribution per site and phase (minimum,
     quartiles, maximum) and the S-P time distribution, recomputed from the
     catalogued arrivals cached by script 10. These replace the median/IQR
     pair in Table 1 and give the "relevant moveout ranges" the editor asked
     for, including the training maximum the forge_19 argument rests on.
  3. LOCATIONS -- (--locate) approximate source positions by trilateration on
     the catalogued S-P times through the site's own 1-D model. S-P time
     removes the origin time, so no absolute timing is needed. These are
     DESCRIPTIVE ONLY: the picker itself uses no velocity model anywhere, and
     the locations are never fed back into any result in the paper.

OUTPUTS
  logs/17_array_coords.json        station coordinates, cached for redraws
  logs/17_geometry_table.csv/.md   extended Table 1 (ranges + S-P + noise)
  logs/17_source_locations.json    located sources + RMS misfit  (--locate)
  PDF/17_a_geometry_ranges.pdf     receiver strings + moveout + S-P ranges
  PDF/17_b_source_receiver.pdf     source-receiver cross-sections (--locate)

All figures: white background, English labels, legends clear of the data.

RUN
  # geometry + ranges (needs the script-10 caches in logs/, and the network
  # once to read the few-kB Attr/*.csv members out of the Zenodo archive):
  pip -q install remotezip
  python src/17_geometry_and_distance.py --geometry
  # with a local copy of the Attr/ directory instead of the network:
  python src/17_geometry_and_distance.py --geometry --attr-dir /path/to/Attr
  # add the approximate source locations:
  python src/17_geometry_and_distance.py --geometry --locate
  # self-test (synthetic, no caches, no network):
  python src/17_geometry_and_distance.py --selftest
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
DATA = cfg.DATA

ZEN_URL = ("https://zenodo.org/records/18944111/files/amber_raw_segys.zip")

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]

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

HILITE = "forge_19"          # the out-of-distribution site, marked throughout


# ----------------------------------------------------------------------------
# AMBER attribute files (station coordinates, velocity models)
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_site(dataset_name: str):
    n = _norm(dataset_name)
    for site, aliases in SITE_ALIASES.items():
        for a in aliases:
            an = _norm(a)
            if an == n or an in n or n in an:
                return site
    return None


def _parse_array_csv(text: str) -> np.ndarray:
    """Array.csv -> (N,3) x, y, depth in metres, sorted by depth."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue                                   # header
    if len(rows) < 2:
        raise ValueError("Array.csv parsed fewer than 2 stations")
    xyz = np.asarray(rows, float)
    return xyz[np.argsort(xyz[:, 2])]


def _parse_vmodel_csv(text: str) -> np.ndarray:
    """Vmodel.csv -> (L,3) layer top depth (m), Vp, Vs (m/s), sorted by depth."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        raise ValueError("Vmodel.csv parsed no layers")
    v = np.asarray(rows, float)
    return v[np.argsort(v[:, 0])]


def _iter_attr(attr_dir, url, suffix):
    if attr_dir:
        for f in sorted(Path(attr_dir).glob(f"*{suffix}")):
            yield f.name.replace(suffix, ""), f.read_text(
                encoding="utf-8", errors="replace")
    else:
        from remotezip import RemoteZip
        with RemoteZip(url) as z:
            names = [n for n in z.namelist() if n.endswith(suffix)]
            print(f"[attr] {len(names)} {suffix} members in the archive")
            for n in names:
                yield Path(n).name.replace(suffix, ""), z.read(n).decode(
                    "utf-8", errors="replace")


def read_attributes(attr_dir, url, want_vmodel: bool) -> dict:
    """{site: {'xyz': (N,3), 'vmodel': (L,3)}} for the eight benchmark sites."""
    out = {}
    for ds, text in _iter_attr(attr_dir, url, ".Array.csv"):
        site = _match_site(ds)
        if site in SITE_ALIASES:
            out.setdefault(site, {})["xyz"] = _parse_array_csv(text)
            out[site]["dataset"] = ds
    if want_vmodel:
        for ds, text in _iter_attr(attr_dir, url, ".Vmodel.csv"):
            site = _match_site(ds)
            if site in out:
                out[site]["vmodel"] = _parse_vmodel_csv(text)
    for site, d in sorted(out.items()):
        n = len(d["xyz"])
        print(f"[attr] {site:<16s} {n:>3d} stations"
              + ("" if "vmodel" not in d else
                 f", {len(d['vmodel'])} velocity layers"))
    return out


# ----------------------------------------------------------------------------
# catalogued arrival statistics, from the script-10 caches
# ----------------------------------------------------------------------------
def load_caches(ckpt: str = "ema") -> dict:
    """{site: stats} -- the catalogued arrivals are identical across
    configurations, so one cache per site is enough."""
    caches = {}
    for f in sorted(cfg.LOGS_DIR.glob(f"10_cache_*_{ckpt}*loso_*.npz")):
        stem = f.stem
        if "_pertrace" in stem or "_shuf" in stem:
            continue
        caches[stem.split("_loso_")[-1]] = dict(np.load(f))
    return caches


def arrival_stats(stats: dict, fs: float) -> dict:
    """Full moveout and S-P distributions from the catalogued arrivals.

    moveout(event, phase) = latest minus earliest catalogued arrival over the
    stations carrying that phase (>= 2 needed), in ms -- the definition used in
    Table 1. S-P is per station, over stations carrying both arrivals.
    """
    eq = ~stats["is_noise"]
    gt = stats["gt_time"]                                  # (E,2,S)
    out = {"n_events": int(eq.sum()),
           "n_noise_windows": int(stats["is_noise"].sum())}
    for ch, name in ((0, "P"), (1, "S")):
        mo = []
        for e in np.where(eq)[0]:
            v = gt[e, ch][gt[e, ch] >= 0]
            if len(v) >= 2:
                mo.append((v.max() - v.min()) / fs * 1000.0)
        mo = np.asarray(mo, float)
        out[name] = _dist(mo)
        out[name]["n_events_with_moveout"] = int(len(mo))
    both = eq[:, None] & (gt[:, 0] >= 0) & (gt[:, 1] >= 0)
    sp = (gt[:, 1] - gt[:, 0])[both] / fs * 1000.0
    out["SP"] = _dist(np.asarray(sp, float))
    out["SP"]["n_station_pairs"] = int(both.sum())
    return out


def _dist(a: np.ndarray) -> dict:
    if a.size == 0:
        return dict(n=0)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return dict(n=int(a.size), min_ms=float(q[0]), q25_ms=float(q[1]),
                median_ms=float(q[2]), q75_ms=float(q[3]), max_ms=float(q[4]),
                iqr_ms=float(q[3] - q[1]))


# ----------------------------------------------------------------------------
# approximate source locations from the catalogued S-P times
# ----------------------------------------------------------------------------
def _depth_average(vmodel: np.ndarray, z0: float, z1: float) -> tuple:
    """Thickness-weighted mean Vp, Vs between two depths (metres)."""
    lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
    if hi - lo < 1.0:                                  # degenerate: nearest layer
        k = int(np.searchsorted(vmodel[:, 0], lo, side="right") - 1)
        k = max(0, min(k, len(vmodel) - 1))
        return float(vmodel[k, 1]), float(vmodel[k, 2])
    tops = vmodel[:, 0]
    bots = np.append(tops[1:], max(hi, tops[-1]) + 1.0)
    w = np.clip(np.minimum(bots, hi) - np.maximum(tops, lo), 0.0, None)
    if w.sum() <= 0:
        k = max(0, min(len(vmodel) - 1,
                       int(np.searchsorted(tops, lo, side="right") - 1)))
        return float(vmodel[k, 1]), float(vmodel[k, 2])
    return (float((w * vmodel[:, 1]).sum() / w.sum()),
            float((w * vmodel[:, 2]).sum() / w.sum()))


def string_axis(xyz: np.ndarray) -> tuple:
    """Centroid and downward unit direction of the station string (PC1)."""
    c = xyz.mean(axis=0)
    u = np.linalg.svd(xyz - c, full_matrices=False)[2][0]
    if u[2] < 0:
        u = -u
    return c, u


def locate_event(xyz: np.ndarray, sp_ms: np.ndarray, vmodel: np.ndarray,
                 n_iter: int = 3) -> tuple:
    """Locate one source in the (radial distance, along-axis) plane.

    For a straight ray through path-averaged velocities,
        t_S - t_P = d (1/Vs - 1/Vp)   ->   d = (t_S - t_P) / (1/Vs - 1/Vp),
    so each station with an S-P time gives a distance and the origin time
    cancels -- no absolute timing is needed.

    The source is NOT solved for in three dimensions. A near-vertical string
    is axially symmetric, so the azimuth around it is not constrained by
    distances alone, and a three-parameter search collapses onto the axis
    (its starting point) and reports a spurious r = 0. What the data DO
    constrain is the pair (r, s): the perpendicular distance from the string
    axis and the position along it. Those two are searched on a grid and then
    refined, which is deterministic and cannot be trapped the way a simplex
    started on the symmetry axis is.

    Returns (xyz_in_the_azimuth_0_plane, r_m, depth_m, rms_m).
    """
    good = np.isfinite(sp_ms) & (sp_ms > 0)
    if good.sum() < 4:
        return None, np.nan, np.nan, np.nan
    pts, sp = xyz[good], sp_ms[good] / 1000.0

    c, u = string_axis(xyz)
    e = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(e) < 1e-6:                  # vertical string
        e = np.array([1.0, 0.0, 0.0])
    e = e / np.linalg.norm(e)
    s_st = (pts - c) @ u                          # stations along the axis

    def rms_for(r, s_ax, d_obs):
        pos = c + s_ax * u + r * e
        return float(np.sqrt(((np.linalg.norm(pts - pos[None, :], axis=1)
                               - d_obs) ** 2).mean()))

    best = (0.0, float(s_st.mean()))
    rms = np.nan
    for _ in range(n_iter):
        depth = float(c[2] + best[1] * u[2])
        vp, vs = _depth_average(vmodel, depth, float(pts[:, 2].mean()))
        slow = 1.0 / vs - 1.0 / vp
        if slow <= 0:
            return None, np.nan, np.nan, np.nan
        d_obs = sp / slow
        span = max(float(d_obs.max()), 1.0)

        # coarse grid over (r, s), then refinements centred on the best cell.
        # The grids are ODD so that the previous best is itself a grid point,
        # and a stage may only improve on it: without both, a refinement window
        # that straddles the optimum can drift along the shallow radial valley
        # and trade a small error in s for a large spurious r.
        r_lo, r_hi = 0.0, 1.3 * span
        s_lo, s_hi = s_st.min() - 1.3 * span, s_st.max() + 1.3 * span
        rms = rms_for(best[0], best[1], d_obs)
        for n_grid, shrink in ((91, 0.12), (41, 0.04), (41, 0.0)):
            rs = np.linspace(r_lo, r_hi, n_grid)
            ss = np.linspace(s_lo, s_hi, n_grid)
            vals = np.array([[rms_for(r, sa, d_obs) for sa in ss] for r in rs])
            i, j = np.unravel_index(int(np.argmin(vals)), vals.shape)
            if float(vals[i, j]) < rms:
                best, rms = (float(rs[i]), float(ss[j])), float(vals[i, j])
            if shrink:
                dr, ds = shrink * (r_hi - r_lo), shrink * (s_hi - s_lo)
                r_lo, r_hi = max(0.0, best[0] - dr), best[0] + dr
                s_lo, s_hi = best[1] - ds, best[1] + ds

    r, s_ax = best
    pos = c + s_ax * u + r * e
    return pos, r, float(pos[2]), rms


def locate_site(stats: dict, xyz: np.ndarray, vmodel: np.ndarray, fs: float,
                max_events: int = 400) -> dict:
    """Locate up to max_events earthquakes of one site in (r, depth)."""
    eq = np.where(~stats["is_noise"])[0]
    if len(eq) > max_events:                     # even thinning, reproducible
        eq = eq[np.linspace(0, len(eq) - 1, max_events).astype(int)]
    gt = stats["gt_time"]
    n_st = min(gt.shape[2], len(xyz))
    radial, depth, rmss = [], [], []
    for e in eq:
        p, s = gt[e, 0, :n_st], gt[e, 1, :n_st]
        sp = np.where((p >= 0) & (s >= 0), (s - p) / fs * 1000.0, np.nan)
        _, r, z, rms = locate_event(xyz[:n_st], sp, vmodel)
        if np.isfinite(rms):
            radial.append(float(r))
            depth.append(float(z))
            rmss.append(float(rms))
    if not radial:
        return dict(n=0)
    return dict(n=len(radial), radial_m=radial, depth_m=depth, rms_m=rmss,
                median_rms_m=float(np.median(rmss)),
                median_radial_m=float(np.median(radial)),
                median_depth_m=float(np.median(depth)),
                note="azimuth is not constrained by S-P distances on a "
                     "near-vertical string, so only (radial distance from the "
                     "string axis, depth) is reported; radial resolution "
                     "degrades close to the axis, where the distances vary "
                     "only as the square of the offset")


# ----------------------------------------------------------------------------
# table
# ----------------------------------------------------------------------------
def build_table(attrs: dict, arr: dict) -> tuple:
    hdr = ("site,n_events,n_noise_windows,n_stations_string,"
           "vertical_extent_m,aperture_3d_m,median_spacing_m,"
           "P_moveout_min_ms,P_moveout_median_ms,P_moveout_max_ms,"
           "S_moveout_min_ms,S_moveout_median_ms,S_moveout_max_ms,"
           "SP_median_ms,SP_min_ms,SP_max_ms")
    csv = [hdr]
    md = ["| Site | Events | Noise windows | Stations | Aperture (m) | "
          "Spacing (m) | P moveout min / median / max (ms) | "
          "S moveout min / median / max (ms) | S-P median (range) (ms) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SITE_ORDER:
        if s not in arr:
            continue
        a, g = arr[s], attrs.get(s, {})
        xyz = g.get("xyz")
        if xyz is not None:
            n_st = len(xyz)
            vert = float(xyz[:, 2].max() - xyz[:, 2].min())
            ap = float(np.linalg.norm(xyz[-1] - xyz[0]))
            sp_m = float(np.median(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
        else:
            n_st, vert, ap, sp_m = 0, float("nan"), float("nan"), float("nan")
        P, S, SP = a["P"], a["S"], a["SP"]
        csv.append(
            f"{s},{a['n_events']},{a['n_noise_windows']},{n_st},"
            f"{vert:.1f},{ap:.1f},{sp_m:.1f},"
            f"{P['min_ms']:.1f},{P['median_ms']:.1f},{P['max_ms']:.1f},"
            f"{S['min_ms']:.1f},{S['median_ms']:.1f},{S['max_ms']:.1f},"
            f"{SP['median_ms']:.1f},{SP['min_ms']:.1f},{SP['max_ms']:.1f}")
        md.append(
            f"| {s} | {a['n_events']} | {a['n_noise_windows']} | {n_st} | "
            f"{ap:.0f} | {sp_m:.1f} | "
            f"{P['min_ms']:.0f} / {P['median_ms']:.0f} / {P['max_ms']:.0f} | "
            f"{S['min_ms']:.0f} / {S['median_ms']:.0f} / {S['max_ms']:.0f} | "
            f"{SP['median_ms']:.0f} ({SP['min_ms']:.0f}–{SP['max_ms']:.0f}) |")
    return "\n".join(csv) + "\n", "\n".join(md) + "\n"


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    return plt


def make_geometry_figure(attrs: dict, arr: dict, out_pdf: Path):
    """Receiver strings to scale, then the full moveout and S-P ranges."""
    plt = _style()
    sites = [s for s in SITE_ORDER if s in arr]
    fig = plt.figure(figsize=(12.4, 9.0))
    gs = fig.add_gridspec(3, len(sites), height_ratios=[1.5, 1.0, 1.0],
                          hspace=0.52, wspace=0.42)

    # (a) receiver geometry, depth relative to the shallowest station.
    # The limits are common to all panels so apertures can be compared by eye,
    # and are taken from the data: pnr-1 and pnr-2 are deviated wells whose
    # horizontal extent runs to several hundred metres, which a fixed range
    # would silently clip.
    dxs, dzs = [], []
    for s in sites:
        xyz = attrs.get(s, {}).get("xyz")
        if xyz is not None:
            dxs.append(float(np.abs(xyz[:, 0] - xyz[0, 0]).max()))
            dzs.append(float((xyz[:, 2] - xyz[:, 2].min()).max()))
    x_lim = max(40.0, (max(dxs) if dxs else 40.0) * 1.25)
    z_lim = (max(dzs) if dzs else 600.0) * 1.12 + 20.0

    for j, s in enumerate(sites):
        ax = fig.add_subplot(gs[0, j])
        xyz = attrs.get(s, {}).get("xyz")
        col = "#1f77b4" if s == HILITE else "0.35"
        if xyz is not None:
            dx = xyz[:, 0] - xyz[0, 0]
            dz = xyz[:, 2] - xyz[:, 2].min()
            ax.plot(dx, dz, "-", color=col, lw=1.0, zorder=1)
            ax.plot(dx, dz, "v", color=col, ms=3.4, zorder=2)
            ax.set_title(f"{s}\n{len(xyz)} st, {dz.max():.0f} m",
                         fontsize=8.2)
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(z_lim, -0.04 * z_lim)
        ax.set_xlabel("x (m)", fontsize=8)
        if j == 0:
            ax.set_ylabel("depth below\nshallowest station (m)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.5)

    # (b) moveout ranges, (c) S-P ranges
    def box(ax, key, label, phases):
        xs = np.arange(len(sites))
        for off, (k, c, nm) in zip((-0.17, 0.17) if len(phases) == 2 else (0.0,),
                                   phases):
            for i, s in enumerate(sites):
                d = arr[s][k]
                if not d.get("n"):
                    continue
                x = xs[i] + off
                ax.plot([x, x], [d["min_ms"], d["max_ms"]], "-", color=c,
                        lw=0.9, alpha=0.55, zorder=1)
                ax.add_patch(plt.Rectangle(
                    (x - 0.13, d["q25_ms"]), 0.26, d["q75_ms"] - d["q25_ms"],
                    facecolor=c, alpha=0.30, edgecolor=c, lw=1.0, zorder=2))
                ax.plot([x - 0.13, x + 0.13], [d["median_ms"]] * 2, "-",
                        color=c, lw=1.8, zorder=3, label=nm if i == 0 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(sites, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
        for i, s in enumerate(sites):
            if s == HILITE:
                ax.axvspan(i - 0.42, i + 0.42, color="#1f77b4", alpha=0.07,
                           zorder=0)

    ax_b = fig.add_subplot(gs[1, :])
    box(ax_b, None, "across-station moveout (ms)",
        [("P", "#1f77b4", "P"), ("S", "#d62728", "S")])
    ax_b.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
                frameon=True, framealpha=0.95, edgecolor="0.8",
                title="phase", title_fontsize=8)

    ax_c = fig.add_subplot(gs[2, :])
    box(ax_c, None, "S–P time (ms)", [("SP", "0.35", "S–P")])
    ax_c.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
                frameon=True, framealpha=0.95, edgecolor="0.8")

    fig.suptitle("Array geometry and the ranges behind the moveout descriptor "
                 "(bars: full range; boxes: inter-quartile; line: median)",
                 fontsize=11.5)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.07)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf} (+ .png)")


def make_source_figure(attrs: dict, locs: dict, out_pdf: Path):
    """Source-receiver cross-sections, one panel per site."""
    plt = _style()
    sites = [s for s in SITE_ORDER if locs.get(s, {}).get("n")]
    if not sites:
        print("[figure] no located sources to draw")
        return
    ncol = 4
    nrow = int(np.ceil(len(sites) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 3.3 * nrow),
                             squeeze=False)
    hs = hr = None
    for k, s in enumerate(sites):
        ax = axes[k // ncol][k % ncol]
        xyz = attrs[s]["xyz"]
        c_ax, u_ax = string_axis(xyz)
        s_st = (xyz - c_ax) @ u_ax
        r_st = np.linalg.norm(xyz - (c_ax[None, :] + s_st[:, None] * u_ax[None, :]),
                              axis=1)
        hs = ax.scatter(locs[s]["radial_m"], locs[s]["depth_m"], s=7,
                        c="#1f77b4", alpha=0.45, lw=0, zorder=2)
        hr, = ax.plot(r_st, xyz[:, 2], "v", color="k", ms=4.5, zorder=3)
        ax.invert_yaxis()
        ax.set_title(f"{s}  (n = {locs[s]['n']}, "
                     f"median misfit {locs[s]['median_rms_m']:.0f} m)",
                     fontsize=9)
        ax.set_xlabel("distance from string axis (m)", fontsize=8)
        ax.set_ylabel("depth (m)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.5)
    for k in range(len(sites), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend([hr, hs], ["stations", "located sources"], loc="lower center",
               ncol=2, fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="0.8", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Approximate source positions from catalogued S–P times "
                 "(descriptive only; a near-vertical string does not "
                 "constrain azimuth)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.055, 1, 0.95])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf} (+ .png)")


# ----------------------------------------------------------------------------
# self-test (synthetic; no caches, no network)
# ----------------------------------------------------------------------------
def _synth_cache(n_ev=120, n_st=12, fs=2000.0, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    gt = np.full((n_ev, 2, n_st), -1, np.int64)
    is_noise = np.zeros(n_ev, bool)
    is_noise[: n_ev // 10] = True
    for e in range(n_ev):
        if is_noise[e]:
            continue
        base = int(rng.integers(300, 500))
        for s in range(n_st):
            gt[e, 0, s] = base + 10 * s
            gt[e, 1, s] = base + 300 + 17 * s
    return dict(gt_time=gt, pr_max=np.ones((n_ev, 2, n_st), np.float32),
                pr_time=gt.copy(), is_noise=is_noise)


def selftest():
    fs = 2000.0

    # parsers
    acsv = "ID,X,Y,Depth\n" + "\n".join(
        f"{i+1},0.0,0.0,{1000 + 30*i:.1f}" for i in range(12))
    xyz = _parse_array_csv(acsv)
    assert xyz.shape == (12, 3)
    assert abs(xyz[:, 2].max() - xyz[:, 2].min() - 330.0) < 1e-6
    vcsv = "Layer,Top,Vp,Vs\n1,0,3000,1700\n2,800,4500,2600\n3,1600,5200,3000"
    vm = _parse_vmodel_csv(vcsv)
    assert vm.shape == (3, 3) and vm[1, 1] == 4500

    # depth average: inside one layer, and spanning two
    vp, vs = _depth_average(vm, 900.0, 1100.0)
    assert abs(vp - 4500) < 1e-6, vp
    vp2, _ = _depth_average(vm, 600.0, 1000.0)
    assert 3000 < vp2 < 4500, vp2

    # arrival statistics on a known synthetic catalogue
    a = arrival_stats(_synth_cache(fs=fs), fs)
    assert a["n_events"] == 108 and a["n_noise_windows"] == 12
    assert abs(a["P"]["median_ms"] - 11 * 10 / fs * 1000.0) < 1e-6, a["P"]
    assert abs(a["SP"]["min_ms"] - 300 / fs * 1000.0) < 1e-6, a["SP"]
    assert a["S"]["max_ms"] >= a["S"]["min_ms"]

    # location: recover a planted source's (radial distance, depth).
    # The azimuth is deliberately NOT recovered -- a vertical string cannot
    # constrain it -- so the test plants a source at a known radius instead.
    pts = xyz.copy()
    r_true, z_true = 134.2, 1500.0
    src = np.array([r_true, 0.0, z_true])
    d = np.linalg.norm(pts - src[None, :], axis=1)
    vp, vs = _depth_average(vm, z_true, float(pts[:, 2].mean()))
    sp_ms = d * (1.0 / vs - 1.0 / vp) * 1000.0
    _, r_got, z_got, rms = locate_event(pts, sp_ms, vm)
    assert rms < 5.0, rms
    assert abs(r_got - r_true) < 25.0, (r_got, r_true)
    assert abs(z_got - z_true) < 25.0, (z_got, z_true)

    # A source ON the axis must come back near r = 0 and at the right depth.
    # The radial tolerance is wider than the depth tolerance on purpose:
    # d = sqrt(r^2 + dz^2) gives dd/dr = r/d, which vanishes at r = 0, so the
    # misfit is flat in r close to the axis and the radius there is resolved
    # only to a few tens of metres -- against an aperture of 330 m and source
    # distances of 270-600 m. Depth stays well resolved.
    src0 = np.array([0.0, 0.0, 1600.0])
    d0 = np.linalg.norm(pts - src0[None, :], axis=1)
    vp0, vs0 = _depth_average(vm, 1600.0, float(pts[:, 2].mean()))
    _, r0, z0, rms0 = locate_event(pts, d0 * (1 / vs0 - 1 / vp0) * 1000.0, vm)
    assert r0 < 50.0, r0
    assert abs(z0 - 1600.0) < 15.0, z0
    assert rms0 < 3.0, rms0

    # too few stations -> no location, not a crash
    none, rn, zn, nan = locate_event(pts, np.full(12, np.nan), vm)
    assert none is None and np.isnan(nan)

    # table + figure paths
    attrs = {"forge_19": {"xyz": xyz, "dataset": "FORGE_2019"},
             "pnr-1": {"xyz": xyz, "dataset": "PNR-1"}}
    arr = {"forge_19": a, "pnr-1": arrival_stats(_synth_cache(seed=1), fs)}
    csv, md = build_table(attrs, arr)
    assert "forge_19" in csv and "P moveout min" in md
    out = cfg.PDF_DIR / "17_selftest_geometry.pdf"
    make_geometry_figure(attrs, arr, out)
    assert out.exists() and out.stat().st_size > 8_000

    locs = {"forge_19": locate_site(_synth_cache(), xyz, vm, fs,
                                    max_events=20)}
    assert locs["forge_19"]["n"] > 0
    out2 = cfg.PDF_DIR / "17_selftest_sources.pdf"
    make_source_figure(attrs, locs, out2)
    assert out2.exists() and out2.stat().st_size > 5_000

    print("[selftest] ALL PASS")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--geometry", action="store_true",
                    help="receiver strings, moveout/S-P ranges, Table 1")
    ap.add_argument("--locate", action="store_true",
                    help="also locate sources from the catalogued S-P times")
    ap.add_argument("--attr-dir", default=None,
                    help="local Attr/ directory instead of the Zenodo archive")
    ap.add_argument("--zenodo-url", default=ZEN_URL)
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--max-events", type=int, default=400,
                    help="events located per site (--locate)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.geometry or a.locate):
        ap.print_help()
        return

    fs = float(getattr(DATA, "FS", 2000.0))
    caches = load_caches(a.ckpt)
    if not caches:
        raise SystemExit(f"[run] no 10_cache_*_{a.ckpt}*loso_*.npz in "
                         f"{cfg.LOGS_DIR} -- build them with script 10 --cache")
    attrs = read_attributes(a.attr_dir, a.zenodo_url, want_vmodel=a.locate)

    arr = {s: arrival_stats(st, fs) for s, st in caches.items()}
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR / "17_array_coords.json").write_text(json.dumps(
        {s: dict(dataset=d.get("dataset"), xyz=d["xyz"].tolist())
         for s, d in attrs.items() if "xyz" in d}, indent=2))
    csv, md = build_table(attrs, arr)
    (cfg.LOGS_DIR / "17_geometry_table.csv").write_text(csv)
    (cfg.LOGS_DIR / "17_geometry_table.md").write_text(md)
    print(f"[table] wrote {cfg.LOGS_DIR/'17_geometry_table.csv'} and .md")
    print(md)
    make_geometry_figure(attrs, arr, cfg.PDF_DIR / "17_a_geometry_ranges.pdf")

    if a.locate:
        locs = {}
        for s, st in sorted(caches.items()):
            d = attrs.get(s, {})
            if "xyz" not in d or "vmodel" not in d:
                print(f"[locate] {s}: no coordinates or velocity model, skipped")
                continue
            locs[s] = locate_site(st, d["xyz"], d["vmodel"], fs, a.max_events)
            print(f"[locate] {s:<16s} {locs[s].get('n', 0):>4d} events, "
                  f"median misfit {locs[s].get('median_rms_m', float('nan')):.0f} m")
        (cfg.LOGS_DIR / "17_source_locations.json").write_text(
            json.dumps(locs, indent=2))
        make_source_figure(attrs, locs,
                           cfg.PDF_DIR / "17_b_source_receiver.pdf")


if __name__ == "__main__":
    main()
