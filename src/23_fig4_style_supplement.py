#!/usr/bin/env python
"""
23_fig4_style_supplement.py -- GJI revision round 2: Figure-4-style panels for
the remaining sites, for the supplement. (Reviewer 2, page 14 annotation)
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Reviewer 2 annotated Section 4.1, where the array model's behaviour across the
eight held-out sites is described, with

    "Add examples like Figure 4 for these cases to the supplement"

Figure 4 shows only two sites: forge_19 (the failure) and mseel_5h (the
in-distribution contrast). The reviewer is asking to see the same waveform and
score panels for the sites the section actually discusses, so that the reader
can check the claim "seven of eight generalize well" against the pictures
rather than against the table alone.

This script produces exactly that for the SIX sites not already in Figure 4:

    pnr-1, mseel_3h, clearfield_mw6, pnr-2, clearfield_mw4, aneth

Every panel convention -- event selection, wiggle plot, score map, colour,
the catalogued-pick markers -- is imported from script 14, so the supplement
figure cannot drift from Figure 4. The event-selection rule is script 14's
patched one, which prefers an event whose arrivals sit inside the window over
one with more stations jammed against its edge (reviewer 1, R1-5).

WHAT IT DOES NOT DO
-------------------
It does not recompute any number that appears in the paper. It is a picture of
model output that already exists, nothing more.

OUTPUTS
  PDF/23_a_fig4_style_sites.pdf   six rows x (waveforms, P score, S score)
                                  (also .png at 300 dpi)
  logs/23_event_choice.json       which event was shown per site, and why

All figures: white background, English labels, legend clear of the data.

RUN (same machine as script 14: Drive mounted, AMBER_H5 set, model/ holding
     the leave-one-site-out checkpoints; a GPU is used if present)
  python src/23_fig4_style_supplement.py --figure
  python src/23_fig4_style_supplement.py --figure --sites pnr-2 aneth
  # self-test (synthetic payload; no AMBER, no checkpoints, no GPU):
  python src/23_fig4_style_supplement.py --selftest
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
    """Numbered files cannot be imported, so load them by path."""
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")
F14 = _load("14_fig4_and_tableS1.py")       # panel conventions come from here

# the sites Section 4.1 discusses that Figure 4 does not already show
DEFAULT_SITES = ("pnr-1", "mseel_3h", "clearfield_mw6",
                 "pnr-2", "clearfield_mw4", "aneth")


# ----------------------------------------------------------------------------
def collect(site: str) -> tuple[dict, dict]:
    """Run script 14's inference and event choice for one site."""
    fs = float(getattr(F14.DATA, "SAMPLE_RATE", 2000.0))
    ds, preds, labels, ctx = F14._infer_site(site)
    order, npick, mo, info = F14._pick_event(labels, fs)
    # Same search as Figure 4: one access per panel (R1-5) plus a scan over
    # candidate events so a larger neighbouring arrival stays out of the panel.
    idx, waves_z, prob, gt, dinfo = F14.pick_panel(ctx, ds, order, fs)
    info.update(event_index=idx, n_p_picks=int(npick[idx]),
                p_moveout_ms=float(mo[idx]), **dinfo)
    payload = dict(waves_z=waves_z, prob=prob, gt=gt, fs=fs)
    return payload, info


# ----------------------------------------------------------------------------
def make_figure(payload: dict, sites, out_pdf: Path):
    """One row per site: Z waveforms, P score map, S score map.

    The legend gets its own strip under the grid, so it never covers data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    n = len(sites)
    fig, axes = plt.subplots(n, 3, figsize=(13.2, 1.95 * n + 1.5),
                             squeeze=False)
    handles = {}
    for i, site in enumerate(sites):
        d = payload[site]
        hP, hS = F14._wiggle_panel(axes[i][0], d["waves_z"], d["gt"], d["fs"])
        handles.setdefault("P pick (catalogue)", hP)
        handles.setdefault("S pick (catalogue)", hS)
        axes[i][0].set_title(f"{site} — Z-component waveforms", fontsize=10)

        imP, hG = F14._prob_panel(axes[i][1], d["prob"][0], d["gt"][0],
                                  d["fs"], "Blues")
        handles.setdefault("catalogued pick on score maps", hG)
        axes[i][1].set_title(f"{site} — predicted P score", fontsize=10)

        imS, _ = F14._prob_panel(axes[i][2], d["prob"][1], d["gt"][1],
                                 d["fs"], "Reds")
        axes[i][2].set_title(f"{site} — predicted S score", fontsize=10)

        fig.colorbar(imP, ax=axes[i][1], fraction=0.045, pad=0.02)
        fig.colorbar(imS, ax=axes[i][2], fraction=0.045, pad=0.02)

    order = ["P pick (catalogue)", "S pick (catalogue)",
             "catalogued pick on score maps"]
    hh = [handles[k] for k in order if handles.get(k) is not None]
    ll = [k for k in order if handles.get(k) is not None]
    fig.suptitle("Waveforms and predicted scores at the remaining held-out "
                 "sites (array configuration)", fontsize=12)
    # reserve a strip at the bottom for the legend; nothing is drawn over data
    fig.tight_layout(rect=[0, 0.055, 1, 0.975])
    fig.legend(hh, ll, loc="lower center", ncol=len(ll), frameon=True,
               framealpha=0.95, edgecolor="0.8", fontsize=9,
               bbox_to_anchor=(0.5, 0.004))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[23] wrote {out_pdf} (+ .png)")


def run(sites) -> None:
    payload, choice = {}, {}
    for site in sites:
        payload[site], choice[site] = collect(site)
        c = choice[site]
        print(f"[23] {site}: event {c['event_index']} "
              f"(P picks {c['n_p_picks']}, moveout {c['p_moveout_ms']:.1f} ms, "
              f"stage '{c['selection_stage']}', events {c['events_tried']}, "
              f"draws {c['draws_tried']}, centred {c['draw_centred']}, "
              f"late-energy {c['late_energy_ratio']:.2f}, ok {c['panel_ok']})")
    make_figure(payload, sites, cfg.PDF_DIR / "23_a_fig4_style_sites.pdf")
    out = cfg.LOGS_DIR / "23_event_choice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(choice, indent=2))
    print(f"[23] wrote {out}")


# ----------------------------------------------------------------------------
# self-test (synthetic payload; no AMBER, no checkpoints, no GPU)
# ----------------------------------------------------------------------------
def selftest() -> None:
    rng = np.random.default_rng(0)
    fs, S, T = 2000.0, 12, 2048
    sites = ("siteA", "siteB", "siteC")
    payload = {}
    for k, site in enumerate(sites):
        waves_z = rng.normal(0, 1, (S, T)).astype(np.float32)
        prob = np.zeros((3, S, T), np.float32)
        gt = np.full((2, S), -1, np.int64)
        for s in range(S):
            tp, ts = 500 + (8 + 4 * k) * s, 900 + (14 + 4 * k) * s
            gt[0, s], gt[1, s] = tp, ts
            prob[0, s, max(0, tp - 20):tp + 20] = 0.9
            prob[1, s, max(0, ts - 20):ts + 20] = 0.85
            waves_z[s, tp:tp + 60] += 4 * np.hanning(60)
        payload[site] = dict(waves_z=waves_z, prob=prob, gt=gt, fs=fs)

    out = cfg.PDF_DIR / "23_selftest_fig4_style.pdf"
    make_figure(payload, sites, out)
    assert out.exists() and out.stat().st_size > 10_000, out
    assert out.with_suffix(".png").exists()

    # the figure must scale with the number of sites, not be fixed at three
    out2 = cfg.PDF_DIR / "23_selftest_one_site.pdf"
    make_figure({k: payload[k] for k in sites[:1]}, sites[:1], out2)
    assert out2.exists() and out2.stat().st_size > 5_000

    # the default site list must not duplicate what Figure 4 already shows
    assert set(DEFAULT_SITES).isdisjoint(set(F14.SITES_FIG4)), DEFAULT_SITES
    assert len(DEFAULT_SITES) == 6, DEFAULT_SITES

    # script 14's patched selection rule must be the one in use
    labels = np.zeros((2, 3, 12, T), np.float32)
    for st in range(12):
        labels[0, 0, st, int(0.90 * T) + st] = 1.0
        labels[0, 1, st, int(0.95 * T) + st] = 1.0
    for st in range(7):
        labels[1, 0, st, int(0.35 * T) + st] = 1.0
        labels[1, 1, st, int(0.45 * T) + st] = 1.0
    order, _, _, info = F14._pick_event(labels, fs)
    assert order[0] == 1, "script 14 is not the patched version (R1-5)"
    assert "selection_stage" in info, "script 14 is not the patched version"

    print("[selftest] ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--sites", nargs="*", default=list(DEFAULT_SITES))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.figure:
        run(list(a.sites))
        return
    print("nothing to do: pass --figure or --selftest (see docstring)")


if __name__ == "__main__":
    main()
