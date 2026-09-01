"""
05_figures_l3.py — MOIRAI L3 publication figures
=================================================
WHAT THIS SCRIPT DOES
---------------------
Produces the paper figures for MOIRAI L3, all to the PDF/ folder, all with:
WHITE background, ENGLISH labels only, and legends placed in the MARGIN
(outside the axes, never overlapping the data). Project figure standard.

  PDF/05_a_probability_example.pdf
      One earthquake test event: Z-component waterfall (station x time) beside
      the predicted P and S probability maps, with ground-truth picks marked.
      (Needs a trained checkpoint + waveforms.hdf5; skipped if unavailable.)
  PDF/05_b_training_curves.pdf
      Training loss and dev F1 (P / S / mean) vs epoch, from
      logs/03_train_history.json.
  PDF/05_c_metrics_summary.pdf
      (1) precision / recall / F1 bars for P and S at the primary tolerance,
      (2) F1 vs tolerance, (3) absolute-error distribution with median marked,
      from logs/04_test_results_<site>_<ckpt>.json.

Each figure is generated independently; missing inputs -> that figure is skipped
with a warning, so partial runs still yield what they can.

RUN
  python 05_figures_l3.py                 # all available figures
  python 05_figures_l3.py --selftest      # render b & c from synthetic logs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# white background everywhere
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "axes.grid": False,
})


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
DATA, EVALU, TRAIN = cfg.DATA, cfg.EVALU, cfg.TRAIN
C = {"P": "#1f6feb", "S": "#d1495b", "mean": "#2a9d8f", "loss": "#404040"}


def _save(fig, name):
    out = cfg.PDF_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] saved {out}")


# ----------------------------------------------------------------------------
# (A) probability-map example
# ----------------------------------------------------------------------------
def figure_example(waves, labels, preds, event_tag="test event"):
    """waves (3,S,T), labels (3,S,T), preds (3,S,T)."""
    S, T = waves.shape[1], waves.shape[2]
    t_ms = np.arange(T) / DATA.FS * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")
    titles = ["Z-component waveform", "Predicted P probability",
              "Predicted S probability"]
    panels = [waves[2], preds[0], preds[1]]
    cmaps = ["Greys", "Blues", "Reds"]
    for ax, dat, ttl, cm in zip(axes, panels, titles, cmaps):
        extent = [t_ms[0], t_ms[-1], S - 0.5, -0.5]
        im = ax.imshow(dat, aspect="auto", cmap=cm, extent=extent,
                       vmin=0 if cm != "Greys" else None,
                       vmax=1 if cm != "Greys" else None)
        ax.set_xlabel("Time (ms)")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Station (depth order)")
    # ground-truth picks as markers on the prediction panels
    for ax, ch, col in ((axes[1], 0, C["P"]), (axes[2], 1, C["S"])):
        for st in range(S):
            row = labels[ch, st]
            if row.max() >= 0.5:
                ax.plot(row.argmax() / DATA.FS * 1e3, st, marker="o",
                        mfc="none", mec="black", ms=7, mew=1.2,
                        label="ground-truth pick" if st == 0 else None)
    fig.suptitle(f"MOIRAI L3 array P/S picking — {event_tag} "
                 f"({S} stations, mseel_3h)", y=1.02, fontsize=12)
    # legend in the right margin, outside the axes
    handles, lbls = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False)
    _save(fig, "05_a_probability_example.pdf")


# ----------------------------------------------------------------------------
# (B) training curves
# ----------------------------------------------------------------------------
def figure_training(history):
    ep = [h["epoch"] for h in history]
    loss = [h["train_loss"] for h in history]
    f1p = [h["dev"]["P"]["f1"] for h in history]
    f1s = [h["dev"]["S"]["f1"] for h in history]
    f1m = [h["dev"]["f1_mean"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    a1.plot(ep, loss, color=C["loss"], lw=1.6, label="train loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Training loss")
    a2.plot(ep, f1p, color=C["P"], lw=1.6, label="dev F1 (P)")
    a2.plot(ep, f1s, color=C["S"], lw=1.6, label="dev F1 (S)")
    a2.plot(ep, f1m, color=C["mean"], lw=2.0, ls="--", label="dev F1 (mean)")
    a2.set_xlabel("Epoch"); a2.set_ylabel("F1"); a2.set_ylim(0, 1.02)
    a2.set_title(f"Dev picking F1 (+/-{EVALU.PRIMARY_TOLERANCE_S*1e3:.0f} ms)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.suptitle("MOIRAI L3 — training history (mseel_3h)", y=1.02, fontsize=12)
    _save(fig, "05_b_training_curves.pdf")


# ----------------------------------------------------------------------------
# (C) metric summary
# ----------------------------------------------------------------------------
def figure_metrics(res):
    tols = res["tolerances_s"]
    prim = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")

    # (1) precision/recall/F1 bars at primary tolerance
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics)); w = 0.36
    pv = [res["by_phase"]["P"][prim][m] for m in metrics]
    sv = [res["by_phase"]["S"][prim][m] for m in metrics]
    a1.bar(x - w/2, pv, w, color=C["P"], label="P")
    a1.bar(x + w/2, sv, w, color=C["S"], label="S")
    a1.set_xticks(x); a1.set_xticklabels([m.capitalize() for m in metrics])
    a1.set_ylim(0, 1.05); a1.set_ylabel("Score")
    a1.set_title(f"At +/-{EVALU.PRIMARY_TOLERANCE_S*1e3:.0f} ms")

    # (2) F1 vs tolerance
    f1p = [res["by_phase"]["P"][f"tol_{t}s"]["f1"] for t in tols]
    f1s = [res["by_phase"]["S"][f"tol_{t}s"]["f1"] for t in tols]
    tms = [t * 1e3 for t in tols]
    a2.plot(tms, f1p, "o-", color=C["P"], label="P")
    a2.plot(tms, f1s, "s-", color=C["S"], label="S")
    a2.set_xlabel("Tolerance (ms)"); a2.set_ylabel("F1"); a2.set_ylim(0, 1.05)
    a2.set_title("F1 vs tolerance")

    # (3) absolute-error medians (bar) with values annotated
    med_p = res["by_phase"]["P"][prim]["median_ae_s"] * 1e3
    med_s = res["by_phase"]["S"][prim]["median_ae_s"] * 1e3
    mae_p = res["by_phase"]["P"][prim]["mae_s"] * 1e3
    mae_s = res["by_phase"]["S"][prim]["mae_s"] * 1e3
    xb = np.arange(2)
    a3.bar(xb - w/2, [med_p, med_s], w, color=[C["P"], C["S"]], alpha=0.9,
           label="median AE")
    a3.bar(xb + w/2, [mae_p, mae_s], w, color=[C["P"], C["S"]], alpha=0.45,
           label="mean AE")
    a3.set_xticks(xb); a3.set_xticklabels(["P", "S"])
    a3.set_ylabel("Absolute error (ms)"); a3.set_title("Pick error (matched)")

    for ax in (a1, a2, a3):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                  frameon=False)
    fig.suptitle(f"MOIRAI L3 — test metrics ({res.get('site','mseel_3h')}, "
                 f"eq {res.get('n_earthquake_events','?')} / "
                 f"noise {res.get('n_noise_events','?')}; "
                 f"noise false-alarm {res.get('noise_false_alarm_per_event',0):.3f}/ev)",
                 y=1.04, fontsize=11)
    _save(fig, "05_c_metrics_summary.pdf")


# ----------------------------------------------------------------------------
# (D) LOSO matrix heatmap
# ----------------------------------------------------------------------------
def loso_rows(ckpt="ema"):
    """Collect per-held-out-site metrics from logs/04_test_results_*_loso_*.json."""
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    rows = []
    for f in sorted(cfg.LOGS_DIR.glob("04_test_results_*_loso_*.json")):
        if any(t in f.name for t in ("_pertrace_", "_shuf_", "_seed")):
            continue                       # ablations/controls/seeds handled separately
        d = json.load(open(f))
        if d.get("ckpt") != ckpt:
            continue
        P, S = d["by_phase"]["P"][pt], d["by_phase"]["S"][pt]
        rows.append(dict(site=d.get("heldout"), P=P["f1"], S=S["f1"],
                         F1=d.get("f1_mean_primary", 0.0),
                         FA=d.get("noise_false_alarm_per_event", 0.0),
                         n=d.get("n_events", 0)))
    return rows


def figure_loso_matrix(rows, ckpt="ema"):
    rows = sorted(rows, key=lambda r: r["F1"], reverse=True)  # best on top
    sites = [f"{r['site']} (n={r['n']})" for r in rows]
    M = np.array([[r["P"], r["S"], r["F1"]] for r in rows])
    FA = np.array([r["FA"] for r in rows])
    n = len(rows)
    fig = plt.figure(figsize=(11, max(2.6, 0.55 * n + 1.8)), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.45)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    im = a1.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    a1.set_xticks(range(3)); a1.set_xticklabels(["P F1", "S F1", "F1 mean"])
    a1.set_yticks(range(n)); a1.set_yticklabels(sites)
    for i in range(n):
        for j in range(3):
            a1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black" if M[i, j] > 0.45 else "white", fontsize=9)
    a1.set_title(f"LOSO held-out F1 (+/-{EVALU.PRIMARY_TOLERANCE_S*1e3:.0f} ms, {ckpt})")
    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04); cb.set_label("F1")

    a2.barh(range(n), FA, color=C["S"])
    a2.set_yticks(range(n)); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel("Noise false-alarm picks / event")
    a2.set_title("Noise false alarms")
    for i, v in enumerate(FA):
        a2.text(v, i, f" {v:.2f}", va="center", ha="left", fontsize=8)
    a2.set_xlim(0, max(0.1, FA.max() * 1.25))

    med = float(np.median([r["F1"] for r in rows])) if rows else 0.0
    nge = sum(r["F1"] >= 0.85 for r in rows)
    fig.suptitle("MOIRAI L3 \u2014 leave-one-site-out matrix (AMBER, full held-out site)\n"
                 f"median F1-mean {med:.3f}; {nge}/{n} sites >= 0.85", y=1.05, fontsize=11)
    _save(fig, f"05_d_loso_matrix_{ckpt}.pdf")


# ----------------------------------------------------------------------------
# (E) forge_19 P-collapse diagnosis: P is a DETECTION failure (sub-threshold
#     probability), not a timing error -- while S transfers normally.
# ----------------------------------------------------------------------------
def _loso_ckpt(site, ckpt):
    base = {"best": TRAIN.CKPT_BEST, "ema": TRAIN.CKPT_EMA,
            "last": TRAIN.CKPT_LAST}[ckpt]
    return cfg.MODEL_DIR / base.replace(".pt", f"_loso_{site}.pt")


def _collect_confidence(site, ckpt, max_ev=150):
    """For a held-out site, run its LOSO model on the full site and return, for
    every (event, station) that HAS a ground-truth pick, the max-over-time
    probability of that phase (= the value compared to PEAK_PROB_THRESHOLD when
    declaring a pick). Also return one earthquake example (waves, labels, preds)."""
    import torch
    model_mod = _load_module("02_picker_model_l3.py")
    loader_mod = _load_module("01_amber_setup.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(_loso_ckpt(site, ckpt), map_location=device)["model_state"])
    model.eval()
    ds = loader_mod.build_amber_dataset("test", loader_mod.prepare_site_csv(site, all_test=True))
    confP, confS, example = [], [], None
    for i in range(min(max_ev, len(ds))):
        waves, labels = ds[i]
        lab = labels.numpy()
        if lab[:2].max() < 0.5:
            continue
        with torch.no_grad():
            pred = model_mod.MoiraiPickerL3.activate(
                model(waves.unsqueeze(0).to(device)))[0].cpu().numpy()
        for st in range(lab.shape[1]):
            if lab[0, st].max() >= 0.5:
                confP.append(float(pred[0, st].max()))
            if lab[1, st].max() >= 0.5:
                confS.append(float(pred[1, st].max()))
        if example is None and lab[0].max() >= 0.5 and lab[1].max() >= 0.5:
            example = (waves.numpy(), lab, pred, i)
    return np.array(confP), np.array(confS), example


def _draw_forge_diagnosis(example, confs, site, ref_site):
    waves, labels, preds, idx = example
    S, T = waves.shape[1], waves.shape[2]
    t_ms = np.arange(T) / DATA.FS * 1e3
    thr = EVALU.PEAK_PROB_THRESHOLD
    fig = plt.figure(figsize=(13, 8.4), facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.5, wspace=1.1)
    axw = fig.add_subplot(gs[0, 0:2])
    axp = fig.add_subplot(gs[0, 2:4])
    axs = fig.add_subplot(gs[0, 4:6])
    # row 1: example probability maps
    for ax, dat, ttl, cm in ((axw, waves[2], "Z-component waveform", "Greys"),
                             (axp, preds[0], "Predicted P probability", "Blues"),
                             (axs, preds[1], "Predicted S probability", "Reds")):
        extent = [t_ms[0], t_ms[-1], S - 0.5, -0.5]
        im = ax.imshow(dat, aspect="auto", cmap=cm, extent=extent,
                       vmin=0 if cm != "Greys" else None,
                       vmax=1 if cm != "Greys" else None)
        ax.set_xlabel("Time (ms)"); ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axw.set_ylabel("Station (depth order)")
    for ax, ch in ((axp, 0), (axs, 1)):
        for st in range(S):
            r = labels[ch, st]
            if r.max() >= 0.5:
                ax.plot(r.argmax() / DATA.FS * 1e3, st, marker="o", mfc="none",
                        mec="black", ms=7, mew=1.2,
                        label="ground-truth pick" if st == 0 else None)
    # row 2: confidence histograms (where a true pick exists), threshold line
    axhp = fig.add_subplot(gs[1, 0:3])
    axhs = fig.add_subplot(gs[1, 3:6])
    bins = np.linspace(0, 1, 26)
    for ax, key, ttl, col in ((axhp, "P", "Max P-probability where a true P exists", C["P"]),
                              (axhs, "S", "Max S-probability where a true S exists", C["S"])):
        ax.hist(confs[site][key], bins=bins, density=True, color=col, alpha=0.6,
                label=f"{site} (held out)")
        if confs.get(ref_site) is not None:
            ax.hist(confs[ref_site][key], bins=bins, density=True, color="#777777",
                    alpha=0.45, label=f"{ref_site} (held out)")
        ax.axvline(thr, color="black", ls="--", lw=1.3,
                   label=f"pick threshold = {thr:.2f}")
        ax.set_xlabel(f"max {key} probability per station"); ax.set_ylabel("density")
        ax.set_title(ttl, fontsize=10); ax.set_xlim(0, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8)
    fig.suptitle(f"MOIRAI L3 \u2014 {site} P-phase collapse is a detection failure "
                 f"(sub-threshold P probability), S transfers normally",
                 y=1.02, fontsize=12)
    _save(fig, f"05_e_{site}_diagnosis.pdf")


def figure_forge_diagnosis(site="forge_19", ref_site="mseel_5h", ckpt="ema"):
    if not _loso_ckpt(site, ckpt).exists() or not Path(cfg.AMBER_H5).exists():
        print(f"[fig] skip 05_e (need {_loso_ckpt(site, ckpt).name} + waveforms.hdf5).")
        return
    cP, cS, ex = _collect_confidence(site, ckpt)
    if ex is None:
        print(f"[fig] skip 05_e (no earthquake example for {site}).")
        return
    confs = {site: {"P": cP, "S": cS}, ref_site: None}
    if _loso_ckpt(ref_site, ckpt).exists():
        rP, rS, _ = _collect_confidence(ref_site, ckpt)
        confs[ref_site] = {"P": rP, "S": rS}
    _draw_forge_diagnosis(ex, confs, site, ref_site)


# ----------------------------------------------------------------------------
# (F) array-level vs per-trace baseline comparison
# ----------------------------------------------------------------------------
def baseline_rows(model="phasenet"):
    """Read logs/06_baseline_<model>_<site>.json -> {site: (P_f1, S_f1)}."""
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob(f"06_baseline_{model}_*.json")):
        d = json.load(open(f))
        out[d.get("site")] = (d["by_phase"]["P"][pt]["f1"], d["by_phase"]["S"][pt]["f1"])
    return out


def pertrace_rows(ckpt="ema"):
    """Read the per-trace ABLATION LOSO results
    (04_test_results_<site>_<ckpt>_pertrace_loso_<site>.json) -> {site:(P,S)}."""
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob("04_test_results_*_pertrace_loso_*.json")):
        if "_seed" in f.name:
            continue
        d = json.load(open(f))
        if d.get("ckpt") != ckpt:
            continue
        site = d.get("heldout") or d.get("site")
        out[site] = (d["by_phase"]["P"][pt]["f1"], d["by_phase"]["S"][pt]["f1"])
    return out


def _draw_baseline(sites, arr, base, base_label, array_ckpt, outname):
    n = len(sites); x = np.arange(n); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(8, 1.15 * n + 3), 4.6),
                                 facecolor="white")
    for ax, key, ttl in ((a1, "P", "P-phase F1"), (a2, "S", "S-phase F1")):
        av = [arr[s][key] for s in sites]
        bv = [base[s][0 if key == "P" else 1] for s in sites]
        ax.bar(x - w / 2, av, w, color=C[key], label=f"array-level (MOIRAI L3, {array_ckpt})")
        ax.bar(x + w / 2, bv, w, color="#777777", label=base_label)
        ax.set_xticks(x); ax.set_xticklabels(sites, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
        ax.set_title(f"{ttl} (+/-{EVALU.PRIMARY_TOLERANCE_S*1e3:.0f} ms)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=8)
    fig.suptitle("MOIRAI L3 \u2014 array-level vs per-trace picking on AMBER held-out sites",
                 y=1.05, fontsize=12)
    _save(fig, outname)


def figure_baseline_comparison(model="phasenet", array_ckpt="ema"):
    arr = {r["site"]: r for r in loso_rows(array_ckpt)}
    base = baseline_rows(model)
    sites = [s for s in arr if s in base]
    if not sites:
        print(f"[fig] skip 05_f ({model}: need both array LOSO and 06 baseline jsons).")
        return
    sites = sorted(sites, key=lambda s: arr[s]["P"], reverse=True)
    _draw_baseline(sites, arr, base, f"per-trace ({model}, off-the-shelf)",
                   array_ckpt, f"05_f_baseline_{model}.pdf")


def figure_pertrace_comparison(array_ckpt="ema"):
    """The clean ablation figure: same model, same data, array vs single-station."""
    arr = {r["site"]: r for r in loso_rows(array_ckpt)}
    pts = pertrace_rows(array_ckpt)
    sites = [s for s in arr if s in pts]
    if not sites:
        print("[fig] skip 05_g (need array LOSO + per-trace ablation jsons).")
        return
    sites = sorted(sites, key=lambda s: arr[s]["P"], reverse=True)
    _draw_baseline(sites, arr, pts, "per-trace (same model, 1 station)",
                   array_ckpt, "05_g_pertrace_ablation.pdf")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# GJI revision extras: seed error bars (05_j), residuals (05_h), Delta-vs-moveout (05_i)
# ----------------------------------------------------------------------------
def shuf_rows(ckpt="ema"):
    """Station-shuffle control -> {site:(P,S,mean)}."""
    pt = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob("04_test_results_*_shuf_loso_*.json")):
        if "_seed" in f.name:
            continue
        d = json.load(open(f))
        if d.get("ckpt") != ckpt:
            continue
        site = d.get("heldout") or d.get("site")
        out[site] = (d["by_phase"]["P"][pt]["f1"], d["by_phase"]["S"][pt]["f1"],
                     d.get("f1_mean_primary", 0.0))
    return out


def seed_rows(method="array", ckpt="ema"):
    """Multi-seed runs -> {site: np.array of f1_mean across seeds} for one method."""
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob("04_test_results_*_seed*.json")):
        d = json.load(open(f))
        if d.get("ckpt") != ckpt or d.get("method") != method:
            continue
        site = d.get("heldout") or d.get("site")
        out.setdefault(site, []).append(d.get("f1_mean_primary", 0.0))
    return {s: np.asarray(v, float) for s, v in out.items()}


def moveout_rows():
    """Read 08_moveout_<site>.json -> {site: (P_median_ms, S_median_ms)}."""
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob("08_moveout_*.json")):
        d = json.load(open(f))
        out[d["site"]] = (d["P"]["median_ms"], d["S"]["median_ms"])
    return out


def _draw_residuals(resid):
    colors = {"array": C["P"], "per-trace": "#777777"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2), facecolor="white")
    for ax, ph in ((a1, "P"), (a2, "S")):
        for label, d in resid.items():
            x = d.get(ph)
            if x is None or len(x) == 0:
                continue
            ax.hist(x, bins=41, range=(-20, 20), histtype="step", density=True,
                    linewidth=1.7, color=colors.get(label, "#333"),
                    label=f"{label} (n={len(x)})")
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("pick residual (ms)"); ax.set_ylabel("density")
        ax.set_title(f"{ph}-phase timing residuals")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=8)
    fig.suptitle("MOIRAI L3 \u2014 matched-pick timing residuals", y=1.04, fontsize=12)
    _save(fig, "05_h_residuals.pdf")


def _load_resid(kind, ckpt):
    agg = {"P": [], "S": []}
    for f in sorted(cfg.LOGS_DIR.glob(f"07_resid_*_{ckpt}_*loso_*.npz")):
        nm = f.name
        if "_seed" in nm:
            continue
        is_pt, is_shuf = "_pertrace_" in nm, "_shuf_" in nm
        if kind == "array" and (is_pt or is_shuf):
            continue
        if kind == "per-trace" and not is_pt:
            continue
        z = np.load(f); agg["P"].append(z["P"]); agg["S"].append(z["S"])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in agg.items()}


def figure_residuals(ckpt="ema"):
    resid = {"array": _load_resid("array", ckpt), "per-trace": _load_resid("per-trace", ckpt)}
    if all(len(resid[m]["P"]) == 0 for m in resid):
        print("[fig] skip 05_h (no 07_resid_*.npz yet).")
        return
    _draw_residuals(resid)


def _draw_delta_moveout(points):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor="white")
    ax.axhspan(-0.02, 0.02, color="#EFEFEF", zorder=0)          # tie band
    ax.axhline(0, color="#999", lw=0.9, zorder=1)

    def col(d):
        return C["P"] if d > 0.02 else ("#C0392B" if d < -0.02 else "#8A8A8A")
    for s, x, y in points:
        ax.scatter(x, y, s=78, color=col(y), zorder=3, edgecolor="white", linewidth=0.6)
    # manual label offsets to de-conflict the ~25-35 ms cluster (points: (dx,dy))
    off = {"clearfield_mw6": (5, -15), "mseel_5h": (5, 7), "clearfield_mw4": (6, 6),
           "mseel_3h": (6, -13), "pnr-2": (7, 4), "pnr-1": (7, 4),
           "aneth": (9, 3), "forge_19": (-7, 11)}
    for s, x, y in points:
        dx, dy = off.get(s, (6, 5))
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color="#333", ha=("right" if dx < 0 else "left"))
    ax.set_xlabel("P across-station moveout, site median (ms)")
    ax.set_ylabel("\u0394 F1-mean  (array \u2212 per-trace)")
    ax.set_title("Array advantage peaks at intermediate, in-distribution moveout")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C["P"],
                      markersize=8, label="array better"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B",
                      markersize=8, label="per-trace better"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#8A8A8A",
                      markersize=8, label="tie (|\u0394|\u22640.02)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              frameon=False, ncol=3, fontsize=8)
    fig.subplots_adjust(bottom=0.24)
    _save(fig, "05_i_delta_vs_moveout.pdf")


def figure_delta_vs_moveout(ckpt="ema"):
    arr = {r["site"]: r for r in loso_rows(ckpt)}
    pts = pertrace_rows(ckpt); mv = moveout_rows()
    points = []
    for s in arr:
        if s in pts and s in mv and not np.isnan(mv[s][0]):
            delta = arr[s]["F1"] - 0.5 * (pts[s][0] + pts[s][1])
            points.append((s, mv[s][0], delta))
    if len(points) < 2:
        print("[fig] skip 05_i (need loso + pertrace + 08 moveout jsons).")
        return
    _draw_delta_moveout(points)


def _draw_seed(sites, arr_stats, pt_stats):
    n = len(sites); x = np.arange(n); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * n + 2), 4.7), facecolor="white")
    am = [arr_stats[s][0] for s in sites]; ae = [arr_stats[s][1] for s in sites]
    pm = [pt_stats[s][0] for s in sites]; pe = [pt_stats[s][1] for s in sites]
    ax.bar(x - w / 2, am, w, yerr=ae, capsize=3, color=C["P"], label="array-level")
    ax.bar(x + w / 2, pm, w, yerr=pe, capsize=3, color="#777777", label="per-trace")
    ax.set_xticks(x); ax.set_xticklabels(sites, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-mean")
    ax.set_title("Array vs per-trace across training seeds (mean \u00b1 std)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=9, ncol=2)
    _save(fig, "05_j_seed_comparison.pdf")


def figure_seed_comparison(ckpt="ema"):
    a = seed_rows("array", ckpt); p = seed_rows("pertrace", ckpt)
    sites = [s for s in a if s in p]
    if not sites:
        print("[fig] skip 05_j (need multi-seed runs of both methods).")
        return
    sites = sorted(sites, key=lambda s: a[s].mean(), reverse=True)
    arr_stats = {s: (float(a[s].mean()), float(a[s].std())) for s in sites}
    pt_stats = {s: (float(p[s].mean()), float(p[s].std())) for s in sites}
    _draw_seed(sites, arr_stats, pt_stats)



def _draw_shuffle(sites, arr, shuf, pt):
    n=len(sites); x=np.arange(n); w=0.26
    fig,ax=plt.subplots(figsize=(max(8,1.4*n+3),4.9),facecolor="white")
    ax.bar(x-w,arr, w,color=C["P"],   label="array (moveout + coupling)")
    ax.bar(x,  shuf,w,color="#E07A28",label="shuffle (coupling only)")
    ax.bar(x+w,pt,  w,color="#8A8A8A",label="per-trace (neither)")
    ax.set_xticks(x); ax.set_xticklabels(sites,rotation=15,ha="right",fontsize=9)
    ax.set_ylim(0,1.0); ax.set_ylabel("F1-mean (held-out site)")
    ax.set_title("Station-shuffle control: coupling vs ordered moveout")
    ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.30),frameon=False,ncol=3,fontsize=8.5)
    fig.subplots_adjust(bottom=0.26)
    _save(fig,"05_k_shuffle_control.pdf")


def figure_shuffle_control(ckpt="ema"):
    sh=shuf_rows(ckpt)
    ar={r["site"]:r["F1"] for r in loso_rows(ckpt)}
    pt=pertrace_rows(ckpt)
    sites=[s for s in sh if s in ar and s in pt]
    if not sites:
        print("[fig] skip 05_k (need array + shuffle + per-trace results).")
        return
    order=["forge_19","clearfield_mw4","pnr-2","aneth"]
    sites=sorted(sites,key=lambda s:(order.index(s) if s in order else 99))
    arr=[ar[s] for s in sites]; shf=[sh[s][2] for s in sites]
    ptm=[0.5*(pt[s][0]+pt[s][1]) for s in sites]
    _draw_shuffle(sites,arr,shf,ptm)


def _try_example():
    """Render Fig A from a real checkpoint + h5 if both exist, else skip."""
    import torch
    ckpt = cfg.MODEL_DIR / TRAIN.CKPT_BEST
    if not ckpt.exists() or not Path(cfg.AMBER_H5).exists():
        print("[fig] skip 05_a (need trained checkpoint + waveforms.hdf5).")
        return
    model_mod = _load_module("02_picker_model_l3.py")
    loader_mod = _load_module("01_amber_setup.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_mod.build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()
    ds = loader_mod.build_amber_dataset("test", loader_mod.prepare_site_csv(DATA.SITE))
    for i in range(min(50, len(ds))):
        waves, labels = ds[i]
        if labels[:2].amax() >= 0.5:  # earthquake event with picks
            with torch.no_grad():
                preds = model_mod.MoiraiPickerL3.activate(
                    model(waves.unsqueeze(0).to(device)))[0].cpu().numpy()
            figure_example(waves.numpy(), labels.numpy(), preds,
                           event_tag=f"test event #{i}")
            return
    print("[fig] skip 05_a (no earthquake event found in first 50 test items).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best", choices=["best", "ema", "last"])
    ap.add_argument("--site", default=DATA.SITE)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    hist_path = cfg.LOGS_DIR / "03_train_history.json"
    if hist_path.exists():
        figure_training(json.load(open(hist_path)))
    else:
        print(f"[fig] skip 05_b (no {hist_path}).")

    res_path = cfg.LOGS_DIR / f"04_test_results_{args.site}_{args.ckpt}.json"
    if res_path.exists():
        figure_metrics(json.load(open(res_path)))
    else:
        print(f"[fig] skip 05_c (no {res_path}).")

    for ck in ("ema", "best"):  # LOSO matrix, if any folds have been evaluated
        rows = loso_rows(ck)
        if rows:
            figure_loso_matrix(rows, ck)
        else:
            print(f"[fig] skip 05_d ({ck}: no LOSO result jsons yet).")

    try:
        _try_example()
    except Exception as e:
        print("[fig] skip 05_a (needs amber pkg + waveforms.hdf5):", e)
    try:
        figure_forge_diagnosis(site="forge_19", ref_site="mseel_5h",
                               ckpt=args.ckpt if args.ckpt in ("ema", "best") else "ema")
    except Exception as e:
        print("[fig] skip 05_e:", e)
    for m in ("phasenet", "eqtransformer"):
        try:
            figure_baseline_comparison(m, "ema")
        except Exception as e:
            print(f"[fig] skip 05_f ({m}):", e)
    try:
        figure_pertrace_comparison("ema")
    except Exception as e:
        print("[fig] skip 05_g:", e)
    for fn, tag in ((figure_seed_comparison, "05_j"), (figure_residuals, "05_h"),
                    (figure_delta_vs_moveout, "05_i"), (figure_shuffle_control, "05_k")):
        try:
            fn("ema")
        except Exception as e:
            print(f"[fig] skip {tag}:", e)
    print("[done] figures in", cfg.PDF_DIR)


# ----------------------------------------------------------------------------
# synthetic self-test (renders b & c to verify layout / white bg / margins)
# ----------------------------------------------------------------------------
def selftest():
    print("=" * 70)
    print("MOIRAI L3 — 05 figures self-test (synthetic logs)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    hist = [{"epoch": e, "train_loss": float(0.3 * np.exp(-e / 20) + 0.02),
             "dev": {"P": {"f1": float(min(0.98, 0.2 + e / 50))},
                     "S": {"f1": float(min(0.93, 0.1 + e / 60))},
                     "f1_mean": float(min(0.95, 0.15 + e / 55))}}
            for e in range(40)]
    figure_training(hist)
    res = {"tolerances_s": list(EVALU.TOLERANCES_S), "site": "mseel_3h",
           "n_earthquake_events": 153, "n_noise_events": 100,
           "noise_false_alarm_per_event": 0.12, "by_phase": {}}
    for name, base in (("P", 0.95), ("S", 0.82)):
        res["by_phase"][name] = {}
        for t in EVALU.TOLERANCES_S:
            res["by_phase"][name][f"tol_{t}s"] = dict(
                precision=base, recall=base - 0.05, f1=base - 0.02,
                fail_rate=0.05, mae_s=0.006, median_ae_s=0.004, n_matched=1500)
    figure_metrics(res)
    # synthetic example panel
    train_mod = _load_module("03_train_l3.py")
    waves, labels = train_mod._make_synthetic_batch(b=1, T=1024)
    figure_example(waves[0].numpy(), labels[0].numpy(), labels[0].numpy(),
                   event_tag="synthetic")
    # synthetic LOSO matrix
    synth = [dict(site=s, P=p, S=q, F1=0.5*(p+q), FA=fa, n=nn) for s, p, q, fa, nn in [
        ("pnr-1", 0.99, 0.98, 0.04, 1258), ("mseel_5h", 0.92, 0.93, 0.0, 1512),
        ("pnr-2", 0.90, 0.90, 1.28, 972), ("clearfield_mw4", 0.90, 0.89, 0.0, 1254),
        ("mseel_3h", 0.85, 0.82, 0.0, 1684), ("forge_19", 0.22, 0.87, 0.03, 1213)]]
    figure_loso_matrix(synth, "ema")
    # synthetic forge diagnosis (P sub-threshold for forge_19, normal elsewhere)
    rng = np.random.default_rng(0)
    confs = {"forge_19": {"P": rng.beta(2, 6, 400), "S": rng.beta(6, 2, 400)},
             "mseel_5h": {"P": rng.beta(6, 2, 400), "S": rng.beta(6, 2, 400)}}
    ex = (waves[0].numpy(), labels[0].numpy(), labels[0].numpy(), 0)
    _draw_forge_diagnosis(ex, confs, "forge_19", "mseel_5h")
    # synthetic baseline comparison (array beats per-trace, esp. P at forge_19)
    sites = ["pnr-1", "mseel_5h", "mseel_3h", "forge_19"]
    arr = {"pnr-1": {"P": 0.99, "S": 0.98}, "mseel_5h": {"P": 0.91, "S": 0.88},
           "mseel_3h": {"P": 0.90, "S": 0.88}, "forge_19": {"P": 0.18, "S": 0.86}}
    base = {"pnr-1": (0.82, 0.70), "mseel_5h": (0.74, 0.61),
            "mseel_3h": (0.70, 0.58), "forge_19": (0.12, 0.55)}
    _draw_baseline(sites, arr, base, "per-trace (phasenet, off-the-shelf)",
                   "ema", "05_f_baseline_phasenet.pdf")
    # per-trace ablation (same model, 1 station): array clearly higher, forge_19 both low
    ptb = {"pnr-1": (0.93, 0.90), "mseel_5h": (0.80, 0.78),
           "mseel_3h": (0.78, 0.75), "forge_19": (0.14, 0.72)}
    _draw_baseline(sites, arr, ptb, "per-trace (same model, 1 station)",
                   "ema", "05_g_pertrace_ablation.pdf")
    # 05_j seed error bars
    seed_sites = ["forge_19", "aneth", "clearfield_mw4", "pnr-2"]
    arr_stats = {"forge_19": (0.52, 0.03), "aneth": (0.89, 0.02),
                 "clearfield_mw4": (0.85, 0.02), "pnr-2": (0.88, 0.02)}
    pt_stats = {"forge_19": (0.84, 0.02), "aneth": (0.94, 0.01),
                "clearfield_mw4": (0.77, 0.03), "pnr-2": (0.84, 0.02)}
    _draw_seed(seed_sites, arr_stats, pt_stats)
    # 05_h residuals
    rng2 = np.random.default_rng(1)
    resid = {"array": {"P": rng2.normal(0, 2, 4000), "S": rng2.normal(0, 3, 4000)},
             "per-trace": {"P": rng2.normal(0.2, 2.2, 4000), "S": rng2.normal(0, 3.1, 4000)}}
    _draw_residuals(resid)
    # 05_i delta vs moveout
    pts_i = [("pnr-1", 6.0, -0.004), ("aneth", 9.0, -0.051), ("mseel_5h", 5.0, 0.010),
             ("pnr-2", 4.0, 0.041), ("clearfield_mw4", 3.5, 0.078), ("forge_19", 12.0, -0.324)]
    _draw_delta_moveout(pts_i)
    _draw_shuffle(["forge_19", "clearfield_mw4", "pnr-2", "aneth"],
                  [0.518, 0.852, 0.881, 0.889], [0.570, 0.866, 0.827, 0.930],
                  [0.842, 0.768, 0.841, 0.941])
    print("  rendered 05_b..05_k (synthetic): PASS")


if __name__ == "__main__":
    main()
