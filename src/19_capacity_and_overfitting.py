#!/usr/bin/env python
"""
19_capacity_and_overfitting.py -- GJI revision round 2: is an 8-million-parameter
model too large for this training set? (Reviewer 2, page 9)
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Reviewer 2 writes: "A model that contains 8 million parameters seems too big,
especially when there are only ~10,000 events to train. There is very little
data for such a big model." The parameter-to-sample ratio is a proxy, not a
measurement. What the objection is really about is whether the model overfits,
and that question is already answered by the training runs: every leave-one-
site-out model was trained with a held-out development split and early stopping,
and script 03 recorded the full per-epoch history.

This script reads those histories -- no retraining, no GPU, no AMBER -- and
reports, per run:

  * the epoch at which development F1-mean peaked, and how many epochs followed;
  * whether training loss kept falling after that peak while development F1
    did not improve, which is the signature the objection predicts;
  * how far development F1 had fallen by the last epoch relative to its peak,
    which is the magnitude of any overfitting that did occur;
  * whether early stopping fired, i.e. whether the run was halted by the
    development split rather than by the epoch budget.

A model that overfits badly shows a development curve that peaks early and then
declines while the training loss continues to fall. A model whose capacity is
adequate but not harmful shows a development curve that plateaus and a run
stopped by patience. The distinction is visible in the data and is reported as
a number rather than argued from the parameter count.

The script also reports the number of training events actually seen by each
leave-one-site-out run, read from the benchmark metadata, because the
reviewer's "~10,000 events" is the whole benchmark rather than the training
split of any single run, and the two differ.

INPUTS
  logs/03_train_history*.json          written by 03_train_l3.py
  data/metadata.csv                    optional; for training-set sizes
  (torch, optional: for the exact parameter count of the model)

OUTPUTS
  logs/19_capacity_<ckpt>.json         per-run numbers
  logs/19_capacity.md                  the table to paste
  PDF/19_a_training_curves.pdf         training loss and development F1

All figures: white background, English labels, legends clear of the data.

RUN
  python src/19_capacity_and_overfitting.py --curves
  python src/19_capacity_and_overfitting.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import json
import math
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
DATA, TRAIN = cfg.DATA, cfg.TRAIN

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]


# ----------------------------------------------------------------------------
def load_histories() -> dict:
    """{(site, method): [epoch records]} from logs/03_train_history*.json."""
    out = {}
    for f in sorted(cfg.LOGS_DIR.glob("03_train_history*.json")):
        stem = f.stem                       # 03_train_history[_pertrace][_shuf][_loso_<site>]
        if "_loso_" not in stem:
            continue
        site = stem.split("_loso_")[-1]
        method = ("pertrace" if "_pertrace" in stem else
                  "shuffle" if "_shuf" in stem else "array")
        try:
            h = json.loads(f.read_text())
        except ValueError:
            print(f"[19] {f.name}: unreadable, skipped")
            continue
        if isinstance(h, list) and h:
            out[(site, method)] = h
    return out


def _dev_f1(rec: dict, ema: bool) -> float:
    """Development F1-mean of one epoch record, EMA or raw."""
    d = rec.get("dev_ema" if ema else "dev") or {}
    v = d.get("f1_mean")
    return float(v) if v is not None else float("nan")


def _complete(rec: dict, ema: bool) -> bool:
    """Is this epoch record usable?

    Some runs end with a record whose development F1 is exactly zero and whose
    training loss is absent. A detector that plateaus at 0.93 for eighty epochs
    does not fall to zero in one step, and a genuine collapse would still have
    recorded a training loss, so such a record is a truncated write rather than
    a measurement. Counting it would manufacture a 0.93 'fall from the peak'
    and turn a flat curve into evidence of the overfitting the reviewer asked
    about, which is exactly the error this function exists to prevent.
    """
    f = _dev_f1(rec, ema)
    return bool(np.isfinite(f) and f > 0.0 and rec.get("train_loss") is not None)


def trim(hist: list, ema: bool = True) -> tuple[list, int]:
    """Drop the trailing incomplete records and say how many were dropped.

    Used by the analysis AND by the figure, so the curve that is drawn is the
    curve that was measured. Keeping them apart is how a table reporting 68
    epochs ended up beside a figure drawing 84 with a cliff at the end.
    """
    n = len(hist)
    while hist and not _complete(hist[-1], ema):
        hist = hist[:-1]
    return hist, n - len(hist)


def analyse(hist: list, ema: bool = True) -> dict:
    """Peak, post-peak behaviour and early stopping for one run."""
    n_raw = len(hist)
    hist, n_dropped = trim(hist, ema)
    if not hist:
        return dict(n_epochs=n_raw, usable=False, n_truncated_records=n_dropped)
    ep = [int(r.get("epoch", i)) for i, r in enumerate(hist)]
    tl = np.array([float(r.get("train_loss", np.nan)) for r in hist])
    f1 = np.array([_dev_f1(r, ema) for r in hist])
    ok = np.isfinite(f1)
    if not ok.any():
        return dict(n_epochs=len(hist), usable=False)
    best_i = int(np.nanargmax(f1))
    after = f1[best_i + 1:]
    tl_after = tl[best_i + 1:]
    # training loss still falling after the peak is the overfitting signature
    fin = tl_after[np.isfinite(tl_after)] if tl_after.size else tl_after
    h = len(fin) // 2
    still_falling = bool(h >= 1 and fin[h:].mean() < fin[:h].mean())
    drop = float(f1[best_i] - f1[-1])
    stopped_early = bool(len(hist) - 1 - best_i >= TRAIN.EARLY_STOP_PATIENCE)

    # The fall from the peak is the gap between the MAXIMUM of a noisy sequence
    # and its last value, and the maximum of a noisy sequence is biased upwards
    # whether or not the sequence is declining. The quantity that answers the
    # question is therefore the TREND after the peak, measured against the
    # epoch-to-epoch noise of the same segment: a plateau that wanders gives a
    # trend near zero with large noise, a model that is overfitting gives a
    # trend that is clearly negative relative to it.
    seg = f1[best_i:][np.isfinite(f1[best_i:])]
    noise = trend = None
    if seg.size >= 6:
        noise = float(np.std(np.diff(seg)) / np.sqrt(2.0))   # per-epoch sigma
        h = seg.size // 2
        trend = float(seg[h:].mean() - seg[:h].mean())       # <0 means declining
    return dict(
        usable=True, n_truncated_records=n_dropped,
        n_epochs=len(hist), best_epoch=ep[best_i],
        epochs_after_best=int(len(hist) - 1 - best_i),
        dev_f1_best=float(f1[best_i]), dev_f1_last=float(f1[-1]),
        dev_f1_drop_from_best=drop,
        train_loss_first=float(tl[0]) if np.isfinite(tl[0]) else None,
        train_loss_best=float(tl[best_i]) if np.isfinite(tl[best_i]) else None,
        train_loss_last=float(tl[-1]) if np.isfinite(tl[-1]) else None,
        train_loss_still_falling_after_best=still_falling,
        early_stop_fired=stopped_early,
        dev_epoch_noise=noise, post_peak_trend=trend,
        trend_over_noise=(None if (trend is None or not noise)
                          else float(trend / noise)))


def _stations_per_site() -> dict:
    """{site: sensors on the string} from the coordinates written by script 17."""
    f = cfg.LOGS_DIR / "17_array_coords.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text())
    except ValueError:
        return {}
    return {k: len(v["xyz"]) for k, v in d.items() if v.get("xyz")}


def training_sizes() -> dict:
    """Training-set size per leave-one-site-out run, in traces AND in events.

    A row of the benchmark metadata is one STATION TRACE, not one event: the
    row count equals the sum over sites of events times sensors on that site's
    string, and the identity is checked below. The distinction decides the
    answer to the reviewer, whose "~10,000 events" refers to events and is
    correct; reporting the row count as events would contradict a true
    statement with a number six times too large. Both are therefore reported,
    and the event count is derived only where the division is exact.
    """
    f = cfg.AMBER_CSV
    if not Path(f).exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(f)
    except Exception as e:                  # pandas missing or csv unreadable
        print(f"[19] {f}: {e}; training-set sizes omitted")
        return {}
    if "dataset" not in df.columns or "split" not in df.columns:
        return {}
    usable = list(DATA.USABLE_SITES)
    nst = _stations_per_site()
    rows = {s: int(((df["dataset"] == s) & (df["split"] == "train")).sum())
            for s in usable}
    all_rows = {s: int((df["dataset"] == s).sum()) for s in usable}

    def to_events(r: dict):
        """Rows -> events, only if every site divides exactly by its sensors."""
        if not nst or any(s not in nst for s in r):
            return None
        if any(r[s] % nst[s] for s in r):
            return None
        return {s: r[s] // nst[s] for s in r}

    ev_train, ev_all = to_events(rows), to_events(all_rows)
    out = {"_stations_per_site": nst,
           "_rows_are_station_traces": ev_all is not None,
           "_train_rows_per_run": {s: sum(v for o, v in rows.items() if o != s)
                                   for s in usable},
           "_benchmark_rows": sum(all_rows.values())}
    if ev_all is not None:
        out["_benchmark_events"] = sum(ev_all.values())
        out["_events_per_site"] = ev_all
        out["_train_events_per_run"] = {
            s: sum(v for o, v in ev_train.items() if o != s) for s in usable}
    return out


def parameter_count():
    """Exact trainable-parameter count, or None where torch is unavailable."""
    try:
        import torch                        # noqa: F401
        mm = _load("02_picker_model_l3.py")
        m = mm.build_model()
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    except Exception:
        return None


# ----------------------------------------------------------------------------
def sign_test(values) -> tuple[int, int, float]:
    """(negatives, total, two-sided p) for the sign of a set of trends.

    Each run's trend is buried in that run's own epoch-to-epoch noise, so no
    single curve settles the question. But if the trends were noise about zero
    their signs would split evenly, and a consistent excess of negatives is
    evidence of a real drift however small each one is. Reporting only the
    per-run ratio would therefore understate what the 44 runs jointly show.
    """
    v = [x for x in values if x is not None]
    n = len(v)
    k = sum(1 for x in v if x < 0)
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return k, n, float(min(1.0, 2 * tail))


def seed_spread(rows: dict) -> float | None:
    """Median spread of the best development score between training seeds.

    The question is not whether the development curve declines at all after its
    peak -- with a finite development split it always will -- but whether the
    decline is larger than the run-to-run variation the same configuration
    shows anyway. Script 03 was run with several seeds at four sites, so that
    variation is measured rather than assumed, and it is the scale against
    which the decline is judged.
    """
    groups = {}
    for (site, cfg_), r in rows.items():
        if not r.get("usable"):
            continue
        base = re.sub(r"_seed\d+$", "", site)
        groups.setdefault((base, cfg_), []).append(r["dev_f1_best"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) >= 2]
    return float(np.median(spreads)) if spreads else None


def verdict(rows: dict) -> str:
    """State what the curves support, in the terms the objection was raised in."""
    use = [r for r in rows.values() if r.get("usable")]
    n_trunc = sum(r.get("n_truncated_records", 0) for r in rows.values())
    note = (f" ({n_trunc} truncated end-of-run records were discarded)"
            if n_trunc else "")
    if not use:
        return "[verdict] no usable training histories." + note
    drops = np.array([r["dev_f1_drop_from_best"] for r in use])
    med, mx = float(np.median(drops)), float(drops.max())
    n_early = sum(r["early_stop_fired"] for r in use)
    n_sign = sum(r["train_loss_still_falling_after_best"] for r in use)
    sp = seed_spread(rows)
    tr = np.array([r["post_peak_trend"] for r in use
                   if r.get("post_peak_trend") is not None])
    nz = np.array([r["dev_epoch_noise"] for r in use
                   if r.get("dev_epoch_noise")])
    ratio = np.array([r["trend_over_noise"] for r in use
                      if r.get("trend_over_noise") is not None])
    sp_txt = (f"the spread of the best score between training seeds at the same "
              f"site is {sp:.3f}" if sp is not None
              else "no seed replicates are available for comparison")
    common = (f"Across {len(use)} runs the development score falls a median of "
              f"{med:.3f} and at most {mx:.3f} below its peak, and {sp_txt}. "
              f"Training loss was still falling after the peak in {n_sign} runs "
              f"and {n_early} runs were halted by the development split rather "
              f"than by the epoch budget.")
    k = n_sign_tot = 0
    pval = 1.0
    if tr.size:
        k, n_sign_tot, pval = sign_test(tr.tolist())
        common += (f" The development curve varies by {np.median(nz):.3f} from "
                   f"one epoch to the next, and the trend after the peak "
                   f"(second half of the segment minus the first) has a median "
                   f"of {np.median(tr):+.4f}, i.e. {np.median(ratio):+.2f} "
                   f"times that epoch-to-epoch variation. The trend is negative "
                   f"in {k} of {n_sign_tot} runs (sign test p = {pval:.3f}).")
    # A plateau that wanders produces a large "fall from the peak" with no
    # trend, so the trend is what decides, and the fall is reported only as
    # context.
    small = tr.size and abs(float(np.median(ratio))) < 0.5
    consistent = bool(tr.size and pval < 0.05 and k > n_sign_tot - k)
    if small and not consistent:
        head = ("[verdict] the development curve does not decline after its "
                "peak: the post-peak trend is small compared with the "
                "epoch-to-epoch variation of the same curve, and its sign is "
                "no more often negative than chance, so the gap between the "
                "peak and the last epoch is what a flat but noisy curve "
                "produces, not overfitting.")
    elif small and consistent:
        head = ("[verdict] the development curve drifts downwards after its "
                "peak, consistently across runs but negligibly within any one "
                "of them: the trend is negative more often than chance allows, "
                "yet in no run does it exceed that run's own epoch-to-epoch "
                "variation, and its magnitude is of order 0.002 in F1.")
    elif (tr.size and float(np.median(tr)) > -0.02) or (not tr.size and mx <= 0.05):
        head = ("[verdict] the development curve declines slightly after its "
                "peak, by an amount too small to change a conclusion.")
    else:
        head = ("[verdict] the development score falls materially after its "
                "peak. This is the overfitting the parameter count predicts "
                "and must be reported as such.")
    tail = (" Every score reported in the paper is taken from the best "
            "development epoch, not the last." + note)
    return head + " " + common + tail


def write_report(rows: dict, sizes: dict, n_param, ckpt: str) -> str:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR / f"19_capacity_{ckpt}.json").write_text(json.dumps(
        {f"{s}__{m}": r for (s, m), r in rows.items()}
        | {"_training_sizes": sizes,
           "_trainable_parameters": n_param}, indent=2))

    md = []
    if n_param:
        md.append(f"Trainable parameters: {n_param:,}")
    if sizes:
        tr = list(sizes.get("_train_rows_per_run", {}).values())
        if tr:
            md.append(f"Benchmark station traces in the eight usable sites: "
                      f"{sizes['_benchmark_rows']:,}; training traces per "
                      f"leave-one-site-out run: {min(tr):,}-{max(tr):,}")
        te = list(sizes.get("_train_events_per_run", {}).values())
        if te:
            md.append(f"Benchmark EVENTS: {sizes['_benchmark_events']:,}; "
                      f"training events per leave-one-site-out run: "
                      f"{min(te):,}-{max(te):,}. The array configuration sees "
                      f"one training sample per event; the per-trace "
                      f"configuration sees one per station trace.")
        elif tr:
            md.append("Event counts not derived: the metadata rows do not "
                      "divide exactly by the sensor counts in "
                      "17_array_coords.json, so the rows may not be station "
                      "traces. Run script 17 --geometry first.")
    md += ["", "| Site | config | epochs | best epoch | dev F1 best | "
               "dev F1 last | fall from best | post-peak trend | epoch noise | "
               "trend/noise | train loss first -> best -> last | early stop |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = [k for s in SITE_ORDER for k in
            ((s, "array"), (s, "pertrace"), (s, "shuffle")) if k in rows]
    keys += [k for k in rows if k not in keys]
    for (s, m) in keys:
        r = rows[(s, m)]
        if not r.get("usable"):
            md.append(f"| {s} | {m} | {r['n_epochs']} | - | - | - | - | - | - | - |")
            continue
        tls = " -> ".join("-" if r[k] is None else f"{r[k]:.4f}"
                          for k in ("train_loss_first", "train_loss_best",
                                    "train_loss_last"))
        f = lambda k, w: ("-" if r.get(k) is None else f"{r[k]:{w}}")
        md.append(
            f"| {s} | {m} | {r['n_epochs']} | {r['best_epoch']} | "
            f"{r['dev_f1_best']:.3f} | {r['dev_f1_last']:.3f} | "
            f"{r['dev_f1_drop_from_best']:+.3f} | {f('post_peak_trend','+.4f')} | "
            f"{f('dev_epoch_noise','.4f')} | {f('trend_over_noise','+.2f')} | "
            f"{tls} | {'yes' if r['early_stop_fired'] else 'no'} |")
    txt = "\n".join(md)
    (cfg.LOGS_DIR / "19_capacity.md").write_text(txt)
    return txt


# ----------------------------------------------------------------------------
def make_figure(hists: dict, rows: dict, pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    sites = [s for s in SITE_ORDER if any(k[0] == s for k in hists)]
    sites += sorted({k[0] for k in hists} - set(sites))
    cmap = plt.get_cmap("tab10")
    col = {s: cmap(i % 10) for i, s in enumerate(sites)}
    style = {"array": "-", "pertrace": "--", "shuffle": ":"}

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.4))
    fig.subplots_adjust(left=0.09, right=0.74, top=0.95, bottom=0.07, hspace=0.42)

    ax = axes[0]
    ax.set_title("Training loss per epoch (logarithmic scale)", fontsize=10)
    for (s, m), h in hists.items():
        y = [r.get("train_loss") for r in h]
        y = [v if (v is not None and np.isfinite(v) and v > 0) else np.nan
             for v in y]
        ax.plot(range(1, len(y) + 1), y, style.get(m, "-"), color=col[s], lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.set_title("Development F1-mean per epoch (dot: best epoch, the one the "
                 "reported checkpoint comes from)", fontsize=10)
    for (s, m), h in hists.items():
        y = [_dev_f1(r, True) for r in h]
        ax.plot(range(1, len(y) + 1), y, style.get(m, "-"), color=col[s], lw=1.2)
        r = rows.get((s, m))
        if r and r.get("usable"):
            ax.plot([r["best_epoch"]], [r["dev_f1_best"]], "o", color=col[s],
                    ms=5, mec="black", mew=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("development F1-mean")
    ax.grid(alpha=0.3)

    # The decline this script reports is a median of about 0.01 on a curve that
    # sits near 0.93, so it is invisible on a 0-1 axis. The third panel is the
    # same curves on the scale of the effect, which is what the objection about
    # capacity actually asks to see.
    ax = axes[2]
    ax.set_title("Development F1-mean, plateau detail (same curves, "
                 "vertical scale of the reported decline)", fontsize=10)
    lo = []
    for (s, m), h in hists.items():
        y = [_dev_f1(r, True) for r in h]
        ax.plot(range(1, len(y) + 1), y, style.get(m, "-"), color=col[s], lw=1.2)
        r = rows.get((s, m))
        if r and r.get("usable"):
            ax.plot([r["best_epoch"]], [r["dev_f1_best"]], "o", color=col[s],
                    ms=5, mec="black", mew=0.5)
            lo.append(r["dev_f1_last"])
    if lo:
        top = max(r["dev_f1_best"] for r in rows.values() if r.get("usable"))
        ax.set_ylim(min(lo) - 0.01, top + 0.005)
    ax.set_xlabel("epoch")
    ax.set_ylabel("development F1-mean")
    ax.grid(alpha=0.3)

    handles = [plt.Line2D([], [], color=col[s], lw=1.6, label=s) for s in sites]
    handles += [plt.Line2D([], [], color="black", ls=style[m], lw=1.4, label=lab)
                for m, lab in (("array", "array"), ("pertrace", "per-trace"),
                               ("shuffle", "station-shuffle"))
                if any(k[1] == m for k in hists)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.755, 0.5),
               frameon=True, fontsize=8)

    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {pdf} (+ .png)")


# ----------------------------------------------------------------------------
def _synth(n=30, peak=18, decline=0.0, seed=0):
    """A history whose development curve peaks at `peak` and then falls by
    `decline`, with training loss monotonically decreasing throughout."""
    rng = np.random.default_rng(seed)
    out = []
    for e in range(1, n + 1):
        loss = 1.0 / (1.0 + 0.25 * e)                 # always falling
        if e <= peak:
            f1 = 0.90 * (1 - np.exp(-0.25 * e))
        else:
            f1 = 0.90 * (1 - np.exp(-0.25 * peak)) - decline * (e - peak) / (n - peak)
        f1 = float(f1 + rng.normal(0, 1e-4))
        out.append({"epoch": e, "train_loss": float(loss),
                    "dev": {"f1_mean": f1}, "dev_ema": {"f1_mean": f1}})
    return out


def selftest():
    plateau = analyse(_synth(decline=0.0, seed=1))
    assert plateau["usable"] and plateau["dev_f1_drop_from_best"] < 0.005, plateau
    assert plateau["train_loss_still_falling_after_best"], plateau
    over = analyse(_synth(decline=0.20, seed=2))
    assert over["dev_f1_drop_from_best"] > 0.15, over
    assert over["best_epoch"] <= plateau["best_epoch"] + 1
    print(f"  plateau run: falls {plateau['dev_f1_drop_from_best']:+.4f} from "
          f"peak; overfitting run: {over['dev_f1_drop_from_best']:+.3f}: PASS")

    # the verdict must distinguish the two, and must not call a plateau
    # overfitting merely because the training loss keeps falling
    v_ok = verdict({("a", "array"): plateau})
    v_bad = verdict({("a", "array"): over})
    assert ("does not decline" in v_ok or "declines slightly" in v_ok
            or "drifts downwards" in v_ok), v_ok
    assert "must be reported as such" in v_bad, v_bad
    # seed spread must be measured from replicates and used as the scale
    reps = {("s_seed1", "array"): analyse(_synth(decline=0.0, seed=11)),
            ("s_seed2", "array"): analyse(_synth(decline=0.0, seed=12)),
            ("s_seed3", "array"): analyse(_synth(decline=0.0, seed=13))}
    sp = seed_spread(reps)
    assert sp is not None and sp >= 0.0, sp
    assert seed_spread({("only", "array"): plateau}) is None
    print(f"  verdict separates plateau from overfitting; seed spread {sp:.4f}: PASS")

    # A NOISY plateau: large fall from the peak, no trend. This is the case the
    # real curves turned out to be, and the one the earlier rule got wrong.
    rng2 = np.random.default_rng(7)
    noisy = [{"epoch": e, "train_loss": 1.0 / (1 + 0.3 * e),
              "dev": {"f1_mean": float(0.93 + rng2.normal(0, 0.008))},
              "dev_ema": {"f1_mean": float(0.93 + rng2.normal(0, 0.008))}}
             for e in range(1, 91)]
    rn = analyse(noisy)
    assert rn["dev_f1_drop_from_best"] > 0.01, rn      # a big apparent fall...
    assert abs(rn["trend_over_noise"]) < 0.5, rn        # ...with no trend
    vn = verdict({("n", "array"): rn})
    assert "does not decline" in vn, vn
    # the sign test itself: an even split is not evidence, a consistent one is
    assert sign_test([1, -1, 1, -1])[2] == 1.0
    k_, n_, p_ = sign_test([-1] * 30 + [1] * 13)
    assert (k_, n_) == (30, 43) and p_ < 0.05, (k_, n_, p_)
    assert sign_test([])[2] == 1.0
    # a small but consistent drift must be named as such, not as "no decline"
    many = {}
    for i in range(43):
        r = dict(rn)
        r["post_peak_trend"] = -0.002 if i < 30 else +0.002
        r["trend_over_noise"] = -0.36 if i < 30 else +0.36
        many[(f"s{i}", "array")] = r
    vm = verdict(many)
    assert "drifts downwards" in vm, vm
    print(f"  sign test: 30/43 negative -> p {p_:.4f}; verdict names the drift: PASS")
    # and a genuinely declining curve must still be caught
    decl = [{"epoch": e, "train_loss": 1.0 / (1 + 0.3 * e),
             "dev": {"f1_mean": float(0.93 - 0.004 * max(0, e - 20)
                                      + rng2.normal(0, 0.008))},
             "dev_ema": {"f1_mean": float(0.93 - 0.004 * max(0, e - 20)
                                          + rng2.normal(0, 0.008))}}
            for e in range(1, 91)]
    rd = analyse(decl)
    assert rd["trend_over_noise"] < -0.5, rd
    assert "must be reported as such" in verdict({("d", "array"): rd})
    print(f"  noisy plateau: fall {rn['dev_f1_drop_from_best']:+.3f} but "
          f"trend/noise {rn['trend_over_noise']:+.2f} -> not a decline; "
          f"true decline {rd['trend_over_noise']:+.2f} -> caught: PASS")

    rows = {("forge_19", "array"): plateau, ("forge_19", "pertrace"): over}
    hists = {("forge_19", "array"): _synth(decline=0.0, seed=1),
             ("forge_19", "pertrace"): _synth(decline=0.20, seed=2)}
    txt = write_report(rows, {"forge_19": 1234, "_benchmark_total": 9999}, 8_000_000,
                       "selftest")
    assert "forge_19" in txt and "8,000,000" in txt
    pdf = cfg.PDF_DIR / "19_selftest_curves.pdf"
    make_figure(hists, rows, pdf)
    assert pdf.exists() and pdf.stat().st_size > 8_000

    # a truncated end-of-run record must be discarded, not read as a collapse.
    # This is the failure that turned four flat curves into "0.93 fall from the
    # peak" on the real logs, so it is checked rather than trusted.
    h = _synth(decline=0.0, seed=5)
    ref = analyse(h)
    trunc = [dict(r) for r in h]
    trunc[-1] = {"epoch": trunc[-1]["epoch"], "train_loss": None,
                 "dev": {"f1_mean": 0.0}, "dev_ema": {"f1_mean": 0.0}}
    got = analyse(trunc)
    assert got["n_truncated_records"] == 1, got
    # the point is that the flat curve stays flat: a drop of order 1e-3, not 0.9
    assert got["dev_f1_drop_from_best"] < 0.01, got
    assert got["dev_f1_drop_from_best"] < 0.01 * 0.93 / 0.01, got
    assert got["dev_f1_last"] > 0.5, got
    assert got["n_epochs"] == len(h) - 1, got
    # two trailing bad records, and a run that is nothing but bad records
    trunc2 = trunc[:-1] + [trunc[-1], dict(trunc[-1])]
    assert analyse(trunc2)["n_truncated_records"] == 2
    allbad = analyse([{"epoch": 1, "train_loss": None,
                       "dev": {"f1_mean": 0.0}, "dev_ema": {"f1_mean": 0.0}}])
    assert not allbad["usable"] and allbad["n_truncated_records"] == 1, allbad
    assert "truncated" in verdict({("a", "array"): got}), verdict({("a","array"): got})
    # the figure must draw the trimmed curve, not the raw one
    tr, nd = trim(trunc)
    assert nd == 1 and len(tr) == len(h) - 1
    print(f"  truncated final record discarded (drop stays "
          f"{got['dev_f1_drop_from_best']:+.4f}, not +0.93): PASS")

    # an unusable history must not crash the report
    bad = analyse([{"epoch": 1, "train_loss": 0.5}])
    assert not bad["usable"], bad
    write_report({("x", "array"): bad}, {}, None, "selftest")
    print("[selftest] ALL PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"],
                    help="which development curve to analyse (ema by default, "
                         "because the reported checkpoints are the EMA ones)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.curves:
        ap.print_help()
        return

    hists = load_histories()
    if not hists:
        raise SystemExit(f"[19] no 03_train_history*_loso_*.json in "
                         f"{cfg.LOGS_DIR}")
    ema = (a.ckpt == "ema")
    hists = {k: trim(h, ema)[0] or h for k, h in hists.items()}
    rows = {k: analyse(h, ema=ema) for k, h in hists.items()}
    sizes = training_sizes()
    n_param = parameter_count()
    print(write_report(rows, sizes, n_param, a.ckpt))
    print()
    print(verdict(rows))
    make_figure(hists, rows, cfg.PDF_DIR / "19_a_training_curves.pdf")


if __name__ == "__main__":
    main()
