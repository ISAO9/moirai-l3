#!/usr/bin/env python
"""
28_seed_statistics.py -- training-seed statistics for every held-out site and
for the station-shuffle control (GJI-26-0613, second revision)
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
The first revision repeated only the four DECISIVE folds (pnr-2, clearfield_mw4,
aneth, forge_19) over three training seeds, in the array and per-trace
configurations, and reported mean +/- standard deviation in Section 4.2 and
Fig. 4.  Two other statements of the paper still rested on ONE training run
each:

  * the "tie" verdict at pnr-1, mseel_5h, clearfield_mw6 and mseel_3h
    (Table 2, Section 4.2), which came from the bootstrap CI of a single run;
  * the station-shuffle control (Table 4, Section 5.3, Fig. 9), which came
    from a single shuffle run per site.

The second-round notebook (MOIRAI_L3_R3_all.ipynb, Cell 9) therefore trained
36 further models: seeds 1-3 of array and per-trace at the four tie sites
(24 runs) and seeds 1-3 of the shuffle control at the four decisive sites
(12 runs).  With the 24 existing seed runs this gives three seeds for every
(site, configuration) cell the paper reports.

This script turns those logs into the numbers and figures the manuscript
quotes, deterministically, so that no value in the text is typed from a
notebook printout.  It also acts as a regression gate: the decisive-site
seed statistics already in Section 4.2 must reproduce from the same logs.

WHAT IT DOES
------------
  1. Reads every logs/04_test_results_*_<ckpt>*_seed*.json (seed repeats)
     and, for reference only, the seed-less base runs (SEED 1234).  Each seed
     run must own a logs/03_train_history*.json with >= 26 epochs, i.e. it
     ran to the early-stopping patience (25) at least once; an evaluation of
     a model whose training was interrupted is refused.
  2. Per (site, configuration): n, mean, sd (ddof = 1), min, max of F1-mean
     (`f1_mean_primary` = mean of P and S F1 at +/-20 ms, the quantity of
     Tables 2 and 4), and the same for the P and S F1 separately.  The base
     run is listed beside the seed statistics but never enters them, so the
     numbers stay comparable with the Section 4.2 values of the first
     revision (three seeds, base run excluded).
  3. Paired-by-seed differences: array - per-trace at all eight sites,
     array - shuffle and per-trace - shuffle at the four shuffle sites.  The
     seed number is shared by both runs of a pair (same data-ordering seed),
     so the difference is paired by seed; the script reports each seed's
     difference, their mean +/- sd, whether every seed has the same sign,
     and whether the mean +/- sd bands of the two configurations overlap.
  4. Equivalence reading for the tie sites: |mean difference| and the largest
     |per-seed difference| against the +/-0.02 margin used by the paired
     inference of script 27, so the text can say in one sentence whether a
     consistent small difference is also a practically negligible one.
  5. Regression gate (--gate, on by default): the decisive-site seed
     statistics must agree with the Section 4.2 values of the first revision
     to within 0.0015 (three-decimal rounding).  If they do not, the logs on
     disk are not the logs the paper was written from, and the script stops.
  6. Figures (white background, legend outside the axes):
       PDF/28_a_fig4_seed_all_sites.pdf  array vs per-trace, all eight
                                          sites, mean +/- sd with the three
                                          seed values overlaid  (new Fig. 4)
       PDF/28_b_fig9_shuffle_seeds.pdf    array / shuffle / per-trace at the
                                          four shuffle sites, mean +/- sd
                                          with seed values       (new Fig. 9)
     Both are also written as 300-dpi PNG for the Word manuscript.
  7. A markdown report with every table in manuscript form (three decimals),
     and a JSON with the full-precision values.

OUTPUTS
-------
  logs/28_seed_statistics_<ckpt>.json
  logs/28_seed_statistics_<ckpt>.md
  PDF/28_a_fig4_seed_all_sites.{pdf,png}
  PDF/28_b_fig9_shuffle_seeds.{pdf,png}

USAGE
-----
  python src/28_seed_statistics.py --ckpt ema          # normal run
  python src/28_seed_statistics.py --ckpt ema --no-gate
  python src/28_seed_statistics.py --selftest          # synthetic logs, no data
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
PRIMARY_TOL_KEY = f"tol_{cfg.EVALU.PRIMARY_TOLERANCE_S}s"      # 'tol_0.02s'
MIN_EPOCHS = cfg.TRAIN.EARLY_STOP_PATIENCE + 1                 # 26
EQUIV_MARGIN = 0.02                                             # script 27 margin

# Table 2 order: the four ties, then the four decisive sites
SITE_ORDER = ["pnr-1", "mseel_5h", "clearfield_mw6", "mseel_3h",
              "pnr-2", "clearfield_mw4", "aneth", "forge_19"]
TIE_SITES = SITE_ORDER[:4]
DECISIVE_SITES = SITE_ORDER[4:]
SHUFFLE_SITES = ["forge_19", "clearfield_mw4", "pnr-2", "aneth",
                 "mseel_5h"]                   # Table 4 order; mseel_5h only if trained
PAIRED_JSON = "27_paired_inference.json"       # optional: per-site paired verdicts
MODES = ("array", "shuf", "pertrace")
MODE_LABEL = {"array": "array", "shuf": "shuffle", "pertrace": "per-trace"}

# Section 4.2 of the first revision (three seeds, base run excluded).
# Used as the regression gate: the logs on disk must reproduce these.
R1_SECTION_4_2 = {
    ("forge_19", "array"): (0.458, 0.107), ("forge_19", "pertrace"): (0.808, 0.050),
    ("aneth", "array"): (0.885, 0.004), ("aneth", "pertrace"): (0.935, 0.012),
    ("clearfield_mw4", "array"): (0.858, 0.004), ("clearfield_mw4", "pertrace"): (0.752, 0.018),
    ("pnr-2", "array"): (0.874, 0.001), ("pnr-2", "pertrace"): (0.812, 0.012),
}
GATE_TOL = 0.0015

COLOR = {"array": "#1f6feb", "shuf": "#E07A28", "pertrace": "#8A8A8A"}


# ----------------------------------------------------------------------------
# 1. reading the evaluation logs
# ----------------------------------------------------------------------------
def _parse_name(nm: str):
    """'04_test_results_<site>_<ckpt>[_pertrace][_shuf]_loso_<site>[_seedN].json'
    -> (mode, heldout, seed | None)."""
    m = re.match(r"04_test_results_.+?_(?:ema|best|last)(.*)\.json$", nm)
    if not m or "_loso_" not in m.group(1):
        return None
    sfx = m.group(1)
    mode = "pertrace" if "_pertrace" in sfx else "shuf" if "_shuf" in sfx else "array"
    tail = sfx.split("_loso_")[-1]
    ms = re.search(r"_seed(\d+)$", tail)
    seed = int(ms.group(1)) if ms else None
    site = tail[: ms.start()] if ms else tail
    return mode, site, seed, sfx


def _history_epochs(logs_dir: Path, sfx: str) -> int:
    h = logs_dir / f"03_train_history{sfx}.json"
    if not h.is_file():
        return 0
    try:
        return len(json.loads(h.read_text()))
    except Exception:
        return 0


def read_runs(logs_dir: Path, ckpt: str = "ema"):
    """Return (seed_runs, base_runs); each a list of dicts with
    site, mode, seed, f1, p, s, p_rec, s_rec, file, n_epochs."""
    seed_runs, base_runs, refused = [], [], []
    for f in sorted(logs_dir.glob(f"04_test_results_*_{ckpt}*.json")):
        parsed = _parse_name(f.name)
        if parsed is None:
            continue
        mode, site, seed, sfx = parsed
        d = json.loads(f.read_text())
        if d.get("ckpt", ckpt) != ckpt:
            continue
        # fields written by the revised 04 must agree with the file name
        if "method" in d:
            assert d["method"] == mode, f"{f.name}: method={d['method']} vs name {mode}"
        if d.get("heldout"):
            assert d["heldout"] == site, f"{f.name}: heldout={d['heldout']} vs name {site}"
        if d.get("seed") is not None and seed is not None:
            assert int(d["seed"]) == seed, f"{f.name}: seed={d['seed']} vs name {seed}"
        P, S = d["by_phase"]["P"][PRIMARY_TOL_KEY], d["by_phase"]["S"][PRIMARY_TOL_KEY]
        f1 = d.get("f1_mean_primary")
        if f1 is None:
            f1 = 0.5 * (P["f1"] + S["f1"])
        assert abs(f1 - 0.5 * (P["f1"] + S["f1"])) < 1e-9, f"{f.name}: f1_mean_primary inconsistent"
        rec = dict(site=site, mode=mode, seed=seed, f1=float(f1),
                   p=float(P["f1"]), s=float(S["f1"]),
                   p_rec=float(P["recall"]), s_rec=float(S["recall"]),
                   file=f.name, n_epochs=_history_epochs(logs_dir, sfx))
        if seed is None:
            base_runs.append(rec)
        elif rec["n_epochs"] < MIN_EPOCHS:
            refused.append((f.name, rec["n_epochs"]))
        else:
            seed_runs.append(rec)
    if refused:
        raise SystemExit("[28] evaluation logs of unfinished training runs "
                         f"(history < {MIN_EPOCHS} epochs) -- retrain first: {refused}")
    return seed_runs, base_runs


# ----------------------------------------------------------------------------
# 2. statistics
# ----------------------------------------------------------------------------
def _ms(v):
    a = np.asarray(v, float)
    sd = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    return float(a.mean()), sd, float(a.min()), float(a.max())


def cell_stats(seed_runs, base_runs):
    """{(site, mode): {...}} with per-seed values and summary statistics."""
    cells = defaultdict(dict)
    for r in seed_runs:
        key = (r["site"], r["mode"])
        assert r["seed"] not in cells[key], f"duplicate seed {r['seed']} at {key}"
        cells[key][r["seed"]] = r
    base = {(r["site"], r["mode"]): r for r in base_runs}
    out = {}
    for key, by_seed in cells.items():
        seeds = sorted(by_seed)
        d = {"site": key[0], "mode": key[1], "seeds": seeds, "n": len(seeds)}
        for q in ("f1", "p", "s"):
            vals = [by_seed[s][q] for s in seeds]
            m, sd, lo, hi = _ms(vals)
            d[q] = dict(values=vals, mean=m, sd=sd, min=lo, max=hi)
        b = base.get(key)
        d["base"] = None if b is None else dict(f1=b["f1"], p=b["p"], s=b["s"], file=b["file"])
        d["files"] = [by_seed[s]["file"] for s in seeds]
        d["n_epochs"] = [by_seed[s]["n_epochs"] for s in seeds]
        out[key] = d
    return out


def paired(cells, site, a_mode, b_mode, q="f1"):
    """a - b paired by seed number. None if a cell is missing."""
    A, B = cells.get((site, a_mode)), cells.get((site, b_mode))
    if A is None or B is None:
        return None
    common = sorted(set(A["seeds"]) & set(B["seeds"]))
    if not common:
        return None
    av = {s: v for s, v in zip(A["seeds"], A[q]["values"])}
    bv = {s: v for s, v in zip(B["seeds"], B[q]["values"])}
    dl = np.asarray([av[s] - bv[s] for s in common], float)
    overlap = not (A[q]["mean"] - A[q]["sd"] > B[q]["mean"] + B[q]["sd"]
                   or B[q]["mean"] - B[q]["sd"] > A[q]["mean"] + A[q]["sd"])
    return dict(site=site, a=a_mode, b=b_mode, quantity=q, seeds=common,
                delta=[float(x) for x in dl], mean=float(dl.mean()),
                sd=float(dl.std(ddof=1)) if dl.size > 1 else float("nan"),
                max_abs=float(np.abs(dl).max()),
                same_sign=bool(dl.size >= 2 and (np.all(dl > 0) or np.all(dl < 0))),
                bands_overlap=bool(overlap),
                within_margin=bool(np.abs(dl).max() < EQUIV_MARGIN))


def regression_gate(cells):
    """Decisive-site seed statistics must reproduce Section 4.2 (first revision)."""
    bad = []
    for key, (m_exp, sd_exp) in R1_SECTION_4_2.items():
        c = cells.get(key)
        if c is None:
            bad.append((key, "missing"))
            continue
        dm, dsd = abs(c["f1"]["mean"] - m_exp), abs(c["f1"]["sd"] - sd_exp)
        if dm > GATE_TOL or dsd > GATE_TOL:
            bad.append((key, f"got {c['f1']['mean']:.4f} +/- {c['f1']['sd']:.4f}, "
                             f"paper {m_exp:.3f} +/- {sd_exp:.3f}"))
    return bad


def history_census(logs_dir: Path):
    """Count training histories by category (for the model counts of 3.4 / S3)."""
    cats = defaultdict(int)
    for h in sorted(logs_dir.glob("03_train_history*.json")):
        nm = h.name
        cat = ("shuffle seed" if "_shuf" in nm and "_seed" in nm else
               "shuffle base" if "_shuf" in nm else
               "per-trace seed" if "_pertrace" in nm and "_seed" in nm else
               "array seed" if "_seed" in nm else
               "per-trace base" if "_pertrace" in nm and "_loso_" in nm else
               "array base" if "_loso_" in nm else "other")
        cats[cat] += 1
    return dict(cats)


def paired_verdicts(logs_dir: Path, convention: str = "fa_incl") -> dict:
    """{site: label} from script 27 (joint bootstrap on Delta), if its log exists.
    Labels: 'equivalent' (CI inside +/-0.02), 'tie' (CI straddles zero),
    'array' / 'per-trace' (CI excludes zero, by the sign of Delta)."""
    f = logs_dir / PAIRED_JSON
    if not f.is_file():
        return {}
    try:
        d = json.loads(f.read_text())
    except ValueError:
        return {}
    out = {}
    for site, row in d.get("sites", {}).items():
        r = row.get(convention) or {}
        v, delta = r.get("verdict_paired"), r.get("delta")
        if v is None or delta is None:
            continue
        out[site] = ("equivalent" if v == "equivalent" else
                     "tie" if v.startswith("tie") else
                     "array" if delta > 0 else "per-trace")
    return out


# ----------------------------------------------------------------------------
# 3. figures
# ----------------------------------------------------------------------------
def _bars_with_seeds(ax, x, cells, sites, modes, width, label_fn):
    """Grouped bars (mean) with sd error bars and the individual seed values."""
    k = len(modes)
    offs = (np.arange(k) - (k - 1) / 2) * width
    for j, mode in enumerate(modes):
        means, sds = [], []
        for s in sites:
            c = cells.get((s, mode))
            means.append(c["f1"]["mean"] if c else np.nan)
            sds.append(c["f1"]["sd"] if c else np.nan)
        ax.bar(x + offs[j], means, width, yerr=sds, capsize=3, color=COLOR[mode],
               edgecolor="none", label=label_fn(mode))
        for i, s in enumerate(sites):
            c = cells.get((s, mode))
            if c:
                ax.plot(np.full(c["n"], x[i] + offs[j]), c["f1"]["values"], "o",
                        ms=3, mfc="white", mec="black", mew=0.7, zorder=5)
    ax.plot([], [], "o", ms=3, mfc="white", mec="black", mew=0.7, label="individual seed")


def _delta_panel(ax, x, cells, sites, pairs_spec, width, margin=None):
    """Paired-by-seed differences: one marker per seed, a bar at the mean."""
    k = len(pairs_spec)
    offs = (np.arange(k) - (k - 1) / 2) * width
    if margin is not None:
        ax.axhspan(-margin, margin, color="#EEEEEE", zorder=0,
                   label=f"\u00b1{margin:.2f} equivalence margin")
    ax.axhline(0, color="black", lw=0.8)
    for j, (a, b, color, label) in enumerate(pairs_spec):
        for i, s in enumerate(sites):
            pr = paired(cells, s, a, b)
            if pr is None:
                continue
            ax.bar(x[i] + offs[j], pr["mean"], width * 0.9, color=color, alpha=0.35,
                   edgecolor="none", zorder=1, label=label if i == 0 else None)
            ax.plot(np.full(len(pr["delta"]), x[i] + offs[j]), pr["delta"], "o", ms=3.2,
                    mfc=color, mec="black", mew=0.6, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=30, ha="right", fontsize=9)


def figure_all_sites(cells, out_dir: Path, verdicts=None):
    sites = [s for s in SITE_ORDER if (s, "array") in cells and (s, "pertrace") in cells]
    x = np.arange(len(sites))
    verdicts = verdicts or {}
    labels = [f"{s}\n[{verdicts[s]}]" if s in verdicts else s for s in sites]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(8.0, 1.05 * len(sites) + 2.2), 8.2),
                                 facecolor="white", gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    _bars_with_seeds(a1, x, cells, sites, ("array", "pertrace"), 0.36,
                     lambda m: f"{MODE_LABEL[m]} (mean \u00b1 sd over three seeds)")
    n_tie = sum(s in TIE_SITES for s in sites)
    for ax in (a1, a2):
        if 0 < n_tie < len(sites):
            ax.axvline(n_tie - 0.5, color="#BBBBBB", lw=0.8, ls="--")   # Table 2 grouping
    a1.set_xticks(x)
    a1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    a1.set_ylim(0, 1.06)
    a1.set_ylabel("F1-mean (held-out site, EMA, \u00b120 ms)")
    a1.set_title("(a) Array versus per-trace across training seeds, all eight held-out sites")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False, ncol=3, fontsize=8.5)
    _delta_panel(a2, x, cells, sites, [("array", "pertrace", COLOR["array"],
                                        "array \u2212 per-trace, paired by seed (bar: mean)")],
                 0.5, margin=EQUIV_MARGIN)
    a2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    # symmetric-log axis: the +/-0.01 tie-site differences and the -0.35 forge_19
    # difference stay readable on one panel; linear inside +/-0.02
    a2.set_yscale("symlog", linthresh=EQUIV_MARGIN, linscale=1.5)
    ticks = [-0.4, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]
    a2.set_yticks(ticks)
    a2.set_yticklabels([f"{t:+.2f}" if t else "0" for t in ticks], fontsize=8)
    a2.set_ylim(-0.5, 0.25)
    a2.set_ylabel("\u0394 F1-mean (array \u2212 per-trace), symlog axis")
    a2.set_title("(b) Paired-by-seed difference (linear within \u00b10.02, logarithmic outside)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False, ncol=2, fontsize=8.5)
    fig.subplots_adjust(hspace=0.75, bottom=0.12)
    _save(fig, out_dir / "28_a_fig4_seed_all_sites")


def figure_shuffle(cells, out_dir: Path):
    sites = [s for s in SHUFFLE_SITES if all((s, m) in cells for m in MODES)]
    x = np.arange(len(sites))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(8.0, 1.5 * len(sites) + 3.0), 8.4),
                                 facecolor="white", gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    names = {"array": "array (moveout + coupling)", "shuf": "shuffle (coupling only)",
             "pertrace": "per-trace (neither)"}
    _bars_with_seeds(a1, x, cells, sites, MODES, 0.26, lambda m: names[m])
    a1.set_xticks(x)
    a1.set_xticklabels(sites, rotation=15, ha="right", fontsize=9)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("F1-mean (held-out site, EMA, \u00b120 ms)")
    a1.set_title("(a) Station-shuffle control, mean \u00b1 sd over three training seeds")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=2, fontsize=8.5)
    _delta_panel(a2, x, cells, sites,
                 [("array", "shuf", COLOR["array"], "array \u2212 shuffle (effect of ordered moveout)"),
                  ("pertrace", "shuf", COLOR["pertrace"], "per-trace \u2212 shuffle (effect of removing coupling)")],
                 0.36)
    a2.set_ylabel("\u0394 F1-mean, paired by seed")
    a2.set_title("(b) Paired-by-seed differences against the shuffle run")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=1, fontsize=8.5)
    fig.subplots_adjust(hspace=0.65, bottom=0.12)
    _save(fig, out_dir / "28_b_fig9_shuffle_seeds")


def _save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {stem}.pdf (+ .png)")


# ----------------------------------------------------------------------------
# 4. report
# ----------------------------------------------------------------------------
def _f(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _pm(c, q="f1"):
    return f"{c[q]['mean']:.3f} ± {c[q]['sd']:.3f}"


def build_report(cells, pairs, census, ckpt, gate_result):
    L = []
    L.append(f"# 28 seed statistics ({ckpt}; F1-mean = mean of P and S F1 at ±20 ms)")
    L.append("")
    L.append(f"Seed runs: {sum(c['n'] for c in cells.values())}; cells: {len(cells)}; "
             f"all cells have n = {sorted({c['n'] for c in cells.values()})}.")
    L.append(f"Training histories on disk by category: {census} "
             f"(total {sum(census.values())}).")
    L.append(f"Regression gate against Section 4.2 (first revision): "
             f"{'PASS' if not gate_result else 'FAIL ' + str(gate_result)}")
    L.append("")
    L.append("## Table A. All eight sites, array versus per-trace (three seeds; base run SEED 1234 for reference)")
    L.append("")
    L.append("| Site | array F1-mean | per-trace F1-mean | array P / S | per-trace P / S | "
             "Δ per seed (array − per-trace) | Δ mean ± sd | same sign | bands overlap | "
             "max|Δ| < 0.02 | base array / per-trace |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SITE_ORDER:
        A, P = cells.get((s, "array")), cells.get((s, "pertrace"))
        if not (A and P):
            L.append(f"| {s} | (missing) |")
            continue
        pr = pairs[("array-pertrace", s)]
        L.append(f"| {s} | {_pm(A)} | {_pm(P)} | {_pm(A,'p')} / {_pm(A,'s')} | "
                 f"{_pm(P,'p')} / {_pm(P,'s')} | "
                 f"{' '.join(f'{d:+.3f}' for d in pr['delta'])} | "
                 f"{pr['mean']:+.3f} ± {_f(pr['sd'])} | {'yes' if pr['same_sign'] else 'no'} | "
                 f"{'yes' if pr['bands_overlap'] else 'no'} | {'yes' if pr['within_margin'] else 'no'} | "
                 f"{_f(A['base']['f1']) if A['base'] else '--'} / {_f(P['base']['f1']) if P['base'] else '--'} |")
    L.append("")
    L.append("## Table B. Station-shuffle control (mean ± sd over three seeds) -- manuscript Table 4; the mseel_5h row appears only if its shuffle runs exist")
    L.append("")
    L.append("| Site | array P / S / mean | shuffle P / S / mean | per-trace P / S / mean |")
    L.append("|---|---|---|---|")
    for s in SHUFFLE_SITES:
        if not all((s, m) in cells for m in MODES):
            continue
        row = [s] + [f"{_pm(cells[(s, m)],'p')} / {_pm(cells[(s, m)],'s')} / "
                     f"{_pm(cells[(s, m)],'f1')}" for m in MODES]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("## Table C. Paired-by-seed differences at the shuffle sites (F1-mean)")
    L.append("")
    L.append("| Site | pair | Δ per seed | Δ mean ± sd | same sign | bands overlap |")
    L.append("|---|---|---|---|---|---|")
    for s in SHUFFLE_SITES:
        for tag in ("array-shuf", "pertrace-shuf", "array-pertrace"):
            pr = pairs.get((tag, s))
            if pr is None:
                continue
            L.append(f"| {s} | {MODE_LABEL[pr['a']]} − {MODE_LABEL[pr['b']]} | "
                     f"{' '.join(f'{d:+.3f}' for d in pr['delta'])} | "
                     f"{pr['mean']:+.3f} ± {_f(pr['sd'])} | {'yes' if pr['same_sign'] else 'no'} | "
                     f"{'yes' if pr['bands_overlap'] else 'no'} |")
    L.append("")
    L.append("## Table D. Per-phase paired differences at the shuffle sites (array − shuffle)")
    L.append("")
    L.append("| Site | ΔP per seed (mean) | ΔS per seed (mean) |")
    L.append("|---|---|---|")
    for s in SHUFFLE_SITES:
        dp, ds = pairs.get(("array-shuf-P", s)), pairs.get(("array-shuf-S", s))
        if dp and ds:
            L.append(f"| {s} | {' '.join(f'{d:+.3f}' for d in dp['delta'])} ({dp['mean']:+.3f}) | "
                     f"{' '.join(f'{d:+.3f}' for d in ds['delta'])} ({ds['mean']:+.3f}) |")
    L.append("")
    L.append("## Sentences for the manuscript (three decimals, seed runs only)")
    L.append("")
    for s in SITE_ORDER:
        A, P = cells.get((s, "array")), cells.get((s, "pertrace"))
        if A and P:
            L.append(f"- {s}: array {_pm(A)} versus per-trace {_pm(P)}")
    for s in SHUFFLE_SITES:
        if all((s, m) in cells for m in MODES):
            L.append(f"- {s} (shuffle): array {_pm(cells[(s,'array')])} → shuffle "
                     f"{_pm(cells[(s,'shuf')])} → per-trace {_pm(cells[(s,'pertrace')])}; "
                     f"P {_pm(cells[(s,'array')],'p')} → {_pm(cells[(s,'shuf')],'p')} → "
                     f"{_pm(cells[(s,'pertrace')],'p')}; "
                     f"S {_pm(cells[(s,'array')],'s')} → {_pm(cells[(s,'shuf')],'s')} → "
                     f"{_pm(cells[(s,'pertrace')],'s')}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# 5. driver
# ----------------------------------------------------------------------------
def run(logs_dir: Path, pdf_dir: Path, ckpt: str = "ema", gate: bool = True, quiet=False):
    seed_runs, base_runs = read_runs(logs_dir, ckpt)
    if not seed_runs:
        raise SystemExit(f"[28] no seed runs found under {logs_dir}")
    cells = cell_stats(seed_runs, base_runs)
    pairs = {}
    for s in SITE_ORDER:
        pr = paired(cells, s, "array", "pertrace")
        if pr:
            pairs[("array-pertrace", s)] = pr
    for s in SHUFFLE_SITES:
        for tag, a, b in (("array-shuf", "array", "shuf"), ("pertrace-shuf", "pertrace", "shuf")):
            pr = paired(cells, s, a, b)
            if pr:
                pairs[(tag, s)] = pr
        for q in ("p", "s"):
            pr = paired(cells, s, "array", "shuf", q)
            if pr:
                pairs[(f"array-shuf-{q.upper()}", s)] = pr
    gate_result = regression_gate(cells)
    if gate and gate_result:
        raise SystemExit("[28] REGRESSION GATE FAILED -- the seed logs on disk do not reproduce "
                         f"Section 4.2 of the first revision: {gate_result}")
    census = history_census(logs_dir)

    report = build_report(cells, pairs, census, ckpt, gate_result)
    (logs_dir / f"28_seed_statistics_{ckpt}.md").write_text(report)
    payload = {
        "metric": "f1_mean_primary = 0.5 * (P F1 + S F1) at +/-20 ms; seed runs only, ddof=1",
        "ckpt": ckpt, "min_epochs": MIN_EPOCHS, "equivalence_margin": EQUIV_MARGIN,
        "n_seed_runs": len(seed_runs), "n_base_runs": len(base_runs),
        "history_census": census, "regression_gate": "PASS" if not gate_result else gate_result,
        "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
        "pairs": {f"{k[0]}|{k[1]}": v for k, v in pairs.items()},
    }
    (logs_dir / f"28_seed_statistics_{ckpt}.json").write_text(json.dumps(payload, indent=2))
    verd = paired_verdicts(logs_dir)
    print(f"[28] paired verdicts from {PAIRED_JSON}: "
          f"{verd if verd else 'not found (figure drawn without verdict labels)'}")
    figure_all_sites(cells, pdf_dir, verd)
    figure_shuffle(cells, pdf_dir)
    if not quiet:
        print(report)
        print(f"[28] wrote {logs_dir / f'28_seed_statistics_{ckpt}.md'} and .json")
    return cells, pairs, gate_result


# ----------------------------------------------------------------------------
# 6. self-test on synthetic logs
# ----------------------------------------------------------------------------
def _synthetic_result(site, mode, seed, p, s, with_fields=True):
    bp = {}
    for ph, f in (("P", p), ("S", s)):
        bp[ph] = {f"tol_{t}s": dict(precision=f, recall=f, f1=f, fail_rate=1 - f, mae_s=0.004,
                                    median_ae_s=0.003, n_matched=100) for t in cfg.EVALU.TOLERANCES_S}
    d = dict(by_phase=bp, tolerances_s=list(cfg.EVALU.TOLERANCES_S), n_events=500,
             n_earthquake_events=400, n_noise_events=100, noise_false_alarm_picks=2,
             noise_false_alarm_per_event=0.02, f1_mean_primary=0.5 * (p + s),
             site=site, split="test", ckpt="ema", peak_thr=0.2)
    if with_fields:
        d.update(heldout=site, seed=seed, method=mode)
    return d


def selftest():
    print("=" * 70)
    print("28_seed_statistics -- self-test on synthetic logs")
    print("=" * 70)
    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as td:
        logs, pdf = Path(td) / "logs", Path(td) / "PDF"
        logs.mkdir(); pdf.mkdir()
        # exact Section 4.2 values are reproduced by construction for the gate:
        # choose three seed values whose mean/sd equal the paper's numbers
        def triple(mean, sd):
            # values mean - sd, mean, mean + sd  ->  mean exact, sd(ddof=1) = sd
            return [mean - sd, mean, mean + sd]
        for (site, mode), (m, sd) in R1_SECTION_4_2.items():
            for seed, f1 in zip((1, 2, 3), triple(m, sd)):
                sfx = ("_pertrace" if mode == "pertrace" else "") + f"_loso_{site}_seed{seed}"
                (logs / f"04_test_results_{site}_ema{sfx}.json").write_text(
                    json.dumps(_synthetic_result(site, mode, seed, f1, f1)))
                (logs / f"03_train_history{sfx}.json").write_text(json.dumps([{}] * 40))
        # tie sites: array higher by exactly 0.010 at every seed, within margin
        for site in [s for s in SITE_ORDER if (s, "array") not in R1_SECTION_4_2]:
            for seed, base in zip((1, 2, 3), (0.88, 0.89, 0.90)):
                for mode, f1 in (("array", base + 0.010), ("pertrace", base)):
                    sfx = ("_pertrace" if mode == "pertrace" else "") + f"_loso_{site}_seed{seed}"
                    (logs / f"04_test_results_{site}_ema{sfx}.json").write_text(
                        json.dumps(_synthetic_result(site, mode, seed, f1, f1)))
                    (logs / f"03_train_history{sfx}.json").write_text(json.dumps([{}] * 30))
        # shuffle at the four decisive sites only (the optional mseel_5h shuffle
        # is absent here, so the report must skip that row rather than fail)
        for site in SHUFFLE_SITES[:4]:
            for seed, (p, s) in zip((1, 2, 3), ((0.80, 0.70), (0.82, 0.72), (0.84, 0.74))):
                sfx = f"_shuf_loso_{site}_seed{seed}"
                (logs / f"04_test_results_{site}_ema{sfx}.json").write_text(
                    json.dumps(_synthetic_result(site, "shuf", seed, p, s)))
                (logs / f"03_train_history{sfx}.json").write_text(json.dumps([{}] * 26))
        # base runs in the OLD structure (no method / heldout / seed keys)
        for site in SITE_ORDER:
            for mode in ("array", "pertrace"):
                sfx = ("_pertrace" if mode == "pertrace" else "") + f"_loso_{site}"
                (logs / f"04_test_results_{site}_ema{sfx}.json").write_text(
                    json.dumps(_synthetic_result(site, mode, None, 0.9, 0.9, with_fields=False)))
                (logs / f"03_train_history{sfx}.json").write_text(json.dumps([{}] * 30))
        # an unfinished run must be refused
        bad_sfx = "_loso_pnr-1_seed9"
        (logs / f"04_test_results_pnr-1_ema{bad_sfx}.json").write_text(
            json.dumps(_synthetic_result("pnr-1", "array", 9, 0.5, 0.5)))
        (logs / f"03_train_history{bad_sfx}.json").write_text(json.dumps([{}] * 2))
        try:
            read_runs(logs, "ema"); refused = False
        except SystemExit:
            refused = True
        ck("evaluation of an unfinished run (2-epoch history) is refused", refused)
        (logs / f"04_test_results_pnr-1_ema{bad_sfx}.json").unlink()
        (logs / f"03_train_history{bad_sfx}.json").unlink()

        # a script-27 log so the verdict labels are exercised
        (logs / PAIRED_JSON).write_text(json.dumps({"sites": {
            "pnr-1": {"fa_incl": {"verdict_paired": "equivalent", "delta": 0.0}},
            "mseel_5h": {"fa_incl": {"verdict_paired": "different", "delta": 0.017}},
            "aneth": {"fa_incl": {"verdict_paired": "different", "delta": -0.057}},
            "clearfield_mw6": {"fa_incl": {"verdict_paired": "tie (inconclusive)", "delta": 0.009}}}}))
        vd = paired_verdicts(logs)
        ck("paired verdict labels read from script 27 (equivalent / array / per-trace / tie)",
           vd == {"pnr-1": "equivalent", "mseel_5h": "array", "aneth": "per-trace",
                  "clearfield_mw6": "tie"})
        ck("missing script-27 log -> no labels", paired_verdicts(Path(td)) == {})
        cells, pairs, gate = run(logs, pdf, "ema", gate=True, quiet=True)
        ck("60 seed runs read (24 decisive + 24 tie + 12 shuffle)",
           sum(c["n"] for c in cells.values()) == 60)
        ck("20 cells, every cell n = 3", len(cells) == 20 and all(c["n"] == 3 for c in cells.values()))
        ck("base runs in old structure read for reference (16)",
           sum(1 for c in cells.values() if c["base"] is not None) == 16)
        ck("regression gate passes on values built to match Section 4.2", gate == [])
        c = cells[("forge_19", "array")]
        ck("mean exact", abs(c["f1"]["mean"] - 0.458) < 1e-12)
        ck("sd uses ddof = 1", abs(c["f1"]["sd"] - 0.107) < 1e-12)
        ck("min / max", abs(c["f1"]["min"] - 0.351) < 1e-12 and abs(c["f1"]["max"] - 0.565) < 1e-12)
        pr = pairs[("array-pertrace", "pnr-1")]
        ck("tie-site paired delta = +0.010 at every seed", all(abs(d - 0.010) < 1e-12 for d in pr["delta"]))
        ck("same_sign true, within 0.02 margin true", pr["same_sign"] and pr["within_margin"])
        ck("bands overlap (sd 0.01 each, gap 0.01)", pr["bands_overlap"])
        pr = pairs[("array-pertrace", "forge_19")]
        ck("decisive site: bands do not overlap, same sign", (not pr["bands_overlap"]) and pr["same_sign"])
        ck("decisive site: not within margin", not pr["within_margin"])
        pr = pairs[("array-shuf", "forge_19")]
        ck("array - shuffle paired over the 3 common seeds", pr["seeds"] == [1, 2, 3] and len(pr["delta"]) == 3)
        pp, ps = pairs[("array-shuf-P", "aneth")], pairs[("array-shuf-S", "aneth")]
        ck("per-phase deltas differ by the 0.10 P/S offset of the shuffle runs",
           abs((ps["mean"] - pp["mean"]) - 0.10) < 1e-9)
        # mixed-sign case
        pr_mixed = paired({("x", "array"): dict(seeds=[1, 2], n=2, f1=dict(values=[0.5, 0.6], mean=0.55, sd=0.07)),
                           ("x", "pertrace"): dict(seeds=[1, 2], n=2, f1=dict(values=[0.6, 0.5], mean=0.55, sd=0.07))},
                          "x", "array", "pertrace")
        ck("mixed-sign deltas -> same_sign false", pr_mixed["same_sign"] is False)
        ck("missing cell -> paired returns None", paired(cells, "nowhere", "array", "pertrace") is None)
        ck("report written", (logs / "28_seed_statistics_ema.md").is_file())
        rep = (logs / "28_seed_statistics_ema.md").read_text()
        ck("Table B lists the four shuffle sites and skips the untrained mseel_5h row",
           rep.split("## Table B")[1].split("## Table C")[0].count("| forge_19 | 0.") == 1
           and "| mseel_5h | 0." in rep.split("## Table A")[1].split("## Table B")[0]
           and rep.split("## Table B")[1].split("## Table C")[0].count("| mseel_5h |") == 0)
        ck("json written with 20 cells", len(json.loads((logs / "28_seed_statistics_ema.json").read_text())["cells"]) == 20)
        ck("Fig. 4 replacement written (pdf + png)",
           (pdf / "28_a_fig4_seed_all_sites.pdf").is_file() and (pdf / "28_a_fig4_seed_all_sites.png").is_file())
        ck("Fig. 9 replacement written (pdf + png)",
           (pdf / "28_b_fig9_shuffle_seeds.pdf").is_file() and (pdf / "28_b_fig9_shuffle_seeds.png").is_file())
        census = history_census(logs)
        ck("history census counts categories (array seed 24, shuffle seed 12, per-trace seed 24)",
           census.get("array seed") == 24 and census.get("shuffle seed") == 12 and census.get("per-trace seed") == 24)
        # gate failure is detected
        bad = dict(cells); bad[("forge_19", "array")] = dict(bad[("forge_19", "array")])
        bad[("forge_19", "array")]["f1"] = dict(bad[("forge_19", "array")]["f1"], mean=0.470)
        ck("gate detects a 0.012 shift of a decisive-site mean", regression_gate(bad) != [])
        # name parser
        ck("parser: array base", _parse_name("04_test_results_pnr-1_ema_loso_pnr-1.json")[:3] == ("array", "pnr-1", None))
        ck("parser: per-trace seed", _parse_name("04_test_results_mseel_3h_ema_pertrace_loso_mseel_3h_seed2.json")[:3]
           == ("pertrace", "mseel_3h", 2))
        ck("parser: shuffle seed", _parse_name("04_test_results_forge_19_ema_shuf_loso_forge_19_seed3.json")[:3]
           == ("shuf", "forge_19", 3))
        ck("parser: non-LOSO file ignored", _parse_name("04_test_results_pnr-1_ema.json") is None)

    n_ok = sum(ok for _, ok in checks)
    print(f"\nself-test: {n_ok}/{len(checks)} PASS")
    if n_ok != len(checks):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ckpt", default="ema")
    ap.add_argument("--no-gate", action="store_true", help="skip the Section 4.2 regression gate")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(cfg.LOGS_DIR, cfg.PDF_DIR, a.ckpt, gate=not a.no_gate)


if __name__ == "__main__":
    main()
