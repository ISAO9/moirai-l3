#!/usr/bin/env python
"""
11_fa_inclusive_scoring.py -- GJI major revision: fold noise false alarms
into the site-level score (Reviewer 2, comment 2; Reviewer 1, lines 176-177).
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
In the submitted paper, precision/recall/F1 are computed on earthquake
events and noise false alarms are reported separately. Reviewer 2 asks
whether pnr-2 remains "array-favoured" once noise-event picks enter the
score. This script recomputes Table 2 in an OPERATIONAL (FA-inclusive)
variant:

    precision_FA = TP / (TP + FP_eq + FA_noise)

i.e. every pick declared on a noise-only event is a false positive for the
phase it was declared on. Recall is unchanged (noise events contribute no
ground truth). Both the original and FA-inclusive scores are reported side
by side at the fixed 0.30 threshold, with EVENT-LEVEL BOOTSTRAP 95 per cent
confidence intervals (earthquake AND noise events resampled jointly, so the
noise-event proportion is preserved in every replicate) and a verdict per
site (array / per-trace / tie) under each scoring convention.

INPUTS
  logs/10_cache_<site>_<ckpt>_loso_*.npz        from 10_threshold_sweep.py
  (run script 10 --cache for array AND --per-trace on every site first)

OUTPUTS
  logs/11_fa_inclusive_<ckpt>.json              full numbers
  logs/11_table2_revised.md / .tex              drop-in revised Table 2

RUN
  python src/11_fa_inclusive_scoring.py
  python src/11_fa_inclusive_scoring.py --selftest
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
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")
DATA, EVALU = cfg.DATA, cfg.EVALU
ts = _load("10_threshold_sweep.py")               # reuse cache + prf_at


# ----------------------------------------------------------------------------
def score_both(stats: dict, thr: float, tol: int) -> dict:
    """Original and FA-inclusive P/S/mean F1 at one threshold."""
    base = ts.prf_at(stats, thr, tol)
    out = {"orig": {ph: base[ph]["f1"] for ph in ("P", "S")},
           "fa_per_noise_event": base["fa_per_noise_event"]}
    out["orig"]["mean"] = base["f1_mean"]
    fa = {}
    for ph in ("P", "S"):
        b = base[ph]
        denom = b["tp"] + b["fp"] + b["fa_noise"]
        prec = b["tp"] / denom if denom else 0.0
        rec = b["recall"]
        fa[ph] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fa["mean"] = 0.5 * (fa["P"] + fa["S"])
    out["fa_incl"] = fa
    return out


def _subset(stats: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in stats.items()}


def bootstrap(stats_a: dict, stats_p: dict, thr: float, tol: int,
              nboot: int = 300, seed: int = 0) -> dict:
    """Joint event bootstrap on F1-mean (orig and FA-inclusive), per config."""
    rng = np.random.default_rng(seed)
    E = stats_a["gt_time"].shape[0]
    acc = {c: {"orig": [], "fa_incl": []} for c in ("array", "pertrace")}
    for _ in range(nboot):
        idx = rng.integers(0, E, E)
        for c, st in (("array", stats_a), ("pertrace", stats_p)):
            r = score_both(_subset(st, idx), thr, tol)
            acc[c]["orig"].append(r["orig"]["mean"])
            acc[c]["fa_incl"].append(r["fa_incl"]["mean"])

    def ci(a):
        a = np.asarray(a)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {c: {k: ci(v) for k, v in d.items()} for c, d in acc.items()}


def verdict(ci_a, ci_p) -> str:
    if ci_a[0] > ci_p[1]:
        return "array"
    if ci_p[0] > ci_a[1]:
        return "per-trace"
    return "tie"


# ----------------------------------------------------------------------------
def run(args):
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    thr = EVALU.PEAK_PROB_THRESHOLD
    caches = ts._load_caches(args.ckpt)
    sites = sorted({s for (s, m) in caches
                    if (s, "array") in caches and (s, "pertrace") in caches})
    if not sites:
        print("[11] need array AND per-trace caches (script 10 --cache).")
        return 1
    rows, out = [], {}
    for site in sites:
        sa, sp = caches[(site, "array")], caches[(site, "pertrace")]
        ra, rp = (score_both(s, thr, tol) for s in (sa, sp))
        cis = bootstrap(sa, sp, thr, tol, args.nboot)
        v_orig = verdict(cis["array"]["orig"], cis["pertrace"]["orig"])
        v_fa = verdict(cis["array"]["fa_incl"], cis["pertrace"]["fa_incl"])
        out[site] = dict(array=ra, pertrace=rp, ci=cis,
                         verdict_orig=v_orig, verdict_fa_incl=v_fa)
        rows.append((site, ra, rp, v_orig, v_fa))
        flip = "  <-- VERDICT CHANGES" if v_orig != v_fa else ""
        print(f"{site:16s} orig: arr {ra['orig']['mean']:.3f} vs pt "
              f"{rp['orig']['mean']:.3f} [{v_orig}] | FA-incl: arr "
              f"{ra['fa_incl']['mean']:.3f} vs pt {rp['fa_incl']['mean']:.3f} "
              f"[{v_fa}]{flip}")

    jpath = cfg.LOGS_DIR / f"11_fa_inclusive_{args.ckpt}.json"
    json.dump(out, open(jpath, "w"), indent=2)

    md = ["| Site | array F1-mean | per-trace F1-mean | array FA-incl | "
          "per-trace FA-incl | array FA/noise ev | verdict (orig) | "
          "verdict (FA-incl) |",
          "|---|---|---|---|---|---|---|---|"]
    for site, ra, rp, vo, vf in rows:
        md.append(f"| {site} | {ra['orig']['mean']:.3f} | "
                  f"{rp['orig']['mean']:.3f} | {ra['fa_incl']['mean']:.3f} | "
                  f"{rp['fa_incl']['mean']:.3f} | "
                  f"{ra['fa_per_noise_event']:.2f} | {vo} | {vf} |")
    (cfg.LOGS_DIR / "11_table2_revised.md").write_text("\n".join(md))
    print(f"[save] {jpath}\n[save] {cfg.LOGS_DIR / '11_table2_revised.md'}")
    return 0


# ----------------------------------------------------------------------------
def selftest():
    """Noise false alarms must lower the FA-inclusive score, never raise it,
    and leave the original score untouched."""
    E, S, T = 10, 12, 256
    labels = np.zeros((E, 3, S, T), np.float32)
    preds = np.zeros_like(labels)
    for e in range(6):                            # 6 eq events, perfect picks
        for st in range(S):
            labels[e, 0, st, 80] = labels[e, 1, st, 150] = 1.0
            preds[e, 0, st, 80] = preds[e, 1, st, 150] = 0.9
    for e in range(6, 10):                        # 4 noise events: model fires!
        preds[e, 0, :, 50] = 0.8
    stats = ts.reduce_to_stats(preds, labels)
    r = score_both(stats, 0.30, 40)
    ok = (abs(r["orig"]["P"] - 1.0) < 1e-9
          and r["fa_incl"]["P"] < 1.0
          and abs(r["orig"]["S"] - r["fa_incl"]["S"]) < 1e-9
          and abs(r["fa_per_noise_event"] - 12.0) < 1e-9)
    print(f"  orig P {r['orig']['P']:.3f} vs FA-incl P {r['fa_incl']['P']:.3f}"
          f" (must drop), FA/noise ev {r['fa_per_noise_event']:.1f}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
