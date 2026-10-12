#!/usr/bin/env python3
"""
27_paired_inference.py -- GJI-26-0613 R3: inference on the PAIRED difference,
and an across-site test of the headline negative result.

WHAT THE SUBMITTED VERSION DOES, AND WHY A REVIEWER WILL OBJECT
---------------------------------------------------------------
11_fa_inclusive_scoring.py already resamples events JOINTLY: in each bootstrap
iteration the same resampled index set is scored under both configurations.  The
per-resample difference is therefore available -- but it is discarded, and the
verdict is read off the NON-OVERLAP OF TWO MARGINAL INTERVALS:

    def verdict(ci_a, ci_p):
        if ci_a[0] > ci_p[1]: return "array"
        ...

Non-overlap of two marginal CIs is not a test of the difference.  It is
conservative when the two estimates are positively correlated, which they are
here by construction: the same events, the same catalogue, the same windows.  A
reviewer will ask for a confidence interval on Delta itself.  Three further gaps:

  * nboot = 300 is too few for 2.5 / 97.5 percentiles (1,000+ is the convention).
  * "tie" is inferred from non-significance, with no equivalence bound stated.
  * the headline claim -- "statistically indistinguishable in median F1
    (0.885 versus 0.878)" across eight sites -- carries no across-site test.

This script closes all four from the SAME cached sufficient statistics, so no
re-inference, no GPU and no AMBER waveforms are needed.

WHAT IT COMPUTES
----------------
per site, under both false-alarm conventions:
  * Delta = F1-mean(array) - F1-mean(per-trace) on the full evaluation set
  * a percentile CI on Delta from the joint bootstrap (default B = 2000)
  * Pr(Delta > 0) over the resamples
  * a TOST-style equivalence verdict against a stated bound (default 0.02):
    "equivalent" when the Delta CI lies wholly inside +/- bound, "tie
    (inconclusive)" when it merely straddles zero
across the eight sites:
  * exact two-sided sign test on the per-site Delta
  * Wilcoxon signed-rank test on the per-site Delta
  * the median and site-mean Delta

The equivalence bound is a declared quantity, not a fitted one.  0.02 is twice
the +/-0.01 window-draw spread that Table 3's caption already concedes, so it is
the smallest bound the evaluation itself can resolve.  Pass --bound to change it;
the value used is written into the log.

RUN
    python src/27_paired_inference.py --all                 # needs logs/10_cache_*.npz
    python src/27_paired_inference.py --all --nboot 5000
    python src/27_paired_inference.py --selftest            # synthetic, no caches

OUTPUT
    logs/27_paired_inference.json
    logs/27_paired_inference.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  HERE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- core, testable
def sign_test(deltas, mu: float = 0.0) -> dict:
    """Exact two-sided sign test. Zeros are dropped, as the test requires."""
    d = np.asarray([x - mu for x in deltas], float)
    d = d[d != 0]
    n = int(d.size)
    k = int((d > 0).sum())
    if n == 0:
        return dict(n=0, n_positive=0, p_value=1.0)
    # two-sided: twice the smaller tail of Binomial(n, 1/2), capped at 1
    def cdf(x):
        return sum(math.comb(n, i) for i in range(0, x + 1)) / 2 ** n
    p = 2 * min(cdf(k), 1 - cdf(k - 1) if k else 1.0)
    return dict(n=n, n_positive=k, p_value=float(min(1.0, p)))


def wilcoxon_signed_rank(deltas, mu: float = 0.0) -> dict:
    """Wilcoxon signed-rank, exact for the small n this study has (n <= 8).

    Ties in |d| share the average rank; zeros are dropped. With n <= 20 the
    null distribution is enumerated rather than approximated, because a normal
    approximation at n = 8 is not defensible.
    """
    d = np.asarray([x - mu for x in deltas], float)
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return dict(n=0, W=None, p_value=1.0, exact=True)
    order = np.argsort(np.abs(d), kind="stable")
    ranks = np.empty(n, float)
    a = np.abs(d)[order]
    i = 0
    while i < n:                               # average ranks within a tie group
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[i:j + 1] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    signed = np.empty(n, float)
    signed[order] = ranks
    w_plus = float(signed[d > 0].sum())
    w_minus = float(signed[d < 0].sum())
    W = min(w_plus, w_minus)
    if n <= 20:                                # enumerate the null exactly
        from itertools import product
        tot = 0
        hit = 0
        for signs in product((-1, 1), repeat=n):
            s = np.asarray(signs, float)
            wp = float(signed[s > 0].sum())
            wm = float(signed[s < 0].sum())
            tot += 1
            hit += (min(wp, wm) <= W + 1e-12)
        p = hit / tot
        return dict(n=n, W=W, W_plus=w_plus, W_minus=w_minus,
                    p_value=float(min(1.0, p)), exact=True)
    return dict(n=n, W=W, W_plus=w_plus, W_minus=w_minus,
                p_value=None, exact=False)


def equivalence(ci: tuple[float, float], bound: float) -> str:
    """TOST-style reading of a two-sided CI against a declared bound."""
    lo, hi = ci
    if lo > 0 or hi < 0:
        return "different"
    if lo >= -bound and hi <= bound:
        return "equivalent"
    return "tie (inconclusive)"


def paired_bootstrap(stats_a, stats_p, score_both, thr, tol,
                     nboot: int, seed: int = 0) -> dict:
    """Joint event bootstrap keeping the per-resample DIFFERENCE.

    Mirrors 11_fa_inclusive_scoring.bootstrap -- same rng construction, same
    resampled index applied to both configurations -- but accumulates
    Delta = mean(array) - mean(per-trace) as well as the two marginals.
    """
    rng = np.random.default_rng(seed)
    n_ev = stats_a["gt_time"].shape[0]
    acc = {k: {"array": [], "pertrace": [], "delta": []}
           for k in ("orig", "fa_incl")}
    for _ in range(nboot):
        idx = rng.integers(0, n_ev, n_ev)
        r_a = score_both(_subset(stats_a, idx), thr, tol)
        r_p = score_both(_subset(stats_p, idx), thr, tol)
        for k in ("orig", "fa_incl"):
            a, p = r_a[k]["mean"], r_p[k]["mean"]
            acc[k]["array"].append(a)
            acc[k]["pertrace"].append(p)
            acc[k]["delta"].append(a - p)
    return acc


def _subset(stats: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if hasattr(v, "shape") and v.shape[:1] == idx.shape[:1]
                else v) for k, v in stats.items()}


def pct_ci(a) -> tuple[float, float]:
    a = np.asarray(a, float)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ------------------------------------------------------------------- selftest
def selftest() -> None:
    # sign test against a hand case: 8 values, 2 positive
    r = sign_test([1, 1, -1, -1, -1, -1, -1, -1])
    assert r["n"] == 8 and r["n_positive"] == 2
    # two-sided p = 2 * P(X <= 2) = 2 * (1+8+28)/256 = 0.2890625
    assert abs(r["p_value"] - 0.2890625) < 1e-9, r
    assert sign_test([0, 0, 1, -1])["n"] == 2, "zeros must be dropped"
    print("  sign_test: 2 of 8 positive -> p = 0.2891: PASS")

    # Wilcoxon on a textbook case: all same sign -> smallest possible p
    r = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6, 7, 8])
    assert r["W"] == 0 and r["exact"]
    assert abs(r["p_value"] - 2 / 256) < 1e-9, r
    # symmetric data -> p = 1
    r2 = wilcoxon_signed_rank([1, -1])
    assert abs(r2["p_value"] - 1.0) < 1e-9, r2
    print("  wilcoxon_signed_rank: all-positive n=8 -> p = 0.0078; "
          "symmetric -> p = 1: PASS")

    assert equivalence((-0.005, 0.004), 0.02) == "equivalent"
    assert equivalence((-0.05, 0.004), 0.02) == "tie (inconclusive)"
    assert equivalence((0.01, 0.05), 0.02) == "different"
    assert equivalence((-0.05, -0.01), 0.02) == "different"
    print("  equivalence: inside bound / straddling / disjoint: PASS")

    # paired bootstrap on synthetic stats: a constant +0.10 offset must come
    # back as a Delta CI around +0.10 that excludes zero
    rng = np.random.default_rng(0)
    n = 400
    sa = {"gt_time": rng.normal(size=(n, 1)), "v": rng.normal(0.8, 0.05, n)}
    sp = {"gt_time": sa["gt_time"], "v": sa["v"] - 0.10}

    def score_both(st, thr, tol):
        m = float(np.mean(st["v"]))
        return {"orig": {"mean": m}, "fa_incl": {"mean": m - 0.01}}

    acc = paired_bootstrap(sa, sp, score_both, 0.3, 40, nboot=400, seed=1)
    ci = pct_ci(acc["orig"]["delta"])
    assert 0.09 < ci[0] and ci[1] < 0.11, ci
    assert equivalence(ci, 0.02) == "different", ci
    # the marginal CIs must be WIDER than the paired one (the point of pairing)
    w_pair = ci[1] - ci[0]
    w_marg = (pct_ci(acc["orig"]["array"])[1] - pct_ci(acc["orig"]["array"])[0])
    assert w_pair < w_marg, (w_pair, w_marg)
    print(f"  paired_bootstrap: Delta CI [{ci[0]:.4f}, {ci[1]:.4f}] excludes 0; "
          f"paired width {w_pair:.4f} < marginal {w_marg:.4f}: PASS")

    print("[selftest] ALL PASS")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--bound", type=float, default=0.02,
                    help="equivalence bound on F1-mean (default 0.02)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.all:
        ap.error("pass --all (or --selftest)")

    cfg = _load("00_config_l3.py")
    ts = _load("10_threshold_sweep.py")
    fa = _load("11_fa_inclusive_scoring.py")
    tol = int(round(cfg.EVALU.PRIMARY_TOLERANCE_S * cfg.DATA.FS))
    thr = cfg.EVALU.PEAK_PROB_THRESHOLD

    caches = ts._load_caches(a.ckpt)
    sites = sorted({s for (s, m) in caches
                    if (s, "array") in caches and (s, "pertrace") in caches})
    assert sites, ("logs/10_cache_*.npz が無い。先に "
                   "10_threshold_sweep.py --cache を各サイト・各配置で回す")
    print(f"[27] sites={len(sites)}  nboot={a.nboot}  bound=±{a.bound}")

    res = {"nboot": a.nboot, "equivalence_bound": a.bound, "ckpt": a.ckpt,
           "note": ("Delta = F1-mean(array) - F1-mean(per-trace). The bootstrap "
                    "is the joint event bootstrap of 11_fa_inclusive_scoring: "
                    "one resampled index set scored under both configurations, "
                    "so Delta is paired within each resample."),
           "sites": {}}
    for s in sites:
        sa, sp = caches[(s, "array")], caches[(s, "pertrace")]
        point_a = fa.score_both(sa, thr, tol)
        point_p = fa.score_both(sp, thr, tol)
        acc = paired_bootstrap(sa, sp, fa.score_both, thr, tol,
                               nboot=a.nboot, seed=0)
        row = {}
        for k in ("orig", "fa_incl"):
            d = np.asarray(acc[k]["delta"], float)
            ci = pct_ci(d)
            row[k] = dict(
                array=point_a[k]["mean"], pertrace=point_p[k]["mean"],
                delta=point_a[k]["mean"] - point_p[k]["mean"],
                delta_ci=[ci[0], ci[1]],
                pr_delta_gt_0=float((d > 0).mean()),
                marginal_ci_array=list(pct_ci(acc[k]["array"])),
                marginal_ci_pertrace=list(pct_ci(acc[k]["pertrace"])),
                verdict_paired=equivalence(ci, a.bound),
            )
        res["sites"][s] = row
        o = row["orig"]
        print(f"  {s:16s} Delta {o['delta']:+.4f}  CI [{o['delta_ci'][0]:+.4f},"
              f" {o['delta_ci'][1]:+.4f}]  Pr(>0)={o['pr_delta_gt_0']:.3f}"
              f"  {o['verdict_paired']}")

    print("\n[27] across the eight sites")
    for k in ("orig", "fa_incl"):
        ds = [res["sites"][s][k]["delta"] for s in sites]
        st, wx = sign_test(ds), wilcoxon_signed_rank(ds)
        res[f"across_sites_{k}"] = dict(
            deltas={s: res["sites"][s][k]["delta"] for s in sites},
            median_delta=float(np.median(ds)), mean_delta=float(np.mean(ds)),
            sign_test=st, wilcoxon=wx)
        print(f"  {k:9s} median Delta {np.median(ds):+.4f}  "
              f"mean {np.mean(ds):+.4f}  "
              f"sign test p={st['p_value']:.3f} ({st['n_positive']}/{st['n']} "
              f"positive)  Wilcoxon p={wx['p_value']:.3f}")

    out = Path(cfg.LOGS_DIR) / "27_paired_inference.json"
    out.write_text(json.dumps(res, indent=2))

    md = ["# Paired inference on the array-minus-per-trace difference", "",
          f"Delta = F1-mean(array) - F1-mean(per-trace). "
          f"B = {a.nboot} joint event-bootstrap resamples; one resampled index "
          f"set is scored under both configurations, so Delta is paired within "
          f"each resample. Equivalence bound ±{a.bound} on F1-mean.", ""]
    for k, title in (("orig", "false alarms excluded"),
                     ("fa_incl", "false alarms included")):
        md += [f"## {title}", "",
               "| Site | array | per-trace | Delta | 95 % CI on Delta | "
               "Pr(Delta>0) | Paired verdict |", "|---|---|---|---|---|---|---|"]
        for s in sites:
            r = res["sites"][s][k]
            md.append(f"| {s} | {r['array']:.3f} | {r['pertrace']:.3f} | "
                      f"{r['delta']:+.3f} | [{r['delta_ci'][0]:+.3f}, "
                      f"{r['delta_ci'][1]:+.3f}] | {r['pr_delta_gt_0']:.3f} | "
                      f"{r['verdict_paired']} |")
        x = res[f"across_sites_{k}"]
        md += ["", f"Across the eight sites: median Delta "
                   f"{x['median_delta']:+.3f}, site-mean "
                   f"{x['mean_delta']:+.3f}; exact sign test "
                   f"p = {x['sign_test']['p_value']:.3f} "
                   f"({x['sign_test']['n_positive']} of "
                   f"{x['sign_test']['n']} positive); Wilcoxon signed-rank "
                   f"p = {x['wilcoxon']['p_value']:.3f}.", ""]
    (Path(cfg.LOGS_DIR) / "27_paired_inference.md").write_text("\n".join(md))
    print(f"\n[27] wrote {out.name} and 27_paired_inference.md")


if __name__ == "__main__":
    main()
