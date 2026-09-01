#!/usr/bin/env python
"""
12_table3_audit.py -- GJI major revision: audit and correct Table 3
(Reviewer 2, comment 4).
=============================================================================
WHAT THIS SCRIPT DOES
---------------------
Reviewer 2 found that three "mean" cells in Table 3 (station-shuffle control)
do not equal the average of the P and S cells on the same row:
    clearfield_mw4 per-trace : (0.793+0.756)/2 = 0.7745, table says 0.768
    pnr-2 shuffle            : (0.847+0.757)/2 = 0.802,  table says 0.827
    aneth shuffle            : (0.922+0.909)/2 = 0.9155, table says 0.930
This script re-reads the AUTHORITATIVE result logs
(logs/04_test_results_*_loso_*.json for array / per-trace / shuffle) and
prints, for each of the four decisive sites and three configurations:
    P F1, S F1, recomputed mean = (P+S)/2
side by side with the manuscript values, flagging every discrepancy, and
writes a corrected, drop-in Table 3 (markdown + LaTeX) whose means are
recomputed from the logged P/S values. Any cell whose LOGGED P/S disagrees
with the manuscript is flagged loudly so the source of the error
(transcription vs stale log) is unambiguous in the response letter.

OUTPUTS
  logs/12_table3_audit.json      per-cell comparison
  logs/12_table3_corrected.md    corrected table (markdown)
  logs/12_table3_corrected.tex   corrected table (LaTeX)

RUN
  python src/12_table3_audit.py
  python src/12_table3_audit.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(path: str):
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")
EVALU = cfg.EVALU

SITES = ("forge_19", "clearfield_mw4", "pnr-2", "aneth")

# manuscript Table 3 as submitted (P, S, mean) -----------------------------
MANUSCRIPT = {
    "forge_19":       {"array": (0.178, 0.857, 0.518),
                       "shuffle": (0.327, 0.813, 0.570),
                       "pertrace": (0.755, 0.928, 0.842)},
    "clearfield_mw4": {"array": (0.860, 0.844, 0.852),
                       "shuffle": (0.878, 0.853, 0.866),
                       "pertrace": (0.793, 0.756, 0.768)},
    "pnr-2":          {"array": (0.890, 0.871, 0.881),
                       "shuffle": (0.847, 0.757, 0.827),
                       "pertrace": (0.887, 0.794, 0.841)},
    "aneth":          {"array": (0.943, 0.835, 0.889),
                       "shuffle": (0.922, 0.909, 0.930),
                       "pertrace": (0.965, 0.916, 0.941)},
}


def _phase_f1(d: dict, ph: str) -> float:
    key = f"tol_{EVALU.PRIMARY_TOLERANCE_S}s"
    return float(d["by_phase"][ph][key]["f1"])


def read_logs(ckpt: str) -> dict:
    """{site: {method: (P, S, mean-from-logs)}} from 04 result jsons."""
    pat = {"array": "04_test_results_{s}_{c}_loso_{s}.json",
           "shuffle": "04_test_results_{s}_{c}_shuf_loso_{s}.json",
           "pertrace": "04_test_results_{s}_{c}_pertrace_loso_{s}.json"}
    out = {}
    for site in SITES:
        out[site] = {}
        for m, tpl in pat.items():
            f = cfg.LOGS_DIR / tpl.format(s=site, c=ckpt)
            if not f.exists():
                out[site][m] = None
                continue
            d = json.load(open(f))
            p, s = _phase_f1(d, "P"), _phase_f1(d, "S")
            out[site][m] = (p, s, 0.5 * (p + s))
    return out


def audit(logged: dict) -> tuple[dict, list[str]]:
    report, lines = {}, []
    hdr = (f"{'site':16s} {'method':9s} | {'manuscript P/S/mean':>24s} | "
           f"{'logged P/S/mean':>24s} | status")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for site in SITES:
        report[site] = {}
        for m in ("array", "shuffle", "pertrace"):
            mp, ms_, mm = MANUSCRIPT[site][m]
            man_mean_ok = abs(0.5 * (mp + ms_) - mm) < 6e-4
            lg = logged.get(site, {}).get(m)
            if lg is None:
                status = ("ARITHMETIC ERROR in manuscript (mean != (P+S)/2); "
                          "log file missing -- rerun 04" if not man_mean_ok
                          else "log file missing")
                lgs = " " * 24
            else:
                lp, ls, lm = lg
                ps_match = abs(lp - mp) < 6e-4 and abs(ls - ms_) < 6e-4
                mean_match = abs(lm - mm) < 6e-4
                if ps_match and mean_match:
                    status = "OK"
                elif ps_match and not mean_match:
                    status = (f"TRANSCRIPTION ERROR: mean should be {lm:.4f} "
                              f"(P/S in the manuscript are correct)")
                else:
                    status = ("P/S DISAGREE with logs -- manuscript cells "
                              "must be replaced from the logs")
                lgs = f"{lp:.3f}/{ls:.3f}/{lm:.4f}".rjust(24)
            report[site][m] = dict(
                manuscript=[mp, ms_, mm],
                logged=list(lg) if lg else None,
                manuscript_mean_consistent=man_mean_ok, status=status)
            lines.append(f"{site:16s} {m:9s} | "
                         f"{f'{mp:.3f}/{ms_:.3f}/{mm:.3f}'.rjust(24)} | "
                         f"{lgs} | {status}")
    return report, lines


def write_corrected(logged: dict):
    md = ["| Site | array P / S / mean | shuffle P / S / mean | "
          "per-trace P / S / mean |", "|---|---|---|---|"]
    tex = [r"\begin{tabular}{lccc}", r"\hline",
           r"Site & array P / S / mean & shuffle P / S / mean & "
           r"per-trace P / S / mean \\", r"\hline"]
    for site in SITES:
        cells_md, cells_tex = [], []
        for m in ("array", "shuffle", "pertrace"):
            src = logged.get(site, {}).get(m)
            if src is None:                       # fall back: recompute mean
                p, s, _ = MANUSCRIPT[site][m]
                src = (p, s, 0.5 * (p + s))
            p, s, mean = src
            cells_md.append(f"{p:.3f} / {s:.3f} / {mean:.3f}")
            cells_tex.append(f"{p:.3f} / {s:.3f} / {mean:.3f}")
        md.append(f"| {site} | " + " | ".join(cells_md) + " |")
        tex.append(site.replace("_", r"\_") + " & "
                   + " & ".join(cells_tex) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    (cfg.LOGS_DIR / "12_table3_corrected.md").write_text("\n".join(md))
    (cfg.LOGS_DIR / "12_table3_corrected.tex").write_text("\n".join(tex))


def selftest():
    """The three reviewer-flagged cells MUST be detected as inconsistent and
    the nine other cells as internally consistent."""
    bad = [(s, m) for s in SITES for m in ("array", "shuffle", "pertrace")
           if abs(0.5 * (MANUSCRIPT[s][m][0] + MANUSCRIPT[s][m][1])
                  - MANUSCRIPT[s][m][2]) >= 6e-4]
    expect = {("clearfield_mw4", "pertrace"), ("pnr-2", "shuffle"),
              ("aneth", "shuffle")}
    ok = set(bad) == expect
    print(f"  inconsistent manuscript cells detected: {sorted(bad)}")
    print(f"  matches reviewer's list: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    logged = read_logs(args.ckpt)
    report, lines = audit(logged)
    print("\n".join(lines))
    json.dump(report, open(cfg.LOGS_DIR / "12_table3_audit.json", "w"),
              indent=2)
    write_corrected(logged)
    print(f"[save] {cfg.LOGS_DIR / '12_table3_audit.json'}")
    print(f"[save] {cfg.LOGS_DIR / '12_table3_corrected.md'} / .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
