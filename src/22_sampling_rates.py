#!/usr/bin/env python
"""
22_sampling_rates.py -- GJI revision round 2: what are AMBER's ORIGINAL
sampling rates, per site? (Reviewer 2, page 6)
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Reviewer 2 annotated the data section with

    "What are the original sampling rates"

The manuscript only states the rate the records are processed TO ("a common
sampling rate (2000 Hz)"). The rate they came FROM is a property of each
site's acquisition and is not in the paper. It is in AMBER's own metadata
table, one row per station-trace, so it needs no waveforms, no GPU and no
model -- just the CSV.

WHAT THIS SCRIPT DOES
---------------------
It is written DISCOVERY-FIRST on purpose. The exact column name is not
assumed: every column whose name mentions a sampling rate is found, and the
distinct values are reported per site with their row counts. That way the
output is informative whether AMBER stores one rate per site, several, or a
single post-resampling constant.

  1. INVENTORY -- list the candidate columns actually present, so a renamed
     column is visible instead of silently producing nothing.
  2. PER SITE  -- for each of the eight sites used in the paper, the distinct
     values of each candidate column, with row counts and the share of rows.
  3. VERDICT   -- state plainly whether the column holds the ORIGINAL rates
     (values differing between or within sites) or has already been
     overwritten by the common processing rate (a single constant). If it is
     a constant, the original rates are NOT in this table and must come from
     the AMBER data paper instead; the script says so rather than letting a
     constant be mistaken for a measurement.

OUTPUTS
  logs/22_sampling_rates.json   per-site distinct values and counts
  logs/22_sampling_rates.md     the same as a table to paste

RUN (no GPU, no waveforms; needs AMBER's metadata CSV only)
  python src/22_sampling_rates.py
  python src/22_sampling_rates.py --csv /path/to/metadata.csv
  # self-test (synthetic CSV, no AMBER required):
  python src/22_sampling_rates.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(path: str):
    """Numbered files cannot be imported, so load the config by path."""
    spec = importlib.util.spec_from_file_location(
        path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


cfg = _load("00_config_l3.py")

# the eight sites the paper uses, in the order of Table 1
SITES = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6",
         "pnr-2", "clearfield_mw4", "aneth", "forge_19"]

# a column is a candidate if its name mentions a rate or a frequency in Hz
CAND = re.compile(r"(sampl|_fs\b|\bfs_|freq|hz)", re.I)
SITE_COL_CAND = ("dataset", "site", "source_dataset")


# ----------------------------------------------------------------------------
def find_site_column(df: pd.DataFrame) -> str:
    """The column naming the site. Refuses to guess beyond the known names."""
    for c in SITE_COL_CAND:
        if c in df.columns:
            return c
    raise KeyError(f"no site column among {SITE_COL_CAND}; "
                   f"columns={list(df.columns)}")


def candidate_columns(df: pd.DataFrame) -> list[str]:
    """Every column whose NAME mentions a sampling rate or a frequency."""
    return [c for c in df.columns if CAND.search(str(c))]


def per_site_values(df: pd.DataFrame, site_col: str, col: str) -> dict:
    """{site: [{value, rows, frac}, ...]} for one candidate column."""
    out = {}
    for site in SITES:
        sub = df[df[site_col].astype(str) == site]
        if sub.empty:
            out[site] = []
            continue
        vc = sub[col].value_counts(dropna=False)
        n = int(vc.sum())
        out[site] = [dict(value=(None if pd.isna(v) else
                                 (float(v) if isinstance(v, (int, float))
                                  else str(v))),
                          rows=int(c), frac=round(float(c) / n, 4))
                     for v, c in vc.items()]
    return out


def verdict(per_site: dict) -> str:
    """Does this column carry ORIGINAL rates, or a post-processing constant?"""
    vals = {e["value"] for rows in per_site.values() for e in rows
            if e["value"] is not None}
    if not vals:
        return ("EMPTY -- the column holds no values for these sites; "
                "the original rates are not here.")
    if len(vals) == 1:
        only = next(iter(vals))
        return (f"CONSTANT at {only} for every site. This is the rate the "
                f"records were processed TO, not the rate they came FROM. "
                f"The original rates are NOT in this table and must be taken "
                f"from the AMBER data paper (Leung et al. 2026) instead.")
    multi = [s for s, rows in per_site.items() if len(rows) > 1]
    return (f"VARIES -- {len(vals)} distinct values across the eight sites"
            + (f"; {len(multi)} site(s) carry more than one value: "
               f"{', '.join(multi)}" if multi else "")
            + ". These are the original rates.")


# ----------------------------------------------------------------------------
def write_report(report: dict, tag: str = "") -> Path:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"22_sampling_rates{tag}"
    (cfg.LOGS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2))

    md = [f"# AMBER sampling rates per site  (source: {report['csv']})", "",
          f"rows read: {report['n_rows']:,}", "",
          f"candidate columns found: {', '.join(report['candidates']) or '(none)'}",
          ""]
    for col, block in report["columns"].items():
        md += [f"## column `{col}`", "", f"**{block['verdict']}**", "",
               "| Site | value | rows | share |", "|---|---|---|---|"]
        for site in SITES:
            rows = block["per_site"][site]
            if not rows:
                md.append(f"| {site} | (no rows) | 0 | — |")
            for e in rows:
                md.append(f"| {site} | {e['value']} | {e['rows']:,} | "
                          f"{e['frac']:.3f} |")
        md.append("")
    p = cfg.LOGS_DIR / f"{stem}.md"
    p.write_text("\n".join(md))
    return p


def run(csv_path: Path, tag: str = "") -> dict:
    if not csv_path.exists():
        raise SystemExit(f"[22] metadata CSV not found: {csv_path}\n"
                         f"     pass --csv, or set AMBER_CSV in the environment.")
    df = pd.read_csv(csv_path, low_memory=False)
    site_col = find_site_column(df)
    cands = candidate_columns(df)
    print(f"[22] {csv_path}  rows {len(df):,}  site column '{site_col}'")
    print(f"[22] candidate columns: {cands or '(none)'}")
    if not cands:
        print("[22] no column name mentions a sampling rate. All columns:")
        for c in df.columns:
            print("      ", c)

    report = dict(csv=str(csv_path), n_rows=int(len(df)),
                  site_column=site_col, candidates=cands, columns={})
    for col in cands:
        ps = per_site_values(df, site_col, col)
        v = verdict(ps)
        report["columns"][col] = dict(per_site=ps, verdict=v)
        print(f"\n[22] column '{col}': {v}")
        for site in SITES:
            vals = ", ".join(f"{e['value']} ({e['rows']:,})" for e in ps[site])
            print(f"       {site:16s} {vals or '(no rows)'}")

    p = write_report(report, tag)
    print(f"\n[22] wrote {p} and {p.with_suffix('.json')}")
    return report


# ----------------------------------------------------------------------------
# self-test (synthetic CSV; no AMBER required)
# ----------------------------------------------------------------------------
def selftest() -> None:
    import tempfile
    rows = []
    # two sites at 4000 Hz, two at 2000, one site mixing both -> VARIES
    plan = {"pnr-1": [4000.0] * 10, "mseel_3h": [2000.0] * 10,
            "mseel_5h": [2000.0] * 10, "clearfield_mw6": [1000.0] * 10,
            "pnr-2": [4000.0] * 6 + [2000.0] * 4, "clearfield_mw4": [2000.0] * 10,
            "aneth": [500.0] * 10, "forge_19": [4000.0] * 10}
    for site, vals in plan.items():
        for v in vals:
            rows.append(dict(dataset=site, trace_sampling_rate_hz=v,
                             unrelated_column="x"))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "metadata.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        rep = run(p, tag="_selftest")

    assert rep["site_column"] == "dataset"
    assert rep["candidates"] == ["trace_sampling_rate_hz"], rep["candidates"]
    blk = rep["columns"]["trace_sampling_rate_hz"]
    assert blk["verdict"].startswith("VARIES"), blk["verdict"]
    assert "pnr-2" in blk["verdict"], blk["verdict"]        # the mixed site
    ps = blk["per_site"]
    assert len(ps["pnr-2"]) == 2, ps["pnr-2"]
    assert ps["pnr-1"][0]["value"] == 4000.0 and ps["pnr-1"][0]["rows"] == 10
    assert abs(sum(e["frac"] for e in ps["pnr-2"]) - 1.0) < 1e-9

    # a column that is constant everywhere must be reported as the processed rate
    const = {s: [dict(value=2000.0, rows=10, frac=1.0)] for s in SITES}
    v = verdict(const)
    assert v.startswith("CONSTANT") and "AMBER data paper" in v, v

    # an empty column must not be mistaken for a measurement
    assert verdict({s: [] for s in SITES}).startswith("EMPTY")

    # the site column must be refused rather than guessed
    try:
        find_site_column(pd.DataFrame({"foo": [1]}))
    except KeyError:
        pass
    else:
        raise SystemExit("NG: unknown site column was not refused")

    print("[selftest] ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", default=None,
                    help="AMBER metadata CSV (default: config AMBER_CSV)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(Path(a.csv) if a.csv else cfg.AMBER_CSV)


if __name__ == "__main__":
    main()
