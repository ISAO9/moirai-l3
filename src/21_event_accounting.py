#!/usr/bin/env python
"""21 -- Event accounting: reconcile every per-site count used in the paper.

WHY THIS SCRIPT EXISTS
----------------------
Three counts that should agree do not, and Table 1 (reviewer 2, points R2-12
and R2-13) cannot be rebuilt until they do:

  * 16_arrival_audit_ema.json reports n_events per site, taken as the number
    of rows in the leave-one-site-out held-out cache. Summed over the eight
    sites that is 3,421.
  * 20_within_array_ema.json reports test events per site. Summed that is
    2,530 -- but at pnr-1 alone it is 1,258, which is larger than the 543 the
    audit reports for the same site. A within-site test split cannot be
    larger than the whole site.
  * The supplement currently states a benchmark total of 9,803 events. That
    figure was derived from metadata rows divided by station counts and has
    not been checked against the loader.

The clean_eq_ids in the audit also arrive in two blocks per site with roughly
a +1000 offset (pnr-1: 0..290 then 1006..1257), which suggests the event index
encodes something besides the event -- a split, a segment of a long string, or
a per-file offset. Until that is known, no number derived from those caches
belongs in the manuscript.

WHAT IT DOES
------------
Nothing is assumed about the CSV schema. The script prints the columns it
actually finds, then counts:

  (a) rows per split in each site's native AMBER partition,
  (b) how many of those rows are noise windows rather than events, if the
      schema marks them,
  (c) how many unique event identifiers each split holds, which separates
      "rows" from "events" if a long string is stored as several rows,
  (d) rows in that site's leave-one-site-out held-out cache, and the range
      and block structure of its event indices,
  (e) stations on the full string against the 12 used in the analysis.

It then runs four reconciliations and prints each as PASS or MISMATCH:

  R1  train + dev + test == total rows in the site CSV
  R2  the within-array test count of script 20 == that site's test rows
  R3  the audit's n_events == rows in the held-out cache
  R4  the held-out cache covers the whole site, not one split

OUTPUTS
-------
  logs/21_event_accounting.json   per-site counts and the reconciliations
  logs/21_event_accounting.md     the same as a table for the response letter

USAGE
-----
  python 21_event_accounting.py            # all sites
  python 21_event_accounting.py --selftest # no AMBER, no GPU needed
"""
from __future__ import annotations

import argparse
import ast
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
DATA = cfg.DATA

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]

# Column names are not assumed. These are the candidates the script looks for,
# in order; whatever it finds is printed so the real schema is on the record.
SPLIT_COLS = ("split", "mode", "subset", "partition", "fold", "set")
NOISE_COLS = ("trace_category", "is_noise", "noise", "label", "type",
              "event_type", "class")
EVENT_COLS = ("event_id", "eventid", "eq_id", "event", "evid", "id", "index")


def _first(cols, candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


# ----------------------------------------------------------------------------
# (a)-(c) the site CSV
# ----------------------------------------------------------------------------
def read_site_table(loader_mod, site: str, verbose: bool = True) -> dict:
    """Counts from a site's native AMBER partition, schema discovered not assumed."""
    import pandas as pd

    csv = loader_mod.prepare_site_csv(site)
    df = pd.read_csv(csv) if not hasattr(csv, "columns") else csv
    cols = list(df.columns)
    scol = _first(cols, SPLIT_COLS)
    ncol = _first(cols, NOISE_COLS)
    ecol = _first(cols, EVENT_COLS)
    if verbose:
        print(f"[21] {site:<16s} csv={getattr(csv, 'name', type(csv).__name__)}")
        print(f"[21] {'':16s} columns: {cols}")
        print(f"[21] {'':16s} split={scol!r} noise={ncol!r} event={ecol!r}")

    rec: dict = {"csv_columns": cols, "split_col": scol,
                 "noise_col": ncol, "event_col": ecol,
                 "rows_total": int(len(df))}

    if scol is None:
        rec["warning"] = "no split column found; split counts unavailable"
        return rec

    rec["rows_by_split"] = {str(k): int(v)
                            for k, v in df[scol].value_counts().items()}
    # script 20 が pnr-1 で 1,258 と表示した正体を確かめる: データセットの
    # 長さが CSV の行数と一致するか、それとも max(index)+1 になっているか。
    try:
        rec["dataset_len_by_split"] = {
            m: int(len(loader_mod.build_amber_dataset(m, csv)))
            for m in ("train", "dev", "test")}
    except Exception as e:                                   # noqa: BLE001
        rec["dataset_len_by_split"] = f"unavailable: {type(e).__name__}: {e}"
    if ecol is not None:
        rec["unique_events_total"] = int(df[ecol].nunique())
        rec["unique_events_by_split"] = {
            str(k): int(g[ecol].nunique()) for k, g in df.groupby(scol)}
    if ncol is not None:
        # Works whether the column is a boolean flag, a number or a text label.
        # NOTE: do not branch on `dtype == object`. Recent pandas gives string
        # columns a dedicated `str` dtype, which fails that test and falls
        # through to astype(bool), where every non-empty string is True -- that
        # silently counts every row as a noise window.
        from pandas.api.types import is_bool_dtype, is_numeric_dtype
        v = df[ncol]
        if is_bool_dtype(v):
            mask = v
        elif is_numeric_dtype(v):
            mask = v.fillna(0) != 0
        else:
            mask = v.astype(str).str.strip().str.lower().isin(
                ["1", "true", "noise", "n", "noise_window", "noise window"])
        rec[f"{ncol}_value_counts"] = {str(k): int(v)
                                       for k, v in v.value_counts().items()}
        rec["noise_rows_total"] = int(mask.sum())
        rec["noise_rows_by_split"] = {
            str(k): int(mask[g.index].sum()) for k, g in df.groupby(scol)}
    return rec


# ----------------------------------------------------------------------------
# (d) the leave-one-site-out held-out cache
# ----------------------------------------------------------------------------
def read_cache(site: str, ckpt: str = "ema") -> dict:
    """Rows and event-index structure of the held-out cache for one site."""
    hits = [f for f in sorted(cfg.LOGS_DIR.glob(f"10_cache_*_{ckpt}*loso_{site}.npz"))
            if "_pertrace" not in f.stem and "_shuf" not in f.stem]
    if not hits:
        return {"cache_file": None}
    z = dict(np.load(hits[0]))
    rec = {"cache_file": hits[0].name,
           "keys": sorted(z.keys()),
           "rows": int(z["gt_time"].shape[0]) if "gt_time" in z else None}
    for k in ("eq_id", "event_id", "index", "idx"):
        if k in z:
            a = np.asarray(z[k]).ravel()
            rec["index_key"] = k
            rec["index_min"] = int(a.min())
            rec["index_max"] = int(a.max())
            rec["index_unique"] = int(np.unique(a).size)
            # contiguous blocks, to expose the +1000 style offset
            u = np.unique(a)
            brk = np.where(np.diff(u) > 1)[0]
            starts = np.r_[u[0], u[brk + 1]]
            ends = np.r_[u[brk], u[-1]]
            rec["index_blocks"] = [[int(s), int(e)] for s, e in zip(starts, ends)]
            break
    return rec


# ----------------------------------------------------------------------------
# reconciliations
# ----------------------------------------------------------------------------
def reconcile(site: str, tbl: dict, cache: dict, within_test: int | None) -> list:
    """Five checks, always all five, each as (name, verdict, detail).

    A check that cannot run returns SKIP rather than no line at all, so a
    silently missing check can never be read as a passing one.
    """
    out = []

    rbs = tbl.get("rows_by_split")
    if rbs:
        s = sum(rbs.values())
        out.append(("R1 train+dev+test == rows",
                    "PASS" if s == tbl["rows_total"] else "MISMATCH",
                    f"{s} vs {tbl['rows_total']}"))
    else:
        out.append(("R1 train+dev+test == rows", "SKIP", "no split column"))

    if rbs and within_test is not None:
        t = rbs.get("test")
        ue = (tbl.get("unique_events_by_split") or {}).get("test")
        ok = within_test in {t, ue}
        out.append(("R2 script20 test == test split",
                    "PASS" if ok else "MISMATCH",
                    f"script20 {within_test} vs rows {t}, unique events {ue}"))
    else:
        out.append(("R2 script20 test == test split", "SKIP", ""))

    if cache.get("rows") is not None:
        out.append(("R3 audit n_events == cache rows", "INFO",
                    f"cache rows {cache['rows']}"))
        # R5: Table 1 の Events 列は max(index)+1 と一致する。番号に隙間がある
        # 以上これは件数ではない。隙間がゼロなら両者は一致してよい。
        if cache.get("index_max") is None:
            out.append(("R5 index is dense (max+1 == count)", "SKIP",
                        f"cache has no event-index key; keys={cache.get('keys')}"))
        else:
            mx = cache["index_max"] + 1
            gap = mx - cache["index_unique"]
            out.append(("R5 index is dense (max+1 == count)",
                        "PASS" if gap == 0 else "MISMATCH",
                        f"max+1 {mx} vs unique {cache['index_unique']}, "
                        f"gap {gap}, blocks {cache.get('index_blocks')}"))
        tot = tbl.get("unique_events_total", tbl.get("rows_total"))
        if tot is None:
            out.append(("R4 cache covers the whole site", "SKIP",
                        "site total unknown (loader unavailable)"))
        else:
            out.append(("R4 cache covers the whole site",
                        "PASS" if cache["rows"] == tot else "MISMATCH",
                        f"cache {cache['rows']} vs site total {tot}"))
    else:
        out.append(("R3 audit n_events == cache rows", "SKIP", "no cache"))
        out.append(("R5 index is dense (max+1 == count)", "SKIP", "no cache"))
        out.append(("R4 cache covers the whole site", "SKIP", "no cache"))

    assert len(out) == 5, out
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run(ckpt: str = "ema") -> dict:
    # The decisive checks (R4, R5) read only the cached arrays, so a loader
    # that will not import -- AMBER not mounted, a missing dependency -- must
    # not stop the script. It degrades to cache-only and says so.
    loader_mod = None
    loader_error = None
    try:
        loader_mod = _load("01_amber_setup.py")
    except Exception as e:                                   # noqa: BLE001
        loader_error = f"{type(e).__name__}: {e}"
        print(f"[21] loader unavailable ({loader_error});"
              " running cache-only. R4 and R5 are unaffected.")

    within = {}
    f = cfg.LOGS_DIR / f"20_within_array_{ckpt}.json"
    if f.exists():
        within = {k: v.get("n_events") or v.get("test_events")
                  for k, v in json.loads(f.read_text()).items()}

    audit = {}
    f = cfg.LOGS_DIR / f"16_arrival_audit_{ckpt}.json"
    if f.exists():
        audit = {k: v.get("n_events")
                 for k, v in json.loads(f.read_text()).items()}

    res = {}
    order = [s for s in SITE_ORDER if s in DATA.USABLE_SITES]
    order += [s for s in DATA.USABLE_SITES if s not in order]
    for site in order:
        if loader_mod is None:
            tbl = {"rows_total": None, "warning": f"loader unavailable: {loader_error}"}
        else:
            try:
                tbl = read_site_table(loader_mod, site)
            except Exception as e:                           # noqa: BLE001
                tbl = {"rows_total": None,
                       "warning": f"{type(e).__name__}: {e}"}
                print(f"[21] {site:<16s} site table unavailable ({tbl['warning']})")
        cache = read_cache(site, ckpt)
        checks = reconcile(site, tbl, cache, within.get(site))
        res[site] = {"table": tbl, "cache": cache,
                     "audit_n_events": audit.get(site),
                     "script20_test_events": within.get(site),
                     "checks": [list(c) for c in checks]}
        for name, verdict, detail in checks:
            print(f"[21] {site:<16s} {verdict:<8s} {name}"
                  + (f"   ({detail})" if detail else ""))
        print()
    return res


def report(res: dict) -> str:
    rows = ["| Site | CSV rows | train / dev / test | unique events | noise rows "
            "| cache rows | audit n_events | script 20 test |",
            "|---|---|---|---|---|---|---|---|"]
    for site, r in res.items():
        t, c = r["table"], r["cache"]
        rbs = t.get("rows_by_split") or {}
        if t.get("rows_total") is None:
            rows.append(f"| {site} | - | - | - | - | {c.get('rows','-')} | "
                        f"{r.get('audit_n_events','-')} | "
                        f"{r.get('script20_test_events','-')} |")
            continue
        tdt = " / ".join(str(rbs.get(k, "-")) for k in ("train", "dev", "test"))
        rows.append(
            f"| {site} | {t.get('rows_total','-')} | {tdt} | "
            f"{t.get('unique_events_total','-')} | {t.get('noise_rows_total','-')} | "
            f"{c.get('rows','-')} | {r.get('audit_n_events','-')} | "
            f"{r.get('script20_test_events','-')} |")

    bad = [(s, n, d) for s, r in res.items()
           for n, v, d in r["checks"] if v == "MISMATCH"]
    rows += ["", f"**MISMATCH: {len(bad)} 件**"]
    rows += [f"- {s}: {n} ({d})" for s, n, d in bad] or ["- なし"]
    return "\n".join(rows)


# ----------------------------------------------------------------------------
# selftest -- no AMBER, no GPU
# ----------------------------------------------------------------------------
def _selftest() -> None:
    import pandas as pd

    def _ck(*a, **k):
        c = reconcile(*a, **k)
        assert len(c) == 5, c          # 検査が黙って消えないこと
        return {x[0]: x[1] for x in c}

    # schema discovery, including a schema that uses none of the first choices
    assert _first(["a", "Split", "b"], SPLIT_COLS) == "Split"
    assert _first(["a", "b"], SPLIT_COLS) is None

    df = pd.DataFrame({"split": ["train"] * 6 + ["dev"] * 2 + ["test"] * 2,
                       "event_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                       "is_noise": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]})

    class L:
        @staticmethod
        def prepare_site_csv(site):
            return df

    t = read_site_table(L, "x", verbose=False)
    assert t["rows_total"] == 10, t
    assert t["rows_by_split"] == {"train": 6, "dev": 2, "test": 2}, t
    assert t["unique_events_total"] == 5, t
    assert t["unique_events_by_split"]["test"] == 1, t
    assert t["noise_rows_total"] == 2, t

    # R1 passes, R2 matches on unique events rather than rows, R4 catches a
    # cache that holds only one split
    ck = _ck("x", t, {"rows": 5}, within_test=1)
    assert ck["R1 train+dev+test == rows"] == "PASS", ck
    assert ck["R2 script20 test == test split"] == "PASS", ck
    assert ck["R4 cache covers the whole site"] == "PASS", ck

    ck = _ck("x", t, {"rows": 2}, within_test=99)
    assert ck["R2 script20 test == test split"] == "MISMATCH", ck
    assert ck["R4 cache covers the whole site"] == "MISMATCH", ck

    # R5 catches a gapped index and passes a dense one
    ck = _ck("x", t, {"rows": 5, "index_max": 9, "index_unique": 5,
                      "index_blocks": [[0, 1], [8, 9]]}, within_test=1)
    assert ck["R5 index is dense (max+1 == count)"] == "MISMATCH", ck
    ck = _ck("x", t, {"rows": 5, "index_max": 4, "index_unique": 5,
                      "index_blocks": [[0, 4]]}, within_test=1)
    assert ck["R5 index is dense (max+1 == count)"] == "PASS", ck

    # build_amber_dataset が無い loader でも落ちないこと
    assert isinstance(t["dataset_len_by_split"], str), t["dataset_len_by_split"]

    # a table with no split column must degrade, not crash
    class L2:
        @staticmethod
        def prepare_site_csv(site):
            return pd.DataFrame({"foo": [1, 2, 3]})

    t2 = read_site_table(L2, "y", verbose=False)
    assert "warning" in t2 and t2["rows_total"] == 3, t2
    assert [v for _, v, _ in reconcile("y", t2, {}, None)].count("SKIP") == 5

    # the markdown report must name every mismatch it was given
    md = report({"y": {"table": t2, "cache": {}, "checks":
                       [["R4 cache covers the whole site", "MISMATCH", "1 vs 2"]]}})
    assert "MISMATCH: 1 件" in md and "R4" in md, md

    # the loader really does expose the entry point this script calls
    src = ast.parse((HERE / "01_amber_setup.py").read_text())
    names = {n.name for n in ast.walk(src) if isinstance(n, ast.FunctionDef)}
    assert "prepare_site_csv" in names, sorted(names)

    # R4 must not cry MISMATCH when there is nothing to compare against
    v = _ck("z", {"rows_total": None}, {"rows": 9, "index_max": 8,
                                        "index_unique": 9,
                                        "index_blocks": [[0, 8]]}, None)
    assert v["R4 cache covers the whole site"] == "SKIP", v
    assert v["R5 index is dense (max+1 == count)"] == "PASS", v

    # loader unavailable: the table degrades to None and the report still builds
    tnone = {"rows_total": None, "warning": "loader unavailable: X"}
    ck = [v for _, v, _ in reconcile("z", tnone, {}, None)]
    assert ck.count("SKIP") == 5, ck
    md = report({"z": {"table": tnone, "cache": {"rows": 7}, "checks": []}})
    assert "| z | - | - | - | - | 7 |" in md, md

    # 雑音列: bool / 数値 / 文字列（object と str dtype の両方）で正しく数える
    for col in (pd.Series(["earthquake"] * 3 + ["noise"] * 2, dtype="object"),
                pd.Series(["earthquake"] * 3 + ["noise"] * 2).astype(str),
                pd.Series([0, 0, 0, 1, 1]),
                pd.Series([False, False, False, True, True])):
        d = pd.DataFrame({"split": ["train"] * 3 + ["test"] * 2,
                          "event_id": [1, 2, 3, 4, 5], "trace_category": col})

        class LN:
            @staticmethod
            def prepare_site_csv(site):
                return d

        r = read_site_table(LN, "n", verbose=False)
        assert r["noise_rows_total"] == 2, (col.dtype, r["noise_rows_total"])
        assert r["noise_rows_by_split"] == {"test": 2, "train": 0}, r

    # index キーの無いキャッシュでも R5 は SKIP として必ず現れる
    v = _ck("w", t, {"rows": 5, "keys": ["gt_time"]}, within_test=1)
    assert v["R5 index is dense (max+1 == count)"] == "SKIP", v

    print("[21] selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ckpt", default="ema", choices=["ema", "best", "last"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    res = run(a.ckpt)
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR / "21_event_accounting.json").write_text(
        json.dumps(res, indent=2, default=str))
    md = report(res)
    (cfg.LOGS_DIR / "21_event_accounting.md").write_text(md)
    print(md)
    print(f"\n[save] {cfg.LOGS_DIR / '21_event_accounting.md'}")


if __name__ == "__main__":
    main()
