#!/usr/bin/env python
"""
26_train_split_overlap.py -- GJI revision round 2 (A-1): the Section 5.2
moveout-support numbers on the population the model was actually trained on.

WHERE THE MANUSCRIPT'S NUMBERS COME FROM
----------------------------------------
Every figure in the moveout-support sentences of the manuscript is in
logs/16_moveout_overlap_ema.json, written by 16_arrival_audit.py --overlap:

    manuscript                                  16_moveout_overlap_ema.json
    "only 9 of 2,788 training events"           n_train_ge_median 9, n_train 2788
    "(0.32 per cent)"                            frac_ge_median 0.00323
    "202 training events ... (8.20 per cent)"    202 / 2463, 0.08201
    "192 of them (95 per cent) ... forge_19"     top_supplier forge_19, 192
    "whose maximum is 104 ms"                    train_max_ms 103.5
    "pnr-1 ... has a maximum of 68 ms"           held_max_ms 67.5

So they reproduce. What script 16 calls the "training set" is, by its own
docstring, "every clean earthquake event of the other seven sites": all splits
(train, dev AND test) of those sites, minus the arrival-audit failures. The
denominators follow exactly from Table S2: the other seven sites' 3,208
earthquakes minus their 420 flagged events is 2,788, and 2,878 - 415 is 2,463.

THE INCONSISTENCY THAT REMAINS
------------------------------
The model is not trained on that population. 01_amber_setup.build_dataloaders
reads only split=='train' (and 'dev' for model selection), so the LOSO training
set is the train split of the other seven sites: 1,527 to 2,002 earthquakes,
which is what supplement S3 reports from script 21. Two consequences:

  * the other sites' dev and test events are counted as "training events"
    although the model never trained on them;
  * pnr-1 carries no train or dev rows at all (AMBER marks all 30,192 pnr-1
    rows split='test'), so pnr-1 contributes NOTHING to any fold's training
    set. "pnr-1, the only other steep site, has a maximum of 68 ms" describes
    pnr-1's catalogue, not the forge_19 fold's training data: in that fold the
    steep-site support is not small, it is absent.

This script therefore recomputes the same quantities, with the same moveout
function and the same audit mask, on three nested populations:

    P16    clean earthquakes, all splits        -- reproduces script 16 exactly
    B      clean earthquakes, split == train    -- what the model trained on
    B+     clean earthquakes, split in train/dev

and, for the record, RAW = the train split before the audit, which is the
population 26_moveout_support.py counted and which the unphysical 500-900 ms
"moveouts" of mis-associated picks contaminate.

HOW IT AVOIDS THE HDF5
----------------------
The script-10 caches hold gt_time (event, phase, station) in dataset order,
and the dataset order is deterministic: AMBER enumerates events with
pandas groupby("event_id"), i.e. event_id ascending, dropping events with fewer
than N_STATION rows (amber/dataloaders.py:153-171). The map from cache index
to event_id, and from there to the CSV split column, is rebuilt from
metadata.csv alone and verified two ways: the number of events must equal the
cache length, and the per-event noise flag rebuilt from trace_category must
equal the cache's is_noise array element for element.

INPUTS (all already in the archive or on the Drive)
--------------------------------------------------
    logs/10_cache_<site>_ema_loso_<site>.npz     gt_time, is_noise
    logs/16_arrival_audit_ema.json               clean_eq_ids per site
    logs/16_moveout_overlap_ema.json             the reference to reproduce
    data/metadata.csv                            event_id, split, dataset,
                                                 trace_category

USAGE
-----
    python src/26_train_split_overlap.py --all
    python src/26_train_split_overlap.py --selftest
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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the moveout function, copied from 16_arrival_audit._event_moveouts so the
# self-test runs without the sibling; main() asserts the two are identical
# ---------------------------------------------------------------------------
def event_moveouts(gt: np.ndarray, events, fs: float) -> np.ndarray:
    """P moveout in ms for each event with at least two picked stations."""
    out = []
    for e in events:
        v = gt[e, 0][gt[e, 0] >= 0]
        if len(v) >= 2:
            out.append((v.max() - v.min()) / fs * 1000.0)
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# cache index -> event_id -> split, from the CSV alone
# ---------------------------------------------------------------------------
def index_map(meta_site, n_station: int):
    """Replicate AMBER's enumeration (dataloaders.py:153-171).

    Returns (event_ids in dataset order, is_noise per event, split per event).
    """
    ids, noise, split = [], [], []
    for eid, df in meta_site.groupby("event_id"):        # sorted by default
        if len(df) < n_station:
            continue
        ids.append(eid)
        noise.append(str(df["trace_category"].iloc[0]) == "noise")
        split.append(str(df["split"].iloc[0]))
    return ids, np.asarray(noise, bool), np.asarray(split, object)


def verify_map(site: str, cache: dict, noise_from_csv: np.ndarray) -> None:
    n_cache = int(cache["gt_time"].shape[0])
    assert len(noise_from_csv) == n_cache, (
        f"{site}: CSV から復元したイベント数 {len(noise_from_csv)} が "
        f"キャッシュの {n_cache} と違う。並び順の前提が崩れている")
    mism = int((noise_from_csv != cache["is_noise"].astype(bool)).sum())
    assert mism == 0, (
        f"{site}: is_noise が {mism} 件一致しない。index -> event_id の対応が "
        "ずれている")


# ---------------------------------------------------------------------------
# the overlap computation, generalised over a population mask
# ---------------------------------------------------------------------------
def population(clean_ids, split_of, allowed_splits) -> list:
    """Clean earthquake indices restricted to the given splits."""
    if allowed_splits is None:
        return list(clean_ids)
    return [e for e in clean_ids if split_of[e] in allowed_splits]


def overlap(mo: dict, held: str, threshold: float) -> dict:
    """n, k, share, max and the supplier breakdown for one held-out site.

    mo: {site: moveout array of the population}; the held-out site's own
    entry is only used to size the report, never counted.
    """
    train_parts = {s: mo[s] for s in mo if s != held}
    train = (np.concatenate([v for v in train_parts.values()])
             if train_parts else np.empty(0))
    sup = {s: int((v >= threshold).sum()) for s, v in train_parts.items()
           if v.size}
    sup = dict(sorted(sup.items(), key=lambda kv: -kv[1]))
    n_ge = int(sum(sup.values()))
    top = next(iter(sup.items()), (None, 0))
    f_med = (n_ge / train.size) if train.size else float("nan")
    if train.size == 0:
        word = "no training events"
    elif f_med == 0.0:
        word = "outside the training range"
    elif f_med < 0.02:
        word = "in the extreme tail of the training range"
    elif f_med < 0.10:
        word = "under-represented in training"
    else:
        word = "well represented in training"
    return dict(heldout=held, threshold_ms=float(threshold),
                n_train=int(train.size),
                n_train_ge_median=n_ge,
                frac_ge_median=float(f_med),
                train_max_ms=float(train.max()) if train.size else None,
                supply_by_site=sup,
                top_supplier=top[0], top_supplier_events=int(top[1]),
                top_supplier_share=(top[1] / n_ge) if n_ge else 0.0,
                n_by_site={s: int(v.size) for s, v in train_parts.items()},
                wording=word)


def fmt_pct(x):
    return "nan" if x != x else f"{100 * x:.2f}%"


def print_block(title: str, rows: dict, ref: dict | None = None) -> None:
    print(f"\n[26] {title}")
    print(f"  {'held out':<16s} {'閾値':>6s} {'n':>6s} {'k':>5s} {'%':>8s} "
          f"{'train max':>9s}  {'largest supplier':<28s} {'wording':<42s}"
          + ("  vs 16" if ref else ""))
    for h, d in rows.items():
        top = (f"{d['top_supplier']} ({d['top_supplier_events']}, "
               f"{100 * d['top_supplier_share']:.0f}%)"
               if d["top_supplier"] else "-")
        mx = f"{d['train_max_ms']:.1f}" if d["train_max_ms"] is not None else "-"
        line = (f"  {h:<16s} {d['threshold_ms']:>6.1f} {d['n_train']:>6d} "
                f"{d['n_train_ge_median']:>5d} {fmt_pct(d['frac_ge_median']):>8s} "
                f"{mx:>9s}  {top:<28s} {d['wording']:<42s}")
        if ref is not None:
            r = ref[h]
            same = (d["n_train"] == r["n_train"]
                    and d["n_train_ge_median"] == r["n_train_ge_median"]
                    and abs(d["train_max_ms"] - r["train_max_ms"]) < 1e-6
                    and d["top_supplier"] == r["top_supplier"]
                    and d["top_supplier_events"] == r["top_supplier_events"])
            line += "  " + ("一致" if same else
                            f"不一致 (16: {r['n_train_ge_median']}/{r['n_train']})")
        print(line)


# ---------------------------------------------------------------------------
def selftest() -> int:
    import pandas as pd
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"  {name}: {'PASS' if cond else 'FAIL'}")

    fs, nst = 2000.0, 3

    # -- index_map replicates AMBER's enumeration --------------------------
    rows = []
    for eid, cat, split, n in (("ev_b", "earthquake", "train", 3),
                               ("ev_a", "earthquake", "test", 3),
                               ("ev_c", "noise", "dev", 3),
                               ("ev_d", "earthquake", "train", 2)):   # too few
        rows += [dict(event_id=eid, trace_category=cat, split=split,
                      dataset="x")] * n
    meta = pd.DataFrame(rows)
    ids, noise, split = index_map(meta, nst)
    check("index_map: event_id 昇順で並ぶ", ids == ["ev_a", "ev_b", "ev_c"])
    check("index_map: 局数不足のイベントは落ちる", "ev_d" not in ids)
    check("index_map: noise フラグが trace_category から出る",
          noise.tolist() == [False, False, True])
    check("index_map: split が元の CSV の値", split.tolist() == ["test", "train", "dev"])

    # -- verify_map ----------------------------------------------------------
    cache = {"gt_time": np.zeros((3, 2, nst), int),
             "is_noise": np.array([False, False, True])}
    try:
        verify_map("x", cache, noise)
        check("verify_map: 件数と is_noise が合えば通る", True)
    except AssertionError:
        check("verify_map: 件数と is_noise が合えば通る", False)
    bad = {"gt_time": np.zeros((3, 2, nst), int),
           "is_noise": np.array([False, True, True])}
    try:
        verify_map("x", bad, noise)
        check("verify_map: is_noise のずれを捕まえる", False)
    except AssertionError:
        check("verify_map: is_noise のずれを捕まえる", True)

    # -- event_moveouts ------------------------------------------------------
    gt = np.full((3, 2, nst), -1, int)
    gt[0, 0] = [100, 110, 120]          # 20 samples -> 10 ms
    gt[1, 0] = [100, -1, 300]           # 200 samples -> 100 ms
    gt[2, 0] = [100, -1, -1]            # one pick -> dropped
    mo = event_moveouts(gt, [0, 1, 2], fs)
    check("event_moveouts: 欠測 (-1) を除いて max-min", mo.tolist() == [10.0, 100.0])

    # -- population ----------------------------------------------------------
    split_of = np.array(["train", "test", "dev"], object)
    check("population: None で全 clean", population([0, 1, 2], split_of, None) == [0, 1, 2])
    check("population: train だけ", population([0, 1, 2], split_of, {"train"}) == [0])
    check("population: train+dev", population([0, 1, 2], split_of, {"train", "dev"}) == [0, 2])

    # -- overlap -------------------------------------------------------------
    mo = {"held": np.array([70.0, 72.0]),
          "a": np.array([10.0, 80.0, 90.0]),
          "b": np.array([5.0]),
          "none": np.empty(0)}
    d = overlap(mo, "held", 70.0)
    check("overlap: 自サイトは数えない", d["n_train"] == 4)
    check("overlap: 閾値以上は a の 2 件", d["n_train_ge_median"] == 2 and
          d["supply_by_site"] == {"a": 2, "b": 0})
    check("overlap: 最大供給元と割合", d["top_supplier"] == "a" and
          abs(d["top_supplier_share"] - 1.0) < 1e-9)
    check("overlap: 学習最大", d["train_max_ms"] == 90.0)
    check("overlap: 空のサイトは n_by_site 0", d["n_by_site"]["none"] == 0)
    check("overlap: wording は 16 と同じ閾値 (2/4 = 50 % -> well represented)",
          d["wording"] == "well represented in training")
    d0 = overlap({"held": np.array([70.0]), "a": np.array([1.0, 2.0])}, "held", 70.0)
    check("overlap: 0 件なら outside", d0["wording"] == "outside the training range")
    d1 = overlap({"held": np.array([70.0]),
                  "a": np.concatenate([np.ones(99), [80.0]])}, "held", 70.0)
    check("overlap: 1/100 = 1 % -> extreme tail", d1["wording"] ==
          "in the extreme tail of the training range")

    # -- the arithmetic that locates the manuscript population --------------
    eq = dict(pnr_1=543, mseel_3h=684, mseel_5h=512, clearfield_mw6=612,
              pnr_2=305, clearfield_mw4=254, aneth=298, forge_19=213)
    fl = dict(pnr_1=6, mseel_3h=107, mseel_5h=90, clearfield_mw6=128,
              pnr_2=18, clearfield_mw4=50, aneth=21, forge_19=1)
    T, F = sum(eq.values()), sum(fl.values())
    check("Table S2: 他7サイト全件 − Flagged = 2,788 (forge_19)",
          (T - eq["forge_19"]) - (F - fl["forge_19"]) == 2788)
    check("Table S2: 他7サイト全件 − Flagged = 2,463 (pnr-1)",
          (T - eq["pnr_1"]) - (F - fl["pnr_1"]) == 2463)

    print(f"[selftest] {'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", default="ema")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.all:
        ap.error("pass --all (or --selftest)")

    import pandas as pd

    cfg = _load("00_config_l3.py")
    audit = _load("16_arrival_audit.py")
    logs = Path(cfg.LOGS_DIR)
    fs = float(cfg.DATA.FS)
    nst = int(cfg.DATA.N_STATION)

    # the moveout function must be the published one
    gt_probe = np.array([[[100, 130, -1, 160]], [[1, -1, -1, 2]]])
    assert np.allclose(event_moveouts(gt_probe, [0, 1], fs),
                       audit._event_moveouts(gt_probe, [0, 1], fs)), \
        "event_moveouts が 16_arrival_audit._event_moveouts と違う"

    caches = audit.load_caches(args.ckpt)
    res = json.loads((logs / f"16_arrival_audit_{args.ckpt}.json").read_text())
    ref = json.loads((logs / f"16_moveout_overlap_{args.ckpt}.json").read_text())
    sites = [s for s in audit.SITE_ORDER if s in caches and s in res]
    assert len(sites) == 8, sites
    print(f"[26] サイト {sites}")
    print("[26] HDF5 も AMBER も GPU も使わない。10 のキャッシュ、16 の監査結果、"
          "metadata.csv だけ")

    # -- index -> event_id -> split, verified against the cache -------------
    meta = pd.read_csv(cfg.AMBER_CSV,
                       usecols=["event_id", "split", "dataset", "trace_category"])
    split_of, n_by_split = {}, {}
    print("\n[26] 0 -- キャッシュの並び順を CSV から復元して照合")
    for s in sites:
        ids, noise, split = index_map(meta[meta["dataset"] == s], nst)
        verify_map(s, caches[s], noise)
        split_of[s] = split
        clean = res[s]["clean_eq_ids"]
        n_by_split[s] = {k: int(sum(1 for e in clean if split[e] == k))
                         for k in ("train", "dev", "test")}
        print(f"  {s:<16s} キャッシュ {len(ids):>5d} 件 = CSV 復元  OK   "
              f"clean 地震 {len(clean):>4d} = "
              f"train {n_by_split[s]['train']:>4d} + dev {n_by_split[s]['dev']:>3d}"
              f" + test {n_by_split[s]['test']:>4d}")

    # -- populations ---------------------------------------------------------
    pops = {
        "P16  clean, all splits  (= script 16 = manuscript)": None,
        "B    clean, train only  (= what the model trained on)": {"train"},
        "B+   clean, train + dev": {"train", "dev"},
    }
    thr = {s: float(ref[s]["held_median_ms"]) for s in sites}
    result = {"threshold_ms": thr, "clean_by_split": n_by_split,
              "populations": {}}

    for label, allowed in pops.items():
        mo = {s: event_moveouts(caches[s]["gt_time"],
                                population(res[s]["clean_eq_ids"], split_of[s],
                                           allowed), fs)
              for s in sites}
        rows = {h: overlap(mo, h, thr[h]) for h in sites}
        result["populations"][label] = rows
        if allowed is None:
            print_block(label, rows, ref)
            bad = [h for h in sites if
                   rows[h]["n_train"] != ref[h]["n_train"] or
                   rows[h]["n_train_ge_median"] != ref[h]["n_train_ge_median"]]
            assert not bad, (
                f"P16 が 16_moveout_overlap を再現しない: {bad}。index の対応か "
                "監査マスクが違う。以下の数値は使わない")
            print("  -> 8 サイトすべて script 16 と一致。回帰ゲート通過")
        else:
            print_block(label, rows)

    # RAW, for the record: the pre-audit train split, i.e. what 26 counted
    mo_raw = {}
    for s in sites:
        idx = [e for e in range(caches[s]["gt_time"].shape[0])
               if not caches[s]["is_noise"][e] and split_of[s][e] == "train"]
        mo_raw[s] = event_moveouts(caches[s]["gt_time"], idx, fs)
    raw_rows = {h: overlap(mo_raw, h, thr[h]) for h in sites}
    result["populations"]["RAW  train only, before the audit"] = raw_rows
    print_block("RAW  train only, BEFORE the audit (26 が数えた母集団。参考)",
                raw_rows)

    # -- the manuscript sentences, P16 against B ----------------------------
    P = result["populations"]["P16  clean, all splits  (= script 16 = manuscript)"]
    B = result["populations"]["B    clean, train only  (= what the model trained on)"]
    print("\n[26] 本文の文 -- 16 の母集団 (P16) と学習母集団 (B)")
    for h, text in (("forge_19", "only 9 of 2,788 training events reach it "
                                 "(0.32 per cent)"),
                    ("pnr-1", "202 training events reach its median ... "
                              "192 of them (95 per cent) come from forge_19")):
        p, b = P[h], B[h]
        print(f"\n  held out {h}   本文: \"{text}\"")
        for name, d in (("P16", p), ("B  ", b)):
            print(f"    {name}: {d['n_train_ge_median']} / {d['n_train']} = "
                  f"{fmt_pct(d['frac_ge_median'])}, largest supplier "
                  f"{d['top_supplier']} ({d['top_supplier_events']}, "
                  f"{100 * d['top_supplier_share']:.0f}%), train max "
                  f"{d['train_max_ms']:.1f} ms, \"{d['wording']}\"")
    print(f"\n  forge_19 の fold で pnr-1 が供給する学習イベント: "
          f"P16 {P['forge_19']['n_by_site']['pnr-1']} 件 / "
          f"B {B['forge_19']['n_by_site']['pnr-1']} 件")
    ratio_p = P["pnr-1"]["frac_ge_median"] / P["forge_19"]["frac_ge_median"]
    ratio_b = (B["pnr-1"]["frac_ge_median"] / B["forge_19"]["frac_ge_median"]
               if B["forge_19"]["frac_ge_median"] else float("inf"))
    print(f"  pnr-1 と forge_19 の支持率の比 (本文 \"factor of twenty-six\"): "
          f"P16 {ratio_p:.1f} 倍 / B {ratio_b:.1f} 倍")
    print(f"  §1 の 1 % 基準: forge_19  P16 {fmt_pct(P['forge_19']['frac_ge_median'])} "
          f"/ B {fmt_pct(B['forge_19']['frac_ge_median'])};  "
          f"pnr-1  P16 {fmt_pct(P['pnr-1']['frac_ge_median'])} / "
          f"B {fmt_pct(B['pnr-1']['frac_ge_median'])}")

    out = logs / f"26_train_split_overlap_{args.ckpt}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[26] {out.name} に書いた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
