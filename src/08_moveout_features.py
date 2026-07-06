#!/usr/bin/env python
"""
08_moveout_features.py  --  GJI revision Tier-1: quantify the geometry that the
array model overfits.

Rather than rely on station coordinates (schema varies across AMBER sites), we
read a purely data-driven proxy directly from the labels: the ACROSS-STATION
MOVEOUT of each event = (max - min) of the P (and S) arrival sample over the
stations that have a pick, in milliseconds. Its per-site distribution
characterises how arrivals sweep across the borehole array; if a held-out site's
moveout distribution sits outside the training sites', that is concrete evidence
of out-of-distribution geometry (the mechanism behind the forge_19 P-collapse).

Output (logs/): 08_moveout_<site>.json  with median / IQR of P and S moveout.
Aggregate all sites, then plot Delta(array-per-trace) vs moveout in 05_i.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(path):
    spec = importlib.util.spec_from_file_location(path[:-3].replace("-", "_"), HERE / path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


cfg = _load("00_config_l3.py")
DATA = cfg.DATA


def event_moveout_ms(label_ch, thr=0.5):
    """label_ch: (n_station, T). Return across-station moveout (ms) using stations
    that carry a pick, or None if fewer than 2 picked stations."""
    idx = []
    for st in range(label_ch.shape[0]):
        tr = label_ch[st]
        if tr.max() > thr:
            idx.append(int(tr.argmax()))
    if len(idx) < 2:
        return None
    return (max(idx) - min(idx)) / DATA.FS * 1000.0


def site_features(site):
    loader_mod = _load("01_amber_setup.py")
    csv = loader_mod.prepare_site_csv(site, all_test=True)
    ds = loader_mod.build_amber_dataset("test", csv)
    mv = {"P": [], "S": []}
    for i in range(len(ds)):
        _, lab = ds[i]
        lab = lab.numpy() if hasattr(lab, "numpy") else np.asarray(lab)
        for ch, name in ((0, "P"), (1, "S")):
            m = event_moveout_ms(lab[ch])
            if m is not None:
                mv[name].append(m)

    def stats(a):
        a = np.asarray(a, float)
        if a.size == 0:
            return dict(median_ms=float("nan"), iqr_ms=float("nan"),
                        q25=float("nan"), q75=float("nan"), n=0)
        q25, q75 = np.percentile(a, [25, 75])
        return dict(median_ms=float(np.median(a)), iqr_ms=float(q75 - q25),
                    q25=float(q25), q75=float(q75), n=int(a.size))
    return {"site": site, "P": stats(mv["P"]), "S": stats(mv["S"])}


def selftest():
    # craft a label with a known 10-sample (=10/FS*1000 ms) P spread over 3 stations
    T, S = 256, 12
    lab = np.zeros((3, S, T), np.float32)
    lab[0, 0, 100] = 1; lab[0, 1, 105] = 1; lab[0, 2, 110] = 1   # P spread 10 samp
    exp = 10 / DATA.FS * 1000.0
    got = event_moveout_ms(lab[0])
    ok = abs(got - exp) < 1e-6
    print(f"  moveout from labels: got {got:.3f} ms, expected {exp:.3f} ms: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="single site; omit to do all USABLE_SITES")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    sites = [args.site] if args.site else list(DATA.USABLE_SITES)
    for site in sites:
        feat = site_features(site)
        out = cfg.LOGS_DIR / f"08_moveout_{site}.json"
        json.dump(feat, open(out, "w"), indent=2)
        print(f"[moveout] {site}: P median {feat['P']['median_ms']:.2f} ms "
              f"(IQR {feat['P']['iqr_ms']:.2f}, n={feat['P']['n']}) | "
              f"S median {feat['S']['median_ms']:.2f} ms  -> {out.name}")


if __name__ == "__main__":
    raise SystemExit(main())
