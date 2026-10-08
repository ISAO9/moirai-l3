#!/usr/bin/env python
"""
16_arrival_audit.py -- GJI major revision round 2: are the catalogued arrivals
in AMBER physically consistent, and what does that do to Table 1?
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
Script 17 recomputed the FULL moveout and S-P distributions that the reviewers
asked for (minimum and maximum, not just a median and an IQR). The medians
reproduce the submitted Table 1 and the project configuration's own note for
mseel_3h ("S-P is tiny, median 110 samples" = 55 ms, recomputed 54 ms), so the
computation is sound. The TAILS are not:

  * P moveout maxima of 554-909 ms across strings 259-335 m long. Treating the
    whole analysed aperture as the station separation, that implies apparent
    velocities of 370-470 m/s. An apparent velocity along a receiver line is
    always at least the medium velocity, which is kilometres per second here,
    so these are not body-wave arrivals.
  * S-P times down to -900 ms. S cannot precede P.
  * forge_19 is the only site whose P moveout range (41-82 ms) is physical
    throughout.

The cause matters, because it decides what may be printed in Table 1 and what
has to be said to Reviewer 2, who asked specifically about label quality in
AMBER and cites Aguilar Suarez & Beroza (2025) on pervasive label errors in
seismological machine-learning datasets.

Two candidate causes were considered and ONE was already ruled out by hand:
label clipping at the window edge cannot be responsible, because the labeller's
taper half-width is 30 samples (15 ms), so an arrival pinned to the window edge
is displaced by at most 15 ms, not by hundreds. An arrival entirely outside the
window leaves the channel below the 0.5 label threshold and is recorded as
absent. What remains is a catalogue problem: picks that belong to a different
event, or to no arrival at all, associated to this event's station.

WHAT THIS SCRIPT DOES
---------------------
  1. PHYSICS   -- per event, the apparent velocity implied by the P (and S)
     moveout, using the station separation of the earliest and latest picked
     stations. Events implying less than V_MIN are flagged as unphysical.
  2. ORDER     -- per event, the number of reversals in the depth-ordered
     arrival sequence. A point source seen by a near-linear string gives a
     monotonic or single-turn sequence; many reversals mean scattered picks.
  3. S-P       -- stations whose catalogued S precedes their catalogued P.
  4. CAUSE     -- where the flagged arrivals sit inside the analysis window.
     Concentration at the first or last samples would point at a windowing
     artefact; a spread points at the catalogue. This is the test that
     separates the two explanations, and it is reported as a number.
  5. EFFECT    -- every distribution recomputed with the flagged events
     removed, so Table 1 can quote ranges that are defensible.
  6. OVERLAP (--overlap) -- what the moveout claim in Section 5.2 may say. The
     submitted text calls the forge_19 P moveout "beyond the training range",
     which is a statement about the training DISTRIBUTION and so has to be
     measured against it, on clean labels. For each held-out site the LOSO
     training set is reassembled from the other seven sites' clean events and
     the share of it lying inside the held-out range, and at or above the
     held-out median, is reported. The last column gives the strongest wording
     those two numbers license, and the paper should use no stronger one.
  7. SURVIVAL (--rescore) -- whether the paper's site-level verdicts survive
     the cleaning. The script-10 caches hold per-event predictions, so the
     clean subset is scored by dropping rows; no retraining is needed. Scoring,
     bootstrap and verdict are imported from script 11, and the all-events
     column is checked automatically against the archived fixed-threshold
     scores, so a wrong mask would show up as a failure to reproduce the
     submitted numbers rather than as a plausible-looking new table.

NOTE ON THE REVERSAL TEST
  Counting turns in the depth-ordered arrival sequence only has power where the
  station-to-station step exceeds the pick quantisation. At aneth the median P
  moveout is about one sample per gap, so integer pick times alone generate
  reversals in sound data. The test therefore ignores sub-resolution steps and
  is disabled entirely, and reported as "n/a", at a site whose median gap step
  is below REVERSAL_MIN_STEP samples.

OUTPUTS
  logs/16_arrival_audit_<ckpt>.json   per-site counts, before/after statistics
  logs/16_arrival_audit_<ckpt>.md     the same as a table to paste
  logs/16_moveout_overlap_<ckpt>.md   training-set overlap per held-out site
  logs/16_clean_rescore_<ckpt>.md     verdicts on all vs clean events
  PDF/16_a_arrival_audit.pdf          apparent velocity, window position,
                                      and the S-P distribution per site

All figures: white background, English labels, legends clear of the data.

RUN (no GPU, no AMBER, no network: the script-10 caches and the station
coordinates written by script 17 are enough)
  python src/16_arrival_audit.py --all            # audit + overlap + rescore
  python src/16_arrival_audit.py --audit
  python src/16_arrival_audit.py --overlap        # (b) only; implies --audit
  python src/16_arrival_audit.py --rescore        # (c) only; implies --audit
  python src/16_arrival_audit.py --audit --v-min 2000
  # self-test (synthetic, nothing external required):
  python src/16_arrival_audit.py --selftest
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

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6", "pnr-2",
              "clearfield_mw4", "aneth", "forge_19"]

V_MIN_DEFAULT = 1500.0      # m/s; below any plausible body-wave velocity here
EDGE_SAMPLES = 60           # "near the window edge" = twice the taper width


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def load_caches(ckpt: str = "ema") -> dict:
    """{site: stats} -- catalogued arrivals are identical across configurations."""
    caches = {}
    for f in sorted(cfg.LOGS_DIR.glob(f"10_cache_*_{ckpt}*loso_*.npz")):
        stem = f.stem
        if "_pertrace" in stem or "_shuf" in stem:
            continue
        caches[stem.split("_loso_")[-1]] = dict(np.load(f))
    return caches


def load_spacing() -> dict:
    """{site: median adjacent station spacing in metres} from script 17.

    The analysis uses 12 depth-ordered stations; at pnr-1 (24 stations on the
    string) and aneth (18) the analysed subset is a contiguous run whose exact
    members the cache does not record. Both strings are regularly spaced, so
    the separation of two analysed stations is taken as the median adjacent
    spacing times their index difference. This is the quantity the physics
    check needs, and it is stated in the output.
    """
    f = cfg.LOGS_DIR / "17_array_coords.json"
    if not f.exists():
        return {}
    out = {}
    for site, d in json.loads(f.read_text()).items():
        xyz = np.asarray(d["xyz"], float)
        if len(xyz) >= 2:
            out[site] = float(np.median(
                np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    return out


# ----------------------------------------------------------------------------
# per-event checks
# ----------------------------------------------------------------------------
REVERSAL_MIN_STEP = 3.0     # samples; below this the reversal test has no power


def _reversals(t: np.ndarray, min_step: float) -> int:
    """Turns in a depth-ordered arrival sequence, ignoring sub-resolution steps.

    A point source seen by a near-linear string gives a monotonic or
    single-turn sequence, so many turns suggest scattered picks. The test only
    has power where the step between adjacent stations is larger than the pick
    quantisation. At aneth the median P moveout is 6 ms over eleven gaps, about
    one sample per gap, so integer pick times alone produce reversals in
    perfectly sound data and counting them measures pick PRECISION rather than
    pick VALIDITY. Steps smaller than min_step are therefore ignored, and the
    caller disables the test altogether for a site that cannot support it.
    """
    d = np.diff(t.astype(float))
    d = d[np.abs(d) >= min_step]
    if d.size < 2:
        return 0
    return int((np.diff(np.sign(d)) != 0).sum())


def _median_gap_step(gt_ch: np.ndarray, events: np.ndarray) -> float:
    """Median per-station arrival step in samples, over events with >=2 picks."""
    steps = []
    for e in events:
        v = gt_ch[e][gt_ch[e] >= 0]
        if len(v) >= 2:
            steps.append((v.max() - v.min()) / (len(v) - 1))
    return float(np.median(steps)) if steps else 0.0


def audit_site(stats: dict, spacing_m: float, fs: float, window: int,
               v_min: float) -> dict:
    """Flag events whose catalogued arrivals cannot be body-wave arrivals.

    The test is deliberately conservative: it flags only what is impossible at
    the chosen velocity floor, so an event contaminated mildly enough to stay
    above the floor survives. The surviving range is therefore an upper bound
    on what the catalogue supports, not a certificate that every remaining
    pick is right -- which is the honest way to quote it in a table.
    """
    gt = stats["gt_time"]                                   # (E,2,S)
    eq = np.where(~stats["is_noise"])[0]
    n_st = gt.shape[2]

    gap_step = _median_gap_step(gt[:, 0], eq)
    rev_ok = gap_step >= REVERSAL_MIN_STEP
    rec = dict(n_events=int(len(eq)), n_stations=int(n_st),
               spacing_m=spacing_m, v_min=v_min, window_samples=int(window),
               median_gap_step_samples=gap_step,
               reversal_test_applied=bool(rev_ok))
    flags = {k: [] for k in ("slow_P", "slow_S", "reversals", "sp_le_zero")}
    v_app = {"P": [], "S": []}
    edge_dist_flagged, edge_dist_clean = [], []
    reversal_counts = []
    sp_all, sp_neg = [], 0
    n_sp_pairs = 0

    for e in eq:
        ev_flag = False
        for ch, name in ((0, "P"), (1, "S")):
            idx = np.where(gt[e, ch] >= 0)[0]
            if len(idx) < 2:
                continue
            t = gt[e, ch, idx]
            i_min, i_max = int(idx[int(np.argmin(t))]), int(idx[int(np.argmax(t))])
            dt = (t.max() - t.min()) / fs                    # seconds
            sep = spacing_m * abs(i_max - i_min)             # metres
            v = sep / dt if dt > 0 else np.inf
            v_app[name].append(v)
            # The window position is recorded for the EXTREME arrivals only --
            # the pair that sets the moveout, one of which is the suspect when
            # the moveout is unphysical. Pooling every arrival of a flagged
            # event instead would dilute one bad pick among two dozen good
            # ones and hide the very signal this test is for.
            ext = np.array([t.min(), t.max()], float)
            d_ext = np.minimum(ext, window - 1 - ext).tolist()
            if dt > 0 and v < v_min:
                flags[f"slow_{name}"].append(int(e))
                ev_flag = True
                edge_dist_flagged.extend(d_ext)
            else:
                edge_dist_clean.extend(d_ext)
            if name == "P" and rev_ok:
                r = _reversals(gt[e, ch, np.sort(idx)], 0.5 * gap_step)
                reversal_counts.append(r)
                if r > 2:
                    flags["reversals"].append(int(e))
                    ev_flag = True

        both = (gt[e, 0] >= 0) & (gt[e, 1] >= 0)
        if both.any():
            sp = (gt[e, 1, both] - gt[e, 0, both]) / fs * 1000.0
            sp_all.extend(sp.tolist())
            n_sp_pairs += int(both.sum())
            if (sp <= 0).any():
                flags["sp_le_zero"].append(int(e))
                sp_neg += int((sp <= 0).sum())
                ev_flag = True
                bad = np.where(both)[0][sp <= 0]
                off = np.concatenate([gt[e, 0, bad], gt[e, 1, bad]]).astype(float)
                edge_dist_flagged.extend(
                    np.minimum(off, window - 1 - off).tolist())

    flagged = sorted({e for v in flags.values() for e in v})
    rec["flagged_events"] = len(flagged)
    rec["flagged_fraction"] = len(flagged) / max(len(eq), 1)
    rec["by_criterion"] = {k: len(set(v)) for k, v in flags.items()}
    rec["n_sp_pairs"] = n_sp_pairs
    rec["n_sp_le_zero"] = sp_neg
    rec["median_reversals"] = (float(np.median(reversal_counts))
                               if reversal_counts else float("nan"))
    for name in ("P", "S"):
        a = np.asarray(v_app[name], float)
        a = a[np.isfinite(a)]
        rec[f"v_apparent_{name}"] = (
            dict(n=int(a.size), p05=float(np.percentile(a, 5)),
                 median=float(np.median(a)), p95=float(np.percentile(a, 95)))
            if a.size else dict(n=0))

    # the cause test: window position of flagged vs clean arrivals
    def edge_summary(a):
        a = np.asarray(a, float)
        if a.size == 0:
            return dict(n=0)
        return dict(n=int(a.size),
                    median_samples_from_edge=float(np.median(a)),
                    frac_within_edge_band=float((a < EDGE_SAMPLES).mean()))
    rec["window_position_flagged"] = edge_summary(edge_dist_flagged)
    rec["window_position_clean"] = edge_summary(edge_dist_clean)

    # effect: the distributions with and without the flagged events
    keep = np.array([e for e in eq if e not in set(flagged)], dtype=int)
    rec["before"] = _distributions(gt, eq, fs)
    rec["after"] = _distributions(gt, keep, fs)
    # the event indices themselves, so --overlap and --rescore apply exactly the
    # same flags as the table rather than recomputing them under other defaults
    rec["flagged_ids"] = sorted(int(e) for e in set(flagged))
    rec["clean_eq_ids"] = [int(e) for e in keep]
    return rec


def _distributions(gt: np.ndarray, events: np.ndarray, fs: float) -> dict:
    out = {}
    for ch, name in ((0, "P"), (1, "S")):
        mo = []
        for e in events:
            v = gt[e, ch][gt[e, ch] >= 0]
            if len(v) >= 2:
                mo.append((v.max() - v.min()) / fs * 1000.0)
        out[name] = _q(np.asarray(mo, float))
    sp = []
    for e in events:
        both = (gt[e, 0] >= 0) & (gt[e, 1] >= 0)
        if both.any():
            sp.extend(((gt[e, 1, both] - gt[e, 0, both]) / fs * 1000.0).tolist())
    out["SP"] = _q(np.asarray(sp, float))
    return out


def _q(a: np.ndarray) -> dict:
    if a.size == 0:
        return dict(n=0)
    p = np.percentile(a, [0, 25, 50, 75, 100])
    return dict(n=int(a.size), min_ms=float(p[0]), q25_ms=float(p[1]),
                median_ms=float(p[2]), q75_ms=float(p[3]), max_ms=float(p[4]))



# ----------------------------------------------------------------------------
# (b) moveout overlap: is the held-out site's moveout really unseen?
# ----------------------------------------------------------------------------
def _ordered(res: dict, have) -> list:
    """SITE_ORDER first, then any other site present, so nothing is dropped."""
    known = [s for s in SITE_ORDER if s in res and s in have]
    rest = sorted(s for s in res if s in have and s not in SITE_ORDER)
    return known + rest


def _event_moveouts(gt: np.ndarray, events, fs: float) -> np.ndarray:
    """P moveout in ms for each event with at least two picked stations."""
    out = []
    for e in events:
        v = gt[e, 0][gt[e, 0] >= 0]
        if len(v) >= 2:
            out.append((v.max() - v.min()) / fs * 1000.0)
    return np.asarray(out, float)


def overlap_table(res: dict, caches: dict, fs: float) -> tuple[dict, str]:
    """How much of each LOSO training set lies in the held-out site's range.

    The submitted Section 5.2 says the forge_19 P moveout lies "beyond the
    training range". That is a claim about the TRAINING DISTRIBUTION, so it has
    to be measured against it, and after label cleaning rather than before --
    an unphysical 900 ms arrival is not evidence that the model has seen steep
    moveout. Under LOSO the training set for a held-out site is every clean
    earthquake event of the other seven sites, which is exactly what is
    assembled here. Three numbers are reported per site:

      train_max_ms      the steepest moveout the model actually sees
      frac_in_range     training events inside the held-out site's [min, max]
      frac_ge_median    training events at or above the held-out site's median

    frac_ge_median is the one that bears on the claim: it is the share of
    training data in the half of the held-out distribution that matters, so a
    small value supports "under-represented" and a value near zero supports
    "outside". The phrase printed in the last column is the wording the
    numbers license, and nothing stronger.
    """
    sites = _ordered(res, caches)
    mo = {s: _event_moveouts(caches[s]["gt_time"], res[s]["clean_eq_ids"], fs)
          for s in sites}

    def _supply(held: str, thr_ms: float) -> dict:
        """Which training sites supply the events at or above thr_ms.

        Under LOSO the training set is the other seven sites, so a held-out
        site whose moveout is rare in training may still be supported by ONE
        other site that happens to resemble it. Naming that site is what turns
        "rare in training" into a mechanism, and it is the difference between
        asserting the asymmetry between forge_19 and pnr-1 and measuring it.
        """
        out = {}
        for o in sites:
            if o == held:
                continue
            v = mo[o]
            if v.size:
                out[o] = int((v >= thr_ms).sum())
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
    out = {}
    for s in sites:
        held = mo[s]
        train = np.concatenate([mo[o] for o in sites if o != s]) \
            if len(sites) > 1 else np.empty(0)
        if held.size == 0 or train.size == 0:
            continue
        lo, hi, med = held.min(), held.max(), float(np.median(held))
        f_rng = float(((train >= lo) & (train <= hi)).mean())
        f_med = float((train >= med).mean())
        if f_med == 0.0:
            word = "outside the training range"
        elif f_med < 0.02:
            word = "in the extreme tail of the training range"
        elif f_med < 0.10:
            word = "under-represented in training"
        else:
            word = "well represented in training"
        sup = _supply(s, med)
        n_ge = sum(sup.values())
        top = next(iter(sup.items()), (None, 0))
        out[s] = dict(n_held=int(held.size), held_min_ms=float(lo),
                      held_median_ms=med, held_max_ms=float(hi),
                      n_train=int(train.size), train_max_ms=float(train.max()),
                      train_median_ms=float(np.median(train)),
                      frac_in_range=f_rng, frac_ge_median=f_med,
                      n_train_ge_median=n_ge,
                      supply_by_site=sup,
                      top_supplier=top[0], top_supplier_events=top[1],
                      top_supplier_share=(top[1] / n_ge) if n_ge else 0.0,
                      wording=word)
    md = ["| Held-out site | clean events | P moveout min/med/max (ms) | "
          "training events | training max (ms) | % of training in site range | "
          "% of training >= site median | events >= median | largest supplier | "
          "supported wording |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        if s not in out:
            continue
        d = out[s]
        md.append(
            f"| {s} | {d['n_held']} | {d['held_min_ms']:.0f} / "
            f"{d['held_median_ms']:.0f} / {d['held_max_ms']:.0f} | "
            f"{d['n_train']} | {d['train_max_ms']:.0f} | "
            f"{100*d['frac_in_range']:.1f}% | {100*d['frac_ge_median']:.2f}% | "
            f"{d['n_train_ge_median']} | "
            f"{d['top_supplier'] or '-'} "
            f"({d['top_supplier_events']}, {100*d['top_supplier_share']:.0f}%) | "
            f"{d['wording']} |")
    return out, "\n".join(md)


# ----------------------------------------------------------------------------
# (c) re-score every LOSO verdict on the clean events only
# ----------------------------------------------------------------------------
def _published_f1(site: str, method: str, ckpt: str):
    """F1-mean at the fixed 0.30 threshold as archived by script 10, or None.

    This is the number printed in the submitted Table 3, so it is the reference
    the "all events" column of the re-scoring MUST reproduce. Checking it
    automatically is what makes the "clean" column trustworthy: if the mask or
    the subset indexing were wrong, the unmasked column would move too.
    """
    f = cfg.LOGS_DIR / f"10_threshold_sweep_{site}_{method}_{ckpt}.json"
    if not f.exists():
        return None
    try:
        return float(json.loads(f.read_text())["at_fixed_030"]["f1_mean"])
    except (KeyError, ValueError, TypeError):
        return None


def rescore_table(res: dict, ckpt: str, nboot: int, spacing: dict,
                  fs: float, window: int, v_min: float) -> tuple[dict, str]:
    """Do the paper's verdicts survive removal of the unphysical events?

    No retraining: the script-10 caches hold the per-event predictions, so the
    clean subset is scored by dropping rows. Scoring, bootstrap and verdict are
    IMPORTED from script 11 rather than reimplemented, so the "all events"
    column must reproduce the submitted Table 3 exactly and any difference in
    the "clean" column is the label cleaning and nothing else.

    Noise windows carry no picks and are therefore never flagged; they are kept
    in full, which keeps the false-alarm denominator and the noise-event share
    of each bootstrap replicate unchanged.

    WHY THE TWO CONFIGURATIONS ARE AUDITED SEPARATELY
    The array and per-trace caches were written by two independent passes over
    the dataset, and AMBER draws the analysis window -- and, where a string
    carries more than N_STATION sensors, which contiguous run of sensors is
    used -- afresh on each pass. Row i is therefore the SAME EVENT in both
    caches (is_noise matches exactly at every site, and that is asserted below)
    but not the same realisation of it, so the catalogued arrivals differ. An
    event may consequently be unphysical in one realisation and sound in the
    other. Transferring one cache's flags to the other by row index would
    silently mix the two cases, so each cache is audited against its OWN
    labels and the UNION of the two flag sets is dropped from both. Both
    configurations are then scored on an identical set of events, and every
    retained event has physically consistent labels in both realisations.
    """
    fa = _load("11_fa_inclusive_scoring.py")
    tol = int(round(EVALU.PRIMARY_TOLERANCE_S * DATA.FS))
    thr = EVALU.PEAK_PROB_THRESHOLD
    caches = fa.ts._load_caches(ckpt)
    pairs = {s for (s, m) in caches
             if (s, "array") in caches and (s, "pertrace") in caches}
    sites = _ordered(res, pairs)
    out = {}
    for s in sites:
        sa, sp = caches[(s, "array")], caches[(s, "pertrace")]
        # row correspondence: same events, same order. is_noise is a property
        # of the event and not of the realisation, so it must match exactly
        if sa["gt_time"].shape != sp["gt_time"].shape or \
                not np.array_equal(sa["is_noise"], sp["is_noise"]):
            print(f"[rescore] {s}: the two caches do not describe the same "
                  f"events in the same order, skipped")
            continue
        E = sa["gt_time"].shape[0]
        flag_a = set(res[s]["flagged_ids"])
        if s in spacing:
            flag_p = set(audit_site(sp, spacing[s], fs, window,
                                    v_min)["flagged_ids"])
        else:
            flag_p = set()
        bad = sorted(flag_a | flag_p)
        mask = np.ones(E, bool)
        if bad:
            mask[np.asarray(bad, int)] = False
        keep = np.where(mask)[0]
        ca, cp = fa._subset(sa, keep), fa._subset(sp, keep)

        row = {}
        for tag, (xa, xp) in (("all", (sa, sp)), ("clean", (ca, cp))):
            ra, rp = fa.score_both(xa, thr, tol), fa.score_both(xp, thr, tol)
            cis = fa.bootstrap(xa, xp, thr, tol, nboot)
            row[tag] = dict(
                n_events=int(xa["gt_time"].shape[0]),
                array=ra["orig"]["mean"], pertrace=rp["orig"]["mean"],
                array_fa=ra["fa_incl"]["mean"], pertrace_fa=rp["fa_incl"]["mean"],
                ci=cis,
                verdict=fa.verdict(cis["array"]["orig"], cis["pertrace"]["orig"]),
                verdict_fa=fa.verdict(cis["array"]["fa_incl"],
                                      cis["pertrace"]["fa_incl"]))
        row["verdict_survives"] = bool(
            row["all"]["verdict"] == row["clean"]["verdict"])
        row["verdict_fa_survives"] = bool(
            row["all"]["verdict_fa"] == row["clean"]["verdict_fa"])
        row["n_dropped"] = int(E - len(keep))
        row["n_flagged_array"] = len(flag_a)
        row["n_flagged_pertrace"] = len(flag_p)
        row["n_flagged_both"] = len(flag_a & flag_p)

        # consistency guard against the archived submission numbers
        chk = {}
        for method, key in (("array", "array"), ("pertrace", "pertrace")):
            ref = _published_f1(s, method, ckpt)
            if ref is None:
                continue
            got = row["all"][key]
            chk[method] = dict(published=ref, recomputed=got,
                               agrees=bool(abs(ref - got) < 1e-6))
            if not chk[method]["agrees"]:
                print(f"[rescore] WARNING {s} {method}: all-events F1-mean "
                      f"{got:.6f} does not reproduce the archived "
                      f"{ref:.6f} -- do not use these numbers until the "
                      f"discrepancy is understood")
        row["reproduces_submitted"] = chk
        out[s] = row
        flip = "" if row["verdict_survives"] else "   <-- VERDICT CHANGES"
        print(f"[rescore] {s:<16s} dropped {row['n_dropped']:>4d}  "
              f"all: arr {row['all']['array']:.3f} / pt "
              f"{row['all']['pertrace']:.3f} [{row['all']['verdict']}]  ->  "
              f"clean: arr {row['clean']['array']:.3f} / pt "
              f"{row['clean']['pertrace']:.3f} "
              f"[{row['clean']['verdict']}]{flip}")

    md = ["| Site | flagged array / per-trace / both | events dropped | "
          "array F1-mean all -> clean | "
          "per-trace F1-mean all -> clean | verdict all | verdict clean | "
          "survives | verdict clean (FA-incl) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        if s not in out:
            continue
        d = out[s]
        md.append(
            f"| {s} | {d['n_flagged_array']} / {d['n_flagged_pertrace']} / "
            f"{d['n_flagged_both']} | {d['n_dropped']} | {d['all']['array']:.3f} -> "
            f"{d['clean']['array']:.3f} | {d['all']['pertrace']:.3f} -> "
            f"{d['clean']['pertrace']:.3f} | {d['all']['verdict']} | "
            f"{d['clean']['verdict']} | "
            f"{'yes' if d['verdict_survives'] else 'NO'} | "
            f"{d['clean']['verdict_fa']} |")
    n_ok = sum(d["verdict_survives"] for d in out.values())
    md += ["", f"{n_ok} of {len(out)} site-level verdicts are unchanged after "
               f"removing the unphysical events."]
    chks = [c["agrees"] for d in out.values()
            for c in d.get("reproduces_submitted", {}).values()]
    if chks:
        md.append(f"The all-events column reproduces the archived fixed-threshold "
                  f"scores in {sum(chks)} of {len(chks)} site x configuration "
                  f"checks" + ("." if all(chks) else " -- SEE WARNINGS ABOVE."))
    return out, "\n".join(md)


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def write_report(res: dict, ckpt: str) -> str:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.LOGS_DIR / f"16_arrival_audit_{ckpt}.json").write_text(
        json.dumps(res, indent=2))

    md = ["| Site | Events | Flagged | slow P | slow S | reversals | S-P<=0 | "
          "gap step (samples) | P moveout min/med/max before (ms) | after (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SITE_ORDER:
        if s not in res:
            continue
        r = res[s]
        b, a = r["before"]["P"], r["after"]["P"]
        c = r["by_criterion"]
        md.append(
            f"| {s} | {r['n_events']} | {r['flagged_events']} "
            f"({100*r['flagged_fraction']:.0f}%) | {c['slow_P']} | "
            f"{c['slow_S']} | "
            f"{c['reversals'] if r['reversal_test_applied'] else 'n/a'} | "
            f"{c['sp_le_zero']} | {r['median_gap_step_samples']:.1f} | "
            f"{b['min_ms']:.0f} / {b['median_ms']:.0f} / {b['max_ms']:.0f} | "
            f"{a.get('min_ms', float('nan')):.0f} / "
            f"{a.get('median_ms', float('nan')):.0f} / "
            f"{a.get('max_ms', float('nan')):.0f} |")

    md += ["", "| Site | flagged arrivals: median samples from window edge | "
               "within edge band | clean arrivals: median | within band |",
           "|---|---|---|---|---|"]
    for s in SITE_ORDER:
        if s not in res:
            continue
        f_, c_ = res[s]["window_position_flagged"], res[s]["window_position_clean"]
        md.append(
            f"| {s} | "
            f"{f_.get('median_samples_from_edge', float('nan')):.0f} | "
            f"{f_.get('frac_within_edge_band', float('nan')):.3f} | "
            f"{c_.get('median_samples_from_edge', float('nan')):.0f} | "
            f"{c_.get('frac_within_edge_band', float('nan')):.3f} |")

    txt = "\n".join(md) + "\n"
    (cfg.LOGS_DIR / f"16_arrival_audit_{ckpt}.md").write_text(txt)
    return txt


def verdict(res: dict) -> str:
    """State which cause the window-position numbers support.

    The two extreme arrivals of a flagged event are recorded, one of which is
    the suspect pick. If the suspect were pinned to a window edge the flagged
    fraction would approach one half; if it sits anywhere in the window the
    expected fraction is only the edge band's share of the window, about
    2 * EDGE_SAMPLES / window, a few per cent. The two hypotheses are an order
    of magnitude apart, so the test is read off the ratio with a floor on the
    absolute level rather than from a knife-edge threshold.
    """
    fl = [r["window_position_flagged"].get("frac_within_edge_band")
          for r in res.values() if r["window_position_flagged"].get("n")]
    cl = [r["window_position_clean"].get("frac_within_edge_band")
          for r in res.values() if r["window_position_clean"].get("n")]
    if not fl:
        return "[verdict] nothing flagged; the catalogued arrivals are consistent."
    f_m, c_m = float(np.mean(fl)), float(np.mean(cl or [0.0]))
    chance = 2.0 * EDGE_SAMPLES / float(DATA.WINDOW_SAMPLES)
    if f_m > 0.25 and f_m > 4.0 * max(c_m, chance):
        return (f"[verdict] the suspect arrivals sit at the window edges "
                f"({100*f_m:.0f}% within {EDGE_SAMPLES} samples, against "
                f"{100*c_m:.0f}% of sound ones and {100*chance:.0f}% expected "
                f"by chance): a WINDOWING artefact.")
    return (f"[verdict] the suspect arrivals are spread through the window "
            f"({100*f_m:.0f}% near an edge, against {100*c_m:.0f}% of sound "
            f"ones and {100*chance:.0f}% expected by chance): the picks "
            f"themselves are inconsistent, i.e. a CATALOGUE problem, not a "
            f"windowing artefact.")


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def make_figure(res: dict, caches: dict, spacing: dict, fs: float,
                window: int, v_min: float, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    sites = [s for s in SITE_ORDER if s in res]
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.6))

    # (a) apparent velocity implied by the P moveout
    ax = axes[0]
    for i, s in enumerate(sites):
        v = res[s]["v_apparent_P"]
        if not v.get("n"):
            continue
        ax.plot([i, i], [v["p05"], v["p95"]], "-", color="0.45", lw=1.0)
        ax.plot(i, v["median"], "o", color="#1f77b4", ms=5)
    ax.axhline(v_min, color="#d62728", ls="--", lw=1.2,
               label=f"physical floor ({v_min:.0f} m/s)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("apparent velocity from P moveout (m/s)")
    ax.set_title("Arrivals below the floor cannot be body waves "
                 "(dot: median, bar: 5th-95th percentile)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="0.8")

    # (b) the cause test: window position of flagged vs clean arrivals
    ax = axes[1]
    w = 0.38
    xs = np.arange(len(sites))
    for off, key, c, lab in ((-w/2, "window_position_clean", "0.55", "clean"),
                             (w/2, "window_position_flagged", "#d62728",
                              "flagged")):
        vals = [res[s][key].get("frac_within_edge_band", np.nan) for s in sites]
        ax.bar(xs + off, vals, width=w, color=c, alpha=0.75, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(sites, rotation=25, ha="right", fontsize=8)
    chance = 2.0 * EDGE_SAMPLES / float(DATA.WINDOW_SAMPLES)
    ax.axhline(chance, color="0.3", ls=":", lw=1.1,
               label=f"expected by chance ({100*chance:.0f}%)")
    ax.set_ylim(0.0, max(0.6, chance * 2))
    ax.set_ylabel(f"fraction within {EDGE_SAMPLES} samples\nof a window edge")
    ax.set_title("A windowing artefact would pile the suspect arrivals up at "
                 "the edges; a catalogue error would not", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="0.8")

    # (c) S-P distribution per site, negative half shaded
    ax = axes[2]
    for i, s in enumerate(sites):
        st = caches[s]
        gt = st["gt_time"]
        eq = np.where(~st["is_noise"])[0]
        sp = []
        for e in eq:
            b = (gt[e, 0] >= 0) & (gt[e, 1] >= 0)
            if b.any():
                sp.extend(((gt[e, 1, b] - gt[e, 0, b]) / fs * 1000.0).tolist())
        if not sp:
            continue
        sp = np.asarray(sp)
        q = np.percentile(sp, [5, 50, 95])
        ax.plot([i, i], [q[0], q[2]], "-", color="0.45", lw=1.0)
        ax.plot(i, q[1], "o", color="#1f77b4", ms=5)
        if (sp <= 0).any():                       # mark the impossible tail
            ax.plot([i], [float(sp.min())], "v", color="#d62728", ms=6)
    ax.axhspan(-1100, 0, color="#d62728", alpha=0.08)
    ax.axhline(0.0, color="#d62728", lw=1.0)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("S–P time (ms)")
    ax.set_title("S cannot precede P: any mass below zero is a label error",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Physical audit of the catalogued arrivals", fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.84, top=0.93, bottom=0.07, hspace=0.62)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf} (+ .png)")


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
def _synth(n_ev=200, n_st=12, window=2048, fs=2000.0, spacing=30.5,
           vp=4000.0, contaminate=0.0, edge_mode=False, seed=0) -> dict:
    """Clean events, optionally contaminated in one of two distinct ways.

    contaminate : fraction of events given one mis-associated station pick
    edge_mode   : place the bad pick AT a window edge instead of at random
    """
    rng = np.random.default_rng(seed)
    gt = np.full((n_ev, 2, n_st), -1, np.int64)
    is_noise = np.zeros(n_ev, bool)
    is_noise[: n_ev // 10] = True
    step = spacing / vp * fs                                # samples per station
    for e in range(n_ev):
        if is_noise[e]:
            continue
        base = int(rng.integers(500, 900))
        for s in range(n_st):
            gt[e, 0, s] = int(base + step * s)
            gt[e, 1, s] = int(base + 220 + 1.7 * step * s)
        if contaminate and rng.random() < contaminate:
            s_bad = int(rng.integers(0, n_st))
            if edge_mode:
                bad = 0 if rng.random() < 0.5 else window - 1
            else:
                bad = int(rng.integers(0, window))
            gt[e, 0, s_bad] = bad
    return dict(gt_time=gt, pr_max=np.ones((n_ev, 2, n_st), np.float32),
                pr_time=gt.copy(), is_noise=is_noise)


def selftest():
    fs, window, spacing, v_min = 2000.0, 2048, 30.5, V_MIN_DEFAULT

    clean = audit_site(_synth(), spacing, fs, window, v_min)
    assert clean["flagged_events"] == 0, clean["by_criterion"]
    assert clean["before"]["P"]["max_ms"] == clean["after"]["P"]["max_ms"]
    assert clean["v_apparent_P"]["median"] > v_min

    # catalogue-style contamination: bad picks anywhere in the window
    cat = audit_site(_synth(contaminate=0.3, seed=1), spacing, fs, window, v_min)
    assert cat["flagged_events"] > 0, cat
    assert cat["before"]["P"]["max_ms"] > cat["after"]["P"]["max_ms"], cat
    # Whatever survives must be physical at the floor, measured against the
    # most generous separation available (the whole analysed aperture).
    aperture_m = spacing * (12 - 1)
    max_allowed_ms = aperture_m / v_min * 1000.0
    assert cat["after"]["P"]["max_ms"] <= max_allowed_ms, \
        (cat["after"]["P"]["max_ms"], max_allowed_ms)
    assert clean["before"]["P"]["max_ms"] <= max_allowed_ms
    assert cat["window_position_flagged"]["frac_within_edge_band"] < 0.5, cat

    # windowing-style contamination: bad picks pinned to the edges
    edge = audit_site(_synth(contaminate=0.3, edge_mode=True, seed=2),
                      spacing, fs, window, v_min)
    assert edge["flagged_events"] > 0
    assert edge["window_position_flagged"]["frac_within_edge_band"] > \
        cat["window_position_flagged"]["frac_within_edge_band"], \
        (edge["window_position_flagged"], cat["window_position_flagged"])

    # the verdict must distinguish the two
    assert "CATALOGUE" in verdict({"a": cat}), verdict({"a": cat})
    assert "WINDOWING" in verdict({"a": edge}), verdict({"a": edge})

    # negative S-P is caught
    bad = _synth(seed=3)
    bad["gt_time"][50, 1, 3] = bad["gt_time"][50, 0, 3] - 100
    r = audit_site(bad, spacing, fs, window, v_min)
    assert r["n_sp_le_zero"] >= 1 and r["by_criterion"]["sp_le_zero"] >= 1

    res = {"forge_19": cat, "pnr-1": clean}
    txt = write_report(res, "selftest")
    assert "forge_19" in txt and "window edge" in txt
    out = cfg.PDF_DIR / "16_selftest_audit.pdf"
    make_figure(res, {"forge_19": _synth(contaminate=0.3, seed=1),
                      "pnr-1": _synth()}, {}, fs, window, v_min, out)
    assert out.exists() and out.stat().st_size > 8_000

    # (b) the overlap table must be computable and self-consistent: a site's
    # own range must contain its own median, and the fractions must be in [0,1]
    sites_syn = {"forge_19": cat, "pnr-1": clean}
    caches_syn = {"forge_19": _synth(contaminate=0.3, seed=1), "pnr-1": _synth()}
    ov, md_ov = overlap_table(sites_syn, caches_syn, fs)
    assert set(ov) == {"forge_19", "pnr-1"}, ov
    for s, d in ov.items():
        assert d["held_min_ms"] <= d["held_median_ms"] <= d["held_max_ms"], d
        assert 0.0 <= d["frac_in_range"] <= 1.0 and 0.0 <= d["frac_ge_median"] <= 1.0
        assert d["n_train"] > 0
    assert "supported wording" in md_ov
    for s_, d in ov.items():
        assert sum(d["supply_by_site"].values()) == d["n_train_ge_median"], d
        assert s_ not in d["supply_by_site"], "held-out site must not supply itself"
        if d["n_train_ge_median"]:
            assert d["top_supplier"] in d["supply_by_site"]
            assert 0.0 < d["top_supplier_share"] <= 1.0
    # the two synthetic sites are drawn from one distribution, so neither can be
    # reported as outside the other's range
    assert all(d["wording"] == "well represented in training"
               for d in ov.values()), ov
    # and a site shifted to steeper moveout than the rest must come out as
    # outside: this is the behaviour the forge_19 claim depends on
    steep = _synth(seed=4)
    gt_s = steep["gt_time"]
    for e in range(gt_s.shape[0]):                  # triple the P moveout
        v = gt_s[e, 0]
        ok = v >= 0
        if ok.sum() >= 2:
            gt_s[e, 0, ok] = v[ok].min() + (v[ok] - v[ok].min()) * 3
    r_steep = audit_site(steep, spacing, fs, window, 300.0)   # keep all events
    ov2, _ = overlap_table({"steep": r_steep, "pnr-1": clean},
                           {"steep": steep, "pnr-1": _synth()}, fs)
    assert ov2["steep"]["frac_ge_median"] < ov2["pnr-1"]["frac_ge_median"], ov2
    assert ov2["steep"]["wording"] != "well represented in training", ov2

    print("[selftest] ALL PASS")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--audit", action="store_true",
                    help="physical audit of the catalogued arrivals")
    ap.add_argument("--overlap", action="store_true",
                    help="(b) moveout overlap with each LOSO training set")
    ap.add_argument("--rescore", action="store_true",
                    help="(c) re-score every LOSO verdict on clean events only")
    ap.add_argument("--all", action="store_true", help="audit + overlap + rescore")
    ap.add_argument("--nboot", type=int, default=300,
                    help="bootstrap replicates for --rescore (default 300)")
    ap.add_argument("--ckpt", default="ema", choices=["best", "ema", "last"])
    ap.add_argument("--v-min", type=float, default=V_MIN_DEFAULT,
                    help="apparent-velocity floor in m/s (default 1500)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.all:
        a.audit = a.overlap = a.rescore = True
    if not (a.audit or a.overlap or a.rescore):
        ap.print_help()
        return
    # the overlap and re-scoring modes consume the audit's flags, so the audit
    # always runs first and with the same settings
    a.audit = True

    fs = float(getattr(DATA, "FS", 2000.0))
    window = int(getattr(DATA, "WINDOW_SAMPLES", 2048))
    caches = load_caches(a.ckpt)
    if not caches:
        raise SystemExit(f"[audit] no 10_cache_*_{a.ckpt}*loso_*.npz in "
                         f"{cfg.LOGS_DIR} -- build them with script 10 --cache")
    spacing = load_spacing()
    if not spacing:
        raise SystemExit(f"[audit] {cfg.LOGS_DIR/'17_array_coords.json'} not "
                         f"found -- run script 17 --geometry first")

    res = {}
    for site, st in sorted(caches.items()):
        if site not in spacing:
            print(f"[audit] {site}: no station spacing, skipped")
            continue
        res[site] = audit_site(st, spacing[site], fs, window, a.v_min)
        r = res[site]
        print(f"[audit] {site:<16s} {r['flagged_events']:>4d}/{r['n_events']:<4d} "
              f"flagged ({100*r['flagged_fraction']:>4.1f}%), "
              f"S-P<=0 on {r['n_sp_le_zero']} station picks, "
              f"P moveout max {r['before']['P']['max_ms']:.0f} -> "
              f"{r['after']['P'].get('max_ms', float('nan')):.0f} ms")
    print(write_report(res, a.ckpt))
    print(verdict(res))
    make_figure(res, caches, spacing, fs, window, a.v_min,
                cfg.PDF_DIR / "16_a_arrival_audit.pdf")

    if a.overlap:
        print("\n=== (b) moveout overlap with the LOSO training set ===")
        ov, md = overlap_table(res, caches, fs)
        (cfg.LOGS_DIR / f"16_moveout_overlap_{a.ckpt}.md").write_text(md)
        (cfg.LOGS_DIR / f"16_moveout_overlap_{a.ckpt}.json").write_text(
            json.dumps(ov, indent=2))
        print(md)
        print(f"[save] {cfg.LOGS_DIR / f'16_moveout_overlap_{a.ckpt}.md'}")

    if a.rescore:
        print("\n=== (c) LOSO verdicts re-scored on clean events only ===")
        rs, md = rescore_table(res, a.ckpt, a.nboot, spacing, fs,
                               window, a.v_min)
        (cfg.LOGS_DIR / f"16_clean_rescore_{a.ckpt}.md").write_text(md)
        (cfg.LOGS_DIR / f"16_clean_rescore_{a.ckpt}.json").write_text(
            json.dumps(rs, indent=2))
        print(md)
        print(f"[save] {cfg.LOGS_DIR / f'16_clean_rescore_{a.ckpt}.md'}")


if __name__ == "__main__":
    main()
