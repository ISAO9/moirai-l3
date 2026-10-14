#!/usr/bin/env python3
"""
29_fig8_paired.py -- Fig. 8 of GJI-26-0613 (second revision) drawn from the
PAIRED array-minus-per-trace difference and the AUDITED moveout medians.

WHY THIS SCRIPT EXISTS
----------------------
Figure 8 (array advantage versus site median P moveout) was produced by
05_figures_l3.py from two quantities that the second revision has since
replaced in the tables:

  * its y-axis was Delta = F1-mean(array) - F1-mean(per-trace) computed from
    the per-phase F1 of the standard evaluation (the per-phase columns of
    Table 2), whereas the Delta column of Table 2, its 95 per cent CI and the
    verdicts now come from the joint event bootstrap of script 27
    (single evaluation pass; e.g. pnr-2 +0.040 against +0.055);
  * its x-axis was the moveout median of logs/08_moveout_<site>.json (before
    the arrival audit), whereas Table 1 quotes the medians AFTER the audit of
    Section S1 (logs/16_arrival_audit_ema.json; e.g. mseel_5h 26.75 against
    27.5 ms).

A figure that shows one Delta while the table beside it lists another is a
fair reviewer complaint.  This script therefore draws Fig. 8 from exactly the
numbers the tables print: Delta, its paired CI and the paired verdict from
logs/27_paired_inference.json (false alarms included, the convention of
Table 2), and the audited P-moveout median from logs/16_arrival_audit_ema.json.

WHAT IT DOES
------------
  1. Reads the two logs above; refuses to run if a site is missing in either.
  2. Regression gate (on by default): the plotted Delta must reproduce the
     Table 2 values of the manuscript to three decimals and the plotted
     medians must reproduce Table 1.  If they do not, the logs on disk are not
     the logs the paper was written from, and the script stops.
  3. Draws PDF/29_a_fig8_paired.pdf and .png: one point per site, a vertical
     bar for the paired 95 per cent CI, colour by the Table 2 verdict (array /
     per-trace / equivalent / tie), the +/-0.02 equivalence band shaded, white
     background, labels in English, legend below the axes.
  4. Writes logs/29_fig8_points.json with the plotted values so that every
     number in the figure can be read back without the figure.

USAGE
-----
  python src/29_fig8_paired.py              # draws the figure, gate on
  python src/29_fig8_paired.py --base DIR   # read DIR/logs, write DIR/PDF and DIR/logs
                                            # (default: the project root of 00_config_l3,
                                            #  i.e. the Drive folder on Colab)
  python src/29_fig8_paired.py --no-gate    # draw whatever the logs hold
  python src/29_fig8_paired.py --selftest   # synthetic logs, no data needed
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
PAIRED_JSON = "27_paired_inference.json"
AUDIT_JSON = "16_arrival_audit_ema.json"
POINTS_JSON = "29_fig8_points.json"
FIG_NAME = "29_a_fig8_paired"
CONVENTION = "fa_incl"            # Table 2 counts noise-window picks as false positives
EQUIV_BOUND = 0.02

SITES = ["pnr-1", "mseel_5h", "clearfield_mw6", "mseel_3h",
         "pnr-2", "clearfield_mw4", "aneth", "forge_19"]

# Table 2 (Delta, paired) and Table 1 (P moveout median, ms) of the manuscript.
# These are the gate, not the input: the figure is drawn from the logs.
TABLE2_DELTA = {"pnr-1": -0.001, "mseel_5h": +0.017, "clearfield_mw6": +0.009,
                "mseel_3h": +0.005, "pnr-2": +0.055, "clearfield_mw4": +0.078,
                "aneth": -0.057, "forge_19": -0.325}
TABLE2_VERDICT = {"pnr-1": "equivalent", "mseel_5h": "array", "clearfield_mw6": "tie",
                  "mseel_3h": "equivalent", "pnr-2": "array", "clearfield_mw4": "array",
                  "aneth": "per-trace", "forge_19": "per-trace"}
TABLE1_P_MEDIAN = {"pnr-1": 62.0, "mseel_3h": 31.5, "mseel_5h": 27.5,
                   "clearfield_mw6": 24.0, "pnr-2": 35.0, "clearfield_mw4": 24.0,
                   "aneth": 6.0, "forge_19": 70.0}

COL = {"array": "#1f6feb", "per-trace": "#d1495b", "equivalent": "#8A8A8A", "tie": "#8A8A8A"}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def verdict_label(rec: dict) -> str:
    """Table 2 wording from a script-27 record: array / per-trace / equivalent / tie."""
    v = rec["verdict_paired"]
    if v == "different":
        return "array" if rec["delta"] > 0 else "per-trace"
    if v == "equivalent":
        return "equivalent"
    return "tie"


def load_points(logs_dir: Path) -> list[dict]:
    paired = json.loads((logs_dir / PAIRED_JSON).read_text())
    audit = json.loads((logs_dir / AUDIT_JSON).read_text())
    pts = []
    for s in SITES:
        if s not in paired["sites"]:
            raise SystemExit(f"{PAIRED_JSON}: site {s} missing")
        if s not in audit:
            raise SystemExit(f"{AUDIT_JSON}: site {s} missing")
        rec = paired["sites"][s][CONVENTION]
        pts.append(dict(site=s,
                        moveout_ms=float(audit[s]["after"]["P"]["median_ms"]),
                        delta=float(rec["delta"]),
                        ci=[float(rec["delta_ci"][0]), float(rec["delta_ci"][1])],
                        verdict=verdict_label(rec)))
    return pts


def gate(pts: list[dict]) -> None:
    bad = []
    for p in pts:
        if round(p["delta"], 3) != TABLE2_DELTA[p["site"]]:
            bad.append(f"{p['site']}: Delta {p['delta']:+.3f} != Table 2 {TABLE2_DELTA[p['site']]:+.3f}")
        if p["verdict"] != TABLE2_VERDICT[p["site"]]:
            bad.append(f"{p['site']}: verdict {p['verdict']} != Table 2 {TABLE2_VERDICT[p['site']]}")
        if abs(p["moveout_ms"] - TABLE1_P_MEDIAN[p["site"]]) > 1e-9:
            bad.append(f"{p['site']}: median {p['moveout_ms']} != Table 1 {TABLE1_P_MEDIAN[p['site']]}")
    if bad:
        raise SystemExit("regression gate failed:\n  " + "\n  ".join(bad))
    print(f"[gate] {len(pts)} sites: Delta, verdict and median reproduce Tables 1 and 2")


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def draw(pts: list[dict], pdf_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.0), facecolor="white")
    ax.axhspan(-EQUIV_BOUND, EQUIV_BOUND, color="#EFEFEF", zorder=0)
    ax.axhline(0, color="#999999", lw=0.9, zorder=1)
    for p in pts:
        c = COL[p["verdict"]]
        ax.plot([p["moveout_ms"]] * 2, p["ci"], color=c, lw=1.4, zorder=2, solid_capstyle="butt")
        ax.scatter(p["moveout_ms"], p["delta"], s=78, color=c, zorder=3,
                   edgecolor="white", linewidth=0.6,
                   marker="o" if p["verdict"] != "tie" else "D")
    # label offsets chosen so that no label overlaps a point or another label
    off = {"clearfield_mw6": (-8, -16), "mseel_5h": (-9, 5), "clearfield_mw4": (8, 4),
           "mseel_3h": (8, -16), "pnr-2": (8, 4), "pnr-1": (8, 5),
           "aneth": (9, 3), "forge_19": (-9, 10)}
    for p in pts:
        dx, dy = off.get(p["site"], (6, 5))
        ax.annotate(p["site"], (p["moveout_ms"], p["delta"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="#333333",
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("P across-station moveout, site median after the arrival audit (ms)")
    ax.set_ylabel("Δ F1-mean  (array − per-trace), paired")
    ax.set_title("Array advantage peaks at intermediate, in-distribution moveout")
    ax.set_xlim(0, 75)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["array"], markersize=8,
               label="array (CI above zero)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["per-trace"], markersize=8,
               label="per-trace (CI below zero)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["equivalent"], markersize=8,
               label="equivalent (CI inside ±0.02)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COL["tie"], markersize=7,
               label="tie (CI straddles zero, exceeds ±0.02)"),
        Line2D([0], [0], color="#555555", lw=1.4, label="paired 95 per cent CI (B = 2,000)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              frameon=False, ncol=3, fontsize=8)
    fig.subplots_adjust(bottom=0.30)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out = pdf_dir / f"{FIG_NAME}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_dir / f"{FIG_NAME}.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"[fig] saved {out} (+ .png)")
    return out


def run(logs_dir: Path, pdf_dir: Path, use_gate: bool) -> list[dict]:
    pts = load_points(logs_dir)
    if use_gate:
        gate(pts)
    draw(pts, pdf_dir)
    (logs_dir / POINTS_JSON).write_text(json.dumps(
        dict(source={"delta": PAIRED_JSON, "moveout": AUDIT_JSON},
             convention=CONVENTION, equivalence_bound=EQUIV_BOUND, points=pts), indent=1))
    print(f"[log] wrote {logs_dir / POINTS_JSON}")
    for p in pts:
        print(f"  {p['site']:15s} moveout {p['moveout_ms']:5.1f} ms  Delta {p['delta']:+.3f} "
              f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  {p['verdict']}")
    return pts


# ----------------------------------------------------------------------------
# self-test on synthetic logs (no data needed)
# ----------------------------------------------------------------------------
def selftest() -> None:
    print("=" * 70)
    print("29_fig8_paired -- self-test on synthetic logs")
    print("=" * 70)
    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as td:
        logs, pdf = Path(td) / "logs", Path(td) / "PDF"
        logs.mkdir(); pdf.mkdir()
        sites = {}
        for s in SITES:
            d = TABLE2_DELTA[s]
            v = TABLE2_VERDICT[s]
            if v in ("array", "per-trace"):
                ci, vp = [d - 0.01, d + 0.01], "different"
            elif v == "equivalent":
                ci, vp = [d - 0.005, d + 0.005], "equivalent"
            else:
                ci, vp = [d - 0.012, d + 0.012], "tie (inconclusive)"
            sites[s] = {CONVENTION: dict(delta=d, delta_ci=ci, verdict_paired=vp)}
        (logs / PAIRED_JSON).write_text(json.dumps(dict(sites=sites)))
        (logs / AUDIT_JSON).write_text(json.dumps(
            {s: {"after": {"P": {"median_ms": TABLE1_P_MEDIAN[s]}}} for s in SITES}))
        pts = run(logs, pdf, use_gate=True)
        ck("eight points drawn", len(pts) == 8)
        ck("gate passes on values equal to Tables 1 and 2", True)
        ck("pdf and png written", (pdf / f"{FIG_NAME}.pdf").exists() and (pdf / f"{FIG_NAME}.png").exists())
        ck("points json written", (logs / POINTS_JSON).exists())
        # the gate must fail when one Delta is off by 0.002
        sites["pnr-2"][CONVENTION]["delta"] = TABLE2_DELTA["pnr-2"] + 0.002
        (logs / PAIRED_JSON).write_text(json.dumps(dict(sites=sites)))
        try:
            run(logs, pdf, use_gate=True)
            ck("gate rejects a Delta that differs from Table 2", False)
        except SystemExit as e:
            ck("gate rejects a Delta that differs from Table 2", "regression gate failed" in str(e))
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f"selftest: {len(checks) - n_fail}/{len(checks)} PASS")
    if n_fail:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-gate", action="store_true", help="skip the Table 1/2 regression gate")
    ap.add_argument("--base", type=Path, default=None,
                    help="project root holding logs/ and PDF/ (default: 00_config_l3 BASE_DIR)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.base is not None:
        logs_dir, pdf_dir = a.base / "logs", a.base / "PDF"
    else:
        cfg = _load_module("00_config_l3.py")
        logs_dir, pdf_dir = Path(cfg.LOGS_DIR), Path(cfg.PDF_DIR)
    run(logs_dir, pdf_dir, use_gate=not a.no_gate)


if __name__ == "__main__":
    main()
