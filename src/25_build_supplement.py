#!/usr/bin/env python
"""
25_build_supplement.py -- GJI revision round 2: assemble the Supporting
Information as ONE separate document, from the real logs and figures.
=============================================================================
WHY THIS SCRIPT EXISTS
----------------------
GJI's Instructions to Authors, section 2.8, say that supplementary files are

    "placed online in exactly the format in which they are provided -- the
     publishers will not modify them in any way"

so the supplement is not typeset by the journal and has to be self-contained.
It also asks for additional figures "(with captions)" as PDF, and for tables
in a machine-readable file with a description of the columns.

The manuscript therefore carries only a one-line index of the supplementary
items, and THIS document is the authoritative supplement. Keeping the captions
in both places would create two sources that must agree; the round-1 version
did that and had already drifted (the manuscript listed Figures S1-S5 while
the text referenced S1-S9).

WHAT IT DOES
------------
  1. INVENTORY  -- report which logs and figure files are present, and what
                   each supplementary item needs. Writes nothing.
  2. BUILD      -- build every table from its log, write each as CSV, render
                   the text and tables, prepend a caption page to each figure,
                   and merge everything into one PDF.
  3. CHECK      -- re-read the logs and verify the numbers QUOTED IN THE PROSE
                   of Sections S5 and S6 against them.

Nothing is invented. A missing log means its table is skipped, listed in the
manifest, and the output is named ..._INCOMPLETE.pdf so it cannot be mistaken
for the submission copy. The script then exits non-zero.

OUTPUTS
  PDF/Supplement_MOIRAI_L3.pdf          the submission copy (complete only)
  PDF/Supplement_MOIRAI_L3_INCOMPLETE.pdf   when an input is missing
  logs/supplement/TableS1.csv .. TableS7.csv   machine-readable tables
  logs/supplement/25_manifest.json      every item, its source, found or not
  logs/supplement/25_number_check.md    prose numbers against the logs

SOURCES (all written by earlier scripts in this archive)
  Table S1 <- logs/14_table_s1.csv
  Table S2 <- logs/16_arrival_audit_ema.json + logs/16_clean_rescore_ema.json
  Table S3 <- logs/18_paired_scores_ema.json
  Table S4 <- logs/19_capacity_ema.json
  Table S5 <- logs/20_within_array_ema.json
  Table S6 <- logs/15_phase_confusion_ema.json
  Table S7 <- logs/24_moveout_ratio.json
  Figures  <- PDF/ and figures/ (see FIGURE_SOURCE)

RUN (no GPU, no AMBER, no checkpoints -- logs and figures only)
  python src/25_build_supplement.py --inventory
  python src/25_build_supplement.py --build
  python src/25_build_supplement.py --build --figdir /content/figures
  # self-test (synthetic logs and figures in a temporary tree):
  python src/25_build_supplement.py --selftest

All figures: white background, English labels, legend clear of the data --
inherited from the scripts that made them; this script does not redraw them.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

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

SITE_ORDER = ["pnr-1", "mseel_3h", "mseel_5h", "clearfield_mw6",
              "pnr-2", "clearfield_mw4", "aneth", "forge_19"]

TITLE = ("Supporting information for: Does receiver-array moveout help a "
         "deep-learning phase picker? A leave-one-site-out benchmark on "
         "downhole arrays")
AUTHOR = "Isao Kurosawa"

PREAMBLE = (
    "This document is the Supporting Information for the manuscript named "
    "above. It contains Sections S1 to S6, Tables S1 to S7 and Figures S1 to "
    "S10. Every number here is produced by the released analysis scripts from "
    "the released result logs; the script and log that produce each table are "
    "named in its caption. Section and item numbers are those used in the "
    "main text.")


# ============================================================================
# 1. The supplementary text. Fixed content: it is the manuscript's own prose.
# ============================================================================
SECTIONS = [
("S1", "Quality of the catalogued arrivals, and whether the results depend on it",
 """Reviewer 2 asked how prevalent unlabelled earthquakes are in the benchmark and
what the overall quality of its labels is. The question cites Aguilar Suarez &
Beroza (2025) on pervasive label errors in seismological machine-learning
datasets. It is answered here in three steps. The catalogued arrivals are tested against physics. The cause of the failures
is identified. Every site-level verdict in the paper is then recomputed with
the failing events removed.

A catalogued arrival sequence must be consistent with a body wave travelling
along the string. Three conditions follow. The apparent velocity implied by the
moveout cannot fall below the medium velocity. That velocity is kilometres per
second in all of these wells, so a floor of 1500 m/s is generous. The arrival
sequence of a point source seen by a near-linear string is monotonic or has a
single turn; many reversals therefore indicate scattered picks. The S arrival cannot precede the P arrival on the same station. Events
violating any of these are flagged. Fig. S6 shows the three tests.

The reversal test applies only where the station-to-station step exceeds the
pick quantization. At aneth the median P moveout is about one sample per gap,
so integer pick times alone produce reversals in sound data; the test is
therefore disabled at sites whose median gap step is below three samples, and
Table S2 reports where it applied.

Between 0 and 21 per cent of events are flagged. The two extremes are
forge_19 (1 of 213) and clearfield_mw6 (128 of 612). The flagged arrivals imply apparent velocities of a few hundred metres per
second. The medium velocities are kilometres per second. The largest raw P moveout reaches 909 ms, and exceeds 350 ms at seven of the
eight sites. These are not body-wave arrivals.

The cause is the catalogue rather than the analysis window. Truncation at a
window edge displaces a label by at most the labeller's taper half-width of 30
samples, not by hundreds. And if the suspect picks were pinned to the edges,
about half of the extreme arrivals would lie within one taper width of one.
They do not. Across sites, 8 per cent of the flagged extreme arrivals lie within 60 samples
of an edge. The figures are 1 per cent for the sound
arrivals and 6 per cent for a random position. The suspect picks are spread
through the window, which is the signature of picks belonging to another event
or to no arrival at all.

Removing the flagged events changes no conclusion. Each configuration was
audited against its own labels, and the union of the two flag sets was dropped
from both. The two are therefore scored on an identical set of events. All
eight site-level verdicts are unchanged (Table S2). At forge_19 only two events are dropped. The array picker moves from 0.516 to
0.517, and the per-trace picker from 0.856 to 0.858. The collapse there is not a label
artefact.

Two further observations are worth recording. First, cleaning raises the absolute scores at every site, by up to 0.074
F1-mean at clearfield_mw6. It
raises both configurations almost equally. The benchmark's absolute numbers are
thus partly a floor set by catalogue quality; the comparison between
configurations is not. Second, the flags agree closely between the two
configurations although their analysis windows were drawn independently. At
mseel_3h, 91 of the 107 and 109 flagged events are common. A flagged event is
therefore a property of the catalogue rather than of the window.

The re-scoring itself is checked. The all-events column of Table S2 reproduces
the archived fixed-threshold scores in all 16 site by configuration
comparisons. The subset indexing therefore cannot be silently wrong."""),

("S2", "Evaluation of the two configurations on identical inputs",
 """The leave-one-site-out results of Section 4 were produced by two independent
passes over the benchmark, one per configuration. The AMBER loader selects the analysis window when an event is accessed. At
strings carrying more than twelve sensors it also selects which contiguous run
of twelve is used. The two
passes therefore did not present identical inputs to the two pickers. The event
set and its order were identical at every site. The selection does not depend
on which configuration is being evaluated, so neither picker was favoured. The
comparison was nevertheless unpaired, and the consequence tracks the number of
sensors. At the six sites carrying exactly twelve sensors the two passes agree
on 98 to 100 per cent of events. At pnr-1 (24 sensors) and aneth (18 sensors),
the only two sites with a subset to choose, they agree on 69 and 52 per cent.

To remove this, both checkpoints were re-evaluated in a single pass over the
data. The array and per-trace probabilities were computed from the same batch.
They were scored against the same label tensor. The resulting caches are paired by
construction, which the analysis asserts before scoring.

No site-level F1-mean moves by more than 0.013 between the submitted and the
paired evaluation, so the independent draws cost essentially nothing (Table
S3). Of the eight site-level verdicts, none reverses direction, six are
identical, and two move from a tie to array-favoured.

Identical inputs also permit a paired bootstrap of the array-minus-per-trace
difference. Resampling the difference cancels the between-event variance the
two configurations share. The paired interval is therefore narrower than the
gap between two marginal intervals, and a difference can be detected where
those intervals overlap. A tie becoming decided is the expected consequence of pairing, not a
disagreement with the submitted result. Both interval forms are reported in
Table S3, and Fig. S7 shows them.

The two observations on which the conclusions rest are unchanged. At forge_19
the array picker remains far behind the per-trace picker under identical inputs
(0.524 against 0.856; paired difference -0.332, 95 per cent interval -0.355 to
-0.310), and at aneth the per-trace picker remains ahead (-0.055, -0.065 to
-0.047). At pnr-2 the array picker retains its advantage (+0.036, +0.027 to
+0.046)."""),

("S3", "Model capacity and overfitting",
 """Reviewer 2 observed that a model with eight million parameters is large for a
benchmark of about ten thousand events. The count is 7,991,491 trainable
parameters. The benchmark figure needs one correction, and it makes the
reviewer's point stronger rather than weaker. The eight usable sites hold 9,803
records, but 6,382 of those are noise windows; the catalogued earthquakes
number 3,421. Each leave-one-site-out run therefore trains on 4,769 to 5,932
records, of which 1,527 to 2,002 are earthquakes. The ratio is larger than the
reviewer supposed. The question it raises is whether the model overfits. That
question is answered here from the training histories rather than argued from
the ratio.

Forty-six models were trained for this paper. Sixteen cover the eight leave-
one-site-out folds in the two configurations. Twenty-four cover three
additional random seeds at the four decisive sites. Four are the station-
shuffle control, and two the within-array experiment of Section S4. The 44 runs
analysed here are all but the last two, which use a different protocol.

Note that the two configurations do not see the same number of training
samples. The array configuration sees one sample per record, while the per-trace
configuration sees one per station trace, which is 58,464 to 72,420 per run.
The array configuration thus has roughly twelve times fewer independent
samples, at a larger input dimension. That is consistent with its being the
configuration that fails where the training distribution is thin.

Across the 44 runs the development score sits near 0.93 and varies by 0.0045
from one epoch to the next. The gap between its maximum and its final value has
a median of 0.011. That gap is largely the upward bias of a maximum taken over
a noisy sequence. It is not by itself evidence of a decline. The quantity
that bears on the question is the trend after the peak. It is measured as the
difference between the means of the second and first halves of the post-peak
segment. Its median is -0.0018 in F1. Measured against each run's own epoch-to-epoch
variation, the median ratio is -0.36. The trend is negative in 30 of 43 runs. The sign is
negative more often than chance allows (sign test p = 0.014). A downward drift is therefore present and consistent. In no run does it exceed
that run's own variation. Fig. S2 shows the curves, and Table S4 gives the
per-run figures.

Forty-one of the 44 runs were halted by the development split; the other
three reached the epoch budget first, having last improved 4, 9 and 18 epochs
earlier. Every score reported in this paper is taken from the best development
epoch. The capacity is used. The training loss falls by a factor of 22 to 41 between
the first epoch and the best one. The median factor is 29, over the 42 runs
whose logged loss is usable. Two runs are excluded. Their best logged loss is
zero or three parts in a million, which is a recording fault rather than a
value. None of this measurably harms generalization."""),

("S4", "Training with the held-out arrays represented",
 """Reviewer 2 proposed training on about 90 per cent of the data from each array
and testing on the remaining 10 per cent. The aim is to compare a model that
saw long moveout in training with one that did not. The leave-one-site-out
protocol of the main text answers a different question: how a picker behaves on
an array it has never seen. Together the two protocols separate the
explanations the editor asked to be distinguished.

The benchmark already carries the required partition. Each site has its own
training, development and test split in the AMBER metadata, so no new partition
was invented. One site is an exception. Every pnr-1 record is labelled test, so that site
contributes no training events. It is not represented in training under either
protocol. It ties under both, so the comparison below is unaffected, but
its row is not a test of representation. One array model and one per-trace
model were trained on the training splits of all eight sites together. The
architecture, optimizer, schedule and early stopping are those of every other
run reported here. Each model was then evaluated on each site's own test split, in a single pass.
The paired interval of Section S2 therefore applies here as well. Fig. S8
shows the outcome, and Table S5 gives the per-site figures.

The experiment can fail. Suppose the collapse at forge_19 follows from the
scarcity of such moveout in training. A model trained with forge_19's own
events should then pick that site normally, and the array-minus-per-trace gap
there should close. Suppose instead that forge_19 is intrinsically harder:
noisier, smaller events, or less reliable picks. The gap should then persist,
and the mechanism proposed in Section 5.2 would have to be revised.

Both models trained without incident and were halted by the development
split, as every other run in this paper was. The eight test splits hold 2,530 records in all, from 47 at aneth to 1,258 at
pnr-1. Of these, 984 are catalogued earthquakes and the rest noise windows. That is 26 per cent of the benchmark rather than the 10 per cent proposed. The
benchmark's own partition was used in place of a new one.

The prediction holds at forge_19. The array F1-mean there rises from 0.516
under leave-one-site-out to 0.950 once the site's own events are in the
training splits. The per-trace model rises far less, from 0.856 to 0.952,
because it had not failed. The gap between the two configurations therefore
falls from 0.340 to 0.002, with a paired interval of [-0.011, +0.008] that
spans zero. Had forge_19 been intrinsically harder, through noise, event
size or pick quality, that gap would have persisted. It does not. The
collapse is a property of the training distribution rather than of the site.

The same applies at aneth, the other site whose leave-one-site-out verdict
favours the per-trace model. Its per-trace advantage of 0.057 disappears:
the within-array difference is +0.003 in the array's favour, on an interval
that spans zero. The site contributes only 47 test events, so this result
carries less weight than the one at forge_19. Both decided verdicts of the
main text none the less track how the training set is composed rather than
the sites themselves.

The conclusion of the main text survives the reviewer's protocol. Across the
eight sites the within-array comparison gives two array wins, one per-trace
win and five ties. Leave-one-site-out gives one, two and five. The median
verdict is a tie under both. No within-array difference exceeds 0.017 in
F1-mean, which is a twentieth of the forge_19 gap it replaces.

Representation in training is not simply an advantage. At mseel_3h both
configurations score lower within-array than under leave-one-site-out, by
0.040 and 0.042, although that site's own events are now in the training
splits. Each model here is fitted to eight sites rather than seven. The effect of
adding one site is therefore not separable from the effect of adding the rest. It is not interpreted further.

These numbers are not an estimate of field performance and are not offered
as one. The test events come from the same arrays and the same acquisitions as the
training events. That is the confound the leave-one-site-out design exists to
remove. The protocol is used here for a single purpose: to vary
whether a given moveout is represented in training while holding the rest
fixed."""),

("S5", "Where the declared picks land: P/S assignment",
 """Reviewer comment R1-6 asks whether the forge_19 collapse is a loss of P
detection or a confusion between the P and S phases. The cached picking statistics answer the question exactly. A pick is the
global maximum of its channel. Its sample index can therefore be compared with
the catalogued arrivals of both phases. Every declared pick on an earthquake event is classified in one of three ways.
On-phase is within the primary tolerance of its own channel's catalogued
arrival. Cross-phase is within that tolerance of the other phase's arrival.
Everything else is elsewhere. Table S6 reports the P channel;
Fig. S9 shows the full residual distributions.

At forge_19 the array P channel declares 2,023 picks. Only 18.1 per cent land on the catalogued P. A larger share, 22.9 per cent,
lands on the catalogued S. The median offset from the catalogued P is +146 ms,
and 48.2 per cent of these picks fall between the two arrivals. The per-trace
model, evaluated on the same site, places 82.4 per cent on the catalogued P
and 0.5 per cent on the S; the array at a site whose moveout is well
represented in training places 91.3 per cent on the catalogued P (mseel_5h).
The behaviour is therefore specific to the array configuration at forge_19 and
not a property of that site's catalogue.

The reviewer's reading is confirmed for the P channel. The detector mis-
assigns rather than merely under-fires. It follows a later apparent-moveout
ridge that coincides with the S arrival. It is refined for the S channel. The array S channel still places 82.4 per cent of its picks on the catalogued
S. Only 5.3 per cent land on the catalogued P. The early S-score ridge visible
in Fig. 6 of the main text is therefore sub-maximal. It does not capture the
pick. The
failure is one-sided: P is drawn towards S, not the reverse."""),

("S6", "How much moveout each site actually carries",
 """Section 5.2 attributes the array's disadvantage at aneth to there being
too little moveout to exploit. That claim rests on a median, so the full
distribution is given here. The quantity is the across-station spread of
the catalogued P arrivals per event, the definition used in Table 1. It
needs no velocity model and no assumption about the ray geometry.

At aneth 86.6 per cent of events (258 of 298) have a P moveout below 10 ms.
That is 20 samples at the common 2000 Hz rate. No other site exceeds 2.4 per
cent, and four carry no such event at all. The site is not merely at the low
end of the moveout range: it occupies a regime the other seven do not enter.
Table S7 gives the per-site distribution.

One further test was carried out and is not used. The ratio of the S to the P moveout of one event equals Vp/Vs when both phases
follow the same ray geometry. It can therefore be compared with each site's
own velocity model. The
estimator is biased high where the P moveout is small and low where it is
large, and that bias appears at every site. Table S7 shows that the sites do
not share a common band of P moveout wide enough to compare them in. The test
is released with the archive for completeness, and no conclusion is drawn from
it."""),
]


# ============================================================================
# 2. Table captions. The caption names the script and log that produced it,
#    so a reader can retrace any number without asking.
# ============================================================================
TABLE_CAPTIONS = {
1: ("Table S1. Per-site array geometry computed from the station coordinates "
    "distributed with AMBER (per-dataset Array.csv files): number of stations "
    "on the full instrument string, vertical extent, three-dimensional "
    "aperture, and median station spacing. Source: script 14, "
    "logs/14_table_s1.csv."),
2: ("Table S2. Catalogued-arrival audit and re-scoring on the surviving "
    "events. Flagged events fail at least one of the three physical tests of "
    "Section S1. \"Gap step\" is the median station-to-station arrival step, "
    "and the reversal test is disabled where it falls below three samples. "
    "Scores are F1-mean at the fixed 0.30 threshold, before and after removing "
    "the union of the events flagged in either configuration. The all-events "
    "columns reproduce the archived scores of Table 3. Source: script 16, "
    "logs/16_arrival_audit_ema.json and logs/16_clean_rescore_ema.json."),
3: ("Table S3. Site-level scores under the submitted and the paired "
    "evaluation, with the paired-bootstrap interval on the "
    "array-minus-per-trace difference and the verdict under each interval "
    "form. \"Reversed\" marks a change between array-favoured and "
    "per-trace-favoured; a move between a tie and a decided verdict is not a "
    "reversal. Source: script 18, logs/18_paired_scores_ema.json."),
4: ("Table S4. Per-run training summary: the best development epoch, the score "
    "there and at the last epoch, the fall from the peak, the trend after the "
    "peak, the epoch-to-epoch variation of the same curve, their ratio, the "
    "training loss at the first, best and last epoch, and whether the run was "
    "halted by the development split. A large fall from the peak with a small "
    "trend-to-variation ratio is a noisy plateau rather than overfitting. "
    "Source: script 19, logs/19_capacity_ema.json."),
5: ("Table S5. Site-level scores under the two protocols, with the "
    "paired-bootstrap interval on the array-minus-per-trace difference under "
    "within-array training and the verdict under each protocol. Source: "
    "script 20, logs/20_within_array_ema.json."),
6: ("Table S6. Where the declared P-channel picks land, by site and "
    "configuration (EMA weights, primary tolerance). On-phase: within "
    "tolerance of the catalogued P. Cross-phase: within tolerance of the "
    "catalogued S. Between: strictly between the two catalogued arrivals. "
    "Median offset is measured from the catalogued P. Source: script 15, "
    "logs/15_phase_confusion_ema.json."),
7: ("Table S7. Distribution of the catalogued P moveout across events, by "
    "site. Moveout is the across-station spread of the catalogued P arrivals "
    "of one event, over events carrying at least two P picks and at least two "
    "S picks and a P moveout of at least 1 ms. Source: script 24, "
    "logs/24_moveout_ratio.json."),
}


# ============================================================================
# 3. Figure captions and sources.
# ============================================================================
FIGURE_CAPTIONS = {
1: ("Figure S1. Example per-station P/S/noise score panel for a representative "
    "event."),
2: ("Figure S2. Training and development curves for all 44 runs (eight "
    "leave-one-site-out folds in the array, per-trace and station-shuffle "
    "configurations, with three additional seeds at four sites). (a) Training "
    "loss per epoch on a logarithmic scale. (b) Development F1-mean per epoch, "
    "with the best epoch marked; the reported checkpoint is taken from that "
    "epoch. (c) The same curves on the vertical scale of the effect under "
    "discussion, which is a post-peak trend of order 0.002 in F1 against an "
    "epoch-to-epoch variation of 0.0045."),
3: "Figure S3. Summary of held-out metrics.",
4: ("Figure S4. Off-the-shelf surface-trained baseline applied to a borehole "
    "site (motivation only; confounded by sampling rate, architecture and "
    "training domain)."),
5: ("Figure S5. Zero-shot transfer of the trained array picker to decimated "
    "arrays (4, 6, 8 stations; mean plus or minus one standard deviation over "
    "five draws) under contiguous-window and random decimation, with the "
    "per-trace model as control."),
6: ("Figure S6. Physical audit of the catalogued arrivals. (a) Apparent "
    "velocity implied by the P moveout at each site (dot: median, bar: 5th to "
    "95th percentile) against the 1500 m/s floor below which an arrival cannot "
    "be a body wave. (b) Fraction of extreme arrivals lying within 60 samples "
    "of a window edge, for flagged and for sound events, against the fraction "
    "expected if their position were random (dotted); a windowing artefact "
    "would pile the flagged arrivals at the edges and does not. (c) "
    "Distribution of catalogued S-P times per site; mass below zero (shaded) "
    "is a label error."),
7: ("Figure S7. The two configurations evaluated on identical inputs. (a) "
    "F1-mean per site under the submitted evaluation, in which the analysis "
    "window and the sensor subset were drawn independently for each "
    "configuration (open symbols), and under a single pass in which both "
    "configurations saw the same batches (filled symbols). (b) Array minus "
    "per-trace on those identical inputs, with paired-bootstrap 95 per cent "
    "intervals; the pairing is legitimate only because the inputs are "
    "shared."),
8: ("Figure S8. Picking performance when the held-out arrays are represented "
    "in training. (a) F1-mean per site under leave-one-site-out (open symbols) "
    "and when the model is trained on the training splits of the eight sites "
    "and evaluated on each site's own test split (filled symbols). (b) Array "
    "minus per-trace on the within-array test splits, with paired-bootstrap 95 "
    "per cent intervals."),
9: ("Figure S9. Where the declared picks land, against both catalogued phases, "
    "for the out-of-distribution site (forge_19) and an in-distribution "
    "contrast (mseel_5h), array and per-trace. Blue: pick time minus the "
    "channel's own catalogued arrival. Red: pick time minus the other phase's "
    "catalogued arrival. A detector that merely fails to fire gives few picks "
    "and a flat residual; one that mis-assigns gives a residual peaking at "
    "zero against the other phase. The forge_19 array P panel shows the "
    "latter."),
10: ("Figure S10. Waveforms and predicted scores at the remaining held-out "
     "sites (array configuration), in the format of Fig. 6 of the main text. "
     "One row per site: Z-component waveforms with the catalogued P and S "
     "picks, the predicted P score map, and the predicted S score map. The "
     "six sites are those that Fig. 6 does not show."),
}

# Each figure's source file, newest first. The first file that exists is used.
FIGURE_SOURCE = {
1: ["05_a_probability_example.pdf", "05_a_probability_example.png"],
2: ["19_a_training_curves.pdf", "19_a_training_curves.png"],
3: ["05_c_metrics_summary.pdf", "05_c_metrics_summary.png"],
4: ["05_f_baseline_phasenet.pdf", "05_f_baseline_phasenet.png"],
5: ["13_a_station_subset.pdf", "13_a_station_subset.png"],
6: ["16_a_arrival_audit.pdf", "16_a_arrival_audit.png"],
7: ["18_a_paired_vs_unpaired.pdf", "18_a_paired_vs_unpaired.png"],
8: ["20_a_within_vs_loso.pdf", "20_a_within_vs_loso.png"],
9: ["15_a_phase_residuals.pdf", "15_a_phase_residuals.png"],
10: ["23_a_fig4_style_sites.pdf", "23_a_fig4_style_sites.png"],
}


# ============================================================================
# 4. Table builders. Each returns (header, rows) or None when its log is
#    absent. None is never replaced by a guess.
# ============================================================================
def _json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _f(x, spec="+.3f", dash="-"):
    """Format a number, or return dash when it is None or not finite."""
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if v != v:                                     # NaN
        return dash
    return f"{v:{spec}}"


# The CSV written by script 14 carries machine column names and an extra
# dataset column. The published table uses these headings, in this order, as
# the submitted manuscript did; a renamed column fails loudly rather than
# putting the wrong numbers under the right heading.
S1_COLUMNS = [("site", "Site"),
              ("n_stations", "Stations (full string)"),
              ("vertical_extent_m", "Vertical extent (m)"),
              ("aperture_3d_m", "3-D aperture (m)"),
              ("median_adjacent_spacing_m", "Median spacing (m)")]


def table_s1(logs: Path):
    p = logs / "14_table_s1.csv"
    if not p.exists():
        return None
    with p.open(newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if len(rows) < 2:
        return None
    cols = {name: i for i, name in enumerate(rows[0])}
    missing = [src for src, _ in S1_COLUMNS if src not in cols]
    assert not missing, f"14_table_s1.csv の列名が変わっている: {missing}"
    by_site = {r[cols["site"]]: r for r in rows[1:]}
    assert set(by_site) == set(SITE_ORDER), sorted(by_site)
    head = [pub for _, pub in S1_COLUMNS]
    out = [[by_site[s][cols[src]] for src, _ in S1_COLUMNS] for s in SITE_ORDER]
    return head, out


def table_s2(logs: Path):
    aud = _json(logs / "16_arrival_audit_ema.json")
    if not aud:
        return None
    res = _json(logs / "16_clean_rescore_ema.json") or {}
    head = ["Site", "Earthquakes", "Flagged", "Flagged %", "Gap step (samples)",
            "Reversal test", "All: array", "All: per-trace",
            "Clean: array", "Clean: per-trace", "Dropped",
            "Verdict (all)", "Verdict (clean)", "Verdict survives"]
    rows = []
    for s in SITE_ORDER:
        a = aud.get(s)
        if not a:
            continue
        r = res.get(s, {})
        allv, cln = r.get("all", {}), r.get("clean", {})
        rows.append([
            s, a["n_events"], a["flagged_events"],
            f"{100 * a['flagged_fraction']:.1f}",
            _f(a.get("median_gap_step_samples"), ".1f"),
            "applied" if a.get("reversal_test_applied") else "disabled",
            _f(allv.get("array"), ".3f"), _f(allv.get("pertrace"), ".3f"),
            _f(cln.get("array"), ".3f"), _f(cln.get("pertrace"), ".3f"),
            r.get("n_dropped", "-"),
            allv.get("verdict", "-"), cln.get("verdict", "-"),
            ("yes" if r.get("verdict_survives") else
             "no" if "verdict_survives" in r else "-"),
        ])
    return (head, rows) if rows else None


def table_s3(logs: Path):
    d = _json(logs / "18_paired_scores_ema.json")
    if not d:
        return None
    head = ["Site", "Events", "Submitted: array", "Submitted: per-trace",
            "Paired: array", "Paired: per-trace", "Difference",
            "Paired 95% CI", "Marginal verdict", "Paired verdict",
            "Submitted verdict", "Reversed"]
    rows = []
    for s in SITE_ORDER:
        r = d.get(s)
        if not r:
            continue
        pc = (r.get("paired_ci") or {}).get("orig", {})
        rows.append([
            s, r.get("n_events", "-"),
            _f(r.get("submitted_array"), ".3f"),
            _f(r.get("submitted_pertrace"), ".3f"),
            _f(r.get("array"), ".3f"), _f(r.get("pertrace"), ".3f"),
            _f(pc.get("median")),
            f"[{_f(pc.get('lo'))}, {_f(pc.get('hi'))}]",
            r.get("verdict_marginal", "-"), r.get("verdict_paired", "-"),
            r.get("submitted_verdict") or "-",
            "yes" if r.get("reversal_vs_submitted") else "no",
        ])
    return (head, rows) if rows else None


def table_s4(logs: Path):
    d = _json(logs / "19_capacity_ema.json")
    if not d:
        return None
    head = ["Site", "Config", "Epochs", "Best epoch", "Dev F1 best",
            "Dev F1 last", "Fall from best", "Post-peak trend",
            "Epoch variation", "Trend / variation",
            "Train loss first", "best", "last", "Early stop"]
    keys = [k for k in d if not k.startswith("_")]
    def order(k):
        site = k.split("__")[0]
        cfg_ = k.split("__")[-1]
        rank = {"array": 0, "pertrace": 1, "shuffle": 2}.get(cfg_, 3)
        return (SITE_ORDER.index(site) if site in SITE_ORDER else 99, rank, k)
    rows = []
    for k in sorted(keys, key=order):
        r = d[k]
        site, _, conf = k.partition("__")
        if not r.get("usable"):
            rows.append([site, conf, r.get("n_epochs", "-")] + ["-"] * 11)
            continue
        rows.append([
            site, conf, r["n_epochs"], r["best_epoch"],
            _f(r.get("dev_f1_best"), ".3f"), _f(r.get("dev_f1_last"), ".3f"),
            _f(r.get("dev_f1_drop_from_best")),
            _f(r.get("post_peak_trend"), "+.4f"),
            _f(r.get("dev_epoch_noise"), ".4f"),
            _f(r.get("trend_over_noise"), "+.2f"),
            # Three significant figures, not four decimals. The best losses
            # are around 7e-4, so ".4f" printed them as 0.0007 -- one
            # significant figure -- and the factor-of-22-to-41 claim in
            # Section S3 could not be recomputed from the table at all.
            _f(r.get("train_loss_first"), ".3g"),
            _f(r.get("train_loss_best"), ".3g"),
            _f(r.get("train_loss_last"), ".3g"),
            "yes" if r.get("early_stop_fired") else "no",
        ])
    return (head, rows) if rows else None


def table_s5(logs: Path):
    d = _json(logs / "20_within_array_ema.json")
    if not d:
        return None
    head = ["Site", "Test events", "LOSO: array", "LOSO: per-trace",
            "Within: array", "Within: per-trace", "Difference",
            "Paired 95% CI", "LOSO verdict", "Within-array verdict"]
    rows = []
    for s in SITE_ORDER:
        r = d.get(s)
        if not r:
            continue
        ci = r.get("diff_ci") or [None, None]
        rows.append([
            s, r.get("n_test_events", "-"),
            _f(r.get("loso_array"), ".3f"), _f(r.get("loso_pertrace"), ".3f"),
            _f(r.get("array"), ".3f"), _f(r.get("pertrace"), ".3f"),
            _f(r.get("diff_median")),
            f"[{_f(ci[0])}, {_f(ci[1])}]",
            r.get("loso_verdict") or "-", r.get("verdict", "-"),
        ])
    return (head, rows) if rows else None


def table_s6(logs: Path):
    d = _json(logs / "15_phase_confusion_ema.json")
    if not d:
        return None
    head = ["Site", "Config", "P picks", "On-phase", "Cross-phase",
            "Elsewhere", "Between", "Median offset (ms)"]
    rows = []
    for s in SITE_ORDER:
        for conf in ("array", "pertrace"):
            r = d.get(f"{s}__{conf}")
            if not r:
                continue
            p = r.get("P", {})
            res = p.get("residuals", {}) or {}
            med = (res.get("own_ms_q25_med_q75") or [None, None, None])[1]
            rows.append([
                s, "array" if conf == "array" else "per-trace",
                p.get("n_picks", "-"),
                _f(p.get("frac_on_phase"), ".3f"), _f(p.get("frac_cross"), ".3f"),
                _f(p.get("frac_elsewhere"), ".3f"),
                _f(res.get("frac_between_phases"), ".3f") if res.get("n") else "-",
                _f(med, "+.1f") if res.get("n") else "-",
            ])
    return (head, rows) if rows else None


# The bin labels script 24 writes, in order. Used as the column set so a
# renamed or reordered bin shows up as a missing column rather than silently
# shifting the numbers into the wrong one.
S7_BINS = ["1-5 ms", "5-10 ms", "10-20 ms", "20-40 ms", ">=40 ms"]


def _s7_bins(site: str, rec: dict) -> dict:
    """{bin label: count} for one site, refusing a renamed or missing bin."""
    bins = {b["bin"]: b["n"] for b in (rec.get("stratified", {})
                                       .get("bins", []))}
    missing = [b for b in S7_BINS if b not in bins]
    assert not missing, f"{site}: script 24 の bin ラベルが変わっている {missing}"
    return bins


def s7_below_10ms(site: str, rec: dict):
    """The share of events below 10 ms, DERIVED from the bin counts.

    The bins are the primitive and are always written. The convenience key
    frac_P_below_10ms was added to script 24 later, so a log produced before
    that is missing it. An earlier version of this function read that key with
    a default of 0.0, which silently put 0.0 into Table S7 for every site --
    aneth included, where the true share is 86.6 per cent. Never default a
    missing number: derive it, or report that it cannot be had.

    Where the key IS present it must agree, so a future change to either
    definition is caught rather than quietly tolerated.
    """
    bins = _s7_bins(site, rec)
    tot = sum(bins[b] for b in S7_BINS)
    if tot == 0:
        return None
    share = (bins["1-5 ms"] + bins["5-10 ms"]) / tot
    logged = rec.get("frac_P_below_10ms")
    if logged is not None:
        assert abs(float(logged) - share) < 1e-3, \
            f"{site}: frac_P_below_10ms={logged} が bin から計算した {share} と違う"
    return share


def table_s7(logs: Path):
    d = _json(logs / "24_moveout_ratio.json")
    if not d:
        return None
    head = ["Site", "Events"] + S7_BINS + ["Below 10 ms (%)"]
    rows = []
    for s in SITE_ORDER:
        r = d.get(s)
        if not r:
            continue
        bins = _s7_bins(s, r)
        tot = sum(bins[b] for b in S7_BINS)
        share = s7_below_10ms(s, r)
        rows.append([s, tot] + [(bins[b] or "-") for b in S7_BINS]
                    + ["-" if share is None else f"{100 * share:.1f}"])
    return (head, rows) if rows else None


TABLE_BUILDERS = {1: table_s1, 2: table_s2, 3: table_s3, 4: table_s4,
                  5: table_s5, 6: table_s6, 7: table_s7}

TABLE_SOURCE = {
1: ["14_table_s1.csv"],
2: ["16_arrival_audit_ema.json", "16_clean_rescore_ema.json"],
3: ["18_paired_scores_ema.json"],
4: ["19_capacity_ema.json"],
5: ["20_within_array_ema.json"],
6: ["15_phase_confusion_ema.json"],
7: ["24_moveout_ratio.json"],
}


def s21_counts(d21) -> dict:
    """{site: {rec, noise, eq, train_rec, train_eq}} from script 21's log.

    RECORDS, not rows. Script 21's table counts STATION TRACES: at pnr-1
    rows_total is 30,192 for 1,258 records, because that string carries 24
    sensors. An earlier version of this reader took rows_total for the record
    count, which would have made all eight sites "usable" and then reported
    the manuscript's correct 9,803 as wrong. The record counts live in
    unique_events_total and unique_events_by_split; the noise counts are rows,
    so they are divided by the sensors per record.

    Two shapes are accepted: script 21's own output, and an earlier derived
    summary in this project that stores rec / eq_tot / nw_tot / eq / nw
    directly. A site whose table is unusable is left out rather than guessed
    at, and the caller refuses to sum unless all eight are present.
    """
    out = {}
    for s, v in (d21 or {}).items():
        if not isinstance(v, dict):
            continue
        t_ = v.get("table")
        if isinstance(t_, dict) and t_.get("unique_events_total") \
                and t_.get("rows_total") is not None \
                and t_.get("noise_rows_total") is not None \
                and isinstance(t_.get("unique_events_by_split"), dict):
            rec = int(t_["unique_events_total"])
            rows = int(t_["rows_total"])
            if rec <= 0 or rows % rec:
                continue           # sensors per record is not an integer
            st = rows // rec
            nz_rows = int(t_["noise_rows_total"])
            nbs = t_.get("noise_rows_by_split") or {}
            tr_nz_rows = int(nbs.get("train", 0))
            if nz_rows % st or tr_nz_rows % st:
                continue           # noise rows do not divide into records
            noise = nz_rows // st
            tr = int(t_["unique_events_by_split"].get("train", 0))
            eq = rec - noise
            # script 21 carries the audit's own earthquake count; if the two
            # disagree the site is dropped rather than averaged over
            aud_n = v.get("audit_n_events")
            if aud_n is not None and int(aud_n) != eq:
                continue
            out[s] = dict(rec=rec, noise=noise, eq=eq,
                          train_rec=tr, train_eq=tr - tr_nz_rows // st)
        elif isinstance(v.get("eq"), dict) and isinstance(v.get("nw"), dict) \
                and "rec" in v and "eq_tot" in v:
            eq_tr = int(v["eq"].get("train", 0))
            nw_tr = int(v["nw"].get("train", 0))
            out[s] = dict(rec=int(v["rec"]),
                          noise=int(v.get("nw_tot", v["rec"] - v["eq_tot"])),
                          eq=int(v["eq_tot"]),
                          train_rec=eq_tr + nw_tr, train_eq=eq_tr)
    return out


# ============================================================================
# 5. The numbers quoted in the prose, checked against the logs.
#    The prose is fixed text, so without this check it could drift away from
#    the logs silently. Each entry is (label, log reader, expected).
# ============================================================================
def check_numbers(logs: Path) -> list:
    """[(label, expected, found, ok)] for every number the prose quotes.

    Everything is read from the JSON logs at full precision, never from the
    rendered table, because the table rounds. On 2026-10-08 an audit of the
    built tables found two statements that the logs did not support -- the
    cleaning gain "up to 0.070" (0.075) and the epoch-to-epoch variation
    "0.005" (0.0045) -- so every quantitative claim in the prose is listed
    here rather than trusted.
    """
    out = []

    def add(label, expected, found, tol=5e-4):
        ok = (found is not None
              and abs(float(found) - float(expected)) <= tol)
        out.append((label, expected, found, ok))

    def flag(label, expected, found, ok):
        out.append((label, expected, found, bool(ok)))

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    # ---- Section S1, from the arrival audit and the re-scoring
    aud = _json(logs / "16_arrival_audit_ema.json")
    res = _json(logs / "16_clean_rescore_ema.json")
    if aud:
        fr = {s: aud[s]["flagged_fraction"] for s in SITE_ORDER if s in aud}
        if fr:
            lo_s, hi_s = min(fr, key=fr.get), max(fr, key=fr.get)
            flag("S1 両極端は forge_19 と clearfield_mw6",
                 "forge_19 / clearfield_mw6", f"{lo_s} / {hi_s}",
                 (lo_s, hi_s) == ("forge_19", "clearfield_mw6"))
            flag("S1 flagged は 0〜21 %", "0-21 %",
                 f"{100*min(fr.values()):.1f}-{100*max(fr.values()):.1f} %",
                 max(fr.values()) <= 0.21)
        for s, n, tot in (("forge_19", 1, 213), ("clearfield_mw6", 128, 612)):
            if s in aud:
                add(f"S1 {s} flagged = {n}", n, aud[s]["flagged_events"], 0.5)
                add(f"S1 {s} earthquakes = {tot}", tot, aud[s]["n_events"], 0.5)
        dis = [s for s in SITE_ORDER
               if s in aud and not aud[s]["reversal_test_applied"]]
        flag("S1 反転検査を切るのは aneth だけ", "aneth",
             ", ".join(dis) or "(なし)", dis == ["aneth"])
        # 窓端の検査: 本文は「flagged 8 %、健全 1 %、偶然なら 6 %」と書く。
        # サイトごとの棒ではなく、件数で重みづけした全体の割合である。
        for key, pct, name in (("window_position_flagged", 8, "flagged"),
                               ("window_position_clean", 1, "健全")):
            num = sum(aud[s][key].get("frac_within_edge_band", 0)
                      * aud[s][key].get("n", 0)
                      for s in aud if key in aud[s])
            den = sum(aud[s][key].get("n", 0) for s in aud if key in aud[s])
            add(f"S1 窓端 60 サンプル以内（{name}）= {pct} %", pct,
                None if not den else round(100 * num / den, 0), 0.5)
        # 「554〜909 ms」と書いていたが、サイト別最大の実測は
        # pnr-1 365, mseel_3h 767, mseel_5h 692, clearfield_mw6 909,
        # pnr-2 608, clearfield_mw4 648, aneth 554, forge_19 82 で、その
        # 範囲の定義を復元できなかった。検証できる記述に改めてある。
        mx = {s: aud[s].get("before", {}).get("P", {}).get("max_ms")
              for s in SITE_ORDER if s in aud}
        mxv = [v for v in mx.values() if v]
        if mxv:
            flag("S1 生の P moveout の最大 = 909 ms", 909,
                 f"{max(mxv):.0f}  ["
                 + ", ".join(f"{s} {v:.0f}" for s, v in mx.items() if v) + "]",
                 abs(max(mxv) - 909) < 1)
            add("S1 350 ms を超えるのは 8 サイト中 7", 7,
                sum(1 for v in mxv if v > 350), 0.5)
    if res:
        f19 = res.get("forge_19", {})
        if f19:
            add("S1 forge_19 の除外は 2 件", 2, f19.get("n_dropped"), 0.5)
            add("S1 forge_19 array 0.516 -> 0.517", 0.517,
                f19.get("clean", {}).get("array"))
            add("S1 forge_19 per-trace 0.856 -> 0.858", 0.858,
                f19.get("clean", {}).get("pertrace"))
        surv = [s for s, r in res.items()
                if isinstance(r, dict) and not r.get("verdict_survives")]
        flag("S1 8 サイトすべて判定不変", "8/8",
             f"{len(res)-len(surv)}/{len(res)}" + (f" 例外 {surv}" if surv else ""),
             not surv and len(res) == 8)
        gains = [(round(r["clean"][k] - r["all"][k], 4), s, k)
                 for s, r in res.items() if isinstance(r, dict)
                 and "clean" in r and "all" in r for k in ("array", "pertrace")]
        if gains:
            flag("S1 清浄化は全サイトで上がる", "all > 0",
                 f"最小 {min(gains)[0]:+.3f} ({min(gains)[1]} {min(gains)[2]})",
                 all(g[0] > 0 for g in gains))
            g = max(gains)
            # 0.075 は丸めた表（0.944-0.869）から取った値で、ログの
            # 実値は 0.0744。期待値も表ではなくログから取ること。
            add("S1 上昇の最大 = 0.074", 0.074, g[0], 5e-4)
            flag("S1 その最大は clearfield_mw6", "clearfield_mw6",
                 f"{g[1]} ({g[2]})", g[1] == "clearfield_mw6")
        if aud:
            n16 = sum(1 for s in res if isinstance(res[s], dict)
                      and "all" in res[s])
            flag("S1 再現検査は 16 件", 16, 2 * n16, 2 * n16 == 16)

    # ---- Section S2, from the paired evaluation
    d18 = _json(logs / "18_paired_scores_ema.json")
    if d18:
        mv = [(round(abs(r[k] - r[f"submitted_{k}"]), 4), s, k)
              for s, r in d18.items() if isinstance(r, dict)
              for k in ("array", "pertrace")
              if r.get(f"submitted_{k}") is not None]
        if mv:
            add("S2 動きの最大 = 0.013", 0.013, max(mv)[0], 5e-4)
        ident = [s for s, r in d18.items() if isinstance(r, dict)
                 and r.get("submitted_verdict") == r.get("verdict_paired")]
        t2a = [s for s, r in d18.items() if isinstance(r, dict)
               and r.get("submitted_verdict") == "tie"
               and r.get("verdict_paired") == "array"]
        flag("S2 同一 6 件 / tie→array 2 件", "6 / 2",
             f"{len(ident)} / {len(t2a)} {t2a}",
             len(ident) == 6 and len(t2a) == 2)
        rev = [s for s, r in d18.items() if isinstance(r, dict)
               and r.get("reversal_vs_submitted")]
        flag("S2 方向転換ゼロ", 0, len(rev), not rev)
        for s, dv, lo, hi in (("forge_19", -0.332, -0.355, -0.310),
                              ("aneth", -0.055, -0.065, -0.047),
                              ("pnr-2", 0.036, 0.027, 0.046)):
            c = (d18.get(s, {}).get("paired_ci") or {}).get("orig", {})
            add(f"S2 {s} 差 = {dv:+.3f}", dv, c.get("median"))
            add(f"S2 {s} CI 下限 = {lo:+.3f}", lo, c.get("lo"))
            add(f"S2 {s} CI 上限 = {hi:+.3f}", hi, c.get("hi"))
        add("S2 forge_19 paired array = 0.524", 0.524,
            d18.get("forge_19", {}).get("array"))

    # ---- Section S3, from the training histories
    d19 = _json(logs / "19_capacity_ema.json")
    if d19:
        runs = {k: v for k, v in d19.items() if not k.startswith("_")}
        add("S3 解析対象 = 44 ラン", 44, len(runs), 0.5)
        add("S3 学習可能パラメータ = 7,991,491", 7991491,
            d19.get("_trainable_parameters"), 0.5)
        es = sum(1 for r in runs.values() if r.get("early_stop_fired"))
        add("S3 dev split で停止 = 41 ラン", 41, es, 0.5)
        late = sorted(r.get("epochs_after_best") for r in runs.values()
                      if not r.get("early_stop_fired")
                      and r.get("epochs_after_best") is not None)
        flag("S3 停止しなかった3ランの最終改善は 4, 9, 18 エポック前",
             "4, 9, 18", ", ".join(str(x) for x in late) or "(なし)",
             late == [4, 9, 18])
        # script 19 の _training_sizes は語彙が違う。_benchmark_rows は
        # 記録ではなく局トレース（134,520）、_benchmark_events が記録
        # （9,803）、_train_events_per_run も記録数である。最初の版はこれを
        # 「記録」「地震」と取り違えて本文を疑ってしまった。
        sz = d19.get("_training_sizes") or {}
        if sz:
            add("S3 ベンチマークの記録数 = 9,803", 9803,
                sz.get("_benchmark_events"), 0.5)
            ev = list((sz.get("_train_events_per_run") or {}).values())
            if ev:
                add("S3 1ランの訓練記録数 下限 = 4,769", 4769, min(ev), 0.5)
                add("S3 1ランの訓練記録数 上限 = 5,932", 5932, max(ev), 0.5)
            rw = list((sz.get("_train_rows_per_run") or {}).values())
            if rw:
                add("S3 1ランの訓練トレース数 下限 = 58,464", 58464,
                    min(rw), 0.5)
                add("S3 1ランの訓練トレース数 上限 = 72,420", 72420,
                    max(rw), 0.5)
        tr = [r.get("post_peak_trend") for r in runs.values()]
        trv = [x for x in tr if x is not None]
        add("S3 trend を持つ = 43 ラン", 43, len(trv), 0.5)
        add("S3 trend が負 = 30 ラン", 30, sum(1 for x in trv if x < 0), 0.5)
        add("S3 trend の中央値 = -0.0018", -0.0018, med(trv), 5e-5)
        add("S3 trend / 変動 の中央値 = -0.36", -0.36,
            med([r.get("trend_over_noise") for r in runs.values()]), 5e-3)
        add("S3 epoch 間変動の中央値 = 0.0045", 0.0045,
            med([r.get("dev_epoch_noise") for r in runs.values()]), 5e-5)
        add("S3 dev スコアの中央値 = 0.93", 0.93,
            med([r.get("dev_f1_best") for r in runs.values()]), 5e-3)
        add("S3 最大からの落差の中央値 = 0.011", 0.011,
            med([r.get("dev_f1_drop_from_best") for r in runs.values()]), 5e-4)
        rat = [r["train_loss_first"] / r["train_loss_best"]
               for r in runs.values()
               if r.get("train_loss_first") and r.get("train_loss_best")
               and r["train_loss_best"] > 1e-5]
        add("S3 損失比が使える = 42 ラン", 42, len(rat), 0.5)
        if rat:
            add("S3 損失比の下限 = 22", 22, min(rat), 0.5)
            add("S3 損失比の上限 = 41", 41, max(rat), 0.5)
            add("S3 損失比の中央値 = 29", 29, med(rat), 0.5)

    # ---- Section S3's benchmark arithmetic, from the event accounting.
    # The counts the prose quotes are defined here, not in script 19's
    # _training_sizes, whose "events" are records and whose "rows" are
    # station traces.
    # script 21 の出力名はプロジェクトの版で 2 通りある。どちらでも読む。
    d21 = (_json(logs / "21_event_accounting.json")
           or _json(logs / "21_derived_counts.json"))
    # ファイルがあっても形が違えば検査は増えない。2026-10-08 の Drive 実行は
    # まさにそれで 7 件が黙って消え、合計が 90 ではなく 83 になった。中身が
    # 使えるかどうかで判定し、使えなければ何を見たかを出す。
    sites = s21_counts(d21)
    if True:                       # 入れ子を保つためのダミー条件ではない:
        if len(sites) == 8:        # 1 サイトでも欠けたら合計が狂うので else へ
            add("S3 記録の総数 = 9,803", 9803,
                sum(v["rec"] for v in sites.values()), 0.5)
            add("S3 地震の総数 = 3,421", 3421,
                sum(v["eq"] for v in sites.values()), 0.5)
            add("S3 雑音窓の総数 = 6,382", 6382,
                sum(v["noise"] for v in sites.values()), 0.5)
            eqt = {s: v["train_eq"] for s, v in sites.items()}
            rect = {s: v["train_rec"] for s, v in sites.items()}
            per_eq = [sum(eqt.values()) - eqt[s] for s in eqt]
            per_rec = [sum(rect.values()) - rect[s] for s in rect]
            add("S3 1ランの訓練地震数 下限 = 1,527", 1527, min(per_eq), 0.5)
            add("S3 1ランの訓練地震数 上限 = 2,002", 2002, max(per_eq), 0.5)
            add("S3 1ランの訓練記録数 下限 = 4,769", 4769, min(per_rec), 0.5)
            add("S3 1ランの訓練記録数 上限 = 5,932", 5932, max(per_rec), 0.5)
        else:
            seen = (", ".join(sorted(d21)[:6]) + " …") if d21 else "(ファイル無し)"
            flag("S3 ベンチマーク内訳の入力が使える"
                 "（21_event_accounting.json か 21_derived_counts.json）",
                 "8 サイト分の rows_total / noise_rows_total / rows_by_split",
                 f"{len(sites)}/8 サイトしか読めない。見えた最上位キー: {seen}",
                 False)

    d15 = _json(logs / "15_phase_confusion_ema.json")
    if d15:
        def p(site, conf, key, sub=None):
            r = (d15.get(f"{site}__{conf}") or {}).get("P", {})
            if sub:
                res = r.get("residuals", {}) or {}
                if sub == "median":
                    v = (res.get("own_ms_q25_med_q75") or [None] * 3)[1]
                    return v
                return res.get(sub)
            return r.get(key)
        add("S5 forge_19 array P: picks = 2,023", 2023,
            p("forge_19", "array", "n_picks"), tol=0.5)
        add("S5 forge_19 array P: on-phase = 0.181", 0.181,
            p("forge_19", "array", "frac_on_phase"))
        add("S5 forge_19 array P: cross-phase = 0.229", 0.229,
            p("forge_19", "array", "frac_cross"))
        add("S5 forge_19 array P: between = 0.482", 0.482,
            p("forge_19", "array", None, "frac_between_phases"))
        add("S5 forge_19 array P: median offset = +146 ms", 146.0,
            p("forge_19", "array", None, "median"), tol=0.5)
        add("S5 forge_19 per-trace P: on-phase = 0.824", 0.824,
            p("forge_19", "pertrace", "frac_on_phase"))
        add("S5 forge_19 per-trace P: cross-phase = 0.005", 0.005,
            p("forge_19", "pertrace", "frac_cross"))
        add("S5 mseel_5h array P: on-phase = 0.913", 0.913,
            p("mseel_5h", "array", "frac_on_phase"))
        s = (d15.get("forge_19__array") or {}).get("S", {})
        add("S5 forge_19 array S: on-phase = 0.824", 0.824,
            s.get("frac_on_phase"))
        add("S5 forge_19 array S: cross-phase = 0.053", 0.053,
            s.get("frac_cross"))

    # All of these are DERIVED from the bin counts, which are the primitive
    # script 24 always writes, so they do not depend on a convenience key the
    # log may predate.
    d24 = _json(logs / "24_moveout_ratio.json")
    if d24:
        a = d24.get("aneth")
        if a:
            # compared as the percentage the prose and Table S7 both print,
            # so the check means what the sentence says
            sh = s7_below_10ms("aneth", a)
            add("S6 aneth: share below 10 ms = 86.6 %", 86.6,
                None if sh is None else round(100 * sh, 1), tol=0.05)
            bins = _s7_bins("aneth", a)
            add("S6 aneth: events below 10 ms = 258",
                258, bins["1-5 ms"] + bins["5-10 ms"], tol=0.5)
            add("S6 aneth: events = 298", 298,
                sum(bins[b] for b in S7_BINS), tol=0.5)
        shares = [s for s in (s7_below_10ms(k, v) for k, v in d24.items()
                              if k != "aneth" and isinstance(v, dict)
                              and "stratified" in v) if s is not None]
        if shares:
            add("S6 other sites: maximum share = 2.4 %", 2.4,
                round(100 * max(shares), 1), tol=0.05)
            n_zero = sum(1 for s in shares if s == 0.0)
            out.append(("S6 four sites carry no event below 10 ms", 4,
                        n_zero, n_zero == 4))

    d20 = _json(logs / "20_within_array_ema.json")
    if d20:
        f = d20.get("forge_19", {})
        add("S4 forge_19 within-array array = 0.950", 0.950,
            f.get("array"), tol=5e-4)
        add("S4 forge_19 within-array per-trace = 0.952", 0.952,
            f.get("pertrace"), tol=5e-4)
        add("S4 forge_19 LOSO array = 0.516", 0.516,
            f.get("loso_array"), tol=5e-4)
        # summed without a default: a site whose count is absent makes the
        # total unknowable, and reporting a short total as if it were the
        # real one is the same fault as defaulting a share to zero
        counts = [v.get("n_test_events") for v in d20.values()
                  if isinstance(v, dict)]
        tot = None if any(c is None for c in counts) else sum(counts)
        add("S4 test records in all = 2,530", 2530, tot, tol=0.5)
        if counts:
            add("S4 最小の test split = 47", 47, min(counts), 0.5)
            add("S4 最大の test split = 1,258", 1258, max(counts), 0.5)
        gl = (f.get("loso_pertrace") or 0) - (f.get("loso_array") or 0)
        gw = (f.get("pertrace") or 0) - (f.get("array") or 0)
        add("S4 forge_19 の差 LOSO = 0.340", 0.340, gl)
        add("S4 forge_19 の差 within = 0.002", 0.002, gw)
        a20 = d20.get("aneth", {})
        add("S4 aneth の per-trace 優位 = 0.057", 0.057,
            (a20.get("loso_pertrace") or 0) - (a20.get("loso_array") or 0))
        add("S4 aneth の within 差 = +0.003", 0.003, a20.get("diff_median"))
        wv = [v.get("verdict") for v in d20.values() if isinstance(v, dict)]
        lv = [v.get("loso_verdict") for v in d20.values() if isinstance(v, dict)]
        flag("S4 within の判定 array 2 / per-trace 1 / tie 5", "2 / 1 / 5",
             f"{wv.count('array')} / {wv.count('per-trace')} / {wv.count('tie')}",
             (wv.count("array"), wv.count("per-trace"), wv.count("tie"))
             == (2, 1, 5))
        flag("S4 LOSO の判定 array 1 / per-trace 2 / tie 5", "1 / 2 / 5",
             f"{lv.count('array')} / {lv.count('per-trace')} / {lv.count('tie')}",
             (lv.count("array"), lv.count("per-trace"), lv.count("tie"))
             == (1, 2, 5))
        diffs = [abs(v.get("diff_median") or 0) for v in d20.values()
                 if isinstance(v, dict)]
        if diffs:
            add("S4 within の差の最大 = 0.017", 0.017, max(diffs))
            flag("S4 『forge_19 の差の20分の1』", "20",
                 f"{gl/max(diffs):.1f}" if max(diffs) else "-",
                 max(diffs) and abs(gl / max(diffs) - 20) < 0.6)
        m3 = d20.get("mseel_3h", {})
        add("S4 mseel_3h array の低下 = 0.040", 0.040,
            (m3.get("loso_array") or 0) - (m3.get("array") or 0))
        add("S4 mseel_3h per-trace の低下 = 0.042", 0.042,
            (m3.get("loso_pertrace") or 0) - (m3.get("pertrace") or 0))

    # A log that is absent silently removes its checks, and a shorter report
    # then looks like a clean one. On 2026-10-08 the run on Drive dropped
    # seven checks this way, because 21_event_accounting.json was not there,
    # and nothing said so. Each missing source is now a failed row.
    for obj, names in ((aud, ["16_arrival_audit_ema.json"]),
                       (res, ["16_clean_rescore_ema.json"]),
                       (d18, ["18_paired_scores_ema.json"]),
                       (d19, ["19_capacity_ema.json"]),
                       (d15, ["15_phase_confusion_ema.json"]),
                       (d24, ["24_moveout_ratio.json"]),
                       (d20, ["20_within_array_ema.json"])):
        if not obj:
            flag(f"検査の入力がある: {' か '.join(names)}", "あり", "無い", False)
    return out


# ============================================================================
# 6. The index. ONE definition, used twice: as this document's contents, and
#    as the list the manuscript's SUPPORTING INFORMATION section must carry.
#    build() writes it to logs/supplement/25_manuscript_index.txt so the two
#    can be compared mechanically instead of by eye.
# ============================================================================
# One line per item. These are the lines the manuscript's SUPPORTING
# INFORMATION section carries, verbatim, so the two documents cannot drift.
# A caption's own first sentence is not used here: some captions open with a
# long parenthetical that does not belong in an index.
INDEX = [
    "Section S1. Quality of the catalogued arrivals, and whether the results "
    "depend on it.",
    "Section S2. Evaluation of the two configurations on identical inputs.",
    "Section S3. Model capacity and overfitting.",
    "Section S4. Training with the held-out arrays represented.",
    "Section S5. Where the declared picks land: P/S assignment.",
    "Section S6. How much moveout each site actually carries.",
    "Figure S1. Example per-station P/S/noise score panel for a representative "
    "event.",
    "Figure S2. Training and development curves for all 44 runs.",
    "Figure S3. Summary of held-out metrics.",
    "Figure S4. Off-the-shelf surface-trained baseline applied to a borehole "
    "site.",
    "Figure S5. Zero-shot transfer of the trained array picker to decimated "
    "arrays.",
    "Figure S6. Physical audit of the catalogued arrivals.",
    "Figure S7. The two configurations evaluated on identical inputs.",
    "Figure S8. Picking performance when the held-out arrays are represented "
    "in training.",
    "Figure S9. Where the declared picks land, against both catalogued phases.",
    "Figure S10. Waveforms and predicted scores at the remaining held-out "
    "sites.",
    "Table S1. Per-site array geometry.",
    "Table S2. Catalogued-arrival audit and re-scoring on the surviving events.",
    "Table S3. Site-level scores under the submitted and the paired evaluation.",
    "Table S4. Per-run training summary.",
    "Table S5. Site-level scores under the leave-one-site-out and within-array "
    "protocols.",
    "Table S6. Where the declared P-channel picks land, by site and "
    "configuration.",
    "Table S7. Distribution of the catalogued P moveout across events, by site.",
]


def manuscript_index() -> list:
    """The index lines, in the order the manuscript lists them."""
    return list(INDEX)


# ============================================================================
# 7. Rendering
# ============================================================================
def _styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=14,
                                leading=18, spaceAfter=10),
        "part": ParagraphStyle("pt", parent=ss["Heading1"], fontSize=13,
                               leading=16, spaceBefore=4, spaceAfter=10),
        "head": ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5,
                               leading=14, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5,
                               leading=13, alignment=TA_JUSTIFY,
                               spaceAfter=7),
        "cap": ParagraphStyle("c", parent=ss["BodyText"], fontSize=9,
                              leading=12, spaceAfter=8),
        "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.4,
                                leading=9),
    }


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _pdf_text(out: Path) -> None:
    """Part A: title page, preamble, contents, Sections S1-S6."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    PageBreak)
    st = _styles()
    flow = [Paragraph(_esc(TITLE), st["title"]),
            Paragraph(_esc(AUTHOR), st["cap"]),
            Spacer(1, 10), Paragraph(_esc(PREAMBLE), st["body"]),
            Spacer(1, 10), Paragraph("Contents", st["head"])]
    for line in manuscript_index():
        flow.append(Paragraph(_esc(line), st["cap"]))
    flow.append(PageBreak())
    flow.append(Paragraph("Supplementary text", st["part"]))
    for num, head, text in SECTIONS:
        flow.append(Paragraph(f"Section {num}. {_esc(head)}", st["head"]))
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            flow.append(Paragraph(_esc(" ".join(para.split())), st["body"]))
    SimpleDocTemplate(str(out), pagesize=A4,
                      leftMargin=48, rightMargin=48,
                      topMargin=48, bottomMargin=40,
                      title="Supporting information",
                      author=AUTHOR).build(flow)


def _table_flowable(head, rows, st):
    """A reportlab table sized to the page, with the header row repeated."""
    from reportlab.lib import colors
    from reportlab.platypus import LongTable, Paragraph
    data = [[Paragraph(f"<b>{_esc(h)}</b>", st["small"]) for h in head]]
    for r in rows:
        data.append([Paragraph(_esc(c), st["small"]) for c in r])
    t = LongTable(data, repeatRows=1, hAlign="LEFT")
    t.setStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])
    return t


def _pdf_tables(out: Path, tables: dict) -> None:
    """Part B: Tables S1-S7, landscape so a wide table is not cut."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    PageBreak)
    st = _styles()
    flow = [Paragraph("Supplementary tables", st["part"])]
    for i, n in enumerate(sorted(tables)):
        head, rows = tables[n]
        if i:
            flow.append(PageBreak())
        flow.append(Paragraph(_esc(TABLE_CAPTIONS[n]), st["cap"]))
        flow.append(Spacer(1, 4))
        flow.append(_table_flowable(head, rows, st))
    SimpleDocTemplate(str(out), pagesize=landscape(A4),
                      leftMargin=30, rightMargin=30,
                      topMargin=34, bottomMargin=28,
                      title="Supplementary tables",
                      author=AUTHOR).build(flow)


def _source_size(src: Path):
    """(width, height) of a figure file, in its own units."""
    if src.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        box = PdfReader(str(src)).pages[0].mediabox
        return float(box.width), float(box.height)
    from PIL import Image
    with Image.open(src) as im:
        return float(im.size[0]), float(im.size[1])


def _figure_page(out: Path, caption: str, src: Path,
                 part_head: str | None) -> None:
    """One page: the caption at the top, the figure below it, scaled to fit.

    A vector figure is composited, not rasterized, so the merged page keeps the
    original's quality. Giving the caption a page of its own would be simpler
    but would add ten nearly empty pages to the submission.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import Paragraph
    from reportlab.pdfgen import canvas

    st = _styles()
    fw, fh = _source_size(src)
    margin = 36

    def layout(page):
        """Free area under the caption on this page size, and the fit scale."""
        pw_, ph_ = page
        top = ph_ - margin
        heads = []
        if part_head:
            p_ = Paragraph(part_head, st["part"])
            _, h_ = p_.wrap(pw_ - 2 * margin, ph_)
            heads.append((p_, h_, 8))
        p_ = Paragraph(_esc(caption), st["cap"])
        _, h_ = p_.wrap(pw_ - 2 * margin, ph_)
        heads.append((p_, h_, 10))
        for _, h_, gap in heads:
            top -= h_ + gap
        aw, ah = pw_ - 2 * margin, top - margin
        return heads, aw, ah, min(aw / fw, ah / fh)

    # Pick the orientation that renders the figure larger. A near-square
    # figure fits better on a portrait page even though it is slightly wider
    # than it is tall, so the aspect ratio alone is the wrong test.
    cands = [(A4, layout(A4)), (landscape(A4), layout(landscape(A4)))]
    page, (heads, avail_w, avail_h, scale) = max(cands, key=lambda x: x[1][3])
    pw, ph = page

    tmp = out.with_name(out.stem + "_cap.pdf")
    c = canvas.Canvas(str(tmp), pagesize=page)
    y = ph - margin
    for p, hh, gap in heads:
        p.drawOn(c, margin, y - hh)
        y -= hh + gap
    dw, dh = fw * scale, fh * scale
    ox, oy = (pw - dw) / 2, margin + (avail_h - dh) / 2

    if src.suffix.lower() != ".pdf":
        c.drawImage(str(src), ox, oy, dw, dh)
        c.showPage()
        c.save()
        tmp.replace(out)
        return

    c.showPage()
    c.save()
    from pypdf import PdfReader, PdfWriter, Transformation
    base = PdfReader(str(tmp)).pages[0]
    fig = PdfReader(str(src)).pages[0]
    box = fig.mediabox
    t = (Transformation()
         .translate(-float(box.left), -float(box.bottom))
         .scale(scale)
         .translate(ox, oy))
    base.merge_transformed_page(fig, t)
    w = PdfWriter()
    w.add_page(base)
    with out.open("wb") as fh:
        w.write(fh)
    w.close()
    tmp.unlink(missing_ok=True)


def _find_figure(num: int, dirs) -> Path | None:
    for name in FIGURE_SOURCE[num]:
        for d in dirs:
            p = d / name
            if p.exists():
                return p
    return None


# ============================================================================
# 7. Driver
# ============================================================================
def inventory(logs: Path, figdirs) -> dict:
    rep = {"logs_dir": str(logs), "figure_dirs": [str(d) for d in figdirs],
           "tables": {}, "figures": {}}
    print(f"[25] logs   : {logs}")
    for d in figdirs:
        print(f"[25] figures: {d}" + ("" if d.exists() else "   (absent)"))
    print("\n--- tables")
    for n in sorted(TABLE_SOURCE):
        need = TABLE_SOURCE[n]
        have = [(logs / f).exists() for f in need]
        rep["tables"][n] = {"needs": need, "present": have}
        mark = "OK " if all(have) else "NG "
        miss = [f for f, h in zip(need, have) if not h]
        print(f"  {mark} Table S{n:<2d} {', '.join(need)}"
              + (f"   missing: {', '.join(miss)}" if miss else ""))
    print("\n--- figures")
    for n in sorted(FIGURE_SOURCE):
        p = _find_figure(n, figdirs)
        rep["figures"][n] = {"candidates": FIGURE_SOURCE[n],
                             "found": str(p) if p else None}
        print(f"  {'OK ' if p else 'NG '} Figure S{n:<2d} "
              + (str(p) if p else f"none of {', '.join(FIGURE_SOURCE[n])}"))
    return rep


def build(logs: Path, figdirs, pdfdir: Path, outdir: Path) -> int:
    """Returns 0 when every item was built, 1 when something was missing."""
    from pypdf import PdfWriter

    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "_parts"
    work.mkdir(exist_ok=True)
    manifest = {"tables": {}, "figures": {}, "missing": []}

    # ---- tables
    tables = {}
    for n in sorted(TABLE_BUILDERS):
        built = TABLE_BUILDERS[n](logs)
        if built is None:
            manifest["tables"][n] = {"built": False,
                                     "needs": TABLE_SOURCE[n]}
            manifest["missing"].append(f"Table S{n} ({', '.join(TABLE_SOURCE[n])})")
            print(f"[25] Table S{n}: NOT BUILT -- needs "
                  f"{', '.join(TABLE_SOURCE[n])}")
            continue
        head, rows = built
        tables[n] = built
        csv_path = outdir / f"TableS{n}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)
        manifest["tables"][n] = {"built": True, "rows": len(rows),
                                 "columns": len(head), "csv": csv_path.name}
        print(f"[25] Table S{n}: {len(rows)} rows x {len(head)} cols "
              f"-> {csv_path.name}")

    # ---- the three rendered parts
    part_text = work / "partA_text.pdf"
    _pdf_text(part_text)
    order = [part_text]
    if tables:
        part_tab = work / "partB_tables.pdf"
        _pdf_tables(part_tab, tables)
        order.append(part_tab)

    first_fig = True
    for n in sorted(FIGURE_CAPTIONS):
        src = _find_figure(n, figdirs)
        if src is None:
            manifest["figures"][n] = {"built": False,
                                      "candidates": FIGURE_SOURCE[n]}
            manifest["missing"].append(
                f"Figure S{n} ({' or '.join(FIGURE_SOURCE[n])})")
            print(f"[25] Figure S{n}: NOT FOUND -- "
                  f"{' or '.join(FIGURE_SOURCE[n])}")
            continue
        page = work / f"figS{n}.pdf"
        _figure_page(page, FIGURE_CAPTIONS[n], src,
                     "Supplementary figures" if first_fig else None)
        first_fig = False
        order.append(page)
        manifest["figures"][n] = {"built": True, "source": str(src)}
        print(f"[25] Figure S{n}: {src.name}")

    complete = not manifest["missing"]
    name = ("Supplement_MOIRAI_L3.pdf" if complete
            else "Supplement_MOIRAI_L3_INCOMPLETE.pdf")
    pdfdir.mkdir(parents=True, exist_ok=True)
    final = pdfdir / name
    w = PdfWriter()
    for p in order:
        w.append(str(p))
    with final.open("wb") as fh:
        w.write(fh)
    w.close()
    manifest["output"] = str(final)
    manifest["complete"] = complete
    try:
        from pypdf import PdfReader
        manifest["pages"] = len(PdfReader(str(final)).pages)
    except Exception:                                   # noqa: BLE001
        manifest["pages"] = None

    # ---- the prose numbers against the logs
    checks = check_numbers(logs)
    md = ["# Numbers quoted in the supplementary prose, against the logs", "",
          "| Quantity | In the text | In the log | |", "|---|---|---|---|"]
    for label, exp, got, ok in checks:
        md.append(f"| {label} | {exp} | "
                  f"{'-' if got is None else got} | {'OK' if ok else 'NG'} |")
    if not checks:
        md.append("| (no log present to check against) | - | - | - |")
    (outdir / "25_number_check.md").write_text("\n".join(md) + "\n")
    bad = [c for c in checks if not c[3]]
    manifest["number_checks"] = {"total": len(checks), "failed": len(bad)}

    # the index the manuscript must carry, so the two can be diffed
    (outdir / "25_manuscript_index.txt").write_text(
        "\n".join(manuscript_index()) + "\n")
    manifest["manuscript_index_lines"] = len(manuscript_index())

    (outdir / "25_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[25] wrote {final}  ({manifest['pages']} pages)")
    print(f"[25] wrote {outdir / '25_manifest.json'} and 25_number_check.md")
    print(f"[25] prose numbers checked: {len(checks)}, failed: {len(bad)}")
    for label, exp, got, _ in bad:
        print(f"       NG {label}: log gives {got}")
    if manifest["missing"]:
        print(f"\n[25] INCOMPLETE -- {len(manifest['missing'])} item(s) missing:")
        for m in manifest["missing"]:
            print(f"       {m}")
        print("[25] run the scripts that produce them, then build again.")
    return 0 if (complete and not bad) else 1


# ============================================================================
# 8. self-test (synthetic logs and figures; no AMBER, no checkpoints, no GPU)
# ============================================================================
# Abbreviations that end in a full stop. Splitting on ". " without protecting
# these cuts "Fig. S9 shows ..." in two, and a long sentence then hides behind
# the cut -- which is what happened on the first run of this check.
_ABBR = re.compile(r"\b(Figs?|Eqs?|Tab|No|cf|vs|al|approx)\.\s")


def sentences(text: str) -> list:
    """The sentences of text, with abbreviation full stops left alone."""
    t = _ABBR.sub(lambda m: m.group(1) + ".\x00", " ".join(text.split()))
    return [s.replace("\x00", " ").strip()
            for s in re.split(r"(?<=[.;])\s+", t) if s.strip()]


def selftest() -> None:
    import tempfile
    import numpy as np

    # the splitter itself, before it is trusted to police the prose
    assert len(sentences("See Fig. 6 of the main text. Then stop.")) == 2
    assert len(sentences("Eq. 1 holds; Eq. 2 does not.")) == 2
    assert sentences("A b c. D e.") == ["A b c.", "D e."]

    # --- the fixed text must obey the paper's own English rules
    long_s, n_sent = [], 0
    for num, head, text in SECTIONS:
        for s in sentences(text):
            n_sent += 1
            if len(s.split()) > 25:
                long_s.append(f"S{num} {len(s.split())}w: {s[:70]}")
    assert not long_s, "25語超の文:\n  " + "\n  ".join(long_s)
    assert n_sent > 100, n_sent
    for num, head, text in SECTIONS:
        assert not re.search(r"\bI\b", text), f"S{num} に一人称 I がある"
        assert text.isascii(), f"S{num} に非ASCII文字がある"
    for d in (TABLE_CAPTIONS, FIGURE_CAPTIONS):
        for n, c in d.items():
            assert c.isascii(), (n, c)
            assert c.rstrip().endswith("."), (n, c)
    assert sorted(TABLE_CAPTIONS) == list(range(1, 8))
    assert sorted(FIGURE_CAPTIONS) == list(range(1, 11))
    assert sorted(FIGURE_SOURCE) == list(range(1, 11))
    assert sorted(TABLE_BUILDERS) == sorted(TABLE_SOURCE) == list(range(1, 8))
    assert [n for n, _, _ in SECTIONS] == [f"S{i}" for i in range(1, 7)]
    # Every section must point at its own table and figure, so the supplement
    # reads on its own. Stated as a map rather than a pattern search: "S6" also
    # matches "Section S6", which let the first version of this check pass
    # while Section S3 cited neither Fig. S2 nor Table S4.
    own = {"S1": ([2], [6]), "S2": ([3], [7]), "S3": ([4], [2]),
           "S4": ([5], [8]), "S5": ([6], [9]), "S6": ([7], [])}
    bodies = {num: " ".join(t.split()) for num, _, t in SECTIONS}
    for num, (tabs, figs) in own.items():
        for n in tabs:
            assert f"Table S{n}" in bodies[num], \
                f"Section {num} が Table S{n} を引いていない"
        for n in figs:
            assert re.search(rf"Fig(?:s?\.|ure)\s*S{n}\b", bodies[num]), \
                f"Section {num} が Fig. S{n} を引いていない"
    # Figures S1, S3, S4, S5 and S10 have no section of their own: the main
    # text is where they are referenced, and verify_r2.py checks that there.
    assert (set(FIGURE_CAPTIONS)
            - {f for _, fs in own.values() for f in fs}) == {1, 3, 4, 5, 10}
    # Table S1 likewise belongs to Section 2 of the main text, not to a
    # supplementary section.
    assert (set(TABLE_CAPTIONS)
            - {t for ts, _ in own.values() for t in ts}) == {1}

    # the index: one line per item, every item once, nothing extra
    idx = manuscript_index()
    assert len(idx) == 23, len(idx)
    assert all(x.isascii() and x.rstrip().endswith(".") for x in idx), idx
    for kind, hi in (("Section", 6), ("Figure", 10), ("Table", 7)):
        nums = [int(m.group(1)) for x in idx
                for m in [re.match(rf"^{kind} S(\d+)\.", x)] if m]
        assert nums == list(range(1, hi + 1)), (kind, nums)
    # an index line must agree with the item it names
    for n, cap in FIGURE_CAPTIONS.items():
        line = next(x for x in idx if x.startswith(f"Figure S{n}."))
        first = sentences(cap[cap.index(".") + 1:])[0].rstrip(".")
        assert first.lower().startswith(
            line[len(f"Figure S{n}."):].strip().rstrip(".").lower()[:28]), \
            f"Figure S{n} の索引行とキャプションが食い違う:\n  {line}\n  {first}"
    for n, cap in TABLE_CAPTIONS.items():
        line = next(x for x in idx if x.startswith(f"Table S{n}."))
        first = sentences(cap[cap.index(".") + 1:])[0].rstrip(".")
        assert first.lower().startswith(
            line[len(f"Table S{n}."):].strip().rstrip(".").lower()[:22]), \
            f"Table S{n} の索引行とキャプションが食い違う:\n  {line}\n  {first}"

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        logs, figs = root / "logs", root / "figs"
        logs.mkdir(); figs.mkdir()

        # ---- synthetic logs in the exact shapes the real scripts write
        with (logs / "14_table_s1.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["site", "dataset", "n_stations", "vertical_extent_m",
                        "aperture_3d_m", "median_adjacent_spacing_m"])
            for s in SITE_ORDER:
                w.writerow([s, s.upper(), 12, 335.3, 335.3, 30.5])

        # The S1, S2 and S4 fixtures carry the MEASURED values, taken from the
        # 2026-10-08 build. A fixture of round numbers exercises the plumbing
        # but proves nothing about the claims in the prose; with the real
        # values the check battery below is a regression suite that runs
        # without Drive.
        #
        # site: (earthquakes, flagged, gap step, reversal test applied,
        #        all array, all per-trace, clean array, clean per-trace,
        #        dropped)
        S1_REAL = {
            "pnr-1":          (543,   6, 11.3, True,  .986, .986, .990, .992,  10),
            "mseel_3h":       (684, 107,  5.8, True,  .891, .888, .945, .941, 125),
            "mseel_5h":       (512,  90,  5.0, True,  .897, .881, .940, .928,  98),
            # clean per-trace はログの実精度（上昇が 0.0744 になる値）。
            # 3桁に丸めた .944 だと 0.075 になり、本文と1桁ずれる。
            "clearfield_mw6": (612, 128,  5.0, True,  .875, .869, .945, .9434, 147),
            "pnr-2":          (305,  18,  6.4, True,  .944, .901, .970, .929,  25),
            "clearfield_mw4": (254,  50,  5.4, True,  .852, .836, .923, .894,  60),
            "aneth":          (298,  21,  1.2, False, .884, .941, .895, .948,  25),
            "forge_19":       (213,   1, 14.3, True,  .516, .856, .517, .858,   2),
        }
        # 清浄化前の生の P moveout 最大（実測）
        RAW_MAX_MS = {"pnr-1": 365.0, "mseel_3h": 767.0, "mseel_5h": 692.0,
                      "clearfield_mw6": 909.0, "pnr-2": 608.0,
                      "clearfield_mw4": 648.0, "aneth": 554.0,
                      "forge_19": 82.0}
        (logs / "16_arrival_audit_ema.json").write_text(json.dumps(
            {s: dict(n_events=v[0], n_stations=12, flagged_events=v[1],
                     flagged_fraction=v[1] / v[0],
                     median_gap_step_samples=v[2],
                     reversal_test_applied=v[3],
                     by_criterion=dict(slow_P=v[1], slow_S=0, reversals=0,
                                       sp_le_zero=0),
                     before={"P": dict(n=v[0], min_ms=1.0, median_ms=30.0,
                                       max_ms=RAW_MAX_MS[s])},
                     after={"P": dict(n=v[0] - v[1], min_ms=1.0,
                                      median_ms=29.0, max_ms=80.0)},
                     # 件数で重みづけした全体の割合が 9 % と 1 % になる値
                     window_position_flagged=dict(
                         n=v[1], median_samples_from_edge=400.0,
                         frac_within_edge_band=0.08),
                     window_position_clean=dict(
                         n=v[0] - v[1], median_samples_from_edge=500.0,
                         frac_within_edge_band=0.01))
             for s, v in S1_REAL.items()}))
        (logs / "16_clean_rescore_ema.json").write_text(json.dumps(
            {s: {"all": dict(n_events=v[0], array=v[4], pertrace=v[5],
                             verdict="tie"),
                 "clean": dict(n_events=v[0] - v[8], array=v[6],
                               pertrace=v[7], verdict="tie"),
                 "n_dropped": v[8], "verdict_survives": True}
             for s, v in S1_REAL.items()}))

        # site: (events, submitted array, submitted per-trace, paired array,
        #        paired per-trace, diff, lo, hi, paired verdict,
        #        submitted verdict)
        S2_REAL = {
            "pnr-1":          (1258, .986, .986, .986, .985, +.002, -.001, +.004, "tie", "tie"),
            "mseel_3h":       (1684, .891, .888, .900, .901, -.001, -.007, +.003, "tie", "tie"),
            "mseel_5h":       (1512, .897, .881, .889, .880, +.009, +.001, +.017, "array", "tie"),
            "clearfield_mw6": (1612, .875, .869, .870, .868, +.002, -.005, +.008, "tie", "tie"),
            "pnr-2":          ( 972, .944, .901, .940, .904, +.036, +.027, +.046, "array", "array"),
            "clearfield_mw4": (1254, .852, .836, .858, .839, +.020, +.009, +.030, "array", "tie"),
            "aneth":          ( 298, .884, .941, .887, .942, -.055, -.065, -.047, "per-trace", "per-trace"),
            "forge_19":       (1213, .516, .856, .524, .856, -.332, -.355, -.310, "per-trace", "per-trace"),
        }
        (logs / "18_paired_scores_ema.json").write_text(json.dumps(
            {s: dict(n_events=v[0], array=v[3], pertrace=v[4],
                     paired_ci={"orig": dict(median=v[5], lo=v[6], hi=v[7])},
                     marginal_ci={}, verdict_marginal="tie",
                     verdict_paired=v[8], submitted_array=v[1],
                     submitted_pertrace=v[2], submitted_verdict=v[9],
                     reversal_vs_submitted=False)
             for s, v in S2_REAL.items()}))

        # 44 runs. The aggregates the prose quotes are reproduced exactly:
        # 41 early-stopped, 43 with a trend of which 30 are negative, median
        # trend -0.0018 at -0.36 of a median variation of 0.0045, median dev
        # score 0.93, median fall 0.011, and 42 usable loss ratios spanning
        # 22 to 41 with a median of 29. The per-run spread is synthetic; only
        # the real logs can settle the per-run numbers, which is why the same
        # battery runs again on Drive.
        keys = ([(s, m) for s in SITE_ORDER for m in ("array", "pertrace")]
                + [(s, "shuffle") for s in SITE_ORDER[:4]]
                + [(s, f"seed{k}") for s in SITE_ORDER[:8] for k in (1, 2, 3)])
        keys = keys[:44]
        assert len(keys) == 44, len(keys)
        # (trend, noise) in four groups. The three medians cannot all come
        # from one group -- the median trend over the median variation is
        # -0.40, not the -0.36 the runs actually give -- so the groups are
        # ordered differently in each field, exactly as real runs are.
        groups = ([(-0.0018, 0.0045)] * 21 + [(-0.0018, 0.0050)]
                  + [(-0.0030, 0.0100)] * 8 + [(+0.0005, 0.0010)] * 13)
        assert len(groups) == 43, len(groups)
        # 42 usable loss ratios: 22 to 41 with a median of 29
        ratios = [22.0] * 20 + [29.0] * 2 + [41.0] * 20
        assert len(ratios) == 42, len(ratios)
        cap = {}
        for i, (s, m) in enumerate(keys):
            trend, noise = groups[i] if i < 43 else (None, None)
            cap[f"{s}__{m}"] = dict(
                usable=True, n_epochs=40 + i, best_epoch=20 + i // 2,
                dev_f1_best=0.93, dev_f1_last=0.919,
                dev_f1_drop_from_best=0.011,
                post_peak_trend=trend, dev_epoch_noise=noise,
                trend_over_noise=(None if trend is None else trend / noise),
                train_loss_first=0.03,
                # the first two runs have an unusable best loss, zero and
                # three parts in a million, so 42 ratios remain
                train_loss_best=(0.0 if i == 0 else 3e-6 if i == 1
                                 else 0.03 / ratios[i - 2]),
                train_loss_last=0.0007,
                epochs_after_best=({41: 4, 42: 9, 43: 18}.get(i, 20)),
                early_stop_fired=(i < 41))
        _tr = [r["post_peak_trend"] for r in cap.values()
               if r["post_peak_trend"] is not None]
        assert len(_tr) == 43 and sum(1 for x in _tr if x < 0) == 30, \
            (len(_tr), sum(1 for x in _tr if x < 0))
        (logs / "19_capacity_ema.json").write_text(json.dumps(
            cap | {"_trainable_parameters": 7991491,
                   "_training_sizes": {
                       # 19 の語彙: rows は局トレース、events は記録
                       "_benchmark_rows": 134520,
                       "_benchmark_events": 9803,
                       "_train_events_per_run": {
                           s: (4769 if i == 0 else 5932 if i == 1
                               else 5000)
                           for i, s in enumerate(SITE_ORDER)},
                       "_train_rows_per_run": {
                           s: (58464 if i == 0 else 72420 if i == 1
                               else 65000)
                           for i, s in enumerate(SITE_ORDER)}}}))

        # site: (test events, LOSO array, LOSO per-trace, within array,
        #        within per-trace, diff, lo, hi, LOSO verdict, within verdict)
        S4_REAL = {
            "pnr-1":          (1258, .986, .986, .985, .986, -.001, -.005, +.002, "tie", "tie"),
            "mseel_3h":       ( 253, .891, .888, .851, .846, +.005, -.004, +.016, "tie", "tie"),
            "mseel_5h":       ( 221, .897, .881, .948, .941, +.006, +.001, +.014, "tie", "array"),
            "clearfield_mw6": ( 228, .875, .869, .871, .890, -.017, -.036, -.003, "tie", "per-trace"),
            "pnr-2":          ( 159, .944, .901, .975, .965, +.010, +.002, +.020, "array", "array"),
            "clearfield_mw4": ( 171, .852, .836, .903, .909, -.006, -.019, +.005, "tie", "tie"),
            "aneth":          (  47, .884, .941, .974, .971, +.003, -.003, +.009, "per-trace", "tie"),
            "forge_19":       ( 193, .516, .856, .950, .952, -.002, -.011, +.008, "per-trace", "tie"),
        }
        # the real split counts, which are what Section S3's benchmark
        # arithmetic is defined on (script 21)
        EQ_TRAIN = {"pnr-1": 0, "mseel_3h": 475, "mseel_5h": 364,
                    "clearfield_mw6": 431, "pnr-2": 207,
                    "clearfield_mw4": 174, "aneth": 206, "forge_19": 145}
        NW_TRAIN = {"pnr-1": 0, "mseel_3h": 688, "mseel_5h": 679,
                    "clearfield_mw6": 717, "pnr-2": 446,
                    "clearfield_mw4": 712, "aneth": 0, "forge_19": 688}
        EQ_TOT = {s: v[0] for s, v in S1_REAL.items()}
        NW_TOT = {"pnr-1": 715, "mseel_3h": 1000, "mseel_5h": 1000,
                  "clearfield_mw6": 1000, "pnr-2": 667,
                  "clearfield_mw4": 1000, "aneth": 0, "forge_19": 1000}
        (logs / "21_event_accounting.json").write_text(json.dumps(
            {s: dict(st=12, rec=EQ_TOT[s] + NW_TOT[s], nw_tot=NW_TOT[s],
                     eq_tot=EQ_TOT[s],
                     eq={"train": EQ_TRAIN[s], "dev": 0,
                         "test": EQ_TOT[s] - EQ_TRAIN[s]},
                     nw={"train": NW_TRAIN[s], "dev": 0,
                         "test": NW_TOT[s] - NW_TRAIN[s]},
                     clean=EQ_TOT[s], audit=EQ_TOT[s])
             for s in SITE_ORDER}))

        assert sum(v[0] for v in S4_REAL.values()) == 2530
        (logs / "20_within_array_ema.json").write_text(json.dumps(
            {s: dict(n_test_events=v[0], array=v[3], pertrace=v[4],
                     diff_median=v[5], diff_ci=[v[6], v[7]], verdict=v[9],
                     loso_array=v[1], loso_pertrace=v[2], loso_verdict=v[8])
             for s, v in S4_REAL.items()}))
        (logs / "15_phase_confusion_ema.json").write_text(json.dumps(
            {f"{s}__{m}": dict(
                site=s, config=m,
                P=dict(n_picks=(2023 if (s, m) == ("forge_19", "array") else 3000),
                       frac_on_phase=(0.181 if (s, m) == ("forge_19", "array")
                                      else 0.824 if (s, m) == ("forge_19", "pertrace")
                                      else 0.913 if (s, m) == ("mseel_5h", "array")
                                      else 0.9),
                       frac_cross=(0.229 if (s, m) == ("forge_19", "array")
                                   else 0.005),
                       frac_elsewhere=0.1,
                       frac_cross_of_missed=0.2,
                       residuals=dict(n=100,
                                      own_ms_q25_med_q75=[0.0, (146.0 if (s, m) == ("forge_19", "array") else 0.0), 1.0],
                                      other_ms_q25_med_q75=[0.0, 0.0, 1.0],
                                      frac_between_phases=(0.482 if (s, m) == ("forge_19", "array") else 0.01))),
                S=dict(n_picks=3000,
                       frac_on_phase=(0.824 if (s, m) == ("forge_19", "array") else 0.9),
                       frac_cross=(0.053 if (s, m) == ("forge_19", "array") else 0.004),
                       frac_elsewhere=0.1, frac_cross_of_missed=0.1,
                       residuals=dict(n=100, own_ms_q25_med_q75=[0.0, 0.0, 1.0],
                                      other_ms_q25_med_q75=[0.0, 0.0, 1.0],
                                      frac_between_phases=0.01)))
             for s in SITE_ORDER for m in ("array", "pertrace")}))
        s7 = {"aneth": [61, 197, 29, 0, 11], "pnr-1": [0, 0, 0, 0, 543],
              "mseel_3h": [0, 0, 63, 479, 142], "mseel_5h": [0, 4, 121, 282, 105],
              "clearfield_mw6": [0, 8, 172, 323, 109], "pnr-2": [0, 0, 0, 273, 32],
              "clearfield_mw4": [0, 6, 65, 118, 65],
              "forge_19": [0, 0, 0, 0, 213]}
        (logs / "24_moveout_ratio.json").write_text(json.dumps(
            {s: dict(n=sum(v), median=1.7, q25=1.5, q75=1.9,
                     stratified=dict(
                         bins=[dict(bin=b, n=k, median=1.7,
                                    p_moveout_median=10.0)
                               for b, k in zip(S7_BINS, v)],
                         well_resolved=dict(threshold_ms=20.0, n=sum(v[2:]),
                                            median=1.7)),
                     frac_P_below_10ms=round(sum(v[:2]) / sum(v), 4),
                     vpvs_string=1.7, verdict="CONSISTENT")
             for s, v in s7.items()}))

        # ---- synthetic figures: one PDF, one PNG, so both paths are exercised
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n, names in FIGURE_SOURCE.items():
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.plot(np.arange(10), np.arange(10) ** 0.5)
            ax.set_xlabel("index")
            ax.set_ylabel("value")
            ax.set_title(f"synthetic Figure S{n}")
            fig.tight_layout()
            name = names[0] if n % 2 else names[1]
            fig.savefig(figs / name, dpi=110)
            plt.close(fig)

        out = root / "out"
        rep = inventory(logs, [figs])
        assert all(all(v["present"]) for v in rep["tables"].values()), rep
        assert all(v["found"] for v in rep["figures"].values()), rep

        rc = build(logs, [figs], root / "PDF", out)
        assert rc == 0, "完全な入力なのに不完全と判定された"
        final = root / "PDF" / "Supplement_MOIRAI_L3.pdf"
        assert final.exists() and final.stat().st_size > 30_000, final
        man = json.loads((out / "25_manifest.json").read_text())
        assert man["complete"] is True, man
        assert man["number_checks"]["failed"] == 0, man["number_checks"]
        assert man["number_checks"]["total"] >= 18, man["number_checks"]
        assert all(v["built"] for v in man["tables"].values())
        assert all(v["built"] for v in man["figures"].values())
        # contents + text + one page per table + one page per figure
        assert man["pages"] >= 1 + 4 + 7 + 10, man["pages"]
        # Table S1 must carry the published headings, not the CSV's own
        with (out / "TableS1.csv").open(newline="") as fh:
            r1 = list(csv.reader(fh))
        assert r1[0] == [pub for _, pub in S1_COLUMNS], r1[0]
        assert [r[0] for r in r1[1:]] == SITE_ORDER, [r[0] for r in r1[1:]]
        for n in range(1, 8):
            p = out / f"TableS{n}.csv"
            assert p.exists(), p
            with p.open(newline="") as fh:
                rows = list(csv.reader(fh))
            assert len(rows) >= 2, (n, rows)
            assert len(set(len(r) for r in rows)) == 1, (n, "列数が揃わない")
        # Table S7 must carry the real aneth distribution, not a rounded one
        with (out / "TableS7.csv").open(newline="") as fh:
            rows = {r[0]: r for r in csv.reader(fh)}
        assert rows["aneth"][2:7] == ["61", "197", "29", "-", "11"], rows["aneth"]
        assert rows["aneth"][-1] == "86.6", rows["aneth"]
        # the whole column, so a zeroed one cannot pass on aneth alone
        want_share = ["0.0", "0.0", "0.8", "1.3", "0.0", "2.4", "86.6", "0.0"]
        assert [rows[s][-1] for s in SITE_ORDER] == want_share, \
            [rows[s][-1] for s in SITE_ORDER]

        # ---- REGRESSION (2026-10-08): a log written before script 24 gained
        # frac_P_below_10ms must still give 86.6 per cent, not 0.0. The first
        # version read that key with a default of 0.0, which silently zeroed
        # the whole column of the built Table S7 on the real Drive log.
        f24 = logs / "24_moveout_ratio.json"
        d24 = json.loads(f24.read_text())
        for v in d24.values():
            v.pop("frac_P_below_10ms", None)
        f24.write_text(json.dumps(d24))
        built = table_s7(logs)
        assert built is not None
        by_site = {r[0]: r for r in built[1]}
        assert [by_site[s][-1] for s in SITE_ORDER] == want_share, \
            [by_site[s][-1] for s in SITE_ORDER]
        ck = {c[0]: c for c in check_numbers(logs)}
        for label in ("S6 aneth: share below 10 ms = 86.6 %",
                      "S6 other sites: maximum share = 2.4 %",
                      "S6 four sites carry no event below 10 ms"):
            assert label in ck and ck[label][3], (label, ck.get(label))

        # ---- a logged share that disagrees with the bins must stop the build
        d24["aneth"]["frac_P_below_10ms"] = 0.5
        f24.write_text(json.dumps(d24))
        try:
            table_s7(logs)
        except AssertionError:
            pass
        else:
            raise SystemExit("NG: bin と矛盾する frac_P_below_10ms が見逃された")
        for v in d24.values():
            v.pop("frac_P_below_10ms", None)
        f24.write_text(json.dumps(d24))

        # ---- s21_counts must read BOTH shapes. The fixture above writes
        # the derived one; script 21's own output nests the per-site record
        # under "table", which the first reader did not know.
        derived = json.loads((logs / "21_event_accounting.json").read_text())
        table_shape = {}
        for s, v in derived.items():
            rbs = {k: v["eq"].get(k, 0) + v["nw"].get(k, 0)
                   for k in ("train", "dev", "test")}
            st = 24 if s == "pnr-1" else 18 if s == "aneth" else 12
            table_shape[s] = {
                "table": {"rows_total": v["rec"] * st,
                          "rows_by_split": {k: n * st
                                            for k, n in rbs.items() if n},
                          "unique_events_total": v["rec"],
                          "unique_events_by_split": {
                              k: n for k, n in rbs.items() if n},
                          "noise_rows_total": v["nw_tot"] * st,
                          "noise_rows_by_split": {
                              k: v["nw"].get(k, 0) * st
                              for k in ("train", "dev", "test")}},
                "cache": {}, "audit_n_events": v["eq_tot"], "checks": []}
        a, b = s21_counts(derived), s21_counts(table_shape)
        assert len(a) == len(b) == 8, (len(a), len(b))
        assert a == b, [k for k in a if a[k] != b[k]]
        assert sum(x["rec"] for x in b.values()) == 9803
        assert sum(x["eq"] for x in b.values()) == 3421
        assert sum(x["noise"] for x in b.values()) == 6382
        tot_eq = sum(x["train_eq"] for x in b.values())
        per = [tot_eq - x["train_eq"] for x in b.values()]
        assert (min(per), max(per)) == (1527, 2002), (min(per), max(per))
        # 行（局トレース）を記録と取り違えないこと。pnr-1 は 24 局なので、
        # rows_total を記録として読めば 1,258 ではなく 30,192 になる。
        assert b["pnr-1"]["rec"] == 1258, b["pnr-1"]
        assert table_shape["pnr-1"]["table"]["rows_total"] == 1258 * 24
        # audit_n_events と食い違うサイトは落とすこと
        bad = {k: json.loads(json.dumps(x)) for k, x in table_shape.items()}
        bad["aneth"]["audit_n_events"] = 999
        assert "aneth" not in s21_counts(bad), "矛盾したサイトが採用された"
        assert len(s21_counts(bad)) == 7

        # ---- REGRESSION (2026-10-08): a log that is PRESENT but shaped
        # differently must not remove its checks silently. The Drive run
        # dropped seven checks that way and still reported "failed: 0".
        f21 = logs / "21_event_accounting.json"
        keep21 = f21.read_text()
        f21.write_text(json.dumps({"summary": {"note": "wrong shape"}}))
        ck21 = {c[0]: c for c in check_numbers(logs)}
        lab = next((k for k in ck21
                    if k.startswith("S3 ベンチマーク内訳の入力が使える")), None)
        assert lab is not None, "形違いが報告されていない"
        assert not ck21[lab][3], ck21[lab]
        assert "summary" in str(ck21[lab][2]), ck21[lab]
        assert not any(k.startswith("S3 記録の総数") for k in ck21), \
            "使えない入力から検査が作られている"
        f21.write_text(keep21)
        ck21 = {c[0]: c for c in check_numbers(logs)}
        assert any(k.startswith("S3 記録の総数") for k in ck21)

        # ---- a missing log must downgrade the output, not be invented
        (logs / "19_capacity_ema.json").unlink()
        (figs / FIGURE_SOURCE[6][0]).unlink(missing_ok=True)
        (figs / FIGURE_SOURCE[6][1]).unlink(missing_ok=True)
        out2 = root / "out2"
        rc2 = build(logs, [figs], root / "PDF2", out2)
        assert rc2 == 1, "欠落があるのに完全と判定された"
        assert (root / "PDF2" / "Supplement_MOIRAI_L3_INCOMPLETE.pdf").exists()
        assert not (root / "PDF2" / "Supplement_MOIRAI_L3.pdf").exists()
        man2 = json.loads((out2 / "25_manifest.json").read_text())
        assert man2["complete"] is False
        assert not (out2 / "TableS4.csv").exists(), "欠落した表が書かれている"
        assert any("Table S4" in m for m in man2["missing"]), man2["missing"]
        assert any("Figure S6" in m for m in man2["missing"]), man2["missing"]

        # ---- a drifted number must be reported, not silently accepted.
        # The drift is applied to the BIN COUNTS, which are what the share is
        # derived from; perturbing the convenience key instead would only test
        # the agreement assertion, which the regression block above covers.
        d24 = json.loads((logs / "24_moveout_ratio.json").read_text())
        d24["aneth"]["stratified"]["bins"][1]["n"] = 20      # was 197
        (logs / "24_moveout_ratio.json").write_text(json.dumps(d24))
        bad = [c for c in check_numbers(logs) if not c[3]]
        assert any("86.6" in c[0] for c in bad), bad
        assert any("258" in c[0] for c in bad), bad

        # ---- a renamed bin label must stop the table, not shift the numbers
        d24["aneth"]["stratified"]["bins"][0]["bin"] = "0-5 ms"
        (logs / "24_moveout_ratio.json").write_text(json.dumps(d24))
        try:
            table_s7(logs)
        except AssertionError:
            pass
        else:
            raise SystemExit("NG: bin ラベルの変更が見逃された")

    print("[selftest] ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--inventory", action="store_true",
                    help="report what is present; write nothing")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--logs", default=None, help="override the logs directory")
    ap.add_argument("--figdir", action="append", default=None,
                    help="extra directory to search for figures (repeatable)")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    logs = Path(a.logs) if a.logs else cfg.LOGS_DIR
    figdirs = [Path(x) for x in (a.figdir or [])]
    figdirs += [cfg.PDF_DIR, HERE / "figures", cfg.BASE_DIR / "figures",
                cfg.BASE_DIR / "paper" / "figures"]
    seen, uniq = set(), []
    for d in figdirs:
        if str(d) not in seen:
            seen.add(str(d)); uniq.append(d)

    if a.inventory:
        inventory(logs, uniq)
        return
    if a.build:
        raise SystemExit(build(logs, uniq, cfg.PDF_DIR,
                               cfg.LOGS_DIR / "supplement"))
    print("nothing to do: pass --inventory, --build or --selftest "
          "(see docstring)")


if __name__ == "__main__":
    main()
