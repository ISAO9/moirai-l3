# Revision analyses (GJI-26-0613, major revision)

These scripts and outputs were added during the major revision of
*"When does array moveout help borehole phase picking? A leave-one-site-out,
confound-free benchmark of array versus per-trace deep learning"*
(Geophysical Journal International, GJI-26-0613).

They run on the checkpoints and evaluation logs already in this repository
(scripts 00–09) and regenerate every new number, table and figure in the
revised manuscript without retraining.

| Script | Purpose | Reviewer point |
|---|---|---|
| `src/10_threshold_sweep.py` | Threshold sweep 0.02–0.98, precision–recall curves, oracle-threshold F1, AUC-PR. Caches sufficient statistics from one inference pass per site. | R2-1, R1-23, R1-37 |
| `src/11_fa_inclusive_scoring.py` | Recomputes the array-versus-per-trace comparison under both false-alarm conventions (noise-event picks included / excluded) with joint event-bootstrap CIs. | R2-2, R1-30, R1-32 |
| `src/12_table3_audit.py` | Audits every cell of the three-way control table against the archived evaluation logs and emits a corrected table. | R2-4 |
| `src/13_station_subset_eval.py` | Zero-shot evaluation of the trained array picker on decimated arrays (4/6/8 stations; contiguous windows and random decimation, 5 draws each), with the per-trace model on identical subsets as control. | R1-40 |
| `src/14_fig4_and_tableS1.py` | Rebuilds the diagnosis figure (wiggle waveforms, explicit legend, in-distribution contrast event) and computes the supplementary array-geometry table from the station coordinates distributed with AMBER. | R1-36, R1-4, R1-5 |

## Reproducing

```bash
# threshold analysis: build caches (GPU), then sweep (CPU)
for s in pnr-1 pnr-2 mseel_3h mseel_5h clearfield_mw4 clearfield_mw6 aneth forge_19; do
  AMBER_H5=$H5 python src/10_threshold_sweep.py --cache --heldout $s
  AMBER_H5=$H5 python src/10_threshold_sweep.py --cache --heldout $s --per-trace
done
python src/10_threshold_sweep.py --sweep --all

python src/11_fa_inclusive_scoring.py --all
python src/12_table3_audit.py
AMBER_H5=$H5 python src/13_station_subset_eval.py --all
AMBER_H5=$H5 python src/14_fig4_and_tableS1.py --fig4
pip install remotezip && python src/14_fig4_and_tableS1.py --table-s1
```

Every script accepts `--selftest`, which validates its logic on synthetic
inputs without AMBER, checkpoints or network access.

## Outputs in this archive

* `logs/10_threshold_sweep_<site>_<config>_ema.json` — per-threshold precision,
  recall, F1, oracle thresholds and false-alarm rates
* `logs/11_fa_inclusive_ema.json` — both scoring conventions, CIs, verdicts
  (source of revised Table 3)
* `logs/12_table3_audit.json` — cell-by-cell audit result (source of the
  corrections in revised Table 4)
* `logs/13_station_subset_<site>_ema.json` — decimation results
* `logs/14_table_s1.csv` / `.md` — array geometry (Supplementary Table S1)
* `logs/14_fig4_event_choice.json` — which events the diagnosis figure shows
  and the selection criteria they satisfy
* `PDF/` — the revision figures: precision–recall curves (Figure 8),
  station decimation (Supplementary Figure S5), rebuilt diagnosis (Figure 4)

The eight `logs/08_moveout_<site>.json` files carry the per-site moveout
medians and inter-quartile ranges quoted in Table 1, including the S-phase
IQRs restored during this revision.
