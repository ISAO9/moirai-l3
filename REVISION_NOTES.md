# Revision analyses (GJI-26-0613)

These scripts and outputs were added during the revision of
*"When does array moveout help borehole phase picking?"*
(Geophysical Journal International, GJI-26-0613).

They run on the checkpoints and evaluation logs already in this repository
(scripts 00–09) and regenerate every new number, table and figure in the
revised manuscript without retraining, with one exception noted below
(script 20 trains new folds).

Every script accepts `--selftest`, which validates its logic on synthetic
inputs without AMBER, checkpoints or network access.

---

## First revision round (scripts 10–14)

| Script | Purpose | Reviewer point |
|---|---|---|
| `src/10_threshold_sweep.py` | Threshold sweep 0.02–0.98, precision–recall curves, oracle-threshold F1, AUC-PR. Caches sufficient statistics from one inference pass per site. | R2-1, R1-23, R1-37 |
| `src/11_fa_inclusive_scoring.py` | Recomputes the array-versus-per-trace comparison under both false-alarm conventions (noise-event picks included / excluded) with joint event-bootstrap CIs. | R2-2, R1-30, R1-32 |
| `src/12_table3_audit.py` | Audits every cell of the three-way control table against the archived evaluation logs and emits a corrected table. | R2-4 |
| `src/13_station_subset_eval.py` | Zero-shot evaluation of the trained array picker on decimated arrays (4/6/8 stations; contiguous windows and random decimation, 5 draws each), with the per-trace model on identical subsets as control. | R1-40 |
| `src/14_fig4_and_tableS1.py` | Rebuilds the diagnosis figure (wiggle waveforms, explicit legend, in-distribution contrast event) and computes the supplementary array-geometry table from the station coordinates distributed with AMBER. | R1-36, R1-4, R1-5 |

### Reproducing

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

### Outputs

* `logs/10_threshold_sweep_<site>_<config>_ema.json` — per-threshold precision,
  recall, F1, oracle thresholds and false-alarm rates
* `logs/11_fa_inclusive_ema.json` — both scoring conventions, CIs, verdicts
* `logs/12_table3_audit.json` — cell-by-cell audit result
* `logs/13_station_subset_<site>_ema.json` — decimation results
* `logs/14_table_s1.csv` / `.md` — array geometry
* `logs/14_fig4_event_choice.json` — which events the diagnosis figure shows
  and the selection criteria they satisfy
* `PDF/` — precision–recall curves, station decimation, rebuilt diagnosis figure

> **Note.** `src/14_fig4_and_tableS1.py` was revised again in the second round
> (reviewer point R1-5: the figure now contrasts the out-of-distribution event
> with an in-distribution one, and the geometry table is computed from the
> station coordinates rather than assumed). Script 23 asserts that the patched
> version is present. The earlier version is in the v0.2.0 release.

---

## Second revision round (scripts 15–28)

| Script | Purpose | Reviewer point |
|---|---|---|
| `src/15_phase_confusion.py` | Compares every declared pick with the catalogued arrivals of **both** phases, giving phase-confusion matrices and signed residual distributions per site and configuration. Establishes that the forge_19 array P failure is a phase mis-assignment, not a loss of detection confidence. | R1-36 (round 2) |
| `src/16_arrival_audit.py` | Audits the catalogued arrivals against physics (S before P, implausible apparent V<sub>p</sub>/V<sub>s</sub>, arrivals outside the analysis window), separates catalogue errors from window-truncation effects, and re-scores both configurations on the surviving events. | Editor point 2, R2 (round 2) |
| `src/17_geometry_and_distance.py` | Reconstructs receiver strings and source positions from the AMBER station coordinates; per-site geometry and source–receiver distance table. Source of the new Figure 1. | R2 (round 2) |
| `src/18_paired_cache.py` | Re-evaluates both configurations on **identical** input records, which the submitted version did not do, and reports the score shift per site. | R2 (round 2) |
| `src/19_capacity_and_overfitting.py` | Parameter count, per-run training-set sizes in records and earthquakes, and the post-peak trend in the development score across all runs, measured against each run's own epoch-to-epoch variation. | Editor point on capacity |
| `src/20_within_array.py` | The within-array experiment: the held-out site's own events are placed in the training splits, isolating geometry from intrinsic site difficulty. **Trains new folds.** | R2 (round 2) |
| `src/21_event_accounting.py` | Record, event and trace accounting for every site and split, with an internal cross-check against the audit counts. | R1, R2 (round 2) |
| `src/22_sampling_rates.py` | Native sampling rates per site, read from the AMBER metadata CSV. | R1 (round 2) |
| `src/23_fig4_style_supplement.py` | Applies the rebuilt diagnosis-figure treatment to the remaining sites. Asserts that script 14 is the patched (R1-5) version. | R1-5 follow-up |
| `src/24_moveout_ratio_check.py` | Per-site moveout distributions, the share of events below the ±10 ms tolerance, and the S/P moveout ratio against the catalogued V<sub>p</sub>/V<sub>s</sub>. | R2 (round 2) |
| `src/25_build_supplement.py` | Assembles the Supporting Information as **one** document (Sections S1–S6, Tables S1–S7, Figures S1–S10) from the logs and figures, and checks every number in its prose against those logs. | GJI §2.8 |
| `src/27_paired_inference.py` | Inference on the **paired** array-minus-per-trace difference. Script 11 already resamples events jointly but reads the verdict off the non-overlap of two marginal intervals, which is not a test of the difference and is conservative when the two scores are correlated. This script keeps the per-resample difference (B = 2,000), gives a percentile CI on Δ and Pr(Δ > 0), applies a declared equivalence bound of ±0.02 (twice the ±0.01 window-draw spread conceded in Table 3), and adds an exact sign test and a Wilcoxon signed-rank test across the eight sites. One verdict changes: mseel_5h, from "tie" to a small but significant array advantage. | Pre-submission audit C-3 (2026-10-12) |
| `src/28_seed_statistics.py` | Seed statistics for every (site, configuration) cell. The first revision repeated only the four sites with a significant difference; 39 further models were trained for this round (seeds 1–3 of array and per-trace at the four remaining sites, and of the station-shuffle control at the five sites with a significant difference). Reports mean ± sd and seed-paired differences, reproduces the Section 4.2 values of the first revision as a regression gate, and draws Figures 4 and 9. Two single-run statements of the shuffle analysis did not survive the repeats and were withdrawn (pnr-2: shuffle is not worse than per-trace; forge_19: the separate effect of ordered moveout is not resolved). | Pre-submission audit C-4 / C-5 (2026-10-12) |
| `src/19_capacity_and_overfitting.py` (v0.3.2b) | Now covers the 83 leave-one-site-out runs and records, per run, the best EMA development score minus the best raw-weight score (the Section 3.3 claim that the EMA score was never below the raw score was unmeasured and is withdrawn: the two best scores differ by a median of +0.0004) and the epoch at which the logged training loss falls abruptly, which marks mini-batches skipped by the non-finite guard of script 03 rather than a lower loss (20 of 83 runs; the reported checkpoints are unaffected). | Pre-submission audit S3 (2026-10-12) |
| `src/26_train_split_overlap.py` | The Section 5.2 training-support figures recomputed on the population the model was trained on. Script 16 defines its "training set" as every audited earthquake of the other seven sites, **all splits**; the model trains on the **train split** only (supplement S3: 1,527–2,002 earthquakes), and pnr-1 contributes no training record in any fold because AMBER assigns its whole catalogue to the test split. The script reproduces the script-16 figures exactly as a regression gate, then restricts the population. Reads only the script-10 caches, the script-16 audit and `metadata.csv`. | Pre-submission audit A-1 (2026-10-09) |
| `model/README.md` (v0.3.3) | Zenodo version v0.3.3 adds the 41 EMA-best checkpoints that v0.3.0–v0.3.2 did not carry (the 39 seed repeats of this round and the two within-array models of script 20), so the record holds all 85 models of Section 3.4. No script or log changes; the layout section of `model/README.md` is corrected to the names `src/03_train_l3.py` actually writes. | Data Availability statement (2026-10-13) |

### Reproducing

```bash
# no GPU, no AMBER, no checkpoints -- logs only
python src/21_event_accounting.py --all
python src/25_build_supplement.py --build --figdir src/figures

# logs + AMBER metadata
python src/22_sampling_rates.py --csv /path/to/metadata.csv
AMBER_CSV=/path/to/metadata.csv python src/26_train_split_overlap.py --all
python src/27_paired_inference.py --all          # script-10 caches only
python src/28_seed_statistics.py --ckpt ema      # script-04 result logs only
python src/19_capacity_and_overfitting.py --curves   # training histories

# AMBER waveforms + checkpoints (GPU)
AMBER_H5=$H5 python src/15_phase_confusion.py --all
AMBER_H5=$H5 python src/16_arrival_audit.py --all --rescore
AMBER_H5=$H5 python src/17_geometry_and_distance.py --all
AMBER_H5=$H5 python src/18_paired_cache.py --all
AMBER_H5=$H5 python src/19_capacity_and_overfitting.py --all
AMBER_H5=$H5 python src/23_fig4_style_supplement.py --all
AMBER_H5=$H5 python src/24_moveout_ratio_check.py --all

# trains new folds (GPU, hours)
AMBER_H5=$H5 python src/20_within_array.py --heldout forge_19
```

### Outputs

* `logs/15_phase_confusion_ema.json` / `.md` — phase-confusion matrices and residuals
* `logs/16_arrival_audit_ema.json` / `.md` — per-site audit failures and causes
* `logs/16_clean_rescore_ema.json` / `.md` — scores on the surviving events
* `logs/16_moveout_overlap_ema.json` / `.md` — window-truncation separation
* `logs/17_array_coords.json`, `logs/17_source_locations.json`,
  `logs/17_geometry_table.csv` / `.md` — geometry (source of Figure 1)
* `logs/18_paired_scores_ema.json` / `.md` — identical-input re-evaluation
* `logs/19_capacity_ema.json`, `logs/19_capacity.md` — capacity and training histories
* `logs/20_within_array_ema.json` / `.md` — the within-array experiment
* `logs/20_within_array_history_array.json`,
  `logs/20_within_array_history_pertrace.json` — the training histories of the
  within-array folds (the only runs in this round that train new models)
* `logs/21_event_accounting.json` / `.md` — record/event/trace accounting
* `logs/22_sampling_rates.json` / `.md` — native sampling rates
* `logs/23_event_choice.json` — events shown in the supplementary diagnosis panels
* `logs/24_moveout_ratio.json` / `.md` — moveout distributions and S/P ratio
* `logs/26_train_split_overlap_ema.json` — training support per held-out site on three nested populations (script-16 population; train split; train+dev), each
* `logs/27_paired_inference.json` / `.md` — paired Δ, its CI, Pr(Δ > 0) and the equivalence verdict per site under both false-alarm conventions; the across-site sign and Wilcoxon tests (source of the Table 2 and Table 3 verdicts)
* `logs/28_seed_statistics_ema.json` / `.md` — mean ± sd over three seeds for every (site, configuration) cell, seed-paired differences, and the regression gate (source of Table 4 and of the seed statistics in Section 4.2) after the arrival audit, with the supplier breakdown. The manuscript quotes the train-split values: forge_19 7 / 1,553 (0.45 %), pnr-1 137 / 1,697 (8.07 %, 129 from forge_19)

The Supporting Information document and its machine-readable parts:

* `logs/supplement/25_manifest.json` — what the supplement built, from which file,
  its page count, and the result of the number check
* `logs/supplement/25_number_check.md` — every prose number in the supplement,
  with the value the logs give and an OK/NG verdict (103 checks)
* `logs/supplement/TableS1.csv` … `TableS7.csv` — Tables S1–S7 as machine-readable
  CSV, as GJI requires for tabular supporting information
* `logs/supplement/25_manuscript_index.txt` — the 23-line index of Sections S1–S6,
  Tables S1–S7 and Figures S1–S10 that the manuscript's SUPPORTING INFORMATION
  section carries, emitted from the same definition so the two cannot drift

The figures these scripts draw, each as PDF and PNG:

* `PDF/15_a_phase_residuals` — phase-assignment residuals (Figure S9/S10)
* `PDF/16_a_arrival_audit` — arrival-audit outcome per site (Figure S1)
* `PDF/17_a_geometry_ranges`, `PDF/17_b_source_receiver` — receiver strings and
  source positions (source of the new Figure 1)
* `PDF/18_a_paired_vs_unpaired` — identical-input re-evaluation (Figure S2)
* `PDF/19_a_training_curves` — development-score histories (Figure S3)
* `PDF/20_a_within_vs_loso` — within-array versus LOSO (Figure S4)
* `PDF/23_a_fig4_style_sites` — the diagnosis treatment on the remaining sites
  (Figures S5–S8)
* `PDF/24_a_moveout_ratio` — per-site moveout distributions (Figure S6 panel)
* `PDF/28_a_fig4_seed_all_sites` — array versus per-trace across seeds, all eight sites (Figure 4)
* `PDF/28_b_fig9_shuffle_seeds` — the station-shuffle control across seeds (Figure 9)

### Figure assets

`src/figures/` holds four published figure files that the supplement builder embeds
and that are not regenerable from the logs alone:
`05_a_probability_example.png`, `05_c_metrics_summary.png`,
`05_f_baseline_phasenet.png` and `13_a_station_subset.pdf`.
Pass the directory with `--figdir src/figures`.

---

The eight `logs/08_moveout_<site>.json` files carry the per-site moveout
medians and inter-quartile ranges quoted in Table 1, including the S-phase
IQRs restored during the first revision.

> **Reproducibility note (2026-10-09).** AMBER draws a random window and a
> random 12-station block per event (`amber/dataloaders.py`, seeded from
> `torch.initial_seed() + index`), and `08_moveout_features.py` does not seed
> torch. The published medians are therefore one unseeded draw; re-measuring
> under three fixed seeds moved every median by at most 1.5 samples (0.75 ms)
> and no event count. Event counts, which the support figures depend on, are
> independent of the draw. Script 26 uses the catalogued arrival samples in
> the script-10 caches, not a new draw.
