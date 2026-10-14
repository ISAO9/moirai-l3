# MOIRAI L3 — P/S phase picking on borehole arrays (array vs per-trace)

Code, trained weights and results accompanying the paper:

> Kurosawa, I. *When does array moveout help borehole phase picking?*
> Submitted to *Geophysical Journal International* (GJI); in revision (GJI-26-0613).
> Preprint (Author's Original Version): EarthArXiv, doi:10.31223/X5121S
> Software archive: Zenodo, concept DOI **10.5281/zenodo.21217614** (always resolves
> to the latest version; v0.1.0 is 10.5281/zenodo.21217615).

A geometry-invariant 2-D U-Net is trained under a strict **leave-one-site-out (LOSO)**
protocol on the **AMBER** benchmark and evaluated zero-shot on the held-out site, in two
configurations that differ **only** in whether the station axis (and hence cross-station
moveout) is available: **array** (full 12-station input) and **per-trace** (single-station
input). A **station-shuffle** control further separates ordered moveout from generic
cross-station coupling.

**Headline result.** Across the eight held-out sites, array and per-trace are statistically
indistinguishable in median F1 (0.885 vs 0.878); the array advantage is *conditional* on
in-distribution moveout and turns negative out-of-distribution (e.g. forge_19). Per-trace is
the safer operational default.

## Repository layout

    src/          numbered pipeline (00–26), each with a header docstring
    src/figures/  published figure assets that the supplement builder (25) embeds
    paper/        preprint PDF + figures/ (05_a … 05_k)
    data/         how to obtain AMBER (data are NOT redistributed here)
    model/        how to obtain the released EMA-best weights
    logs/         result logs; every figure and table regenerates from these
    PDF/          where the figure scripts write their PDFs
    docs/         RELEASE_AND_DOI.md — GitHub + Zenodo DOI workflow

## Data (not redistributed here)

AMBER is openly available under CC-BY-4.0 and must be obtained from its own archive
(see `data/README.md`):

> Verdon, J., Lim, C.S.Y., Leung, K., Lapins, S., Rodriguez-Pradilla, G., Read, E. &
> Werner, M.J., 2026. *The AI-Ready Downhole Microseismic Benchmark Database (AMBER)*
> (Version v1) [Data set]. Zenodo. doi:10.5281/zenodo.18944111 ·
> Code: https://github.com/kelleuseis/AMBER_Public

This study uses 8 of the 10 AMBER sub-datasets (excluding Cotton Valley Stage B and FORGE 2022).

## Environment (uv)

Python 3.10–3.12.

    pip install uv
    uv venv && source .venv/bin/activate
    uv pip install --no-deps "amber @ git+https://github.com/kelleuseis/AMBER_Public"
    uv pip install -r requirements.txt

Do not downgrade NumPy after installing AMBER.

## Pipeline (`src/`, run in order)

### Core pipeline (00–09)

- `00_config_l3.py` — paths, constants (FS=2000 Hz, window=2048, 12 stations, tolerances, seed 1234)
- `01_amber_setup.py` — AMBER → SeisBench loaders; per-site CSV; LOSO split; per-trace split
- `02_picker_model_l3.py` — MoiraiPickerL3 2-D U-Net (station axis convolved, never pooled)
- `03_train_l3.py` — training; flags `--per-trace`, `--shuffle-stations`, `--seed`, `--heldout`; EMA + best/last checkpoints
- `04_evaluate_l3.py` — scoring (peak-pick, ±10/20/50 ms, F1/P/R, residuals) → `logs/04_test_results_*.json`
- `05_figures_l3.py` — figures 05_a…05_k and Tables 1–2
- `06_baseline_pertrace.py` — off-the-shelf SeisBench baseline (motivation only; confounded)
- `07_bootstrap_ci.py` — event-level bootstrap 95% CIs + matched-pick residuals
- `08_moveout_features.py` — per-site across-station moveout (geometry descriptor)
- `09_ingest_external_to_amber_l3.py` — convert your own borehole array (miniSEED/SAC/SEG-Y/NumPy) → L3 input, for zero-shot inference

### First revision round (10–14)

- `10_threshold_sweep.py` — threshold sweep 0.02–0.98, PR curves, oracle-threshold F1, AUC-PR
- `11_fa_inclusive_scoring.py` — both false-alarm conventions with joint event-bootstrap CIs
- `12_table3_audit.py` — cell-by-cell audit of the three-way control table
- `13_station_subset_eval.py` — zero-shot evaluation on decimated arrays (4/6/8 stations)
- `14_fig4_and_tableS1.py` — rebuilt diagnosis figure and the array-geometry table

### Second revision round (15–28)

- `15_phase_confusion.py` — every declared pick compared with the catalogued arrivals of **both** phases; phase-confusion matrices and residual distributions
- `16_arrival_audit.py` — physical audit of the catalogued arrivals (S before P, implausible V<sub>p</sub>/V<sub>s</sub>, arrivals outside the window), plus re-scoring on the surviving events
- `17_geometry_and_distance.py` — receiver strings and source positions from the AMBER station coordinates; per-site geometry table
- `18_paired_cache.py` — re-evaluation of both configurations on **identical** inputs; `--figure-only` redraws Figure S7 from the scores log (since v0.3.4 the figure labels the independent-pass scores "unpaired")
- `19_capacity_and_overfitting.py` — parameter count, per-run training-set sizes, post-peak trend in the development score across all runs; since v0.3.2 also the best-EMA-minus-best-raw development score per run and the detection of late falls of the logged training loss (skipped non-finite mini-batches)
- `20_within_array.py` — the within-array experiment: the held-out site's own events placed in the training splits
- `21_event_accounting.py` — record/event/trace accounting for every site and split
- `22_sampling_rates.py` — native sampling rates per site from the AMBER metadata
- `23_fig4_style_supplement.py` — the diagnosis-figure treatment applied to the remaining sites
- `24_moveout_ratio_check.py` — per-site moveout distributions and the S/P moveout ratio
- `25_build_supplement.py` — assembles the Supporting Information as one document (Sections S1–S6, Tables S1–S7, Figures S1–S10) from the logs and figures, and checks every number in its prose against those logs (104 checks). v0.3.4 carries the text of the resubmitted supplement: the published title, no references to the review process, the independent-pass evaluation named "unpaired" (Section S2, Table S3, Figure S7), and the per-trace training-trace count corrected to 12 × records per run (57,228–71,184)
- `26_train_split_overlap.py` — the Section 5.2 training-support figures (share of training earthquakes reaching the held-out site's median moveout) recomputed on the population the model was trained on: the LOSO **train split** after the arrival audit of script 16. Reproduces the script-16 figures first, then restricts the population. No HDF5, no GPU
- `27_paired_inference.py` — the array-versus-per-trace verdict as a **paired** joint event bootstrap (B = 2,000) on Δ = F1-mean(array) − F1-mean(per-trace), with a declared ±0.02 equivalence bound and an across-site sign test and Wilcoxon signed-rank test; replaces the marginal-CI-overlap rule of the first revision. Reads the script-10 caches only
- `28_seed_statistics.py` — training-seed statistics for every held-out site (array, per-trace) and for the station-shuffle control: mean ± sd over three seeds, seed-paired differences, a regression gate against the first-revision values, and the sources of Figures 4 and 9. Reads the script-04 result logs only
- `29_fig8_paired.py` — Figure 8 (array advantage versus site median P moveout) drawn from the **paired** Δ, its 95 per cent CI and the verdict of script 27 (the Table 2 quantities) and from the moveout medians **after** the arrival audit of script 16 (the Table 1 quantities); a regression gate checks both against the tables. Replaces the `05_i` panel of script 05, which used the standard-evaluation per-phase means and the pre-audit medians. Reads two logs only

Every script in 10–29 accepts `--selftest`, which validates its logic on synthetic inputs
without AMBER, checkpoints or network access.

## Reproducing the paper

Figures and tables read the JSON results in `logs/`, so they regenerate **without retraining
or weights**:

    AMBER_H5=/path/to/waveforms.hdf5 python src/05_figures_l3.py --ckpt ema

The Supporting Information document rebuilds from the logs alone — no GPU, no AMBER, no
checkpoints:

    python src/25_build_supplement.py --build --figdir src/figures

The paired inference and the seed statistics likewise rebuild from the logs alone:

    python src/27_paired_inference.py --all
    python src/28_seed_statistics.py --ckpt ema
    python src/29_fig8_paired.py                 # Figure 8 from logs/27 and logs/16

To retrain a fold (example: held-out forge_19, array, seed 1234):

    AMBER_H5=/path/to/waveforms.hdf5 python src/03_train_l3.py --heldout forge_19 --seed 1234
    # add --per-trace or --shuffle-stations for the ablations

See `REVISION_NOTES.md` for the full command sequence of the revision analyses and the log
files each one writes. Full training/eval is also provided as notebooks (see release assets).

## Citation

See `CITATION.cff`. Please cite **both** the paper (preprint/GJI) and the software archive
(Zenodo concept DOI 10.5281/zenodo.21217614). See `docs/RELEASE_AND_DOI.md` for the release
workflow.

> The archive cites the **concept** DOI 10.5281/zenodo.21217614, which always resolves
> to the latest version. The per-version DOIs are 10.5281/zenodo.21217615 (v0.1.0) and
> 10.5281/zenodo.22224441 (v0.2.0).

## License

- Software (`src/` etc.): **MIT** (see `LICENSE`).
- Manuscript and figures (`paper/`): **CC BY 4.0** (see `NOTICE`).
- AMBER data: under its own CC-BY-4.0 license (not redistributed here).
