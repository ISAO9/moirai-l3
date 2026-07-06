# MOIRAI L3 — velocity-free P/S phase picking on borehole arrays (array vs per-trace)

Code, trained weights and results accompanying the paper:

> Kurosawa, I. *When does array moveout help borehole phase picking? A leave-one-site-out,
> confound-free benchmark of array versus per-trace deep learning.
> Submitted to *Geophysical Journal International* (GJI).
> Preprint: EarthArXiv, **DOI: _to be assigned_**. Software archive: Zenodo, DOI: 10.5281/zenodo.21217615

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

    src/     numbered pipeline (00–09), each with a header docstring
    paper/   preprint PDF + figures/ (05_a … 05_k)
    data/    how to obtain AMBER (data are NOT redistributed here)
    model/   how to obtain the released EMA-best weights
    PDF/     where src/05_figures_l3.py writes figure PDFs
    docs/    RELEASE_AND_DOI.md — GitHub + Zenodo DOI workflow

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

- `00_config_l3.py` — paths, constants (FS=2000 Hz, window=2048, 12 stations, tolerances, seed 1234)
- `01_amber_setup.py` — AMBER → SeisBench loaders; per-site CSV; LOSO split; per-trace split
- `02_picker_model_l3.py` — MoiraiPickerL3 2-D U-Net (station axis convolved, never pooled)
- `03_train_l3.py` — training; flags `--per-trace`, `--shuffle-stations`, `--seed`, `--heldout`; EMA + best/last checkpoints
- `04_evaluate_l3.py` — scoring (peak-pick, ±10/20/50 ms, F1/P/R, residuals) → `logs/04_test_results_*.json`
- `05_figures_l3.py` — all figures 05_a…05_k and Tables 1–2 (white background, English labels, legends in margin, saved as PDF)
- `06_baseline_pertrace.py` — off-the-shelf SeisBench baseline (motivation only; confounded)
- `07_bootstrap_ci.py` — event-level bootstrap 95% CIs + matched-pick residuals
- `08_moveout_features.py` — per-site across-station moveout (geometry descriptor)
- `09_ingest_external_to_amber_l3.py` — convert your own borehole array (miniSEED/SAC/SEG-Y/NumPy) → L3 input, for zero-shot inference

## Reproducing the paper

Figures and tables read the JSON results in `logs/`, so they regenerate **without retraining
or weights**:

    AMBER_H5=/path/to/waveforms.hdf5 python src/05_figures_l3.py --ckpt ema

To retrain a fold (example: held-out forge_19, array, seed 1234):

    AMBER_H5=/path/to/waveforms.hdf5 python src/03_train_l3.py --heldout forge_19 --seed 1234
    # add --per-trace or --shuffle-stations for the ablations

Full training/eval also provided as notebooks (see release assets).

## Citation

See `CITATION.cff`. Please cite **both** the paper (preprint/GJI) and the software archive
(Zenodo DOI). Fill in the DOIs once assigned (see `docs/RELEASE_AND_DOI.md`).

## License

- Software (`src/` etc.): **MIT** (see `LICENSE`).
- Manuscript and figures (`paper/`): **CC BY 4.0** (see `NOTICE`).
- AMBER data: under its own CC-BY-4.0 license (not redistributed here).
