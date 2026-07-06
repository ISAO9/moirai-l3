# data/

Third-party benchmark data are **not** redistributed in this repository.

## AMBER (required)
Obtain `waveforms.hdf5` + `metadata.csv` from the AMBER archive:

> Verdon, J., Lim, C.S.Y., Leung, K., Lapins, S., Rodriguez-Pradilla, G., Read, E. &
> Werner, M.J., 2026. *The AI-Ready Downhole Microseismic Benchmark Database (AMBER)*
> (Version v1) [Data set]. Zenodo. doi:10.5281/zenodo.18944111
> Code: https://github.com/kelleuseis/AMBER_Public

Point `AMBER_H5` (see `src/00_config_l3.py`) at the downloaded `waveforms.hdf5`.
This study uses 8 of the 10 AMBER sub-datasets (excluding Cotton Valley Stage B
and FORGE 2022).

## External arrays (optional)
To run the trained picker zero-shot on your own borehole array, convert your
waveforms with `src/09_ingest_external_to_amber_l3.py` (miniSEED/SAC/SEG-Y or
NumPy → L3 input); see its header docstring.
