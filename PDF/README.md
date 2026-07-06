# PDF/

Figure PDFs (white background, English labels, legends in the margin) are
written here by `src/05_figures_l3.py`. They are regenerated from the result
logs in `logs/` and therefore require no retraining:

    AMBER_H5=/path/to/waveforms.hdf5 python src/05_figures_l3.py --ckpt ema

The PNG renders embedded in the manuscript are under `paper/figures/`.
