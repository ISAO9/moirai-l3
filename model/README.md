# model/

Trained weights are **not** committed to git (size). The EMA-best checkpoints
used for the paper are attached to the **Zenodo record** of this archive
(concept DOI 10.5281/zenodo.21217614, which always resolves to the latest
version). They are not attached to the GitHub releases: the GitHub-to-Zenodo
integration archives the repository source only, so Zenodo is the single place
the weights live.

The weights are **not** needed to reproduce the figures and tables of the paper;
those regenerate from the result logs in `logs/` (see `REVISION_NOTES.md`). The
weights are for running the trained picker on new data, for example through
`src/09_ingest_external_to_amber_l3.py`.

Expected layout once downloaded (matches `src/00_config_l3.py` / `03_train_l3.py`):

    model/
      loso_<site>_array_seed1234_ema_best.pt
      loso_<site>_pertrace_seed1234_ema_best.pt
      ...

Both a `best` (highest dev F1-mean) and an EMA checkpoint are saved during
training; the paper reports the EMA-best weights.
