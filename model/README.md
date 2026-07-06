# model/

Trained weights are **not** committed to git (size). The EMA-best checkpoints
used for the paper are released as assets on the **Zenodo archive** and/or the
**GitHub Release** for this repository.

Expected layout once downloaded (matches `src/00_config_l3.py` / `03_train_l3.py`):

    model/
      loso_<site>_array_seed1234_ema_best.pt
      loso_<site>_pertrace_seed1234_ema_best.pt
      ...

Both a `best` (highest dev F1-mean) and an EMA checkpoint are saved during
training; the paper reports the EMA-best weights.
