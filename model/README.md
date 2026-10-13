# model/

Trained weights are **not** committed to git (size). The EMA-best checkpoints
of every model reported in the paper are attached to the **Zenodo record** of
this archive (concept DOI 10.5281/zenodo.21217614, which always resolves to the
latest version). They are not attached to any GitHub release: the
GitHub-to-Zenodo integration archives the repository source only, so Zenodo is
the single place the weights live.

The weights are **not** needed to reproduce the figures and tables of the paper;
those regenerate from the result logs in `logs/` (see `REVISION_NOTES.md`). The
weights are for running the trained pickers on new data, for example through
`src/09_ingest_external_to_amber_l3.py`, and for re-running the evaluation
(`src/04_evaluate_l3.py --ckpt ema ...`).

## What is on Zenodo

| Zenodo version | EMA-best checkpoints |
|---|---|
| v0.1.0 – v0.2.0 | none (repository source only) |
| v0.3.0 – v0.3.2 | 44: the 16 leave-one-site-out models, the 24 seed repeats of the first revision (four sites) and the 4 station-shuffle controls |
| **v0.3.3** | **85**: every model of Section 3.4 of the paper — the 16 leave-one-site-out models, 48 seed repeats (three seeds of every fold in both configurations), 19 station-shuffle controls and the 2 within-array models of Section S4 |

Each `.pt` is a `torch.save` dictionary with the keys `epoch`, `model_state`,
`metrics` (the development-split scores at the epoch the checkpoint was written),
`site` and `config` (see `save_ckpt` in `src/03_train_l3.py`). Every file is
about 32 MB.

## File names

The names are produced by `src/03_train_l3.py` (`suffix` in `train()`), which
appends `_pertrace`, `_shuf`, `_loso_<site>` and `_seed<n>` to
`moirai_l3_ema_best.pt`. Sites: `pnr-1`, `pnr-2`, `mseel_3h`, `mseel_5h`,
`clearfield_mw4`, `clearfield_mw6`, `aneth`, `forge_19`.

| Group | Count | Pattern |
|---|---|---|
| Leave-one-site-out, array (seed 1234) | 8 | `moirai_l3_ema_best_loso_<site>.pt` |
| Leave-one-site-out, per-trace (seed 1234) | 8 | `moirai_l3_ema_best_pertrace_loso_<site>.pt` |
| Seed repeats, array | 24 | `moirai_l3_ema_best_loso_<site>_seed<n>.pt`, n = 1, 2, 3 |
| Seed repeats, per-trace | 24 | `moirai_l3_ema_best_pertrace_loso_<site>_seed<n>.pt`, n = 1, 2, 3 |
| Station-shuffle control (seed 1234) | 4 | `moirai_l3_ema_best_shuf_loso_<site>.pt`, sites `aneth`, `clearfield_mw4`, `forge_19`, `pnr-2` |
| Station-shuffle control, seed repeats | 15 | `moirai_l3_ema_best_shuf_loso_<site>_seed<n>.pt`, the four sites above and `mseel_5h`, n = 1, 2, 3 |
| Within-array experiment (script 20) | 2 | `within_array__moirai_l3_ema_best.pt`, `within_array__moirai_l3_ema_best_pertrace.pt` |

The station-shuffle control is reported as three-seed statistics at five sites
(Section 5.3, Table 4); a seed-1234 run preceded the repeats at four of them and
none was trained at `mseel_5h`, so the shuffle group has 4 + 15 = 19 models.

## Expected layout once downloaded

Place the 83 leave-one-site-out, seed-repeat and shuffle files directly in
`model/`, under their Zenodo names (this is where `src/03_train_l3.py` wrote them
and where `src/04_evaluate_l3.py` looks for them):

    model/
      moirai_l3_ema_best_loso_<site>.pt
      moirai_l3_ema_best_loso_<site>_seed<n>.pt
      moirai_l3_ema_best_pertrace_loso_<site>.pt
      moirai_l3_ema_best_pertrace_loso_<site>_seed<n>.pt
      moirai_l3_ema_best_shuf_loso_<site>.pt
      moirai_l3_ema_best_shuf_loso_<site>_seed<n>.pt

The two within-array files carry the prefix `within_array__` on Zenodo only
because a Zenodo record is a flat list. Strip the prefix and place them in the
sub-directory `src/20_within_array.py` uses:

    model/within_array/
      moirai_l3_ema_best.pt
      moirai_l3_ema_best_pertrace.pt

Both a raw `best` (highest development F1-mean) and an EMA checkpoint are saved
during training; the paper reports the EMA-best weights, and only those are
released.
