# Codex navigation

Treat this repository as the canonical source of truth. Before any material experiment, read:

1. `docs/RESEARCH_CONTEXT.md`
2. `docs/EXPERIMENT_LOG.md`
3. `docs/DECISIONS.md`
4. `docs/NEXT_TASK.md`

Stable rules:

- Do not delete or overwrite negative results; append a new run or update the log with provenance.
- Save every experiment config and distinguish model, dataset, shuffle, and augmentation/sampling seeds.
- Never use OOD test results for hyperparameter or checkpoint selection.
- Do not change dependency versions without recording the reason and a compatibility rerun.
- Preserve `maze-dataset==1.1.0`; version 1.4.2 is incompatible with the pretrained raster pipeline.
- Run the pretrained compatibility smoke test before GPU training.
- Do not commit datasets, model weights, checkpoints, virtual environments, or other large artifacts to ordinary Git without explicit approval. Record them in `docs/ARTIFACTS.md` instead.
- Update `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md`, and artifact metadata after every major experiment.
