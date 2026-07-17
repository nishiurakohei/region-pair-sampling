# Collaborator Handoff Guide

## Purpose

This note is the practical handoff entry point for sharing the current pipeline with collaborators.

There are two different things to share.

- `GitHub`: code, configs, and documentation
- `Dropbox`: real data, sample outputs, and heavier image artifacts

## Recommended Split

The recommended default is:

- use `GitHub` for day-to-day collaboration,
- use `Dropbox` for data delivery and generated image folders.

This keeps versioned code separate from heavier, dataset-specific assets.

## Minimum GitHub Package

At minimum, the collaborator should receive:

- `README.md`
- `requirements.txt`
- `docs/shared_region_pair_pipeline/`
- `configs/pair_generation_default.yaml`
- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_default.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`
- `src/generate_region_pairs.py`
- `src/run_pair_corruption_evaluation.py`
- `src/connectivity_experiment/`

Recommended reading order:

1. [`QUICKSTART_en.md`](QUICKSTART_en.md)
2. [`DEMO_RUN_SUMMARY_en.md`](DEMO_RUN_SUMMARY_en.md)
3. [`README_en.md`](README_en.md)

## What To Place In Dropbox

Dropbox should hold the assets that are large or dataset-specific.

- `data/masks/` or any real mask dataset
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/`
- `outputs/demo_pair_corruption_stare_realdata/`
- any extra visualizations or analysis outputs

## Automatic Dropbox Bundle

This repository includes a script that creates a collaborator-ready bundle.

Run:

```bash
python scripts/create_collaborator_bundle.py
```

By default it creates:

```text
outputs/share_bundles/collaborator_handoff_<date>/
```

That bundle contains:

- documentation,
- required code,
- main configs,
- a minimal test subset,
- demo outputs when they are available locally.

## Suggested Message To Send

This short message works well in practice.

```text
Please start from docs/shared_region_pair_pipeline/QUICKSTART_en.md.
The main code entry points are src/generate_region_pairs.py and src/run_pair_corruption_evaluation.py.
Real data and sample outputs are provided separately through Dropbox.
```

## Notes

- `data/masks/` is intentionally not tracked in Git.
- `outputs/` is also normally untracked.
- Because of that, GitHub alone will not include real data or sample outputs.
- Before creating a Dropbox bundle, make sure the sample outputs you want to share already exist locally.
