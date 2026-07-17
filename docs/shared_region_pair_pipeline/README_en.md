# Region-Pair Pipeline Sharing Guide

## Purpose

This guide is intended for sharing the current pipeline with collaborators in a structured way.

If the first goal is packaging and handoff, open [`HANDOFF_en.md`](HANDOFF_en.md).
If the first goal is running the code quickly, open [`QUICKSTART_en.md`](QUICKSTART_en.md).
If the first goal is seeing one concrete example, open [`DEMO_RUN_SUMMARY_en.md`](DEMO_RUN_SUMMARY_en.md).

The pipeline covers one full sequence:
1. start from binary mask images,
2. generate point pairs,
3. record connectivity-related features for each pair,
4. apply masking / occlusion / noise corruption to fixed pairs,
5. export machine-readable outputs for later model or analysis work.

## Recommended Entry Points

If a collaborator wants only the practical entry points, start from these files:

- Pair generation CLI: [`src/generate_region_pairs.py`](../../src/generate_region_pairs.py)
- Pair generation core module: [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
- Corruption evaluation CLI: [`src/run_pair_corruption_evaluation.py`](../../src/run_pair_corruption_evaluation.py)
- Corruption evaluation core module: [`src/connectivity_experiment/pair_corruption_evaluation.py`](../../src/connectivity_experiment/pair_corruption_evaluation.py)

## Shareable Structure

```text
docs/shared_region_pair_pipeline/
  README.md
  README_en.md
  README_ja.md
  HANDOFF_en.md
  HANDOFF_ja.md
  QUICKSTART_en.md
  QUICKSTART_ja.md
  DEMO_RUN_SUMMARY_en.md
  DEMO_RUN_SUMMARY_ja.md

configs/
  pair_generation_default.yaml
  pair_generation_stare_demo.yaml
  pair_corruption_evaluation_default.yaml
  pair_corruption_evaluation_stare_demo.yaml

scripts/
  create_collaborator_bundle.py

src/
  generate_region_pairs.py
  run_pair_corruption_evaluation.py
  connectivity_experiment/
    config.py
    io_utils.py
    region_pair_sampling.py
    region_pair_generation.py
    pair_corruption_evaluation.py

outputs/
  demo_pair_generation_stare_realdata_pairwise_check2/
  demo_pair_corruption_stare_realdata/
  share_bundles/
    collaborator_handoff_<date>/
```

## End-to-End Workflow

### Step 1. Prepare binary masks

- Input is a directory of binary mask images such as `data/masks/`.
- The current loader accepts grayscale or multi-channel images and converts them internally to a boolean foreground/background mask.
- Small connected components can be removed with `min_component_size`.

Relevant code:
- [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
- [`src/connectivity_experiment/region_pair_sampling.py`](../../src/connectivity_experiment/region_pair_sampling.py)

### Step 2. Generate point pairs and record features

Run:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
```

This stage:
- samples valid points from the mask,
- constructs pairs,
- keeps the final number of pairs fixed per image when possible,
- excludes trivial foreground-background cross-region pairs by default,
- assigns `connected` / `disconnected`,
- computes geometric features,
- generates deterministic explanation text and review flags,
- writes CSV / JSONL / TXT / metadata outputs.

Main output files:
- `pairs.csv`
- `pairs.jsonl`
- `pairs_explanation.txt`
- `pairs_for_human_review.csv`
- `per_image_text/<original_name>.txt`
- `visualizations/<original_name>_<pair_id>_vis.png`
- `metadata.json`

Important recorded fields include:
- `connected`
- `euclidean_distance`
- `geodesic_distance`
- `geodesic_euclidean_ratio`
- `boundary_crossing_count`
- `distance_to_boundary_min`
- `difficulty_bucket`
- `reason_codes`
- `why_selected`
- `review_reasons`

### Step 3. Inspect pair-level outputs

The main file for downstream analysis is `pairs.csv`.

Suggested reading order:
1. `metadata.json` for run-level context.
2. `pairs.csv` for structured analysis.
3. `per_image_text/` for one text file per original image.
4. `visualizations/` for one `_vis.png` overlay per pair on the original input image.
5. `pairs_explanation.txt` for one aggregated explanation file.
6. `pairs_for_human_review.csv` for pairs that may need manual checking.

### Step 4. Apply masking / noise corruption to fixed pairs

Run:

```bash
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

This stage:
- reads already-generated pair records,
- keeps the point pair fixed,
- applies corruption to the image or mask,
- recomputes connectivity after corruption only as auxiliary information,
- records whether the structure changed,
- writes corruption-level summary files.

Current corruption methods include:
- `none`
- `random_rectangle_mask`
- `random_circle_mask`
- `line_occlusion`
- `point_neighborhood_mask`
- `salt_and_pepper`
- `gaussian_noise`
- `blur`
- `erosion`
- `dilation`

Main output files:
- `corrupted_pair_inputs.csv`
- `corrupted_pair_inputs.jsonl`
- `corruption_explanation.txt`
- `corruption_review.csv`
- `robustness_summary.csv`
- `metadata.json`

### Step 5. Use outputs in downstream experiments

The current pipeline does not force a specific downstream model.

Typical downstream uses are:
- train a classifier on `pairs.csv`,
- analyze how difficulty relates to geometry,
- analyze how corruption affects structural stability,
- join corruption outputs with model predictions later.

## Input Flexibility

The current pair-generation loader is intended to be reusable not only for vessel masks, but also for road-segmentation and crack-segmentation masks.

Useful config options are:
- `mask_channel: auto`
- `binarization_mode: auto`
- `foreground_threshold`
- `foreground_values`
- `invert_mask`
- `use_alpha_channel`
- `auto_invert_if_foreground_dominates`

Practical meaning:
- `mask_channel: auto` uses the strongest signal across RGB channels when the segmentation color is not tied to one channel.
- `binarization_mode: auto` handles common binary encodings such as `0/1`, `0/255`, and simple colored segmentations.
- `foreground_values` can be used when the segmentation is encoded by specific pixel values rather than thresholding.
- `invert_mask` and `auto_invert_if_foreground_dominates` help when the foreground/background polarity differs across datasets.

## Code Map

### Pair generation layer

- [`src/generate_region_pairs.py`](../../src/generate_region_pairs.py)
  CLI entry point.
- [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
  Dataset-agnostic pair generation, explanation, review export.
- [`src/connectivity_experiment/region_pair_sampling.py`](../../src/connectivity_experiment/region_pair_sampling.py)
  Shared geometry and sampling utilities.

### Corruption evaluation layer

- [`src/run_pair_corruption_evaluation.py`](../../src/run_pair_corruption_evaluation.py)
  CLI entry point.
- [`src/connectivity_experiment/pair_corruption_evaluation.py`](../../src/connectivity_experiment/pair_corruption_evaluation.py)
  Fixed-pair corruption application, structural summary, review logic.

### Shared utilities

- [`src/connectivity_experiment/config.py`](../../src/connectivity_experiment/config.py)
  Lightweight YAML parser.
- [`src/connectivity_experiment/io_utils.py`](../../src/connectivity_experiment/io_utils.py)
  CSV / JSON / JSONL helpers.

### Validation

- [`tests/test_region_pair_generation.py`](../../tests/test_region_pair_generation.py)
- [`tests/test_pair_corruption_evaluation.py`](../../tests/test_pair_corruption_evaluation.py)
- [`tests/test_config_loader.py`](../../tests/test_config_loader.py)

## Default Configs vs Demo Configs

Use the default configs when you want a generic starting point:
- [`configs/pair_generation_default.yaml`](../../configs/pair_generation_default.yaml)
- [`configs/pair_corruption_evaluation_default.yaml`](../../configs/pair_corruption_evaluation_default.yaml)

Use the demo configs when you want one concrete real-data example:
- [`configs/pair_generation_stare_demo.yaml`](../../configs/pair_generation_stare_demo.yaml)
- [`configs/pair_corruption_evaluation_stare_demo.yaml`](../../configs/pair_corruption_evaluation_stare_demo.yaml)

## Demo Run Reference

The concrete real-data demo is documented here:
- [`outputs/reports/stare_realdata_demo_run_2026-07-16.md`](../../outputs/reports/stare_realdata_demo_run_2026-07-16.md)

That report shows:
- what was run,
- which configs were used,
- how many pairs were generated,
- which corruption methods were applied,
- what structural changes were observed.

## Minimal Bundle To Share

If you do not want to share the full repository, the minimum practical subset is:

- `docs/shared_region_pair_pipeline/`
- `configs/pair_generation_default.yaml`
- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_default.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`
- `src/generate_region_pairs.py`
- `src/run_pair_corruption_evaluation.py`
- `src/connectivity_experiment/config.py`
- `src/connectivity_experiment/io_utils.py`
- `src/connectivity_experiment/region_pair_sampling.py`
- `src/connectivity_experiment/region_pair_generation.py`
- `src/connectivity_experiment/pair_corruption_evaluation.py`
- `outputs/reports/stare_realdata_demo_run_2026-07-16.md`

## Notes For Collaborators

- The pair-generation module is dataset-agnostic at the directory level. Changing `input_dir` is usually enough to start on a new mask dataset.
- The corruption module assumes that pairs are already fixed in `pairs.csv`.
- The label used for corruption evaluation is still the original label by default. Structural change after corruption is recorded separately as auxiliary information.
- The current demo is a pipeline demonstration, not yet a statistically representative evaluation over the entire real dataset.
