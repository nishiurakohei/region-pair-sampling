# Share-Safe Demo Run Summary

## Overview

This note summarizes the real-data demo run from `2026-07-16` in a form that is easy to share with collaborators.

Configs used:

- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`

Commands:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

## Pair Generation Demo

Input:

- `data/masks`

Main outputs:

- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.csv`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.jsonl`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs_explanation.txt`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/per_image_text/`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/visualizations/`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/metadata.json`

Key points:

- 8 images were processed
- 192 pairs were generated
- one text file is written per image
- one `_vis.png` is written per pair
- trivial foreground-background cross-region pairs are excluded by default

## Corruption Demo

Main outputs:

- `outputs/demo_pair_corruption_stare_realdata/corrupted_pair_inputs.csv`
- `outputs/demo_pair_corruption_stare_realdata/corruption_review.csv`
- `outputs/demo_pair_corruption_stare_realdata/robustness_summary.csv`
- `outputs/demo_pair_corruption_stare_realdata/corrupted_images/`
- `outputs/demo_pair_corruption_stare_realdata/metadata.json`

Key points:

- multiple corruptions are applied to already fixed pairs
- the demo covered `none`, `random_rectangle_mask`, `line_occlusion`, `point_neighborhood_mask`, `gaussian_noise`, and `blur`
- structural changes and label changes are recorded in CSV outputs

## Limitation Of This Demo

- The first 8 processed images were biased toward the same STARE family.
- Because of that, this run should be read as an end-to-end demonstration, not as a broad dataset estimate.

## Good First Files To Show A Collaborator

These files usually communicate the state of the pipeline fastest.

1. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.csv`
2. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/per_image_text/`
3. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/visualizations/`
4. `outputs/demo_pair_corruption_stare_realdata/robustness_summary.csv`
