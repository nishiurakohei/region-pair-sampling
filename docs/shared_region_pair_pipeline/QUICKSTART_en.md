# Quickstart

## Purpose

This is the shortest path for a collaborator to run the pipeline for the first time.

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The main dependencies are `numpy`, `Pillow`, `scipy`, and `matplotlib`.

## 2. Place mask images

The simplest option is to put binary mask images under `data/masks/`.

If you want to use another location, update `input_dir` in the config.

## 3. Generate point pairs

Example:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
```

This stage:

- chooses a fixed number of pairs per image,
- excludes trivial foreground-background pairs by default,
- computes geometric features,
- writes one text file per image,
- writes one visualization PNG per pair.

Main outputs:

- `pairs.csv`
- `pairs.jsonl`
- `pairs_explanation.txt`
- `per_image_text/<original_name>.txt`
- `visualizations/<original_name>_<pair_id>_vis.png`
- `metadata.json`

## 4. Inspect the results

The recommended order is:

1. `metadata.json`
2. `pairs.csv`
3. `per_image_text/`
4. `visualizations/`

## 5. Run corruption evaluation if needed

```bash
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

This stage applies masking or noise to already fixed point pairs.

## 6. First config fields to edit for a new dataset

Start from these fields.

- `input_dir`
- `output_dir`
- `num_pairs_per_image`
- `mask_channel`
- `binarization_mode`
- `sample_foreground`
- `sample_background`
- `pair_component_mode`

## 7. Minimal handoff path

When sending the pipeline to a collaborator, pointing them to these three files is usually enough.

1. this `QUICKSTART_en.md`
2. [`HANDOFF_en.md`](HANDOFF_en.md)
3. [`DEMO_RUN_SUMMARY_en.md`](DEMO_RUN_SUMMARY_en.md)
