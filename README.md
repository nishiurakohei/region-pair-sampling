# Region Pair Sampling for Biconnectivity Experiments

This repository provides a pipeline for **assigning point pairs to binary region images** — the core annotation step in a study examining the relationship between topological quantities and the difficulty of two-point connectivity classification.

Given a binary mask image (e.g., a retinal vessel segmentation), the pipeline samples pairs of points (A, B) from the foreground (or background) region and records geometric features for each pair: Euclidean distance, geodesic distance, geodesic/Euclidean ratio, boundary-crossing count, connected-component membership, and a composite difficulty proxy. These labelled pairs serve as the dataset for downstream connectivity classification experiments.

> **Context.** This repository covers the *pair assignment* side of the project. A collaborating group is independently running topology-based and classification-difficulty experiments on the resulting pair records.

## Collaborator Guides

For the current dataset-agnostic pair-generation plus corruption pipeline, start here:

- Shared guide hub: [`docs/shared_region_pair_pipeline/README.md`](docs/shared_region_pair_pipeline/README.md)
- Japanese guide: [`docs/shared_region_pair_pipeline/README_ja.md`](docs/shared_region_pair_pipeline/README_ja.md)
- English guide: [`docs/shared_region_pair_pipeline/README_en.md`](docs/shared_region_pair_pipeline/README_en.md)
- Handoff guide (Japanese): [`docs/shared_region_pair_pipeline/HANDOFF_ja.md`](docs/shared_region_pair_pipeline/HANDOFF_ja.md)
- Handoff guide (English): [`docs/shared_region_pair_pipeline/HANDOFF_en.md`](docs/shared_region_pair_pipeline/HANDOFF_en.md)
- Quickstart (Japanese): [`docs/shared_region_pair_pipeline/QUICKSTART_ja.md`](docs/shared_region_pair_pipeline/QUICKSTART_ja.md)
- Quickstart (English): [`docs/shared_region_pair_pipeline/QUICKSTART_en.md`](docs/shared_region_pair_pipeline/QUICKSTART_en.md)
- Demo run summary (Japanese): [`docs/shared_region_pair_pipeline/DEMO_RUN_SUMMARY_ja.md`](docs/shared_region_pair_pipeline/DEMO_RUN_SUMMARY_ja.md)
- Demo run summary (English): [`docs/shared_region_pair_pipeline/DEMO_RUN_SUMMARY_en.md`](docs/shared_region_pair_pipeline/DEMO_RUN_SUMMARY_en.md)
- Dropbox bundle creator: [`scripts/create_collaborator_bundle.py`](scripts/create_collaborator_bundle.py)

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── region_pair_sampling_v1.yaml          # minimal smoke-test (synthetic masks)
│   ├── region_pair_sampling_stare_subset_v1.yaml
│   ├── region_pair_sampling_stare_subset_v2.yaml  # STARE subset (recommended start)
│   ├── region_pair_sampling_stare_full_v1.yaml
│   ├── region_pair_sampling_stare_full_random_v1.yaml
│   └── region_pair_sampling_meiyan_template.yaml  # template for other datasets
├── data/
│   └── masks/                                # place your binary mask PNGs here
├── src/
│   ├── run_region_pair_sampling.py           # main entry point
│   ├── connectivity_experiment/
│   │   ├── region_pair_sampling.py           # core logic
│   │   ├── connectivity_oracles.py           # pixel connectivity (4/8-neighbour BFS)
│   │   ├── geometry.py                       # geodesic distance, boundary crossing
│   │   ├── config.py                         # YAML loader
│   │   └── io_utils.py                       # CSV / JSON / JSONL writers
│   └── scripts/
│       ├── visualize_region_pair_examples.py
│       └── plot_difficulty_by_method.py
├── tests/
│   └── test_region_pair_sampling.py
└── outputs/                                  # created at runtime
    └── <experiment_name>/
        ├── phase0/   # mask & distance previews
        ├── phase1/   # valid-mask & sampled-point previews
        ├── phase2/   # pair contact sheets
        ├── phase3/   # distribution plots
        ├── metrics/  # CSV / JSONL pair records
        └── report/   # auto-generated Markdown report
```

## How It Works

The pipeline proceeds in four conceptual phases for each input mask.

**Phase 0 — mask loading.** Each PNG is loaded and binarised according to `foreground_threshold`. Small connected components below `min_component_area` are removed. A distance transform (`scipy.ndimage.distance_transform_edt`) is computed to measure how far each foreground pixel is from the nearest boundary.

**Phase 1 — valid-point sampling.** Points that are too close to the mask boundary (controlled by `valid_thresholds`) are excluded. Candidate points are sampled from the valid region using one of four strategies: `random`, `euclidean_fps` (farthest-point sampling by Euclidean distance), `geodesic_fps` (farthest-point sampling by geodesic distance), or `feature_diversity` (bin-stratified coverage of distance-to-boundary values).

**Phase 2 — pair construction and labelling.** All pairs (A, B) are formed from the sampled points. For each pair the following features are recorded:

| Feature | Description |
|---|---|
| `euclidean_dist` | Straight-line pixel distance |
| `geodesic_dist` | Shortest path distance within the foreground mask |
| `geodesic_euclidean_ratio` | Tortuosity proxy |
| `boundary_crossing_count` | Number of foreground/background transitions on the straight line A→B |
| `connected` | Whether A and B belong to the same connected component (pixel 8-connectivity) |
| `near_miss_score` | How close a disconnected pair is to being connected |
| `difficulty_proxy` | Composite score combining ratio, crossings, and near-miss |
| `difficulty_bucket` | Discretised label: Easy / Medium / Hard / Very Hard |

**Phase 3 — reporting.** Distribution plots, feature-coverage heatmaps, pair contact sheets (sorted by difficulty), and a summary Markdown report are written to the output directory.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists `numpy`, `Pillow`, `scipy`, and `matplotlib`. No deep-learning framework is required.

### 2. Place your mask images

Put binary mask PNGs under `data/masks/` (or point `mask_glob` in the config to your own directory).

### 3. Run the pipeline

```bash
# Smoke-test with the bundled synthetic masks (6 images)
python src/run_region_pair_sampling.py --config configs/region_pair_sampling_v1.yaml --overwrite

# STARE subset (recommended starting point)
python src/run_region_pair_sampling.py --config configs/region_pair_sampling_stare_subset_v2.yaml --overwrite

# Limit to N images for a quick check
python src/run_region_pair_sampling.py --config configs/region_pair_sampling_stare_subset_v2.yaml --limit-images 5 --overwrite
```

### 4. Inspect the outputs

```
outputs/<experiment_name>/metrics/pair_records.csv   ← one row per pair, all features
outputs/<experiment_name>/report/region_pair_sampling_report.md
outputs/<experiment_name>/phase2/pairs_high_difficulty.png
```

## Configuration Reference

All options live in a YAML file under `configs/`. The key fields are:

```yaml
experiment:
  seed: 42                    # global random seed

dataset:
  mask_glob: data/masks/*.png # glob pattern for mask images (relative to repo root)
  sample_limit: 20            # max number of images to process
  mask_channel: 0             # which PNG channel encodes the mask
  foreground_threshold: 128   # pixel value >= threshold → foreground
  invert_mask: false          # set true if background is bright
  min_component_area: 32      # discard tiny connected components

sampling:
  valid_thresholds: [1, 2, 3] # minimum distance-to-boundary for a point to be valid
  methods:                    # one or more of: random, euclidean_fps, geodesic_fps, feature_diversity
    - random
    - euclidean_fps
    - geodesic_fps
    - feature_diversity
  sample_foreground: true
  sample_background: true
  foreground_points_per_method: 8
  background_points_per_method: 8
  candidate_pool_limit: 3000  # upper bound on candidate points before pair construction
  pair_component_mode: all_pairs   # all_pairs | same_component | cross_component

paths:
  output_dir: outputs/my_experiment
```

To adapt this pipeline to a new dataset, copy `configs/region_pair_sampling_meiyan_template.yaml`, update `mask_glob` and `output_dir`, and run as above.

## Output Files

| File | Contents |
|---|---|
| `metrics/pair_records.csv` | One row per (A, B) pair with all geometric features and labels |
| `metrics/pair_records.jsonl` | Same data in JSONL format |
| `metrics/pair_feature_triplets.json` | (image_id, pair_id, feature_vector) triples for downstream use |
| `metrics/dataset_summary.csv` | Per-image statistics (area, component count, bounding box) |
| `metrics/difficulty_bucket_summary.csv` | Pair counts per Easy/Medium/Hard/Very Hard bucket |
| `report/region_pair_sampling_report.md` | Auto-generated summary report |

## Codebase Map (pair-assignment relevant files)

```
src/run_region_pair_sampling.py          CLI entry point and phase orchestration
src/connectivity_experiment/
    region_pair_sampling.py              Point sampling, pair construction, difficulty scoring,
                                         all visualization helpers
    connectivity_oracles.py              BFS-based pixel connectivity check (4/8-neighbour)
    geometry.py                          Geodesic distance via Dijkstra, boundary crossing counter
    config.py                            Lightweight YAML parser (no PyYAML dependency)
    io_utils.py                          write_csv / write_json / write_jsonl helpers
scripts/
    visualize_region_pair_examples.py    Standalone visualizer for a saved pair_records.csv
    plot_difficulty_by_method.py         Per-method difficulty distribution plots
tests/
    test_region_pair_sampling.py         Unit tests for sampling and feature computation
```

## Notes

- The YAML loader is a minimal in-house parser; no `PyYAML` installation is needed.
- `geodesic_fps` and `geodesic_dist` require `scipy` (for `distance_transform_edt`) and fall back gracefully if a path cannot be found within the foreground mask.
- All random operations are seeded deterministically from `experiment.seed` plus per-image offsets, so results are fully reproducible.
- The `connected` label uses 8-connectivity pixel BFS on the binary mask, which may differ from a topology-theoretic definition; both the raw connectivity flag and the component-membership information are preserved in `pair_records.csv` for downstream analysis.# region-pair-sampling
