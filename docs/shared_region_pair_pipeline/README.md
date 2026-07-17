# Shared Region-Pair Pipeline Guide

This folder is the collaborator-facing entry point for the current pipeline:

1. create point pairs from binary mask images,
2. record geometric / connectivity-related features,
3. write one text summary per source image and one visualization image per pair,
4. apply masking or noise corruption to fixed pairs,
5. export structured outputs for downstream analysis.

Language guides:
- Japanese: [README_ja.md](README_ja.md)
- English: [README_en.md](README_en.md)

Handoff / onboarding guides:
- Japanese handoff: [HANDOFF_ja.md](HANDOFF_ja.md)
- English handoff: [HANDOFF_en.md](HANDOFF_en.md)
- Japanese quickstart: [QUICKSTART_ja.md](QUICKSTART_ja.md)
- English quickstart: [QUICKSTART_en.md](QUICKSTART_en.md)
- Japanese demo summary: [DEMO_RUN_SUMMARY_ja.md](DEMO_RUN_SUMMARY_ja.md)
- English demo summary: [DEMO_RUN_SUMMARY_en.md](DEMO_RUN_SUMMARY_en.md)

Key files:
- Pair generation CLI: [`src/generate_region_pairs.py`](../../src/generate_region_pairs.py)
- Pair generation core: [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
- Corruption evaluation CLI: [`src/run_pair_corruption_evaluation.py`](../../src/run_pair_corruption_evaluation.py)
- Corruption evaluation core: [`src/connectivity_experiment/pair_corruption_evaluation.py`](../../src/connectivity_experiment/pair_corruption_evaluation.py)
- Demo generation config: [`configs/pair_generation_stare_demo.yaml`](../../configs/pair_generation_stare_demo.yaml)
- Demo corruption config: [`configs/pair_corruption_evaluation_stare_demo.yaml`](../../configs/pair_corruption_evaluation_stare_demo.yaml)
- Demo bundle creator: [`scripts/create_collaborator_bundle.py`](../../scripts/create_collaborator_bundle.py)

Recommended reading order:
1. Open either [HANDOFF_ja.md](HANDOFF_ja.md) or [HANDOFF_en.md](HANDOFF_en.md).
2. Follow [QUICKSTART_ja.md](QUICKSTART_ja.md) or [QUICKSTART_en.md](QUICKSTART_en.md) for the first run.
3. Read [DEMO_RUN_SUMMARY_ja.md](DEMO_RUN_SUMMARY_ja.md) or [DEMO_RUN_SUMMARY_en.md](DEMO_RUN_SUMMARY_en.md) for one concrete example.
4. Check the demo configs to see the expected runtime setup.
5. Move to the CLI and core modules if code-level modification is needed.
