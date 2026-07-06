from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.region_pair_example_visualization import save_example_grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize representative region-pair CNN predictions by sampling method.")
    parser.add_argument(
        "--predictions",
        default="outputs/region_pair_cnn_stare_instruction_subset_v1/metrics/test_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/region_pair_cnn_stare_instruction_subset_v1/figures/example_pairs",
    )
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = save_example_grids(
        predictions_path=ROOT / args.predictions,
        out_dir=ROOT / args.out_dir,
        per_group=args.per_group,
        n_cols=args.cols,
        random_seed=args.seed,
    )
    print(f"Saved {len(output_paths)} files to {ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
