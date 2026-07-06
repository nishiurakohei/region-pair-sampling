from __future__ import annotations

import argparse

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.io_utils import read_csv
from connectivity_experiment.visualize import build_visualization_sets, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sample grids for qualitative inspection.")
    parser.add_argument("--config", default="configs/curved_v1.yaml")
    parser.add_argument(
        "--input-csv",
        default="data/raw/test/metadata.csv",
        help="CSV with image_path and metadata columns.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    figure_dir = repo_root / args.output_dir if args.output_dir else repo_root / config["paths"]["sample_dir"]
    records = read_csv(repo_root / args.input_csv)
    visuals = build_visualization_sets(records)
    for name, subset in visuals.items():
        make_contact_sheet(
            subset,
            repo_root=repo_root,
            out_path=figure_dir / f"{name}.png",
            title=name.replace("_", " "),
        )
        print(f"[visualize_samples] wrote {figure_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
