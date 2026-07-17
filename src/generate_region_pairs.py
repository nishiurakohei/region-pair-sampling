from __future__ import annotations

import argparse

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.region_pair_generation import generate_region_pairs_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset-agnostic connected/disconnected point pairs from binary masks.")
    parser.add_argument("--config", default="configs/pair_generation_default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    result = generate_region_pairs_from_config(config=config, repo_root=repo_root)
    records = result["records"]
    review_count = sum(int(record.needs_human_review) == 1 for record in records)
    print(f"Generated {len(records)} pairs.")
    print(f"Review pairs: {review_count}")
    for name, path in result["output_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
