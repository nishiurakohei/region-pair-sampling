from __future__ import annotations

import argparse

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.pair_corruption_evaluation import run_pair_corruption_evaluation_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply corruption methods to fixed point pairs and summarize robustness signals.")
    parser.add_argument("--config", default="configs/pair_corruption_evaluation_default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    result = run_pair_corruption_evaluation_from_config(config=config, repo_root=repo_root)
    records = result["records"]
    review_count = sum(int(record.get("needs_human_review", 0)) == 1 for record in records)
    print(f"Generated {len(records)} corrupted samples.")
    print(f"Review samples: {review_count}")
    for name, path in result["output_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
