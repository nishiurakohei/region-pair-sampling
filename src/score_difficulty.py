from __future__ import annotations

import argparse
from pathlib import Path

from connectivity_experiment.difficulty import attach_difficulty_proxy
from connectivity_experiment.io_utils import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach difficulty_proxy to an existing CSV.")
    parser.add_argument("input_csv")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path
    records = read_csv(input_path)
    attach_difficulty_proxy(records)
    write_csv(output_path, records)
    print(f"[score_difficulty] wrote {len(records)} rows to {output_path}")


if __name__ == "__main__":
    main()

