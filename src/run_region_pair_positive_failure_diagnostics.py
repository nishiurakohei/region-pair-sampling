from __future__ import annotations

import argparse
from pathlib import Path

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.io_utils import ensure_dir, write_csv, write_json
from connectivity_experiment.region_pair_cnn import assign_image_splits, load_pair_records
from connectivity_experiment.region_pair_diagnostics import (
    filter_easy_only_records_v2,
    run_model_grid,
    save_named_prediction_grids,
)
from connectivity_experiment.region_pair_sampling import attach_region_pair_difficulty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run positive-failure diagnostics for region pair CNN experiments.")
    parser.add_argument("--config", default="configs/region_pair_positive_failure_diagnostics_stare_instruction_subset_v2.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)

    records = load_pair_records(repo_root / config["data"]["pair_metrics_csv"])
    if not records:
        raise RuntimeError("No pair records were loaded.")
    attach_region_pair_difficulty(records)

    seed = int(config["experiment"]["seed"])
    split_config = dict(config["split"])
    base_model_config = dict(config["model"])
    output_dir = ensure_dir(repo_root / config["paths"]["output_dir"])
    metrics_dir = ensure_dir(output_dir / "metrics")
    report_dir = ensure_dir(output_dir / "report")
    figure_dir = ensure_dir(output_dir / "figures")

    easy_only_records = filter_easy_only_records_v2(records, seed=seed)
    if not easy_only_records:
        raise RuntimeError("easy_only_records_v2 ended up empty.")
    write_csv(metrics_dir / "easy_only_v2_records.csv", easy_only_records)
    write_csv(metrics_dir / "pair_records_with_easy_v2_flags.csv", records)

    conditions = build_conditions(
        input_modes=[str(value) for value in config["comparison"]["input_modes"]],
        pool_sizes=[int(value) for value in config["comparison"]["pool_sizes"]],
        pos_weight_modes=[str(value) for value in config["comparison"]["pos_weight_modes"]],
        line_width=int(config["comparison"].get("line_width", 1)),
    )

    easy_split = assign_image_splits(
        easy_only_records,
        seed=seed,
        train_fraction=float(split_config["train_fraction"]),
        val_fraction=float(split_config["val_fraction"]),
    )
    easy_results, easy_rows = run_model_grid(
        records=easy_only_records,
        output_dir=output_dir / "easy_only_v2",
        split_config=split_config,
        base_model_config=base_model_config,
        seed=seed,
        conditions=conditions,
        split_by_image=easy_split,
    )

    subset_split = assign_image_splits(
        records,
        seed=seed,
        train_fraction=float(split_config["train_fraction"]),
        val_fraction=float(split_config["val_fraction"]),
    )
    subset_results, subset_rows = run_model_grid(
        records=records,
        output_dir=output_dir / "subset_v2",
        split_config=split_config,
        base_model_config=base_model_config,
        seed=seed,
        conditions=conditions,
        split_by_image=subset_split,
    )

    reference_condition = str(config["visualization"]["reference_condition"])
    if reference_condition not in easy_results:
        raise KeyError(f"reference_condition {reference_condition!r} was not found in easy_only_v2 results.")
    reference_predictions = easy_results[reference_condition].test_predictions
    example_paths = save_named_prediction_grids(
        predictions=reference_predictions,
        out_dir=figure_dir / "easy_v2_examples",
        groups=build_easy_v2_example_groups(
            reference_predictions,
            per_group=int(config["visualization"].get("per_group", 16)),
        ),
        n_cols=int(config["visualization"].get("n_cols", 4)),
    )

    write_report(
        out_path=report_dir / "region_pair_positive_failure_diagnostics.md",
        easy_rows=easy_rows,
        subset_rows=subset_rows,
        easy_only_records=easy_only_records,
        reference_condition=reference_condition,
        example_paths=example_paths,
    )
    write_json(
        metrics_dir / "diagnostics_manifest.json",
        {
            "conditions": conditions,
            "easy_only_record_count": len(easy_only_records),
            "reference_condition": reference_condition,
            "example_paths": [str(path) for path in example_paths],
        },
    )
    print(
        "[region_pair_positive_failure_diagnostics] "
        f"easy_records={len(easy_only_records)} "
        f"easy_best_acc={max(float(row['accuracy']) for row in easy_rows):.4f} "
        f"subset_best_acc={max(float(row['accuracy']) for row in subset_rows):.4f} "
        f"-> {output_dir}"
    )


def build_conditions(
    input_modes: list[str],
    pool_sizes: list[int],
    pos_weight_modes: list[str],
    line_width: int,
) -> list[dict]:
    conditions: list[dict] = []
    for input_mode in input_modes:
        for pool_size in pool_sizes:
            for pos_weight_mode in pos_weight_modes:
                condition_name = f"{input_mode}_pool{pool_size}_pos_weight_{pos_weight_mode}"
                conditions.append(
                    {
                        "name": condition_name,
                        "input_mode": input_mode,
                        "pool_size": int(pool_size),
                        "pos_weight_mode": pos_weight_mode,
                        "line_width": int(line_width),
                    }
                )
    return conditions


def build_easy_v2_example_groups(predictions: list[dict], per_group: int) -> list[dict]:
    easy_positive = [dict(record) for record in predictions if int(record.get("easy_positive_v2", 0)) == 1]
    easy_negative = [dict(record) for record in predictions if int(record.get("easy_negative_v2", 0)) == 1]
    false_negative = [record for record in easy_positive if int(record.get("pred_label", 0)) == 0]
    false_positive = [record for record in easy_negative if int(record.get("pred_label", 0)) == 1]
    groups = [
        {
            "name": "easy_positive_v2_false_negative_examples",
            "title": "easy_positive_v2 false negatives",
            "records": sorted(false_negative, key=lambda record: float(record.get("loss", 0.0)), reverse=True)[:per_group],
        },
        {
            "name": "easy_negative_v2_false_positive_examples",
            "title": "easy_negative_v2 false positives",
            "records": sorted(false_positive, key=lambda record: float(record.get("loss", 0.0)), reverse=True)[:per_group],
        },
        {
            "name": "easy_positive_v2_high_loss_examples",
            "title": "easy_positive_v2 high loss",
            "records": sorted(easy_positive, key=lambda record: float(record.get("loss", 0.0)), reverse=True)[:per_group],
        },
        {
            "name": "easy_negative_v2_high_loss_examples",
            "title": "easy_negative_v2 high loss",
            "records": sorted(easy_negative, key=lambda record: float(record.get("loss", 0.0)), reverse=True)[:per_group],
        },
    ]
    return groups


def write_report(
    out_path: str | Path,
    easy_rows: list[dict],
    subset_rows: list[dict],
    easy_only_records: list[dict],
    reference_condition: str,
    example_paths: list[Path],
) -> None:
    easy_positive = sum(int(record.get("connected", 0)) for record in easy_only_records)
    easy_negative = len(easy_only_records) - easy_positive
    lines = [
        "# Region Pair Positive Failure Diagnostics",
        "",
        "## Easy-only v2 Dataset",
        "",
        f"- records: `{len(easy_only_records)}`",
        f"- positive: `{easy_positive}`",
        f"- negative: `{easy_negative}`",
        "",
        "## Easy-only v2 Results",
        "",
        "| condition_name | input_mode | pool_size | pos_weight_mode | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in easy_rows:
        lines.append(
            f"| {row['condition_name']} | {row['input_mode']} | {row['pool_size']} | {row['pos_weight_mode']} | "
            f"{row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | "
            f"{row['mean_loss']} | {row['mean_uncertainty']} |"
        )

    lines.extend(
        [
            "",
            "## Subset v2 Results",
            "",
            "| condition_name | input_mode | pool_size | pos_weight_mode | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in subset_rows:
        lines.append(
            f"| {row['condition_name']} | {row['input_mode']} | {row['pool_size']} | {row['pos_weight_mode']} | "
            f"{row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | "
            f"{row['mean_loss']} | {row['mean_uncertainty']} |"
        )

    lines.extend(
        [
            "",
            "## Easy-v2 Example Files",
            "",
            f"- reference_condition: `{reference_condition}`",
        ]
    )
    for path in example_paths:
        lines.append(f"- [{Path(path).name}]({path})")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
