from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from connectivity_experiment.io_utils import ensure_dir, write_csv, write_json
from connectivity_experiment.region_pair_cnn import (
    SimplePairCNN,
    assign_image_splits,
    build_dataloaders,
    correlation_report,
    input_channels_for_mode,
    performance_by_group,
    predict_with_records,
    quantile_bin_summary,
    save_analysis_outputs,
    summarize_predictions,
    train_method_vs_test_method_matrix,
    train_model,
)
from connectivity_experiment.region_pair_example_visualization import save_example_grids, save_grid


@dataclass
class PairCNNExperimentResult:
    output_dir: Path
    split_by_image: dict[str, str]
    split_records: dict[str, list[dict]]
    test_predictions: list[dict]
    test_summary: dict
    correlation_summary: list[dict]
    bin_summaries: list[dict]
    sampling_method_performance: list[dict]
    train_test_method_matrix: list[dict]
    training_history: list[dict]
    best_val_loss: float


def run_pair_cnn_experiment(
    records: list[dict],
    output_dir: str | Path,
    split_config: dict,
    model_config: dict,
    seed: int,
    split_by_image: dict[str, str] | None = None,
    save_outputs: bool = True,
) -> PairCNNExperimentResult:
    output_dir = Path(output_dir)
    input_mode = str(model_config.get("input_mode", "mask_point_i_point_j"))
    if split_by_image is None:
        split_by_image = assign_image_splits(
            records,
            seed=seed,
            train_fraction=float(split_config["train_fraction"]),
            val_fraction=float(split_config["val_fraction"]),
        )
    split_records, datasets, loaders = build_dataloaders(
        records,
        split_by_image=split_by_image,
        input_size=int(model_config["input_size"]),
        marker_radius=int(model_config["marker_radius"]),
        batch_size=int(model_config["batch_size"]),
        input_mode=input_mode,
        line_width=int(model_config.get("line_width", 1)),
    )

    device = select_device(model_config)
    model = SimplePairCNN(
        input_channels=input_channels_for_mode(input_mode),
        pool_size=int(model_config.get("pool_size", 4)),
    ).to(device)
    artifacts = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        train_labels=[int(float(record["connected"])) for record in split_records["train"]],
        output_dir=output_dir / "model",
        epochs=int(model_config["epochs"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
        seed=seed,
        device=device,
        pos_weight_mode=str(model_config.get("pos_weight_mode", "balanced")),
    )
    test_predictions = predict_with_records(
        model=model,
        loader=loaders["test"],
        records=split_records["test"],
        device=device,
    )
    test_summary = summarize_predictions(test_predictions)
    test_summary["best_val_loss"] = round(float(artifacts.best_val_loss), 6)
    test_summary["device"] = device
    test_summary["input_mode"] = input_mode
    test_summary["pool_size"] = int(model_config.get("pool_size", 4))
    test_summary["line_width"] = int(model_config.get("line_width", 1))
    test_summary["pos_weight_mode"] = str(model_config.get("pos_weight_mode", "balanced"))
    test_summary["n_train"] = len(split_records["train"])
    test_summary["n_val"] = len(split_records["val"])
    test_summary["n_test"] = len(split_records["test"])
    test_summary["n_train_images"] = len({record["image_id"] for record in split_records["train"]})
    test_summary["n_val_images"] = len({record["image_id"] for record in split_records["val"]})
    test_summary["n_test_images"] = len({record["image_id"] for record in split_records["test"]})

    correlation_summary = correlation_report(test_predictions)
    sampling_method_performance = performance_by_group(test_predictions, key="sampling_method")
    train_test_method_rows = train_method_vs_test_method_matrix(test_predictions, train_method="mixed_train")
    bin_summaries = []
    for subset_name, feature_key in [
        ("all", "difficulty_proxy"),
        ("all", "boundary_crossing_count"),
        ("all", "euclidean_distance"),
        ("connected_only", "geodesic_distance"),
        ("connected_only", "geodesic_euclidean_ratio"),
    ]:
        bin_summaries.extend(
            quantile_bin_summary(
                test_predictions,
                feature_key=feature_key,
                subset_name=subset_name,
                n_bins=4,
            )
        )

    if save_outputs:
        save_analysis_outputs(
            out_dir=output_dir,
            split_records=split_records,
            test_predictions=test_predictions,
            training_history=artifacts.history,
            test_summary=test_summary,
            correlation_summary=correlation_summary,
            bin_summaries=bin_summaries,
            sampling_method_performance=sampling_method_performance,
            train_test_method_matrix=train_test_method_rows,
            split_by_image=split_by_image,
        )

    return PairCNNExperimentResult(
        output_dir=output_dir,
        split_by_image=split_by_image,
        split_records=split_records,
        test_predictions=test_predictions,
        test_summary=test_summary,
        correlation_summary=correlation_summary,
        bin_summaries=bin_summaries,
        sampling_method_performance=sampling_method_performance,
        train_test_method_matrix=train_test_method_rows,
        training_history=artifacts.history,
        best_val_loss=float(artifacts.best_val_loss),
    )


def filter_easy_only_records(records: list[dict]) -> list[dict]:
    return [
        dict(record)
        for record in records
        if int(record.get("is_easy_positive", 0)) == 1 or int(record.get("is_easy_negative", 0)) == 1
    ]


def filter_easy_only_records_v2(records: list[dict], seed: int = 0) -> list[dict]:
    return _balanced_flagged_records_by_image(
        records=records,
        positive_flag_key="easy_positive_v2",
        negative_flag_key="easy_negative_v2",
        seed=seed,
    )


def run_input_ablation(
    records: list[dict],
    output_dir: str | Path,
    split_config: dict,
    model_config: dict,
    seed: int,
    input_modes: list[str],
    split_by_image: dict[str, str] | None = None,
) -> tuple[dict[str, PairCNNExperimentResult], list[dict]]:
    output_dir = ensure_dir(output_dir)
    results: dict[str, PairCNNExperimentResult] = {}
    summary_rows: list[dict] = []
    shared_split_by_image = split_by_image
    for input_mode in input_modes:
        mode_model_config = dict(model_config)
        mode_model_config["input_mode"] = input_mode
        mode_output_dir = output_dir / input_mode
        result = run_pair_cnn_experiment(
            records=records,
            output_dir=mode_output_dir,
            split_config=split_config,
            model_config=mode_model_config,
            seed=seed,
            split_by_image=shared_split_by_image,
            save_outputs=True,
        )
        if shared_split_by_image is None:
            shared_split_by_image = result.split_by_image
        results[input_mode] = result
        summary_rows.append(
            {
                "input_mode": input_mode,
                "accuracy": result.test_summary["accuracy"],
                "balanced_accuracy": result.test_summary["balanced_accuracy"],
                "positive_accuracy": result.test_summary["positive_accuracy"],
                "negative_accuracy": result.test_summary["negative_accuracy"],
                "mean_loss": result.test_summary["mean_loss"],
                "mean_uncertainty": result.test_summary["mean_uncertainty"],
                "mean_true_class_confidence": result.test_summary["mean_true_class_confidence"],
                "n_test": result.test_summary["n_test"],
            }
        )
    write_csv(output_dir / "input_ablation_summary.csv", summary_rows)
    _write_input_ablation_report(output_dir / "input_ablation_summary.md", summary_rows)
    return results, summary_rows


def run_model_grid(
    records: list[dict],
    output_dir: str | Path,
    split_config: dict,
    base_model_config: dict,
    seed: int,
    conditions: list[dict],
    split_by_image: dict[str, str] | None = None,
) -> tuple[dict[str, PairCNNExperimentResult], list[dict]]:
    output_dir = ensure_dir(output_dir)
    results: dict[str, PairCNNExperimentResult] = {}
    rows: list[dict] = []
    shared_split_by_image = split_by_image
    for condition in conditions:
        condition_name = str(condition["name"])
        mode_model_config = dict(base_model_config)
        for key, value in condition.items():
            if key != "name":
                mode_model_config[key] = value
        result = run_pair_cnn_experiment(
            records=records,
            output_dir=output_dir / condition_name,
            split_config=split_config,
            model_config=mode_model_config,
            seed=seed,
            split_by_image=shared_split_by_image,
            save_outputs=True,
        )
        if shared_split_by_image is None:
            shared_split_by_image = result.split_by_image
        results[condition_name] = result
        rows.append(
            {
                "condition_name": condition_name,
                "input_mode": result.test_summary["input_mode"],
                "pool_size": result.test_summary["pool_size"],
                "line_width": result.test_summary["line_width"],
                "pos_weight_mode": result.test_summary["pos_weight_mode"],
                "accuracy": result.test_summary["accuracy"],
                "balanced_accuracy": result.test_summary["balanced_accuracy"],
                "positive_accuracy": result.test_summary["positive_accuracy"],
                "negative_accuracy": result.test_summary["negative_accuracy"],
                "mean_loss": result.test_summary["mean_loss"],
                "mean_uncertainty": result.test_summary["mean_uncertainty"],
                "mean_true_class_confidence": result.test_summary["mean_true_class_confidence"],
                "n_test": result.test_summary["n_test"],
            }
        )
    write_csv(output_dir / "model_grid_summary.csv", rows)
    _write_model_grid_report(output_dir / "model_grid_summary.md", rows)
    return results, rows


def analyze_random_high_vs_low_loss(
    predictions: list[dict],
    out_dir: str | Path,
    fraction: float = 0.2,
) -> list[dict]:
    out_dir = ensure_dir(out_dir)
    random_records = [dict(record) for record in predictions if str(record.get("sampling_method")) == "random"]
    if not random_records:
        raise ValueError("No random sampling_method predictions were available for analysis.")
    sorted_by_loss = sorted(random_records, key=lambda record: float(record["loss"]))
    bucket_size = max(1, int(round(len(sorted_by_loss) * fraction)))
    low_loss = sorted_by_loss[:bucket_size]
    high_loss = sorted_by_loss[-bucket_size:]

    feature_specs = [
        ("all", "euclidean_distance"),
        ("all", "boundary_crossing_count"),
        ("all", "distance_to_boundary_min"),
        ("all", "foreground_fraction"),
        ("all", "component_count"),
        ("all", "component_area_i"),
        ("all", "component_area_j"),
        ("connected_only", "geodesic_distance"),
        ("connected_only", "geodesic_euclidean_ratio"),
        ("disconnected_only", "component_gap_distance"),
        ("disconnected_only", "max_outside_segment_length"),
    ]
    rows = []
    for subset_name, feature_key in feature_specs:
        high_subset = _filter_prediction_subset(high_loss, subset_name)
        low_subset = _filter_prediction_subset(low_loss, subset_name)
        high_values = _numeric_values(high_subset, feature_key)
        low_values = _numeric_values(low_subset, feature_key)
        if not high_values and not low_values:
            continue
        rows.append(
            {
                "subset": subset_name,
                "feature": feature_key,
                "high_count": len(high_values),
                "high_mean": _round_or_blank(_safe_mean(high_values)),
                "high_median": _round_or_blank(_safe_median(high_values)),
                "high_q25": _round_or_blank(_safe_quantile(high_values, 0.25)),
                "high_q75": _round_or_blank(_safe_quantile(high_values, 0.75)),
                "low_count": len(low_values),
                "low_mean": _round_or_blank(_safe_mean(low_values)),
                "low_median": _round_or_blank(_safe_median(low_values)),
                "low_q25": _round_or_blank(_safe_quantile(low_values, 0.25)),
                "low_q75": _round_or_blank(_safe_quantile(low_values, 0.75)),
                "delta_mean_high_minus_low": _round_or_blank(_safe_mean(high_values) - _safe_mean(low_values) if high_values and low_values else None),
            }
        )

    write_csv(out_dir / "random_high_vs_low_loss_feature_summary.csv", rows)
    _write_random_high_low_report(
        out_path=out_dir / "random_high_vs_low_loss_feature_summary.md",
        rows=rows,
        n_random=len(random_records),
        n_bucket=bucket_size,
    )
    return rows


def write_followup_report(
    out_path: str | Path,
    baseline_result: PairCNNExperimentResult,
    easy_only_result: PairCNNExperimentResult,
    easy_only_count: int,
    ablation_rows: list[dict],
    random_analysis_rows: list[dict],
    pair_summary_rows: list[dict],
    coverage_rows: list[dict],
    difficulty_rows: list[dict],
    example_paths: list[str | Path] | None = None,
) -> None:
    compare_methods = {
        "boundary_crossing_targeted",
        "boundary_crossing_stratified",
        "hard_pair_targeted",
        "hard_pair_targeted_v2",
    }
    lines = [
        "# Region Pair Follow-up Diagnostics",
        "",
        "## Easy-only Sanity Check",
        "",
        f"- records: `{easy_only_count}`",
        f"- accuracy: `{easy_only_result.test_summary['accuracy']}`",
        f"- balanced_accuracy: `{easy_only_result.test_summary['balanced_accuracy']}`",
        f"- positive_accuracy: `{easy_only_result.test_summary['positive_accuracy']}`",
        f"- negative_accuracy: `{easy_only_result.test_summary['negative_accuracy']}`",
        "",
        "## Input Ablation",
        "",
        "| input_mode | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['input_mode']} | {row['accuracy']} | {row['balanced_accuracy']} | "
            f"{row['positive_accuracy']} | {row['negative_accuracy']} | {row['mean_loss']} |"
        )

    lines.extend(
        [
            "",
            "## Random High vs Low Loss",
            "",
            "| subset | feature | high_mean | low_mean | delta_mean_high_minus_low |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in random_analysis_rows:
        lines.append(
            f"| {row['subset']} | {row['feature']} | {row['high_mean']} | {row['low_mean']} | {row['delta_mean_high_minus_low']} |"
        )

    lines.extend(
        [
            "",
            "## Example Visualizations",
            "",
        ]
    )
    if example_paths:
        for path in example_paths:
            path_obj = Path(path)
            lines.append(f"- [{path_obj.name}]({path_obj})")
    else:
        lines.append("- No example visualizations were recorded.")

    lines.extend(
        [
            "",
            "## Method Summary Comparison",
            "",
            "| threshold | method | n_pairs | connected | disconnected | mean_boundary_crossing_count | hard_pairs |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in pair_summary_rows:
        if row["sampling_method"] not in compare_methods:
            continue
        lines.append(
            f"| {row['valid_threshold']} | {row['sampling_method']} | {row['n_pairs']} | {row['connected_pairs']} | {row['disconnected_pairs']} | "
            f"{row['mean_boundary_crossing_count']} | {row['hard_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## Feature Coverage Comparison",
            "",
            "| threshold | method | subset | coverage_ratio | entropy |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in coverage_rows:
        if row["sampling_method"] not in compare_methods:
            continue
        lines.append(
            f"| {row['valid_threshold']} | {row['sampling_method']} | {row['subset']} | {row['coverage_ratio']} | {row['entropy']} |"
        )

    lines.extend(
        [
            "",
            "## Difficulty Bucket Comparison",
            "",
            "| threshold | method | Easy | Medium | Hard | Very hard |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in difficulty_rows:
        if row["sampling_method"] not in compare_methods:
            continue
        lines.append(
            f"| {row['valid_threshold']} | {row['sampling_method']} | {row.get('Easy', 0)} | {row.get('Medium', 0)} | {row.get('Hard', 0)} | {row.get('Very hard', 0)} |"
        )

    lines.extend(
        [
            "",
            "## CNN Performance Comparison",
            "",
            "| method | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in baseline_result.sampling_method_performance:
        if row["group_value"] not in compare_methods:
            continue
        lines.append(
            f"| {row['group_value']} | {row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | {row['mean_loss']} |"
        )

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_example_visualizations(
    predictions_csv: str | Path,
    out_dir: str | Path,
    per_group: int,
) -> list[Path]:
    return save_example_grids(predictions_path=predictions_csv, out_dir=out_dir, per_group=per_group)


def save_named_prediction_grids(
    predictions: list[dict],
    out_dir: str | Path,
    groups: list[dict],
    n_cols: int = 4,
) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    output_paths: list[Path] = []
    for group in groups:
        name = str(group["name"])
        title = str(group.get("title", name.replace("_", " ")))
        rows = list(group["records"])
        png_path = out_dir / f"{name}.png"
        csv_path = out_dir / f"{name}.csv"
        if rows:
            save_grid(rows, out_path=png_path, title=title, n_cols=n_cols)
            write_csv(csv_path, rows)
        else:
            _save_placeholder_grid(png_path, title=f"{title} (no examples)")
            write_csv(
                csv_path,
                [
                    {
                        "example_group": name,
                        "status": "no_examples",
                    }
                ],
            )
        output_paths.extend([png_path, csv_path])
    return output_paths


def select_device(model_config: dict) -> str:
    use_cuda = bool(model_config.get("use_cuda", False))
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    if bool(model_config.get("use_mps", True)) and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_input_ablation_report(out_path: str | Path, rows: list[dict]) -> None:
    lines = [
        "# Input Ablation Summary",
        "",
        "| input_mode | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty | mean_true_class_confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['input_mode']} | {row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | "
            f"{row['negative_accuracy']} | {row['mean_loss']} | {row['mean_uncertainty']} | {row['mean_true_class_confidence']} |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_grid_report(out_path: str | Path, rows: list[dict]) -> None:
    lines = [
        "# Model Grid Summary",
        "",
        "| condition_name | input_mode | pool_size | line_width | pos_weight_mode | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition_name']} | {row['input_mode']} | {row['pool_size']} | {row['line_width']} | {row['pos_weight_mode']} | "
            f"{row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | {row['mean_loss']} | {row['mean_uncertainty']} |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_placeholder_grid(out_path: str | Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_random_high_low_report(
    out_path: str | Path,
    rows: list[dict],
    n_random: int,
    n_bucket: int,
) -> None:
    lines = [
        "# Random High vs Low Loss Feature Summary",
        "",
        f"- random_test_pairs: `{n_random}`",
        f"- high_loss_bucket_size: `{n_bucket}`",
        f"- low_loss_bucket_size: `{n_bucket}`",
        "",
        "| subset | feature | high_count | high_mean | low_count | low_mean | delta_mean_high_minus_low |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset']} | {row['feature']} | {row['high_count']} | {row['high_mean']} | "
            f"{row['low_count']} | {row['low_mean']} | {row['delta_mean_high_minus_low']} |"
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_prediction_subset(records: list[dict], subset_name: str) -> list[dict]:
    if subset_name == "connected_only":
        return [record for record in records if int(record.get("connected", 0)) == 1]
    if subset_name == "disconnected_only":
        return [record for record in records if int(record.get("connected", 0)) == 0]
    return list(records)


def _numeric_values(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if value in (None, "", "NA", "None"):
            continue
        values.append(float(value))
    return values


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _round_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 6)


def _balanced_flagged_records_by_image(
    records: list[dict],
    positive_flag_key: str,
    negative_flag_key: str,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        group = grouped.setdefault(image_id, {"positive": [], "negative": []})
        if int(record.get(positive_flag_key, 0)) == 1:
            group["positive"].append(dict(record))
        if int(record.get(negative_flag_key, 0)) == 1:
            group["negative"].append(dict(record))

    selected: list[dict] = []
    for image_id in sorted(grouped):
        positive_records = grouped[image_id]["positive"]
        negative_records = grouped[image_id]["negative"]
        target = min(len(positive_records), len(negative_records))
        if target <= 0:
            continue
        positive_indices = np.arange(len(positive_records))
        negative_indices = np.arange(len(negative_records))
        if len(positive_indices) > target:
            positive_indices = np.sort(rng.choice(positive_indices, size=target, replace=False))
        if len(negative_indices) > target:
            negative_indices = np.sort(rng.choice(negative_indices, size=target, replace=False))
        selected.extend(positive_records[int(index)] for index in positive_indices.tolist())
        selected.extend(negative_records[int(index)] for index in negative_indices.tolist())
    selected.sort(key=lambda record: str(record.get("pair_id", "")))
    return selected
