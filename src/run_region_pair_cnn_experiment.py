from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.region_pair_cnn import (
    SimplePairCNN,
    assign_image_splits,
    build_dataloaders,
    correlation_report,
    input_channels_for_mode,
    load_pair_records,
    normalize_input_mode,
    performance_by_group,
    predict_with_records,
    quantile_bin_summary,
    save_analysis_outputs,
    summarize_predictions,
    train_method_vs_test_method_matrix,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN on region pair connectivity data.")
    parser.add_argument("--config", default="configs/region_pair_cnn_stare20_v1.yaml")
    parser.add_argument("--regenerate-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)

    pair_generation_config = str(config["data"]["pair_generation_config"])
    if args.regenerate_pairs or bool(config["data"].get("regenerate_pairs", False)):
        command = [sys.executable, str(repo_root / "src" / "run_region_pair_sampling.py"), "--config", pair_generation_config, "--overwrite"]
        subprocess.run(command, check=True, cwd=repo_root)

    pair_metrics_csv = repo_root / config["data"]["pair_metrics_csv"]
    records = load_pair_records(pair_metrics_csv)
    split_by_image = assign_image_splits(
        records,
        seed=int(config["experiment"]["seed"]),
        train_fraction=float(config["split"]["train_fraction"]),
        val_fraction=float(config["split"]["val_fraction"]),
    )
    split_records, datasets, loaders = build_dataloaders(
        records,
        split_by_image=split_by_image,
        input_size=int(config["model"]["input_size"]),
        marker_radius=int(config["model"]["marker_radius"]),
        batch_size=int(config["model"]["batch_size"]),
        input_mode=str(config["model"].get("input_mode", "mask_point_i_point_j")),
        line_width=int(config["model"].get("line_width", 1)),
    )

    use_cuda = bool(config["model"].get("use_cuda", False))
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
    elif bool(config["model"].get("use_mps", True)) and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    input_mode = normalize_input_mode(str(config["model"].get("input_mode", "mask_point_i_point_j")))
    model = SimplePairCNN(
        input_channels=input_channels_for_mode(input_mode),
        pool_size=int(config["model"].get("pool_size", 4)),
    ).to(device)
    artifacts = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        train_labels=[int(float(record["connected"])) for record in split_records["train"]],
        output_dir=repo_root / config["paths"]["output_dir"] / "model",
        epochs=int(config["model"]["epochs"]),
        learning_rate=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
        seed=int(config["experiment"]["seed"]),
        device=device,
        pos_weight_mode=str(config["model"].get("pos_weight_mode", "balanced")),
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
    test_summary["pool_size"] = int(config["model"].get("pool_size", 4))
    test_summary["line_width"] = int(config["model"].get("line_width", 1))
    test_summary["pos_weight_mode"] = str(config["model"].get("pos_weight_mode", "balanced"))
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
        bin_summaries.extend(quantile_bin_summary(test_predictions, feature_key=feature_key, subset_name=subset_name, n_bins=4))

    save_analysis_outputs(
        out_dir=repo_root / config["paths"]["output_dir"],
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
    print(
        f"[region_pair_cnn] test_accuracy={test_summary['accuracy']:.4f} "
        f"mean_conf={test_summary['mean_true_class_confidence']:.4f} "
        f"-> {repo_root / config['paths']['output_dir']}"
    )


if __name__ == "__main__":
    main()
