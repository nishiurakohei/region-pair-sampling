from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.io_utils import write_csv, write_json, write_jsonl
from connectivity_experiment.region_pair_sampling import (
    attach_region_pair_difficulty,
    build_pair_records,
    build_valid_mask,
    compute_distance_transform,
    discover_mask_paths,
    load_binary_mask,
    pair_feature_triplet_records,
    sample_points,
    save_distribution_plots,
    save_distance_preview,
    save_feature_coverage_heatmap,
    save_mask_preview,
    save_pair_contact_sheet,
    save_points_preview,
    save_valid_mask_preview,
    select_pair_records,
    summarize_mask,
    summarize_difficulty_buckets,
    summarize_feature_coverage,
    summarize_pairs,
    write_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run region pair sampling experiment on existing masks.")
    parser.add_argument("--config", default="configs/region_pair_sampling_v1.yaml")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    output_dir = repo_root / config["paths"]["output_dir"]
    if args.overwrite and output_dir.exists():
        for child in sorted(output_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    phase0_dir = output_dir / "phase0"
    phase1_dir = output_dir / "phase1"
    phase2_dir = output_dir / "phase2"
    phase3_dir = output_dir / "phase3"
    metrics_dir = output_dir / "metrics"
    report_dir = output_dir / "report"
    for directory in [phase0_dir, phase1_dir, phase2_dir, phase3_dir, metrics_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    limit_images = args.limit_images if args.limit_images is not None else int(config["dataset"].get("sample_limit", 6))
    mask_paths = discover_mask_paths(
        mask_glob=str(config["dataset"]["mask_glob"]),
        repo_root=repo_root,
        limit=limit_images,
    )
    if not mask_paths:
        raise FileNotFoundError(
            f"No mask files matched {config['dataset']['mask_glob']!r}. Update the config to point to your masks."
        )

    seed = int(config["experiment"]["seed"])
    mask_channel = int(config["dataset"].get("mask_channel", 0))
    foreground_threshold = int(config["dataset"].get("foreground_threshold", 16))
    invert_mask = bool(config["dataset"].get("invert_mask", False))
    min_component_area = int(config["dataset"].get("min_component_area", 1))
    valid_thresholds = [float(value) for value in config["sampling"]["valid_thresholds"]]
    methods = [str(value) for value in config["sampling"]["methods"]]
    sample_foreground = bool(config["sampling"].get("sample_foreground", True))
    sample_background = bool(config["sampling"].get("sample_background", False))
    points_per_method = int(config["sampling"].get("points_per_method", 12))
    foreground_points_per_method = int(config["sampling"].get("foreground_points_per_method", points_per_method))
    background_points_per_method = int(config["sampling"].get("background_points_per_method", points_per_method if sample_background else 0))
    candidate_pool_limit = int(config["sampling"].get("candidate_pool_limit", 4096))
    n_pairs_per_image = config["sampling"].get("n_pairs_per_image")
    n_pairs_per_image = int(n_pairs_per_image) if n_pairs_per_image not in (None, "", 0) else None
    feature_bin_count = int(config["sampling"].get("feature_bin_count", 4))
    pair_component_mode = str(config["sampling"].get("pair_component_mode", "same_component"))
    preview_limit_images = int(config.get("visualization", {}).get("preview_limit_images", 1000000))

    dataset_summary = []
    threshold_summary = []
    pair_records = []
    mask_lookup: dict[str, np.ndarray] = {}

    for image_index, mask_path in enumerate(mask_paths):
        image_id = f"{image_index:03d}_{mask_path.stem}"
        mask = load_binary_mask(
            mask_path,
            mask_channel=mask_channel,
            foreground_threshold=foreground_threshold,
            invert_mask=invert_mask,
            min_component_area=min_component_area,
        )
        if int(mask.sum()) == 0:
            continue
        mask_lookup[image_id] = mask
        dataset_record = summarize_mask(mask, image_id=image_id, image_path=mask_path)
        dataset_summary.append(dataset_record)
        should_save_previews = image_index < preview_limit_images
        if should_save_previews:
            save_mask_preview(mask, phase0_dir / f"{image_id}_mask.png", title=f"{image_id} mask")
        distance_map = compute_distance_transform(mask)
        background_mask = ~mask
        background_distance_map = compute_distance_transform(background_mask)
        if should_save_previews:
            save_distance_preview(mask, distance_map, phase0_dir / f"{image_id}_foreground_distance.png", title=f"{image_id} foreground distance")
            save_distance_preview(background_mask, background_distance_map, phase0_dir / f"{image_id}_background_distance.png", title=f"{image_id} background distance")

        for threshold_index, valid_threshold in enumerate(valid_thresholds):
            foreground_valid_mask = build_valid_mask(mask, distance_map, min_distance_to_boundary=valid_threshold)
            background_valid_mask = build_valid_mask(background_mask, background_distance_map, min_distance_to_boundary=valid_threshold)
            foreground_valid_pixels = int(foreground_valid_mask.sum())
            background_valid_pixels = int(background_valid_mask.sum())
            if sample_foreground:
                threshold_summary.append(
                    {
                        "image_id": image_id,
                        "image_path": str(mask_path),
                        "region_name": "foreground",
                        "valid_threshold": valid_threshold,
                        "valid_pixels": foreground_valid_pixels,
                        "valid_fraction": round(foreground_valid_pixels / max(1, int(mask.sum())), 6),
                    }
                )
                if should_save_previews:
                    save_valid_mask_preview(
                        mask,
                        foreground_valid_mask,
                        phase1_dir / f"{image_id}_t{_tag(valid_threshold)}_foreground_valid.png",
                        title=f"{image_id} foreground valid threshold={valid_threshold}",
                    )
            if sample_background:
                threshold_summary.append(
                    {
                        "image_id": image_id,
                        "image_path": str(mask_path),
                        "region_name": "background",
                        "valid_threshold": valid_threshold,
                        "valid_pixels": background_valid_pixels,
                        "valid_fraction": round(background_valid_pixels / max(1, int(background_mask.sum())), 6),
                    }
                )
                if should_save_previews:
                    save_valid_mask_preview(
                        background_mask,
                        background_valid_mask,
                        phase1_dir / f"{image_id}_t{_tag(valid_threshold)}_background_valid.png",
                        title=f"{image_id} background valid threshold={valid_threshold}",
                    )
            if foreground_valid_pixels == 0 and background_valid_pixels == 0:
                continue

            for method_index, method in enumerate(methods):
                points: list[dict] = []
                if sample_foreground and foreground_valid_pixels > 0 and foreground_points_per_method > 0:
                    rng = np.random.default_rng(seed + image_index * 1009 + threshold_index * 131 + method_index * 17 + 1)
                    foreground_points = sample_points(
                        valid_mask=foreground_valid_mask,
                        distance_map=distance_map,
                        method=method,
                        n_points=foreground_points_per_method,
                        candidate_pool_limit=candidate_pool_limit,
                        rng=rng,
                    )
                    points.extend(
                        [
                            {
                                "row": int(row),
                                "col": int(col),
                                "region_value": 1,
                                "region_name": "foreground",
                                "distance_to_boundary": float(distance_map[row, col]),
                            }
                            for row, col in foreground_points
                        ]
                    )
                if sample_background and background_valid_pixels > 0 and background_points_per_method > 0:
                    rng = np.random.default_rng(seed + image_index * 1009 + threshold_index * 131 + method_index * 17 + 10001)
                    background_points = sample_points(
                        valid_mask=background_valid_mask,
                        distance_map=background_distance_map,
                        method=method,
                        n_points=background_points_per_method,
                        candidate_pool_limit=candidate_pool_limit,
                        rng=rng,
                    )
                    points.extend(
                        [
                            {
                                "row": int(row),
                                "col": int(col),
                                "region_value": 0,
                                "region_name": "background",
                                "distance_to_boundary": float(background_distance_map[row, col]),
                            }
                            for row, col in background_points
                        ]
                    )
                if not points:
                    continue
                if should_save_previews:
                    save_points_preview(
                        mask,
                        np.logical_or(foreground_valid_mask, background_valid_mask),
                        points,
                        phase1_dir / f"{image_id}_t{_tag(valid_threshold)}_{method}_points.png",
                        title=f"{image_id} {method} threshold={valid_threshold} fg={sum(int(point['region_value'] == 1) for point in points)} bg={sum(int(point['region_value'] == 0) for point in points)}",
                    )
                candidate_pair_records = build_pair_records(
                    mask=mask,
                    distance_map=distance_map,
                    points=points,
                    image_id=image_id,
                    image_path=mask_path,
                    valid_threshold=valid_threshold,
                    sampling_method=method,
                    background_distance_map=background_distance_map,
                    pair_component_mode=pair_component_mode,
                )
                attach_region_pair_difficulty(candidate_pair_records)
                selection_rng = np.random.default_rng(seed + image_index * 4001 + threshold_index * 257 + method_index * 29 + 200001)
                pair_records.extend(
                    select_pair_records(
                        candidate_pair_records,
                        method=method,
                        target_count=n_pairs_per_image,
                        rng=selection_rng,
                        feature_bin_count=feature_bin_count,
                    )
                )

    attach_region_pair_difficulty(pair_records)
    pair_summary = summarize_pairs(pair_records)
    feature_coverage_summary = summarize_feature_coverage(pair_records, n_bins=feature_bin_count)
    difficulty_bucket_summary = summarize_difficulty_buckets(pair_records)
    triplet_records = pair_feature_triplet_records(pair_records)

    if not dataset_summary:
        raise RuntimeError("No non-empty masks were available after binarization.")
    if not pair_records:
        raise RuntimeError("No valid point pairs were produced. Lower the valid threshold or increase points_per_method.")

    write_csv(metrics_dir / "dataset_summary.csv", dataset_summary)
    write_csv(metrics_dir / "threshold_summary.csv", threshold_summary)
    write_csv(metrics_dir / "pair_records.csv", pair_records)
    write_jsonl(metrics_dir / "pair_records.jsonl", pair_records)
    write_csv(metrics_dir / "pair_summary.csv", pair_summary)
    write_csv(metrics_dir / "sampling_method_summary.csv", pair_summary)
    write_csv(metrics_dir / "feature_coverage_summary.csv", feature_coverage_summary)
    write_csv(metrics_dir / "difficulty_bucket_summary.csv", difficulty_bucket_summary)
    write_json(metrics_dir / "pair_feature_triplets.json", triplet_records)
    write_jsonl(metrics_dir / "pair_feature_triplets.jsonl", triplet_records)

    plot_notes = save_distribution_plots(pair_records, pair_summary, out_dir=phase3_dir)
    save_feature_coverage_heatmap(feature_coverage_summary, phase3_dir / "feature_coverage_heatmap.png")
    save_pair_contact_sheet(
        pair_records,
        mask_lookup=mask_lookup,
        out_path=phase2_dir / "pairs_high_ratio.png",
        title="High geodesic/euclidean ratio pairs",
        sort_key="geodesic_euclidean_ratio",
    )
    save_pair_contact_sheet(
        pair_records,
        mask_lookup=mask_lookup,
        out_path=phase2_dir / "pairs_high_boundary_crossing.png",
        title="High boundary crossing pairs",
        sort_key="boundary_crossing_count",
    )
    save_pair_contact_sheet(
        pair_records,
        mask_lookup=mask_lookup,
        out_path=phase2_dir / "pairs_high_difficulty.png",
        title="High difficulty proxy pairs",
        sort_key="difficulty_proxy",
    )
    save_pair_contact_sheet(
        [record for record in pair_records if int(record.get("connected", 0)) == 0],
        mask_lookup=mask_lookup,
        out_path=phase2_dir / "pairs_near_miss_negative.png",
        title="Near-miss negative pairs",
        sort_key="near_miss_score",
    )
    save_pair_contact_sheet(
        pair_records,
        mask_lookup=mask_lookup,
        out_path=phase2_dir / "pairs_hard_pair_target_score.png",
        title="Hard pair targeted examples",
        sort_key="hard_pair_score",
    )
    write_experiment_report(
        out_path=report_dir / "region_pair_sampling_report.md",
        config_path=args.config,
        dataset_summary=dataset_summary,
        threshold_summary=threshold_summary,
        pair_summary=pair_summary,
        feature_coverage_summary=feature_coverage_summary,
        difficulty_bucket_summary=difficulty_bucket_summary,
        pair_records=pair_records,
        plot_notes=plot_notes,
    )
    print(f"[run_region_pair_sampling] images={len(dataset_summary)} pairs={len(pair_records)} -> {output_dir}")


def _tag(value: float) -> str:
    if int(value) == value:
        return str(int(value))
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
