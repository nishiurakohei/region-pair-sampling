from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .io_utils import ensure_dir, read_csv, write_json, write_jsonl
from .region_pair_sampling import bresenham_line, connected_components
from .region_pair_generation import CSV_FIELD_ORDER, DEFAULT_EXTENSIONS, load_binary_mask

DEFAULT_REVIEW_THRESHOLDS = {
    "foreground_ratio_delta_threshold": 0.15,
    "component_count_delta_threshold": 3,
    "changed_pixel_ratio_threshold": 0.35,
    "line_changed_fraction_threshold": 0.4,
    "near_empty_foreground_threshold": 0.01,
    "near_full_foreground_threshold": 0.99,
}

CORRUPTION_FIELD_ORDER = [
    "pair_id",
    "image_id",
    "mask_path",
    "point_a_y",
    "point_a_x",
    "point_b_y",
    "point_b_x",
    "connected_original",
    "connected_after_corruption",
    "label_changed_by_corruption",
    "label_source",
    "corruption_method",
    "corruption_trial_id",
    "corruption_params_json",
    "corrupted_image_path",
    "euclidean_distance_original",
    "geodesic_distance_original",
    "geodesic_euclidean_ratio_original",
    "boundary_crossing_count_original",
    "distance_to_boundary_min_original",
    "foreground_ratio_original",
    "foreground_ratio_corrupted",
    "changed_pixel_ratio",
    "component_count_original",
    "component_count_corrupted",
    "needs_human_review",
    "review_reasons",
]

ROBUSTNESS_SUMMARY_FIELD_ORDER = [
    "corruption_method",
    "num_samples",
    "accuracy_original",
    "accuracy_corrupted",
    "accuracy_drop",
    "mean_confidence_drop",
    "mean_loss_increase",
    "prediction_flip_rate",
    "label_changed_rate",
    "human_review_rate",
    "mean_foreground_ratio_original",
    "mean_foreground_ratio_corrupted",
    "mean_foreground_ratio_delta",
    "mean_changed_pixel_ratio",
    "mean_component_count_original",
    "mean_component_count_corrupted",
    "mean_component_count_delta",
    "point_a_masked_rate",
    "point_b_masked_rate",
    "mean_line_changed_fraction",
]


def apply_corruption_to_fixed_pair_input(
    image: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    method: str,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    method = _normalize_corruption_method(method)
    mask = image.astype(bool)
    if method == "none":
        return mask.copy(), {"method": "none"}
    if method == "random_rectangle_mask":
        return _apply_random_rectangle_mask(mask, params=params, rng=rng)
    if method == "random_circle_mask":
        return _apply_random_circle_mask(mask, params=params, rng=rng)
    if method == "line_occlusion":
        return _apply_line_occlusion(mask, params=params, rng=rng)
    if method == "point_neighborhood_mask":
        return _apply_point_neighborhood_mask(mask, point_a=point_a, point_b=point_b, params=params)
    if method == "salt_and_pepper":
        return _apply_salt_and_pepper(mask, params=params, rng=rng)
    if method == "gaussian_noise":
        return _apply_gaussian_noise(mask, params=params, rng=rng)
    if method == "blur":
        return _apply_blur(mask, params=params)
    if method == "erosion":
        return _apply_morphology(mask, params=params, operation="erosion")
    if method == "dilation":
        return _apply_morphology(mask, params=params, operation="dilation")
    raise ValueError(f"Unknown corruption method: {method}")


def compute_corruption_summary(
    original_mask: np.ndarray,
    corrupted_mask: np.ndarray,
) -> dict[str, Any]:
    original_mask = original_mask.astype(bool)
    corrupted_mask = corrupted_mask.astype(bool)
    _, original_component_sizes = connected_components(original_mask)
    _, corrupted_component_sizes = connected_components(corrupted_mask)
    total_pixels = max(1, original_mask.shape[0] * original_mask.shape[1])
    return {
        "foreground_ratio_original": round(float(original_mask.sum()) / total_pixels, 6),
        "foreground_ratio_corrupted": round(float(corrupted_mask.sum()) / total_pixels, 6),
        "changed_pixel_ratio": round(float(np.mean(original_mask != corrupted_mask)), 6),
        "component_count_original": int(len(original_component_sizes)),
        "component_count_corrupted": int(len(corrupted_component_sizes)),
    }


def recompute_label_after_corruption(
    corrupted_mask: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
) -> int:
    if not _is_inside(corrupted_mask.shape, point_a) or not _is_inside(corrupted_mask.shape, point_b):
        return 0
    if not bool(corrupted_mask[point_a]) or not bool(corrupted_mask[point_b]):
        return 0
    labels, _ = connected_components(corrupted_mask.astype(bool))
    component_a = int(labels[point_a])
    component_b = int(labels[point_b])
    return int(component_a >= 0 and component_a == component_b)


def build_corruption_explanation(
    corruption_method: str,
    corruption_params: dict[str, Any],
    connected_original: int,
    connected_after_corruption: int,
    corruption_summary: dict[str, Any],
) -> tuple[str, list[str], bool]:
    thresholds = dict(DEFAULT_REVIEW_THRESHOLDS)
    thresholds.update(corruption_summary.get("_review_thresholds", {}))
    review_reasons: list[str] = []

    if int(connected_original) != int(connected_after_corruption):
        review_reasons.append("LABEL_CHANGED_AFTER_CORRUPTION")
    foreground_delta = abs(
        float(corruption_summary["foreground_ratio_original"]) - float(corruption_summary["foreground_ratio_corrupted"])
    )
    if foreground_delta >= float(thresholds["foreground_ratio_delta_threshold"]):
        review_reasons.append("FOREGROUND_RATIO_CHANGED_LARGE")
    component_delta = abs(
        int(corruption_summary["component_count_original"]) - int(corruption_summary["component_count_corrupted"])
    )
    if component_delta >= int(thresholds["component_count_delta_threshold"]):
        review_reasons.append("COMPONENT_COUNT_CHANGED_LARGE")
    if float(corruption_summary["changed_pixel_ratio"]) >= float(thresholds["changed_pixel_ratio_threshold"]):
        review_reasons.append("TOO_MANY_PIXELS_CHANGED")
    if int(corruption_summary.get("point_a_preserved", 1)) == 0 or int(corruption_summary.get("point_b_preserved", 1)) == 0:
        review_reasons.append("POINT_LOCATION_MASKED_OR_FLIPPED")
    if (
        corruption_method in {"line_occlusion", "random_rectangle_mask", "random_circle_mask", "point_neighborhood_mask"}
        and float(corruption_summary.get("line_changed_fraction", 0.0)) >= float(thresholds["line_changed_fraction_threshold"])
    ):
        review_reasons.append("PAIR_PATH_STRONGLY_OCCLUDED")
    if (
        float(corruption_summary["foreground_ratio_corrupted"]) <= float(thresholds["near_empty_foreground_threshold"])
        or float(corruption_summary["foreground_ratio_corrupted"]) >= float(thresholds["near_full_foreground_threshold"])
    ):
        review_reasons.append("CORRUPTED_MASK_DEGENERATE")

    sentences = [
        _corruption_method_sentence(corruption_method),
        "The point pair and original label are kept fixed so the experiment measures robustness to post-hoc image degradation.",
    ]
    if int(connected_original) != int(connected_after_corruption):
        sentences.append("This corruption also changes the underlying connectivity on the corrupted mask, so the sample should be interpreted carefully.")
    if corruption_params:
        sentences.append(f"Applied parameters: {json.dumps(corruption_params, ensure_ascii=False)}.")
    return " ".join(sentences), review_reasons, bool(review_reasons)


def compute_robustness_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["corruption_method"]), []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for method, method_records in sorted(grouped.items()):
        summary_rows.append(
            {
                "corruption_method": method,
                "num_samples": len(method_records),
                "accuracy_original": _optional_accuracy(method_records, pred_key="pred_original", label_key="connected_original"),
                "accuracy_corrupted": _optional_accuracy(method_records, pred_key="pred_corrupted", label_key="connected_original"),
                "accuracy_drop": _optional_difference(
                    _optional_accuracy(method_records, pred_key="pred_original", label_key="connected_original"),
                    _optional_accuracy(method_records, pred_key="pred_corrupted", label_key="connected_original"),
                ),
                "mean_confidence_drop": _optional_mean_difference(method_records, "prob_true_label_original", "prob_true_label_corrupted"),
                "mean_loss_increase": _optional_mean_difference(method_records, "loss_corrupted", "loss_original"),
                "prediction_flip_rate": _optional_prediction_flip_rate(method_records),
                "label_changed_rate": _rate(method_records, "label_changed_by_corruption"),
                "human_review_rate": _rate(method_records, "needs_human_review"),
                "mean_foreground_ratio_original": _mean(method_records, "foreground_ratio_original"),
                "mean_foreground_ratio_corrupted": _mean(method_records, "foreground_ratio_corrupted"),
                "mean_foreground_ratio_delta": round(
                    float(
                        np.mean(
                            [
                                abs(float(record["foreground_ratio_original"]) - float(record["foreground_ratio_corrupted"]))
                                for record in method_records
                            ]
                        )
                    ),
                    6,
                ),
                "mean_changed_pixel_ratio": _mean(method_records, "changed_pixel_ratio"),
                "mean_component_count_original": _mean(method_records, "component_count_original"),
                "mean_component_count_corrupted": _mean(method_records, "component_count_corrupted"),
                "mean_component_count_delta": round(
                    float(
                        np.mean(
                            [
                                abs(float(record["component_count_original"]) - float(record["component_count_corrupted"]))
                                for record in method_records
                            ]
                        )
                    ),
                    6,
                ),
                "point_a_masked_rate": round(
                    float(np.mean([1.0 - float(record.get("point_a_preserved", 1)) for record in method_records])),
                    6,
                ),
                "point_b_masked_rate": round(
                    float(np.mean([1.0 - float(record.get("point_b_preserved", 1)) for record in method_records])),
                    6,
                ),
                "mean_line_changed_fraction": _mean(method_records, "line_changed_fraction"),
            }
        )
    return summary_rows


def run_pair_corruption_evaluation_from_config(
    config: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    input_pairs_csv = _resolve_path(config.get("input_pairs_csv", "outputs/generated_pairs/pairs.csv"), repo_root=repo_root)
    output_dir = ensure_dir(_resolve_path(config.get("output_dir", "outputs/pair_corruption_evaluation"), repo_root=repo_root))
    corrupted_image_dir = ensure_dir(output_dir / "corrupted_images")
    input_image_dir_value = config.get("input_image_dir")
    input_image_dir = _resolve_path(input_image_dir_value, repo_root=repo_root) if input_image_dir_value else None

    seed = int(config.get("seed", 0))
    label_source = str(config.get("label_source", "original"))
    mask_channel = int(config.get("mask_channel", 0))
    foreground_threshold = int(config.get("foreground_threshold", 16))
    invert_mask = bool(config.get("invert_mask", False))
    min_component_size = int(config.get("min_component_size", 1))
    max_pairs = config.get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs not in (None, "", 0) else None
    pair_selection_strategy = str(config.get("pair_selection_strategy", "head"))

    corruption_config = dict(config.get("corruption", {}))
    methods = [
        _normalize_corruption_method(value)
        for value in corruption_config.get("methods", ["none", "random_rectangle_mask", "random_circle_mask", "line_occlusion", "point_neighborhood_mask", "salt_and_pepper", "gaussian_noise", "blur"])
    ]
    num_trials_per_method = int(corruption_config.get("num_trials_per_method", 3))
    review_thresholds = dict(DEFAULT_REVIEW_THRESHOLDS)
    review_thresholds.update(corruption_config.get("review", {}))

    pair_rows = read_csv(input_pairs_csv)
    pair_rows = _select_input_pairs(
        pair_rows=pair_rows,
        max_pairs=max_pairs,
        strategy=pair_selection_strategy,
    )

    corrupted_records: list[dict[str, Any]] = []
    mask_cache: dict[str, np.ndarray] = {}

    for pair_index, pair_row in enumerate(pair_rows):
        mask_path = _resolve_pair_mask_path(pair_row, input_image_dir=input_image_dir, repo_root=repo_root)
        cache_key = str(mask_path)
        if cache_key not in mask_cache:
            mask_cache[cache_key] = load_binary_mask(
                image_path=mask_path,
                mask_channel=mask_channel,
                foreground_threshold=foreground_threshold,
                invert_mask=invert_mask,
                min_component_size=min_component_size,
            )
        original_mask = mask_cache[cache_key]
        point_a = _extract_point(pair_row, "a")
        point_b = _extract_point(pair_row, "b")
        region_value_i = int(_get_first_available(pair_row, ["region_value_i"], default=1))
        region_value_j = int(_get_first_available(pair_row, ["region_value_j"], default=1))
        connected_original = int(_get_first_available(pair_row, ["connected_original", "connected"], default=0))
        line_changed_reference = bresenham_line(point_a, point_b)

        for method_index, method in enumerate(methods):
            trials = 1 if method == "none" else max(1, num_trials_per_method)
            params = dict(corruption_config.get(method, {}) or {})
            for trial_index in range(trials):
                trial_rng = np.random.default_rng(seed + pair_index * 1009 + method_index * 131 + trial_index * 17 + 1)
                corrupted_mask, used_params = apply_corruption_to_fixed_pair_input(
                    image=original_mask,
                    point_a=point_a,
                    point_b=point_b,
                    method=method,
                    params=params,
                    rng=trial_rng,
                )
                connected_after_corruption = _recompute_label_after_corruption_for_pair(
                    corrupted_mask=corrupted_mask,
                    point_a=point_a,
                    point_b=point_b,
                    region_value_i=region_value_i,
                    region_value_j=region_value_j,
                )
                corruption_summary = compute_corruption_summary(original_mask=original_mask, corrupted_mask=corrupted_mask)
                corruption_summary["point_a_preserved"] = int(_point_matches_region(corrupted_mask, point_a, region_value_i))
                corruption_summary["point_b_preserved"] = int(_point_matches_region(corrupted_mask, point_b, region_value_j))
                corruption_summary["line_changed_fraction"] = round(
                    float(
                        np.mean(
                            [
                                int(bool(original_mask[point]) != bool(corrupted_mask[point]))
                                for point in line_changed_reference
                                if _is_inside(original_mask.shape, point)
                            ]
                        )
                    )
                    if line_changed_reference
                    else 0.0,
                    6,
                )
                corruption_summary["_review_thresholds"] = review_thresholds

                corruption_reason, review_reasons, needs_human_review = build_corruption_explanation(
                    corruption_method=method,
                    corruption_params=used_params,
                    connected_original=connected_original,
                    connected_after_corruption=connected_after_corruption,
                    corruption_summary=corruption_summary,
                )

                trial_id = f"{method}_{trial_index:03d}"
                corrupted_image_path = corrupted_image_dir / f"{pair_row['pair_id']}__{trial_id}.png"
                _save_binary_mask(corrupted_image_path, corrupted_mask)

                record = {
                    "pair_id": str(pair_row["pair_id"]),
                    "image_id": str(pair_row.get("image_id", Path(mask_path).stem)),
                    "mask_path": str(mask_path),
                    "point_a_y": point_a[0],
                    "point_a_x": point_a[1],
                    "point_b_y": point_b[0],
                    "point_b_x": point_b[1],
                    "connected_original": connected_original,
                    "connected_after_corruption": int(connected_after_corruption),
                    "label_changed_by_corruption": int(connected_original != connected_after_corruption),
                    "label_source": label_source,
                    "corruption_method": method,
                    "corruption_trial_id": trial_id,
                    "corruption_params_json": json.dumps(used_params, ensure_ascii=False),
                    "corrupted_image_path": str(corrupted_image_path),
                    "euclidean_distance_original": _get_first_available(pair_row, ["euclidean_distance_original", "euclidean_distance"], default=""),
                    "geodesic_distance_original": _get_first_available(pair_row, ["geodesic_distance_original", "geodesic_distance"], default=""),
                    "geodesic_euclidean_ratio_original": _get_first_available(
                        pair_row,
                        ["geodesic_euclidean_ratio_original", "geodesic_euclidean_ratio"],
                        default="",
                    ),
                    "boundary_crossing_count_original": _get_first_available(
                        pair_row,
                        ["boundary_crossing_count_original", "boundary_crossing_count"],
                        default="",
                    ),
                    "distance_to_boundary_min_original": _get_first_available(
                        pair_row,
                        ["distance_to_boundary_min_original", "distance_to_boundary_min"],
                        default="",
                    ),
                    "foreground_ratio_original": corruption_summary["foreground_ratio_original"],
                    "foreground_ratio_corrupted": corruption_summary["foreground_ratio_corrupted"],
                    "changed_pixel_ratio": corruption_summary["changed_pixel_ratio"],
                    "component_count_original": corruption_summary["component_count_original"],
                    "component_count_corrupted": corruption_summary["component_count_corrupted"],
                    "point_a_preserved": corruption_summary["point_a_preserved"],
                    "point_b_preserved": corruption_summary["point_b_preserved"],
                    "line_changed_fraction": corruption_summary["line_changed_fraction"],
                    "needs_human_review": int(needs_human_review),
                    "review_reasons": ";".join(review_reasons) if review_reasons else "none",
                    "corruption_reason": corruption_reason,
                    "region_value_i": region_value_i,
                    "region_value_j": region_value_j,
                }
                corrupted_records.append(record)

    robustness_summary = compute_robustness_summary(corrupted_records)
    output_paths = _write_corruption_outputs(
        output_dir=output_dir,
        records=corrupted_records,
        robustness_summary=robustness_summary,
        metadata={
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_pairs_csv": str(input_pairs_csv),
            "input_image_dir": str(input_image_dir) if input_image_dir is not None else "",
            "output_dir": str(output_dir),
            "label_source": label_source,
            "pair_selection_strategy": pair_selection_strategy,
            "num_pairs": len(pair_rows),
            "num_corrupted_samples": len(corrupted_records),
            "methods": methods,
            "config": config,
        },
    )
    return {
        "records": corrupted_records,
        "robustness_summary": robustness_summary,
        "output_paths": output_paths,
    }


def _apply_random_rectangle_mask(
    mask: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrupted = mask.copy()
    num_masks = max(1, int(params.get("num_masks", 3)))
    min_size = max(1, int(params.get("min_size", 8)))
    max_size = max(min_size, int(params.get("max_size", 32)))
    rectangles = []
    height, width = corrupted.shape
    for _ in range(num_masks):
        rect_height = min(height, int(rng.integers(min_size, max_size + 1)))
        rect_width = min(width, int(rng.integers(min_size, max_size + 1)))
        row0 = int(rng.integers(0, max(1, height - rect_height + 1)))
        col0 = int(rng.integers(0, max(1, width - rect_width + 1)))
        row1 = min(height, row0 + rect_height)
        col1 = min(width, col0 + rect_width)
        corrupted[row0:row1, col0:col1] = False
        rectangles.append({"row0": row0, "col0": col0, "row1": row1, "col1": col1})
    return corrupted, {"method": "random_rectangle_mask", "rectangles": rectangles}


def _apply_random_circle_mask(
    mask: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrupted = mask.copy()
    num_masks = max(1, int(params.get("num_masks", 3)))
    min_radius = max(1, int(params.get("min_radius", 4)))
    max_radius = max(min_radius, int(params.get("max_radius", 16)))
    circles = []
    height, width = corrupted.shape
    row_grid, col_grid = np.ogrid[:height, :width]
    for _ in range(num_masks):
        radius = int(rng.integers(min_radius, max_radius + 1))
        center_row = int(rng.integers(0, height))
        center_col = int(rng.integers(0, width))
        disk = (row_grid - center_row) ** 2 + (col_grid - center_col) ** 2 <= radius**2
        corrupted[disk] = False
        circles.append({"center_row": center_row, "center_col": center_col, "radius": radius})
    return corrupted, {"method": "random_circle_mask", "circles": circles}


def _apply_line_occlusion(
    mask: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_lines = max(1, int(params.get("num_lines", 2)))
    width = max(1, int(params.get("width", 3)))
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    draw = ImageDraw.Draw(image)
    height, image_width = mask.shape
    lines = []
    for _ in range(num_lines):
        start = (int(rng.integers(0, image_width)), int(rng.integers(0, height)))
        end = (int(rng.integers(0, image_width)), int(rng.integers(0, height)))
        draw.line([start, end], fill=0, width=width)
        lines.append({"start_x": start[0], "start_y": start[1], "end_x": end[0], "end_y": end[1], "width": width})
    return np.asarray(image) >= 128, {"method": "line_occlusion", "lines": lines}


def _apply_point_neighborhood_mask(
    mask: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    corrupted = mask.copy()
    radius = max(1, int(params.get("radius", 8)))
    targets = params.get("target", ["point_a", "point_b"])
    if isinstance(targets, str):
        targets = [targets]
    centers = []
    for target in targets:
        if target == "point_a":
            centers.append(("point_a", point_a))
        elif target == "point_b":
            centers.append(("point_b", point_b))
    row_grid, col_grid = np.ogrid[:mask.shape[0], :mask.shape[1]]
    for label, center in centers:
        disk = (row_grid - center[0]) ** 2 + (col_grid - center[1]) ** 2 <= radius**2
        corrupted[disk] = False
    return corrupted, {"method": "point_neighborhood_mask", "radius": radius, "target": list(targets)}


def _apply_salt_and_pepper(
    mask: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrupted = mask.copy().reshape(-1)
    amount = max(0.0, float(params.get("amount", 0.01)))
    num_flips = min(len(corrupted), int(round(amount * len(corrupted))))
    if num_flips > 0:
        flip_indices = rng.choice(len(corrupted), size=num_flips, replace=False)
        corrupted[flip_indices] = rng.integers(0, 2, size=num_flips).astype(bool)
    return corrupted.reshape(mask.shape), {"method": "salt_and_pepper", "amount": amount, "num_flips": num_flips}


def _apply_gaussian_noise(
    mask: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = max(0.0, float(params.get("sigma", 0.05)))
    noisy = mask.astype(np.float32) + rng.normal(0.0, sigma, size=mask.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy >= 0.5, {"method": "gaussian_noise", "sigma": sigma}


def _apply_blur(mask: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = max(0.0, float(params.get("sigma", 1.0)))
    try:
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        return blurred >= 0.5, {"method": "blur", "sigma": sigma}
    except ImportError:
        image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(blurred) >= 128, {"method": "blur", "sigma": sigma}


def _apply_morphology(
    mask: np.ndarray,
    params: dict[str, Any],
    operation: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    radius = max(1, int(params.get("radius", 1)))
    structure = _disk_structure(radius)
    try:
        if operation == "erosion":
            from scipy.ndimage import binary_erosion

            return binary_erosion(mask, structure=structure), {"method": "erosion", "radius": radius}
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=structure), {"method": "dilation", "radius": radius}
    except ImportError:
        return _binary_morphology_fallback(mask, structure=structure, operation=operation), {"method": operation, "radius": radius}


def _disk_structure(radius: int) -> np.ndarray:
    row_grid, col_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((row_grid**2 + col_grid**2) <= radius**2).astype(bool)


def _binary_morphology_fallback(mask: np.ndarray, structure: np.ndarray, operation: str) -> np.ndarray:
    structure = structure.astype(bool)
    center_row = structure.shape[0] // 2
    center_col = structure.shape[1] // 2
    offsets = np.argwhere(structure) - np.array([center_row, center_col])
    padded = np.pad(mask.astype(bool), ((center_row, center_row), (center_col, center_col)), mode="constant")
    if operation == "dilation":
        result = np.zeros_like(mask, dtype=bool)
        for delta_row, delta_col in offsets:
            row_start = center_row + int(delta_row)
            col_start = center_col + int(delta_col)
            result |= padded[row_start : row_start + mask.shape[0], col_start : col_start + mask.shape[1]]
        return result
    result = np.ones_like(mask, dtype=bool)
    for delta_row, delta_col in offsets:
        row_start = center_row + int(delta_row)
        col_start = center_col + int(delta_col)
        result &= padded[row_start : row_start + mask.shape[0], col_start : col_start + mask.shape[1]]
    return result


def _point_matches_region(mask: np.ndarray, point: tuple[int, int], region_value: int) -> bool:
    if not _is_inside(mask.shape, point):
        return False
    return int(bool(mask[point])) == int(region_value)


def _recompute_label_after_corruption_for_pair(
    corrupted_mask: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    region_value_i: int,
    region_value_j: int,
) -> int:
    if int(region_value_i) != int(region_value_j):
        return 0
    domain_mask = corrupted_mask if int(region_value_i) == 1 else ~corrupted_mask
    return recompute_label_after_corruption(domain_mask, point_a=point_a, point_b=point_b)


def _write_corruption_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    robustness_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    csv_path = output_dir / "corrupted_pair_inputs.csv"
    jsonl_path = output_dir / "corrupted_pair_inputs.jsonl"
    explanation_path = output_dir / "corruption_explanation.txt"
    review_path = output_dir / "corruption_review.csv"
    summary_path = output_dir / "robustness_summary.csv"
    metadata_path = output_dir / "metadata.json"

    _write_csv_allow_empty(csv_path, records, fieldnames=_collect_fieldnames(records, preferred=CORRUPTION_FIELD_ORDER))
    write_jsonl(jsonl_path, [_to_json_record(record) for record in records])
    explanation_path.write_text(_build_corruption_explanation_text(records), encoding="utf-8")
    review_records = [record for record in records if int(record.get("needs_human_review", 0)) == 1]
    _write_csv_allow_empty(review_path, review_records, fieldnames=_collect_fieldnames(records, preferred=CORRUPTION_FIELD_ORDER))
    _write_csv_allow_empty(summary_path, robustness_summary, fieldnames=ROBUSTNESS_SUMMARY_FIELD_ORDER)
    write_json(metadata_path, metadata)
    return {
        "corrupted_csv": csv_path,
        "corrupted_jsonl": jsonl_path,
        "corruption_explanation": explanation_path,
        "corruption_review_csv": review_path,
        "robustness_summary_csv": summary_path,
        "metadata": metadata_path,
    }


def _build_corruption_explanation_text(records: list[dict[str, Any]]) -> str:
    blocks = []
    for record in records:
        blocks.append(
            "\n".join(
                [
                    "=" * 80,
                    f"pair_id: {record['pair_id']}",
                    f"image_id: {record['image_id']}",
                    f"label_source: {record['label_source']}",
                    f"connected_original: {record['connected_original']}",
                    f"connected_after_corruption: {record['connected_after_corruption']}",
                    f"label_changed_by_corruption: {record['label_changed_by_corruption']}",
                    "",
                    f"point_a: ({record['point_a_y']}, {record['point_a_x']})",
                    f"point_b: ({record['point_b_y']}, {record['point_b_x']})",
                    "",
                    "corruption:",
                    f"  method: {record['corruption_method']}",
                    f"  trial_id: {record['corruption_trial_id']}",
                    f"  params: {record['corruption_params_json']}",
                    "",
                    "corruption_summary:",
                    f"  foreground_ratio_original: {record['foreground_ratio_original']}",
                    f"  foreground_ratio_corrupted: {record['foreground_ratio_corrupted']}",
                    f"  changed_pixel_ratio: {record['changed_pixel_ratio']}",
                    f"  component_count_original: {record['component_count_original']}",
                    f"  component_count_corrupted: {record['component_count_corrupted']}",
                    "",
                    "corruption_reason:",
                    f"  {record['corruption_reason']}",
                    "",
                    f"needs_human_review: {record['needs_human_review']}",
                    f"review_reasons: {record['review_reasons']}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _resolve_pair_mask_path(
    record: dict[str, Any],
    input_image_dir: Path | None,
    repo_root: Path,
) -> Path:
    path_value = _get_first_available(record, ["mask_path", "image_path"], default="")
    if path_value:
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        candidate = repo_root / path
        if candidate.exists():
            return candidate
        if input_image_dir is not None:
            candidate = input_image_dir / path.name
            if candidate.exists():
                return candidate
    if input_image_dir is None:
        raise FileNotFoundError(f"Could not resolve mask path for pair_id={record.get('pair_id')}.")
    image_id = str(record.get("image_id", ""))
    for extension in DEFAULT_EXTENSIONS:
        candidate = input_image_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve mask path for pair_id={record.get('pair_id')}.")


def _select_input_pairs(
    pair_rows: list[dict[str, Any]],
    max_pairs: int | None,
    strategy: str,
) -> list[dict[str, Any]]:
    if max_pairs is None or max_pairs <= 0 or len(pair_rows) <= max_pairs:
        return list(pair_rows)
    strategy = str(strategy).strip().lower()
    if strategy == "head":
        return list(pair_rows[:max_pairs])
    if strategy == "round_robin_image":
        return _round_robin_pair_rows(pair_rows, max_pairs=max_pairs, key_fields=["image_id"])
    if strategy == "round_robin_image_method":
        return _round_robin_pair_rows(pair_rows, max_pairs=max_pairs, key_fields=["image_id", "sampling_method"])
    if strategy == "balanced_label_round_robin_image_method":
        positives = [row for row in pair_rows if int(float(row.get("connected_original") or row.get("connected") or 0)) == 1]
        negatives = [row for row in pair_rows if int(float(row.get("connected_original") or row.get("connected") or 0)) == 0]
        positive_target = min(len(positives), max_pairs // 2)
        negative_target = min(len(negatives), max_pairs // 2)
        while positive_target + negative_target < max_pairs:
            if len(positives) - positive_target > len(negatives) - negative_target and positive_target < len(positives):
                positive_target += 1
                continue
            if negative_target < len(negatives):
                negative_target += 1
                continue
            if positive_target < len(positives):
                positive_target += 1
                continue
            break
        selected_positive = _round_robin_pair_rows(
            positives,
            max_pairs=positive_target,
            key_fields=["image_id", "sampling_method"],
        )
        selected_negative = _round_robin_pair_rows(
            negatives,
            max_pairs=negative_target,
            key_fields=["image_id", "sampling_method"],
        )
        return _interleave_records(selected_positive, selected_negative)[:max_pairs]
    raise ValueError(f"Unknown pair selection strategy: {strategy}")


def _round_robin_pair_rows(
    pair_rows: list[dict[str, Any]],
    max_pairs: int,
    key_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    group_order: list[tuple[str, ...]] = []
    for row in pair_rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_pairs:
        advanced = False
        for key in group_order:
            values = grouped[key]
            if not values:
                continue
            selected.append(values.pop(0))
            advanced = True
            if len(selected) >= max_pairs:
                break
        if not advanced:
            break
    return selected


def _interleave_records(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    max_length = max(len(primary), len(secondary))
    for index in range(max_length):
        if index < len(primary):
            merged.append(primary[index])
        if index < len(secondary):
            merged.append(secondary[index])
    return merged


def _extract_point(record: dict[str, Any], label: str) -> tuple[int, int]:
    if label == "a":
        row_value = _get_first_available(record, ["point_a_y", "row_i"], default=0)
        col_value = _get_first_available(record, ["point_a_x", "col_i"], default=0)
    else:
        row_value = _get_first_available(record, ["point_b_y", "row_j"], default=0)
        col_value = _get_first_available(record, ["point_b_x", "col_j"], default=0)
    return int(float(row_value)), int(float(col_value))


def _get_first_available(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in ("", None):
            return record[key]
    return default


def _write_csv_allow_empty(path: str | Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _collect_fieldnames(records: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = set(preferred)
    fieldnames = list(preferred)
    for record in records:
        for key in record.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def _save_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _is_inside(shape: tuple[int, int], point: tuple[int, int]) -> bool:
    return 0 <= int(point[0]) < int(shape[0]) and 0 <= int(point[1]) < int(shape[1])


def _to_json_record(record: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in record.items():
        if key == "corruption_params_json":
            try:
                converted[key] = json.loads(str(value))
            except json.JSONDecodeError:
                converted[key] = str(value)
            continue
        if isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted


def _mean(records: list[dict[str, Any]], key: str) -> Any:
    values = [float(record[key]) for record in records if record.get(key) not in ("", None)]
    if not values:
        return ""
    return round(float(np.mean(values)), 6)


def _rate(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0)) for record in records]
    return round(float(np.mean(values)), 6) if values else 0.0


def _optional_accuracy(records: list[dict[str, Any]], pred_key: str, label_key: str) -> Any:
    pairs = [
        (int(float(record[pred_key])), int(float(record[label_key])))
        for record in records
        if record.get(pred_key) not in ("", None) and record.get(label_key) not in ("", None)
    ]
    if not pairs:
        return ""
    return round(float(np.mean([int(pred == label) for pred, label in pairs])), 6)


def _optional_difference(value_a: Any, value_b: Any) -> Any:
    if value_a in ("", None) or value_b in ("", None):
        return ""
    return round(float(value_a) - float(value_b), 6)


def _optional_mean_difference(records: list[dict[str, Any]], high_key: str, low_key: str) -> Any:
    values = []
    for record in records:
        high = record.get(high_key)
        low = record.get(low_key)
        if high in ("", None) or low in ("", None):
            continue
        values.append(float(high) - float(low))
    if not values:
        return ""
    return round(float(np.mean(values)), 6)


def _optional_prediction_flip_rate(records: list[dict[str, Any]]) -> Any:
    flips = []
    for record in records:
        pred_original = record.get("pred_original")
        pred_corrupted = record.get("pred_corrupted")
        if pred_original in ("", None) or pred_corrupted in ("", None):
            continue
        flips.append(int(float(pred_original) != float(pred_corrupted)))
    if not flips:
        return ""
    return round(float(np.mean(flips)), 6)


def _corruption_method_sentence(method: str) -> str:
    method = _normalize_corruption_method(method)
    return {
        "none": "This sample keeps the original mask unchanged and acts as a control condition.",
        "random_rectangle_mask": "This sample applies random rectangular masking to create local occlusions.",
        "random_circle_mask": "This sample applies random circular masking to create blob-like occlusions.",
        "line_occlusion": "This sample draws dark lines over the mask to mimic path-disrupting occlusion.",
        "point_neighborhood_mask": "This sample hides a neighborhood around one or both query points.",
        "salt_and_pepper": "This sample flips sparse pixels to test sensitivity to impulsive noise.",
        "gaussian_noise": "This sample adds Gaussian noise before thresholding back to a binary mask.",
        "blur": "This sample blurs the mask before thresholding, which softens narrow structures.",
        "erosion": "This sample erodes the mask, shrinking thin structures and bottlenecks.",
        "dilation": "This sample dilates the mask, thickening structures and potentially merging components.",
    }.get(method, f"This sample applies the corruption method '{method}'.")


def _normalize_corruption_method(method: Any) -> str:
    if method is None:
        return "none"
    return str(method).strip().lower()
