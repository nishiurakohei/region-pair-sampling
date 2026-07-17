from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .io_utils import ensure_dir, write_json, write_jsonl
from .region_pair_sampling import (
    attach_region_pair_difficulty,
    build_pair_records,
    build_valid_mask,
    compute_distance_transform,
    connected_components,
    load_binary_mask as sampling_load_binary_mask,
    sample_points,
    select_pair_records,
)

DEFAULT_EXPLANATION_THRESHOLDS = {
    "geodesic_long_threshold": 150.0,
    "geodesic_very_long_threshold": 300.0,
    "ratio_high_threshold": 2.0,
    "ratio_very_high_threshold": 4.0,
    "boundary_crossing_multiple_threshold": 3,
    "boundary_crossing_many_threshold": 6,
    "near_boundary_threshold": 2.0,
    "close_disconnected_euclidean_threshold": 20.0,
}

DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DEFAULT_VISUALIZATION_CONFIG = {
    "enabled": True,
    "point_radius": 3,
    "line_width": 2,
}

CSV_FIELD_ORDER = [
    "pair_id",
    "image_id",
    "mask_path",
    "image_path",
    "sampling_method",
    "valid_threshold",
    "point_a_y",
    "point_a_x",
    "point_b_y",
    "point_b_x",
    "row_i",
    "col_i",
    "row_j",
    "col_j",
    "connected",
    "euclidean_distance",
    "geodesic_distance",
    "geodesic_euclidean_ratio",
    "boundary_crossing_count",
    "distance_to_boundary_min",
    "difficulty_bucket",
    "reason_codes",
    "rule_score",
    "why_selected",
    "needs_human_review",
    "review_reasons",
]


@dataclass
class PairRecord:
    pair_id: str
    image_id: str
    mask_path: str
    sampling_method: str
    point_a_y: int
    point_a_x: int
    point_b_y: int
    point_b_x: int
    connected: int
    euclidean_distance: float
    geodesic_distance: float | None
    geodesic_euclidean_ratio: float | None
    boundary_crossing_count: int
    distance_to_boundary_min: float
    difficulty_bucket: str
    reason_codes: list[str]
    rule_score: float
    why_selected: str
    needs_human_review: int
    review_reasons: list[str]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_csv_record(self) -> dict[str, Any]:
        record = {
            "pair_id": self.pair_id,
            "image_id": self.image_id,
            "mask_path": self.mask_path,
            "image_path": self.extras.get("image_path", self.mask_path),
            "sampling_method": self.sampling_method,
            "valid_threshold": _maybe_round(self.extras.get("valid_threshold")),
            "point_a_y": self.point_a_y,
            "point_a_x": self.point_a_x,
            "point_b_y": self.point_b_y,
            "point_b_x": self.point_b_x,
            "row_i": self.extras.get("row_i", self.point_a_y),
            "col_i": self.extras.get("col_i", self.point_a_x),
            "row_j": self.extras.get("row_j", self.point_b_y),
            "col_j": self.extras.get("col_j", self.point_b_x),
            "connected": int(self.connected),
            "euclidean_distance": _maybe_round(self.euclidean_distance),
            "geodesic_distance": _maybe_round(self.geodesic_distance),
            "geodesic_euclidean_ratio": _maybe_round(self.geodesic_euclidean_ratio),
            "boundary_crossing_count": int(self.boundary_crossing_count),
            "distance_to_boundary_min": _maybe_round(self.distance_to_boundary_min),
            "difficulty_bucket": self.difficulty_bucket,
            "reason_codes": ";".join(self.reason_codes) if self.reason_codes else "NONE",
            "rule_score": _maybe_round(self.rule_score),
            "why_selected": self.why_selected,
            "needs_human_review": int(self.needs_human_review),
            "review_reasons": ";".join(self.review_reasons) if self.review_reasons else "none",
        }
        for key, value in self.extras.items():
            if key in record:
                continue
            record[key] = _sanitize_scalar(value, for_json=False)
        return record

    def to_json_record(self) -> dict[str, Any]:
        record = {
            "pair_id": self.pair_id,
            "image_id": self.image_id,
            "mask_path": self.mask_path,
            "sampling_method": self.sampling_method,
            "point_a_y": self.point_a_y,
            "point_a_x": self.point_a_x,
            "point_b_y": self.point_b_y,
            "point_b_x": self.point_b_x,
            "connected": int(self.connected),
            "euclidean_distance": _sanitize_scalar(self.euclidean_distance, for_json=True),
            "geodesic_distance": _sanitize_scalar(self.geodesic_distance, for_json=True),
            "geodesic_euclidean_ratio": _sanitize_scalar(self.geodesic_euclidean_ratio, for_json=True),
            "boundary_crossing_count": int(self.boundary_crossing_count),
            "distance_to_boundary_min": _sanitize_scalar(self.distance_to_boundary_min, for_json=True),
            "difficulty_bucket": self.difficulty_bucket,
            "reason_codes": list(self.reason_codes),
            "rule_score": _sanitize_scalar(self.rule_score, for_json=True),
            "why_selected": self.why_selected,
            "needs_human_review": int(self.needs_human_review),
            "review_reasons": list(self.review_reasons),
        }
        for key, value in self.extras.items():
            if key in record:
                continue
            record[key] = _sanitize_scalar(value, for_json=True)
        return record


def discover_input_mask_paths(
    input_dir: str | Path,
    recursive: bool = True,
    extensions: list[str] | None = None,
) -> list[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.glob("*")
    allowed = {suffix.lower() for suffix in (extensions or DEFAULT_EXTENSIONS)}
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in allowed]
    return sorted(paths)


def filter_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if int(min_component_size) <= 1:
        return mask.astype(bool)
    labels, component_sizes = connected_components(mask.astype(bool))
    keep_ids = {index for index, size in enumerate(component_sizes) if int(size) >= int(min_component_size)}
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, list(keep_ids))


def load_binary_mask(
    image_path: str | Path,
    mask_channel: int | str = "auto",
    foreground_threshold: int = 16,
    invert_mask: bool = False,
    min_component_size: int = 1,
    binarization_mode: str = "auto",
    foreground_values: list[int] | None = None,
    use_alpha_channel: bool = False,
    auto_invert_if_foreground_dominates: bool = False,
) -> np.ndarray:
    image = Image.open(image_path)
    values = _extract_mask_values(
        image=np.asarray(image),
        mask_channel=mask_channel,
        use_alpha_channel=use_alpha_channel,
    )
    mask = _binarize_values(
        values=values,
        binarization_mode=binarization_mode,
        foreground_threshold=foreground_threshold,
        foreground_values=foreground_values,
    )
    if auto_invert_if_foreground_dominates and float(mask.mean()) > 0.5:
        mask = ~mask
    if invert_mask:
        mask = ~mask
    return filter_small_components(mask.astype(bool), min_component_size=min_component_size)


def score_pair_for_method(record: dict[str, Any], method: str) -> float:
    euclidean_distance = _to_float(record.get("euclidean_distance")) or 0.0
    geodesic_distance = _to_float(record.get("geodesic_distance"))
    ratio = _to_float(record.get("geodesic_euclidean_ratio")) or 0.0
    boundary_crossings = _to_float(record.get("boundary_crossing_count")) or 0.0
    distance_to_boundary_min = _to_float(record.get("distance_to_boundary_min")) or 0.0
    hard_pair_score = _to_float(record.get("hard_pair_score")) or 0.0
    hard_pair_score_v2 = _to_float(record.get("hard_pair_score_v2")) or 0.0
    difficulty_proxy = _to_float(record.get("difficulty_proxy")) or 0.0
    line_inside_fraction = _to_float(record.get("line_inside_fraction")) or 0.0
    component_gap_distance = _to_float(record.get("component_gap_distance")) or 0.0
    connected = int(record.get("connected", 0))

    if method in {"euclidean_fps", "euclidean_farthest"}:
        return euclidean_distance
    if method in {"geodesic_fps", "geodesic_farthest"}:
        return geodesic_distance if geodesic_distance is not None else euclidean_distance
    if method in {"boundary_crossing_targeted", "boundary_crossing_stratified"}:
        return boundary_crossings * 100.0 + euclidean_distance
    if method == "feature_space_diverse":
        return difficulty_proxy + ratio * 10.0 + boundary_crossings - distance_to_boundary_min * 0.5
    if method == "hard_pair_targeted":
        return hard_pair_score
    if method == "hard_pair_targeted_v2":
        return hard_pair_score_v2
    if method == "random":
        return 0.0
    if connected == 1:
        return (geodesic_distance or 0.0) + ratio * 10.0 + boundary_crossings
    return boundary_crossings * 10.0 + line_inside_fraction * 10.0 - component_gap_distance


def select_pair(
    records: list[dict[str, Any]],
    method: str,
    target_count: int,
    positive_ratio: float,
    rng: np.random.Generator,
    feature_bin_count: int = 4,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []

    positives = [record for record in records if int(record.get("connected", 0)) == 1]
    negatives = [record for record in records if int(record.get("connected", 0)) == 0]
    maximum = min(int(target_count), len(records))
    if not positives or not negatives:
        chosen = select_pair_records(
            records,
            method=method,
            target_count=maximum,
            rng=rng,
            feature_bin_count=feature_bin_count,
        )
        return _attach_selection_scores(chosen, method=method)

    positive_target = min(len(positives), int(round(maximum * float(positive_ratio))))
    negative_target = min(len(negatives), maximum - positive_target)
    while positive_target + negative_target < maximum:
        positive_remaining = len(positives) - positive_target
        negative_remaining = len(negatives) - negative_target
        if positive_remaining <= 0 and negative_remaining <= 0:
            break
        current_total = max(1, positive_target + negative_target)
        current_ratio = positive_target / current_total
        if current_ratio < float(positive_ratio) and positive_remaining > 0:
            positive_target += 1
            continue
        if negative_remaining > 0:
            negative_target += 1
            continue
        if positive_remaining > 0:
            positive_target += 1

    selected: list[dict[str, Any]] = []
    if positive_target > 0:
        positive_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        selected.extend(
            select_pair_records(
                positives,
                method=method,
                target_count=positive_target,
                rng=positive_rng,
                feature_bin_count=feature_bin_count,
            )
        )
    if negative_target > 0:
        negative_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        selected.extend(
            select_pair_records(
                negatives,
                method=method,
                target_count=negative_target,
                rng=negative_rng,
                feature_bin_count=feature_bin_count,
            )
        )

    if len(selected) < maximum:
        used_ids = {record["pair_id"] for record in selected}
        leftovers = [record for record in records if record["pair_id"] not in used_ids]
        if leftovers:
            filler_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            selected.extend(
                select_pair_records(
                    leftovers,
                    method=method,
                    target_count=maximum - len(selected),
                    rng=filler_rng,
                    feature_bin_count=feature_bin_count,
                )
            )
    selected = _deduplicate_by_pair_id(selected)
    return _attach_selection_scores(selected[:maximum], method=method)


def explain_pair_selection(
    record: dict[str, Any],
    explanation_config: dict[str, Any],
) -> tuple[list[str], str, list[str], int]:
    thresholds = dict(DEFAULT_EXPLANATION_THRESHOLDS)
    thresholds.update(explanation_config or {})

    connected = int(record.get("connected", 0))
    method = str(record.get("sampling_method", "unknown"))
    euclidean_distance = _to_float(record.get("euclidean_distance")) or 0.0
    geodesic_distance = _to_float(record.get("geodesic_distance"))
    ratio = _to_float(record.get("geodesic_euclidean_ratio"))
    boundary_crossings = int(_to_float(record.get("boundary_crossing_count")) or 0.0)
    distance_to_boundary_min = _to_float(record.get("distance_to_boundary_min")) or 0.0
    line_inside_fraction = _to_float(record.get("line_inside_fraction"))
    pair_region_type = str(record.get("pair_region_type", "foreground_foreground"))

    reason_codes = [_method_reason_code(method), "CONNECTED_PAIR" if connected == 1 else "DISCONNECTED_PAIR"]
    if geodesic_distance is not None:
        if geodesic_distance >= float(thresholds["geodesic_very_long_threshold"]):
            reason_codes.append("VERY_LONG_GEODESIC_PATH")
        elif geodesic_distance >= float(thresholds["geodesic_long_threshold"]):
            reason_codes.append("LONG_GEODESIC_PATH")
    if ratio is not None:
        if ratio >= float(thresholds["ratio_very_high_threshold"]):
            reason_codes.append("VERY_HIGH_GEODESIC_EUCLIDEAN_RATIO")
        elif ratio >= float(thresholds["ratio_high_threshold"]):
            reason_codes.append("HIGH_GEODESIC_EUCLIDEAN_RATIO")
        elif connected == 1 and ratio <= 1.1:
            reason_codes.append("LOW_GEODESIC_EUCLIDEAN_RATIO")
    if boundary_crossings >= int(thresholds["boundary_crossing_many_threshold"]):
        reason_codes.append("MANY_BOUNDARY_CROSSINGS")
    elif boundary_crossings >= int(thresholds["boundary_crossing_multiple_threshold"]):
        reason_codes.append("MULTIPLE_BOUNDARY_CROSSINGS")
    elif boundary_crossings == 0:
        reason_codes.append("NO_BOUNDARY_CROSSING")
    if distance_to_boundary_min <= float(thresholds["near_boundary_threshold"]):
        reason_codes.append("NEAR_BOUNDARY")
    if connected == 0 and euclidean_distance <= float(thresholds["close_disconnected_euclidean_threshold"]):
        reason_codes.append("CLOSE_DISCONNECTED_PAIR")
    if int(record.get("is_hard_positive_v2", 0)) == 1:
        reason_codes.append("HARD_POSITIVE_RULE_V2")
    if int(record.get("is_hard_negative_v2", 0)) == 1:
        reason_codes.append("HARD_NEGATIVE_RULE_V2")
    if int(record.get("easy_positive_v2", 0)) == 1:
        reason_codes.append("EASY_POSITIVE_RULE_V2")
    if int(record.get("easy_negative_v2", 0)) == 1:
        reason_codes.append("EASY_NEGATIVE_RULE_V2")

    review_reasons: list[str] = []
    if connected == 1 and geodesic_distance is None:
        review_reasons.append("GEODESIC_COMPUTATION_FAILED")
    if boundary_crossings >= int(thresholds["boundary_crossing_many_threshold"]):
        review_reasons.append("VERY_HIGH_BOUNDARY_CROSSING_COUNT")
    if distance_to_boundary_min <= max(0.5, float(thresholds["near_boundary_threshold"]) * 0.5):
        review_reasons.append("POINT_EXTREMELY_CLOSE_TO_BOUNDARY")
    if connected == 1 and line_inside_fraction is not None and line_inside_fraction < 0.25:
        review_reasons.append("CONNECTED_PAIR_WITH_LOW_LINE_INSIDE_FRACTION")
    if pair_region_type == "cross_region":
        review_reasons.append("CROSS_REGION_PAIR")

    why_selected = build_why_selected(record=record, reason_codes=reason_codes)
    return reason_codes, why_selected, review_reasons, int(bool(review_reasons))


def build_why_selected(record: dict[str, Any], reason_codes: list[str]) -> str:
    method = str(record.get("sampling_method", "unknown"))
    connected = int(record.get("connected", 0))
    euclidean_distance = _to_float(record.get("euclidean_distance"))
    geodesic_distance = _to_float(record.get("geodesic_distance"))
    ratio = _to_float(record.get("geodesic_euclidean_ratio"))
    boundary_crossings = int(_to_float(record.get("boundary_crossing_count")) or 0.0)
    distance_to_boundary_min = _to_float(record.get("distance_to_boundary_min"))

    sentences = [_method_sentence(method)]
    sentences.append(
        "The pair is labeled connected because both points fall in the same connected component."
        if connected == 1
        else "The pair is labeled disconnected because the two points fall in different connected components."
    )
    if "VERY_LONG_GEODESIC_PATH" in reason_codes or "LONG_GEODESIC_PATH" in reason_codes:
        sentences.append(
            f"The in-mask path is long (geodesic={_fmt_float(geodesic_distance)}), which can force the model to follow an extended route."
        )
    if "VERY_HIGH_GEODESIC_EUCLIDEAN_RATIO" in reason_codes or "HIGH_GEODESIC_EUCLIDEAN_RATIO" in reason_codes:
        sentences.append(
            f"The path is indirect relative to the straight-line distance (ratio={_fmt_float(ratio)})."
        )
    if "MULTIPLE_BOUNDARY_CROSSINGS" in reason_codes or "MANY_BOUNDARY_CROSSINGS" in reason_codes:
        sentences.append(
            f"The straight segment crosses the boundary {boundary_crossings} times, so local shape concavity matters."
        )
    if "CLOSE_DISCONNECTED_PAIR" in reason_codes:
        sentences.append(
            f"The points are spatially close (euclidean={_fmt_float(euclidean_distance)}) despite being disconnected."
        )
    if "NEAR_BOUNDARY" in reason_codes:
        sentences.append(
            f"At least one point is near the boundary (min distance={_fmt_float(distance_to_boundary_min)}), which makes the sample sensitive to local masking."
        )
    if "LOW_GEODESIC_EUCLIDEAN_RATIO" in reason_codes and connected == 1:
        sentences.append("The geodesic and Euclidean distances are close, so this pair should be comparatively straightforward.")
    return " ".join(sentence.strip() for sentence in sentences if sentence).strip()


def make_pair_record(
    record: dict[str, Any],
    mask_path: str | Path,
    explanation_config: dict[str, Any],
) -> PairRecord:
    reason_codes, why_selected, review_reasons, needs_human_review = explain_pair_selection(
        record=record,
        explanation_config=explanation_config,
    )
    mapped_keys = {
        "pair_id",
        "image_id",
        "sampling_method",
        "row_i",
        "col_i",
        "row_j",
        "col_j",
        "connected",
        "euclidean_distance",
        "geodesic_distance",
        "geodesic_euclidean_ratio",
        "boundary_crossing_count",
        "distance_to_boundary_min",
        "difficulty_bucket",
        "selection_score",
    }
    extras = {key: value for key, value in record.items() if key not in mapped_keys}
    extras["image_path"] = str(mask_path)
    extras["row_i"] = int(record["row_i"])
    extras["col_i"] = int(record["col_i"])
    extras["row_j"] = int(record["row_j"])
    extras["col_j"] = int(record["col_j"])
    return PairRecord(
        pair_id=str(record["pair_id"]),
        image_id=str(record["image_id"]),
        mask_path=str(mask_path),
        sampling_method=str(record["sampling_method"]),
        point_a_y=int(record["row_i"]),
        point_a_x=int(record["col_i"]),
        point_b_y=int(record["row_j"]),
        point_b_x=int(record["col_j"]),
        connected=int(record["connected"]),
        euclidean_distance=float(_to_float(record.get("euclidean_distance")) or 0.0),
        geodesic_distance=_to_float(record.get("geodesic_distance")),
        geodesic_euclidean_ratio=_to_float(record.get("geodesic_euclidean_ratio")),
        boundary_crossing_count=int(_to_float(record.get("boundary_crossing_count")) or 0.0),
        distance_to_boundary_min=float(_to_float(record.get("distance_to_boundary_min")) or 0.0),
        difficulty_bucket=str(record.get("difficulty_bucket", "Unknown")),
        reason_codes=reason_codes,
        rule_score=float(_to_float(record.get("selection_score")) or score_pair_for_method(record, str(record["sampling_method"]))),
        why_selected=why_selected,
        needs_human_review=needs_human_review,
        review_reasons=review_reasons,
        extras=extras,
    )


def build_explanation_text(records: list[PairRecord]) -> str:
    blocks = []
    for record in records:
        block = [
            "=" * 80,
            f"pair_id: {record.pair_id}",
            f"image_id: {record.image_id}",
            f"mask_path: {record.mask_path}",
            f"sampling_method: {record.sampling_method}",
            f"connected: {record.connected}",
            "",
            f"point_a: ({record.point_a_y}, {record.point_a_x})",
            f"point_b: ({record.point_b_y}, {record.point_b_x})",
            "",
            "features:",
            f"  euclidean_distance: {_fmt_float(record.euclidean_distance)}",
            f"  geodesic_distance: {_fmt_float(record.geodesic_distance)}",
            f"  geodesic_euclidean_ratio: {_fmt_float(record.geodesic_euclidean_ratio)}",
            f"  boundary_crossing_count: {record.boundary_crossing_count}",
            f"  distance_to_boundary_min: {_fmt_float(record.distance_to_boundary_min)}",
            f"  difficulty_bucket: {record.difficulty_bucket}",
            "",
            f"reason_codes: {';'.join(record.reason_codes) if record.reason_codes else 'NONE'}",
            f"rule_score: {_fmt_float(record.rule_score)}",
            "why_selected:",
            f"  {record.why_selected}",
            "",
            f"needs_human_review: {record.needs_human_review}",
            f"review_reasons: {';'.join(record.review_reasons) if record.review_reasons else 'none'}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def build_image_explanation_text(
    image_path: str | Path,
    records: list[PairRecord],
) -> str:
    connected_count = sum(int(record.connected) == 1 for record in records)
    disconnected_count = sum(int(record.connected) == 0 for record in records)
    header = [
        "=" * 80,
        f"image_path: {image_path}",
        f"num_pairs: {len(records)}",
        f"connected_pairs: {connected_count}",
        f"disconnected_pairs: {disconnected_count}",
        "",
    ]
    return "\n".join(header) + build_explanation_text(records)


def write_review_file(path: str | Path, records: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    fieldnames = _collect_fieldnames(records, preferred=field_order or CSV_FIELD_ORDER)
    _write_csv_allow_empty(path, records, fieldnames=fieldnames)


def write_outputs(
    output_dir: str | Path,
    records: list[PairRecord],
    metadata: dict[str, Any],
    input_dir: str | Path,
    visualization_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    csv_records = [record.to_csv_record() for record in records]
    json_records = [record.to_json_record() for record in records]
    review_records = [record for record in csv_records if int(record.get("needs_human_review", 0)) == 1]
    explanation_text = build_explanation_text(records)
    per_image_text_dir = ensure_dir(output_dir / "per_image_text")
    visualization_dir = ensure_dir(output_dir / "visualizations")

    pairs_csv_path = output_dir / "pairs.csv"
    pairs_jsonl_path = output_dir / "pairs.jsonl"
    explanation_path = output_dir / "pairs_explanation.txt"
    review_csv_path = output_dir / "pairs_for_human_review.csv"
    metadata_path = output_dir / "metadata.json"

    _write_csv_allow_empty(pairs_csv_path, csv_records, fieldnames=_collect_fieldnames(csv_records, preferred=CSV_FIELD_ORDER))
    write_jsonl(pairs_jsonl_path, json_records)
    explanation_path.write_text(explanation_text, encoding="utf-8")
    write_review_file(review_csv_path, review_records, field_order=CSV_FIELD_ORDER)
    write_per_image_texts(
        output_dir=per_image_text_dir,
        input_dir=input_dir,
        records=records,
    )
    save_pair_visualizations(
        output_dir=visualization_dir,
        input_dir=input_dir,
        records=records,
        visualization_config=visualization_config or {},
    )
    write_json(metadata_path, metadata)
    return {
        "pairs_csv": pairs_csv_path,
        "pairs_jsonl": pairs_jsonl_path,
        "pairs_explanation": explanation_path,
        "pairs_review_csv": review_csv_path,
        "per_image_text_dir": per_image_text_dir,
        "visualization_dir": visualization_dir,
        "metadata": metadata_path,
    }


def generate_region_pairs_from_config(
    config: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    input_dir = _resolve_path(config.get("input_dir", "data/masks"), repo_root=repo_root)
    output_dir = _resolve_path(config.get("output_dir", "outputs/generated_pairs"), repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(value) for value in config.get("methods", ["random", "euclidean_farthest", "geodesic_farthest"])]
    if not methods:
        raise ValueError("At least one sampling method must be provided.")
    valid_thresholds = config.get("valid_thresholds")
    if valid_thresholds is None:
        valid_thresholds = [config.get("valid_threshold", 2.0)]
    valid_thresholds = [float(value) for value in valid_thresholds]
    num_pairs_per_image = int(config.get("num_pairs_per_image", 30))
    seed = int(config.get("seed", 0))
    positive_ratio = float(config.get("positive_ratio", 0.5))
    min_component_size = int(config.get("min_component_size", 20))
    candidate_pool_size = int(config.get("candidate_pool_size", 300))
    mask_channel = config.get("mask_channel", "auto")
    foreground_threshold = int(config.get("foreground_threshold", 16))
    invert_mask = bool(config.get("invert_mask", False))
    binarization_mode = str(config.get("binarization_mode", "auto"))
    foreground_values = config.get("foreground_values")
    foreground_values = [int(value) for value in foreground_values] if foreground_values else None
    use_alpha_channel = bool(config.get("use_alpha_channel", False))
    auto_invert_if_foreground_dominates = bool(config.get("auto_invert_if_foreground_dominates", False))
    recursive = bool(config.get("recursive", True))
    extensions = [str(value) for value in config.get("input_extensions", DEFAULT_EXTENSIONS)]
    max_images = config.get("max_images")
    max_images = int(max_images) if max_images not in (None, "", 0) else None
    feature_bin_count = int(config.get("feature_bin_count", 4))
    sample_foreground = bool(config.get("sample_foreground", True))
    sample_background = bool(config.get("sample_background", False))
    pair_component_mode = str(config.get("pair_component_mode", "same_region"))
    strict_num_pairs_per_image = bool(config.get("strict_num_pairs_per_image", True))
    points_per_method = config.get("points_per_method")
    explanation_config = dict(config.get("explanation", {}))
    visualization_config = dict(DEFAULT_VISUALIZATION_CONFIG)
    visualization_config.update(config.get("visualization", {}))

    mask_paths = discover_input_mask_paths(input_dir=input_dir, recursive=recursive, extensions=extensions)
    if max_images is not None:
        mask_paths = mask_paths[:max_images]
    if not mask_paths:
        raise FileNotFoundError(f"No mask files were found under {input_dir}.")

    pair_records: list[PairRecord] = []
    generation_summary: list[dict[str, Any]] = []
    skipped_images: list[dict[str, Any]] = []
    combination_targets = _combination_targets(num_pairs_per_image, methods, valid_thresholds)

    for image_index, mask_path in enumerate(mask_paths):
        image_id = _image_id_from_path(index=image_index, path=mask_path, input_dir=input_dir)
        image_selected_records: list[dict[str, Any]] = []
        image_candidate_records: list[dict[str, Any]] = []
        mask = load_binary_mask(
            image_path=mask_path,
            mask_channel=mask_channel,
            foreground_threshold=foreground_threshold,
            invert_mask=invert_mask,
            min_component_size=min_component_size,
            binarization_mode=binarization_mode,
            foreground_values=foreground_values,
            use_alpha_channel=use_alpha_channel,
            auto_invert_if_foreground_dominates=auto_invert_if_foreground_dominates,
        )
        if int(mask.sum()) == 0:
            skipped_images.append({"image_id": image_id, "mask_path": str(mask_path), "reason": "empty_after_filtering"})
            continue

        distance_map = compute_distance_transform(mask)
        background_mask = ~mask
        background_distance_map = compute_distance_transform(background_mask)

        for threshold_index, valid_threshold in enumerate(valid_thresholds):
            foreground_valid_mask = build_valid_mask(mask, distance_map, min_distance_to_boundary=valid_threshold)
            background_valid_mask = build_valid_mask(background_mask, background_distance_map, min_distance_to_boundary=valid_threshold)
            if sample_foreground and int(foreground_valid_mask.sum()) == 0 and (not sample_background or int(background_valid_mask.sum()) == 0):
                continue

            for method_index, method in enumerate(methods):
                target_count = combination_targets[(method, float(valid_threshold))]
                if target_count <= 0:
                    continue
                n_points = int(points_per_method) if points_per_method not in (None, "", 0) else _default_points_per_method(target_count)
                points: list[dict[str, Any]] = []
                if sample_foreground and int(foreground_valid_mask.sum()) > 0:
                    point_rng = np.random.default_rng(seed + image_index * 1009 + threshold_index * 131 + method_index * 17 + 1)
                    points.extend(
                        _sample_region_points(
                            valid_mask=foreground_valid_mask,
                            distance_map=distance_map,
                            method=method,
                            n_points=n_points,
                            candidate_pool_size=candidate_pool_size,
                            rng=point_rng,
                            region_value=1,
                            region_name="foreground",
                        )
                    )
                if sample_background and int(background_valid_mask.sum()) > 0:
                    point_rng = np.random.default_rng(seed + image_index * 1009 + threshold_index * 131 + method_index * 17 + 10001)
                    points.extend(
                        _sample_region_points(
                            valid_mask=background_valid_mask,
                            distance_map=background_distance_map,
                            method=method,
                            n_points=n_points,
                            candidate_pool_size=candidate_pool_size,
                            rng=point_rng,
                            region_value=0,
                            region_name="background",
                        )
                    )
                if len(points) < 2:
                    generation_summary.append(
                        {
                            "image_id": image_id,
                            "mask_path": str(mask_path),
                            "valid_threshold": float(valid_threshold),
                            "sampling_method": method,
                            "candidate_pairs": 0,
                            "selected_pairs": 0,
                            "positive_available": 0,
                            "negative_available": 0,
                            "positive_selected": 0,
                            "negative_selected": 0,
                        }
                    )
                    continue

                candidate_records = build_pair_records(
                    mask=mask,
                    distance_map=distance_map,
                    background_distance_map=background_distance_map,
                    points=points,
                    image_id=image_id,
                    image_path=mask_path,
                    valid_threshold=valid_threshold,
                    sampling_method=method,
                    pair_component_mode=pair_component_mode,
                )
                if not candidate_records:
                    generation_summary.append(
                        {
                            "image_id": image_id,
                            "mask_path": str(mask_path),
                            "valid_threshold": float(valid_threshold),
                            "sampling_method": method,
                            "candidate_pairs": 0,
                            "selected_pairs": 0,
                            "positive_available": 0,
                            "negative_available": 0,
                            "positive_selected": 0,
                            "negative_selected": 0,
                        }
                    )
                    continue
                attach_region_pair_difficulty(candidate_records)
                image_candidate_records.extend(_attach_selection_scores(candidate_records, method=method))
                selection_rng = np.random.default_rng(seed + image_index * 4001 + threshold_index * 257 + method_index * 29 + 200001)
                selected_records = select_pair(
                    records=candidate_records,
                    method=method,
                    target_count=target_count,
                    positive_ratio=positive_ratio,
                    rng=selection_rng,
                    feature_bin_count=feature_bin_count,
                )
                image_selected_records.extend(selected_records)

                generation_summary.append(
                    {
                        "image_id": image_id,
                        "mask_path": str(mask_path),
                        "valid_threshold": float(valid_threshold),
                        "sampling_method": method,
                        "candidate_pairs": len(candidate_records),
                        "selected_pairs": len(selected_records),
                        "positive_available": sum(int(record.get("connected", 0)) == 1 for record in candidate_records),
                        "negative_available": sum(int(record.get("connected", 0)) == 0 for record in candidate_records),
                        "positive_selected": sum(int(record.get("connected", 0)) == 1 for record in selected_records),
                        "negative_selected": sum(int(record.get("connected", 0)) == 0 for record in selected_records),
                    }
                )

        image_selected_records = _ensure_fixed_pairs_per_image(
            selected_records=image_selected_records,
            candidate_records=image_candidate_records,
            target_count=num_pairs_per_image,
            positive_ratio=positive_ratio,
            strict=strict_num_pairs_per_image,
            rng=np.random.default_rng(seed + image_index * 9001 + 17),
            image_id=image_id,
            mask_path=mask_path,
        )
        for selection_rank, selected_record in enumerate(image_selected_records, start=1):
            enriched_record = dict(selected_record)
            enriched_record["selection_rank"] = selection_rank
            pair_records.append(
                make_pair_record(
                    record=enriched_record,
                    mask_path=mask_path,
                    explanation_config=explanation_config,
                )
            )

    output_paths = write_outputs(
        output_dir=output_dir,
        records=pair_records,
        metadata=_build_metadata(
            config=config,
            input_dir=input_dir,
            output_dir=output_dir,
            methods=methods,
            valid_thresholds=valid_thresholds,
            records=pair_records,
            generation_summary=generation_summary,
            skipped_images=skipped_images,
        ),
        input_dir=input_dir,
        visualization_config=visualization_config,
    )
    return {
        "records": pair_records,
        "output_paths": output_paths,
        "generation_summary": generation_summary,
        "skipped_images": skipped_images,
    }


def _attach_selection_scores(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    enriched = []
    for record in records:
        copied = dict(record)
        copied["selection_score"] = score_pair_for_method(copied, method=method)
        enriched.append(copied)
    return enriched


def _build_metadata(
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    methods: list[str],
    valid_thresholds: list[float],
    records: list[PairRecord],
    generation_summary: list[dict[str, Any]],
    skipped_images: list[dict[str, Any]],
) -> dict[str, Any]:
    csv_records = [record.to_csv_record() for record in records]
    label_counts = {
        "connected": sum(int(record.get("connected", 0)) == 1 for record in csv_records),
        "disconnected": sum(int(record.get("connected", 0)) == 0 for record in csv_records),
    }
    method_counts: dict[str, int] = {}
    for record in csv_records:
        method = str(record.get("sampling_method", "unknown"))
        method_counts[method] = method_counts.get(method, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_pairs": len(records),
        "num_review_pairs": sum(int(record.get("needs_human_review", 0)) == 1 for record in csv_records),
        "label_counts": label_counts,
        "method_counts": method_counts,
        "methods": methods,
        "valid_thresholds": valid_thresholds,
        "per_image_text_dir": str(output_dir / "per_image_text"),
        "visualization_dir": str(output_dir / "visualizations"),
        "generation_summary": generation_summary,
        "skipped_images": skipped_images,
        "config": config,
    }


def _combination_targets(num_pairs_per_image: int, methods: list[str], valid_thresholds: list[float]) -> dict[tuple[str, float], int]:
    combinations = [(method, float(valid_threshold)) for valid_threshold in valid_thresholds for method in methods]
    if not combinations:
        return {}
    base = int(num_pairs_per_image) // len(combinations)
    remainder = int(num_pairs_per_image) % len(combinations)
    targets: dict[tuple[str, float], int] = {}
    for index, key in enumerate(combinations):
        targets[key] = base + (1 if index < remainder else 0)
    return targets


def _default_points_per_method(target_count: int) -> int:
    if target_count <= 0:
        return 8
    return max(8, int(math.sqrt(max(4, target_count * 8))) + 2)


def _sample_region_points(
    valid_mask: np.ndarray,
    distance_map: np.ndarray,
    method: str,
    n_points: int,
    candidate_pool_size: int,
    rng: np.random.Generator,
    region_value: int,
    region_name: str,
) -> list[dict[str, Any]]:
    sampled_points = sample_points(
        valid_mask=valid_mask,
        distance_map=distance_map,
        method=method,
        n_points=n_points,
        candidate_pool_limit=candidate_pool_size,
        rng=rng,
    )
    return [
        {
            "row": int(row),
            "col": int(col),
            "region_value": int(region_value),
            "region_name": str(region_name),
            "distance_to_boundary": float(distance_map[row, col]),
        }
        for row, col in sampled_points
    ]


def _deduplicate_by_pair_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    deduplicated = []
    for record in records:
        pair_id = str(record.get("pair_id"))
        if pair_id in seen_ids:
            continue
        seen_ids.add(pair_id)
        deduplicated.append(record)
    return deduplicated


def _ensure_fixed_pairs_per_image(
    selected_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    target_count: int,
    positive_ratio: float,
    strict: bool,
    rng: np.random.Generator,
    image_id: str,
    mask_path: str | Path,
) -> list[dict[str, Any]]:
    selected_records = _deduplicate_by_pair_id(selected_records)
    if len(selected_records) >= target_count:
        return selected_records[:target_count]
    selected_ids = {record["pair_id"] for record in selected_records}
    leftovers = [record for record in _deduplicate_by_pair_id(candidate_records) if record["pair_id"] not in selected_ids]
    deficit = target_count - len(selected_records)
    if deficit > 0 and leftovers:
        selected_records.extend(
            _balanced_fill_selection(
                records=leftovers,
                target_count=deficit,
                positive_ratio=positive_ratio,
                rng=rng,
            )
        )
        selected_records = _deduplicate_by_pair_id(selected_records)
    if strict and len(selected_records) != target_count:
        raise RuntimeError(
            f"Could not produce the requested fixed pair count for image_id={image_id} "
            f"(mask_path={mask_path}, requested={target_count}, obtained={len(selected_records)}). "
            "Lower valid_thresholds, increase points_per_method, or set strict_num_pairs_per_image=false."
        )
    return selected_records[:target_count]


def _balanced_fill_selection(
    records: list[dict[str, Any]],
    target_count: int,
    positive_ratio: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    positives = [record for record in records if int(record.get("connected", 0)) == 1]
    negatives = [record for record in records if int(record.get("connected", 0)) == 0]
    positive_target = min(len(positives), int(round(target_count * positive_ratio)))
    negative_target = min(len(negatives), target_count - positive_target)
    while positive_target + negative_target < target_count:
        positive_remaining = len(positives) - positive_target
        negative_remaining = len(negatives) - negative_target
        if positive_remaining <= 0 and negative_remaining <= 0:
            break
        if positive_remaining > negative_remaining and positive_remaining > 0:
            positive_target += 1
            continue
        if negative_remaining > 0:
            negative_target += 1
            continue
        if positive_remaining > 0:
            positive_target += 1
            continue
        break

    selected: list[dict[str, Any]] = []
    selected.extend(_ranked_subset_selection(positives, positive_target, rng=rng))
    selected.extend(_ranked_subset_selection(negatives, negative_target, rng=rng))
    if len(selected) < target_count:
        used_ids = {record["pair_id"] for record in selected}
        leftovers = [record for record in records if record["pair_id"] not in used_ids]
        selected.extend(_ranked_subset_selection(leftovers, target_count - len(selected), rng=rng))
    return selected[:target_count]


def _ranked_subset_selection(
    records: list[dict[str, Any]],
    target_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    prepared = list(records)
    rng.shuffle(prepared)
    prepared.sort(key=lambda record: float(_to_float(record.get("selection_score")) or 0.0), reverse=True)
    return prepared[:target_count]


def write_per_image_texts(
    output_dir: str | Path,
    input_dir: str | Path,
    records: list[PairRecord],
) -> None:
    grouped_records = _group_records_by_image(records)
    for image_path, image_records in grouped_records.items():
        text_path = _per_image_output_path(
            base_dir=output_dir,
            input_dir=input_dir,
            image_path=image_path,
            suffix=".txt",
        )
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            build_image_explanation_text(image_path=image_path, records=image_records),
            encoding="utf-8",
        )


def save_pair_visualizations(
    output_dir: str | Path,
    input_dir: str | Path,
    records: list[PairRecord],
    visualization_config: dict[str, Any],
) -> None:
    if not bool(visualization_config.get("enabled", True)):
        return
    point_radius = max(2, int(visualization_config.get("point_radius", 3)))
    line_width = max(1, int(visualization_config.get("line_width", 2)))
    for record in records:
        out_path = _per_pair_visualization_path(
            base_dir=output_dir,
            input_dir=input_dir,
            image_path=record.mask_path,
            pair_id=record.pair_id,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_pair_visualization(
            image_path=record.mask_path,
            record=record,
            out_path=out_path,
            point_radius=point_radius,
            line_width=line_width,
        )


def save_pair_visualization(
    image_path: str | Path,
    record: PairRecord,
    out_path: str | Path,
    point_radius: int = 3,
    line_width: int = 2,
) -> None:
    image = _load_visualization_base_image(image_path)
    overlay = Image.new("RGBA", image.size, color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start = (int(record.point_a_x), int(record.point_a_y))
    end = (int(record.point_b_x), int(record.point_b_y))
    if int(record.connected) == 1:
        line_color = (56, 214, 86, 150)
        status_text = "connected=1"
    else:
        line_color = (255, 170, 0, 170)
        status_text = "connected=0"
    draw.line([start, end], fill=line_color, width=line_width)
    _draw_point(draw, center=start, radius=point_radius, fill=(255, 0, 120, 230), outline=(255, 255, 255, 230))
    _draw_point(draw, center=end, radius=point_radius, fill=(0, 175, 255, 230), outline=(255, 255, 255, 230))

    banner_height = 18
    banner = Image.new("RGBA", (image.width, banner_height), color=(0, 0, 0, 180))
    banner_draw = ImageDraw.Draw(banner)
    banner_draw.text(
        (4, 3),
        f"{record.pair_id} {record.sampling_method} {status_text}",
        fill=(255, 255, 255, 255),
    )
    composed = Image.alpha_composite(image, overlay)
    canvas = Image.new("RGBA", (image.width, image.height + banner_height), color=(0, 0, 0, 255))
    canvas.paste(banner, (0, 0))
    canvas.paste(composed, (0, banner_height))
    canvas.convert("RGB").save(out_path)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    left = int(center[0]) - radius
    top = int(center[1]) - radius
    right = int(center[0]) + radius
    bottom = int(center[1]) + radius
    draw.ellipse([left, top, right, bottom], fill=fill, outline=outline, width=1)


def _load_visualization_base_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    array = np.asarray(image)
    if array.ndim == 2:
        array = _normalize_display_array(array)
        rgb = np.repeat(array[:, :, None], 3, axis=2)
        return Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("RGBA")
    display = _normalize_display_array(array[:, :, : min(3, array.shape[2])])
    if display.ndim == 2:
        display = np.repeat(display[:, :, None], 3, axis=2)
    return Image.fromarray(display.astype(np.uint8), mode="RGB").convert("RGBA")


def _normalize_display_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return array.astype(np.uint8)
    max_value = float(array.max())
    if max_value <= 1.0:
        return (array.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    if max_value < 16.0:
        scaled = array.astype(np.float32) * (255.0 / max_value)
        return scaled.clip(0, 255).astype(np.uint8)
    return array.astype(np.uint8)


def _group_records_by_image(records: list[PairRecord]) -> dict[str, list[PairRecord]]:
    grouped: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.mask_path)].append(record)
    return grouped


def _per_image_output_path(
    base_dir: str | Path,
    input_dir: str | Path,
    image_path: str | Path,
    suffix: str,
) -> Path:
    base_dir = Path(base_dir)
    input_dir = Path(input_dir)
    image_path = Path(image_path)
    try:
        relative = image_path.relative_to(input_dir)
    except ValueError:
        relative = Path(image_path.name)
    parent = relative.parent
    stem = relative.stem
    return base_dir / parent / f"{stem}{suffix}"


def _per_pair_visualization_path(
    base_dir: str | Path,
    input_dir: str | Path,
    image_path: str | Path,
    pair_id: str,
) -> Path:
    base_dir = Path(base_dir)
    input_dir = Path(input_dir)
    image_path = Path(image_path)
    try:
        relative = image_path.relative_to(input_dir)
    except ValueError:
        relative = Path(image_path.name)
    parent = relative.parent
    stem = relative.stem
    safe_pair_id = _safe_filename_token(pair_id)
    return base_dir / parent / f"{stem}_{safe_pair_id}_vis.png"


def _safe_filename_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return token.strip("._") or "pair"


def _collect_fieldnames(records: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    seen = set(preferred)
    fieldnames = list(preferred)
    for record in records:
        for key in record.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def _write_csv_allow_empty(path: str | Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _image_id_from_path(index: int, path: Path, input_dir: Path) -> str:
    try:
        relative = path.relative_to(input_dir)
        stem = relative.with_suffix("").as_posix().replace("/", "__")
    except ValueError:
        stem = path.stem
    return f"{index:04d}_{stem}"


def _method_reason_code(method: str) -> str:
    return {
        "random": "SELECTED_BY_RANDOM",
        "euclidean_fps": "SELECTED_BY_EUCLIDEAN_FARTHEST",
        "euclidean_farthest": "SELECTED_BY_EUCLIDEAN_FARTHEST",
        "geodesic_fps": "SELECTED_BY_GEODESIC_FARTHEST",
        "geodesic_farthest": "SELECTED_BY_GEODESIC_FARTHEST",
        "boundary_crossing_targeted": "SELECTED_BY_BOUNDARY_CROSSING_RULE",
        "boundary_crossing_stratified": "SELECTED_BY_BOUNDARY_CROSSING_STRATIFICATION",
        "feature_space_diverse": "SELECTED_BY_FEATURE_SPACE_DIVERSITY",
        "hard_pair_targeted": "SELECTED_BY_HARDNESS_RULE",
        "hard_pair_targeted_v2": "SELECTED_BY_HARDNESS_RULE_V2",
    }.get(method, f"SELECTED_BY_{method.upper()}")


def _method_sentence(method: str) -> str:
    return {
        "random": "This pair was sampled from the valid point set without extra geometric prioritization.",
        "euclidean_fps": "This pair comes from points that were spread apart in Euclidean space.",
        "euclidean_farthest": "This pair comes from points that were spread apart in Euclidean space.",
        "geodesic_fps": "This pair comes from points that were spread apart along the in-mask geodesic geometry.",
        "geodesic_farthest": "This pair comes from points that were spread apart along the in-mask geodesic geometry.",
        "boundary_crossing_targeted": "This pair was prioritized because the straight segment tends to interact with the boundary.",
        "boundary_crossing_stratified": "This pair was retained to cover several boundary-crossing regimes while preserving label diversity.",
        "feature_space_diverse": "This pair was retained to cover a distinct region of the feature space.",
        "hard_pair_targeted": "This pair was prioritized by a rule-based hardness score.",
        "hard_pair_targeted_v2": "This pair was prioritized by the strengthened rule-based hardness score.",
    }.get(method, f"This pair was selected by the {method} sampling rule.")


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _extract_mask_values(
    image: np.ndarray,
    mask_channel: int | str,
    use_alpha_channel: bool,
) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if use_alpha_channel and array.shape[2] >= 4:
        return array[:, :, 3]
    if str(mask_channel).lower() == "auto":
        limit = 3 if array.shape[2] >= 3 else array.shape[2]
        return array[:, :, :limit].max(axis=2)
    channel = int(mask_channel)
    channel = max(0, min(channel, array.shape[2] - 1))
    return array[:, :, channel]


def _binarize_values(
    values: np.ndarray,
    binarization_mode: str,
    foreground_threshold: int,
    foreground_values: list[int] | None,
) -> np.ndarray:
    if foreground_values:
        return np.isin(values, foreground_values)
    mode = str(binarization_mode).strip().lower()
    if mode == "nonzero":
        return values > 0
    if mode == "threshold":
        return values >= foreground_threshold
    if mode == "auto":
        max_value = float(np.max(values)) if values.size else 0.0
        unique_values = np.unique(values)
        if max_value <= 1.0:
            return values > 0
        if len(unique_values) <= 4:
            return values > 0
        return values >= foreground_threshold
    raise ValueError(f"Unknown binarization mode: {binarization_mode}")


def _sanitize_scalar(value: Any, for_json: bool) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value) if for_json else str(tuple(value))
    if isinstance(value, list):
        if for_json:
            return [_sanitize_scalar(item, for_json=True) for item in value]
        return json.dumps([_sanitize_scalar(item, for_json=True) for item in value], ensure_ascii=False)
    if isinstance(value, dict):
        converted = {str(key): _sanitize_scalar(item, for_json=True) for key, item in value.items()}
        return converted if for_json else json.dumps(converted, ensure_ascii=False)
    return value


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_round(value: Any, digits: int = 6) -> Any:
    numeric = _to_float(value)
    if numeric is None:
        return ""
    return round(numeric, digits)


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"
