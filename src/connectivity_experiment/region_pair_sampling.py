from __future__ import annotations

import glob
import heapq
import math
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


def discover_mask_paths(mask_glob: str, repo_root: str | Path, limit: int | None = None) -> list[Path]:
    repo_root = Path(repo_root)
    pattern = mask_glob if Path(mask_glob).is_absolute() else str(repo_root / mask_glob)
    paths = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
    if limit is not None:
        return paths[:limit]
    return paths


def load_binary_mask(
    image_path: str | Path,
    mask_channel: int = 0,
    foreground_threshold: int = 16,
    invert_mask: bool = False,
    min_component_area: int = 1,
) -> np.ndarray:
    image = Image.open(image_path)
    array = np.asarray(image)
    if array.ndim == 3:
        channel = max(0, min(mask_channel, array.shape[2] - 1))
        values = array[:, :, channel]
    else:
        values = array
    mask = values >= foreground_threshold
    if invert_mask:
        mask = ~mask
    mask = _keep_large_components(mask, min_component_area=min_component_area)
    return mask.astype(bool)


def summarize_mask(mask: np.ndarray, image_id: str, image_path: str | Path) -> dict:
    labels, component_sizes = connected_components(mask)
    height, width = mask.shape
    area = int(mask.sum())
    largest_component_area = max(component_sizes, default=0)
    bbox = _foreground_bbox(mask)
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "height": height,
        "width": width,
        "foreground_area": area,
        "foreground_fraction": round(area / max(1, height * width), 6),
        "n_components": len(component_sizes),
        "largest_component_area": int(largest_component_area),
        "bbox_row_min": bbox[0],
        "bbox_col_min": bbox[1],
        "bbox_row_max": bbox[2],
        "bbox_col_max": bbox[3],
        "is_single_component": int(len(component_sizes) <= 1),
    }


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return _distance_transform_fallback(mask.astype(bool))
    return distance_transform_edt(mask)


def build_valid_mask(mask: np.ndarray, distance_map: np.ndarray, min_distance_to_boundary: float) -> np.ndarray:
    return np.logical_and(mask, distance_map > float(min_distance_to_boundary))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    labels = -np.ones(mask.shape, dtype=np.int32)
    component_sizes: list[int] = []
    component_id = 0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or labels[row, col] != -1:
                continue
            stack = [(row, col)]
            labels[row, col] = component_id
            size = 0
            while stack:
                cur_row, cur_col = stack.pop()
                size += 1
                for delta_row, delta_col, _ in _NEIGHBORS_8:
                    next_row = cur_row + delta_row
                    next_col = cur_col + delta_col
                    if not _is_inside(mask.shape, next_row, next_col):
                        continue
                    if not mask[next_row, next_col] or labels[next_row, next_col] != -1:
                        continue
                    labels[next_row, next_col] = component_id
                    stack.append((next_row, next_col))
            component_sizes.append(size)
            component_id += 1
    return labels, component_sizes


def sample_points(
    valid_mask: np.ndarray,
    distance_map: np.ndarray,
    method: str,
    n_points: int,
    candidate_pool_limit: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates = np.argwhere(valid_mask)
    if len(candidates) == 0:
        return []
    candidate_pool = _candidate_pool(candidates, candidate_pool_limit=candidate_pool_limit, rng=rng)
    if len(candidate_pool) <= n_points:
        return [tuple(int(value) for value in point) for point in candidate_pool]
    point_method = _resolve_point_sampling_method(method)
    if point_method == "random":
        chosen_indices = rng.choice(len(candidate_pool), size=n_points, replace=False)
        return [tuple(int(value) for value in candidate_pool[index]) for index in chosen_indices]
    if point_method == "euclidean_fps":
        return _farthest_point_sampling(candidate_pool, n_points=n_points, seed_index=_seed_index(candidate_pool, distance_map))
    if point_method == "feature_diversity":
        return _feature_diversity_sampling(candidate_pool, distance_map, n_points=n_points)
    if point_method == "geodesic_fps":
        return _geodesic_farthest_point_sampling(candidate_pool, valid_mask, distance_map, n_points=n_points)
    raise ValueError(f"Unknown sampling method: {method}")


def _resolve_point_sampling_method(method: str) -> str:
    aliases = {
        "random": "random",
        "euclidean_fps": "euclidean_fps",
        "euclidean_farthest": "euclidean_fps",
        "geodesic_fps": "geodesic_fps",
        "geodesic_farthest": "geodesic_fps",
        "feature_diversity": "feature_diversity",
        "feature_space_diverse": "random",
        "boundary_crossing_targeted": "random",
        "hard_pair_targeted": "random",
        "hard_pair_targeted_v2": "random",
        "boundary_crossing_stratified": "random",
    }
    resolved = aliases.get(str(method))
    if resolved is None:
        raise ValueError(f"Unknown sampling method: {method}")
    return resolved


def build_pair_records(
    mask: np.ndarray,
    distance_map: np.ndarray,
    points: list[tuple[int, int]] | list[dict],
    image_id: str,
    image_path: str | Path,
    valid_threshold: float,
    sampling_method: str,
    background_distance_map: np.ndarray | None = None,
    pair_component_mode: str = "same_component",
) -> list[dict]:
    background_mask = ~mask
    background_distance_map = compute_distance_transform(background_mask) if background_distance_map is None else background_distance_map
    foreground_labels, foreground_component_sizes = connected_components(mask)
    background_labels, background_component_sizes = connected_components(background_mask)
    foreground_fraction = float(mask.sum()) / max(1, mask.shape[0] * mask.shape[1])
    component_count = int(len(foreground_component_sizes))
    point_entries = [
        _normalize_point_entry(
            point,
            mask=mask,
            foreground_distance_map=distance_map,
            background_distance_map=background_distance_map,
        )
        for point in points
    ]
    geodesic_pair_cache: dict[tuple[int, int, int], float] = {}
    component_distance_cache: dict[tuple[int, int], np.ndarray] = {}
    component_gap_cache: dict[tuple[int, int, int], float] = {}
    records: list[dict] = []

    for point_index, (start_entry, end_entry) in enumerate(combinations(point_entries, 2)):
        start = (int(start_entry["row"]), int(start_entry["col"]))
        end = (int(end_entry["row"]), int(end_entry["col"]))
        region_value_i = int(start_entry["region_value"])
        region_value_j = int(end_entry["region_value"])
        same_region_value = int(region_value_i == region_value_j)
        component_start = int(foreground_labels[start] if region_value_i == 1 else background_labels[start])
        component_end = int(foreground_labels[end] if region_value_j == 1 else background_labels[end])
        component_area_start = _component_area(
            region_value=region_value_i,
            component_id=component_start,
            foreground_component_sizes=foreground_component_sizes,
            background_component_sizes=background_component_sizes,
        )
        component_area_end = _component_area(
            region_value=region_value_j,
            component_id=component_end,
            foreground_component_sizes=foreground_component_sizes,
            background_component_sizes=background_component_sizes,
        )
        same_component = int(same_region_value == 1 and component_start == component_end and component_start >= 0)
        connected = int(same_component == 1)
        if pair_component_mode == "same_component" and not connected:
            continue
        if pair_component_mode == "same_region" and not same_region_value:
            continue

        euclidean_distance = float(math.hypot(end[1] - start[1], end[0] - start[0]))
        geodesic_distance = None
        domain_mask = mask if region_value_i == 1 else background_mask
        path_exists = connected == 1
        if path_exists:
            start_flat = start[0] * mask.shape[1] + start[1]
            end_flat = end[0] * mask.shape[1] + end[1]
            low_flat, high_flat = sorted((start_flat, end_flat))
            cache_key = (region_value_i, low_flat, high_flat)
            geodesic_distance = geodesic_pair_cache.get(cache_key)
            if geodesic_distance is None:
                geodesic_distance = geodesic_distance_between(domain_mask, start, end)
                if geodesic_distance is not None:
                    geodesic_pair_cache[cache_key] = float(geodesic_distance)

        distance_to_boundary_start = float(start_entry["distance_to_boundary"])
        distance_to_boundary_end = float(end_entry["distance_to_boundary"])
        distance_to_boundary_min = min(distance_to_boundary_start, distance_to_boundary_end)
        boundary_crossings = count_boundary_crossings(mask, start, end)
        line_inside_fraction = fraction_line_inside_mask(mask, start, end)
        max_outside_segment_length = longest_outside_segment_length(mask, start, end)
        geo_euc_ratio = geodesic_distance / euclidean_distance if geodesic_distance is not None and euclidean_distance > 0 else None
        component_gap_distance = None
        if same_region_value == 1 and connected == 0 and component_start >= 0 and component_end >= 0:
            component_gap_distance = _component_gap_distance(
                region_value=region_value_i,
                component_a=component_start,
                component_b=component_end,
                labels=foreground_labels if region_value_i == 1 else background_labels,
                distance_cache=component_distance_cache,
                gap_cache=component_gap_cache,
            )

        record = {
            "pair_id": f"{image_id}_{sampling_method}_t{_threshold_tag(valid_threshold)}_{point_index:04d}",
            "image_id": image_id,
            "image_path": str(image_path),
            "valid_threshold": float(valid_threshold),
            "sampling_method": sampling_method,
            "point_count": len(point_entries),
            "row_i": int(start[0]),
            "col_i": int(start[1]),
            "row_j": int(end[0]),
            "col_j": int(end[1]),
            "point_region_i": str(start_entry["region_name"]),
            "point_region_j": str(end_entry["region_name"]),
            "region_value_i": region_value_i,
            "region_value_j": region_value_j,
            "pair_region_type": _pair_region_type(region_value_i, region_value_j),
            "same_region_value": same_region_value,
            "component_i": component_start,
            "component_j": component_end,
            "component_area_i": int(component_area_start),
            "component_area_j": int(component_area_end),
            "same_component": same_component,
            "connected": connected,
            "foreground_fraction": round(foreground_fraction, 6),
            "component_count": component_count,
            "distance_to_boundary_i": round(distance_to_boundary_start, 6),
            "distance_to_boundary_j": round(distance_to_boundary_end, 6),
            "distance_to_boundary_min": round(distance_to_boundary_min, 6),
            "euclidean_distance": round(euclidean_distance, 6),
            "geodesic_distance": round(geodesic_distance, 6) if geodesic_distance is not None else "",
            "geodesic_euclidean_ratio": round(geo_euc_ratio, 6) if geo_euc_ratio is not None else "",
            "boundary_crossing_count": int(boundary_crossings),
            "line_inside_fraction": round(line_inside_fraction, 6),
            "max_outside_segment_length": round(max_outside_segment_length, 6),
            "component_gap_distance": round(component_gap_distance, 6) if component_gap_distance is not None else "",
            "feature_triplet": [
                round(euclidean_distance, 6),
                round(geodesic_distance, 6) if geodesic_distance is not None else None,
                int(boundary_crossings),
            ],
        }
        records.append(record)
    return records


def geodesic_distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    if not bool(mask[start]):
        raise ValueError("Geodesic source must lie inside the mask.")
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = [(0.0, int(start[0]), int(start[1]))]
    distances[start] = 0.0
    while heap:
        cost, row, col = heapq.heappop(heap)
        if cost > distances[row, col]:
            continue
        for delta_row, delta_col, weight in _NEIGHBORS_8:
            next_row = row + delta_row
            next_col = col + delta_col
            if not _is_inside(mask.shape, next_row, next_col) or not mask[next_row, next_col]:
                continue
            next_cost = cost + weight
            if next_cost < distances[next_row, next_col]:
                distances[next_row, next_col] = next_cost
                heapq.heappush(heap, (next_cost, next_row, next_col))
    return distances


def geodesic_distance_between(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> float | None:
    if not bool(mask[start]) or not bool(mask[end]):
        return None
    if start == end:
        return 0.0
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, float, int, int]] = [(_heuristic_cost(start, end), 0.0, int(start[0]), int(start[1]))]
    distances[start] = 0.0
    while heap:
        _, cost, row, col = heapq.heappop(heap)
        if cost > distances[row, col]:
            continue
        if (row, col) == end:
            return float(cost)
        for delta_row, delta_col, weight in _NEIGHBORS_8:
            next_row = row + delta_row
            next_col = col + delta_col
            if not _is_inside(mask.shape, next_row, next_col) or not mask[next_row, next_col]:
                continue
            next_cost = cost + weight
            if next_cost < distances[next_row, next_col]:
                distances[next_row, next_col] = next_cost
                priority = next_cost + _heuristic_cost((next_row, next_col), end)
                heapq.heappush(heap, (priority, next_cost, next_row, next_col))
    return None


def shortest_path_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if not bool(mask[start]) or not bool(mask[end]):
        return []
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    parents_row = np.full(mask.shape, -1, dtype=np.int32)
    parents_col = np.full(mask.shape, -1, dtype=np.int32)
    heap: list[tuple[float, int, int]] = [(0.0, int(start[0]), int(start[1]))]
    distances[start] = 0.0
    while heap:
        cost, row, col = heapq.heappop(heap)
        if cost > distances[row, col]:
            continue
        if (row, col) == end:
            break
        for delta_row, delta_col, weight in _NEIGHBORS_8:
            next_row = row + delta_row
            next_col = col + delta_col
            if not _is_inside(mask.shape, next_row, next_col) or not mask[next_row, next_col]:
                continue
            next_cost = cost + weight
            if next_cost < distances[next_row, next_col]:
                distances[next_row, next_col] = next_cost
                parents_row[next_row, next_col] = row
                parents_col[next_row, next_col] = col
                heapq.heappush(heap, (next_cost, next_row, next_col))
    if not np.isfinite(distances[end]):
        return []
    path = [end]
    cursor = end
    while cursor != start:
        parent_row = int(parents_row[cursor])
        parent_col = int(parents_col[cursor])
        if parent_row < 0 or parent_col < 0:
            return []
        cursor = (parent_row, parent_col)
        path.append(cursor)
    path.reverse()
    return path


def count_boundary_crossings(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> int:
    line = bresenham_line(start, end)
    transitions = 0
    previous_state = bool(mask[line[0]])
    for point in line[1:]:
        current_state = bool(mask[point])
        if current_state != previous_state:
            transitions += 1
            previous_state = current_state
    return transitions


def fraction_line_inside_mask(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> float:
    line = bresenham_line(start, end)
    inside = sum(int(bool(mask[point])) for point in line)
    return inside / max(1, len(line))


def longest_outside_segment_length(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> float:
    line = bresenham_line(start, end)
    if len(line) < 2:
        return 0.0
    max_length = 0.0
    current_length = 0.0
    previous_point = line[0]
    previous_inside = bool(mask[previous_point])
    for point in line[1:]:
        step_length = float(math.hypot(point[0] - previous_point[0], point[1] - previous_point[1]))
        current_inside = bool(mask[point])
        if not current_inside:
            current_length = current_length + step_length if not previous_inside else step_length
            max_length = max(max_length, current_length)
        else:
            current_length = 0.0
        previous_point = point
        previous_inside = current_inside
    return max_length


def bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = start
    row1, col1 = end
    delta_row = abs(row1 - row0)
    delta_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    err = delta_col - delta_row
    line = []

    while True:
        line.append((row0, col0))
        if row0 == row1 and col0 == col1:
            break
        twice_err = 2 * err
        if twice_err > -delta_row:
            err -= delta_row
            col0 += step_col
        if twice_err < delta_col:
            err += delta_col
            row0 += step_row
    return line


def _heuristic_cost(start: tuple[int, int], end: tuple[int, int]) -> float:
    return float(math.hypot(end[0] - start[0], end[1] - start[1]))


def _component_gap_distance(
    region_value: int,
    component_a: int,
    component_b: int,
    labels: np.ndarray,
    distance_cache: dict[tuple[int, int], np.ndarray],
    gap_cache: dict[tuple[int, int, int], float],
) -> float | None:
    if component_a < 0 or component_b < 0:
        return None
    if component_a == component_b:
        return 0.0
    low_component, high_component = sorted((int(component_a), int(component_b)))
    cache_key = (int(region_value), low_component, high_component)
    cached = gap_cache.get(cache_key)
    if cached is not None:
        return cached
    distance_map_key = (int(region_value), low_component)
    distance_map = distance_cache.get(distance_map_key)
    if distance_map is None:
        distance_map = compute_distance_transform(labels != low_component)
        distance_cache[distance_map_key] = distance_map
    target_coords = np.argwhere(labels == high_component)
    if len(target_coords) == 0:
        return None
    gap_value = float(distance_map[target_coords[:, 0], target_coords[:, 1]].min())
    gap_cache[cache_key] = gap_value
    return gap_value


def _component_area(
    region_value: int,
    component_id: int,
    foreground_component_sizes: list[int],
    background_component_sizes: list[int],
) -> int:
    if component_id < 0:
        return 0
    component_sizes = foreground_component_sizes if int(region_value) == 1 else background_component_sizes
    if component_id >= len(component_sizes):
        return 0
    return int(component_sizes[component_id])


def select_pair_records(
    records: list[dict],
    method: str,
    target_count: int | None,
    rng: np.random.Generator,
    feature_bin_count: int = 4,
) -> list[dict]:
    if not records:
        return []
    resolved_method = str(method)
    if target_count is None or target_count <= 0 or target_count >= len(records):
        if resolved_method == "hard_pair_targeted":
            return sorted(records, key=lambda record: _pair_selection_score(record), reverse=True)
        if resolved_method == "hard_pair_targeted_v2":
            return sorted(records, key=lambda record: _pair_selection_score_v2(record), reverse=True)
        return list(records)
    if resolved_method in {"random", "euclidean_fps", "euclidean_farthest", "geodesic_fps", "geodesic_farthest", "feature_diversity"}:
        chosen_indices = rng.choice(len(records), size=target_count, replace=False)
        return [records[index] for index in np.sort(chosen_indices)]
    if resolved_method == "boundary_crossing_targeted":
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(_boundary_crossing_bin(record.get("boundary_crossing_count")), []).append(record)
        return _round_robin_group_selection(groups, target_count=target_count, rng=rng)
    if resolved_method == "boundary_crossing_stratified":
        return _select_boundary_crossing_stratified(records, target_count=target_count, rng=rng, n_bins=feature_bin_count)
    if resolved_method == "feature_space_diverse":
        return _select_feature_space_diverse(records, target_count=target_count, rng=rng, n_bins=feature_bin_count)
    if resolved_method == "hard_pair_targeted":
        return _select_hard_pairs(records, target_count=target_count, rng=rng)
    if resolved_method == "hard_pair_targeted_v2":
        return _select_hard_pairs_v2(records, target_count=target_count, rng=rng)
    raise ValueError(f"Unknown sampling method: {method}")


def _select_feature_space_diverse(
    records: list[dict],
    target_count: int,
    rng: np.random.Generator,
    n_bins: int = 4,
) -> list[dict]:
    connected_records = [record for record in records if int(record.get("connected", 0)) == 1]
    disconnected_records = [record for record in records if int(record.get("connected", 0)) == 0]
    connected_target = int(round(target_count * len(connected_records) / max(1, len(records))))
    disconnected_target = target_count - connected_target
    if connected_records and disconnected_target == target_count:
        connected_target = 1
        disconnected_target = target_count - 1
    if disconnected_records and connected_target == target_count:
        disconnected_target = 1
        connected_target = target_count - 1
    selected = []
    selected.extend(_feature_space_subset_selection(connected_records, connected_target, rng=rng, n_bins=n_bins, connected_only=True))
    selected.extend(_feature_space_subset_selection(disconnected_records, disconnected_target, rng=rng, n_bins=n_bins, connected_only=False))
    if len(selected) >= target_count:
        return selected[:target_count]
    used_ids = {record["pair_id"] for record in selected}
    leftovers = [record for record in records if record["pair_id"] not in used_ids]
    if leftovers:
        extras = _round_robin_group_selection(
            {"leftover": leftovers},
            target_count=target_count - len(selected),
            rng=rng,
        )
        selected.extend(extras)
    return selected[:target_count]


def _feature_space_subset_selection(
    records: list[dict],
    target_count: int,
    rng: np.random.Generator,
    n_bins: int,
    connected_only: bool,
) -> list[dict]:
    if not records or target_count <= 0:
        return []
    euclidean_edges = _quantile_edges(records, "euclidean_distance", n_bins=n_bins)
    boundary_keys = {}
    if connected_only:
        geodesic_edges = _quantile_edges(records, "geodesic_distance", n_bins=n_bins)
        ratio_edges = _quantile_edges(records, "geodesic_euclidean_ratio", n_bins=n_bins)
        for record in records:
            boundary_keys.setdefault(
                (
                    _assign_quantile_bin(record.get("euclidean_distance"), euclidean_edges),
                    _boundary_crossing_bin(record.get("boundary_crossing_count")),
                    _assign_quantile_bin(record.get("geodesic_distance"), geodesic_edges),
                    _assign_quantile_bin(record.get("geodesic_euclidean_ratio"), ratio_edges),
                ),
                [],
            ).append(record)
    else:
        margin_edges = _quantile_edges(records, "distance_to_boundary_min", n_bins=n_bins)
        inside_edges = _quantile_edges(records, "line_inside_fraction", n_bins=n_bins)
        for record in records:
            boundary_keys.setdefault(
                (
                    _assign_quantile_bin(record.get("euclidean_distance"), euclidean_edges),
                    _boundary_crossing_bin(record.get("boundary_crossing_count")),
                    _assign_quantile_bin(record.get("distance_to_boundary_min"), margin_edges),
                    _assign_quantile_bin(record.get("line_inside_fraction"), inside_edges),
                ),
                [],
            ).append(record)
    return _round_robin_group_selection(boundary_keys, target_count=target_count, rng=rng)


def _select_hard_pairs(records: list[dict], target_count: int, rng: np.random.Generator) -> list[dict]:
    connected_records = sorted(
        [record for record in records if int(record.get("connected", 0)) == 1],
        key=lambda record: _pair_selection_score(record),
        reverse=True,
    )
    disconnected_records = sorted(
        [record for record in records if int(record.get("connected", 0)) == 0],
        key=lambda record: _pair_selection_score(record),
        reverse=True,
    )
    connected_target = min(len(connected_records), target_count // 2)
    disconnected_target = min(len(disconnected_records), target_count - connected_target)
    if disconnected_target < target_count // 2 and connected_records:
        connected_target = min(len(connected_records), target_count - disconnected_target)
    if connected_target < target_count // 2 and disconnected_records:
        disconnected_target = min(len(disconnected_records), target_count - connected_target)
    selected = connected_records[:connected_target] + disconnected_records[:disconnected_target]
    if len(selected) < target_count:
        used_ids = {record["pair_id"] for record in selected}
        leftovers = [record for record in sorted(records, key=lambda record: _pair_selection_score(record), reverse=True) if record["pair_id"] not in used_ids]
        selected.extend(leftovers[: target_count - len(selected)])
    rng.shuffle(selected)
    return selected[:target_count]


def _select_hard_pairs_v2(records: list[dict], target_count: int, rng: np.random.Generator) -> list[dict]:
    positive_candidates = sorted(
        [record for record in records if int(record.get("is_hard_positive_v2", 0)) == 1],
        key=lambda record: _pair_selection_score_v2(record),
        reverse=True,
    )
    negative_candidates = sorted(
        [record for record in records if int(record.get("is_hard_negative_v2", 0)) == 1],
        key=lambda record: _pair_selection_score_v2(record),
        reverse=True,
    )
    selected = _balanced_positive_negative_selection(
        positive_candidates=positive_candidates,
        negative_candidates=negative_candidates,
        all_records=records,
        target_count=target_count,
        score_fn=_pair_selection_score_v2,
        rng=rng,
    )
    return selected[:target_count]


def _select_boundary_crossing_stratified(
    records: list[dict],
    target_count: int,
    rng: np.random.Generator,
    n_bins: int = 4,
) -> list[dict]:
    euclidean_edges = _quantile_edges(records, "euclidean_distance", n_bins=n_bins)
    groups: dict[tuple[str, int | None, int], list[dict]] = {}
    for record in records:
        groups.setdefault(
            (
                _boundary_crossing_bin(record.get("boundary_crossing_count")),
                _assign_quantile_bin(record.get("euclidean_distance"), euclidean_edges),
                int(record.get("connected", 0)),
            ),
            [],
        ).append(record)
    return _round_robin_group_selection(groups, target_count=target_count, rng=rng)


def _balanced_positive_negative_selection(
    positive_candidates: list[dict],
    negative_candidates: list[dict],
    all_records: list[dict],
    target_count: int,
    score_fn,
    rng: np.random.Generator,
) -> list[dict]:
    positive_target = min(len(positive_candidates), target_count // 2)
    negative_target = min(len(negative_candidates), target_count - positive_target)
    if negative_target < target_count // 2 and positive_candidates:
        positive_target = min(len(positive_candidates), target_count - negative_target)
    if positive_target < target_count // 2 and negative_candidates:
        negative_target = min(len(negative_candidates), target_count - positive_target)
    selected = positive_candidates[:positive_target] + negative_candidates[:negative_target]
    if len(selected) < target_count:
        used_ids = {record["pair_id"] for record in selected}
        leftovers = [
            record
            for record in sorted(all_records, key=score_fn, reverse=True)
            if record["pair_id"] not in used_ids
        ]
        selected.extend(leftovers[: target_count - len(selected)])
    rng.shuffle(selected)
    return selected


def _pair_selection_score(record: dict) -> float:
    precomputed = _to_float(record.get("hard_pair_score"))
    if precomputed is not None:
        return precomputed
    connected = int(record.get("connected", 0))
    euclidean = _to_float(record.get("euclidean_distance")) or 0.0
    boundary = _to_float(record.get("boundary_crossing_count")) or 0.0
    ratio = _to_float(record.get("geodesic_euclidean_ratio")) or 0.0
    geodesic = _to_float(record.get("geodesic_distance")) or 0.0
    inside_fraction = _to_float(record.get("line_inside_fraction")) or 0.0
    component_gap = _to_float(record.get("component_gap_distance"))
    max_outside = _to_float(record.get("max_outside_segment_length"))
    if connected == 1:
        return geodesic + ratio * 20.0 + boundary * 8.0 - euclidean * 0.15
    return boundary * 10.0 + inside_fraction * 20.0 - euclidean * 0.2 - (component_gap or 0.0) * 2.0 - (max_outside or 0.0)


def _pair_selection_score_v2(record: dict) -> float:
    precomputed = _to_float(record.get("hard_pair_score_v2"))
    if precomputed is not None:
        return precomputed
    return _pair_selection_score(record)


def _round_robin_group_selection(
    groups: dict[object, list[dict]],
    target_count: int,
    rng: np.random.Generator,
) -> list[dict]:
    prepared = {}
    for key, values in groups.items():
        values = list(values)
        if not values:
            continue
        rng.shuffle(values)
        prepared[key] = values
    selected: list[dict] = []
    per_group_counts = {key: 0 for key in prepared}
    while len(selected) < target_count:
        available_keys = [key for key, values in prepared.items() if values]
        if not available_keys:
            break
        available_keys.sort(key=lambda key: (per_group_counts[key], -len(prepared[key]), str(key)))
        for key in available_keys:
            if len(selected) >= target_count or not prepared[key]:
                break
            selected.append(prepared[key].pop())
            per_group_counts[key] += 1
    return selected


def _boundary_crossing_bin(value) -> str:
    crossings = int(float(value))
    if crossings <= 0:
        return "0"
    if crossings == 1:
        return "1"
    if crossings == 2:
        return "2"
    return "3_plus"


def attach_region_pair_difficulty(records: list[dict]) -> list[dict]:
    if not records:
        return records

    connected_records = [record for record in records if int(record.get("connected", 0)) == 1]
    disconnected_records = [record for record in records if int(record.get("connected", 0)) == 0]
    geodesic_values = _collect_numeric(records, "geodesic_distance")
    ratio_values = _collect_numeric(records, "geodesic_euclidean_ratio")
    boundary_values = _collect_numeric(records, "boundary_crossing_count")
    boundary_margin_values = _collect_numeric(records, "distance_to_boundary_min")
    euclidean_values = _collect_numeric(records, "euclidean_distance")
    line_inside_values = _collect_numeric(records, "line_inside_fraction")
    component_gap_values = _collect_numeric(records, "component_gap_distance")
    outside_segment_values = _collect_numeric(records, "max_outside_segment_length")
    positive_euclidean_values = _collect_numeric(connected_records, "euclidean_distance")
    positive_geodesic_values = _collect_numeric(connected_records, "geodesic_distance")
    positive_ratio_values = _collect_numeric(connected_records, "geodesic_euclidean_ratio")
    positive_margin_values = _collect_numeric(connected_records, "distance_to_boundary_min")
    positive_width_values = _collect_numeric(connected_records, "distance_to_boundary_i") + _collect_numeric(connected_records, "distance_to_boundary_j")
    negative_euclidean_values = _collect_numeric(disconnected_records, "euclidean_distance")
    negative_gap_values = _collect_numeric(disconnected_records, "component_gap_distance")
    negative_inside_values = _collect_numeric(disconnected_records, "line_inside_fraction")
    negative_outside_values = _collect_numeric(disconnected_records, "max_outside_segment_length")

    geodesic_mean, geodesic_std = _mean_std(geodesic_values)
    ratio_mean, ratio_std = _mean_std(ratio_values)
    crossing_mean, crossing_std = _mean_std(boundary_values)
    margin_mean, margin_std = _mean_std(boundary_margin_values)
    euclidean_mean, euclidean_std = _mean_std(euclidean_values)
    inside_mean, inside_std = _mean_std(line_inside_values)
    gap_mean, gap_std = _mean_std(component_gap_values)
    outside_mean, outside_std = _mean_std(outside_segment_values)
    euclidean_q33, euclidean_q66 = _quantiles(euclidean_values, [0.33, 0.66])
    geodesic_q33, geodesic_q66 = _quantiles(geodesic_values, [0.33, 0.66])
    positive_euclidean_q25 = _quantiles(positive_euclidean_values, [0.25])[0]
    positive_euclidean_q20 = _quantiles(positive_euclidean_values, [0.20])[0]
    positive_geodesic_q25, positive_geodesic_q75 = _quantiles(positive_geodesic_values, [0.25, 0.75])
    positive_geodesic_q20 = _quantiles(positive_geodesic_values, [0.20])[0]
    positive_ratio_q75 = _quantiles(positive_ratio_values, [0.75])[0]
    positive_margin_q50 = _quantiles(positive_margin_values, [0.50])[0]
    positive_margin_q25 = _quantiles(positive_margin_values, [0.25])[0]
    positive_width_q50 = _quantiles(positive_width_values, [0.50])[0]
    negative_euclidean_q50 = _quantiles(negative_euclidean_values, [0.5])[0]
    negative_gap_q50 = _quantiles(negative_gap_values, [0.5])[0]
    negative_inside_q50 = _quantiles(negative_inside_values, [0.5])[0]
    negative_outside_q50 = _quantiles(negative_outside_values, [0.5])[0]
    ratio_q75 = _quantiles(ratio_values, [0.75])[0]
    crossing_q75 = _quantiles(boundary_values, [0.75])[0]

    for record in records:
        geodesic_distance = _to_float(record.get("geodesic_distance"))
        ratio = _to_float(record.get("geodesic_euclidean_ratio"))
        boundary_crossings = _to_float(record.get("boundary_crossing_count"))
        boundary_margin = _to_float(record.get("distance_to_boundary_min"))
        boundary_i = _to_float(record.get("distance_to_boundary_i"))
        boundary_j = _to_float(record.get("distance_to_boundary_j"))
        euclidean_distance = _to_float(record.get("euclidean_distance"))
        line_inside_fraction = _to_float(record.get("line_inside_fraction"))
        component_gap_distance = _to_float(record.get("component_gap_distance"))
        max_outside_segment_length = _to_float(record.get("max_outside_segment_length"))
        proxy = (
            _zscore_value(geodesic_distance, geodesic_mean, geodesic_std)
            + _zscore_value(ratio, ratio_mean, ratio_std)
            + _zscore_value(boundary_crossings, crossing_mean, crossing_std)
            - _zscore_value(boundary_margin, margin_mean, margin_std)
        )
        connected = int(record.get("connected", 0))
        easy_positive = int(
            connected == 1
            and euclidean_distance is not None
            and geodesic_distance is not None
            and euclidean_distance <= positive_euclidean_q25
            and geodesic_distance <= positive_geodesic_q25
            and (boundary_crossings or 0.0) <= 0.0
        )
        easy_negative = int(
            connected == 0
            and euclidean_distance is not None
            and euclidean_distance >= negative_euclidean_q50
            and (boundary_crossings or 0.0) <= 1.0
            and (
                component_gap_distance is None
                or component_gap_distance >= negative_gap_q50
            )
        )
        easy_positive_v2 = int(
            connected == 1
            and euclidean_distance is not None
            and geodesic_distance is not None
            and ratio is not None
            and boundary_margin is not None
            and boundary_i is not None
            and boundary_j is not None
            and euclidean_distance <= positive_euclidean_q20
            and geodesic_distance <= positive_geodesic_q20
            and ratio <= 1.1
            and (boundary_crossings or 0.0) <= 0.0
            and boundary_margin >= positive_margin_q50
            and (line_inside_fraction is None or line_inside_fraction >= 0.95)
            and boundary_i >= positive_width_q50
            and boundary_j >= positive_width_q50
        )
        easy_negative_v2 = int(
            connected == 0
            and euclidean_distance is not None
            and euclidean_distance >= negative_euclidean_q50
            and (boundary_crossings or 0.0) <= 1.0
            and (max_outside_segment_length is None or max_outside_segment_length >= negative_outside_q50)
            and (line_inside_fraction is None or line_inside_fraction <= negative_inside_q50)
        )
        is_hard_positive_v2 = int(
            connected == 1
            and geodesic_distance is not None
            and geodesic_distance >= positive_geodesic_q75
            and (
                (ratio is not None and ratio >= positive_ratio_q75)
                or (boundary_crossings is not None and boundary_crossings >= 2.0)
                or (boundary_margin is not None and boundary_margin <= positive_margin_q25)
            )
        )
        is_hard_negative_v2 = int(
            connected == 0
            and euclidean_distance is not None
            and euclidean_distance <= negative_euclidean_q50
            and (
                component_gap_distance is None
                or component_gap_distance <= negative_gap_q50
            )
            and (
                (max_outside_segment_length is not None and max_outside_segment_length <= negative_outside_q50)
                or (boundary_crossings is not None and boundary_crossings >= 1.0)
                or (line_inside_fraction is not None and line_inside_fraction >= negative_inside_q50)
            )
        )
        hard_positive_score = (
            _zscore_value(geodesic_distance, geodesic_mean, geodesic_std)
            + _zscore_value(ratio, ratio_mean, ratio_std)
            + _zscore_value(boundary_crossings, crossing_mean, crossing_std)
            - 0.5 * _zscore_value(euclidean_distance, euclidean_mean, euclidean_std)
        )
        near_miss_score = (
            -_zscore_value(euclidean_distance, euclidean_mean, euclidean_std)
            + _zscore_value(boundary_crossings, crossing_mean, crossing_std)
            + _zscore_value(line_inside_fraction, inside_mean, inside_std)
            - _zscore_value(component_gap_distance, gap_mean, gap_std)
            - _zscore_value(max_outside_segment_length, outside_mean, outside_std)
        )
        hard_pair_score = hard_positive_score if connected == 1 else near_miss_score
        hard_pair_score_v2 = (
            hard_positive_score + 3.0 * is_hard_positive_v2
            if connected == 1
            else near_miss_score + 3.0 * is_hard_negative_v2
        )
        record["difficulty_proxy"] = round(proxy, 6)
        record["difficulty_bucket"] = _difficulty_bucket(
            euclidean_distance=euclidean_distance,
            geodesic_distance=geodesic_distance,
            ratio=ratio,
            boundary_crossings=boundary_crossings,
            euclidean_q33=euclidean_q33,
            euclidean_q66=euclidean_q66,
            geodesic_q33=geodesic_q33,
            geodesic_q66=geodesic_q66,
            ratio_q75=ratio_q75,
            crossing_q75=crossing_q75,
        )
        record["is_easy_positive"] = easy_positive
        record["is_easy_negative"] = easy_negative
        record["easy_positive_v2"] = easy_positive_v2
        record["easy_negative_v2"] = easy_negative_v2
        record["is_hard_positive_v2"] = is_hard_positive_v2
        record["is_hard_negative_v2"] = is_hard_negative_v2
        record["near_miss_score"] = round(near_miss_score, 6)
        record["hard_pair_score"] = round(hard_pair_score, 6)
        record["hard_pair_score_v2"] = round(hard_pair_score_v2, 6)
    return records


def summarize_pairs(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (_threshold_tag(float(record["valid_threshold"])), str(record["sampling_method"]))
        grouped.setdefault(key, []).append(record)

    summary_records = []
    for (threshold_tag, method), subset in sorted(grouped.items()):
        ratio_values = [_to_float(record["geodesic_euclidean_ratio"]) for record in subset if _to_float(record["geodesic_euclidean_ratio"]) is not None]
        geodesic_values = [_to_float(record["geodesic_distance"]) for record in subset if _to_float(record["geodesic_distance"]) is not None]
        euclidean_values = [_to_float(record["euclidean_distance"]) for record in subset if _to_float(record["euclidean_distance"]) is not None]
        crossings = [_to_float(record["boundary_crossing_count"]) for record in subset if _to_float(record["boundary_crossing_count"]) is not None]
        summary_records.append(
            {
                "valid_threshold": threshold_tag,
                "sampling_method": method,
                "n_pairs": len(subset),
                "n_images": len({record["image_id"] for record in subset}),
                "connected_pairs": sum(int(record.get("connected", 0)) for record in subset),
                "disconnected_pairs": sum(1 - int(record.get("connected", 0)) for record in subset),
                "foreground_foreground_pairs": sum(int(record.get("pair_region_type") == "foreground_foreground") for record in subset),
                "background_background_pairs": sum(int(record.get("pair_region_type") == "background_background") for record in subset),
                "cross_region_pairs": sum(int(record.get("pair_region_type") == "cross_region") for record in subset),
                "mean_euclidean_distance": round(_safe_mean(euclidean_values), 6),
                "mean_geodesic_distance": round(_safe_mean(geodesic_values), 6),
                "mean_geodesic_euclidean_ratio": round(_safe_mean(ratio_values), 6),
                "mean_boundary_crossing_count": round(_safe_mean(crossings), 6),
                "hard_pairs": sum(int(record.get("difficulty_bucket") in {"Hard", "Very hard"}) for record in subset),
                "very_hard_pairs": sum(int(record.get("difficulty_bucket") == "Very hard") for record in subset),
            }
        )
    return summary_records


def summarize_feature_coverage(records: list[dict], n_bins: int = 4) -> list[dict]:
    grouped_by_threshold: dict[str, list[dict]] = {}
    for record in records:
        grouped_by_threshold.setdefault(_threshold_tag(float(record["valid_threshold"])), []).append(record)

    summary_rows = []
    for threshold_tag, threshold_records in sorted(grouped_by_threshold.items()):
        all_edges = _quantile_edges(threshold_records, "euclidean_distance", n_bins=n_bins)
        connected_records = [record for record in threshold_records if int(record.get("connected", 0)) == 1]
        disconnected_records = [record for record in threshold_records if int(record.get("connected", 0)) == 0]
        geodesic_edges = _quantile_edges(connected_records, "geodesic_distance", n_bins=n_bins)
        ratio_edges = _quantile_edges(connected_records, "geodesic_euclidean_ratio", n_bins=n_bins)
        margin_edges = _quantile_edges(disconnected_records, "distance_to_boundary_min", n_bins=n_bins)
        inside_edges = _quantile_edges(disconnected_records, "line_inside_fraction", n_bins=n_bins)

        for method in sorted({str(record["sampling_method"]) for record in threshold_records}):
            subset = [record for record in threshold_records if str(record["sampling_method"]) == method]
            summary_rows.extend(
                [
                    _coverage_summary_row(
                        subset=subset,
                        threshold_tag=threshold_tag,
                        method=method,
                        subset_name="all_pairs_2d",
                        total_bins=n_bins * 4,
                        key_builder=lambda record: (
                            _assign_quantile_bin(record.get("euclidean_distance"), all_edges),
                            _boundary_crossing_bin(record.get("boundary_crossing_count")),
                        ),
                    ),
                    _coverage_summary_row(
                        subset=[record for record in subset if int(record.get("connected", 0)) == 1],
                        threshold_tag=threshold_tag,
                        method=method,
                        subset_name="connected_only_4d",
                        total_bins=n_bins * 4 * n_bins * n_bins,
                        key_builder=lambda record: (
                            _assign_quantile_bin(record.get("euclidean_distance"), all_edges),
                            _boundary_crossing_bin(record.get("boundary_crossing_count")),
                            _assign_quantile_bin(record.get("geodesic_distance"), geodesic_edges),
                            _assign_quantile_bin(record.get("geodesic_euclidean_ratio"), ratio_edges),
                        ),
                    ),
                    _coverage_summary_row(
                        subset=[record for record in subset if int(record.get("connected", 0)) == 0],
                        threshold_tag=threshold_tag,
                        method=method,
                        subset_name="disconnected_only_4d",
                        total_bins=n_bins * 4 * n_bins * n_bins,
                        key_builder=lambda record: (
                            _assign_quantile_bin(record.get("euclidean_distance"), all_edges),
                            _boundary_crossing_bin(record.get("boundary_crossing_count")),
                            _assign_quantile_bin(record.get("distance_to_boundary_min"), margin_edges),
                            _assign_quantile_bin(record.get("line_inside_fraction"), inside_edges),
                        ),
                    ),
                ]
            )
    return [row for row in summary_rows if row["n_pairs"] > 0]


def summarize_difficulty_buckets(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (_threshold_tag(float(record["valid_threshold"])), str(record["sampling_method"]))
        grouped.setdefault(key, []).append(record)
    rows = []
    for (threshold_tag, method), subset in sorted(grouped.items()):
        row = {
            "valid_threshold": threshold_tag,
            "sampling_method": method,
            "Easy": 0,
            "Medium": 0,
            "Hard": 0,
            "Very hard": 0,
        }
        for record in subset:
            bucket = str(record.get("difficulty_bucket", "Medium"))
            if bucket not in row:
                row[bucket] = 0
            row[bucket] += 1
        rows.append(row)
    return rows


def pair_feature_triplet_records(records: list[dict]) -> list[dict]:
    triplet_records = []
    for record in records:
        geodesic_value = _to_float(record.get("geodesic_distance"))
        triplet_records.append(
            {
                "pair_id": record["pair_id"],
                "image_id": record["image_id"],
                "sampling_method": record["sampling_method"],
                "valid_threshold": record["valid_threshold"],
                "connected": int(record.get("connected", 0)),
                "pair_region_type": record.get("pair_region_type", ""),
                "feature_triplet": [
                    float(record["euclidean_distance"]),
                    geodesic_value,
                    int(record["boundary_crossing_count"]),
                ],
            }
        )
    return triplet_records


def save_mask_preview(mask: np.ndarray, out_path: str | Path, title: str) -> None:
    canvas = _base_canvas(mask)
    _draw_title(canvas, title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_distance_preview(mask: np.ndarray, distance_map: np.ndarray, out_path: str | Path, title: str) -> None:
    max_distance = float(distance_map.max()) if float(distance_map.max()) > 0 else 1.0
    scaled = np.clip(distance_map / max_distance, 0.0, 1.0)
    red = (scaled * 255).astype(np.uint8)
    green = (np.sqrt(scaled) * 220).astype(np.uint8)
    blue = np.where(mask, 40, 8).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~mask] = np.array([10, 10, 10], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 22), color=(18, 18, 18))
    canvas.paste(image, (0, 22))
    _draw_title(canvas, title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_valid_mask_preview(mask: np.ndarray, valid_mask: np.ndarray, out_path: str | Path, title: str) -> None:
    base = _mask_rgb(mask)
    base[valid_mask] = np.array([80, 200, 120], dtype=np.uint8)
    image = Image.fromarray(base, mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 22), color=(18, 18, 18))
    canvas.paste(image, (0, 22))
    _draw_title(canvas, title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_points_preview(
    mask: np.ndarray,
    valid_mask: np.ndarray,
    points: list[tuple[int, int]] | list[dict],
    out_path: str | Path,
    title: str,
) -> None:
    image = _overlay_points(mask, valid_mask, points)
    canvas = Image.new("RGB", (image.width, image.height + 34), color=(18, 18, 18))
    canvas.paste(image, (0, 34))
    _draw_title(canvas, title)
    _draw_points_legend(canvas, origin=(6, 18))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_pair_contact_sheet(
    records: list[dict],
    mask_lookup: dict[str, np.ndarray],
    out_path: str | Path,
    title: str,
    sort_key: str,
    reverse: bool = True,
    count: int = 12,
    columns: int = 4,
) -> None:
    chosen = sorted(records, key=lambda record: _sortable_metric(record, sort_key), reverse=reverse)[:count]
    if not chosen:
        return
    font = ImageFont.load_default()
    tiles = []
    for record in chosen:
        mask = mask_lookup[str(record["image_id"])]
        start = (int(record["row_i"]), int(record["col_i"]))
        end = (int(record["row_j"]), int(record["col_j"]))
        path_points = shortest_path_points(mask, start, end) if _to_float(record.get("geodesic_distance")) is not None else []
        image = _overlay_pair(mask, start, end, path_points)
        tile = Image.new("RGB", (image.width, image.height + 32), color=(16, 16, 16))
        tile.paste(image, (0, 0))
        draw = ImageDraw.Draw(tile)
        caption = (
            f"{record['sampling_method']} t={record['valid_threshold']} "
            f"r={_fmt_metric(record.get('geodesic_euclidean_ratio'))} "
            f"b={record.get('boundary_crossing_count', '-')}"
        )
        draw.text((4, image.height + 4), caption, fill=(240, 240, 240), font=font)
        tiles.append(tile)

    rows = (len(tiles) + columns - 1) // columns
    tile_width, tile_height = tiles[0].size
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height + 22), color=(12, 12, 12))
    _draw_title(canvas, title)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        canvas.paste(tile, (col * tile_width, row * tile_height + 22))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_distribution_plots(records: list[dict], summary_records: list[dict], out_dir: str | Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib is not installed; histogram/scatter outputs were skipped."]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    metric_specs = [
        ("euclidean_distance", "Euclidean Distance"),
        ("geodesic_euclidean_ratio", "Geodesic / Euclidean Ratio"),
        ("boundary_crossing_count", "Boundary Crossing Count"),
        ("geodesic_distance", "Geodesic Distance"),
    ]
    methods = sorted({str(record["sampling_method"]) for record in records})

    for metric_key, metric_title in metric_specs:
        plt.figure(figsize=(8, 5))
        for method in methods:
            values = [_to_float(record.get(metric_key)) for record in records if str(record["sampling_method"]) == method]
            values = [value for value in values if value is not None]
            if not values:
                continue
            plt.hist(values, bins=20, alpha=0.45, label=method)
        plt.title(metric_title)
        plt.xlabel(metric_title)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_key}_hist.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 6))
    for method in methods:
        subset = [record for record in records if str(record["sampling_method"]) == method]
        xs = [_to_float(record.get("euclidean_distance")) for record in subset]
        ys = [_to_float(record.get("geodesic_distance")) for record in subset]
        points = [(x_value, y_value) for x_value, y_value in zip(xs, ys) if x_value is not None and y_value is not None]
        if not points:
            continue
        plt.scatter([point[0] for point in points], [point[1] for point in points], s=18, alpha=0.5, label=method)
    plt.title("Euclidean vs Geodesic Distance")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Geodesic Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "euclidean_vs_geodesic.png", dpi=160)
    plt.close()

    if summary_records:
        plt.figure(figsize=(9, 5))
        x_labels = [f"{record['sampling_method']}\nT={record['valid_threshold']}" for record in summary_records]
        y_values = [float(record["mean_geodesic_euclidean_ratio"]) for record in summary_records]
        plt.bar(x_labels, y_values, color="#4c78a8")
        plt.title("Mean Geodesic/Euclidean Ratio by Method")
        plt.ylabel("Mean Ratio")
        plt.tight_layout()
        plt.savefig(out_dir / "method_ratio_summary.png", dpi=160)
        plt.close()

    return notes


def save_feature_coverage_heatmap(coverage_summary: list[dict], out_path: str | Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [row for row in coverage_summary if row.get("n_pairs", 0) > 0]
    if not rows:
        return
    row_labels = sorted({f"{row['sampling_method']} | T={row['valid_threshold']}" for row in rows})
    col_labels = sorted({str(row["subset"]) for row in rows})
    matrix = np.zeros((len(row_labels), len(col_labels)), dtype=np.float64)
    row_index = {label: index for index, label in enumerate(row_labels)}
    col_index = {label: index for index, label in enumerate(col_labels)}
    for row in rows:
        matrix[row_index[f"{row['sampling_method']} | T={row['valid_threshold']}"], col_index[str(row["subset"])]] = float(row["coverage_ratio"])

    plt.figure(figsize=(max(6, len(col_labels) * 2.2), max(4, len(row_labels) * 0.5 + 2)))
    plt.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks(range(len(col_labels)), col_labels, rotation=20, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.title("Feature Coverage Ratio Heatmap")
    plt.colorbar(label="Coverage ratio")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_experiment_report(
    out_path: str | Path,
    config_path: str,
    dataset_summary: list[dict],
    threshold_summary: list[dict],
    pair_summary: list[dict],
    feature_coverage_summary: list[dict],
    difficulty_bucket_summary: list[dict],
    pair_records: list[dict],
    plot_notes: list[str],
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hard_counts = {}
    for record in pair_records:
        bucket = str(record.get("difficulty_bucket", ""))
        hard_counts[bucket] = hard_counts.get(bucket, 0) + 1

    lines = [
        "# Region Pair Sampling Experiment Report",
        "",
        f"- config: `{config_path}`",
        f"- images: `{len(dataset_summary)}`",
        f"- pairs: `{len(pair_records)}`",
        "",
        "## Dataset Summary",
        "",
        "| image_id | size | components | foreground_fraction |",
        "| --- | --- | --- | --- |",
    ]
    for record in dataset_summary[:12]:
        size_text = f"{record['height']}x{record['width']}"
        lines.append(
            f"| {record['image_id']} | {size_text} | {record['n_components']} | {record['foreground_fraction']} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Summary",
            "",
            "| image_id | region | threshold | valid_pixels | valid_fraction |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in threshold_summary[:24]:
        lines.append(
            f"| {record['image_id']} | {record.get('region_name', '-') } | {record['valid_threshold']} | {record['valid_pixels']} | {record['valid_fraction']} |"
        )

    lines.extend(
        [
            "",
            "## Pair Summary",
            "",
            "| threshold | method | n_pairs | connected | disconnected | mean_ratio | mean_boundary_crossing_count | hard_pairs |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in pair_summary:
        lines.append(
            f"| {record['valid_threshold']} | {record['sampling_method']} | {record['n_pairs']} | "
            f"{record['connected_pairs']} | {record['disconnected_pairs']} | "
            f"{record['mean_geodesic_euclidean_ratio']} | {record['mean_boundary_crossing_count']} | {record['hard_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## Feature Coverage Summary",
            "",
            "| threshold | method | subset | coverage_ratio | entropy | occupied_bins | total_bins |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in feature_coverage_summary:
        lines.append(
            f"| {record['valid_threshold']} | {record['sampling_method']} | {record['subset']} | "
            f"{record['coverage_ratio']} | {record['entropy']} | {record['occupied_bins']} | {record['total_bins']} |"
        )

    lines.extend(
        [
            "",
            "## Difficulty Bucket Summary By Method",
            "",
            "| threshold | method | Easy | Medium | Hard | Very hard |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in difficulty_bucket_summary:
        lines.append(
            f"| {record['valid_threshold']} | {record['sampling_method']} | "
            f"{record.get('Easy', 0)} | {record.get('Medium', 0)} | {record.get('Hard', 0)} | {record.get('Very hard', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Difficulty Buckets",
            "",
            "| bucket | count |",
            "| --- | --- |",
        ]
    )
    for bucket in ["Easy", "Medium", "Hard", "Very hard"]:
        lines.append(f"| {bucket} | {hard_counts.get(bucket, 0)} |")

    if plot_notes:
        lines.extend(["", "## Notes", ""])
        for note in plot_notes:
            lines.append(f"- {note}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_pool(candidates: np.ndarray, candidate_pool_limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) <= candidate_pool_limit:
        return candidates
    chosen_indices = rng.choice(len(candidates), size=candidate_pool_limit, replace=False)
    return candidates[np.sort(chosen_indices)]


def _normalize_point_entry(
    point,
    mask: np.ndarray,
    foreground_distance_map: np.ndarray,
    background_distance_map: np.ndarray,
) -> dict:
    if isinstance(point, dict):
        row = int(point["row"])
        col = int(point["col"])
        region_value = int(point.get("region_value", int(bool(mask[row, col]))))
        distance_to_boundary = point.get("distance_to_boundary")
        if distance_to_boundary in (None, "", "NA", "None"):
            distance_to_boundary = (
                float(foreground_distance_map[row, col])
                if region_value == 1
                else float(background_distance_map[row, col])
            )
        region_name = str(point.get("region_name", _region_name(region_value)))
        return {
            "row": row,
            "col": col,
            "region_value": region_value,
            "region_name": region_name,
            "distance_to_boundary": float(distance_to_boundary),
        }

    row, col = int(point[0]), int(point[1])
    region_value = int(bool(mask[row, col]))
    distance_to_boundary = float(foreground_distance_map[row, col]) if region_value == 1 else float(background_distance_map[row, col])
    return {
        "row": row,
        "col": col,
        "region_value": region_value,
        "region_name": _region_name(region_value),
        "distance_to_boundary": distance_to_boundary,
    }


def _farthest_point_sampling(candidates: np.ndarray, n_points: int, seed_index: int = 0) -> list[tuple[int, int]]:
    seed_index = max(0, min(seed_index, len(candidates) - 1))
    selected_indices = [seed_index]
    selected_points = candidates[selected_indices].astype(np.float64)
    candidate_points = candidates.astype(np.float64)
    min_distances = np.linalg.norm(candidate_points - selected_points[0], axis=1)
    while len(selected_indices) < n_points:
        next_index = int(np.argmax(min_distances))
        if next_index in selected_indices:
            break
        selected_indices.append(next_index)
        new_distances = np.linalg.norm(candidate_points - candidate_points[next_index], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return [tuple(int(value) for value in candidates[index]) for index in selected_indices]


def _feature_diversity_sampling(candidates: np.ndarray, distance_map: np.ndarray, n_points: int) -> list[tuple[int, int]]:
    height, width = distance_map.shape
    distance_values = distance_map[candidates[:, 0], candidates[:, 1]]
    max_distance = float(distance_values.max()) if len(distance_values) else 1.0
    features = np.column_stack(
        [
            candidates[:, 0] / max(1, height - 1),
            candidates[:, 1] / max(1, width - 1),
            distance_values / max(1e-6, max_distance),
        ]
    )
    seed_index = int(np.argmax(distance_values))
    selected_indices = [seed_index]
    min_distances = np.linalg.norm(features - features[seed_index], axis=1)
    while len(selected_indices) < n_points:
        next_index = int(np.argmax(min_distances))
        if next_index in selected_indices:
            break
        selected_indices.append(next_index)
        new_distances = np.linalg.norm(features - features[next_index], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return [tuple(int(value) for value in candidates[index]) for index in selected_indices]


def _geodesic_farthest_point_sampling(
    candidates: np.ndarray,
    valid_mask: np.ndarray,
    distance_map: np.ndarray,
    n_points: int,
) -> list[tuple[int, int]]:
    seed_index = _seed_index(candidates, distance_map)
    selected_indices = [seed_index]
    min_distances = np.full(len(candidates), np.inf, dtype=np.float64)

    while len(selected_indices) < n_points:
        last_point = tuple(int(value) for value in candidates[selected_indices[-1]])
        geodesic_map = geodesic_distance_map(valid_mask, last_point)
        candidate_distances = geodesic_map[candidates[:, 0], candidates[:, 1]]
        min_distances = np.minimum(min_distances, candidate_distances)
        min_distances[selected_indices] = -1.0
        next_index = int(np.argmax(min_distances))
        if min_distances[next_index] <= 0 or next_index in selected_indices:
            break
        selected_indices.append(next_index)
    return [tuple(int(value) for value in candidates[index]) for index in selected_indices]


def _seed_index(candidates: np.ndarray, distance_map: np.ndarray) -> int:
    values = distance_map[candidates[:, 0], candidates[:, 1]]
    return int(np.argmax(values))


def _keep_large_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1:
        return mask.astype(bool)
    labels, component_sizes = connected_components(mask.astype(bool))
    keep_ids = {index for index, size in enumerate(component_sizes) if size >= min_component_area}
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, list(keep_ids))


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0, 0, 0, 0
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    return int(row_min), int(col_min), int(row_max), int(col_max)


def _distance_transform_fallback(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    distances = np.full(mask.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col]:
                distances[row, col] = 0.0
                heapq.heappush(heap, (0.0, row, col))
    if not heap:
        return distances
    while heap:
        cost, row, col = heapq.heappop(heap)
        if cost > distances[row, col]:
            continue
        for delta_row, delta_col, weight in _NEIGHBORS_8:
            next_row = row + delta_row
            next_col = col + delta_col
            if not _is_inside(mask.shape, next_row, next_col):
                continue
            next_cost = cost + weight
            if next_cost < distances[next_row, next_col]:
                distances[next_row, next_col] = next_cost
                heapq.heappush(heap, (next_cost, next_row, next_col))
    return distances


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = np.array([238, 238, 238], dtype=np.uint8)
    rgb[~mask] = np.array([18, 18, 18], dtype=np.uint8)
    return rgb


def _base_canvas(mask: np.ndarray) -> Image.Image:
    image = Image.fromarray(_mask_rgb(mask), mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 22), color=(18, 18, 18))
    canvas.paste(image, (0, 22))
    return canvas


def _overlay_points(mask: np.ndarray, valid_mask: np.ndarray, points: list[tuple[int, int]] | list[dict]) -> Image.Image:
    image = Image.fromarray(_mask_rgb(mask), mode="RGB")
    draw = ImageDraw.Draw(image)
    overlay = np.array(image)
    overlay[valid_mask] = np.array([90, 160, 110], dtype=np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    for point in points:
        if isinstance(point, dict):
            row = int(point["row"])
            col = int(point["col"])
            region_value = int(point.get("region_value", int(bool(mask[row, col]))))
        else:
            row = int(point[0])
            col = int(point[1])
            region_value = int(bool(mask[row, col]))
        color = (245, 72, 72) if region_value == 1 else (72, 144, 245)
        draw.ellipse((col - 2, row - 2, col + 2, row + 2), fill=color, outline=(255, 255, 255))
    return image


def _overlay_pair(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    path_points: list[tuple[int, int]],
) -> Image.Image:
    image = Image.fromarray(_mask_rgb(mask), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.line((start[1], start[0], end[1], end[0]), fill=(70, 140, 250), width=1)
    if path_points:
        flattened = [(point[1], point[0]) for point in path_points]
        draw.line(flattened, fill=(255, 160, 0), width=2)
    for row, col, color in [(start[0], start[1], (240, 60, 60)), (end[0], end[1], (40, 210, 120))]:
        draw.ellipse((col - 2, row - 2, col + 2, row + 2), fill=color, outline=(255, 255, 255))
    return image


def _draw_title(image: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.text((6, 5), title, fill=(255, 255, 255), font=ImageFont.load_default())


def _draw_points_legend(image: Image.Image, origin: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0 = origin
    draw.text((x0, y0), "Red point = foreground, Blue point = background", fill=(230, 230, 230), font=font)


def _threshold_tag(value: float) -> str:
    if int(value) == value:
        return str(int(value))
    return str(value).replace(".", "p")


def _region_name(region_value: int) -> str:
    return "foreground" if int(region_value) == 1 else "background"


def _pair_region_type(region_value_i: int, region_value_j: int) -> str:
    if int(region_value_i) == 1 and int(region_value_j) == 1:
        return "foreground_foreground"
    if int(region_value_i) == 0 and int(region_value_j) == 0:
        return "background_background"
    return "cross_region"


def _is_inside(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _to_float(value) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    return float(value)


def _collect_numeric(records: list[dict], key: str) -> list[float]:
    values = [_to_float(record.get(key)) for record in records]
    return [value for value in values if value is not None]


def _quantile_edges(records: list[dict], key: str, n_bins: int) -> np.ndarray:
    values = _collect_numeric(records, key)
    if not values:
        return np.asarray([], dtype=np.float64)
    edges = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0.0, 1.0, n_bins + 1))
    return _strictly_increasing_edges(edges)


def _assign_quantile_bin(value, edges: np.ndarray) -> int | None:
    numeric = _to_float(value)
    if numeric is None or len(edges) < 2:
        return None
    index = int(np.searchsorted(edges, float(numeric), side="right") - 1)
    return max(0, min(len(edges) - 2, index))


def _coverage_summary_row(
    subset: list[dict],
    threshold_tag: str,
    method: str,
    subset_name: str,
    total_bins: int,
    key_builder,
) -> dict:
    if not subset:
        return {
            "valid_threshold": threshold_tag,
            "sampling_method": method,
            "subset": subset_name,
            "n_pairs": 0,
            "occupied_bins": 0,
            "total_bins": total_bins,
            "coverage_ratio": 0.0,
            "entropy": 0.0,
            "min_bin_count": 0,
            "max_bin_count": 0,
            "mean_bin_count": 0.0,
        }
    counts: dict[tuple, int] = {}
    for record in subset:
        key = key_builder(record)
        if any(value is None for value in key):
            continue
        counts[key] = counts.get(key, 0) + 1
    occupied_bins = len(counts)
    probabilities = [count / max(1, len(subset)) for count in counts.values() if count > 0]
    entropy = 0.0
    if probabilities and total_bins > 1:
        entropy = -sum(probability * math.log(probability + 1e-12) for probability in probabilities) / math.log(total_bins)
    zero_bins = max(0, total_bins - occupied_bins)
    min_bin_count = 0 if zero_bins > 0 else min(counts.values(), default=0)
    max_bin_count = max(counts.values(), default=0)
    mean_bin_count = len(subset) / max(1, total_bins)
    return {
        "valid_threshold": threshold_tag,
        "sampling_method": method,
        "subset": subset_name,
        "n_pairs": len(subset),
        "occupied_bins": occupied_bins,
        "total_bins": total_bins,
        "coverage_ratio": round(occupied_bins / max(1, total_bins), 6),
        "entropy": round(entropy, 6),
        "min_bin_count": int(min_bin_count),
        "max_bin_count": int(max_bin_count),
        "mean_bin_count": round(mean_bin_count, 6),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    array = np.asarray(values, dtype=np.float64)
    std_value = float(array.std())
    return float(array.mean()), std_value if std_value > 0 else 1.0


def _quantiles(values: list[float], quantiles: list[float]) -> list[float]:
    if not values:
        return [0.0 for _ in quantiles]
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, quantile)) for quantile in quantiles]


def _zscore_value(value: float | None, mean_value: float, std_value: float) -> float:
    if value is None:
        return 0.0
    return (value - mean_value) / std_value


def _difficulty_bucket(
    euclidean_distance: float | None,
    geodesic_distance: float | None,
    ratio: float | None,
    boundary_crossings: float | None,
    euclidean_q33: float,
    euclidean_q66: float,
    geodesic_q33: float,
    geodesic_q66: float,
    ratio_q75: float,
    crossing_q75: float,
) -> str:
    if (
        euclidean_distance is not None
        and geodesic_distance is not None
        and ratio is not None
        and euclidean_distance <= euclidean_q33
        and geodesic_distance >= geodesic_q66
        and ratio >= ratio_q75
        and boundary_crossings is not None
        and boundary_crossings >= max(1.0, crossing_q75)
    ):
        return "Very hard"
    if (
        euclidean_distance is not None
        and geodesic_distance is not None
        and ratio is not None
        and euclidean_distance <= euclidean_q66
        and geodesic_distance >= geodesic_q66
        and ratio >= ratio_q75
    ):
        return "Hard"
    if (
        euclidean_distance is not None
        and geodesic_distance is not None
        and euclidean_distance <= euclidean_q33
        and geodesic_distance <= geodesic_q33
    ):
        return "Easy"
    return "Medium"


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _sortable_metric(record: dict, key: str) -> float:
    value = _to_float(record.get(key))
    if value is None:
        return float("-inf")
    return value


def _fmt_metric(value) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.2f}"


def _strictly_increasing_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    unique_values = [float(values[0])]
    for value in values[1:]:
        numeric = float(value)
        if numeric <= unique_values[-1]:
            numeric = unique_values[-1] + 1e-6
        unique_values.append(numeric)
    return np.asarray(unique_values, dtype=np.float64)


_NEIGHBORS_8 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
]
