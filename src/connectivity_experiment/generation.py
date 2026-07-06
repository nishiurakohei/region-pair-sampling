from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np

from connectivity_experiment.connectivity_oracles import pixel_connected
from connectivity_experiment.generate_curves import (
    choose_weighted_key,
    curve_length,
    curvature_stats,
    generate_bezier_bend_curve,
    generate_c_shape_curve,
    generate_straight_curve,
    max_deviation_from_segment,
    resample_polyline,
    self_proximity,
    total_turning_angle,
)
from connectivity_experiment.geometry import Circle, build_adjacency, component_labels, edge_margin, radius_stats, summarize_graph
from connectivity_experiment.render_shapes import (
    bottleneck_radii,
    build_multichannel_image,
    circles_from_curve_points,
    image_from_array,
    random_noise_circles,
    render_circles,
)


def generate_sample_pair(config: dict, split: str, pair_index: int, base_seed: int) -> tuple[dict, dict]:
    dataset_cfg = config["dataset"]
    rng = random.Random(base_seed + pair_index * 10007 + _split_seed_offset(split))
    shape_families = dataset_cfg.get("shape_families", {"straight": 1.0})
    negative_modes = dataset_cfg.get("negative_modes", {"break_bottleneck": 1.0})

    for _ in range(200):
        shape_family = choose_weighted_key(rng, shape_families)
        negative_mode = choose_weighted_key(rng, negative_modes)
        positive_base = _generate_family_positive(config, rng, split, pair_index, shape_family)
        if positive_base is None:
            continue

        negative_base = _generate_negative_variant(config, rng, positive_base, negative_mode)
        if negative_base is None:
            continue

        positive_circles, negative_circles, extra_counts = _augment_with_shared_extras(config, rng, positive_base, negative_base)
        positive = _build_sample_record(
            config=config,
            circles=positive_circles,
            base=positive_base,
            split=split,
            sample_id=f"{split}_{pair_index:05d}_pos",
            connected=1,
            negative_mode="",
            generation_mode="curved_v1_positive",
            extra_counts=extra_counts,
            sample_seed=base_seed + pair_index * 2,
        )
        negative = _build_sample_record(
            config=config,
            circles=negative_circles,
            base=negative_base,
            split=split,
            sample_id=f"{split}_{pair_index:05d}_neg",
            connected=0,
            negative_mode=negative_mode,
            generation_mode="curved_v1_negative",
            extra_counts=extra_counts,
            sample_seed=base_seed + pair_index * 2 + 1,
        )
        return positive, negative
    raise RuntimeError(f"Failed to generate curved sample pair for split={split}, pair_index={pair_index}")


def save_sample(sample: dict, raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    split_dir = raw_dir / sample["split"]
    image_dir = split_dir / "images"
    geometry_dir = split_dir / "geometry"
    image_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{sample['sample_id']}.png"
    geometry_path = geometry_dir / f"{sample['sample_id']}.json"
    sample["image"].save(image_path)

    geometry_payload = sample.get("geometry_payload")
    if geometry_payload is None:
        geometry_payload = {
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "connected": sample["connected"],
            "shape_family": sample["metadata"]["shape_family"],
            "curve_points": [[round(point[0], 6), round(point[1], 6)] for point in sample["curve_points"]],
            "circles": [circle.to_dict() for circle in sample["circles"]],
            "latent_path": sample["latent_path"],
        }
    geometry_path.write_text(json.dumps(geometry_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    record = dict(sample["metadata"])
    record["image_path"] = image_path.relative_to(raw_dir.parent.parent).as_posix()
    record["geometry_path"] = geometry_path.relative_to(raw_dir.parent.parent).as_posix()
    return record


def _split_seed_offset(split: str) -> int:
    return {"train": 11, "val": 23, "test": 37}.get(split, 101)


def _generate_family_positive(config: dict, rng: random.Random, split: str, pair_index: int, shape_family: str) -> dict | None:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    target_distance_range = tuple(dataset_cfg["target_distance_range"])
    tube_width_range = tuple(dataset_cfg.get("tube_width_range", dataset_cfg["radius_range"]))

    for _ in range(120):
        if shape_family == "straight":
            point_a, point_b = _sample_target_pair(rng, image_size, target_distance_range, margin=tube_width_range[1] + 8.0)
            coarse_points = generate_straight_curve(point_a, point_b, n_points=24)
            family_params = {
                "curve_type": "straight",
                "bend_strength": 0.0,
                "arc_angle": 0.0,
                "arc_radius": 0.0,
            }
        elif shape_family == "bezier_bend":
            point_a, point_b = _sample_target_pair(rng, image_size, target_distance_range, margin=tube_width_range[1] + 16.0)
            bend_range = tuple(dataset_cfg.get("bend_strength_range", [10, 40]))
            bend_strength = rng.uniform(max(6.0, bend_range[0]), bend_range[1]) * rng.choice([-1.0, 1.0])
            coarse_points, control = generate_bezier_bend_curve(point_a, point_b, bend_strength=bend_strength, n_points=28)
            family_params = {
                "curve_type": "bezier_bend",
                "bend_strength": bend_strength,
                "arc_angle": 0.0,
                "arc_radius": 0.0,
                "control_point_x": round(control[0], 6),
                "control_point_y": round(control[1], 6),
            }
        elif shape_family == "c_shape":
            c_shape = _sample_c_shape(rng, image_size, target_distance_range, dataset_cfg)
            if c_shape is None:
                continue
            point_a, point_b, coarse_points, family_params = c_shape
        else:
            raise ValueError(f"Unsupported shape family: {shape_family}")

        local_width_min = rng.uniform(tube_width_range[0], max(tube_width_range[0], tube_width_range[1] - 0.8))
        local_width_max = rng.uniform(max(local_width_min + 0.2, tube_width_range[0]), tube_width_range[1])
        dense_points = resample_polyline(coarse_points, max_step=max(1.0, local_width_min * 0.55))
        radii = bottleneck_radii(len(dense_points), local_width_min, local_width_max, shape_family, rng)

        if not _curve_inside_image(dense_points, radii, image_size):
            continue

        circles, latent_path = circles_from_curve_points(dense_points, radii)
        graph_summary = summarize_graph(circles, 0, len(latent_path) - 1)
        if graph_summary["connected"] != 1:
            continue

        curve_metrics = _curve_metrics(dense_points, point_a, point_b)
        return {
            "shape_family": shape_family,
            "curve_points": dense_points,
            "point_a": dense_points[0],
            "point_b": dense_points[-1],
            "circles": circles,
            "latent_path": latent_path,
            "tube_width_min": local_width_min,
            "tube_width_max": local_width_max,
            "params": family_params,
            "curve_metrics": curve_metrics,
        }
    return None


def _sample_c_shape(
    rng: random.Random,
    image_size: int,
    target_distance_range: tuple[float, float],
    dataset_cfg: dict,
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]], dict] | None:
    tube_width_range = tuple(dataset_cfg.get("tube_width_range", [3, 8]))
    arc_angle_range = tuple(dataset_cfg.get("arc_angle_range", [120, 280]))
    arc_radius_range = tuple(dataset_cfg.get("arc_radius_range", [25, 60]))
    margin = arc_radius_range[1] + tube_width_range[1] + 4.0

    for _ in range(160):
        center = (rng.uniform(margin, image_size - margin), rng.uniform(margin, image_size - margin))
        arc_angle_deg = rng.uniform(max(120.0, arc_angle_range[0]), arc_angle_range[1])
        arc_radius = rng.uniform(arc_radius_range[0], arc_radius_range[1])
        chord_distance = 2.0 * arc_radius * math.sin(math.radians(arc_angle_deg) / 2.0)
        if not (target_distance_range[0] <= chord_distance <= target_distance_range[1]):
            continue
        rotation = rng.uniform(0.0, math.tau)
        theta_start = rotation - math.radians(arc_angle_deg) / 2.0
        theta_end = rotation + math.radians(arc_angle_deg) / 2.0
        points = generate_c_shape_curve(center, arc_radius, theta_start, theta_end, n_points=52)
        if not _points_inside(points, arc_radius_range[0] * 0.08 + tube_width_range[1], image_size):
            continue
        params = {
            "curve_type": "circular_arc",
            "bend_strength": 0.0,
            "arc_angle": arc_angle_deg,
            "arc_radius": arc_radius,
            "rotation_angle": math.degrees(rotation),
        }
        return points[0], points[-1], points, params
    return None


def _generate_negative_variant(config: dict, rng: random.Random, positive_base: dict, negative_mode: str) -> dict | None:
    circles = deepcopy(positive_base["circles"])
    latent_path = list(positive_base["latent_path"])
    image_size = int(config["dataset"]["image_size"])

    if negative_mode == "near_gap":
        negative_circles = _introduce_near_gap(rng, circles, latent_path, image_size)
    elif negative_mode == "break_bottleneck":
        negative_circles = _break_bottleneck(rng, circles, latent_path, image_size)
    else:
        negative_circles = None

    if negative_circles is None:
        return None
    summary = summarize_graph(negative_circles, 0, len(latent_path) - 1)
    if summary["connected"] != 0:
        return None
    if _is_pixel_connected(negative_circles, len(latent_path), image_size):
        return None

    negative_base = deepcopy(positive_base)
    negative_base["circles"] = negative_circles
    negative_base["point_a"] = (negative_circles[0].x, negative_circles[0].y)
    negative_base["point_b"] = (negative_circles[len(latent_path) - 1].x, negative_circles[len(latent_path) - 1].y)
    negative_base["curve_points"] = [(circle.x, circle.y) for circle in negative_circles[: len(latent_path)]]
    negative_base["curve_metrics"] = _curve_metrics(negative_base["curve_points"], negative_base["point_a"], negative_base["point_b"])
    return negative_base


def _augment_with_shared_extras(config: dict, rng: random.Random, positive_base: dict, negative_base: dict) -> tuple[list[Circle], list[Circle], dict]:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    radius_range = tuple(dataset_cfg["radius_range"])
    noise_range = tuple(dataset_cfg.get("noise_circles_range", dataset_cfg.get("n_noise_circles_range", [0, 20])))
    distractor_range = tuple(dataset_cfg.get("distractor_curves_range", [0, 0]))

    positive_circles = deepcopy(positive_base["circles"])
    negative_circles = deepcopy(negative_base["circles"])

    accepted_noise = 0
    target_noise = rng.randint(noise_range[0], noise_range[1])
    for candidate in random_noise_circles(rng, image_size, radius_range, target_noise):
        if _preserves_labels_with_append(positive_circles, negative_circles, [candidate], len(positive_base["latent_path"])):
            positive_circles.append(deepcopy(candidate))
            negative_circles.append(deepcopy(candidate))
            accepted_noise += 1

    accepted_distractors = 0
    target_distractors = rng.randint(distractor_range[0], distractor_range[1])
    for _ in range(target_distractors):
        distractor = _generate_distractor_curve(rng, config)
        if distractor and _preserves_labels_with_append(positive_circles, negative_circles, distractor, len(positive_base["latent_path"])):
            positive_circles.extend(deepcopy(distractor))
            negative_circles.extend(deepcopy(distractor))
            accepted_distractors += 1

    return positive_circles, negative_circles, {"n_noise_circles": accepted_noise, "n_distractor_curves": accepted_distractors}


def _generate_distractor_curve(rng: random.Random, config: dict) -> list[Circle] | None:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    radius_range = tuple(dataset_cfg["radius_range"])
    tube_width_range = tuple(dataset_cfg.get("tube_width_range", radius_range))
    family = rng.choice(["straight", "bezier_bend", "c_shape"])

    if family == "c_shape":
        c_shape = _sample_c_shape(rng, image_size, (16, 48), dataset_cfg | {"arc_angle_range": [120, 220], "arc_radius_range": [14, 28]})
        if c_shape is None:
            return None
        _, _, coarse_points, _ = c_shape
    else:
        point_a, point_b = _sample_target_pair(rng, image_size, (18, 48), margin=tube_width_range[1] + 8.0)
        if family == "straight":
            coarse_points = generate_straight_curve(point_a, point_b, n_points=12)
        else:
            bend_strength = rng.uniform(8.0, 24.0) * rng.choice([-1.0, 1.0])
            coarse_points, _ = generate_bezier_bend_curve(point_a, point_b, bend_strength=bend_strength, n_points=14)

    local_width_min = rng.uniform(max(1.2, tube_width_range[0] * 0.6), max(1.6, tube_width_range[1] * 0.7))
    local_width_max = rng.uniform(local_width_min, max(local_width_min + 0.2, tube_width_range[1]))
    dense_points = resample_polyline(coarse_points, max_step=max(1.0, local_width_min * 0.55))
    radii = bottleneck_radii(len(dense_points), local_width_min, local_width_max, family, rng)
    if not _curve_inside_image(dense_points, radii, image_size):
        return None
    circles, _ = circles_from_curve_points(dense_points, radii)
    for circle in circles:
        circle.kind = "noise"
        circle.latent_path_index = None
    return circles


def _introduce_near_gap(rng: random.Random, circles: list[Circle], latent_path: list[int], image_size: int) -> list[Circle] | None:
    ranked_edges = []
    for index in range(len(latent_path) - 1):
        left = circles[latent_path[index]]
        right = circles[latent_path[index + 1]]
        ranked_edges.append((edge_margin(left, right), index))
    ranked_edges.sort()

    for _, index in ranked_edges:
        for desired_gap in (1.5, 2.2, 3.0, 4.0):
            candidate = deepcopy(circles)
            left = candidate[latent_path[index]]
            right = candidate[latent_path[index + 1]]
            direction = (right.x - left.x, right.y - left.y)
            norm = math.hypot(direction[0], direction[1])
            if norm == 0:
                continue
            unit = (direction[0] / norm, direction[1] / norm)
            shift_amount = edge_margin(left, right) + desired_gap
            for path_index in latent_path[index + 1 :]:
                candidate[path_index].x += unit[0] * shift_amount
                candidate[path_index].y += unit[1] * shift_amount
            if not all(_inside_image((circle.x, circle.y), circle.radius, image_size) for circle in candidate[: len(latent_path)]):
                continue
            if summarize_graph(candidate, 0, len(latent_path) - 1)["connected"] == 0 and not _is_pixel_connected(candidate, len(latent_path), image_size):
                return candidate
    return None


def _break_bottleneck(rng: random.Random, circles: list[Circle], latent_path: list[int], image_size: int) -> list[Circle] | None:
    candidates = []
    for index in latent_path[1:-1]:
        left = circles[index - 1]
        center = circles[index]
        right = circles[index + 1]
        local_margin = min(edge_margin(left, center), edge_margin(center, right))
        candidates.append((center.radius, local_margin, index))
    candidates.sort()

    for _, _, index in candidates:
        for window_radius in (1, 2, 3):
            candidate = deepcopy(circles)
            start = max(1, index - window_radius)
            end = min(len(latent_path) - 2, index + window_radius)
            for current_index in range(start, end + 1):
                candidate[current_index].radius = 0.6
            if summarize_graph(candidate, 0, len(latent_path) - 1)["connected"] == 0 and not _is_pixel_connected(candidate, len(latent_path), image_size):
                return candidate
    return None


def _build_sample_record(
    config: dict,
    circles: list[Circle],
    base: dict,
    split: str,
    sample_id: str,
    connected: int,
    negative_mode: str,
    generation_mode: str,
    extra_counts: dict,
    sample_seed: int,
) -> dict:
    image_size = int(config["dataset"]["image_size"])
    latent_path = base["latent_path"]
    graph_summary = summarize_graph(circles, 0, len(latent_path) - 1)
    mean_radius, std_radius = radius_stats(circles)
    target_a = (circles[0].x, circles[0].y)
    target_b = (circles[len(latent_path) - 1].x, circles[len(latent_path) - 1].y)

    shape_channel = render_circles(circles, image_size=image_size)
    image_array = build_multichannel_image(shape_channel, target_a, target_b, image_size=image_size)
    image = image_from_array(image_array)

    connected_pixel_4 = int(pixel_connected(image_array, target_a, target_b, connectivity=4))
    connected_pixel_8 = int(pixel_connected(image_array, target_a, target_b, connectivity=8))
    label_disagreement = int(connected != connected_pixel_8)

    actual_path = graph_summary["shortest_path"] or latent_path
    min_bridge_radius = min(circles[index].radius for index in actual_path)
    area, perimeter, compactness, total_ink_pixels = _shape_stats(shape_channel)
    curve_metrics = base["curve_metrics"]

    metadata = {
        "sample_id": sample_id,
        "split": split,
        "connected": connected,
        "connected_geometry": connected,
        "connected_pixel_4": connected_pixel_4,
        "connected_pixel_8": connected_pixel_8,
        "label_disagreement": label_disagreement,
        "seed": sample_seed,
        "image_size": image_size,
        "shape_family": base["shape_family"],
        "curve_type": base["params"]["curve_type"],
        "generation_mode": generation_mode,
        "negative_mode": negative_mode,
        "n_circles": len(circles),
        "n_noise_circles": extra_counts["n_noise_circles"],
        "n_distractor_curves": extra_counts["n_distractor_curves"],
        "target_ax": round(target_a[0], 6),
        "target_ay": round(target_a[1], 6),
        "target_bx": round(target_b[0], 6),
        "target_by": round(target_b[1], 6),
        "target_distance": round(graph_summary["target_distance"], 6),
        "path_length": _optional_round(graph_summary["path_length"]),
        "tortuosity": _optional_round(graph_summary["tortuosity"]),
        "shortest_path_steps": _optional_round(graph_summary["shortest_path_steps"]),
        "tube_width_min": round(base["tube_width_min"], 6),
        "tube_width_max": round(base["tube_width_max"], 6),
        "mean_radius": round(mean_radius, 6),
        "std_radius": round(std_radius, 6),
        "min_bridge_radius": round(min_bridge_radius, 6),
        "bend_strength": round(base["params"].get("bend_strength", 0.0), 6),
        "arc_angle": round(base["params"].get("arc_angle", 0.0), 6),
        "arc_radius": round(base["params"].get("arc_radius", 0.0), 6),
        "max_deviation_from_AB": round(curve_metrics["max_deviation_from_AB"], 6),
        "total_turning_angle": round(curve_metrics["total_turning_angle"], 6),
        "mean_curvature": round(curve_metrics["mean_curvature"], 6),
        "max_curvature": round(curve_metrics["max_curvature"], 6),
        "self_proximity": round(curve_metrics["self_proximity"], 6),
        "neck_width": round(curve_metrics["neck_width"], 6),
        "concavity_depth": round(curve_metrics["concavity_depth"], 6),
        "bottleneck_width": round(base["tube_width_min"], 6),
        "connection_margin": round(abs(graph_summary["connection_margin"]), 6),
        "signed_connection_margin": round(graph_summary["connection_margin"], 6),
        "area": area,
        "perimeter": perimeter,
        "compactness": round(compactness, 6),
        "total_ink_pixels": total_ink_pixels,
    }
    return {
        "sample_id": sample_id,
        "split": split,
        "connected": connected,
        "circles": circles,
        "curve_points": base["curve_points"],
        "latent_path": latent_path,
        "image": image,
        "metadata": metadata,
    }


def _sample_target_pair(
    rng: random.Random,
    image_size: int,
    distance_range: tuple[float, float],
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(80):
        distance = rng.uniform(distance_range[0], distance_range[1])
        angle = rng.uniform(0.0, math.tau)
        dx = math.cos(angle) * distance / 2.0
        dy = math.sin(angle) * distance / 2.0
        x_min = margin + abs(dx)
        x_max = image_size - margin - abs(dx)
        y_min = margin + abs(dy)
        y_max = image_size - margin - abs(dy)
        if x_min >= x_max or y_min >= y_max:
            continue
        center_x = rng.uniform(x_min, x_max)
        center_y = rng.uniform(y_min, y_max)
        return (center_x - dx, center_y - dy), (center_x + dx, center_y + dy)
    raise RuntimeError("Failed to sample target pair within image bounds.")


def _curve_metrics(
    curve_points: list[tuple[float, float]],
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> dict:
    path_length = curve_length(curve_points)
    target_distance = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
    deviation = max_deviation_from_segment(curve_points, point_a, point_b)
    turning = total_turning_angle(curve_points)
    mean_curvature, max_curvature = curvature_stats(curve_points)
    proximity = self_proximity(curve_points)
    neck_width = proximity if math.isfinite(proximity) else 0.0
    return {
        "path_length": path_length,
        "tortuosity": path_length / target_distance if target_distance > 0 else 1.0,
        "max_deviation_from_AB": deviation,
        "total_turning_angle": turning,
        "mean_curvature": mean_curvature,
        "max_curvature": max_curvature,
        "self_proximity": proximity if math.isfinite(proximity) else 0.0,
        "neck_width": neck_width,
        "concavity_depth": deviation,
    }


def _shape_stats(shape_channel: np.ndarray) -> tuple[int, int, float, int]:
    mask = shape_channel > 0
    area = int(mask.sum())
    padded = np.pad(mask.astype(np.uint8), 1)
    perimeter = int(
        np.sum(padded[1:-1, 1:-1] != padded[:-2, 1:-1])
        + np.sum(padded[1:-1, 1:-1] != padded[2:, 1:-1])
        + np.sum(padded[1:-1, 1:-1] != padded[1:-1, :-2])
        + np.sum(padded[1:-1, 1:-1] != padded[1:-1, 2:])
    )
    compactness = 0.0 if perimeter == 0 else (4.0 * math.pi * area) / (perimeter**2)
    return area, perimeter, compactness, area


def _preserves_labels_with_append(
    positive_circles: list[Circle],
    negative_circles: list[Circle],
    extras: list[Circle],
    target_end_index: int,
) -> bool:
    positive_summary = summarize_graph(positive_circles + extras, 0, target_end_index)
    negative_summary = summarize_graph(negative_circles + extras, 0, target_end_index)
    return positive_summary["connected"] == 1 and negative_summary["connected"] == 0


def _curve_inside_image(curve_points: list[tuple[float, float]], radii: list[float], image_size: int) -> bool:
    return all(_inside_image(point, radius, image_size) for point, radius in zip(curve_points, radii))


def _points_inside(points: list[tuple[float, float]], margin: float, image_size: int) -> bool:
    return all(margin <= point[0] <= image_size - margin and margin <= point[1] <= image_size - margin for point in points)


def _inside_image(point: tuple[float, float], radius: float, image_size: int) -> bool:
    margin = radius + 2.0
    return margin <= point[0] <= image_size - margin and margin <= point[1] <= image_size - margin


def _is_pixel_connected(circles: list[Circle], target_count: int, image_size: int) -> bool:
    shape_channel = render_circles(circles, image_size=image_size)
    image_array = build_multichannel_image(
        shape_channel,
        (circles[0].x, circles[0].y),
        (circles[target_count - 1].x, circles[target_count - 1].y),
        image_size=image_size,
    )
    return pixel_connected(image_array, (circles[0].x, circles[0].y), (circles[target_count - 1].x, circles[target_count - 1].y), connectivity=8)


def _optional_round(value: float | int | None) -> str | float | int:
    if value is None:
        return ""
    return round(value, 6) if isinstance(value, float) else value


def generate_scene_query_samples(
    config: dict,
    split: str,
    scene_index: int,
    base_seed: int,
    target_positive: int,
    target_negative: int,
) -> list[dict]:
    rng = random.Random(base_seed + scene_index * 17011 + _split_seed_offset(split))
    for _ in range(120):
        scene = _generate_multi_magatama_scene(config, rng, split, scene_index)
        if scene is None:
            continue
        samples = _sample_queries_from_scene(
            config=config,
            rng=rng,
            scene=scene,
            split=split,
            scene_index=scene_index,
            target_positive=target_positive,
            target_negative=target_negative,
            base_seed=base_seed,
        )
        if samples:
            return samples
    raise RuntimeError(f"Failed to generate multi-scene samples for split={split}, scene_index={scene_index}")


def _generate_multi_magatama_scene(config: dict, rng: random.Random, split: str, scene_index: int) -> dict | None:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    scene_shape_count_range = tuple(dataset_cfg.get("scene_shape_count_range", [3, 4]))
    shape_count = rng.randint(scene_shape_count_range[0], scene_shape_count_range[1])
    scene_shape_families = dataset_cfg.get(
        "scene_shape_families",
        {
            "crescent": 0.55,
            "c_shape": 0.25,
            "bezier_bend": 0.20,
        },
    )
    scene_min_gap = float(dataset_cfg.get("scene_min_inter_shape_gap", 3.0))
    require_cross_source_positive = bool(dataset_cfg.get("scene_require_cross_source_positive", True))
    max_scene_attempts = int(dataset_cfg.get("scene_generation_attempts", 120))

    for _ in range(max_scene_attempts):
        centers = _scene_candidate_centers(image_size=image_size, shape_count=shape_count, rng=rng)
        all_circles: list[Circle] = []
        scene_shapes = []
        success = True

        for shape_index, center in enumerate(centers):
            family = choose_weighted_key(rng, scene_shape_families)
            base = _generate_scene_shape(config, rng, center, family)
            if base is None or not _scene_shape_placement_ok(base["circles"], all_circles, scene_min_gap):
                success = False
                break
            start_index = len(all_circles)
            all_circles.extend(deepcopy(base["circles"]))
            scene_shapes.append(
                {
                    **base,
                    "source_pair_id": shape_index,
                    "anchor_circle_indices": (start_index, start_index + len(base["circles"]) - 1),
                }
            )

        if not success or len(scene_shapes) != shape_count:
            continue

        bridges = _scene_add_bridges(config, rng, scene_shapes, all_circles)
        anchors = _scene_build_anchors(scene_shapes, all_circles)
        extra_counts = _scene_add_extras(config, rng, scene_shapes, all_circles, anchors)
        labels = component_labels(build_adjacency(all_circles))
        query_candidates = _scene_query_candidates(anchors, labels)
        has_cross_source_positive = any(candidate["connected"] and not candidate["same_source_pair"] for candidate in query_candidates)
        has_negative = any(not candidate["connected"] for candidate in query_candidates)
        if require_cross_source_positive and not has_cross_source_positive:
            continue
        if not has_negative:
            continue

        shape_channel = render_circles(all_circles, image_size=image_size)
        component_ids = {labels[anchor["circle_index"]] for anchor in anchors}
        scene = {
            "scene_id": f"{split}_scene_{scene_index:05d}",
            "circles": all_circles,
            "anchors": anchors,
            "scene_shapes": scene_shapes,
            "bridges": bridges,
            "shape_channel": shape_channel,
            "component_labels": labels,
            "query_candidates": query_candidates,
            "extra_counts": extra_counts,
            "scene_metrics": {
                "n_scene_shapes": len(scene_shapes),
                "n_scene_components": len(component_ids),
                "n_scene_bridges": len(bridges),
                "n_query_points": len(anchors),
            },
        }
        return scene
    return None


def _scene_candidate_centers(image_size: int, shape_count: int, rng: random.Random) -> list[tuple[float, float]]:
    templates = [
        (0.26, 0.26),
        (0.74, 0.26),
        (0.26, 0.74),
        (0.74, 0.74),
        (0.50, 0.18),
        (0.50, 0.82),
    ]
    jitter = image_size * 0.055
    chosen = rng.sample(templates, k=shape_count)
    centers = []
    for x_ratio, y_ratio in chosen:
        centers.append(
            (
                min(image_size - 18.0, max(18.0, x_ratio * image_size + rng.uniform(-jitter, jitter))),
                min(image_size - 18.0, max(18.0, y_ratio * image_size + rng.uniform(-jitter, jitter))),
            )
        )
    return centers


def _generate_scene_shape(
    config: dict,
    rng: random.Random,
    center: tuple[float, float],
    family: str,
) -> dict | None:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    scene_target_distance_range = tuple(dataset_cfg.get("scene_target_distance_range", [18, 40]))
    scene_bend_strength_range = tuple(dataset_cfg.get("scene_bend_strength_range", [8, 24]))
    scene_arc_radius_range = tuple(dataset_cfg.get("scene_arc_radius_range", [14, 26]))
    scene_arc_angle_range = tuple(dataset_cfg.get("scene_arc_angle_range", [160, 320]))
    scene_tube_width_range = tuple(dataset_cfg.get("scene_tube_width_range", [2.5, 6.5]))

    for _ in range(80):
        if family == "straight":
            distance = rng.uniform(scene_target_distance_range[0], scene_target_distance_range[1])
            angle = rng.uniform(0.0, math.tau)
            point_a, point_b = _pair_from_center(center, distance, angle)
            coarse_points = generate_straight_curve(point_a, point_b, n_points=18)
            params = {"curve_type": "straight", "bend_strength": 0.0, "arc_angle": 0.0, "arc_radius": 0.0}
        elif family == "bezier_bend":
            distance = rng.uniform(scene_target_distance_range[0], scene_target_distance_range[1])
            angle = rng.uniform(0.0, math.tau)
            point_a, point_b = _pair_from_center(center, distance, angle)
            bend_strength = rng.uniform(scene_bend_strength_range[0], scene_bend_strength_range[1]) * rng.choice([-1.0, 1.0])
            coarse_points, control = generate_bezier_bend_curve(point_a, point_b, bend_strength=bend_strength, n_points=22)
            params = {
                "curve_type": "bezier_bend",
                "bend_strength": bend_strength,
                "arc_angle": 0.0,
                "arc_radius": 0.0,
                "control_point_x": round(control[0], 6),
                "control_point_y": round(control[1], 6),
            }
        elif family in {"c_shape", "crescent"}:
            shape = _sample_scene_c_shape(
                rng=rng,
                image_size=image_size,
                center=center,
                arc_radius_range=scene_arc_radius_range,
                arc_angle_range=scene_arc_angle_range,
                family=family,
            )
            if shape is None:
                continue
            point_a, point_b, coarse_points, params = shape
        else:
            return None

        tube_min = rng.uniform(scene_tube_width_range[0], max(scene_tube_width_range[0], scene_tube_width_range[1] - 0.6))
        tube_max = rng.uniform(max(tube_min + 0.2, scene_tube_width_range[0]), scene_tube_width_range[1])
        dense_points = resample_polyline(coarse_points, max_step=max(0.9, tube_min * 0.52))
        radii = bottleneck_radii(len(dense_points), tube_min, tube_max, family, rng)
        if not _curve_inside_image(dense_points, radii, image_size):
            continue
        circles, latent_path = circles_from_curve_points(dense_points, radii)
        summary = summarize_graph(circles, 0, len(latent_path) - 1)
        if summary["connected"] != 1:
            continue
        return {
            "shape_family": family,
            "curve_points": dense_points,
            "point_a": dense_points[0],
            "point_b": dense_points[-1],
            "circles": circles,
            "latent_path": latent_path,
            "tube_width_min": tube_min,
            "tube_width_max": tube_max,
            "params": params,
            "curve_metrics": _curve_metrics(dense_points, dense_points[0], dense_points[-1]),
        }
    return None


def _sample_scene_c_shape(
    rng: random.Random,
    image_size: int,
    center: tuple[float, float],
    arc_radius_range: tuple[float, float],
    arc_angle_range: tuple[float, float],
    family: str,
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]], dict] | None:
    for _ in range(80):
        if family == "crescent":
            arc_angle_deg = rng.uniform(max(190.0, arc_angle_range[0]), min(330.0, arc_angle_range[1] + 20.0))
            arc_radius = rng.uniform(arc_radius_range[0], arc_radius_range[1] + 3.0)
        else:
            arc_angle_deg = rng.uniform(max(145.0, arc_angle_range[0]), min(290.0, arc_angle_range[1]))
            arc_radius = rng.uniform(arc_radius_range[0], arc_radius_range[1])
        rotation = rng.uniform(0.0, math.tau)
        theta_start = rotation - math.radians(arc_angle_deg) / 2.0
        theta_end = rotation + math.radians(arc_angle_deg) / 2.0
        points = generate_c_shape_curve(center, arc_radius, theta_start, theta_end, n_points=36 if family == "crescent" else 30)
        if not _points_inside(points, arc_radius * 0.18, image_size):
            continue
        params = {
            "curve_type": "circular_arc" if family == "c_shape" else "crescent_arc",
            "bend_strength": 0.0,
            "arc_angle": arc_angle_deg,
            "arc_radius": arc_radius,
            "rotation_angle": math.degrees(rotation),
        }
        return points[0], points[-1], points, params
    return None


def _pair_from_center(center: tuple[float, float], distance: float, angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = math.cos(angle) * distance / 2.0
    dy = math.sin(angle) * distance / 2.0
    return (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy)


def _scene_shape_placement_ok(new_circles: list[Circle], existing_circles: list[Circle], min_gap: float) -> bool:
    for new_circle in new_circles:
        for existing_circle in existing_circles:
            if math.hypot(new_circle.x - existing_circle.x, new_circle.y - existing_circle.y) <= new_circle.radius + existing_circle.radius + min_gap:
                return False
    return True


def _scene_build_anchors(scene_shapes: list[dict], circles: list[Circle]) -> list[dict]:
    anchors = []
    anchor_id = 0
    for shape in scene_shapes:
        start_idx, end_idx = shape["anchor_circle_indices"]
        start_circle = circles[start_idx]
        end_circle = circles[end_idx]
        anchors.append(
            {
                "anchor_id": anchor_id,
                "circle_index": start_idx,
                "x": start_circle.x,
                "y": start_circle.y,
                "source_pair_id": shape["source_pair_id"],
                "family": shape["shape_family"],
                "endpoint_role": "head",
            }
        )
        anchor_id += 1
        anchors.append(
            {
                "anchor_id": anchor_id,
                "circle_index": end_idx,
                "x": end_circle.x,
                "y": end_circle.y,
                "source_pair_id": shape["source_pair_id"],
                "family": shape["shape_family"],
                "endpoint_role": "tail",
            }
        )
        anchor_id += 1
    return anchors


def _scene_add_bridges(config: dict, rng: random.Random, scene_shapes: list[dict], circles: list[Circle]) -> list[dict]:
    dataset_cfg = config["dataset"]
    bridge_count_range = tuple(dataset_cfg.get("scene_bridge_count_range", [1, 2]))
    bridge_target = rng.randint(bridge_count_range[0], min(bridge_count_range[1], max(1, len(scene_shapes) - 1)))
    pairs = list(combinations(range(len(scene_shapes)), 2))
    rng.shuffle(pairs)
    bridges = []
    for left_index, right_index in pairs:
        if len(bridges) >= bridge_target:
            break
        bridge = _scene_bridge_between_shapes(config, rng, scene_shapes[left_index], scene_shapes[right_index])
        if bridge is None:
            continue
        candidate_circles = circles + bridge["circles"]
        labels = component_labels(build_adjacency(candidate_circles))
        left_anchor = scene_shapes[left_index]["anchor_circle_indices"][0]
        right_anchor = scene_shapes[right_index]["anchor_circle_indices"][0]
        if labels[left_anchor] != labels[right_anchor]:
            continue
        bridge["circle_index_start"] = len(circles)
        circles.extend(bridge["circles"])
        bridges.append(bridge)
    return bridges


def _scene_bridge_between_shapes(config: dict, rng: random.Random, shape_a: dict, shape_b: dict) -> dict | None:
    dataset_cfg = config["dataset"]
    scene_bridge_width_range = tuple(dataset_cfg.get("scene_bridge_width_range", [1.6, 3.2]))
    candidate_pairs = [
        (shape_a["point_a"], shape_b["point_a"], shape_a["anchor_circle_indices"][0], shape_b["anchor_circle_indices"][0]),
        (shape_a["point_a"], shape_b["point_b"], shape_a["anchor_circle_indices"][0], shape_b["anchor_circle_indices"][1]),
        (shape_a["point_b"], shape_b["point_a"], shape_a["anchor_circle_indices"][1], shape_b["anchor_circle_indices"][0]),
        (shape_a["point_b"], shape_b["point_b"], shape_a["anchor_circle_indices"][1], shape_b["anchor_circle_indices"][1]),
    ]
    candidate_pairs.sort(key=lambda item: math.hypot(item[0][0] - item[1][0], item[0][1] - item[1][1]))
    for point_a, point_b, anchor_a_idx, anchor_b_idx in candidate_pairs:
        distance = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
        if not 12.0 <= distance <= 72.0:
            continue
        bend_strength = rng.uniform(4.0, max(6.0, distance * 0.22)) * rng.choice([-1.0, 1.0])
        coarse_points, control = generate_bezier_bend_curve(point_a, point_b, bend_strength=bend_strength, n_points=18)
        bridge_width = rng.uniform(scene_bridge_width_range[0], scene_bridge_width_range[1])
        dense_points = resample_polyline(coarse_points, max_step=max(0.8, bridge_width * 0.55))
        if len(dense_points) <= 2:
            continue
        bridge_points = dense_points[1:-1]
        bridge_circles = [
            Circle(x=point[0], y=point[1], radius=bridge_width, kind="path", latent_path_index=None)
            for point in bridge_points
        ]
        return {
            "from_source_pair_id": shape_a["source_pair_id"],
            "to_source_pair_id": shape_b["source_pair_id"],
            "from_anchor_circle_index": anchor_a_idx,
            "to_anchor_circle_index": anchor_b_idx,
            "bend_strength": bend_strength,
            "control_point": [round(control[0], 6), round(control[1], 6)],
            "circles": bridge_circles,
        }
    return None


def _scene_add_extras(
    config: dict,
    rng: random.Random,
    scene_shapes: list[dict],
    circles: list[Circle],
    anchors: list[dict],
) -> dict:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    radius_range = tuple(dataset_cfg["radius_range"])
    noise_range = tuple(dataset_cfg.get("noise_circles_range", [0, 16]))
    distractor_range = tuple(dataset_cfg.get("distractor_curves_range", [0, 2]))
    baseline_signature = _scene_equivalence_signature(circles, anchors)
    accepted_noise = 0
    for candidate in random_noise_circles(rng, image_size, radius_range, rng.randint(noise_range[0], noise_range[1])):
        if _scene_equivalence_signature(circles + [candidate], anchors) == baseline_signature:
            circles.append(candidate)
            accepted_noise += 1
    accepted_distractors = 0
    for _ in range(rng.randint(distractor_range[0], distractor_range[1])):
        distractor = _generate_scene_distractor(config, rng)
        if distractor and _scene_equivalence_signature(circles + distractor, anchors) == baseline_signature:
            circles.extend(distractor)
            accepted_distractors += 1
    return {"n_noise_circles": accepted_noise, "n_distractor_curves": accepted_distractors}


def _generate_scene_distractor(config: dict, rng: random.Random) -> list[Circle] | None:
    dataset_cfg = config["dataset"]
    image_size = int(dataset_cfg["image_size"])
    center = (
        rng.uniform(image_size * 0.18, image_size * 0.82),
        rng.uniform(image_size * 0.18, image_size * 0.82),
    )
    family = rng.choice(["bezier_bend", "c_shape", "crescent"])
    base = _generate_scene_shape(config, rng, center, family)
    if base is None:
        return None
    circles = deepcopy(base["circles"])
    for circle in circles:
        circle.kind = "noise"
        circle.latent_path_index = None
    return circles


def _scene_equivalence_signature(circles: list[Circle], anchors: list[dict]) -> tuple[bool, ...]:
    labels = component_labels(build_adjacency(circles))
    signature = []
    for anchor_a, anchor_b in combinations(anchors, 2):
        signature.append(labels[anchor_a["circle_index"]] == labels[anchor_b["circle_index"]])
    return tuple(signature)


def _scene_query_candidates(anchors: list[dict], labels: list[int]) -> list[dict]:
    candidates = []
    for anchor_a, anchor_b in combinations(anchors, 2):
        connected = labels[anchor_a["circle_index"]] == labels[anchor_b["circle_index"]]
        families = sorted([anchor_a["family"], anchor_b["family"]])
        candidates.append(
            {
                "anchor_a": anchor_a,
                "anchor_b": anchor_b,
                "connected": int(connected),
                "same_source_pair": int(anchor_a["source_pair_id"] == anchor_b["source_pair_id"]),
                "query_family_pair": f"{families[0]}|{families[1]}",
                "target_distance": math.hypot(anchor_a["x"] - anchor_b["x"], anchor_a["y"] - anchor_b["y"]),
            }
        )
    return candidates


def _sample_queries_from_scene(
    config: dict,
    rng: random.Random,
    scene: dict,
    split: str,
    scene_index: int,
    target_positive: int,
    target_negative: int,
    base_seed: int,
) -> list[dict]:
    dataset_cfg = config["dataset"]
    queries_per_scene = int(dataset_cfg.get("scene_queries_per_scene", 8))
    cross_source_positive_ratio = float(dataset_cfg.get("scene_cross_source_positive_ratio", 0.5))
    positives = [candidate for candidate in scene["query_candidates"] if candidate["connected"] == 1]
    negatives = [candidate for candidate in scene["query_candidates"] if candidate["connected"] == 0]
    cross_source_positives = [candidate for candidate in positives if candidate["same_source_pair"] == 0]
    same_source_positives = [candidate for candidate in positives if candidate["same_source_pair"] == 1]
    rng.shuffle(cross_source_positives)
    rng.shuffle(same_source_positives)
    rng.shuffle(negatives)

    positive_quota = min(target_positive, max(1, queries_per_scene // 2) if target_positive > 0 else 0, len(positives))
    negative_quota = min(target_negative, queries_per_scene - positive_quota, len(negatives))
    if positive_quota + negative_quota == 0:
        return []

    cross_source_quota = min(len(cross_source_positives), round(positive_quota * cross_source_positive_ratio))
    selected = cross_source_positives[:cross_source_quota]
    selected.extend(same_source_positives[: max(0, positive_quota - len(selected))])
    if len(selected) < positive_quota:
        remaining = [candidate for candidate in positives if candidate not in selected]
        selected.extend(remaining[: positive_quota - len(selected)])
    selected_negatives = _match_scene_negatives(selected, negatives, negative_quota)
    selected.extend(selected_negatives)
    rng.shuffle(selected)

    samples = []
    for query_index, query in enumerate(selected):
        sample_seed = base_seed + scene_index * 97 + query_index
        samples.append(
            _build_scene_sample_record(
                config=config,
                scene=scene,
                query=query,
                split=split,
                sample_id=f"{split}_scene{scene_index:05d}_query{query_index:03d}",
                sample_seed=sample_seed,
            )
        )
    return samples


def _build_scene_sample_record(
    config: dict,
    scene: dict,
    query: dict,
    split: str,
    sample_id: str,
    sample_seed: int,
) -> dict:
    image_size = int(config["dataset"]["image_size"])
    circles = scene["circles"]
    anchor_a = query["anchor_a"]
    anchor_b = query["anchor_b"]
    point_a = (anchor_a["x"], anchor_a["y"])
    point_b = (anchor_b["x"], anchor_b["y"])
    graph_summary = summarize_graph(circles, anchor_a["circle_index"], anchor_b["circle_index"])
    image_array = build_multichannel_image(scene["shape_channel"], point_a, point_b, image_size=image_size)
    image = image_from_array(image_array)

    connected = int(query["connected"])
    connected_pixel_4 = int(pixel_connected(image_array, point_a, point_b, connectivity=4))
    connected_pixel_8 = int(pixel_connected(image_array, point_a, point_b, connectivity=8))
    label_disagreement = int(connected != connected_pixel_8)
    actual_path = graph_summary["shortest_path"] or []
    path_points = [(circles[index].x, circles[index].y) for index in actual_path] if actual_path else []
    query_metrics = _scene_query_curve_metrics(scene, query, graph_summary, path_points)
    mean_radius, std_radius = radius_stats(circles)
    min_bridge_radius = min(circles[index].radius for index in actual_path) if actual_path else min(
        circles[anchor_a["circle_index"]].radius,
        circles[anchor_b["circle_index"]].radius,
    )
    area, perimeter, compactness, total_ink_pixels = _shape_stats(scene["shape_channel"])
    shape_family = query["query_family_pair"]

    metadata = {
        "sample_id": sample_id,
        "scene_id": scene["scene_id"],
        "split": split,
        "connected": connected,
        "connected_geometry": connected,
        "connected_pixel_4": connected_pixel_4,
        "connected_pixel_8": connected_pixel_8,
        "label_disagreement": label_disagreement,
        "seed": sample_seed,
        "image_size": image_size,
        "shape_family": shape_family,
        "curve_type": "scene_query",
        "generation_mode": "multi_magatama_scene_v1_query",
        "negative_mode": "" if connected else "scene_cross_component",
        "query_case": _scene_query_case(query),
        "same_source_pair": query["same_source_pair"],
        "same_component": connected,
        "query_point_i": anchor_a["anchor_id"],
        "query_point_j": anchor_b["anchor_id"],
        "source_pair_i": anchor_a["source_pair_id"],
        "source_pair_j": anchor_b["source_pair_id"],
        "query_shape_family_a": anchor_a["family"],
        "query_shape_family_b": anchor_b["family"],
        "n_circles": len(circles),
        "n_noise_circles": scene["extra_counts"]["n_noise_circles"],
        "n_distractor_curves": scene["extra_counts"]["n_distractor_curves"],
        "n_scene_shapes": scene["scene_metrics"]["n_scene_shapes"],
        "n_scene_components": scene["scene_metrics"]["n_scene_components"],
        "n_scene_bridges": scene["scene_metrics"]["n_scene_bridges"],
        "n_query_points": scene["scene_metrics"]["n_query_points"],
        "target_ax": round(point_a[0], 6),
        "target_ay": round(point_a[1], 6),
        "target_bx": round(point_b[0], 6),
        "target_by": round(point_b[1], 6),
        "target_distance": round(graph_summary["target_distance"], 6),
        "path_length": _optional_round(graph_summary["path_length"]),
        "tortuosity": _optional_round(graph_summary["tortuosity"]),
        "shortest_path_steps": _optional_round(graph_summary["shortest_path_steps"]),
        "tube_width_min": round(query_metrics["tube_width_min"], 6),
        "tube_width_max": round(query_metrics["tube_width_max"], 6),
        "mean_radius": round(mean_radius, 6),
        "std_radius": round(std_radius, 6),
        "min_bridge_radius": round(min_bridge_radius, 6),
        "bend_strength": round(query_metrics["bend_strength"], 6),
        "arc_angle": round(query_metrics["arc_angle"], 6),
        "arc_radius": round(query_metrics["arc_radius"], 6),
        "max_deviation_from_AB": round(query_metrics["max_deviation_from_AB"], 6),
        "total_turning_angle": round(query_metrics["total_turning_angle"], 6),
        "mean_curvature": round(query_metrics["mean_curvature"], 6),
        "max_curvature": round(query_metrics["max_curvature"], 6),
        "self_proximity": round(query_metrics["self_proximity"], 6),
        "neck_width": round(query_metrics["neck_width"], 6),
        "concavity_depth": round(query_metrics["concavity_depth"], 6),
        "bottleneck_width": round(query_metrics["tube_width_min"], 6),
        "connection_margin": round(abs(graph_summary["connection_margin"]), 6),
        "signed_connection_margin": round(graph_summary["connection_margin"], 6),
        "area": area,
        "perimeter": perimeter,
        "compactness": round(compactness, 6),
        "total_ink_pixels": total_ink_pixels,
    }
    geometry_payload = {
        "sample_id": sample_id,
        "scene_id": scene["scene_id"],
        "split": split,
        "connected": connected,
        "query_anchor_ids": [anchor_a["anchor_id"], anchor_b["anchor_id"]],
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "circle_index": anchor["circle_index"],
                "x": round(anchor["x"], 6),
                "y": round(anchor["y"], 6),
                "source_pair_id": anchor["source_pair_id"],
                "family": anchor["family"],
                "endpoint_role": anchor["endpoint_role"],
            }
            for anchor in scene["anchors"]
        ],
        "scene_shapes": [
            {
                "source_pair_id": shape["source_pair_id"],
                "family": shape["shape_family"],
                "anchor_circle_indices": list(shape["anchor_circle_indices"]),
                "curve_points": [[round(point[0], 6), round(point[1], 6)] for point in shape["curve_points"]],
                "params": shape["params"],
            }
            for shape in scene["scene_shapes"]
        ],
        "bridges": [
            {
                "from_source_pair_id": bridge["from_source_pair_id"],
                "to_source_pair_id": bridge["to_source_pair_id"],
                "from_anchor_circle_index": bridge["from_anchor_circle_index"],
                "to_anchor_circle_index": bridge["to_anchor_circle_index"],
                "bend_strength": round(bridge["bend_strength"], 6),
                "control_point": bridge["control_point"],
            }
            for bridge in scene["bridges"]
        ],
        "circles": [circle.to_dict() for circle in circles],
        "component_labels": scene["component_labels"],
    }
    return {
        "sample_id": sample_id,
        "split": split,
        "connected": connected,
        "circles": circles,
        "curve_points": path_points,
        "latent_path": actual_path,
        "image": image,
        "metadata": metadata,
        "geometry_payload": geometry_payload,
    }


def _scene_query_curve_metrics(scene: dict, query: dict, graph_summary: dict, path_points: list[tuple[float, float]]) -> dict:
    anchor_a = query["anchor_a"]
    anchor_b = query["anchor_b"]
    shape_a = scene["scene_shapes"][anchor_a["source_pair_id"]]
    shape_b = scene["scene_shapes"][anchor_b["source_pair_id"]]
    if path_points:
        metrics = _curve_metrics(path_points, (anchor_a["x"], anchor_a["y"]), (anchor_b["x"], anchor_b["y"]))
        radii = [scene["circles"][index].radius for index in graph_summary["shortest_path"]]
        tube_width_min = min(radii)
        tube_width_max = max(radii)
    else:
        metrics = {
            key: (shape_a["curve_metrics"][key] + shape_b["curve_metrics"][key]) / 2.0
            for key in ("max_deviation_from_AB", "total_turning_angle", "mean_curvature", "max_curvature", "self_proximity", "neck_width", "concavity_depth")
        }
        tube_width_min = min(shape_a["tube_width_min"], shape_b["tube_width_min"])
        tube_width_max = max(shape_a["tube_width_max"], shape_b["tube_width_max"])
    return {
        **metrics,
        "tube_width_min": tube_width_min,
        "tube_width_max": tube_width_max,
        "bend_strength": (shape_a["params"].get("bend_strength", 0.0) + shape_b["params"].get("bend_strength", 0.0)) / 2.0,
        "arc_angle": (shape_a["params"].get("arc_angle", 0.0) + shape_b["params"].get("arc_angle", 0.0)) / 2.0,
        "arc_radius": (shape_a["params"].get("arc_radius", 0.0) + shape_b["params"].get("arc_radius", 0.0)) / 2.0,
    }


def _scene_query_case(query: dict) -> str:
    if query["connected"] == 1 and query["same_source_pair"] == 1:
        return "same_source_positive"
    if query["connected"] == 1 and query["same_source_pair"] == 0:
        return "cross_source_positive"
    return "cross_component_negative"


def _match_scene_negatives(selected_positives: list[dict], negatives: list[dict], negative_quota: int) -> list[dict]:
    remaining = negatives[:]
    selected = []
    for positive in selected_positives:
        if len(selected) >= negative_quota:
            break
        best_index = None
        best_score = None
        for index, negative in enumerate(remaining):
            family_penalty = 0 if negative["query_family_pair"] == positive["query_family_pair"] else 1
            distance_penalty = abs(negative["target_distance"] - positive["target_distance"])
            score = (family_penalty, distance_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        if best_index is not None:
            selected.append(remaining.pop(best_index))
    if len(selected) < negative_quota:
        selected.extend(remaining[: negative_quota - len(selected)])
    return selected
