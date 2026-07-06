from __future__ import annotations

import math
import random


def generate_straight_curve(point_a: tuple[float, float], point_b: tuple[float, float], n_points: int) -> list[tuple[float, float]]:
    return [_lerp(point_a, point_b, index / (n_points - 1)) for index in range(n_points)]


def generate_bezier_bend_curve(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    bend_strength: float,
    n_points: int,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    midpoint = ((_lerp(point_a, point_b, 0.5))[0], (_lerp(point_a, point_b, 0.5))[1])
    normal = _unit_normal(point_a, point_b)
    control = (
        midpoint[0] + normal[0] * bend_strength,
        midpoint[1] + normal[1] * bend_strength,
    )
    points = []
    for index in range(n_points):
        t = index / (n_points - 1)
        points.append(
            (
                (1 - t) ** 2 * point_a[0] + 2 * (1 - t) * t * control[0] + t**2 * point_b[0],
                (1 - t) ** 2 * point_a[1] + 2 * (1 - t) * t * control[1] + t**2 * point_b[1],
            )
        )
    return points, control


def generate_s_curve(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    bend_strength: float,
    n_points: int,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> tuple[list[tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]:
    normal = _unit_normal(point_a, point_b)
    point_1 = (
        point_a[0] + alpha * (point_b[0] - point_a[0]) + bend_strength * normal[0],
        point_a[1] + alpha * (point_b[1] - point_a[1]) + bend_strength * normal[1],
    )
    point_2 = (
        point_a[0] + beta * (point_b[0] - point_a[0]) - bend_strength * normal[0],
        point_a[1] + beta * (point_b[1] - point_a[1]) - bend_strength * normal[1],
    )
    points = []
    for index in range(n_points):
        t = index / (n_points - 1)
        points.append(
            (
                (1 - t) ** 3 * point_a[0]
                + 3 * (1 - t) ** 2 * t * point_1[0]
                + 3 * (1 - t) * t**2 * point_2[0]
                + t**3 * point_b[0],
                (1 - t) ** 3 * point_a[1]
                + 3 * (1 - t) ** 2 * t * point_1[1]
                + 3 * (1 - t) * t**2 * point_2[1]
                + t**3 * point_b[1],
            )
        )
    return points, (point_1, point_2)


def generate_c_shape_curve(
    center: tuple[float, float],
    radius: float,
    theta_start: float,
    theta_end: float,
    n_points: int,
) -> list[tuple[float, float]]:
    points = []
    for index in range(n_points):
        ratio = index / (n_points - 1)
        theta = theta_start + (theta_end - theta_start) * ratio
        points.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
    return points


def choose_weighted_key(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    total = sum(weights.values())
    threshold = rng.uniform(0.0, total)
    running = 0.0
    for key in keys:
        running += weights[key]
        if threshold <= running:
            return key
    return keys[-1]


def curve_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1]) for index in range(len(points) - 1))


def max_deviation_from_segment(points: list[tuple[float, float]], point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return max(_distance_to_segment(point, point_a, point_b) for point in points)


def total_turning_angle(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(1, len(points) - 1):
        prev_vec = (points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        next_vec = (points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        prev_norm = math.hypot(prev_vec[0], prev_vec[1])
        next_norm = math.hypot(next_vec[0], next_vec[1])
        if prev_norm == 0 or next_norm == 0:
            continue
        cosine = (prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1]) / (prev_norm * next_norm)
        total += math.acos(max(-1.0, min(1.0, cosine)))
    return total


def curvature_stats(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    curvatures = []
    for index in range(1, len(points) - 1):
        left = points[index - 1]
        center = points[index]
        right = points[index + 1]
        left_vec = (center[0] - left[0], center[1] - left[1])
        right_vec = (right[0] - center[0], right[1] - center[1])
        left_len = math.hypot(left_vec[0], left_vec[1])
        right_len = math.hypot(right_vec[0], right_vec[1])
        chord_len = math.hypot(right[0] - left[0], right[1] - left[1])
        if left_len == 0 or right_len == 0 or chord_len == 0:
            continue
        area_twice = abs(left_vec[0] * right_vec[1] - left_vec[1] * right_vec[0])
        curvatures.append(area_twice / (left_len * right_len * chord_len))
    if not curvatures:
        return 0.0, 0.0
    return sum(curvatures) / len(curvatures), max(curvatures)


def self_proximity(points: list[tuple[float, float]], skip: int = 8) -> float:
    if len(points) <= skip + 1:
        return float("inf")
    best = float("inf")
    for left in range(len(points)):
        for right in range(left + skip, len(points)):
            best = min(best, math.hypot(points[left][0] - points[right][0], points[left][1] - points[right][1]))
    return best


def resample_polyline(points: list[tuple[float, float]], max_step: float) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return points[:]
    sampled = [points[0]]
    for start, end in zip(points, points[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, math.ceil(segment_length / max_step))
        for step in range(1, steps + 1):
            sampled.append(_lerp(start, end, step / steps))
    return sampled


def _lerp(point_a: tuple[float, float], point_b: tuple[float, float], t: float) -> tuple[float, float]:
    return ((1 - t) * point_a[0] + t * point_b[0], (1 - t) * point_a[1] + t * point_b[1])


def _unit_normal(point_a: tuple[float, float], point_b: tuple[float, float]) -> tuple[float, float]:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (0.0, 1.0)
    return (-dy / norm, dx / norm)


def _distance_to_segment(point: tuple[float, float], seg_a: tuple[float, float], seg_b: tuple[float, float]) -> float:
    vx = seg_b[0] - seg_a[0]
    vy = seg_b[1] - seg_a[1]
    wx = point[0] - seg_a[0]
    wy = point[1] - seg_a[1]
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(point[0] - seg_a[0], point[1] - seg_a[1])
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(point[0] - seg_b[0], point[1] - seg_b[1])
    ratio = c1 / c2
    projection = (seg_a[0] + ratio * vx, seg_a[1] + ratio * vy)
    return math.hypot(point[0] - projection[0], point[1] - projection[1])
