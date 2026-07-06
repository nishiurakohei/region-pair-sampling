from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from connectivity_experiment.geometry import Circle


def render_tube_from_curve(
    curve_points: list[tuple[float, float]],
    radii: list[float],
    image_size: int,
    fill_value: int = 255,
) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(image)
    for point, radius in zip(curve_points, radii):
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill_value)
    return np.asarray(image, dtype=np.uint8)


def render_circles(circles: list[Circle], image_size: int) -> np.ndarray:
    image = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(image)
    for circle in circles:
        fill_value = 255 if circle.kind in {"path", "target"} else 180
        draw.ellipse((circle.x - circle.radius, circle.y - circle.radius, circle.x + circle.radius, circle.y + circle.radius), fill=fill_value)
    return np.asarray(image, dtype=np.uint8)


def render_target_markers(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    image_size: int,
    marker_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    channel_a = Image.new("L", (image_size, image_size), color=0)
    channel_b = Image.new("L", (image_size, image_size), color=0)
    draw_a = ImageDraw.Draw(channel_a)
    draw_b = ImageDraw.Draw(channel_b)
    _draw_marker(draw_a, point_a, marker_radius)
    _draw_marker(draw_b, point_b, marker_radius)
    return np.asarray(channel_a, dtype=np.uint8), np.asarray(channel_b, dtype=np.uint8)


def build_multichannel_image(
    shape_channel: np.ndarray,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    image_size: int,
    marker_radius: int = 3,
) -> np.ndarray:
    marker_a, marker_b = render_target_markers(point_a, point_b, image_size=image_size, marker_radius=marker_radius)
    return np.stack([shape_channel, marker_a, marker_b], axis=2)


def image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="RGB")


def circles_from_curve_points(
    curve_points: list[tuple[float, float]],
    radii: list[float],
    target_count: int | None = None,
) -> tuple[list[Circle], list[int]]:
    circles = []
    for index, (point, radius) in enumerate(zip(curve_points, radii)):
        kind = "target" if index in {0, len(curve_points) - 1} else "path"
        circles.append(Circle(x=point[0], y=point[1], radius=radius, kind=kind, latent_path_index=index))
    latent_path = list(range(len(circles)))
    return circles, latent_path


def random_noise_circles(
    rng: random.Random,
    image_size: int,
    radius_range: tuple[float, float],
    count: int,
) -> list[Circle]:
    circles = []
    for _ in range(count):
        radius = rng.uniform(radius_range[0], radius_range[1])
        circles.append(
            Circle(
                x=rng.uniform(radius + 2.0, image_size - radius - 2.0),
                y=rng.uniform(radius + 2.0, image_size - radius - 2.0),
                radius=radius,
                kind="noise",
            )
        )
    return circles


def bottleneck_radii(
    n_points: int,
    base_width_min: float,
    base_width_max: float,
    family: str,
    rng: random.Random,
) -> list[float]:
    if family == "crescent":
        return crescent_radii(n_points, base_width_min, base_width_max, rng)
    if family == "c_shape":
        neck_center = rng.uniform(0.45, 0.65)
        bottleneck_depth = rng.uniform(0.8, 2.0)
    elif family == "bezier_bend":
        neck_center = rng.uniform(0.4, 0.6)
        bottleneck_depth = rng.uniform(0.4, 1.4)
    else:
        neck_center = 0.5
        bottleneck_depth = rng.uniform(0.1, 0.7)

    radii = []
    for index in range(n_points):
        t = index / max(1, n_points - 1)
        base = rng.uniform(base_width_min, base_width_max)
        neck = bottleneck_depth * math.exp(-((t - neck_center) ** 2) / 0.02)
        radii.append(max(1.2, base - neck))
    return radii


def crescent_radii(
    n_points: int,
    base_width_min: float,
    base_width_max: float,
    rng: random.Random,
) -> list[float]:
    head_bias = rng.uniform(0.18, 0.32)
    belly_center = rng.uniform(0.32, 0.48)
    belly_strength = rng.uniform(0.35, 0.7) * (base_width_max - base_width_min)
    tail_taper = rng.uniform(1.1, 1.7)
    neck_center = rng.uniform(0.72, 0.86)
    neck_depth = rng.uniform(0.3, 0.9) * (base_width_max - base_width_min)
    radii = []
    for index in range(n_points):
        t = index / max(1, n_points - 1)
        base = base_width_max - (base_width_max - base_width_min) * (t**tail_taper)
        head = 0.18 * (base_width_max - base_width_min) * math.exp(-((t - head_bias) ** 2) / 0.012)
        belly = belly_strength * math.exp(-((t - belly_center) ** 2) / 0.028)
        neck = neck_depth * math.exp(-((t - neck_center) ** 2) / 0.018)
        tail = 0.15 * (base_width_max - base_width_min) * t
        radius = base + head + belly - neck - tail
        radii.append(max(1.0, radius))
    return radii


def _draw_marker(draw: ImageDraw.ImageDraw, point: tuple[float, float], marker_radius: int) -> None:
    draw.ellipse(
        (
            point[0] - marker_radius,
            point[1] - marker_radius,
            point[0] + marker_radius,
            point[1] + marker_radius,
        ),
        fill=255,
    )
