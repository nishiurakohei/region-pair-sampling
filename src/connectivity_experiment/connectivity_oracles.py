from __future__ import annotations

from collections import deque

import numpy as np

from connectivity_experiment.geometry import Circle, build_adjacency, component_labels, summarize_graph


def build_circle_graph(centers: list[tuple[float, float]], radii: list[float]) -> list[list[int]]:
    circles = [Circle(x=center[0], y=center[1], radius=radius, kind="path") for center, radius in zip(centers, radii)]
    return build_adjacency(circles)


def geometry_connected(point_a_index: int, point_b_index: int, centers: list[tuple[float, float]], radii: list[float]) -> bool:
    circles = [Circle(x=center[0], y=center[1], radius=radius, kind="path") for center, radius in zip(centers, radii)]
    summary = summarize_graph(circles, point_a_index, point_b_index)
    return bool(summary["connected"])


def pixel_connected(image: np.ndarray, point_a: tuple[float, float], point_b: tuple[float, float], connectivity: int = 8, threshold: int = 16) -> bool:
    if image.ndim == 3:
        mask = image[:, :, 0] >= threshold
    else:
        mask = image >= threshold

    start = (int(round(point_a[1])), int(round(point_a[0])))
    end = (int(round(point_b[1])), int(round(point_b[0])))
    if not _valid(mask, start) or not _valid(mask, end):
        return False

    neighbors = _neighbors(connectivity)
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        for delta_y, delta_x in neighbors:
            next_node = (node[0] + delta_y, node[1] + delta_x)
            if next_node in visited or not _valid(mask, next_node):
                continue
            visited.add(next_node)
            queue.append(next_node)
    return False


def pixel_component_labels(image: np.ndarray, connectivity: int = 8, threshold: int = 16) -> np.ndarray:
    if image.ndim == 3:
        mask = image[:, :, 0] >= threshold
    else:
        mask = image >= threshold
    labels = -np.ones(mask.shape, dtype=np.int32)
    component_id = 0
    neighbors = _neighbors(connectivity)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or labels[row, col] != -1:
                continue
            queue = deque([(row, col)])
            labels[row, col] = component_id
            while queue:
                node = queue.popleft()
                for delta_y, delta_x in neighbors:
                    next_node = (node[0] + delta_y, node[1] + delta_x)
                    if not _valid(mask, next_node):
                        continue
                    if labels[next_node] != -1:
                        continue
                    labels[next_node] = component_id
                    queue.append(next_node)
            component_id += 1
    return labels


def geometry_component_labels(circles: list[Circle]) -> list[int]:
    return component_labels(build_adjacency(circles))


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _valid(mask: np.ndarray, point: tuple[int, int]) -> bool:
    row, col = point
    if row < 0 or col < 0 or row >= mask.shape[0] or col >= mask.shape[1]:
        return False
    return bool(mask[row, col])
