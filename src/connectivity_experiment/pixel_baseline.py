from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def pixel_connected(image_path: str, point_a: tuple[float, float], point_b: tuple[float, float], threshold: int = 32) -> bool:
    image = Image.open(image_path).convert("L")
    mask = np.array(image) >= threshold

    start = (int(round(point_a[1])), int(round(point_a[0])))
    end = (int(round(point_b[1])), int(round(point_b[0])))

    if not _valid(mask, start) or not _valid(mask, end):
        return False

    queue = deque([start])
    visited = {start}
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            return True
        for delta_y, delta_x in neighbors:
            next_node = (y + delta_y, x + delta_x)
            if next_node in visited or not _valid(mask, next_node):
                continue
            visited.add(next_node)
            queue.append(next_node)
    return False


def _valid(mask: np.ndarray, point: tuple[int, int]) -> bool:
    y, x = point
    if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])

