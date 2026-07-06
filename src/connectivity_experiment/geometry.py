from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, asdict


@dataclass
class Circle:
    x: float
    y: float
    radius: float
    kind: str
    latent_path_index: int | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("x", "y", "radius"):
            payload[key] = round(payload[key], 6)
        return payload


def distance_between(a: Circle, b: Circle) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def edge_margin(a: Circle, b: Circle) -> float:
    return a.radius + b.radius - distance_between(a, b)


def circles_overlap(a: Circle, b: Circle) -> bool:
    return edge_margin(a, b) >= 0.0


def build_adjacency(circles: list[Circle]) -> list[list[int]]:
    adjacency = [[] for _ in circles]
    for left in range(len(circles)):
        for right in range(left + 1, len(circles)):
            if circles_overlap(circles[left], circles[right]):
                adjacency[left].append(right)
                adjacency[right].append(left)
    return adjacency


def bfs_shortest_path(adjacency: list[list[int]], start: int, end: int) -> list[int] | None:
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if end not in parent:
        return None
    path = []
    cursor = end
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return path


def component_labels(adjacency: list[list[int]]) -> list[int]:
    labels = [-1] * len(adjacency)
    component_id = 0
    for start in range(len(adjacency)):
        if labels[start] != -1:
            continue
        queue = deque([start])
        labels[start] = component_id
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return labels


def weighted_shortest_path(circles: list[Circle], adjacency: list[list[int]], start: int, end: int) -> tuple[list[int] | None, float | None]:
    heap: list[tuple[float, int]] = [(0.0, start)]
    best = {start: 0.0}
    parent = {start: None}
    while heap:
        cost, node = heapq.heappop(heap)
        if node == end:
            break
        if cost > best[node]:
            continue
        for neighbor in adjacency[node]:
            new_cost = cost + distance_between(circles[node], circles[neighbor])
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor))
    if end not in parent:
        return None, None
    path = []
    cursor = end
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return path, best[end]


def target_distance(circles: list[Circle], start: int, end: int) -> float:
    return distance_between(circles[start], circles[end])


def connection_margin(circles: list[Circle], adjacency: list[list[int]], start: int, end: int, path: list[int] | None) -> float:
    if path:
        margins = [edge_margin(circles[path[index]], circles[path[index + 1]]) for index in range(len(path) - 1)]
        return min(margins) if margins else 0.0

    labels = component_labels(adjacency)
    start_label = labels[start]
    end_label = labels[end]
    min_gap = float("inf")
    for left, left_label in enumerate(labels):
        if left_label != start_label:
            continue
        for right, right_label in enumerate(labels):
            if right_label != end_label:
                continue
            gap = distance_between(circles[left], circles[right]) - circles[left].radius - circles[right].radius
            if gap < min_gap:
                min_gap = gap
    return -min_gap if min_gap < float("inf") else 0.0


def summarize_graph(circles: list[Circle], start: int, end: int) -> dict:
    adjacency = build_adjacency(circles)
    steps_path = bfs_shortest_path(adjacency, start, end)
    weighted_path, weighted_length = weighted_shortest_path(circles, adjacency, start, end)
    connected = 1 if steps_path else 0
    direct_distance = target_distance(circles, start, end)

    if steps_path and weighted_path:
        shortest_steps = len(steps_path) - 1
        path_length = weighted_length if weighted_length is not None else 0.0
        tortuosity = path_length / direct_distance if direct_distance > 0 else 1.0
        path_for_metrics = weighted_path
    else:
        shortest_steps = None
        path_length = None
        tortuosity = None
        path_for_metrics = None

    return {
        "connected": connected,
        "adjacency": adjacency,
        "shortest_path": path_for_metrics,
        "shortest_path_steps": shortest_steps,
        "path_length": path_length,
        "tortuosity": tortuosity,
        "target_distance": direct_distance,
        "connection_margin": connection_margin(circles, adjacency, start, end, path_for_metrics),
    }


def radius_stats(circles: list[Circle]) -> tuple[float, float]:
    radii = [circle.radius for circle in circles]
    if not radii:
        return 0.0, 0.0
    mean_value = sum(radii) / len(radii)
    variance = sum((radius - mean_value) ** 2 for radius in radii) / len(radii)
    return mean_value, math.sqrt(variance)

