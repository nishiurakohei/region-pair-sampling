from __future__ import annotations

from typing import Iterable


def _to_float(value) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    return float(value)


def _resolve_metric(record: dict, primary: str, fallback: str) -> float:
    primary_value = _to_float(record.get(primary))
    if primary_value is not None:
        return primary_value
    fallback_value = _to_float(record.get(fallback))
    if fallback_value is None:
        return 0.0
    return fallback_value


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance ** 0.5
    if std_value == 0:
        return [0.0 for _ in values]
    return [(value - mean_value) / std_value for value in values]


def attach_difficulty_proxy(records: list[dict]) -> list[dict]:
    if not records:
        return records

    resolved_tortuosity = [_resolve_metric(record, "tortuosity", "resolved_tortuosity") for record in records]
    turning_angles = [_resolve_metric(record, "total_turning_angle", "total_turning_angle") for record in records]
    noise_counts = [_resolve_metric(record, "n_noise_circles", "n_noise_circles") for record in records]
    distractor_counts = [_resolve_metric(record, "n_distractor_curves", "n_distractor_curves") for record in records]
    deviations = [_resolve_metric(record, "max_deviation_from_AB", "max_deviation_from_AB") for record in records]
    min_bridge_radii = [_resolve_metric(record, "min_bridge_radius", "min_bridge_radius") for record in records]
    signed_margins = [abs(_resolve_metric(record, "signed_connection_margin", "connection_margin")) for record in records]

    z_tortuosity = _zscore(resolved_tortuosity)
    z_turning = _zscore(turning_angles)
    z_noise = _zscore(noise_counts)
    z_distractor = _zscore(distractor_counts)
    z_deviation = _zscore(deviations)
    z_min_radius = _zscore(min_bridge_radii)
    z_margin = _zscore(signed_margins)

    for index, record in enumerate(records):
        record["resolved_tortuosity"] = round(resolved_tortuosity[index], 6)
        record["difficulty_proxy"] = round(
            z_tortuosity[index]
            + z_turning[index]
            + z_noise[index]
            + z_distractor[index]
            + z_deviation[index]
            - z_min_radius[index]
            - z_margin[index],
            6,
        )
    return records


def sort_by_numeric(records: Iterable[dict], key: str, reverse: bool = False) -> list[dict]:
    def numeric_value(record: dict) -> float:
        value = record.get(key)
        if value in (None, "", "NA", "None"):
            return float("-inf") if reverse else float("inf")
        return float(value)

    return sorted(records, key=numeric_value, reverse=reverse)
