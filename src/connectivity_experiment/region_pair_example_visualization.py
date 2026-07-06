from __future__ import annotations

import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from connectivity_experiment.io_utils import ensure_dir, read_csv, write_csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


def load_prediction_records(path: str | Path) -> list[dict]:
    return read_csv(path)


def build_example_groups(
    records: list[dict],
    per_group: int = 20,
    random_seed: int = 0,
) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        method = str(record.get("sampling_method", "")).strip()
        if not method:
            continue
        grouped[method].append(record)

    rng = random.Random(random_seed)
    output: dict[str, dict[str, list[dict]]] = {}
    for method in sorted(grouped):
        subset = list(grouped[method])
        output[method] = {
            "random_examples": _sample_records(subset, count=per_group, rng=rng),
            "high_loss_examples": _sort_records(subset, key="loss", reverse=True)[:per_group],
            "hard_positive_examples": _sort_records(
                [record for record in subset if int(_to_float(record.get("connected")) or 0) == 1],
                key="loss",
                reverse=True,
            )[:per_group],
            "hard_negative_examples": _sort_records(
                [record for record in subset if int(_to_float(record.get("connected")) or 0) == 0],
                key="loss",
                reverse=True,
            )[:per_group],
        }
    return output


def save_example_grids(
    predictions_path: str | Path,
    out_dir: str | Path,
    per_group: int = 20,
    n_cols: int = 4,
    random_seed: int = 0,
) -> list[Path]:
    records = load_prediction_records(predictions_path)
    grouped_examples = build_example_groups(records, per_group=per_group, random_seed=random_seed)
    out_dir = ensure_dir(out_dir)
    output_paths: list[Path] = []

    for method, example_groups in grouped_examples.items():
        for group_name, rows in example_groups.items():
            if not rows:
                continue
            stem = f"{method}_{group_name}"
            out_path = out_dir / f"{stem}.png"
            csv_path = out_dir / f"{stem}.csv"
            title = f"{method} | {group_name.replace('_', ' ')}"
            save_grid(rows, out_path=out_path, title=title, n_cols=n_cols)
            write_csv(csv_path, _example_csv_rows(rows, example_type=group_name))
            output_paths.append(out_path)
            output_paths.append(csv_path)

    report_path = out_dir / "example_pairs_report.md"
    report_lines = [
        "# Region Pair Example Visualizations",
        "",
        f"- predictions: `{Path(predictions_path)}`",
        f"- output_dir: `{out_dir}`",
        f"- per_group: `{per_group}`",
        "",
        "## Files",
        "",
    ]
    for method, example_groups in grouped_examples.items():
        report_lines.extend([f"### {method}", ""])
        for group_name, rows in example_groups.items():
            if not rows:
                continue
            stem = f"{method}_{group_name}"
            png_path = out_dir / f"{stem}.png"
            csv_path = out_dir / f"{stem}.csv"
            report_lines.append(
                f"- `{group_name}`: [PNG]({png_path}) | [CSV]({csv_path})"
            )
        report_lines.append("")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    output_paths.append(report_path)
    return output_paths


def save_grid(
    records: list[dict],
    out_path: str | Path,
    title: str,
    n_cols: int = 4,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for example pair visualization.") from exc

    if not records:
        return
    n_rows = int(math.ceil(len(records) / max(1, n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")

    for index, record in enumerate(records):
        draw_pair(axes[index], record)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_pair(ax, record: dict) -> None:
    image_path = Path(str(record["image_path"]))
    image = Image.open(image_path).convert("L")
    mask = np.asarray(image)
    row_i = int(_to_float(record.get("row_i")) or 0)
    col_i = int(_to_float(record.get("col_i")) or 0)
    row_j = int(_to_float(record.get("row_j")) or 0)
    col_j = int(_to_float(record.get("col_j")) or 0)

    ax.imshow(mask, cmap="gray")
    ax.plot([col_i, col_j], [row_i, row_j], color="#f2c14e", linewidth=1.5)
    ax.scatter([col_i], [row_i], s=34, marker="o", c="#e45756", edgecolors="white", linewidths=0.6)
    ax.scatter([col_j], [row_j], s=34, marker="x", c="#4c78a8", linewidths=1.2)

    correct = int(_to_float(record.get("is_correct")) or 0) == 1
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_edgecolor("#2ca02c" if correct else "#d62728")

    connected = int(_to_float(record.get("connected")) or 0)
    pred = int(_to_float(record.get("pred_label")) or 0) if record.get("pred_label", "") not in {"", None} else "NA"
    title = (
        f"{record.get('sampling_method', 'NA')}\n"
        f"y={connected}, pred={pred}, p={_fmt(record.get('pred_prob_connected'))}, loss={_fmt(record.get('loss'))}\n"
        f"euc={_fmt(record.get('euclidean_distance'))}, "
        f"geo={_fmt(record.get('geodesic_distance'))}, "
        f"ratio={_fmt(record.get('geodesic_euclidean_ratio'))}, "
        f"cross={_fmt(record.get('boundary_crossing_count'), integer=True)}\n"
        f"margin={_fmt(record.get('distance_to_boundary_min'))}, "
        f"inside={_fmt(record.get('line_inside_fraction'))}"
    )
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _sample_records(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(records) <= count:
        return list(records)
    indices = list(range(len(records)))
    chosen_indices = sorted(rng.sample(indices, count))
    return [records[index] for index in chosen_indices]


def _example_csv_rows(records: list[dict], example_type: str) -> list[dict]:
    csv_rows = []
    for record in records:
        csv_rows.append(
            {
                "sampling_method": str(record.get("sampling_method", "")),
                "example_type": example_type,
                "image_id": str(record.get("image_id", "")),
                "row_i": record.get("row_i", ""),
                "col_i": record.get("col_i", ""),
                "row_j": record.get("row_j", ""),
                "col_j": record.get("col_j", ""),
                "connected": record.get("connected", ""),
                "pred_label": record.get("pred_label", ""),
                "pred_prob_connected": record.get("pred_prob_connected", ""),
                "is_correct": record.get("is_correct", ""),
                "loss": record.get("loss", ""),
                "euclidean_distance": record.get("euclidean_distance", ""),
                "geodesic_distance": record.get("geodesic_distance", ""),
                "geodesic_euclidean_ratio": record.get("geodesic_euclidean_ratio", ""),
                "boundary_crossing_count": record.get("boundary_crossing_count", ""),
                "distance_to_boundary_min": record.get("distance_to_boundary_min", ""),
                "line_inside_fraction": record.get("line_inside_fraction", ""),
                "component_gap_distance": record.get("component_gap_distance", ""),
                "max_outside_segment_length": record.get("max_outside_segment_length", ""),
                "difficulty_bucket": record.get("difficulty_bucket", ""),
            }
        )
    return csv_rows


def _sort_records(records: list[dict], key: str, reverse: bool) -> list[dict]:
    return sorted(records, key=lambda record: _sortable_numeric(record.get(key)), reverse=reverse)


def _sortable_numeric(value) -> float:
    numeric = _to_float(value)
    if numeric is None:
        return float("-inf")
    return numeric


def _to_float(value) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    return float(value)


def _fmt(value, integer: bool = False) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "-"
    if integer:
        return str(int(round(numeric)))
    return f"{numeric:.3f}"
