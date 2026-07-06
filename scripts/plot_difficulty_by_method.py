"""Plot difficulty distribution per sampling method.

Reads pair_records.csv from the instruction_subset_v2 sampling run and
produces two figures:

- difficulty_by_method_primary.png:
  focus on the three methods described in the meeting summary
  (random / euclidean_farthest / geodesic_farthest).

- difficulty_by_method_all.png:
  same panels but including all eight sampling methods.

Panels in each figure:
  1. difficulty_proxy histogram (overlaid, normalized)
  2. difficulty bucket stacked bar (Easy / Medium / Hard / Very hard)
  3. geodesic/Euclidean ratio (connected pairs only, overlaid)
  4. boundary_crossing_count (all pairs, overlaid)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "region_pair_sampling_stare_instruction_subset_v2" / "metrics" / "pair_records.csv"
OUT_DIR = ROOT / "outputs" / "figures_difficulty_by_method"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_METHODS = ["random", "euclidean_farthest", "geodesic_farthest"]
ALL_METHODS = [
    "random",
    "euclidean_farthest",
    "geodesic_farthest",
    "feature_space_diverse",
    "boundary_crossing_stratified",
    "boundary_crossing_targeted",
    "hard_pair_targeted",
    "hard_pair_targeted_v2",
]
BUCKET_ORDER = ["Easy", "Medium", "Hard", "Very hard"]
BUCKET_COLORS = {
    "Easy": "#4c9ac6",
    "Medium": "#f5c04a",
    "Hard": "#e07a5f",
    "Very hard": "#8e3b46",
}
METHOD_COLORS = {
    "random": "#2b8cbe",
    "euclidean_farthest": "#e34a33",
    "geodesic_farthest": "#31a354",
    "feature_space_diverse": "#756bb1",
    "boundary_crossing_stratified": "#fdae6b",
    "boundary_crossing_targeted": "#c994c7",
    "hard_pair_targeted": "#636363",
    "hard_pair_targeted_v2": "#000000",
}


def load(methods):
    data = {m: {"proxy": [], "ratio_connected": [], "geodesic_connected": [],
                "crossing": [], "euclid": [], "buckets": {b: 0 for b in BUCKET_ORDER}}
            for m in methods}
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["sampling_method"]
            if m not in data:
                continue
            try:
                proxy = float(row["difficulty_proxy"])
                euclid = float(row["euclidean_distance"])
                crossing = float(row["boundary_crossing_count"])
            except ValueError:
                continue
            data[m]["proxy"].append(proxy)
            data[m]["euclid"].append(euclid)
            data[m]["crossing"].append(crossing)
            connected = row["connected"] in ("1", "True", "true")
            if connected:
                try:
                    gd = float(row["geodesic_distance"])
                    gr = float(row["geodesic_euclidean_ratio"])
                except ValueError:
                    gd = gr = 0.0
                if gd > 0:
                    data[m]["geodesic_connected"].append(gd)
                if gr > 0:
                    data[m]["ratio_connected"].append(gr)
            b = row["difficulty_bucket"]
            if b in data[m]["buckets"]:
                data[m]["buckets"][b] += 1
    return data


def plot_panel_proxy(ax, data, methods):
    all_vals = np.concatenate([np.asarray(data[m]["proxy"]) for m in methods])
    lo, hi = np.quantile(all_vals, [0.005, 0.995])
    bins = np.linspace(lo, hi, 45)
    for m in methods:
        vals = np.clip(np.asarray(data[m]["proxy"]), lo, hi)
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.8,
                color=METHOD_COLORS.get(m, None), density=True, label=m)
    ax.set_title("difficulty_proxy distribution")
    ax.set_xlabel("difficulty_proxy (z-normalized composite)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)


def plot_panel_buckets_frac(ax, data, methods):
    x = np.arange(len(methods))
    totals = np.array([sum(data[m]["buckets"].values()) for m in methods])
    bottom = np.zeros(len(methods))
    for b in BUCKET_ORDER:
        heights = np.array([data[m]["buckets"][b] for m in methods]) / totals
        ax.bar(x, heights, bottom=bottom, color=BUCKET_COLORS[b], label=b,
               edgecolor="white", linewidth=0.5)
        bottom += heights
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_title("difficulty bucket fraction per method")
    ax.set_ylabel("fraction of pairs")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")


def plot_panel_hard_counts(ax, data, methods):
    x = np.arange(len(methods))
    hard = np.array([data[m]["buckets"]["Hard"] for m in methods])
    vhard = np.array([data[m]["buckets"]["Very hard"] for m in methods])
    ax.bar(x, hard, color=BUCKET_COLORS["Hard"], label="Hard",
           edgecolor="white", linewidth=0.5)
    ax.bar(x, vhard, bottom=hard, color=BUCKET_COLORS["Very hard"],
           label="Very hard", edgecolor="white", linewidth=0.5)
    for i, (h, v) in enumerate(zip(hard, vhard)):
        ax.text(i, h + v + max(hard.max(), 1) * 0.02, f"{h + v}",
                ha="center", fontsize=8, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_title("Hard + Very-hard pair count per method")
    ax.set_ylabel("pair count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y")


def plot_panel_euclid(ax, data, methods):
    all_vals = np.concatenate([np.asarray(data[m]["euclid"]) for m in methods])
    hi = np.quantile(all_vals, 0.995)
    bins = np.linspace(0, hi, 40)
    for m in methods:
        vals = np.clip(np.asarray(data[m]["euclid"]), 0, hi)
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.8,
                color=METHOD_COLORS.get(m, None), density=True, label=m)
    ax.set_title("Euclidean distance distribution")
    ax.set_xlabel("Euclidean distance (pixels)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)


def plot_panel_ratio(ax, data, methods):
    all_vals = np.concatenate(
        [np.asarray(data[m]["ratio_connected"]) for m in methods
         if data[m]["ratio_connected"]]
    )
    hi = min(np.quantile(all_vals, 0.995), 3.5)
    bins = np.linspace(1.0, hi, 40)
    for m in methods:
        vals = np.clip(np.asarray(data[m]["ratio_connected"]), 1.0, hi)
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.8,
                color=METHOD_COLORS.get(m, None), density=True, label=m)
    ax.set_title("geodesic / Euclidean ratio  (connected only)")
    ax.set_xlabel("ratio")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)


def plot_panel_crossing(ax, data, methods):
    # Discrete-ish: use integer bins up to 99th percentile
    all_vals = np.concatenate([np.asarray(data[m]["crossing"]) for m in methods])
    hi = int(np.quantile(all_vals, 0.99)) + 1
    bins = np.arange(-0.5, hi + 1.5, 1.0)
    for m in methods:
        vals = np.clip(np.asarray(data[m]["crossing"]), 0, hi)
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.8,
                color=METHOD_COLORS.get(m, None), density=True, label=m)
    ax.set_title("boundary_crossing_count distribution")
    ax.set_xlabel("crossings")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)


def make_figure(methods, out_path, suptitle):
    data = load(methods)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    plot_panel_proxy(axes[0, 0], data, methods)
    plot_panel_buckets_frac(axes[0, 1], data, methods)
    plot_panel_ratio(axes[1, 0], data, methods)
    plot_panel_crossing(axes[1, 1], data, methods)
    plot_panel_euclid(axes[2, 0], data, methods)
    plot_panel_hard_counts(axes[2, 1], data, methods)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    make_figure(
        PRIMARY_METHODS,
        OUT_DIR / "difficulty_by_method_primary.png",
        "Difficulty distribution per sampling method  (3 primary methods)",
    )
    make_figure(
        ALL_METHODS,
        OUT_DIR / "difficulty_by_method_all.png",
        "Difficulty distribution per sampling method  (all 8 methods)",
    )


if __name__ == "__main__":
    main()
