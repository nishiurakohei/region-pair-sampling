"""Figures 1--7 for the Hard-Maze report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import MazeData
from graph import degrees

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_hard_maze"
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
LEVELS = ["H0", "H1", "H2", "H3"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def plot_maze(ax, H, V, pivot=None, control=None, title=""):
    n = H.shape[0]
    for r in range(n):
        for c in range(n - 1):
            if H[r, c]:
                ax.plot([c, c + 1], [n - 1 - r, n - 1 - r], color="#4a5568", lw=1)
    for r in range(n - 1):
        for c in range(n):
            if V[r, c]:
                ax.plot([c, c], [n - 1 - r, n - 2 - r], color="#4a5568", lw=1)
    ax.scatter([0, n - 1], [n - 1, 0], c=["#2f855a", "#c53030"], s=35, zorder=4)
    for edge, color, label in ((pivot, "#e53e3e", "pivotal"), (control, "#dd6b20", "control")):
        if edge is None or int(edge[0]) < 0:
            continue
        axis, r, c = map(int, edge)
        if axis == 0:
            x, y = [c, c + 1], [n - 1 - r] * 2
        else:
            x, y = [c, c], [n - 1 - r, n - 2 - r]
        ax.plot(x, y, color=color, lw=3, label=label, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


def distributions(field, filename, xlabel):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
    for ax, level in zip(axes, LEVELS):
        data = MazeData(level, "test")
        if field in data.arrays:
            vals = data.arrays[field]
        else:
            vals = np.asarray([(h.sum() + v.sum()) if field == "edge_count" else
                               int(h.sum() + v.sum() - 225 + len(np.unique(_component_labels(h, v))))
                               for h, v in zip(data.H, data.V)])
        for y, color, label in ((1, "#2b6cb0", "positive"), (0, "#c05621", "negative")):
            ax.hist(vals[data.y == y], bins=18, density=True, alpha=.55, color=color, label=label)
        ax.set_title(level); ax.set_xlabel(xlabel); ax.grid(alpha=.2)
    axes[0].set_ylabel("density"); axes[-1].legend(fontsize=8)
    save(fig, filename)


def _component_labels(H, V):
    from graph import components
    return components(H, V)[0]


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    shortcuts = json.loads((RESULTS / "shortcut_baselines.json").read_text())
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = .22
    for j, (name, color) in enumerate((("cnn", "#2b6cb0"), ("resnet", "#c05621"))):
        mu = [metrics["levels"][l][name]["aggregate"]["accuracy"]["mean"] for l in LEVELS]
        sd = [metrics["levels"][l][name]["aggregate"]["accuracy"]["std"] for l in LEVELS]
        ax.bar(x + (j - .5) * width, mu, width, yerr=sd, label=name.upper(), color=color, capsize=3)
    scalar = [shortcuts[l]["scalar_logistic"]["accuracy"] for l in LEVELS]
    ax.plot(x, scalar, "o--", color="#2f855a", label="scalar shortcut")
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xticks(x, LEVELS); ax.set_ylim(.45, 1); ax.set_ylabel("test accuracy")
    ax.set_title("Figure 1. Accuracy across the hardness ladder"); ax.legend(); ax.grid(alpha=.2, axis="y")
    save(fig, "fig1_accuracy_h0_h3.png")

    distributions("edge_count", "fig2_edge_count.png", "open-edge count")
    distributions("beta1", "fig3_beta1.png", r"cycle rank $\beta_1$")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
    for ax, level in zip(axes, LEVELS):
        data = MazeData(level, "test")
        for y, color, marker, label in ((1, "#2b6cb0", "o", "positive"), (0, "#c05621", "s", "negative")):
            hist = np.asarray([np.bincount(degrees(h, v).ravel(), minlength=5)[:5]
                               for h, v in zip(data.H[data.y == y], data.V[data.y == y])]).mean(0)
            ax.plot(range(5), hist, marker=marker, color=color, label=label)
        ax.set_title(level); ax.set_xlabel("node degree"); ax.grid(alpha=.2)
    axes[0].set_ylabel("mean node count"); axes[-1].legend(fontsize=8)
    save(fig, "fig4_degree_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for level, color in zip(("H1", "H2", "H3"), ("#718096", "#2b6cb0", "#c05621")):
        z = MazeData(level, "test").arrays
        ax.hist(z["shortest_path_length"][z["y"] == 1], bins=np.arange(27.5, 55, 2),
                density=True, histtype="step", lw=2, color=color, label=level)
    ax.set_xlabel("positive shortest-path length"); ax.set_ylabel("density")
    ax.set_title("Figure 5. Positive path-length distribution"); ax.legend(); ax.grid(alpha=.2)
    save(fig, "fig5_path_length.png")

    data = MazeData("H3", "test")
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, pair in enumerate((0, 7)):
        pos, neg = 2 * pair, 2 * pair + 1
        pivot, ctrl = data.arrays["pivotal_edge"][pos], data.arrays["control_edge"][pos]
        plot_maze(axes[row, 0], data.H[pos], data.V[pos], pivot=pivot,
                  title=f"pair {pair}: positive (pivotal red)")
        plot_maze(axes[row, 1], data.H[neg], data.V[neg], control=ctrl,
                  title=f"pair {pair}: negative (control orange)")
    fig.suptitle("Figure 6. Representative H3 matched pairs")
    save(fig, "fig6_matched_pairs.png")

    z = np.load(RESULTS / "pred_H3_cnn_s1.npz")
    pred = z["prob"] > .5
    bad = [i for i in range(len(data.y) // 2) if not (pred[2*i] and not pred[2*i+1])]
    chosen = bad[:2]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, pair in enumerate(chosen):
        for col, i in enumerate((2 * pair, 2 * pair + 1)):
            plot_maze(axes[row, col], data.H[i], data.V[i],
                      pivot=data.arrays["pivotal_edge"][i] if data.y[i] else None,
                      control=data.arrays["control_edge"][i] if not data.y[i] else None,
                      title=f"y={int(data.y[i])}, CNN p={z['prob'][i]:.3f}")
    fig.suptitle("Figure 7. Representative H3 CNN pair failures")
    save(fig, "fig7_pair_failures.png")
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
