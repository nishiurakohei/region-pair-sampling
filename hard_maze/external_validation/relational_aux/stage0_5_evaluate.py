"""Evaluate scratch DT-Net checkpoints on the fixed Stage-0 screening suite."""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
UP = HERE / "upstream" / "maze-extrapolation"
sys.path.insert(0, str(UP))
from src.models.dt_net import DTNet
from stage0_reproduce import evaluate, make_data


def load_scratch(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = DTNet()
    model.load_state_dict(saved["net"], strict=True)
    return model.eval().to(device), saved


def conditions(config):
    for axis, spec in config.items():
        if axis in {"model", "seeds", "batch_size"}:
            continue
        for size in spec["maze_sizes"]:
            for p in spec["percolations"]:
                for deadend in spec["deadend_starts"]:
                    key = (size, p, deadend, spec["num_mazes"])
                    yield axis, key, spec["iterations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "stage0_screen.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "stage0_5_training")
    ap.add_argument("--dataset-seed", type=int, default=42)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoints = sorted((args.out / "checkpoints").glob("seed_*/best_torch.pt"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"expected 3 checkpoints, found {len(checkpoints)}")

    datasets = {}
    for axis, key, iters in conditions(cfg):
        size, p, deadend, count = key
        if key not in datasets:
            x, y, digest = make_data(size, count, p, deadend, args.dataset_seed, HERE / "data")
            datasets[key] = (x, y, digest)

    rows = []
    for checkpoint in checkpoints:
        model, saved = load_scratch(checkpoint, device)
        model_seed = int(saved["model_seed"])
        for axis, key, iters in conditions(cfg):
            size, p, deadend, count = key
            x, y, digest = datasets[key]
            scores = evaluate(model, x, y, iters, cfg["batch_size"], device)
            for iteration, (accuracy, correct, seconds) in scores.items():
                row = {
                    "model_seed": model_seed, "dataset_seed": args.dataset_seed,
                    "axis": axis, "maze_size": size, "percolation": p,
                    "deadend_start": deadend, "iterations": iteration,
                    "num_mazes": len(x), "accuracy": accuracy,
                    "correct_count": int(sum(correct)), "seconds": seconds,
                    "dataset_sha256": digest,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "checkpoint": str(checkpoint.relative_to(args.out)),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    with (args.out / "scratch_eval_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    (args.out / "evaluation_environment.json").write_text(json.dumps({
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "dataset_seed": args.dataset_seed, "fixed_dataset_across_models": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
