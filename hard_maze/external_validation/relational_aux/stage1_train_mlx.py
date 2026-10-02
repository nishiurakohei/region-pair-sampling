"""Auxiliary-only random-pair graph-distance training with the DT-Net core."""
import argparse, csv, json, math, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from dt_net_mlx import torch_initialized

HERE = Path(__file__).resolve().parents[1]
NODE_PIXELS = [3, 7, 11, 15, 19]


class AuxBase(nn.Module):
    def __init__(self, projection, hidden):
        super().__init__(); self.projection = projection
        self.fc1 = nn.Linear(128 * 5, hidden); self.fc2 = nn.Linear(hidden, 1)

    def predict(self, nodes, global_feature, u, v):
        batch = mx.arange(nodes.shape[0])[:, None]
        hu, hv = nodes[batch, u], nodes[batch, v]
        g = mx.broadcast_to(global_feature[:, None, :], hu.shape)
        pair = mx.concatenate([hu, hv, mx.abs(hu - hv), hu * hv, g], axis=-1)
        return self.fc2(nn.relu(self.fc1(pair)))[..., 0]


class AuxDistanceModel(nn.Module):
    def __init__(self, seed, hidden):
        super().__init__(); core = torch_initialized(seed); mx.random.seed(seed + 10000)
        self.base = AuxBase(core.base.projection, hidden); self.recur = core.recur

    def __call__(self, x, u, v, iterations):
        z = nn.relu(self.base.projection(x))
        for _ in range(iterations): z = self.recur(z, x)
        nodes = mx.stack([mx.mean(z[:, r:r + 2, c:c + 2, :], axis=(1, 2))
                          for r in NODE_PIXELS for c in NODE_PIXELS], axis=1)
        mask = mx.max(x, axis=-1) > 0
        global_feature = mx.sum(z * mask[..., None], axis=(1, 2)) / mx.sum(mask, axis=(1, 2))[..., None]
        return self.base.predict(nodes, global_feature, u, v)


def lr_schedule(base_lr, steps_per_epoch, warmup_epochs):
    def value(step):
        epoch = mx.floor(step.astype(mx.float32) / steps_per_epoch)
        return base_lr * mx.where(epoch < warmup_epochs,
                                  1 - mx.exp(-(epoch + 1) / warmup_epochs), 1.0)
    return value


def graph_input(images):
    open_mask = np.max(images, axis=1, keepdims=True)
    return np.repeat(open_mask, 3, axis=1).transpose(0, 2, 3, 1).astype(np.float32)


def split_arrays(pair_file, maze_count):
    z = np.load(pair_file); k = len(z["u"]) // maze_count
    return {name: z[name].reshape(maze_count, k) for name in
            ("u", "v", "graph_distance", "manhattan_distance", "normalized_distance", "detour_ratio")}, z["source_index"].reshape(maze_count, k)[:, 0]


def evaluate(model, images, pairs, sources, maze_indices, batch_size, iterations):
    predictions = []
    for lo in range(0, len(maze_indices), batch_size):
        idx = maze_indices[lo:lo + batch_size]
        x = mx.array(graph_input(np.asarray(images[sources[idx]])))
        u, v = mx.array(pairs["u"][idx].astype(np.int32)), mx.array(pairs["v"][idx].astype(np.int32))
        pred = model(x, u, v, iterations); mx.eval(pred); predictions.append(np.array(pred))
    return np.concatenate(predictions).reshape(-1)


def rankdata(values):
    order = np.argsort(values, kind="mergesort"); sorted_values = values[order]; ranks = np.empty(len(values), dtype=np.float64)
    lo = 0
    while lo < len(values):
        hi = lo + 1
        while hi < len(values) and sorted_values[hi] == sorted_values[lo]: hi += 1
        ranks[order[lo:hi]] = (lo + hi - 1) / 2
        lo = hi
    return ranks


def metrics(y, pred):
    error = pred - y; denom = np.sum((y - y.mean()) ** 2)
    pearson = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    spearman = float(np.corrcoef(rankdata(y), rankdata(pred))[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"raw_mae": float(np.mean(np.abs(error))), "normalized_mae": float(np.mean(np.abs(error)) / 9),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "r2": float(1 - np.sum(error ** 2) / denom),
            "pearson": pearson, "spearman": spearman}


def train(cfg, out, pair_root):
    data_root = HERE / "data" / "easy-to-hard-data"
    train_images = np.load(data_root / "maze_data_train_9" / "inputs.npy", mmap_mode="r")
    test_images = np.load(data_root / "maze_data_test_9" / "inputs.npy", mmap_mode="r")
    train_pairs, train_sources = split_arrays(pair_root / "fixed_train_k16.npz", cfg["train_mazes"])
    val_pairs, val_sources = split_arrays(pair_root / "fixed_val_k16.npz", cfg["val_mazes"])
    test_pairs, test_sources = split_arrays(pair_root / "fixed_test_k16.npz", cfg["test_mazes"])
    train_idx = np.arange(min(cfg["train_subset_mazes"], cfg["train_mazes"])); val_idx = np.arange(min(cfg["val_subset_mazes"], cfg["val_mazes"]))
    summaries = []
    for seed in cfg["model_seeds"]:
        model = AuxDistanceModel(seed, cfg["pair_head_hidden"]); rng = np.random.default_rng(seed + 1)
        steps = len(train_idx) // cfg["batch_mazes"]
        base_opt = optim.Adam(lr_schedule(cfg["base_lr"], steps, cfg["warmup_epochs"]), bias_correction=True)
        recur_opt = optim.Adam(lr_schedule(cfg["base_lr"] / cfg["recurrent_lr_divisor"], steps, cfg["warmup_epochs"]), bias_correction=True)
        wd = cfg["weight_decay"]
        def loss_fn(params, x, u, v, target):
            model.update(params); pred = model(x, u, v, cfg["training_iterations"])
            loss = mx.mean((pred - target) ** 2)
            return loss, mx.mean(mx.abs(pred - target))
        value_grad = mx.value_and_grad(loss_fn); rows = []; best = float("inf"); seed_dir = out / "checkpoints" / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(cfg["epochs"]):
            started = time.time(); order = rng.permutation(train_idx); loss_sum = mae_sum = 0.0
            for lo in range(0, steps * cfg["batch_mazes"], cfg["batch_mazes"]):
                idx = order[lo:lo + cfg["batch_mazes"]]
                x = mx.array(graph_input(np.asarray(train_images[train_sources[idx]])))
                u = mx.array(train_pairs["u"][idx].astype(np.int32)); v = mx.array(train_pairs["v"][idx].astype(np.int32))
                target = mx.array(train_pairs["normalized_distance"][idx].astype(np.float32))
                values, grads = value_grad(model.trainable_parameters(), x, u, v, target)
                bg = tree_map(lambda g, p: g + wd * p, grads["base"], model.base.trainable_parameters())
                rg = tree_map(lambda g, p: g + wd * p, grads["recur"], model.recur.trainable_parameters())
                base_opt.update(model.base, bg); recur_opt.update(model.recur, rg)
                mx.eval(values, model.parameters(), base_opt.state, recur_opt.state)
                loss_sum += float(values[0]); mae_sum += float(values[1])
            val_pred = evaluate(model, train_images, val_pairs, val_sources, val_idx, cfg["batch_mazes"], cfg["training_iterations"])
            val_true = val_pairs["normalized_distance"][val_idx].reshape(-1); val_mse = float(np.mean((val_pred - val_true) ** 2))
            row = {"model_seed": seed, "epoch": epoch + 1, "train_mse": loss_sum / steps, "train_normalized_mae": mae_sum / steps,
                   "validation_mse": val_mse, "validation_normalized_mae": float(np.mean(np.abs(val_pred - val_true))),
                   "seconds": time.time() - started, "base_lr": float(base_opt.learning_rate), "recurrent_lr": float(recur_opt.learning_rate)}
            rows.append(row); print(json.dumps(row), flush=True); model.save_weights(str(seed_dir / "last.safetensors"))
            if val_mse < best: best = val_mse; model.save_weights(str(seed_dir / "best.safetensors"))
        model.load_weights(str(seed_dir / "best.safetensors"))
        with (seed_dir / "training_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        full_test_idx = np.arange(min(cfg.get("test_subset_mazes", cfg["test_mazes"]), cfg["test_mazes"]))
        true = test_pairs["graph_distance"][full_test_idx].reshape(-1).astype(float)
        iteration_results = {}
        for iterations in cfg["test_iterations"]:
            pred_norm = evaluate(model, test_images, test_pairs, test_sources, full_test_idx, cfg["batch_mazes"], iterations)
            iteration_results[str(iterations)] = {"metrics": metrics(true, pred_norm * 9), "predictions": pred_norm}
        summaries.append({"model_seed": seed, "iterations": {k: v["metrics"] for k, v in iteration_results.items()}})
        primary = iteration_results[str(cfg["training_iterations"])]["predictions"] * 9
        np.savez_compressed(seed_dir / "test_predictions.npz", true=true, predicted=primary,
                            manhattan=test_pairs["manhattan_distance"][full_test_idx].reshape(-1),
                            detour_ratio=test_pairs["detour_ratio"][full_test_idx].reshape(-1))
    return summaries, train_pairs, test_pairs


def baselines_and_report(cfg, out, train_pairs, test_pairs, model_summaries):
    train_y = train_pairs["graph_distance"].reshape(-1).astype(float); train_m = train_pairs["manhattan_distance"].reshape(-1).astype(float)
    test_count = min(cfg.get("test_subset_mazes", cfg["test_mazes"]), cfg["test_mazes"])
    y = test_pairs["graph_distance"][:test_count].reshape(-1).astype(float)
    m = test_pairs["manhattan_distance"][:test_count].reshape(-1).astype(float)
    slope, intercept = np.polyfit(train_m, train_y, 1)
    baselines = {"constant_mean": metrics(y, np.full_like(y, train_y.mean())), "manhattan": metrics(y, m),
                 "manhattan_linear": metrics(y, slope * m + intercept),
                 "linear_coefficients": {"slope": float(slope), "intercept": float(intercept)}}
    result = {"model": model_summaries, "baselines": baselines}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=HERE / "configs" / "stage1_aux.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "stage1_aux_learnability")
    ap.add_argument("--pair-root", type=Path); ap.add_argument("--epochs", type=int)
    ap.add_argument("--train-subset-mazes", type=int); ap.add_argument("--val-subset-mazes", type=int)
    ap.add_argument("--test-subset-mazes", type=int)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    if args.epochs: cfg["epochs"] = args.epochs
    if args.train_subset_mazes: cfg["train_subset_mazes"] = args.train_subset_mazes
    if args.val_subset_mazes: cfg["val_subset_mazes"] = args.val_subset_mazes
    if args.test_subset_mazes: cfg["test_subset_mazes"] = args.test_subset_mazes
    pair_root = args.pair_root or args.out; args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    summaries, train_pairs, test_pairs = train(cfg, args.out, pair_root)
    print(json.dumps(baselines_and_report(cfg, args.out, train_pairs, test_pairs, summaries), indent=2))


if __name__ == "__main__": main()
