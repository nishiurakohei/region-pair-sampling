"""Fast, checkpointed DT-Net training with an exactly checked MLX backend."""
import argparse, csv, hashlib, json, math, subprocess, time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np
import torch

from dt_net_mlx import export_torch_state, torch_initialized

HERE = Path(__file__).resolve().parents[1]


def digest(model):
    h = hashlib.sha256()
    for value in export_torch_state(model).values(): h.update(value.numpy().tobytes())
    return h.hexdigest()


def schedule(base_lr, steps_per_epoch, warmup_epochs, milestones, factor):
    def value(step):
        epoch = mx.floor(step.astype(mx.float32) / steps_per_epoch)
        warm = mx.where(epoch < warmup_epochs, 1 - mx.exp(-(epoch + 1) / warmup_epochs), 1.0)
        decay = mx.array(1.0)
        for milestone in milestones: decay = mx.where(epoch >= milestone, decay * factor, decay)
        return base_lr * warm * decay
    return value


def save_tree(path, tree): mx.save_safetensors(str(path), dict(tree_flatten(tree)))


def load_tree(path):
    arrays = mx.load(str(path))
    return tree_unflatten(sorted(arrays.items()))


def batch_arrays(x, y, indices):
    xb = np.asarray(x[indices], dtype=np.float32).transpose(0, 2, 3, 1)
    yb = np.asarray(y[indices], dtype=np.int32)
    return mx.array(xb), mx.array(yb)


def evaluate(model, x, y, indices, batch_size, iterations):
    total_loss = total_pixels = correct = 0.0
    for lo in range(0, len(indices), batch_size):
        xb, yb = batch_arrays(x, y, indices[lo:lo + batch_size])
        logits = model(xb, iterations)[0]
        mask = mx.max(xb, axis=-1) > 0
        losses = mx.take_along_axis(mx.logsumexp(logits, axis=-1, keepdims=True) - logits,
                                    yb[..., None], axis=-1)[..., 0]
        batch_loss = mx.sum(losses * mask); pixels = mx.sum(mask)
        batch_correct = mx.sum(mx.all(mx.argmax(logits, axis=-1) * mask == yb, axis=(1, 2)))
        mx.eval(batch_loss, pixels, batch_correct)
        total_loss += float(batch_loss); total_pixels += float(pixels); correct += float(batch_correct)
    return total_loss / total_pixels, correct / len(indices)


def train_seed(cfg, seed_index, x, y, train_idx, val_idx, out, epochs, sample_limit=None, val_limit=None):
    model_seed = cfg["model_seeds"][seed_index]
    shuffle_seed = cfg["shuffle_seeds"][seed_index]
    progressive_seed = cfg["progressive_seeds"][seed_index]
    model = torch_initialized(model_seed)
    initial_hash = digest(model)
    selected = np.asarray(train_idx[:sample_limit] if sample_limit else train_idx)
    validation = np.asarray(val_idx[:val_limit] if val_limit else val_idx)
    steps_per_epoch = len(selected) // cfg["batch_size"]
    base_opt = optim.Adam(schedule(cfg["base_lr"], steps_per_epoch, cfg["warmup_epochs"],
                                   cfg["lr_milestones"], cfg["lr_factor"]), bias_correction=True)
    recur_opt = optim.Adam(schedule(cfg["base_lr"] / cfg["recurrent_lr_divisor"], steps_per_epoch,
                                    cfg["warmup_epochs"], cfg["lr_milestones"], cfg["lr_factor"]),
                           bias_correction=True)
    seed_dir = out / "checkpoints" / f"seed_{model_seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = seed_dir / "last.json"
    shuffle_rng, progressive_rng = np.random.default_rng(shuffle_seed), np.random.default_rng(progressive_seed)
    rows, start_epoch, best_acc = [], 0, -1.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text()); model.load_weights(str(seed_dir / "last.safetensors"))
        base_opt.state = load_tree(seed_dir / "base_optimizer.safetensors")
        recur_opt.state = load_tree(seed_dir / "recur_optimizer.safetensors")
        shuffle_rng.bit_generator.state = meta["shuffle_rng"]; progressive_rng.bit_generator.state = meta["progressive_rng"]
        rows, start_epoch, best_acc = meta["rows"], meta["epoch"] + 1, meta["best_val_accuracy"]

    alpha, max_iters, weight_decay = cfg["progressive_alpha"], cfg["train_iterations"], cfg["weight_decay"]
    def loss_fn(params, xb, yb, n, k):
        model.update(params)
        full_logits, _, captured = model(xb, max_iters, capture=n)
        prog_logits = model(xb, k, interim=None if n == 0 else mx.stop_gradient(captured))[0]
        mask = mx.max(xb, axis=-1) > 0
        full_losses = mx.take_along_axis(mx.logsumexp(full_logits, axis=-1, keepdims=True) - full_logits,
                                         yb[..., None], axis=-1)[..., 0]
        prog_losses = mx.take_along_axis(mx.logsumexp(prog_logits, axis=-1, keepdims=True) - prog_logits,
                                         yb[..., None], axis=-1)[..., 0]
        full_loss = mx.sum(full_losses * mask) / mx.sum(mask)
        prog_loss = mx.sum(prog_losses * mask) / mx.sum(mask)
        loss = (1 - alpha) * full_loss + alpha * prog_loss
        correct = mx.sum(mx.all(mx.argmax(full_logits, axis=-1) * mask == yb, axis=(1, 2)))
        return loss, correct, full_loss, prog_loss

    value_grad = mx.value_and_grad(loss_fn)
    for epoch in range(start_epoch, epochs):
        started = time.time(); order = shuffle_rng.permutation(selected); loss_sum = correct = total = 0.0
        for lo in range(0, steps_per_epoch * cfg["batch_size"], cfg["batch_size"]):
            xb, yb = batch_arrays(x, y, order[lo:lo + cfg["batch_size"]])
            n = int(progressive_rng.integers(0, max_iters)); k = int(progressive_rng.integers(1, max_iters - n + 1))
            (values, grads) = value_grad(model.trainable_parameters(), xb, yb, n, k)
            base_grads = tree_map(lambda g, p: g + weight_decay * p, grads["base"], model.base.trainable_parameters())
            recur_grads = tree_map(lambda g, p: g + weight_decay * p, grads["recur"], model.recur.trainable_parameters())
            base_opt.update(model.base, base_grads); recur_opt.update(model.recur, recur_grads)
            mx.eval(values, model.parameters(), base_opt.state, recur_opt.state)
            loss_sum += float(values[0]); correct += float(values[1]); total += cfg["batch_size"]
        val_loss, val_acc = evaluate(model, x, y, validation, cfg["eval_batch_size"], max_iters)
        row = {"model_seed": model_seed, "data_split_seed": cfg["data_split_seed"], "shuffle_seed": shuffle_seed,
               "progressive_seed": progressive_seed, "epoch": epoch + 1, "train_loss": loss_sum / steps_per_epoch,
               "train_exact_accuracy": correct / total, "validation_loss": val_loss,
               "validation_exact_accuracy": val_acc, "seconds": time.time() - started,
               "base_lr": float(base_opt.learning_rate), "recurrent_lr": float(recur_opt.learning_rate),
               "backend": "mlx-0.29.3"}
        rows.append(row); print(json.dumps(row), flush=True)
        improved = val_acc > best_acc
        if improved: best_acc = val_acc; model.save_weights(str(seed_dir / "best.safetensors"))
        model.save_weights(str(seed_dir / "last.safetensors")); save_tree(seed_dir / "base_optimizer.safetensors", base_opt.state)
        save_tree(seed_dir / "recur_optimizer.safetensors", recur_opt.state)
        meta = {"epoch": epoch, "best_val_accuracy": best_acc, "rows": rows,
                "shuffle_rng": shuffle_rng.bit_generator.state, "progressive_rng": progressive_rng.bit_generator.state,
                "initial_weight_sha256": initial_hash, "pretrained_loaded": False, "model_seed": model_seed}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    model.load_weights(str(seed_dir / "best.safetensors"))
    torch.save({"net": export_torch_state(model), "model_seed": model_seed, "pretrained_loaded": False,
                "initial_weight_sha256": initial_hash, "best_val_accuracy": best_acc}, seed_dir / "best_torch.pt")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "stage0_5_training.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "stage0_5_training")
    ap.add_argument("--seed-index", type=int, action="append"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-samples", type=int); ap.add_argument("--max-val-samples", type=int)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); epochs = args.epochs or cfg["epochs"]
    args.out.mkdir(parents=True, exist_ok=True); effective = dict(cfg, backend="mlx-0.29.3", epochs_run=epochs,
        max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)
    (args.out / "config.json").write_text(json.dumps(effective, indent=2) + "\n")
    root = HERE / "data" / "easy-to-hard-data" / "maze_data_train_9"
    x, y = np.load(root / "inputs.npy", mmap_mode="r"), np.load(root / "solutions.npy", mmap_mode="r")
    split = torch.randperm(len(x), generator=torch.Generator().manual_seed(cfg["data_split_seed"])).numpy()
    train_idx = split[:cfg["train_examples"]]; val_idx = split[cfg["train_examples"]:cfg["train_examples"] + cfg["val_examples"]]
    all_rows = []
    for i in (args.seed_index if args.seed_index is not None else range(len(cfg["model_seeds"]))):
        all_rows.extend(train_seed(cfg, i, x, y, train_idx, val_idx, args.out, epochs,
                                   args.max_train_samples, args.max_val_samples))
    if all_rows:
        with (args.out / "per_seed_training_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0]); writer.writeheader(); writer.writerows(all_rows)
    (args.out / "upstream_commit.txt").write_text(subprocess.check_output(
        ["git", "-C", str(HERE / "upstream" / "maze-extrapolation"), "rev-parse", "HEAD"], text=True))


if __name__ == "__main__": main()
