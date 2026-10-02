"""Scratch-train the published DT-Net with the official Deep Thinking recipe."""
import argparse, csv, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parents[1]
UP = HERE / "upstream" / "maze-extrapolation"
sys.path.insert(0, str(UP))
from src.models.dt_net import DTNet


class MazeArrays(Dataset):
    def __init__(self, root):
        self.x = np.load(root / "inputs.npy", mmap_mode="r")
        self.y = np.load(root / "solutions.npy", mmap_mode="r")

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i].astype(np.float32)), torch.from_numpy(self.y[i].astype(np.int64))


def state_hash(model):
    h = hashlib.sha256()
    for value in model.state_dict().values(): h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def exact_and_loss(model, loader, device, iterations):
    model.eval(); total = correct = 0; loss_sum = 0.0; pixel_count = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(memory_format=torch.channels_last).to(device); y = y.to(device)
            logits, _ = model(x, iterations)
            flat_logits, flat_y = logits.flatten(2), y.flatten(1)
            mask = x.flatten(2).amax(1) > 0
            losses = torch.nn.functional.cross_entropy(flat_logits, flat_y, reduction="none")
            loss_sum += losses[mask].sum().item(); pixel_count += mask.sum().item()
            pred = logits.argmax(1) * mask.reshape_as(y)
            correct += (pred == y).flatten(1).all(1).sum().item(); total += len(x)
    return loss_sum / pixel_count, correct / total


def run_seed(cfg, seed_index, data, train_indices, val_indices, out, device, epoch_limit=None, sample_limit=None):
    model_seed = cfg["model_seeds"][seed_index]
    shuffle_seed = cfg["shuffle_seeds"][seed_index]
    progressive_seed = cfg["progressive_seeds"][seed_index]
    torch.manual_seed(model_seed); np.random.seed(model_seed); random.seed(model_seed)
    model = DTNet().to(memory_format=torch.channels_last).to(device)
    if model.projection[0].out_channels != cfg["width"]:
        raise RuntimeError("DT-Net width differs from the recorded configuration")
    initial_hash = state_hash(model)
    pretrained = torch.load(UP / "models" / "dt_net.pth", map_location="cpu", weights_only=True)["net"]
    pretrained_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in pretrained.values())).hexdigest()
    if initial_hash == pretrained_hash: raise RuntimeError("scratch initialization unexpectedly matches pretrained weights")

    base = [p for n, p in model.named_parameters() if "recur" not in n]
    recur = [p for n, p in model.named_parameters() if "recur" in n]
    optimizer = torch.optim.Adam(
        [{"params": base}, {"params": recur, "lr": cfg["base_lr"] / cfg["recurrent_lr_divisor"]}],
        lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg["lr_milestones"], gamma=cfg["lr_factor"])
    base_lrs = [cfg["base_lr"], cfg["base_lr"] / cfg["recurrent_lr_divisor"]]
    for group, lr in zip(optimizer.param_groups, base_lrs): group["lr"] = lr * (1 - math.exp(-1 / cfg["warmup_epochs"]))

    selected = train_indices[:sample_limit] if sample_limit else train_indices
    generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(Subset(data, selected), batch_size=cfg["batch_size"], shuffle=True,
                              generator=generator, num_workers=0, drop_last=True)
    val_loader = DataLoader(Subset(data, val_indices), batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=0)
    rng = random.Random(progressive_seed)
    seed_dir = out / "checkpoints" / f"seed_{model_seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = seed_dir / "last.pt", seed_dir / "best.pt"
    rows, start_epoch, best_acc, best_loss, best_epoch = [], 0, -1.0, float("inf"), None
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        rng.setstate(saved["progressive_rng"]); generator.set_state(saved["shuffle_rng"])
        rows, start_epoch, best_acc = saved["rows"], saved["epoch"] + 1, saved["best_val_accuracy"]
        best_loss = saved.get("best_val_loss", float("inf")); best_epoch = saved.get("best_epoch")
    epochs = epoch_limit or cfg["epochs"]
    alpha, max_iters = cfg["progressive_alpha"], cfg["train_iterations"]
    for epoch in range(start_epoch, epochs):
        started = time.time(); model.train(); loss_sum = correct = total = 0
        for x, y in train_loader:
            x = x.to(memory_format=torch.channels_last).to(device); y = y.to(device)
            flat_y = y.flatten(1); mask = x.flatten(2).amax(1) > 0
            optimizer.zero_grad(set_to_none=True)
            n = rng.randrange(0, max_iters); k = rng.randrange(1, max_iters - n + 1)
            interim = model.projection(x)
            detached_at_n = None
            for step in range(max_iters):
                interim = model.recur_block(torch.cat([interim, x], 1))
                if step + 1 == n: detached_at_n = interim.detach()
            full_logits = model.head(interim)
            full_loss = torch.nn.functional.cross_entropy(full_logits.flatten(2), flat_y, reduction="none")[mask].mean()
            prog_logits, _ = model(x, k, interim_thought=detached_at_n)
            prog_loss = torch.nn.functional.cross_entropy(prog_logits.flatten(2), flat_y, reduction="none")[mask].mean()
            loss = (1 - alpha) * full_loss + alpha * prog_loss
            loss.backward(); optimizer.step(); loss_sum += loss.item()
            pred = full_logits.argmax(1) * mask.reshape_as(y)
            correct += (pred == y).flatten(1).all(1).sum().item(); total += len(x)
        val_loss, val_acc = exact_and_loss(model, val_loader, device, max_iters)
        row = {"model_seed": model_seed, "data_split_seed": cfg["data_split_seed"], "shuffle_seed": shuffle_seed,
               "progressive_seed": progressive_seed, "epoch": epoch + 1, "train_loss": loss_sum / len(train_loader),
               "train_exact_accuracy": correct / total, "validation_loss": val_loss,
               "validation_exact_accuracy": val_acc, "seconds": time.time() - started,
               "base_lr": optimizer.param_groups[0]["lr"], "recurrent_lr": optimizer.param_groups[1]["lr"]}
        rows.append(row); print(json.dumps(row), flush=True)
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        if improved: best_acc, best_loss, best_epoch = val_acc, val_loss, epoch + 1
        scheduler.step()
        step = epoch + 1
        if step < cfg["warmup_epochs"]:
            factor = 1 - math.exp(-(step + 1) / cfg["warmup_epochs"])
            for group, lr in zip(optimizer.param_groups, base_lrs): group["lr"] = factor * lr
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "progressive_rng": rng.getstate(), "shuffle_rng": generator.get_state(), "rows": rows,
                 "epoch": epoch, "best_val_accuracy": best_acc, "initial_weight_sha256": initial_hash,
                 "best_val_loss": best_loss, "best_epoch": best_epoch,
                 "pretrained_loaded": False, "model_seed": model_seed, "device": str(device)}
        torch.save(state, last_path)
        if improved: torch.save(state, best_path)
        if epoch + 1 in cfg.get("checkpoint_epochs", []):
            torch.save(state, seed_dir / f"epoch_{epoch + 1:03d}.pt")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "stage0_5_training.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "stage0_5_training")
    ap.add_argument("--seed-index", type=int, action="append")
    ap.add_argument("--epochs", type=int); ap.add_argument("--max-train-samples", type=int)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    data_root = HERE / "data" / "easy-to-hard-data" / "maze_data_train_9"
    data = MazeArrays(data_root)
    split = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg["data_split_seed"])).tolist()
    train_indices, val_indices = split[:cfg["train_examples"]], split[cfg["train_examples"]:cfg["train_examples"] + cfg["val_examples"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(json.dumps({"device": str(device), "torch": torch.__version__}), flush=True)
    all_rows = []
    for i in (args.seed_index if args.seed_index is not None else range(len(cfg["model_seeds"]))):
        all_rows.extend(run_seed(cfg, i, data, train_indices, val_indices, args.out, device, args.epochs, args.max_train_samples))
    if all_rows:
        with (args.out / "per_seed_training_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0]); writer.writeheader(); writer.writerows(all_rows)
    (args.out / "upstream_commit.txt").write_text(subprocess.check_output(["git", "-C", str(UP), "rev-parse", "HEAD"], text=True))


if __name__ == "__main__": main()
