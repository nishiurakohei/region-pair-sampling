"""Train the unchanged Phase-1 model family as normalized beta1 regressors."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

P2 = Path(__file__).resolve().parent
ROOT = P2.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import MazeData
from models import build, count_params
from nn import Adam

OUT = ROOT / "results_phase2" / "aux_only"


def evaluate(model, X, target, bs=256):
    pred = np.empty(len(target), dtype=np.float32)
    for s in range(0, len(target), bs):
        pred[s:s + bs] = model.forward(X[s:s + bs], train=False)
    return float(np.mean((pred - target) ** 2)), pred


def save(path, model, opt, meta, best, best_bn):
    d = {f"P::{k}": v for k, v in model.params().items()}
    d.update({f"BN::{k}": v for k, v in model.bn_state().items()})
    d.update({f"OPT::{k}": v for k, v in opt.state().items()})
    d.update({f"BEST::{k}": v for k, v in best.items()})
    d.update({f"BESTBN::{k}": v for k, v in best_bn.items()})
    d["META"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **d)


def load(path, model, opt):
    z = np.load(path, allow_pickle=False)
    for k, v in model.params().items(): v[...] = z[f"P::{k}"]
    model.load_bn_state({k[4:]: z[k] for k in z.files if k.startswith("BN::")})
    opt.load_state({k[5:]: z[k] for k in z.files if k.startswith("OPT::")})
    return (json.loads(bytes(z["META"]).decode()),
            {k[6:]: z[k] for k in z.files if k.startswith("BEST::")},
            {k[8:]: z[k] for k in z.files if k.startswith("BESTBN::")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["cnn", "resnet"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--budget", type=float, default=150)
    a = ap.parse_args()
    cfg = json.loads((P2 / "config.json").read_text())
    ckdir = OUT / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / f"pred_{a.model}_s{a.seed}.npz"
    if pred_path.exists():
        print("ALREADY DONE"); return
    ckpt = ckdir / f"train_{a.model}_s{a.seed}.npz"
    tr, va, te = [MazeData("H3", s) for s in ("train", "val", "test")]
    mu, sd = float(tr.arrays["beta1"].mean()), float(tr.arrays["beta1"].std())
    targets = [(d.arrays["beta1"].astype(np.float32) - mu) / sd for d in (tr, va, te)]
    Xtr, Xva = tr.raster_all(), va.raster_all()
    rng = np.random.default_rng(1000 * a.seed + (0 if a.model == "cnn" else 500))
    model = build(a.model, rng)
    opt = Adam(model.params(), lr=cfg["learning_rate"], wd=cfg["weight_decay"])
    if ckpt.exists():
        meta, best, best_bn = load(ckpt, model, opt)
    else:
        meta = {"model": a.model, "seed": a.seed, "epoch": 0, "best_mse": 1e9,
                "best_epoch": -1, "patience": 0, "history": [], "params": count_params(model),
                "target_mean": mu, "target_std": sd}
        best = {k: v.copy() for k, v in model.params().items()}
        best_bn = {k: v.copy() for k, v in model.bn_state().items()}
    order_rng = np.random.default_rng(9999 * a.seed)
    t0, done = time.time(), False
    while meta["epoch"] < cfg["max_epochs"] and time.time() - t0 < a.budget:
        total = 0.0
        order = order_rng.permutation(tr.n)
        for s in range(0, tr.n, cfg["batch_size"]):
            idx = order[s:s + cfg["batch_size"]]
            model.zero_grad()
            pred = model.forward(Xtr[idx], train=True)
            err = pred - targets[0][idx]
            loss = float(np.mean(err ** 2))
            model.backward(2 * err / len(idx))
            opt.step(model.grads())
            total += loss * len(idx)
        meta["epoch"] += 1
        vmse, _ = evaluate(model, Xva, targets[1])
        meta["history"].append({"epoch": meta["epoch"], "train_mse": total / tr.n, "val_mse": vmse})
        if vmse < meta["best_mse"] - 1e-4:
            meta["best_mse"], meta["best_epoch"], meta["patience"] = vmse, meta["epoch"], 0
            best = {k: v.copy() for k, v in model.params().items()}
            best_bn = {k: v.copy() for k, v in model.bn_state().items()}
        else:
            meta["patience"] += 1
        print(f"[aux {a.model} s{a.seed}] ep{meta['epoch']} tr={total/tr.n:.4f} "
              f"val={vmse:.4f} best={meta['best_mse']:.4f}@{meta['best_epoch']}", flush=True)
        done = meta["patience"] >= cfg["patience"] or meta["epoch"] >= cfg["max_epochs"]
        save(ckpt, model, opt, meta, best, best_bn)
        if done: break
    if not done:
        print("PARTIAL", meta["epoch"]); return
    for k, v in model.params().items(): v[...] = best[k]
    model.load_bn_state(best_bn)
    Xte = te.raster_all()
    _, pred_z = evaluate(model, Xte, targets[2])
    np.savez_compressed(pred_path, pred_beta1=pred_z * sd + mu, true_beta1=te.arrays["beta1"],
                        mean=mu, std=sd, best_epoch=meta["best_epoch"], history=json.dumps(meta["history"]))
    print("DONE", pred_path)


if __name__ == "__main__":
    main()
