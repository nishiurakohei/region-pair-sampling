"""Train one Phase-1 CNN/ResNet baseline on a fully observed hardness level."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dataset import MazeData
from models import build, count_params
from nn import Adam, bce_with_logits

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_hard_maze"


def evaluate(model, X, y, bs=256):
    logits = np.empty(len(y), dtype=np.float32)
    for s in range(0, len(y), bs):
        logits[s:s + bs] = model.forward(X[s:s + bs], train=False)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    loss, _ = bce_with_logits(logits, y)
    acc = ((p > 0.5) == y).mean()
    return float(loss), float(acc), p


def save_checkpoint(path, model, opt, meta, best_params, best_bn):
    d = {f"P::{k}": v for k, v in model.params().items()}
    d.update({f"BN::{k}": v for k, v in model.bn_state().items()})
    d.update({f"OPT::{k}": v for k, v in opt.state().items()})
    d.update({f"BEST::{k}": v for k, v in best_params.items()})
    d.update({f"BESTBN::{k}": v for k, v in best_bn.items()})
    d["META"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **d)


def load_checkpoint(path, model, opt):
    z = np.load(path, allow_pickle=False)
    for k, v in model.params().items():
        v[...] = z[f"P::{k}"]
    model.load_bn_state({k[4:]: z[k] for k in z.files if k.startswith("BN::")})
    opt.load_state({k[5:]: z[k] for k in z.files if k.startswith("OPT::")})
    meta = json.loads(bytes(z["META"]).decode())
    best = {k[6:]: z[k] for k in z.files if k.startswith("BEST::")}
    best_bn = {k[8:]: z[k] for k in z.files if k.startswith("BESTBN::")}
    return meta, best, best_bn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["H0", "H1", "H2", "H3"])
    parser.add_argument("model", choices=["cnn", "resnet"])
    parser.add_argument("seed", type=int)
    parser.add_argument("--budget", type=float, default=150)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text())["training"]
    ckpt = RESULTS / f"train_{args.level}_{args.model}_s{args.seed}.npz"
    pred = RESULTS / f"pred_{args.level}_{args.model}_s{args.seed}.npz"
    if pred.exists():
        print("ALREADY DONE", pred)
        return

    rng = np.random.default_rng(1000 * args.seed + (0 if args.model == "cnn" else 500))
    model = build(args.model, rng)
    opt = Adam(model.params(), lr=cfg["learning_rate"], wd=cfg["weight_decay"])
    if ckpt.exists():
        meta, best_params, best_bn = load_checkpoint(ckpt, model, opt)
    else:
        meta = {"level": args.level, "model": args.model, "seed": args.seed,
                "epoch": 0, "best_bce": 1e9, "best_epoch": -1,
                "patience_counter": 0, "history": [], "params": count_params(model)}
        best_params = {k: v.copy() for k, v in model.params().items()}
        best_bn = {k: v.copy() for k, v in model.bn_state().items()}

    tr, va = MazeData(args.level, "train"), MazeData(args.level, "val")
    Xtr, Xva = tr.raster_all(), va.raster_all()
    perm_rng = np.random.default_rng(9999 * args.seed)
    t0 = time.time()
    done = False
    while meta["epoch"] < cfg["max_epochs"] and time.time() - t0 < args.budget:
        order = perm_rng.permutation(tr.n)
        total = 0.0
        for s in range(0, tr.n, cfg["batch_size"]):
            idx = order[s:s + cfg["batch_size"]]
            model.zero_grad()
            logits = model.forward(Xtr[idx], train=True)
            loss, grad = bce_with_logits(logits, tr.y[idx])
            model.backward(grad)
            opt.step(model.grads())
            total += loss * len(idx)
        meta["epoch"] += 1
        vbce, vacc, _ = evaluate(model, Xva, va.y)
        meta["history"].append({"epoch": meta["epoch"], "train_bce": float(total / tr.n),
                                "val_bce": vbce, "val_acc": vacc})
        if vbce < meta["best_bce"] - 1e-4:
            meta["best_bce"], meta["best_epoch"], meta["patience_counter"] = vbce, meta["epoch"], 0
            best_params = {k: v.copy() for k, v in model.params().items()}
            best_bn = {k: v.copy() for k, v in model.bn_state().items()}
        else:
            meta["patience_counter"] += 1
        print(f"[{args.level} {args.model} s{args.seed}] ep{meta['epoch']} "
              f"tr={total/tr.n:.4f} val={vbce:.4f} acc={vacc:.4f} "
              f"best={meta['best_bce']:.4f}@{meta['best_epoch']}", flush=True)
        done = meta["patience_counter"] >= cfg["patience"] or meta["epoch"] >= cfg["max_epochs"]
        save_checkpoint(ckpt, model, opt, meta, best_params, best_bn)
        if done:
            break
    if not done:
        print("PARTIAL", meta["epoch"])
        return
    for k, v in model.params().items():
        v[...] = best_params[k]
    model.load_bn_state(best_bn)
    te = MazeData(args.level, "test")
    Xte = te.raster_all()
    tbce, tacc, prob = evaluate(model, Xte, te.y)
    vbce, vacc, _ = evaluate(model, Xva, va.y)
    np.savez_compressed(pred, prob=prob, y=te.y, pair_id=te.arrays["pair_id"],
                        test_bce=tbce, test_acc=tacc, val_bce=vbce, val_acc=vacc,
                        best_epoch=meta["best_epoch"], params=meta["params"])
    print(f"DONE test_acc={tacc:.4f} test_bce={tbce:.4f}", flush=True)


if __name__ == "__main__":
    main()
