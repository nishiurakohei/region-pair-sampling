"""Train one H3 main/true-beta1/shuffled-beta1 configuration."""
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
from nn import Adam, bce_with_logits
from multitask_models import build

OUT = ROOT / "results_phase2" / "multitask"


def tag(value): return str(value).replace(".", "p")


def evaluate(model, X, y, z, bs=256):
    logits, aux = np.empty(len(y), np.float32), np.empty(len(y), np.float32)
    for s in range(0, len(y), bs):
        logits[s:s + bs], aux[s:s + bs] = model.forward(X[s:s + bs], train=False)
    p = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    bce, _ = bce_with_logits(logits, y)
    return float(bce), float(np.mean((aux - z) ** 2)), p, aux


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
    ap.add_argument("condition", choices=["main", "true", "shuffled"])
    ap.add_argument("lambda_", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--budget", type=float, default=150)
    a = ap.parse_args()
    if (a.condition == "main") != (a.lambda_ == 0):
        raise ValueError("main requires lambda=0; auxiliary conditions require lambda>0")
    cfg = json.loads((P2 / "config.json").read_text())
    name = f"{a.model}_{a.condition}_l{tag(a.lambda_)}_s{a.seed}"
    ckdir = OUT / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    pred_path, ckpt = OUT / f"pred_{name}.npz", ckdir / f"train_{name}.npz"
    if pred_path.exists(): print("ALREADY DONE"); return
    tr, va, te = [MazeData("H3", s) for s in ("train", "val", "test")]
    mu, sd = float(tr.arrays["beta1"].mean()), float(tr.arrays["beta1"].std())
    ztr = (tr.arrays["beta1"].astype(np.float32) - mu) / sd
    zva = (va.arrays["beta1"].astype(np.float32) - mu) / sd
    zte = (te.arrays["beta1"].astype(np.float32) - mu) / sd
    if a.condition == "shuffled":
        shuffle_seed = cfg["shuffle_seed"] + a.seed
        ztr_used = ztr[np.random.default_rng(shuffle_seed).permutation(tr.n)]
    else:
        shuffle_seed, ztr_used = None, ztr
    Xtr, Xva = tr.raster_all(), va.raster_all()
    rng = np.random.default_rng(1000 * a.seed + (0 if a.model == "cnn" else 500))
    model = build(a.model, rng)
    opt = Adam(model.params(), lr=cfg["learning_rate"], wd=cfg["weight_decay"])
    if ckpt.exists():
        meta, best, best_bn = load(ckpt, model, opt)
    else:
        meta = {"model": a.model, "condition": a.condition, "lambda": a.lambda_, "seed": a.seed,
                "shuffle_seed": shuffle_seed, "epoch": 0, "best_main_bce": 1e9,
                "best_epoch": -1, "patience": 0, "history": [], "target_mean": mu, "target_std": sd}
        best = {k: v.copy() for k, v in model.params().items()}
        best_bn = {k: v.copy() for k, v in model.bn_state().items()}
    order_rng = np.random.default_rng(9999 * a.seed)
    t0, done = time.time(), False
    while meta["epoch"] < cfg["max_epochs"] and time.time() - t0 < a.budget:
        order, main_total, aux_total = order_rng.permutation(tr.n), 0.0, 0.0
        for s in range(0, tr.n, cfg["batch_size"]):
            idx = order[s:s + cfg["batch_size"]]
            model.zero_grad()
            logits, aux = model.forward(Xtr[idx], train=True)
            main_loss, gr = bce_with_logits(logits, tr.y[idx])
            err = aux - ztr_used[idx]
            aux_loss = float(np.mean(err ** 2))
            ga = (2 * a.lambda_ / len(idx)) * err
            model.backward(gr, ga)
            opt.step(model.grads())
            main_total += main_loss * len(idx); aux_total += aux_loss * len(idx)
        meta["epoch"] += 1
        vbce, vaux, _, _ = evaluate(model, Xva, va.y, zva)
        meta["history"].append({"epoch": meta["epoch"], "train_main_bce": float(main_total / tr.n),
                                "train_aux_mse": float(aux_total / tr.n), "val_main_bce": vbce,
                                "val_aux_mse_true": vaux})
        if vbce < meta["best_main_bce"] - 1e-4:
            meta["best_main_bce"], meta["best_epoch"], meta["patience"] = vbce, meta["epoch"], 0
            best = {k: v.copy() for k, v in model.params().items()}
            best_bn = {k: v.copy() for k, v in model.bn_state().items()}
        else: meta["patience"] += 1
        print(f"[{name}] ep{meta['epoch']} main={main_total/tr.n:.4f}/{vbce:.4f} "
              f"aux={aux_total/tr.n:.4f}/{vaux:.4f} best={meta['best_main_bce']:.4f}", flush=True)
        done = meta["patience"] >= cfg["patience"] or meta["epoch"] >= cfg["max_epochs"]
        save(ckpt, model, opt, meta, best, best_bn)
        if done: break
    if not done: print("PARTIAL", meta["epoch"]); return
    for k, v in model.params().items(): v[...] = best[k]
    model.load_bn_state(best_bn)
    Xte = te.raster_all()
    _, _, prob, aux_z = evaluate(model, Xte, te.y, zte)
    np.savez_compressed(pred_path, prob=prob, y=te.y, pair_id=te.arrays["pair_id"],
                        pred_beta1=aux_z * sd + mu, true_beta1=te.arrays["beta1"],
                        best_epoch=meta["best_epoch"], history=json.dumps(meta["history"]),
                        shuffle_seed=-1 if shuffle_seed is None else shuffle_seed)
    print("DONE", pred_path)


if __name__ == "__main__":
    main()
