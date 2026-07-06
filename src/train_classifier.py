from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.dataset import load_split_arrays
from connectivity_experiment.difficulty import attach_difficulty_proxy
from connectivity_experiment.io_utils import write_csv
from connectivity_experiment.model import (
    LargeReceptiveFieldCNN,
    SmallCNN,
    binary_cross_entropy_with_logits,
    per_sample_binary_loss,
    sigmoid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNNs for curved connectivity classification.")
    parser.add_argument("--config", default="configs/curved_v1.yaml")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", choices=["small", "large"], default="small")
    parser.add_argument("--drop-label-disagreement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    model_dir = repo_root / config["paths"]["model_dir"]
    metric_dir = repo_root / config["paths"]["metric_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_records = load_split_arrays(
        repo_root,
        "train",
        input_size=args.input_size,
        drop_label_disagreement=args.drop_label_disagreement,
    )
    val_x, val_y, _ = load_split_arrays(
        repo_root,
        "val",
        input_size=args.input_size,
        drop_label_disagreement=args.drop_label_disagreement,
    )
    test_x, test_y, test_records = load_split_arrays(
        repo_root,
        "test",
        input_size=args.input_size,
        drop_label_disagreement=args.drop_label_disagreement,
    )

    model = _build_model(args.model, input_size=args.input_size, seed=int(config["experiment"]["seed"]), input_channels=train_x.shape[1])
    best_val_loss = float("inf")
    best_model_path = model_dir / f"{args.model}_cnn_best.npz"
    history = []
    rng = np.random.default_rng(int(config["experiment"]["seed"]))

    for epoch in range(args.epochs):
        indices = rng.permutation(len(train_x))
        train_loss_total = 0.0
        train_correct = 0

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            logits = model.forward(batch_x)
            loss, grad = binary_cross_entropy_with_logits(logits, batch_y)
            model.backward(grad)
            model.step(args.learning_rate, weight_decay=args.weight_decay)
            probs = sigmoid(logits.reshape(-1))
            train_loss_total += loss * len(batch_indices)
            train_correct += int(np.sum((probs >= 0.5) == (batch_y >= 0.5)))

        train_loss = train_loss_total / len(train_x)
        train_acc = train_correct / len(train_x)
        val_loss, val_acc = evaluate(model, val_x, val_y, batch_size=args.batch_size)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
            }
        )
        print(
            f"[train_classifier] model={args.model} epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(best_model_path))

    model.load(str(best_model_path))
    test_logits = model.predict_logits(test_x, batch_size=args.batch_size).reshape(-1)
    test_probs = sigmoid(test_logits)
    test_losses = per_sample_binary_loss(test_logits, test_y)

    attach_difficulty_proxy(test_records)
    for index, record in enumerate(test_records):
        label = int(float(record["connected"]))
        prob = float(test_probs[index])
        pred_label = int(prob >= 0.5)
        correct_prob = prob if label == 1 else 1.0 - prob
        record["pred_prob"] = round(prob, 6)
        record["pred_label"] = pred_label
        record["is_correct"] = int(pred_label == label)
        record["loss"] = round(float(test_losses[index]), 6)
        record["confidence_margin"] = round(abs(correct_prob - 0.5), 6)
        record["pixel_connected_pred_4"] = int(record["connected_pixel_4"])
        record["pixel_connected_pred_8"] = int(record["connected_pixel_8"])
        record["geometry_connected_pred"] = int(record["connected_geometry"])

    metric_stem = f"{args.model}_test_predictions"
    write_csv(metric_dir / f"{metric_stem}.csv", test_records)
    (metric_dir / f"{args.model}_training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata_accuracy = _metadata_only_baseline(train_records, test_records)
    test_acc = sum(int(record["is_correct"]) for record in test_records) / len(test_records)
    pixel_acc_4 = sum(int(record["pixel_connected_pred_4"] == int(record["connected"])) for record in test_records) / len(test_records)
    pixel_acc_8 = sum(int(record["pixel_connected_pred_8"] == int(record["connected"])) for record in test_records) / len(test_records)
    shape_family_accuracy = _accuracy_by_key(test_records, "shape_family")
    query_case_accuracy = _accuracy_by_key(test_records, "query_case")
    negative_mode_accuracy = _accuracy_by_key([record for record in test_records if int(record["connected"]) == 0], "negative_mode")
    difficulty_loss_corr = _correlation([float(record["difficulty_proxy"]) for record in test_records], [float(record["loss"]) for record in test_records])

    summary = {
        "model": args.model,
        "test_accuracy": round(test_acc, 6),
        "pixel_baseline_accuracy_4": round(pixel_acc_4, 6),
        "pixel_baseline_accuracy_8": round(pixel_acc_8, 6),
        "geometry_baseline_accuracy": 1.0,
        "metadata_only_accuracy": round(metadata_accuracy, 6),
        "best_val_loss": round(best_val_loss, 6),
        "input_size": args.input_size,
        "epochs": args.epochs,
        "drop_label_disagreement": args.drop_label_disagreement,
        "shape_family_accuracy": shape_family_accuracy,
        "query_case_accuracy": query_case_accuracy,
        "negative_mode_accuracy": negative_mode_accuracy,
        "difficulty_proxy_loss_correlation": round(difficulty_loss_corr, 6),
    }
    (metric_dir / f"{args.model}_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[train_classifier] model={args.model} test_accuracy={test_acc:.4f} "
        f"pixel4={pixel_acc_4:.4f} pixel8={pixel_acc_8:.4f} metadata={metadata_accuracy:.4f}"
    )


def evaluate(model, features: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[float, float]:
    logits = model.predict_logits(features, batch_size=batch_size).reshape(-1)
    loss = float(np.mean(per_sample_binary_loss(logits, labels)))
    probs = sigmoid(logits)
    accuracy = float(np.mean((probs >= 0.5) == (labels >= 0.5)))
    return loss, accuracy


def _build_model(model_name: str, input_size: int, seed: int, input_channels: int):
    if model_name == "small":
        return SmallCNN(input_size=input_size, seed=seed, input_channels=input_channels)
    return LargeReceptiveFieldCNN(input_size=input_size, seed=seed, input_channels=input_channels)


def _accuracy_by_key(records: list[dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[int]] = {}
    for record in records:
        bucket = str(record.get(key, ""))
        buckets.setdefault(bucket, []).append(int(record["is_correct"]))
    return {bucket: round(sum(values) / len(values), 6) for bucket, values in buckets.items() if bucket}


def _correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(values_a), np.asarray(values_b))[0, 1])


def _metadata_only_baseline(train_records: list[dict], test_records: list[dict]) -> float:
    train_x, train_y, enc, means, stds = _metadata_feature_matrix(train_records)
    test_x, test_y, _, _, _ = _metadata_feature_matrix(test_records, family_vocab=enc, means=means, stds=stds)
    weights, bias = _fit_logistic_regression(train_x, train_y)
    probs = sigmoid(test_x @ weights + bias)
    preds = (probs >= 0.5).astype(np.float32)
    return float(np.mean(preds.reshape(-1) == test_y.reshape(-1)))


def _metadata_feature_matrix(
    records: list[dict],
    family_vocab: list[str] | None = None,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    numeric_keys = [
        "n_circles",
        "n_noise_circles",
        "n_distractor_curves",
        "mean_radius",
        "std_radius",
        "target_distance",
        "total_ink_pixels",
        "area",
        "perimeter",
        "compactness",
    ]
    if family_vocab is None:
        family_vocab = sorted({str(record["shape_family"]) for record in records})
    family_index = {name: idx for idx, name in enumerate(family_vocab)}
    matrix = []
    labels = []
    for record in records:
        row = [float(record.get(key, 0.0) or 0.0) for key in numeric_keys]
        one_hot = [0.0] * len(family_vocab)
        family = str(record["shape_family"])
        if family in family_index:
            one_hot[family_index[family]] = 1.0
        matrix.append(row + one_hot)
        labels.append(float(record["connected"]))
    features = np.asarray(matrix, dtype=np.float32)
    if means is None:
        means = features.mean(axis=0, keepdims=True)
    if stds is None:
        stds = features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    return (features - means) / stds, np.asarray(labels, dtype=np.float32), family_vocab, means, stds


def _fit_logistic_regression(features: np.ndarray, labels: np.ndarray, steps: int = 400, learning_rate: float = 0.1) -> tuple[np.ndarray, float]:
    weights = np.zeros((features.shape[1], 1), dtype=np.float32)
    bias = 0.0
    targets = labels.reshape(-1, 1)
    for _ in range(steps):
        logits = features @ weights + bias
        probs = sigmoid(logits)
        error = probs - targets
        grad_w = (features.T @ error) / len(features)
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


if __name__ == "__main__":
    main()
