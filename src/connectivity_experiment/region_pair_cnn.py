from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from connectivity_experiment.io_utils import ensure_dir, read_csv, write_csv, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


def load_pair_records(path: str | Path) -> list[dict]:
    return read_csv(path)


def normalize_input_mode(input_mode: str) -> str:
    aliases = {
        "mask_point_i_point_j": "mask_point_i_point_j",
        "mask+point_i+point_j": "mask_point_i_point_j",
        "3ch": "mask_point_i_point_j",
        "point_i_point_j_only": "point_i_point_j_only",
        "point_i+point_j_only": "point_i_point_j_only",
        "point_i+point_j": "point_i_point_j_only",
        "mask_only": "mask_only",
        "mask": "mask_only",
        "mask_merged_points": "mask_merged_points",
        "mask+merged_points": "mask_merged_points",
        "mask+merged_point": "mask_merged_points",
        "mask_point_i_point_j_line": "mask_point_i_point_j_line",
        "mask+point_i+point_j+line": "mask_point_i_point_j_line",
        "4ch_line": "mask_point_i_point_j_line",
    }
    normalized = aliases.get(str(input_mode))
    if normalized is None:
        raise ValueError(f"Unknown input mode: {input_mode}")
    return normalized


def input_channels_for_mode(input_mode: str) -> int:
    normalized = normalize_input_mode(input_mode)
    if normalized == "mask_point_i_point_j":
        return 3
    if normalized == "mask_point_i_point_j_line":
        return 4
    if normalized == "point_i_point_j_only":
        return 2
    if normalized == "mask_only":
        return 1
    if normalized == "mask_merged_points":
        return 2
    raise ValueError(f"Unknown input mode: {input_mode}")


def normalize_pos_weight_mode(pos_weight_mode) -> str:
    aliases = {
        "balanced": "balanced",
        "auto": "balanced",
        "class_balanced": "balanced",
        "one": "one",
        "1": "one",
        1: "one",
        1.0: "one",
        "fixed_1": "one",
    }
    normalized = aliases.get(pos_weight_mode, aliases.get(str(pos_weight_mode)))
    if normalized is None:
        raise ValueError(f"Unknown pos_weight mode: {pos_weight_mode}")
    return normalized


def resolve_pos_weight_value(pos_count: int, neg_count: int, pos_weight_mode) -> float:
    normalized = normalize_pos_weight_mode(pos_weight_mode)
    if normalized == "one":
        return 1.0
    return float(max(1, neg_count) / max(1, pos_count))


def assign_image_splits(
    records: list[dict],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, str]:
    image_ids = sorted({str(record["image_id"]) for record in records})
    rng = np.random.default_rng(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    n_images = len(shuffled)
    n_train = max(1, int(round(n_images * train_fraction)))
    n_val = max(1, int(round(n_images * val_fraction)))
    if n_train + n_val >= n_images:
        n_val = max(1, n_images - n_train - 1)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])
    if not test_ids:
        moved = sorted(val_ids)[-1]
        val_ids.remove(moved)
        test_ids.add(moved)
    split_by_image = {}
    for image_id in image_ids:
        if image_id in train_ids:
            split_by_image[image_id] = "train"
        elif image_id in val_ids:
            split_by_image[image_id] = "val"
        else:
            split_by_image[image_id] = "test"
    return split_by_image


class PairRecordDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        input_size: int,
        marker_radius: int,
        input_mode: str = "mask_point_i_point_j",
        line_width: int = 1,
        threshold: int = 128,
    ) -> None:
        self.records = records
        self.input_size = int(input_size)
        self.marker_radius = int(marker_radius)
        self.input_mode = normalize_input_mode(input_mode)
        self.line_width = max(1, int(line_width))
        self.threshold = int(threshold)
        self.mask_cache: dict[str, np.ndarray] = {}
        self.shape_cache: dict[str, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = str(record["image_path"])
        mask = self._load_resized_mask(image_path)
        original_height, original_width = self.shape_cache[image_path]

        point_a = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        point_b = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        row_i, col_i = _scale_point(
            int(record["row_i"]),
            int(record["col_i"]),
            original_height=original_height,
            original_width=original_width,
            target_size=self.input_size,
        )
        row_j, col_j = _scale_point(
            int(record["row_j"]),
            int(record["col_j"]),
            original_height=original_height,
            original_width=original_width,
            target_size=self.input_size,
        )
        _draw_disk(point_a, row_i, col_i, radius=self.marker_radius, value=1.0)
        _draw_disk(point_b, row_j, col_j, radius=self.marker_radius, value=1.0)
        image = _compose_input_channels(
            mask,
            point_a,
            point_b,
            input_mode=self.input_mode,
            line_width=self.line_width,
            start=(row_i, col_i),
            end=(row_j, col_j),
        )
        label = np.float32(float(record["connected"]))
        return torch.from_numpy(image), torch.tensor(label), index

    def _load_resized_mask(self, image_path: str) -> np.ndarray:
        cached = self.mask_cache.get(image_path)
        if cached is not None:
            return cached
        image = Image.open(image_path).convert("L")
        self.shape_cache[image_path] = (image.height, image.width)
        if image.size != (self.input_size, self.input_size):
            image = image.resize((self.input_size, self.input_size), Image.NEAREST)
        mask = (np.asarray(image, dtype=np.uint8) >= self.threshold).astype(np.float32)
        self.mask_cache[image_path] = mask
        return mask


class SimplePairCNN(nn.Module):
    def __init__(self, input_channels: int = 3, pool_size: int = 4) -> None:
        super().__init__()
        self.pool_size = max(1, int(pool_size))
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.pool_size * self.pool_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


@dataclass
class TrainingArtifacts:
    best_model_path: Path
    history: list[dict]
    best_val_loss: float


def build_dataloaders(
    records: list[dict],
    split_by_image: dict[str, str],
    input_size: int,
    marker_radius: int,
    batch_size: int,
    input_mode: str = "mask_point_i_point_j",
    line_width: int = 1,
) -> tuple[dict[str, list[dict]], dict[str, PairRecordDataset], dict[str, DataLoader]]:
    split_records = {
        split: [record for record in records if split_by_image[str(record["image_id"])] == split]
        for split in ["train", "val", "test"]
    }
    datasets = {
        split: PairRecordDataset(
            split_records[split],
            input_size=input_size,
            marker_radius=marker_radius,
            input_mode=input_mode,
            line_width=line_width,
        )
        for split in split_records
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    return split_records, datasets, loaders


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: list[int],
    output_dir: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    pos_weight_mode: str = "balanced",
) -> TrainingArtifacts:
    torch.manual_seed(seed)
    output_dir = ensure_dir(output_dir)
    best_model_path = output_dir / "best_model.pt"
    pos_count = max(1, sum(int(label) for label in train_labels))
    neg_count = max(1, len(train_labels) - pos_count)
    pos_weight_value = resolve_pos_weight_value(pos_count=pos_count, neg_count=neg_count, pos_weight_mode=pos_weight_mode)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict] = []
    best_val_loss = math.inf

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_count = 0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_loss_total += float(loss.item()) * len(labels)
            train_correct += int((preds == labels).sum().item())
            train_count += len(labels)

        train_loss = train_loss_total / max(1, train_count)
        train_accuracy = train_correct / max(1, train_count)
        val_metrics = evaluate_model(model, val_loader, device=device, criterion=criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        print(
            f"[region_pair_cnn] epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return TrainingArtifacts(best_model_path=best_model_path, history=history, best_val_loss=float(best_val_loss))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss() if criterion is None else criterion
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_loss += float(loss.item()) * len(labels)
            total_correct += int((preds == labels).sum().item())
            total_count += len(labels)
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
    }


def predict_with_records(
    model: nn.Module,
    loader: DataLoader,
    records: list[dict],
    device: str,
) -> list[dict]:
    model.eval()
    predictions = [dict(record) for record in records]
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for inputs, labels, indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            losses = criterion(logits, labels)
            preds = (probs >= 0.5).int()
            for offset, record_index in enumerate(indices.tolist()):
                label = int(float(records[record_index]["connected"]))
                prob = float(probs[offset].item())
                pred_label = int(preds[offset].item())
                true_conf = prob if label == 1 else 1.0 - prob
                uncertainty = 1.0 - abs(prob - 0.5) * 2.0
                predictions[record_index]["pred_prob_connected"] = round(prob, 6)
                predictions[record_index]["pred_label"] = pred_label
                predictions[record_index]["is_correct"] = int(pred_label == label)
                predictions[record_index]["loss"] = round(float(losses[offset].item()), 6)
                predictions[record_index]["true_class_confidence"] = round(true_conf, 6)
                predictions[record_index]["confidence_margin"] = round(abs(prob - 0.5), 6)
                predictions[record_index]["uncertainty"] = round(uncertainty, 6)
    return predictions


def summarize_predictions(records: list[dict]) -> dict:
    summary = _performance_metrics(records)
    summary.update(
        {
            "connected_accuracy": summary["positive_accuracy"],
            "pair_region_type_accuracy": _accuracy_by_key(records, "pair_region_type"),
            "sampling_method_accuracy": _accuracy_by_key(records, "sampling_method"),
            "valid_threshold_accuracy": _accuracy_by_key(records, "valid_threshold"),
        }
    )
    return summary


def performance_by_group(records: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        group = str(record.get(key, ""))
        if not group:
            continue
        grouped.setdefault(group, []).append(record)
    rows = []
    for group, subset in sorted(grouped.items()):
        row = {"group_key": key, "group_value": group}
        row.update(_performance_metrics(subset))
        rows.append(row)
    return rows


def train_method_vs_test_method_matrix(records: list[dict], train_method: str = "mixed_train") -> list[dict]:
    rows = []
    for row in performance_by_group(records, key="sampling_method"):
        rows.append(
            {
                "train_method": train_method,
                "test_method": row["group_value"],
                "count": row["count"],
                "positive": row["positive"],
                "negative": row["negative"],
                "accuracy": row["accuracy"],
                "balanced_accuracy": row["balanced_accuracy"],
                "positive_accuracy": row["positive_accuracy"],
                "negative_accuracy": row["negative_accuracy"],
                "mean_loss": row["mean_loss"],
                "mean_uncertainty": row["mean_uncertainty"],
                "mean_true_class_confidence": row["mean_true_class_confidence"],
            }
        )
    return rows


def correlation_report(records: list[dict]) -> list[dict]:
    specs = [
        ("all", "difficulty_proxy"),
        ("all", "euclidean_distance"),
        ("all", "boundary_crossing_count"),
        ("all", "distance_to_boundary_min"),
        ("connected_only", "geodesic_distance"),
        ("connected_only", "geodesic_euclidean_ratio"),
    ]
    summary = []
    for subset_name, feature_key in specs:
        if subset_name == "connected_only":
            subset = [record for record in records if int(record["connected"]) == 1]
        else:
            subset = list(records)
        feature_values = [_to_float(record.get(feature_key)) for record in subset]
        loss_values = [_to_float(record.get("loss")) for record in subset]
        conf_values = [_to_float(record.get("true_class_confidence")) for record in subset]
        paired_loss = [(feature, loss) for feature, loss in zip(feature_values, loss_values) if feature is not None and loss is not None]
        paired_conf = [(feature, conf) for feature, conf in zip(feature_values, conf_values) if feature is not None and conf is not None]
        summary.append(
            {
                "subset": subset_name,
                "feature": feature_key,
                "count_loss": len(paired_loss),
                "loss_correlation": round(_correlation([a for a, _ in paired_loss], [b for _, b in paired_loss]), 6),
                "confidence_correlation": round(_correlation([a for a, _ in paired_conf], [b for _, b in paired_conf]), 6),
            }
        )
    return summary


def quantile_bin_summary(
    records: list[dict],
    feature_key: str,
    subset_name: str = "all",
    n_bins: int = 4,
) -> list[dict]:
    if subset_name == "connected_only":
        subset = [record for record in records if int(record["connected"]) == 1]
    else:
        subset = list(records)
    values = [_to_float(record.get(feature_key)) for record in subset]
    valid = [(record, value) for record, value in zip(subset, values) if value is not None]
    if len(valid) < n_bins:
        return []
    sorted_valid = sorted(valid, key=lambda item: item[1])
    bins: list[list[tuple[dict, float]]] = [[] for _ in range(n_bins)]
    for index, item in enumerate(sorted_valid):
        bin_index = min(n_bins - 1, int(index * n_bins / len(sorted_valid)))
        bins[bin_index].append(item)
    summaries = []
    for bin_index, bin_items in enumerate(bins):
        records_in_bin = [record for record, _ in bin_items]
        feature_values = [value for _, value in bin_items]
        summaries.append(
            {
                "feature": feature_key,
                "subset": subset_name,
                "bin_index": bin_index,
                "count": len(records_in_bin),
                "positive": sum(int(record["connected"]) for record in records_in_bin),
                "negative": len(records_in_bin) - sum(int(record["connected"]) for record in records_in_bin),
                "feature_min": round(min(feature_values), 6),
                "feature_max": round(max(feature_values), 6),
                "mean_loss": round(_safe_mean([float(record["loss"]) for record in records_in_bin]), 6),
                "mean_uncertainty": round(_safe_mean([float(record["uncertainty"]) for record in records_in_bin]), 6),
                "mean_true_class_confidence": round(_safe_mean([float(record["true_class_confidence"]) for record in records_in_bin]), 6),
                "accuracy": round(_safe_mean([float(record["is_correct"]) for record in records_in_bin]), 6),
                "balanced_accuracy": _balanced_accuracy(records_in_bin),
                "positive_accuracy": _bucket_accuracy(records_in_bin, "connected", "1"),
                "negative_accuracy": _bucket_accuracy(records_in_bin, "connected", "0"),
            }
        )
    return summaries


def save_analysis_outputs(
    out_dir: str | Path,
    split_records: dict[str, list[dict]],
    test_predictions: list[dict],
    training_history: list[dict],
    test_summary: dict,
    correlation_summary: list[dict],
    bin_summaries: list[dict],
    sampling_method_performance: list[dict],
    train_test_method_matrix: list[dict],
    split_by_image: dict[str, str],
) -> None:
    out_dir = ensure_dir(out_dir)
    metrics_dir = ensure_dir(out_dir / "metrics")
    report_dir = ensure_dir(out_dir / "report")
    figure_dir = ensure_dir(out_dir / "figures")

    write_csv(metrics_dir / "test_predictions.csv", test_predictions)
    write_csv(metrics_dir / "correlation_summary.csv", correlation_summary)
    if bin_summaries:
        write_csv(metrics_dir / "feature_bin_summary.csv", bin_summaries)
        write_csv(metrics_dir / "feature_bin_summary_by_label.csv", bin_summaries)
    if sampling_method_performance:
        write_csv(metrics_dir / "sampling_method_performance.csv", sampling_method_performance)
    if train_test_method_matrix:
        write_csv(metrics_dir / "train_method_vs_test_method_matrix.csv", train_test_method_matrix)
    write_json(metrics_dir / "training_history.json", training_history)
    write_json(metrics_dir / "test_summary.json", test_summary)
    split_rows = [
        {"image_id": image_id, "split": split}
        for image_id, split in sorted(split_by_image.items())
    ]
    write_csv(metrics_dir / "image_splits.csv", split_rows)
    _write_report(
        report_dir / "region_pair_cnn_report.md",
        split_records=split_records,
        test_summary=test_summary,
        correlation_summary=correlation_summary,
        bin_summaries=bin_summaries,
        sampling_method_performance=sampling_method_performance,
        train_test_method_matrix=train_test_method_matrix,
    )
    _save_scatter_figure(test_predictions, feature_key="difficulty_proxy", value_key="loss", out_path=figure_dir / "loss_vs_difficulty_proxy.png")
    _save_scatter_figure(test_predictions, feature_key="boundary_crossing_count", value_key="uncertainty", out_path=figure_dir / "uncertainty_vs_boundary_crossing.png")
    _save_scatter_figure(
        [record for record in test_predictions if _to_float(record.get("geodesic_euclidean_ratio")) is not None],
        feature_key="geodesic_euclidean_ratio",
        value_key="loss",
        out_path=figure_dir / "loss_vs_geodesic_ratio_connected.png",
    )
    _save_feature_difficulty_heatmap(
        test_predictions,
        feature_key="euclidean_distance",
        difficulty_key="loss",
        out_path=figure_dir / "heatmap_euclidean_distance_vs_loss.png",
        subset_name="all",
    )
    _save_feature_difficulty_heatmap(
        test_predictions,
        feature_key="boundary_crossing_count",
        difficulty_key="loss",
        out_path=figure_dir / "heatmap_boundary_crossing_count_vs_loss.png",
        subset_name="all",
    )
    _save_feature_difficulty_heatmap(
        [record for record in test_predictions if int(record["connected"]) == 1],
        feature_key="geodesic_distance",
        difficulty_key="loss",
        out_path=figure_dir / "heatmap_geodesic_distance_vs_loss_connected.png",
        subset_name="connected_only",
    )
    _save_feature_difficulty_heatmap(
        [record for record in test_predictions if _to_float(record.get("geodesic_euclidean_ratio")) is not None],
        feature_key="geodesic_euclidean_ratio",
        difficulty_key="loss",
        out_path=figure_dir / "heatmap_geodesic_ratio_vs_loss_connected.png",
        subset_name="connected_only",
    )


def _write_report(
    out_path: str | Path,
    split_records: dict[str, list[dict]],
    test_summary: dict,
    correlation_summary: list[dict],
    bin_summaries: list[dict],
    sampling_method_performance: list[dict],
    train_test_method_matrix: list[dict],
) -> None:
    lines = [
        "# Region Pair CNN Experiment Report",
        "",
        "## Split Summary",
        "",
        "| split | n_pairs | positive | negative |",
        "| --- | --- | --- | --- |",
    ]
    for split in ["train", "val", "test"]:
        records = split_records[split]
        positives = sum(int(record["connected"]) for record in records)
        negatives = len(records) - positives
        lines.append(f"| {split} | {len(records)} | {positives} | {negatives} |")

    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            f"- accuracy: `{test_summary['accuracy']}`",
            f"- balanced accuracy: `{test_summary['balanced_accuracy']}`",
            f"- positive accuracy: `{test_summary['positive_accuracy']}`",
            f"- negative accuracy: `{test_summary['negative_accuracy']}`",
            f"- mean loss: `{test_summary['mean_loss']}`",
            f"- mean uncertainty: `{test_summary['mean_uncertainty']}`",
            f"- mean true-class confidence: `{test_summary['mean_true_class_confidence']}`",
            "",
            "## Correlation Summary",
            "",
            "| subset | feature | n | corr(loss) | corr(confidence) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in correlation_summary:
        lines.append(
            f"| {row['subset']} | {row['feature']} | {row['count_loss']} | {row['loss_correlation']} | {row['confidence_correlation']} |"
        )

    if sampling_method_performance:
        lines.extend(
            [
                "",
                "## CNN Performance By Sampling Method",
                "",
                "| method | count | positive | negative | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty | mean_confidence |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sampling_method_performance:
            lines.append(
                f"| {row['group_value']} | {row['count']} | {row['positive']} | {row['negative']} | "
                f"{row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | "
                f"{row['mean_loss']} | {row['mean_uncertainty']} | {row['mean_true_class_confidence']} |"
            )

    if train_test_method_matrix:
        lines.extend(
            [
                "",
                "## Train Method Vs Test Method",
                "",
                "| train_method | test_method | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty | mean_confidence |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in train_test_method_matrix:
            lines.append(
                f"| {row['train_method']} | {row['test_method']} | {row['accuracy']} | {row['balanced_accuracy']} | "
                f"{row['positive_accuracy']} | {row['negative_accuracy']} | {row['mean_loss']} | {row['mean_uncertainty']} | {row['mean_true_class_confidence']} |"
            )

    if bin_summaries:
        lines.extend(
            [
                "",
                "## Feature Bin Summary",
                "",
                "| feature | subset | bin | range | count | positive | negative | accuracy | balanced_accuracy | positive_accuracy | negative_accuracy | mean_loss | mean_uncertainty | mean_confidence |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in bin_summaries:
            range_text = f"{row['feature_min']}..{row['feature_max']}"
            lines.append(
                f"| {row['feature']} | {row['subset']} | {row['bin_index']} | {range_text} | "
                f"{row['count']} | {row['positive']} | {row['negative']} | "
                f"{row['accuracy']} | {row['balanced_accuracy']} | {row['positive_accuracy']} | {row['negative_accuracy']} | "
                f"{row['mean_loss']} | {row['mean_uncertainty']} | {row['mean_true_class_confidence']} |"
            )

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_scatter_figure(records: list[dict], feature_key: str, value_key: str, out_path: str | Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = [
        (_to_float(record.get(feature_key)), _to_float(record.get(value_key)), int(record.get("connected", 0)))
        for record in records
    ]
    points = [(x_value, y_value, label) for x_value, y_value, label in points if x_value is not None and y_value is not None]
    if not points:
        return
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    labels = [point[2] for point in points]
    plt.figure(figsize=(7, 5))
    colors = ["#4c78a8" if label == 1 else "#e45756" for label in labels]
    plt.scatter(x_values, y_values, c=colors, s=14, alpha=0.45)
    plt.xlabel(feature_key)
    plt.ylabel(value_key)
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label="Blue: connected pair", markerfacecolor="#4c78a8", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="Red: disconnected pair", markerfacecolor="#e45756", markersize=7),
    ]
    plt.legend(handles=legend_items, loc="best")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_feature_difficulty_heatmap(
    records: list[dict],
    feature_key: str,
    difficulty_key: str,
    out_path: str | Path,
    subset_name: str,
    n_feature_bins: int = 24,
    n_difficulty_bins: int = 24,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pairs = [
        (_to_float(record.get(feature_key)), _to_float(record.get(difficulty_key)))
        for record in records
    ]
    pairs = [(feature, difficulty) for feature, difficulty in pairs if feature is not None and difficulty is not None]
    if len(pairs) < 4:
        return
    features = np.asarray([feature for feature, _ in pairs], dtype=np.float64)
    difficulties = np.asarray([difficulty for _, difficulty in pairs], dtype=np.float64)
    feature_edges = np.quantile(features, np.linspace(0.0, 1.0, n_feature_bins + 1))
    difficulty_edges = np.quantile(difficulties, np.linspace(0.0, 1.0, n_difficulty_bins + 1))
    feature_edges = _strictly_increasing_edges(feature_edges)
    difficulty_edges = _strictly_increasing_edges(difficulty_edges)
    if len(feature_edges) < 2 or len(difficulty_edges) < 2:
        return
    hist, _, _ = np.histogram2d(features, difficulties, bins=[feature_edges, difficulty_edges])
    plt.figure(figsize=(7, 5))
    plt.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[feature_edges[0], feature_edges[-1], difficulty_edges[0], difficulty_edges[-1]],
    )
    plt.xlabel(feature_key)
    plt.ylabel(f"{difficulty_key} (BCE difficulty)")
    plt.title(f"{feature_key} vs {difficulty_key} [{subset_name}]")
    plt.colorbar(label="Pair count")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _bucket_accuracy(records: list[dict], key: str, expected_value: str) -> float:
    subset = [record for record in records if str(record.get(key)) == expected_value]
    if not subset:
        return 0.0
    return round(_safe_mean([float(record["is_correct"]) for record in subset]), 6)


def _balanced_accuracy(records: list[dict]) -> float:
    positive_records = [record for record in records if int(record.get("connected", 0)) == 1]
    negative_records = [record for record in records if int(record.get("connected", 0)) == 0]
    if positive_records and negative_records:
        positive_accuracy = _safe_mean([float(record["is_correct"]) for record in positive_records])
        negative_accuracy = _safe_mean([float(record["is_correct"]) for record in negative_records])
        return round((positive_accuracy + negative_accuracy) / 2.0, 6)
    if positive_records or negative_records:
        return round(_safe_mean([float(record["is_correct"]) for record in records]), 6)
    return 0.0


def _performance_metrics(records: list[dict]) -> dict:
    accuracy = _safe_mean([float(record["is_correct"]) for record in records])
    loss_mean = _safe_mean([float(record["loss"]) for record in records])
    confidence_mean = _safe_mean([float(record["true_class_confidence"]) for record in records])
    uncertainty_mean = _safe_mean([float(record["uncertainty"]) for record in records])
    positive = sum(int(record.get("connected", 0)) for record in records)
    negative = len(records) - positive
    return {
        "count": len(records),
        "positive": positive,
        "negative": negative,
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": _balanced_accuracy(records),
        "positive_accuracy": _bucket_accuracy(records, "connected", "1"),
        "negative_accuracy": _bucket_accuracy(records, "connected", "0"),
        "mean_loss": round(loss_mean, 6),
        "mean_uncertainty": round(uncertainty_mean, 6),
        "mean_true_class_confidence": round(confidence_mean, 6),
    }


def _accuracy_by_key(records: list[dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        bucket = str(record.get(key, ""))
        buckets.setdefault(bucket, []).append(float(record["is_correct"]))
    return {bucket: round(_safe_mean(values), 6) for bucket, values in buckets.items() if bucket}


def _draw_disk(array: np.ndarray, row: int, col: int, radius: int, value: float) -> None:
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            if delta_row * delta_row + delta_col * delta_col > radius * radius:
                continue
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < array.shape[0] and 0 <= next_col < array.shape[1]:
                array[next_row, next_col] = value


def _compose_input_channels(
    mask: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    input_mode: str,
    line_width: int = 1,
    start: tuple[int, int] | None = None,
    end: tuple[int, int] | None = None,
) -> np.ndarray:
    normalized = normalize_input_mode(input_mode)
    if normalized == "mask_point_i_point_j":
        image = np.stack([mask, point_a, point_b], axis=0)
    elif normalized == "mask_point_i_point_j_line":
        if start is None or end is None:
            raise ValueError("start/end points are required for mask_point_i_point_j_line input mode")
        line = np.zeros_like(mask, dtype=np.float32)
        _draw_line(line, start[0], start[1], end[0], end[1], line_width=line_width, value=1.0)
        image = np.stack([mask, point_a, point_b, line], axis=0)
    elif normalized == "point_i_point_j_only":
        image = np.stack([point_a, point_b], axis=0)
    elif normalized == "mask_only":
        image = np.expand_dims(mask, axis=0)
    elif normalized == "mask_merged_points":
        merged_points = np.maximum(point_a, point_b)
        image = np.stack([mask, merged_points], axis=0)
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")
    return image.astype(np.float32)


def _draw_line(array: np.ndarray, row_i: int, col_i: int, row_j: int, col_j: int, line_width: int, value: float) -> None:
    radius = max(0, int(line_width) // 2)
    for row, col in _bresenham_line(row_i, col_i, row_j, col_j):
        _draw_disk(array, row, col, radius=radius, value=value)


def _bresenham_line(row_i: int, col_i: int, row_j: int, col_j: int) -> list[tuple[int, int]]:
    delta_row = abs(row_j - row_i)
    delta_col = abs(col_j - col_i)
    step_row = 1 if row_i < row_j else -1
    step_col = 1 if col_i < col_j else -1
    err = delta_col - delta_row
    line = []

    while True:
        line.append((row_i, col_i))
        if row_i == row_j and col_i == col_j:
            break
        twice_err = 2 * err
        if twice_err > -delta_row:
            err -= delta_row
            col_i += step_col
        if twice_err < delta_col:
            err += delta_col
            row_i += step_row
    return line


def _scale_point(row: int, col: int, original_height: int, original_width: int, target_size: int) -> tuple[int, int]:
    scaled_row = int(round(row * (target_size - 1) / max(1, original_height - 1)))
    scaled_col = int(round(col * (target_size - 1) / max(1, original_width - 1)))
    return scaled_row, scaled_col


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_float(value) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    return float(value)


def _correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(values_a), np.asarray(values_b))[0, 1])


def _strictly_increasing_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    unique_values = [float(values[0])]
    for value in values[1:]:
        value = float(value)
        if value <= unique_values[-1]:
            value = unique_values[-1] + 1e-6
        unique_values.append(value)
    return np.asarray(unique_values, dtype=np.float64)
