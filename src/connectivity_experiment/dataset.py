from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from connectivity_experiment.io_utils import read_csv


def load_split_records(repo_root: str | Path, split: str) -> list[dict]:
    repo_root = Path(repo_root)
    metadata_path = repo_root / "data" / "raw" / split / "metadata.csv"
    return read_csv(metadata_path)


def load_split_arrays(
    repo_root: str | Path,
    split: str,
    input_size: int,
    drop_label_disagreement: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    repo_root = Path(repo_root)
    records = load_split_records(repo_root, split)
    if drop_label_disagreement:
        records = [record for record in records if int(record.get("label_disagreement", 0)) == 0]
    images = []
    labels = []
    for record in records:
        image_path = repo_root / record["image_path"]
        image = Image.open(image_path).convert("RGB")
        if image.size != (input_size, input_size):
            image = image.resize((input_size, input_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(np.transpose(array, (2, 0, 1)))
        labels.append(float(record["connected"]))
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.float32), records
