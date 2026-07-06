from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Cannot write empty CSV.")
    fieldnames = list(records[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

