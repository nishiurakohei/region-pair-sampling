from __future__ import annotations

import argparse

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.difficulty import attach_difficulty_proxy
from connectivity_experiment.generation import generate_sample_pair, generate_scene_query_samples, save_sample
from connectivity_experiment.io_utils import write_csv, write_jsonl
from connectivity_experiment.visualize import build_visualization_sets, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate connectivity synthetic dataset.")
    parser.add_argument("--config", default="configs/curved_v1.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Override per-split sample count for quick checks.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)
    dataset_mode = str(config["dataset"].get("dataset_mode", "pair"))
    raw_dir = repo_root / config["paths"]["raw_dir"]
    split_targets = {
        "train": int(config["dataset"]["n_train"]),
        "val": int(config["dataset"]["n_val"]),
        "test": int(config["dataset"]["n_test"]),
    }
    positive_ratio = float(config["dataset"].get("positive_ratio", 0.5))
    if args.limit is not None:
        split_targets = {split: min(count, args.limit) for split, count in split_targets.items()}

    if args.overwrite:
        for split in split_targets:
            split_dir = raw_dir / split
            if split_dir.exists():
                for child in split_dir.rglob("*"):
                    if child.is_file():
                        child.unlink()
    processed_dir = repo_root / config["paths"]["processed_dir"]
    sample_dir = repo_root / config["paths"]["sample_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(config["experiment"]["seed"])
    all_records = []

    for split, total_samples in split_targets.items():
        records = []
        positive_remaining = round(total_samples * positive_ratio)
        negative_remaining = total_samples - positive_remaining
        generation_index = 0
        while positive_remaining > 0 or negative_remaining > 0:
            if dataset_mode == "scene_query":
                samples = generate_scene_query_samples(
                    config=config,
                    split=split,
                    scene_index=generation_index,
                    base_seed=base_seed,
                    target_positive=positive_remaining,
                    target_negative=negative_remaining,
                )
                for sample in samples:
                    if sample["connected"] == 1 and positive_remaining <= 0:
                        continue
                    if sample["connected"] == 0 and negative_remaining <= 0:
                        continue
                    records.append(save_sample(sample, raw_dir))
                    if sample["connected"] == 1:
                        positive_remaining -= 1
                    else:
                        negative_remaining -= 1
            else:
                positive, negative = generate_sample_pair(config, split, generation_index, base_seed)
                if positive_remaining > 0:
                    records.append(save_sample(positive, raw_dir))
                    positive_remaining -= 1
                if negative_remaining > 0:
                    records.append(save_sample(negative, raw_dir))
                    negative_remaining -= 1
            generation_index += 1

        attach_difficulty_proxy(records)
        split_dir = raw_dir / split
        write_csv(split_dir / "metadata.csv", records)
        write_jsonl(split_dir / "metadata.jsonl", records)
        all_records.extend(records)
        print(f"[generate_dataset] split={split} samples={len(records)} -> {split_dir}")

    write_csv(processed_dir / "metadata.csv", all_records)
    write_jsonl(processed_dir / "metadata.jsonl", all_records)
    test_records = [record for record in all_records if record["split"] == "test"]
    visuals = build_visualization_sets(test_records)
    for name, subset in visuals.items():
        make_contact_sheet(
            subset,
            repo_root=repo_root,
            out_path=sample_dir / f"{name}.png",
            title=name.replace("_", " "),
        )
    print(f"[generate_dataset] aggregate metadata -> {processed_dir / 'metadata.csv'}")
    print(f"[generate_dataset] sample grids -> {sample_dir}")


if __name__ == "__main__":
    main()
