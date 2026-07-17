from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date, datetime
from pathlib import Path


IGNORE_NAMES = {".DS_Store", "__pycache__"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}

CORE_RELATIVE_PATHS = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("src/README.md"),
    Path("src/generate_region_pairs.py"),
    Path("src/run_pair_corruption_evaluation.py"),
    Path("src/connectivity_experiment"),
    Path("tests/test_config_loader.py"),
    Path("tests/test_pair_corruption_evaluation.py"),
    Path("tests/test_region_pair_generation.py"),
    Path("docs/shared_region_pair_pipeline"),
    Path("configs/pair_generation_default.yaml"),
    Path("configs/pair_generation_stare_demo.yaml"),
    Path("configs/pair_corruption_evaluation_default.yaml"),
    Path("configs/pair_corruption_evaluation_stare_demo.yaml"),
    Path("scripts/create_collaborator_bundle.py"),
]

OPTIONAL_DEMO_PATHS = [
    Path("outputs/demo_pair_generation_stare_realdata_pairwise_check2"),
    Path("outputs/demo_pair_corruption_stare_realdata"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a collaborator-ready share bundle.")
    parser.add_argument(
        "--output-dir",
        default=f"outputs/share_bundles/collaborator_handoff_{date.today().isoformat()}",
        help="Bundle output directory, relative to the repository root.",
    )
    parser.add_argument(
        "--skip-demo-outputs",
        action="store_true",
        help="Do not include local demo output directories in the bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Re-run with --overwrite or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    bundle_paths = list(CORE_RELATIVE_PATHS)
    if not args.skip_demo_outputs:
        bundle_paths.extend(OPTIONAL_DEMO_PATHS)

    for relative_path in bundle_paths:
        src = repo_root / relative_path
        dst = output_dir / relative_path
        status = copy_entry(src=src, dst=dst)
        manifest_rows.append(
            {
                "relative_path": relative_path.as_posix(),
                "kind": "directory" if src.is_dir() else "file",
                "status": status,
            }
        )

    write_start_here(output_dir=output_dir, skip_demo_outputs=args.skip_demo_outputs)
    write_manifest(output_dir=output_dir, rows=manifest_rows)
    write_bundle_info(output_dir=output_dir, rows=manifest_rows, skip_demo_outputs=args.skip_demo_outputs)
    write_tree(output_dir=output_dir)

    copied_count = sum(row["status"] == "copied" for row in manifest_rows)
    missing_count = sum(row["status"] == "missing" for row in manifest_rows)
    print(f"Bundle created: {output_dir}")
    print(f"Copied entries: {copied_count}")
    print(f"Missing optional entries: {missing_count}")


def copy_entry(src: Path, dst: Path) -> str:
    if not src.exists():
        return "missing"
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_copytree_ignore)
        return "copied"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def _copytree_ignore(directory: str, names: list[str]) -> list[str]:
    ignored: list[str] = []
    for name in names:
        path = Path(directory) / name
        if name in IGNORE_NAMES:
            ignored.append(name)
            continue
        if path.suffix in IGNORE_SUFFIXES:
            ignored.append(name)
    return ignored


def write_start_here(output_dir: Path, skip_demo_outputs: bool) -> None:
    demo_line = (
        "- Demo outputs were not included in this bundle because `--skip-demo-outputs` was used.\n"
        if skip_demo_outputs
        else "- Demo outputs were included when they were available locally.\n"
    )
    text = (
        "# README_START_HERE\n\n"
        "This folder is a collaborator-ready snapshot of the region-pair pipeline.\n\n"
        "Recommended reading order:\n\n"
        "1. `docs/shared_region_pair_pipeline/HANDOFF_ja.md` or `HANDOFF_en.md`\n"
        "2. `docs/shared_region_pair_pipeline/QUICKSTART_ja.md` or `QUICKSTART_en.md`\n"
        "3. `docs/shared_region_pair_pipeline/DEMO_RUN_SUMMARY_ja.md` or `DEMO_RUN_SUMMARY_en.md`\n"
        "4. `docs/shared_region_pair_pipeline/README_ja.md` or `README_en.md`\n\n"
        "Main code entry points:\n\n"
        "- `src/generate_region_pairs.py`\n"
        "- `src/run_pair_corruption_evaluation.py`\n"
        "- `src/connectivity_experiment/region_pair_generation.py`\n"
        "- `src/connectivity_experiment/pair_corruption_evaluation.py`\n\n"
        "Main configs:\n\n"
        "- `configs/pair_generation_stare_demo.yaml`\n"
        "- `configs/pair_corruption_evaluation_stare_demo.yaml`\n\n"
        "Notes:\n\n"
        "- Real mask data is not tracked in Git by default.\n"
        "- `outputs/` is normally generated locally.\n"
        f"{demo_line}"
    )
    (output_dir / "README_START_HERE.md").write_text(text, encoding="utf-8")


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    path = output_dir / "FILE_MANIFEST.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "kind", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bundle_info(output_dir: Path, rows: list[dict[str, str]], skip_demo_outputs: bool) -> None:
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "skip_demo_outputs": bool(skip_demo_outputs),
        "entries": rows,
    }
    (output_dir / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_tree(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if path.name in IGNORE_NAMES:
            continue
        if path.suffix in IGNORE_SUFFIXES:
            continue
        relative = path.relative_to(output_dir)
        depth = len(relative.parts) - 1
        prefix = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{prefix}{relative.name}{suffix}")
    (output_dir / "TREE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
