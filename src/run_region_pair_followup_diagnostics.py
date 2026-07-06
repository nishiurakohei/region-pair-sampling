from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from connectivity_experiment.config import load_config, repo_root_from_script
from connectivity_experiment.io_utils import ensure_dir, read_csv, write_csv, write_json
from connectivity_experiment.region_pair_cnn import load_pair_records
from connectivity_experiment.region_pair_diagnostics import (
    analyze_random_high_vs_low_loss,
    filter_easy_only_records,
    run_input_ablation,
    run_pair_cnn_experiment,
    save_example_visualizations,
    write_followup_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run follow-up diagnostics for region pair experiments.")
    parser.add_argument("--config", default="configs/region_pair_followup_diagnostics_stare_instruction_subset_v2.yaml")
    parser.add_argument("--regenerate-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = load_config(repo_root / args.config)

    pair_generation_config = str(config["data"]["pair_generation_config"])
    if args.regenerate_pairs or bool(config["data"].get("regenerate_pairs", False)):
        command = [sys.executable, str(repo_root / "src" / "run_region_pair_sampling.py"), "--config", pair_generation_config, "--overwrite"]
        subprocess.run(command, check=True, cwd=repo_root)

    pair_metrics_csv = repo_root / config["data"]["pair_metrics_csv"]
    records = load_pair_records(pair_metrics_csv)
    if not records:
        raise RuntimeError(f"No pair records found in {pair_metrics_csv}")

    seed = int(config["experiment"]["seed"])
    split_config = dict(config["split"])
    model_config = dict(config["model"])
    baseline_output_dir = repo_root / config["paths"]["baseline_output_dir"]
    diagnostics_output_dir = ensure_dir(repo_root / config["paths"]["output_dir"])
    metrics_dir = ensure_dir(diagnostics_output_dir / "metrics")
    report_dir = ensure_dir(diagnostics_output_dir / "report")

    baseline_result = run_pair_cnn_experiment(
        records=records,
        output_dir=baseline_output_dir,
        split_config=split_config,
        model_config=model_config,
        seed=seed,
        split_by_image=None,
        save_outputs=True,
    )
    example_paths = save_example_visualizations(
        predictions_csv=baseline_output_dir / "metrics" / "test_predictions.csv",
        out_dir=baseline_output_dir / "figures" / "example_pairs",
        per_group=int(config["diagnostics"].get("example_visualization_per_group", 20)),
    )

    easy_only_records = filter_easy_only_records(records)
    if not easy_only_records:
        raise RuntimeError("Easy-only sanity dataset ended up empty.")
    write_csv(metrics_dir / "easy_only_records.csv", easy_only_records)
    easy_only_result = run_pair_cnn_experiment(
        records=easy_only_records,
        output_dir=diagnostics_output_dir / "easy_only_sanity",
        split_config=split_config,
        model_config=model_config,
        seed=seed,
        split_by_image=None,
        save_outputs=True,
    )
    write_json(metrics_dir / "easy_only_test_summary.json", easy_only_result.test_summary)

    input_modes = [str(value) for value in config["diagnostics"]["ablation_input_modes"]]
    _, ablation_rows = run_input_ablation(
        records=records,
        output_dir=diagnostics_output_dir / "input_ablation",
        split_config=split_config,
        model_config=model_config,
        seed=seed,
        input_modes=input_modes,
        split_by_image=baseline_result.split_by_image,
    )

    random_analysis_rows = analyze_random_high_vs_low_loss(
        predictions=baseline_result.test_predictions,
        out_dir=diagnostics_output_dir / "analysis",
        fraction=float(config["diagnostics"].get("random_loss_fraction", 0.2)),
    )

    pair_generation_output_dir = repo_root / config["data"]["pair_generation_output_dir"]
    pair_summary_rows = read_csv(pair_generation_output_dir / "metrics" / "pair_summary.csv")
    coverage_rows = read_csv(pair_generation_output_dir / "metrics" / "feature_coverage_summary.csv")
    difficulty_rows = read_csv(pair_generation_output_dir / "metrics" / "difficulty_bucket_summary.csv")

    write_followup_report(
        out_path=report_dir / "region_pair_followup_diagnostics.md",
        baseline_result=baseline_result,
        easy_only_result=easy_only_result,
        easy_only_count=len(easy_only_records),
        ablation_rows=ablation_rows,
        random_analysis_rows=random_analysis_rows,
        pair_summary_rows=pair_summary_rows,
        coverage_rows=coverage_rows,
        difficulty_rows=difficulty_rows,
        example_paths=example_paths,
    )

    write_json(
        metrics_dir / "diagnostics_manifest.json",
        {
            "baseline_output_dir": str(baseline_output_dir),
            "diagnostics_output_dir": str(diagnostics_output_dir),
            "example_visualizations": [str(path) for path in example_paths],
            "easy_only_count": len(easy_only_records),
        },
    )
    print(
        "[region_pair_followup_diagnostics] "
        f"baseline_acc={baseline_result.test_summary['accuracy']:.4f} "
        f"easy_acc={easy_only_result.test_summary['accuracy']:.4f} "
        f"-> {diagnostics_output_dir}"
    )


if __name__ == "__main__":
    main()
