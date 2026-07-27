#!/usr/bin/env python3
"""Run unlabeled eval predictions and agreement checks across two configs."""

from __future__ import annotations

import argparse
import logging
from contextlib import redirect_stdout
from pathlib import Path

from src.classify_tickets import (
    CLASSIFICATION_PROMPT_PATH,
    DEFAULT_RESULTS_ROOT,
    copy_config_snapshot,
    create_run_directory,
    get_pipeline_config,
    run_pipeline,
)
from src.config import CONFIGS
from src.evaluate_predictions import evaluate_agreement, read_jsonl


DEFAULT_INPUT_PATH = Path("novig_files/tickets_eval.jsonl")
PREDICTIONS_FILENAME = "predictions.jsonl"
AGREEMENT_REPORT_FILENAME = "agreement_report.txt"


def default_config_pair() -> tuple[str, str]:
    config_names = list(CONFIGS)
    if len(config_names) < 2:
        available = ", ".join(config_names) or "none"
        raise ValueError(f"Agreement eval requires at least two configs. Available configs: {available}")
    return config_names[0], config_names[1]


def comparison_predictions_filename(config_name: str) -> str:
    safe_config_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in config_name)
    return f"comparison_predictions_{safe_config_name}.jsonl"


def write_agreement_report(
    first_predictions_path: Path,
    second_predictions_path: Path,
    report_path: Path,
    first_config_name: str,
    second_config_name: str,
) -> None:
    first_predictions = read_jsonl(first_predictions_path)
    second_predictions = read_jsonl(second_predictions_path)

    with report_path.open("w", encoding="utf-8") as outfile:
        with redirect_stdout(outfile):
            evaluate_agreement(
                first_predictions,
                second_predictions,
                first_config_name=first_config_name,
                second_config_name=second_config_name,
            )


def run_eval(
    input_path: Path,
    results_root: Path,
    first_config_name: str,
    second_config_name: str,
) -> Path:
    if first_config_name == second_config_name:
        raise ValueError("Agreement eval requires two different configs.")

    first_config = get_pipeline_config(first_config_name)
    second_config = get_pipeline_config(second_config_name)

    run_dir = create_run_directory(results_root, f"{first_config_name}_vs_{second_config_name}")
    first_predictions_path = run_dir / PREDICTIONS_FILENAME
    second_predictions_path = run_dir / comparison_predictions_filename(second_config_name)
    agreement_report_path = run_dir / AGREEMENT_REPORT_FILENAME
    config_snapshot_path = copy_config_snapshot(run_dir)

    print(f"Run directory: {run_dir}")
    print(f"Config snapshot: {config_snapshot_path}")

    print(f"\nRunning first config: {first_config_name}")
    print(f"Official predictions output: {first_predictions_path}")
    total, successful, failed, draft_failed = run_pipeline(
        input_path,
        first_predictions_path,
        CLASSIFICATION_PROMPT_PATH,
        first_config,
    )
    print(f"First config summary: total={total}, successful={successful}, failed={failed}, draft_failed={draft_failed}")

    print(f"\nRunning second config: {second_config_name}")
    print(f"Comparison predictions output: {second_predictions_path}")
    total, successful, failed, draft_failed = run_pipeline(
        input_path,
        second_predictions_path,
        CLASSIFICATION_PROMPT_PATH,
        second_config,
    )
    print(f"Second config summary: total={total}, successful={successful}, failed={failed}, draft_failed={draft_failed}")

    write_agreement_report(
        first_predictions_path,
        second_predictions_path,
        agreement_report_path,
        first_config_name,
        second_config_name,
    )
    print(f"\nAgreement report: {agreement_report_path}")

    return run_dir


def parse_args() -> argparse.Namespace:
    first_default, second_default = default_config_pair()
    parser = argparse.ArgumentParser(
        description="Run unlabeled eval predictions and compare agreement between two configs."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the unlabeled eval JSONL file.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory where timestamped eval run folders are created.",
    )
    parser.add_argument(
        "--first-config",
        default=first_default,
        help=f"Primary config for official predictions. Available: {', '.join(CONFIGS)}.",
    )
    parser.add_argument(
        "--second-config",
        default=second_default,
        help=f"Comparison config for agreement checks. Available: {', '.join(CONFIGS)}.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        run_eval(args.input_path, args.results_root, args.first_config, args.second_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
