#!/usr/bin/env python3
"""Run the support-ticket classification prompt over a JSONL dataset."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
import logging
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

try:
    from .config import CONFIGS
    from .draft import generate_draft_response
    from .llm import LLMClient, create_llm_client, load_prompt, parse_json_object
except ImportError:
    from config import CONFIGS
    from draft import generate_draft_response
    from llm import LLMClient, create_llm_client, load_prompt, parse_json_object


CLASSIFICATION_PROMPT_PATH = Path("prompts/classification.md")
CONFIG_PATH = Path(__file__).resolve().with_name("config.py")
DEFAULT_RESULTS_ROOT = Path("results")
CATEGORY_OUTPUT_FILENAME = "category_output.jsonl"
EVAL_OUTPUT_FILENAME = "eval_output.txt"
CONFIG_SNAPSHOT_FILENAME = "config.py"
TICKET_INPUT_FIELDS = ("ticket_id", "subject", "body", "metadata")
REQUIRED_FIELDS = {
    "ticket_id",
    "category",
    "urgency",
    "should_draft",
    "no_draft_reason",
    "draft_response",
    "confidence",
}


def default_config_name() -> str:
    return next(iter(CONFIGS))


def get_pipeline_config(config_name: str) -> Mapping[str, str]:
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS)
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available}")

    config = CONFIGS[config_name]
    missing_fields = [
        field
        for field in ("classification_model", "draft_model")
        if field not in config
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Config '{config_name}' is missing required fields: {missing}")

    return config


RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "ticket_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ticket_id": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "account_access",
                        "kyc_verification",
                        "deposits_withdrawals",
                        "trading_mechanics",
                        "market_questions",
                        "bug_report",
                        "tax_documents",
                        "account_compromise",
                        "problem_gambling",
                        "legal_regulatory",
                        "other",
                    ],
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "escalate_immediately"],
                },
                "should_draft": {"type": "boolean"},
                "no_draft_reason": {"type": ["string", "null"]},
                "draft_response": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": sorted(REQUIRED_FIELDS),
        },
    }
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")

            yield record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_run_directory(results_root: Path, config_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_config_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in config_name)
    run_name = f"{timestamp}_{safe_config_name}"
    run_dir = results_root / run_name
    suffix = 1

    while run_dir.exists():
        run_dir = results_root / f"{run_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def copy_config_snapshot(run_dir: Path, config_path: Path = CONFIG_PATH) -> Path:
    destination = run_dir / CONFIG_SNAPSHOT_FILENAME
    shutil.copy2(config_path, destination)
    return destination


def validate_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    missing_fields = REQUIRED_FIELDS - set(prediction)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Model response missing required fields: {missing}")

    return prediction


def prepare_ticket_for_model(ticket: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in TICKET_INPUT_FIELDS if field not in ticket]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Ticket missing required input fields: {missing}")

    return {field: ticket[field] for field in TICKET_INPUT_FIELDS}


def classify_ticket(
    llm_client: LLMClient,
    instructions: str,
    ticket: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    model_input = prepare_ticket_for_model(ticket)
    response_text = llm_client.generate(instructions, model_input, model, RESPONSE_FORMAT)
    prediction = parse_json_object(response_text)
    return validate_prediction(prediction)


def fallback_prediction(ticket: dict[str, Any], reason: str) -> dict[str, Any]:
    ticket_id = ticket.get("ticket_id")
    if not isinstance(ticket_id, str):
        ticket_id = ""

    return {
        "ticket_id": ticket_id,
        "category": "other",
        "urgency": "medium",
        "should_draft": False,
        "no_draft_reason": f"classification failed: {reason}",
        "draft_response": None,
        "confidence": 0.0,
    }


def process_ticket(
    llm_client: LLMClient,
    classification_instructions: str,
    ticket: dict[str, Any],
    config: Mapping[str, str],
) -> tuple[dict[str, Any], bool]:
    classification_model = config["classification_model"]
    draft_model = config["draft_model"]
    prediction = classify_ticket(llm_client, classification_instructions, ticket, classification_model)
    draft_failed = False

    if prediction.get("should_draft") is True:
        try:
            model_ticket = prepare_ticket_for_model(ticket)
            prediction["draft_response"] = generate_draft_response(
                llm_client,
                model_ticket,
                prediction,
                draft_model,
            )
        except Exception:
            draft_failed = True
            prediction["draft_response"] = None
            logging.exception("Failed to draft response for ticket %s", prediction.get("ticket_id"))

    return prediction, draft_failed


def run_pipeline(
    input_path: Path,
    output_path: Path,
    prompt_path: Path,
    config: Mapping[str, str],
) -> tuple[int, int, int, int]:
    classification_instructions = load_prompt(prompt_path)
    llm_client = create_llm_client()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    total = 0
    successful = 0
    failed = 0
    draft_failed = 0

    for ticket in read_jsonl(input_path):
        total += 1
        ticket_id = ticket.get("ticket_id", f"line_{total}")
        print(f"Processing ticket {total}: {ticket_id}", flush=True)

        try:
            prediction, ticket_draft_failed = process_ticket(llm_client, classification_instructions, ticket, config)
            successful += 1
            draft_failed += int(ticket_draft_failed)
        except Exception as exc:
            failed += 1
            logging.exception("Failed to classify ticket %s", ticket_id)
            prediction = fallback_prediction(ticket, str(exc))

        append_jsonl(output_path, prediction)

    return total, successful, failed, draft_failed


def write_eval_report(ground_truth_path: Path, predictions_path: Path, eval_output_path: Path) -> None:
    try:
        from . import evaluate_predictions
    except ImportError:
        import evaluate_predictions

    ground_truth_records = evaluate_predictions.read_jsonl(ground_truth_path)
    prediction_records = evaluate_predictions.read_jsonl(predictions_path)

    with eval_output_path.open("w", encoding="utf-8") as outfile:
        with redirect_stdout(outfile):
            evaluate_predictions.evaluate(ground_truth_records, prediction_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and draft support-ticket responses using the configured LLM providers."
    )
    parser.add_argument("input_path", type=Path, help="Path to the input .jsonl dataset")
    parser.add_argument(
        "--config",
        default=default_config_name(),
        help=f"Named config to run. Available: {', '.join(CONFIGS)}. Defaults to the first config in src/config.py.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help="Optional labeled JSONL file to evaluate against. Defaults to input_path.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        config = get_pipeline_config(args.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = create_run_directory(args.results_root, args.config)
    category_output_path = run_dir / CATEGORY_OUTPUT_FILENAME
    eval_output_path = run_dir / EVAL_OUTPUT_FILENAME
    ground_truth_path = args.ground_truth_path or args.input_path

    print(f"Run directory: {run_dir}")
    print(f"Config: {args.config}")
    print(f"Classification model: {config['classification_model']}")
    print(f"Draft model: {config['draft_model']}")
    config_snapshot_path = copy_config_snapshot(run_dir)
    print(f"Config snapshot: {config_snapshot_path}")
    print(f"Prediction output: {category_output_path}")
    total, successful, failed, draft_failed = run_pipeline(
        args.input_path,
        category_output_path,
        CLASSIFICATION_PROMPT_PATH,
        config,
    )

    print("\nSummary")
    print(f"Total tickets processed: {total}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    print(f"Failed drafts: {draft_failed}")

    write_eval_report(ground_truth_path, category_output_path, eval_output_path)
    print(f"Eval output: {eval_output_path}")


if __name__ == "__main__":
    main()
