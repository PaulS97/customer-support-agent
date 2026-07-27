#!/usr/bin/env python3
"""Run the support-ticket classification prompt over a JSONL dataset."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from draft import generate_draft_response
from llm import call_responses_api, create_openai_client, load_prompt, parse_json_object


CLASSIFICATION_PROMPT_PATH = Path("prompts/classification.md")
DEFAULT_RESULTS_ROOT = Path("results")
CATEGORY_OUTPUT_FILENAME = "category_output.jsonl"
EVAL_OUTPUT_FILENAME = "eval_output.txt"
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


def create_run_directory(results_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / timestamp
    suffix = 1

    while run_dir.exists():
        run_dir = results_root / f"{timestamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


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


def classify_ticket(client: Any, instructions: str, ticket: dict[str, Any]) -> dict[str, Any]:
    model_input = prepare_ticket_for_model(ticket)
    response_text = call_responses_api(client, instructions, model_input, RESPONSE_FORMAT)
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


def process_ticket(client: Any, classification_instructions: str, ticket: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    prediction = classify_ticket(client, classification_instructions, ticket)
    draft_failed = False

    if prediction.get("should_draft") is True:
        try:
            model_ticket = prepare_ticket_for_model(ticket)
            prediction["draft_response"] = generate_draft_response(client, model_ticket, prediction)
        except Exception:
            draft_failed = True
            prediction["draft_response"] = None
            logging.exception("Failed to draft response for ticket %s", prediction.get("ticket_id"))

    return prediction, draft_failed


def run_pipeline(input_path: Path, output_path: Path, prompt_path: Path) -> tuple[int, int, int, int]:
    classification_instructions = load_prompt(prompt_path)
    client = create_openai_client()

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
            prediction, ticket_draft_failed = process_ticket(client, classification_instructions, ticket)
            successful += 1
            draft_failed += int(ticket_draft_failed)
        except Exception as exc:
            failed += 1
            logging.exception("Failed to classify ticket %s", ticket_id)
            prediction = fallback_prediction(ticket, str(exc))

        append_jsonl(output_path, prediction)

    return total, successful, failed, draft_failed


def write_eval_report(ground_truth_path: Path, predictions_path: Path, eval_output_path: Path) -> None:
    import evaluate_predictions

    ground_truth_records = evaluate_predictions.read_jsonl(ground_truth_path)
    prediction_records = evaluate_predictions.read_jsonl(predictions_path)

    with eval_output_path.open("w", encoding="utf-8") as outfile:
        with redirect_stdout(outfile):
            evaluate_predictions.evaluate(ground_truth_records, prediction_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify support tickets from a JSONL file using the OpenAI Responses API."
    )
    parser.add_argument("input_path", type=Path, help="Path to the input .jsonl dataset")
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
    run_dir = create_run_directory(args.results_root)
    category_output_path = run_dir / CATEGORY_OUTPUT_FILENAME
    eval_output_path = run_dir / EVAL_OUTPUT_FILENAME
    ground_truth_path = args.ground_truth_path or args.input_path

    print(f"Run directory: {run_dir}")
    print(f"Prediction output: {category_output_path}")
    total, successful, failed, draft_failed = run_pipeline(
        args.input_path,
        category_output_path,
        CLASSIFICATION_PROMPT_PATH,
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
