#!/usr/bin/env python3
"""Evaluate ticket triage predictions against labeled JSONL ground truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


URGENCY_CLASSES = ("low", "medium", "high", "escalate_immediately")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON in {path} line {line_number}: {exc}")
                continue

            if not isinstance(record, dict):
                print(f"Skipping non-object JSON in {path} line {line_number}")
                continue

            records.append(record)

    return records


def index_by_ticket_id(records: list[dict[str, Any]], source_name: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()

    for index, record in enumerate(records, start=1):
        ticket_id = record.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id:
            print(f"{source_name}: record {index} is missing a valid ticket_id")
            continue

        if ticket_id in indexed:
            duplicates.add(ticket_id)
            continue

        indexed[ticket_id] = record

    return indexed, duplicates


def get_label(ground_truth: dict[str, Any]) -> dict[str, Any] | None:
    label = ground_truth.get("label")
    if not isinstance(label, dict):
        return None
    return label


def format_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2%}"


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_average(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_list_report(title: str, values: set[str]) -> None:
    if not values:
        print(f"{title}: none")
        return

    print(f"{title}: {len(values)}")
    for value in sorted(values):
        print(f"  - {value}")


def print_confusion_matrix(matrix: dict[str, Counter[str]]) -> None:
    row_label = "actual \\ predicted"
    column_widths = {
        urgency: max(len(urgency), 5)
        for urgency in URGENCY_CLASSES
    }
    label_width = max(len(row_label), *(len(urgency) for urgency in URGENCY_CLASSES))

    header = row_label.ljust(label_width)
    for urgency in URGENCY_CLASSES:
        header += "  " + urgency.rjust(column_widths[urgency])
    print(header)
    print("-" * len(header))

    for actual in URGENCY_CLASSES:
        row = actual.ljust(label_width)
        for predicted in URGENCY_CLASSES:
            row += "  " + str(matrix[actual][predicted]).rjust(column_widths[predicted])
        print(row)


def print_verbose_mismatch(
    ticket_id: str,
    truth: dict[str, Any],
    mismatches: list[tuple[str, Any, Any]],
) -> None:
    print(f"\nTicket: {ticket_id}")
    print(f"Subject: {truth.get('subject', '')}")
    print(f"Body: {truth.get('body', '')}")
    for field, truth_value, predicted_value in mismatches:
        print(f"True {field}: {truth_value}")
        print(f"Predicted {field}: {predicted_value}")


def evaluate(
    ground_truth_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    verbose: bool = False,
) -> None:
    ground_truth_by_id, duplicate_truth_ids = index_by_ticket_id(ground_truth_records, "ground truth")
    predictions_by_id, duplicate_prediction_ids = index_by_ticket_id(prediction_records, "predictions")

    truth_ids = set(ground_truth_by_id)
    prediction_ids = set(predictions_by_id)
    missing_prediction_ids = truth_ids - prediction_ids
    extra_prediction_ids = prediction_ids - truth_ids
    matched_ids = sorted(truth_ids & prediction_ids)

    category_correct = 0
    urgency_correct = 0
    should_draft_correct = 0
    sensitive_total = 0
    false_drafts = 0
    draftable_total = 0
    false_no_drafts = 0
    evaluated = 0
    verbose_mismatches: list[tuple[str, dict[str, Any], list[tuple[str, Any, Any]]]] = []

    confidence_values: list[float] = []
    correct_confidence_values: list[float] = []
    incorrect_confidence_values: list[float] = []
    confusion_matrix: dict[str, Counter[str]] = {
        urgency: Counter({predicted: 0 for predicted in URGENCY_CLASSES})
        for urgency in URGENCY_CLASSES
    }

    for ticket_id in matched_ids:
        truth = ground_truth_by_id[ticket_id]
        prediction = predictions_by_id[ticket_id]
        label = get_label(truth)

        if label is None:
            print(f"Skipping {ticket_id}: ground truth record missing label object")
            continue

        evaluated += 1
        truth_category = label.get("category")
        truth_urgency = label.get("urgency")
        truth_should_draft = label.get("should_draft")

        predicted_category = prediction.get("category")
        predicted_urgency = prediction.get("urgency")
        predicted_should_draft = prediction.get("should_draft")

        category_matches = predicted_category == truth_category
        urgency_matches = predicted_urgency == truth_urgency
        should_draft_matches = predicted_should_draft == truth_should_draft
        all_core_fields_match = category_matches and urgency_matches and should_draft_matches

        category_correct += int(category_matches)
        urgency_correct += int(urgency_matches)
        should_draft_correct += int(should_draft_matches)

        if truth_urgency in URGENCY_CLASSES and predicted_urgency in URGENCY_CLASSES:
            confusion_matrix[truth_urgency][predicted_urgency] += 1

        if truth_should_draft is False:
            sensitive_total += 1
            false_drafts += int(predicted_should_draft is True)

        if truth_should_draft is True:
            draftable_total += 1
            false_no_drafts += int(predicted_should_draft is False)

        if verbose and not all_core_fields_match:
            mismatches: list[tuple[str, Any, Any]] = []
            if not category_matches:
                mismatches.append(("category", truth_category, predicted_category))
            if not urgency_matches:
                mismatches.append(("urgency", truth_urgency, predicted_urgency))
            if not should_draft_matches:
                mismatches.append(("should_draft", truth_should_draft, predicted_should_draft))
            verbose_mismatches.append((ticket_id, truth, mismatches))

        confidence = prediction.get("confidence")
        if isinstance(confidence, int | float):
            confidence_values.append(float(confidence))
            if all_core_fields_match:
                correct_confidence_values.append(float(confidence))
            else:
                incorrect_confidence_values.append(float(confidence))

    print("ID checks")
    print_list_report("Duplicate ground truth ticket IDs", duplicate_truth_ids)
    print_list_report("Duplicate prediction ticket IDs", duplicate_prediction_ids)
    print_list_report("Missing prediction ticket IDs", missing_prediction_ids)
    print_list_report("Extra prediction ticket IDs", extra_prediction_ids)

    print("\nCore metrics")
    print(f"Evaluated matched tickets: {evaluated}")
    print(f"Category accuracy: {format_percent(category_correct, evaluated)} ({category_correct}/{evaluated})")
    print(f"Urgency accuracy: {format_percent(urgency_correct, evaluated)} ({urgency_correct}/{evaluated})")
    print(
        f"Should-draft accuracy: {format_percent(should_draft_correct, evaluated)} "
        f"({should_draft_correct}/{evaluated})"
    )

    print("\nUrgency confusion matrix")
    print_confusion_matrix(confusion_matrix)

    print("\nFalse draft on sensitive tickets")
    print(f"Total sensitive tickets: {sensitive_total}")
    print(f"False drafts: {false_drafts}")
    print(f"False draft rate: {format_percent(false_drafts, sensitive_total)}")

    print("\nFalse no-draft on draftable tickets")
    print(f"Total draftable tickets: {draftable_total}")
    print(f"False no-drafts: {false_no_drafts}")
    print(f"False no-draft rate: {format_percent(false_no_drafts, draftable_total)}")

    print("\nAdditional metrics")
    print(f"Average confidence: {format_average(average(confidence_values))}")
    print(f"Average confidence for correct predictions: {format_average(average(correct_confidence_values))}")
    print(f"Average confidence for incorrect predictions: {format_average(average(incorrect_confidence_values))}")

    if verbose:
        print("\nVerbose mismatches")
        if not verbose_mismatches:
            print("No category, urgency, or should_draft mismatches.")
        for ticket_id, truth, mismatches in verbose_mismatches:
            print_verbose_mismatch(ticket_id, truth, mismatches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate support-ticket triage predictions against labeled JSONL ground truth."
    )
    parser.add_argument("ground_truth_path", type=Path, help="Path to labeled ground truth .jsonl")
    parser.add_argument("predictions_path", type=Path, help="Path to prediction .jsonl")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subject, body, and field-level details for incorrect category, urgency, or should_draft predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth_records = read_jsonl(args.ground_truth_path)
    prediction_records = read_jsonl(args.predictions_path)
    evaluate(ground_truth_records, prediction_records, verbose=args.verbose)


if __name__ == "__main__":
    main()
