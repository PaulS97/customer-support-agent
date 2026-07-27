"""Draft-response generation for tickets classified as safe to draft."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import call_responses_api, load_prompt


DRAFT_PROMPT_PATH = Path("prompts/draft.md")


def generate_draft_response(
    client: Any,
    ticket: dict[str, Any],
    classification: dict[str, Any],
    prompt_path: Path = DRAFT_PROMPT_PATH,
) -> str:
    instructions = load_prompt(prompt_path)
    payload = {
        "ticket": ticket,
        "classification": classification,
    }
    return call_responses_api(client, instructions, payload).strip()
