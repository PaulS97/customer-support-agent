"""Draft-response generation for tickets classified as safe to draft."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .llm import LLMClient, load_prompt
except ImportError:
    from llm import LLMClient, load_prompt


DRAFT_PROMPT_PATH = Path("prompts/draft.md")


def generate_draft_response(
    llm_client: LLMClient,
    ticket: dict[str, Any],
    classification: dict[str, Any],
    model: str,
    prompt_path: Path = DRAFT_PROMPT_PATH,
) -> str:
    instructions = load_prompt(prompt_path)
    payload = {
        "ticket": ticket,
        "classification": classification,
    }
    return llm_client.generate(instructions, payload, model).strip()
