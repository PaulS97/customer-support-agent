"""Shared OpenAI Responses API helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MODEL = "gpt-4.1-mini"


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def create_openai_client() -> Any:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install with: python3 -m pip install -r requirements.txt"
        ) from exc

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    return OpenAI()


def call_responses_api(
    client: Any,
    instructions: str,
    payload: dict[str, Any],
    response_format: dict[str, Any] | None = None,
) -> str:
    response_kwargs: dict[str, Any] = {
        "model": MODEL,
        "instructions": instructions,
        "input": json.dumps(payload, indent=2, ensure_ascii=False),
    }
    if response_format is not None:
        response_kwargs["text"] = response_format

    response = client.responses.create(**response_kwargs)
    return response.output_text


def parse_json_object(response_text: str) -> dict[str, Any]:
    parsed = json.loads(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed
