"""Provider-agnostic LLM interface for pipeline model calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal


Provider = Literal["openai", "anthropic"]
DEFAULT_MAX_TOKENS = 2000


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def detect_provider(model: str) -> Provider:
    normalized = model.lower()
    if normalized.startswith("claude-"):
        return "anthropic"
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    raise ValueError(f"Could not detect provider from model name: {model}")


def parse_json_object(response_text: str) -> dict[str, Any]:
    parsed = json.loads(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed


class LLMClient:
    """Shared model client that isolates provider-specific SDK calls."""

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency. Install with: python3 -m pip install -r requirements.txt"
            ) from exc

        load_dotenv()
        self._clients: dict[Provider, Any] = {}

    def generate(
        self,
        prompt: str,
        payload: dict[str, Any],
        model: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        provider = detect_provider(model)
        if provider == "openai":
            return self._generate_openai(prompt, payload, model, response_format)
        if provider == "anthropic":
            return self._generate_anthropic(prompt, payload, model, response_format)

        raise ValueError(f"Unsupported provider: {provider}")

    def _get_openai_client(self) -> Any:
        if "openai" not in self._clients:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Missing OpenAI dependency. Install with: python3 -m pip install -r requirements.txt"
                ) from exc
            self._clients["openai"] = OpenAI()

        return self._clients["openai"]

    def _get_anthropic_client(self) -> Any:
        if "anthropic" not in self._clients:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Missing Anthropic dependency. Install with: python3 -m pip install -r requirements.txt"
                ) from exc
            self._clients["anthropic"] = Anthropic()

        return self._clients["anthropic"]

    def _generate_openai(
        self,
        prompt: str,
        payload: dict[str, Any],
        model: str,
        response_format: dict[str, Any] | None,
    ) -> str:
        request: dict[str, Any] = {
            "model": model,
            "instructions": prompt,
            "input": self._format_payload(payload),
        }
        if response_format is not None:
            request["text"] = response_format

        response = self._get_openai_client().responses.create(**request)
        return response.output_text

    def _generate_anthropic(
        self,
        prompt: str,
        payload: dict[str, Any],
        model: str,
        response_format: dict[str, Any] | None,
    ) -> str:
        if response_format is not None:
            prompt = f"{prompt}\n\nReturn only a valid JSON object with no markdown fences or commentary."

        response = self._get_anthropic_client().messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": self._format_payload(payload),
                }
            ],
        )
        return self._anthropic_text(response)

    @staticmethod
    def _format_payload(payload: dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _anthropic_text(response: Any) -> str:
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


def create_llm_client() -> LLMClient:
    return LLMClient()
