"""
Text-generation runtime for post-OCR agents.

Supports:
  - OpenRouter (https://openrouter.ai) — access to hundreds of models
  - mock mode for offline/local verification
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs):
        return False


ENV_FILE = Path(__file__).resolve().parent / ".env"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def _load_env_file(env_file: Path = ENV_FILE) -> None:
    """Load .env variables with a stdlib fallback when python-dotenv is absent."""
    if load_dotenv(env_file):
        return

    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()


def _get_int_env(*names: str) -> int | None:
    """Return the first valid positive integer found in the given env vars."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


@dataclass
class RuntimeSelection:
    provider: str
    mode: str


def resolve_runtime(allow_mock_fallback: bool = True) -> RuntimeSelection:
    if os.environ.get("OPENROUTER_API_KEY"):
        return RuntimeSelection(provider="openrouter", mode="live")

    if allow_mock_fallback:
        return RuntimeSelection(provider="mock", mode="mock")

    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
    )


class LLMTextRuntime:
    """Thin wrapper around OpenRouter used by the pipeline agents."""

    def __init__(
        self,
        model: str | None = None,
        allow_mock_fallback: bool = True,
        # kept for backward-compat with pipeline_runner.py CLI
        provider: str = "auto",
    ):
        if provider == "mock":
            self.provider = "mock"
            self.mode = "mock"
        else:
            selection = resolve_runtime(allow_mock_fallback=allow_mock_fallback)
            self.provider = selection.provider
            self.mode = selection.mode

        self.model = (
            "mock"
            if self.mode == "mock"
            else (model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
        )

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 10000,
        temperature: float = 0.2,
    ) -> str:
        """Generate plain-text output via OpenRouter."""
        if self.is_mock:
            raise RuntimeError("Mock runtime does not support free-form generation.")

        import requests

        effective_max_tokens = (
            _get_int_env("OPENROUTER_MAX_TOKENS") or max_tokens
        )

        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/malay-fyp/classical-malay-pipeline",
            "X-Title": "Classical Malay Multi-Agent Pipeline",
        }
        payload = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=120
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        return _extract_text(response.json())


def _extract_text(data: dict) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"OpenRouter response missing choices: {json.dumps(data)[:500]}"
        )

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    finish_reason = choice.get("finish_reason")
    model_name = data.get("model", "unknown")
    if finish_reason == "length":
        raise RuntimeError(
            f"Model '{model_name}' hit the output token limit before returning usable text. "
            f"Try increasing max_tokens or set OPENROUTER_MAX_TOKENS in your .env."
        )

    raise RuntimeError(
        f"OpenRouter returned empty content for model '{model_name}' "
        f"(finish_reason={finish_reason})."
    )


# Backward-compat alias used in tests
_extract_openai_compatible_text = _extract_text


def extract_json_object(text: str) -> dict:
    """Parse JSON from a response that may contain surrounding prose/code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found in model response.")

    return json.loads(text[start : end + 1])
