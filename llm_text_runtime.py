"""
Text-generation runtime for post-OCR agents.

Supports:
  - OpenRouter (https://openrouter.ai) — access to hundreds of models,
    called through LangChain's ChatOpenAI pointed at the OpenRouter base URL
  - mock mode for offline/local verification

Live mode sends manuscript text to OpenRouter, so it needs both
OPENROUTER_API_KEY and ALLOW_EXTERNAL_MANUSCRIPT_API=true. Mock mode is never
chosen silently: the caller must ask for it (provider="mock") or opt in to the
fallback (allow_mock_fallback=True, used by the CLI and by APP_MODE=demo_mock).
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT_SECONDS = 120
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEMO_MOCK_MODE = "demo_mock"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class LLMNotConfiguredError(EnvironmentError):
    """Live calls to an external provider are not possible with the current configuration."""


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


def env_value(name: str, default: str = "") -> str:
    """Read an env var, treating unset, empty and whitespace-only values as the default."""
    return (os.environ.get(name) or "").strip() or default


def env_flag(name: str) -> bool:
    return env_value(name).lower() in _TRUE_VALUES


def demo_mock_enabled() -> bool:
    """True when APP_MODE=demo_mock, the only mode in which the API may serve mock output."""
    return env_value("APP_MODE").lower() == DEMO_MOCK_MODE


def external_api_allowed() -> bool:
    """True when manuscript text or images may be sent to an external provider."""
    return env_flag("ALLOW_EXTERNAL_MANUSCRIPT_API")


def default_model() -> str:
    return env_value("OPENROUTER_MODEL", DEFAULT_MODEL)


def allowed_models() -> list[str]:
    """Models an API client may request, from OPENROUTER_ALLOWED_MODELS (comma-separated)."""
    names = [name.strip() for name in env_value("OPENROUTER_ALLOWED_MODELS").split(",")]
    return list(dict.fromkeys(name for name in names if name)) or [DEFAULT_MODEL]


def live_config_problem() -> str | None:
    """Say why live OpenRouter calls are unavailable, or return None when they are allowed."""
    if not env_value("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY is not set. Add it to your .env file."
    if not external_api_allowed():
        return (
            "ALLOW_EXTERNAL_MANUSCRIPT_API is not true, so manuscript text may not be sent "
            "to OpenRouter. Set ALLOW_EXTERNAL_MANUSCRIPT_API=true in your .env file to allow it."
        )
    return None


@dataclass
class RuntimeSelection:
    provider: str
    mode: str


def resolve_runtime(allow_mock_fallback: bool = False) -> RuntimeSelection:
    problem = live_config_problem()
    if problem is None:
        return RuntimeSelection(provider="openrouter", mode="live")

    if allow_mock_fallback:
        return RuntimeSelection(provider="mock", mode="mock")

    raise LLMNotConfiguredError(problem)


class LLMTextRuntime:
    """Thin wrapper around LangChain's ChatOpenAI, pointed at OpenRouter, used by the pipeline agents."""

    def __init__(
        self,
        model: str | None = None,
        allow_mock_fallback: bool = False,
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

        self.model = "mock" if self.mode == "mock" else (model or default_model())
        self._chat = None  # ChatOpenAI, built on the first live call

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
        """Generate plain-text output via OpenRouter, using LangChain's ChatOpenAI."""
        if self.is_mock:
            raise RuntimeError("Mock runtime does not support free-form generation.")

        # Checked again at call time so no code path can send text once the flag is off.
        problem = live_config_problem()
        if problem:
            raise LLMNotConfiguredError(problem)

        from langchain_core.messages import HumanMessage, SystemMessage
        from openai import APIStatusError

        try:
            reply = self._chat_model().invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                temperature=temperature,
                # Sent in the request body as "max_tokens", the field OpenRouter documents.
                # ChatOpenAI would rename its own max_tokens setting to "max_completion_tokens".
                extra_body={"max_tokens": _get_int_env("OPENROUTER_MAX_TOKENS") or max_tokens},
            )
        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenRouter API error {exc.status_code}: {exc.response.text[:500]}"
            ) from exc

        return _extract_text(_as_openai_response(reply))

    def _chat_model(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=self.model,
                base_url=OPENROUTER_BASE_URL,
                api_key=env_value("OPENROUTER_API_KEY"),
                timeout=OPENROUTER_TIMEOUT_SECONDS,
                # The Runtime Fault Handler owns retries. The SDK default of 2 would multiply them.
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/malay-fyp/classical-malay-pipeline",
                    "X-Title": "Classical Malay Multi-Agent Pipeline",
                },
            )
        return self._chat


def _as_openai_response(reply) -> dict:
    """Put a LangChain AIMessage back into the response shape `_extract_text` checks."""
    metadata = reply.response_metadata or {}
    return {
        "model": metadata.get("model_name", "unknown"),
        "choices": [
            {"finish_reason": metadata.get("finish_reason"), "message": {"content": reply.content}}
        ],
    }


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
