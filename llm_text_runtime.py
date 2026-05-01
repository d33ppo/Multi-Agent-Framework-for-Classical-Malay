"""
Reusable text-generation runtime for post-OCR agents.

Supports:
  - Claude via Anthropic SDK
  - ILMU via OpenAI-compatible REST API
  - z.AI via OpenAI-compatible REST API
  - mock mode for offline/local verification
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in minimal environments

    def load_dotenv(*_args, **_kwargs):
        return False


ENV_FILE = Path(__file__).resolve().parent / ".env"


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


AUTO_PROVIDER_ORDER = ("claude", "ilmu", "zai")


@dataclass
class RuntimeSelection:
    provider: str
    mode: str


def _has_key(provider: str) -> bool:
    env_map = {
        "claude": "ANTHROPIC_API_KEY",
        "ilmu": "ILMU_API_KEY",
        "zai": "ZAI_API_KEY",
    }
    env_name = env_map.get(provider)
    return bool(env_name and os.environ.get(env_name))


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


def resolve_runtime(
    provider: str = "auto", allow_mock_fallback: bool = True
) -> RuntimeSelection:
    provider = provider.lower().strip()

    if provider == "mock":
        return RuntimeSelection(provider="mock", mode="mock")

    if provider != "auto":
        if _has_key(provider):
            return RuntimeSelection(provider=provider, mode="live")
        if allow_mock_fallback:
            return RuntimeSelection(provider="mock", mode="mock")
        raise EnvironmentError(
            f"Provider '{provider}' is not configured in environment."
        )

    for candidate in AUTO_PROVIDER_ORDER:
        if _has_key(candidate):
            return RuntimeSelection(provider=candidate, mode="live")

    if allow_mock_fallback:
        return RuntimeSelection(provider="mock", mode="mock")

    raise EnvironmentError(
        "No text-generation provider configured and mock fallback disabled."
    )


class LLMTextRuntime:
    """Small wrapper around text-only providers used by the pipeline agents."""

    def __init__(self, provider: str = "auto", allow_mock_fallback: bool = True):
        selection = resolve_runtime(
            provider=provider, allow_mock_fallback=allow_mock_fallback
        )
        self.provider = selection.provider
        self.mode = selection.mode

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
        """Generate plain-text output using the selected provider."""
        if self.is_mock:
            raise RuntimeError("Mock runtime does not support free-form generation.")

        if self.provider == "claude":
            return self._generate_claude(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        if self.provider == "ilmu":
            ilmu_model = os.environ.get(
                "PIPELINE_ILMU_MODEL", os.environ.get("ILMU_MODEL", "ilmu-vision")
            )
            ilmu_max_tokens = (
                _get_int_env("PIPELINE_ILMU_MAX_TOKENS", "ILMU_MAX_TOKENS")
                or max_tokens
            )
            return self._generate_openai_compatible(
                api_key=os.environ["ILMU_API_KEY"],
                api_url=os.environ.get(
                    "ILMU_API_URL", "https://api.ilmu.ai/v1/chat/completions"
                ),
                model_name=ilmu_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=ilmu_max_tokens,
                temperature=temperature,
            )
        if self.provider == "zai":
            return self._generate_openai_compatible(
                api_key=os.environ["ZAI_API_KEY"],
                api_url=os.environ.get(
                    "ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions"
                ),
                model_name=os.environ.get(
                    "PIPELINE_ZAI_MODEL", os.environ.get("ZAI_MODEL", "glm-4.5")
                ),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_claude(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=os.environ.get("PIPELINE_CLAUDE_MODEL", "claude-sonnet-4-6"),
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()

    def _generate_openai_compatible(
        self,
        *,
        api_key: str,
        api_url: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Provider API error {response.status_code}: {response.text}"
            )

        data = response.json()
        return _extract_openai_compatible_text(data)


def _extract_openai_compatible_text(data: dict) -> str:
    """Extract assistant text from OpenAI-compatible responses."""
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"Provider response missing choices: {json.dumps(data)[:500]}"
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
            f"Try increasing max_tokens, setting PIPELINE_ILMU_MAX_TOKENS/ILMU_MAX_TOKENS, "
            f"shortening the prompt, or switching models. Raw response: {json.dumps(data)[:300]}"
        )

    raise RuntimeError(
        f"Provider returned empty assistant content for model '{model_name}' "
        f"(finish_reason={finish_reason})."
    )


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
