"""
Translation subagent: romanised Classical Malay -> modern Malay.
"""

from __future__ import annotations

from llm_text_runtime import LLMTextRuntime


SYSTEM_PROMPT = """You are a careful translator for Classical Malay texts.
Translate the input into modern standard Malay.

Rules:
- Preserve meaning faithfully.
- Prefer clear contemporary Malay.
- Keep personal names, place names, and dates intact.
- Do not summarise or omit details.
- Output only the translated modern Malay text."""


def run_translation_agent(romanized_text: str, runtime: LLMTextRuntime) -> str:
    if runtime.is_mock:
        return _mock_translation(romanized_text)

    return runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Terjemahkan teks Melayu Klasik beroman ini ke Bahasa Melayu moden:\n\n{romanized_text}",
        max_tokens=10000,
        temperature=0.15,
    )


def _mock_translation(text: str) -> str:
    preview = text.splitlines()[0] if text.strip() else ""
    return (
        "[MOCK TRANSLATION]\n"
        "Terjemahan Bahasa Melayu moden memerlukan provider LLM/API.\n"
        f"Input preview: {preview}"
    )
