"""
Romanisation subagent: Jawi OCR text -> romanised Classical Malay.
"""

from __future__ import annotations

from llm_text_runtime import LLMTextRuntime


SYSTEM_PROMPT = """You are a specialist in Classical Malay written in Jawi.
Convert Jawi script into romanised Classical Malay in Rumi script.

Rules:
- Preserve names, dates, titles, and line structure where reasonable.
- Do not translate into modern Malay yet.
- Do not explain your reasoning.
- Output only the romanised Classical Malay text."""


def run_romanisation_agent(ocr_jawi_text: str, runtime: LLMTextRuntime) -> str:
    if runtime.is_mock:
        return _mock_romanisation(ocr_jawi_text)

    return runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Romanise this Jawi OCR text into Classical Malay Rumi:\n\n{ocr_jawi_text}",
        max_tokens=10000,
        temperature=0.1,
    )


def _mock_romanisation(text: str) -> str:
    first_line = text.splitlines()[0] if text.strip() else ""
    return (
        "[MOCK ROMANISATION]\n"
        "Romanised Classical Malay output requires an LLM/API provider.\n"
        f"Source preview: {first_line}"
    )
