"""
Summarisation subagent: modern Malay text -> concise summary.
"""

from __future__ import annotations

from llm_text_runtime import LLMTextRuntime


SYSTEM_PROMPT = """You summarise historical Malay texts for researchers.

Rules:
- Write in modern Malay.
- Keep the summary concise and factual.
- Focus on the core topic, actors, place, and time if present.
- Do not invent missing details.
- Output only the summary."""


def run_summary_agent(modern_malay_translation: str, runtime: LLMTextRuntime) -> str:
    if runtime.is_mock:
        return _mock_summary(modern_malay_translation)

    return runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Ringkaskan teks Bahasa Melayu moden berikut dalam 3-5 ayat:\n\n{modern_malay_translation}",
        max_tokens=10000,
        temperature=0.2,
    )


def _mock_summary(text: str) -> str:
    preview = text.splitlines()[0] if text.strip() else ""
    return (
        "Ringkasan mock: Teks ini diproses melalui pipeline pasca-OCR. "
        f"Baris awal input ialah '{preview}'."
    )
