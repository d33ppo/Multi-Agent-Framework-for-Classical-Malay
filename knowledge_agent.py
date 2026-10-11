"""
Knowledge extraction subagent: modern Malay text -> structured facts.
"""

from __future__ import annotations

from llm_text_runtime import LLMTextRuntime, extract_json_object


SYSTEM_PROMPT = """You extract structured knowledge from historical Malay texts.

Return valid JSON only with this exact schema:
{
  "people": [],
  "places": [],
  "dates": [],
  "organizations": [],
  "themes": [],
  "key_claims": []
}

Rules:
- Use modern Malay labels/content where appropriate.
- If a field has no evidence, return an empty list.
- Do not add keys outside the schema.
- Do not include markdown fences or explanation."""


def run_knowledge_agent(modern_malay_translation: str, runtime: LLMTextRuntime) -> dict:
    if runtime.is_mock:
        return _mock_knowledge(modern_malay_translation)

    response = runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Ekstrak pengetahuan berstruktur daripada teks ini:\n\n{modern_malay_translation}",
        max_tokens=10000,
        temperature=0.1,
    )
    return extract_json_object(response)


def _mock_knowledge(text: str) -> dict:
    preview = text.splitlines()[0] if text.strip() else ""
    return {
        "people": [],
        "places": [],
        "dates": [],
        "organizations": [],
        "themes": ["ujian pipeline pasca-OCR"],
        "key_claims": [preview] if preview else [],
    }
