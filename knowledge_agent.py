"""
Knowledge extraction subagent: modern Malay text -> structured facts.

The model output is validated against a fixed schema. Invalid output gets one
repair call. If it is still invalid the agent raises KnowledgeValidationError,
and the Runtime Fault Handler marks the stage incomplete.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_text_runtime import LLMTextRuntime, extract_json_object

logger = logging.getLogger(__name__)

SCHEMA_KEYS = ("people", "places", "dates", "organizations", "themes", "key_claims")
EVIDENCE_KEY = "evidence"
MAX_REPAIR_INPUT_CHARS = 20000

SYSTEM_PROMPT = """You extract structured knowledge from historical Malay texts.

Return valid JSON only with this exact schema:
{
  "people": [],
  "places": [],
  "dates": [],
  "organizations": [],
  "themes": [],
  "key_claims": [],
  "evidence": [{"entity": "", "quote": ""}]
}

Rules:
- Every field except "evidence" is a list of plain strings.
- Use modern Malay labels/content where appropriate.
- If a field has no evidence, return an empty list.
- "evidence" is optional. Each item names one extracted entity and gives a short quote
  copied word for word from the input text that supports it. Never paraphrase a quote.
- Do not add keys outside the schema.
- Do not include markdown fences or explanation."""


class KnowledgeValidationError(ValueError):
    """The model output cannot be used as a knowledge extraction result."""


def run_knowledge_agent(modern_malay_translation: str, runtime: LLMTextRuntime) -> dict:
    if runtime.is_mock:
        return _mock_knowledge(modern_malay_translation)

    response = runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Ekstrak pengetahuan berstruktur daripada teks ini:\n\n{modern_malay_translation}",
        max_tokens=10000,
        temperature=0.1,
    )
    try:
        return parse_knowledge(response, modern_malay_translation)
    except KnowledgeValidationError as exc:
        problem = str(exc)

    logger.warning("Knowledge extraction output was invalid (%s). Making one repair call.", problem)
    repaired = runtime.generate_text(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_repair_prompt(response, problem),
        max_tokens=10000,
        temperature=0.0,
    )
    try:
        return parse_knowledge(repaired, modern_malay_translation)
    except KnowledgeValidationError as exc:
        raise KnowledgeValidationError(f"Output still invalid after one repair call: {exc}") from exc


def parse_knowledge(response: str, source_text: str) -> dict:
    """Parse a model response and validate it. Raises KnowledgeValidationError if it cannot be used."""
    try:
        data = extract_json_object(response)
    except ValueError as exc:  # also covers json.JSONDecodeError
        raise KnowledgeValidationError(f"Response is not valid JSON: {exc}") from exc
    return validate_knowledge(data, source_text)


def validate_knowledge(data: Any, source_text: str) -> dict:
    """
    Return the result with exactly the six schema keys plus the verified evidence list.

    Unknown keys are dropped and a single string is accepted as a one-item list.
    A missing key, or a value that is not a list of strings, is an error.
    """
    if not isinstance(data, dict):
        raise KnowledgeValidationError("Top-level JSON value must be an object.")

    missing = [key for key in SCHEMA_KEYS if key not in data]
    if missing:
        raise KnowledgeValidationError(f"Missing schema keys: {', '.join(missing)}.")

    result: dict[str, Any] = {key: _string_list(key, data[key]) for key in SCHEMA_KEYS}
    result[EVIDENCE_KEY] = _verified_evidence(data.get(EVIDENCE_KEY), source_text)
    return result


def _string_list(key: str, value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise KnowledgeValidationError(f"'{key}' must be a list of strings.")
    return [item.strip() for item in value if item.strip()]


def _verified_evidence(value: Any, source_text: str) -> list[dict[str, str]]:
    """
    Keep only evidence whose quote appears word for word in the source text.

    Evidence is optional, so malformed or unsupported items are dropped rather than
    failing the stage. Whitespace runs are collapsed before comparing; case is kept.
    """
    if not isinstance(value, list):
        return []

    haystack = _collapse_whitespace(source_text)
    kept = []
    for item in value:
        if not isinstance(item, dict):
            continue
        entity, quote = item.get("entity"), item.get("quote")
        if not isinstance(entity, str) or not isinstance(quote, str):
            continue
        entity, quote = entity.strip(), _collapse_whitespace(quote)
        if entity and quote and quote in haystack:
            kept.append({"entity": entity, "quote": quote})

    if len(kept) < len(value):
        logger.info("Dropped %d evidence item(s) not found in the input text.", len(value) - len(kept))
    return kept


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _repair_prompt(bad_output: str, problem: str) -> str:
    return (
        "Your previous answer could not be used.\n"
        f"Problem: {problem}\n\n"
        "Previous answer:\n"
        f"{bad_output[:MAX_REPAIR_INPUT_CHARS]}\n\n"
        "Return the same information as valid JSON only, following the schema exactly. "
        "No markdown fences and no explanation."
    )


def _mock_knowledge(text: str) -> dict:
    preview = text.splitlines()[0] if text.strip() else ""
    return {
        "people": [],
        "places": [],
        "dates": [],
        "organizations": [],
        "themes": ["ujian pipeline pasca-OCR"],
        "key_claims": [preview] if preview else [],
        "evidence": [],
    }
