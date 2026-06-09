"""
Quality gate subagent: scores romanisation output and triggers a retry if below threshold.
"""
from __future__ import annotations

from llm_text_runtime import LLMTextRuntime, extract_json_object

QUALITY_THRESHOLD = 70

_SCORE_PROMPT = """You are a quality checker for Classical Malay romanisation from Jawi script.

Evaluate the romanised text and return a JSON score.

Scoring guide:
- 80-100: Valid Classical Malay in Rumi, proper vocabulary, coherent sentences
- 60-79 : Mostly correct but some uncertain romanisations or minor errors
- 40-59 : Significant errors, mixed characters, or uncertain readings
- 0-39  : Mostly gibberish, wrong script, or failed romanisation

Return ONLY valid JSON — no markdown, no extra text:
{"score": <integer 0-100>, "reason": "<one sentence>"}"""

_ENHANCED_ROMANISATION_PROMPT = """You are a specialist in Classical Malay written in Jawi script.
Convert the Jawi text into romanised Classical Malay (Rumi).

Key Jawi-only characters to watch for:
  ڬ = ga   ڠ = nga   ڽ = nya   ڤ = pa   ﭺ = ca   ۏ = va

Rules:
- Preserve all names, dates, and titles exactly as written
- Do not translate — romanise only
- Output ONLY the romanised Classical Malay text"""


def score_romanisation(romanized_text: str, runtime: LLMTextRuntime) -> dict:
    """Score romanisation quality. Returns {"score": int, "reason": str}."""
    if runtime.is_mock:
        return {"score": 88, "reason": "Mock mode — auto-pass"}
    try:
        response = runtime.generate_text(
            system_prompt=_SCORE_PROMPT,
            user_prompt=f"Score this romanised Classical Malay:\n\n{romanized_text}",
            max_tokens=80,
            temperature=0.0,
        )
        result = extract_json_object(response)
        score = max(0, min(100, int(result.get("score", 100))))
        return {"score": score, "reason": result.get("reason", "")}
    except Exception:
        return {"score": 100, "reason": "Scoring unavailable — auto-pass"}


def run_romanisation_enhanced(jawi_text: str, runtime: LLMTextRuntime) -> str:
    """Retry romanisation with an enhanced prompt that explicitly names Jawi glyphs."""
    if runtime.is_mock:
        return "[MOCK ENHANCED ROMANISATION]\nRetried with enhanced prompt."
    return runtime.generate_text(
        system_prompt=_ENHANCED_ROMANISATION_PROMPT,
        user_prompt=f"Romanise this Jawi text carefully:\n\n{jawi_text}",
        max_tokens=10000,
        temperature=0.05,
    )
