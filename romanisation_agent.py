"""
SubAgent 1 — Romanisation: Jawi OCR text -> romanised Classical Malay (Rumi).

Runs two independent methods in parallel on the same input:
  - Method A (neural):     romanisation_neural.NeuralRomaniser (jawi-nanot5-small)
  - Method B (dictionary): jawi_dictionary.JawiRumiDictionary (reverse lookup)
then passes both through the Inter-Method Agreement Gate.

Method A is the primary output passed to Translation; Method B is the
independent check. If Method A fails, Method B's text is passed through
instead; if both fail, the original OCR text is passed through. Each fallback
is flagged by the Runtime Fault Handler.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from agreement_gate import AgreementGateResult, run_agreement_gate
from fault_handler import StageOutcome, run_stage
from jawi_dictionary import DictionaryRomanisation, JawiRumiDictionary
from romanisation_neural import NeuralRomaniser


class Romaniser(Protocol):
    def romanise(self, jawi_text: str) -> str: ...


class DictionaryRomaniser(Protocol):
    def romanise(self, jawi_text: str) -> DictionaryRomanisation: ...


class MockNeuralRomaniser:
    """Offline stand-in for Method A used in mock mode and tests."""

    def romanise(self, jawi_text: str) -> str:
        first_line = jawi_text.splitlines()[0] if jawi_text.strip() else ""
        return f"[MOCK NEURAL ROMANISATION] {first_line}"


@lru_cache(maxsize=1)
def default_neural_romaniser() -> NeuralRomaniser:
    return NeuralRomaniser()


@lru_cache(maxsize=1)
def default_dictionary() -> JawiRumiDictionary:
    return JawiRumiDictionary.load()


class _LazyDictionary:
    """Defers loading the lookup table so a missing file fails inside the fault handler."""

    def romanise(self, jawi_text: str) -> DictionaryRomanisation:
        return default_dictionary().romanise(jawi_text)


@dataclass
class RomanisationOutcome:
    text: str  # passed to Translation
    method_a: StageOutcome
    method_b: StageOutcome
    gate: AgreementGateResult | None  # None when either method failed
    needs_human_review: bool

    def method_b_details(self) -> dict | None:
        result = self.method_b.output
        return result.to_dict() if isinstance(result, DictionaryRomanisation) else None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "method_a": {**self.method_a.to_dict(), "text": self.method_a.output},
            "method_b": {**self.method_b.to_dict(), "details": self.method_b_details()},
            "agreement_gate": self.gate.to_dict() if self.gate else {
                "status": "skipped",
                "reason": "Gate skipped because at least one romanisation method failed.",
            },
            "needs_human_review": self.needs_human_review,
        }


def run_romanisation_agent(
    ocr_jawi_text: str,
    *,
    neural: Romaniser | None = None,
    dictionary: DictionaryRomaniser | None = None,
    timeout: float | None = None,
) -> RomanisationOutcome:
    neural = neural or default_neural_romaniser()
    dictionary = dictionary or _LazyDictionary()
    stage_kwargs = {} if timeout is None else {"timeout": timeout}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            run_stage, "romanisation_method_a", neural.romanise, ocr_jawi_text, fallback=None, **stage_kwargs
        )
        future_b = pool.submit(
            run_stage, "romanisation_method_b", dictionary.romanise, ocr_jawi_text, fallback=None, **stage_kwargs
        )
        method_a, method_b = future_a.result(), future_b.result()

    method_b_text = method_b.output.text if isinstance(method_b.output, DictionaryRomanisation) else None

    gate = None
    if not method_a.incomplete and method_b_text is not None:
        gate = run_agreement_gate(
            method_a.output,
            method_b_text,
            method_b_coverage=method_b.output.coverage,
        )

    if not method_a.incomplete:
        text = method_a.output
    elif method_b_text:
        text = method_b_text
    else:
        text = ocr_jawi_text  # last valid output: the original OCR text

    needs_review = gate is None or not gate.passed
    return RomanisationOutcome(text.strip(), method_a, method_b, gate, needs_review)
