"""
Inter-Method Agreement Gate.

Runtime quality gate that sits between SubAgent 1 (Romanisation) and SubAgent 2
(Translation). It compares Method A (neural) and Method B (dictionary lookup)
romanisations of the same Jawi input at character level.

This is NOT a stage evaluation metric (those are computed offline against gold
data in stage_metrics.py) and NOT fault handling (see fault_handler.py).

Interpretation note for the thesis: both methods are built from the same modern
Jawi source data, so agreement measures consistency between the two readings,
not correctness against the original manuscript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import Levenshtein

from jawi_normalise import normalise_rumi_for_comparison

AGREEMENT_THRESHOLD = 0.85


def character_agreement(text_a: str, text_b: str) -> float:
    """
    1 - (character edit distance / length of the longer text), after
    lowercasing and removing punctuation. 1.0 means identical.
    Unknown words from Method B stay in Jawi script, so they count as disagreement.
    """
    a = normalise_rumi_for_comparison(text_a)
    b = normalise_rumi_for_comparison(text_b)
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1 - Levenshtein.distance(a, b) / longest


@dataclass
class AgreementGateResult:
    agreement: float
    threshold: float
    passed: bool
    method_b_coverage: float | None
    status: str  # "passed" | "flagged"
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_agreement_gate(
    method_a_text: str,
    method_b_text: str,
    *,
    method_b_coverage: float | None = None,
    threshold: float = AGREEMENT_THRESHOLD,
) -> AgreementGateResult:
    """Compare the two romanisations and decide whether the item needs human review."""
    agreement = round(character_agreement(method_a_text, method_b_text), 4)
    passed = agreement >= threshold

    if passed:
        reason = f"Methods A and B agree at {agreement:.1%} (threshold {threshold:.0%})."
    else:
        reason = f"Methods A and B agree at only {agreement:.1%} (threshold {threshold:.0%}); flagged for human review."
        if method_b_coverage is not None and method_b_coverage < threshold:
            reason += f" Method B dictionary covered {method_b_coverage:.1%} of words, which may explain part of the gap."

    return AgreementGateResult(
        agreement=agreement,
        threshold=threshold,
        passed=passed,
        method_b_coverage=None if method_b_coverage is None else round(method_b_coverage, 4),
        status="passed" if passed else "flagged",
        reason=reason,
    )
