"""
Stage Evaluation Metrics (offline, for thesis reporting — NOT a runtime gate).

| Stage                | Method                                             |
|----------------------|----------------------------------------------------|
| Romanisation         | BLEU + chrF vs Majalah Qalam romanised gold text   |
| Translation          | Human rubric (adequacy + fluency, 1-5), aggregated |
| Summarisation        | ROUGE-1/2/L vs human-written reference summaries   |
| Knowledge extraction | Precision/recall/F1 vs manually tagged gold entities |

LLM-as-a-judge is deliberately not used anywhere (rejected by panel/supervisor).

Inputs:
  romanisation / summary : JSONL, one {"id", "hypothesis", "reference"} per line
  knowledge              : JSONL, one {"id", "prediction": {...}, "gold": {...}} per line
  rubric                 : CSV with columns item_id, rater, adequacy, fluency

Usage:
    python stage_metrics.py romanisation eval/romanisation.jsonl
    python stage_metrics.py summary eval/summaries.jsonl
    python stage_metrics.py knowledge eval/knowledge.jsonl
    python stage_metrics.py rubric eval/translation_rubric.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

# Themes and claims are free text, so exact-match F1 is only meaningful for named entities.
ENTITY_FIELDS = ("people", "places", "dates", "organizations")
RUBRIC_SCALE = (1, 5)


def romanisation_scores(hypotheses: list[str], references: list[str]) -> dict:
    """Corpus BLEU and chrF (sacrebleu defaults) against gold Rumi text."""
    import sacrebleu

    _check_aligned(hypotheses, references)
    return {
        "items": len(hypotheses),
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2),
    }


def summary_scores(hypotheses: list[str], references: list[str]) -> dict:
    """Mean ROUGE-1/2/L F-measure. No stemming: the Porter stemmer is English-only."""
    from rouge_score import rouge_scorer

    _check_aligned(hypotheses, references)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp) for hyp, ref in zip(hypotheses, references)]
    return {
        "items": len(scores),
        **{name: round(mean(s[name].fmeasure for s in scores), 4) for name in ("rouge1", "rouge2", "rougeL")},
    }


def _norm_entity(value: object) -> str:
    return " ".join(str(value).lower().split())


def entity_f1(predictions: list[dict], golds: list[dict], fields: tuple[str, ...] = ENTITY_FIELDS) -> dict:
    """Micro-averaged exact-match precision/recall/F1 per field and overall."""
    _check_aligned(predictions, golds)
    totals = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}

    for pred, gold in zip(predictions, golds):
        for f in fields:
            p = {_norm_entity(v) for v in pred.get(f, []) if str(v).strip()}
            g = {_norm_entity(v) for v in gold.get(f, []) if str(v).strip()}
            totals[f]["tp"] += len(p & g)
            totals[f]["fp"] += len(p - g)
            totals[f]["fn"] += len(g - p)

    def prf(tp: int, fp: int, fn: int) -> dict:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    overall = {k: sum(t[k] for t in totals.values()) for k in ("tp", "fp", "fn")}
    return {
        "items": len(golds),
        "per_field": {f: prf(**t) for f, t in totals.items()},
        "micro": prf(**overall),
    }


def rubric_summary(rows: list[dict]) -> dict:
    """Aggregate human adequacy/fluency ratings (1-5) for the translation stage."""
    low, high = RUBRIC_SCALE
    result = {"ratings": len(rows), "items": len({r["item_id"] for r in rows}),
              "raters": sorted({r["rater"] for r in rows})}
    for criterion in ("adequacy", "fluency"):
        values = [int(r[criterion]) for r in rows]
        bad = [v for v in values if not low <= v <= high]
        if bad:
            raise ValueError(f"{criterion} ratings outside {low}-{high}: {bad}")
        result[criterion] = {
            "mean": round(mean(values), 3) if values else None,
            "stdev": round(stdev(values), 3) if len(values) > 1 else None,
        }
    return result


def _check_aligned(a: list, b: list) -> None:
    if len(a) != len(b):
        raise ValueError(f"Hypotheses ({len(a)}) and references ({len(b)}) differ in length.")
    if not a:
        raise ValueError("No items to evaluate.")


def _read_jsonl(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Offline stage evaluation metrics (no LLM judge).")
    parser.add_argument("stage", choices=["romanisation", "summary", "knowledge", "rubric"])
    parser.add_argument("path")
    args = parser.parse_args()

    if args.stage == "rubric":
        with open(args.path, encoding="utf-8", newline="") as handle:
            result = rubric_summary(list(csv.DictReader(handle)))
    elif args.stage == "knowledge":
        rows = _read_jsonl(args.path)
        result = entity_f1([r["prediction"] for r in rows], [r["gold"] for r in rows])
    else:
        rows = _read_jsonl(args.path)
        scorer = romanisation_scores if args.stage == "romanisation" else summary_scores
        result = scorer([r["hypothesis"] for r in rows], [r["reference"] for r in rows])

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
