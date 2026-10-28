"""
Stage Evaluation Metrics (offline, for thesis reporting — NOT a runtime gate).

| Stage                | Method                                             |
|----------------------|----------------------------------------------------|
| OCR (page level)     | Matched-line CER and page CER vs jawi_corrected    |
| Romanisation         | BLEU + chrF vs Majalah Qalam romanised gold text   |
| Translation          | Human rubric (adequacy + fluency, 1-5), aggregated |
| Summarisation        | ROUGE-1/2/L vs human-written reference summaries   |
| Knowledge extraction | Precision/recall/F1 vs manually tagged gold entities |

LLM-as-a-judge is deliberately not used anywhere (rejected by panel/supervisor).

Inputs:
  romanisation / summary : JSONL, one {"id", "hypothesis", "reference"} per line
  knowledge              : JSONL, one {"id", "prediction": {...}, "gold": {...}} per line
  rubric                 : CSV with columns item_id, rater, adequacy, fluency
  OCR page metrics       : run scripts/eval_page_cer.py, which calls the functions below

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
import re
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


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_lines(pred_boxes: list, gold_boxes: list, threshold: float = 0.5) -> list[tuple[int, int]]:
    """
    Greedy one-to-one matching at IoU >= threshold, most confident prediction first
    (the rule of yolo_prefill.match_stats). Boxes are (x1, y1, x2, y2[, conf]).
    Returns (pred_index, gold_index) pairs.
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: -(pred_boxes[i][4] if len(pred_boxes[i]) > 4 else 1))
    used, pairs = set(), []
    for i in order:
        best, best_j = 0.0, None
        for j, gold in enumerate(gold_boxes):
            if j in used:
                continue
            value = _box_iou(pred_boxes[i], gold)
            if value > best:
                best, best_j = value, j
        if best >= threshold:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def matched_line_cer(pred_lines: list[tuple], gold_lines: list[tuple], threshold: float = 0.5) -> dict:
    """
    Page-level OCR error that does not depend on reading order. Lines are (box, text).
    Matched pairs are scored by edit distance, a missed gold line counts as deleting
    all its characters and an extra predicted line as inserting all of its own.
    CER = edits / gold characters, so it can go above 1 on a page full of false lines.
    Also returns detector precision and recall at the same IoU threshold.
    """
    from Levenshtein import distance

    pairs = match_lines([box for box, _ in pred_lines], [box for box, _ in gold_lines], threshold)
    matched_pred = {i for i, _ in pairs}
    matched_gold = {j for _, j in pairs}
    matched_edits = sum(distance(gold_lines[j][1], pred_lines[i][1]) for i, j in pairs)
    missed_chars = sum(len(text) for j, (_, text) in enumerate(gold_lines) if j not in matched_gold)
    extra_chars = sum(len(text) for i, (_, text) in enumerate(pred_lines) if i not in matched_pred)
    gold_chars = sum(len(text) for _, text in gold_lines)
    edits = matched_edits + missed_chars + extra_chars
    return {
        "edits": edits,
        "matched_edits": matched_edits,
        "missed_chars": missed_chars,
        "extra_chars": extra_chars,
        "gold_chars": gold_chars,
        "cer": _ratio(edits, gold_chars),
        "gold_lines": len(gold_lines),
        "pred_lines": len(pred_lines),
        "matched_lines": len(pairs),
        "precision": _ratio(len(pairs), len(pred_lines)),
        "recall": _ratio(len(pairs), len(gold_lines)),
    }


LENGTH_MISMATCH_FRAC = 0.15  # a prediction this much longer or shorter than gold
MIN_REPEAT_RUN = 5  # this many identical characters in a row
_REPEAT_RUN_RE = re.compile(r"(.)\1{%d,}" % (MIN_REPEAT_RUN - 1), re.DOTALL)


def _intersects(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _new_repeat_run(pred: str, gold: str) -> bool:
    """True when pred has a run of MIN_REPEAT_RUN+ identical characters that gold does not contain."""
    return any(match.group(0) not in gold for match in _REPEAT_RUN_RE.finditer(pred))


def line_error_breakdown(pred_lines: list[tuple], gold_lines: list[tuple], threshold: float = 0.5) -> dict:
    """
    Where the matched-line CER edits come from. Lines are (box, text); matching is
    match_lines at the same threshold. The categories overlap (a line can be both
    too long and repeating), so they do not add up to the total.

    - recognition: matched pairs only, edits / gold characters of the matched lines
    - length_mismatch: matched pairs whose prediction is more than 15% longer or shorter than gold
    - repeated_run: predicted lines with a 5+ identical-character run that their gold line
      does not contain (an extra line is compared with empty gold)
    - extra: unmatched predictions that touch a gold box vs touch none
    - missed: unmatched gold lines that a prediction partly covers vs that no prediction touches
    """
    from Levenshtein import distance

    pred_boxes = [box for box, _ in pred_lines]
    gold_boxes = [box for box, _ in gold_lines]
    pairs = match_lines(pred_boxes, gold_boxes, threshold)
    matched_pred = {i for i, _ in pairs}
    matched_gold = {j for _, j in pairs}

    def bucket() -> dict:
        return {"lines": 0, "edits": 0}

    result = {
        "recognition": {"lines": 0, "edits": 0, "gold_chars": 0},
        "length_mismatch": {"longer": bucket(), "shorter": bucket()},
        "repeated_run": bucket(),
        "extra": {"overlaps_gold": bucket(), "no_gold_overlap": bucket()},
        "missed": {"partly_covered": bucket(), "not_detected": bucket()},
    }

    def add(target: dict, edits: int) -> None:
        target["lines"] += 1
        target["edits"] += edits

    for i, j in pairs:
        pred, gold = pred_lines[i][1], gold_lines[j][1]
        edits = distance(gold, pred)
        add(result["recognition"], edits)
        result["recognition"]["gold_chars"] += len(gold)
        if len(pred) > (1 + LENGTH_MISMATCH_FRAC) * len(gold):
            add(result["length_mismatch"]["longer"], edits)
        elif len(pred) < (1 - LENGTH_MISMATCH_FRAC) * len(gold):
            add(result["length_mismatch"]["shorter"], edits)
        if _new_repeat_run(pred, gold):
            add(result["repeated_run"], edits)

    for i, (box, text) in enumerate(pred_lines):
        if i in matched_pred:
            continue
        touches = any(_intersects(box, gold) for gold in gold_boxes)
        add(result["extra"]["overlaps_gold" if touches else "no_gold_overlap"], len(text))
        if _new_repeat_run(text, ""):
            add(result["repeated_run"], len(text))

    for j, (box, text) in enumerate(gold_lines):
        if j in matched_gold:
            continue
        covered = any(_intersects(box, pred) for pred in pred_boxes)
        add(result["missed"]["partly_covered" if covered else "not_detected"], len(text))

    result["recognition"]["cer"] = _ratio(result["recognition"]["edits"], result["recognition"]["gold_chars"])
    return result


def sum_error_breakdowns(breakdowns: list[dict]) -> dict:
    """Add up per-page line_error_breakdown results and recompute the recognition CER."""
    def add(total: dict, part: dict) -> None:
        for key, value in part.items():
            if isinstance(value, dict):
                add(total.setdefault(key, {}), value)
            elif key != "cer":
                total[key] = total.get(key, 0) + value

    total: dict = {}
    for breakdown in breakdowns:
        add(total, breakdown)
    if "recognition" in total:
        total["recognition"]["cer"] = _ratio(total["recognition"]["edits"], total["recognition"]["gold_chars"])
    return total


def page_cer(pred_text: str, gold_text: str) -> dict:
    """CER of the whole page text. Sensitive to reading order as well as recognition."""
    from Levenshtein import distance

    edits = distance(gold_text, pred_text)
    return {"edits": edits, "gold_chars": len(gold_text), "cer": _ratio(edits, len(gold_text))}


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
