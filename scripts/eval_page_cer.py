"""
Page-level OCR evaluation of the full TrOCR provider on the locked test articles.

Runs the product pipeline (YOLO boxes -> reading order -> eval crops -> TrOCR)
on the full-resolution page PNGs and compares it with the Label Studio ground
truth. The line-level test CER (3.78% greedy) used hand-drawn boxes; this script
measures what the product actually gets, with detected boxes.

Ground truth: the Label Studio JSON export, read with yolo_prefill.load_pages
(latest annotation per page, task 1 mapped to 1950_11_002 page 0), filtered to
ocr_training/splits/test_articles.json and normalised with
trocr_dataset.normalise_text. Lines are dropped exactly as the dataset builder
drops them (empty text, box under 3 px), so the line count should match the
641 test lines.

Metrics, per page and overall:
  (a) matched-line CER (primary): predicted and gold boxes matched at IoU >= 0.5;
      matched pairs scored by edit distance, missed gold lines as full deletions,
      extra predicted lines as full insertions
  (b) page CER (secondary): page text joined with newlines; predicted lines in
      reading order, gold lines in the order of the export, so it is order sensitive
  (c) detector precision and recall at IoU >= 0.5
  (d) seconds per page (model loading reported separately)

Outputs in --out-dir: page_eval_<decoding>.json and .csv (metrics only, no
Qalam text) and predictions_<decoding>.jsonl (texts, gitignored).

Usage:
    python scripts/eval_page_cer.py --export exports/label_studio_export.json --pages-base data/dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
for folder in (PROJECT_DIR, PROJECT_DIR / "ocr_training"):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

import stage_metrics  # noqa: E402
import trocr_dataset as td  # noqa: E402
import yolo_prefill as yp  # noqa: E402

TASK_OVERRIDES = {1: ("1950_11_002", 0)}  # task 1 has no metadata in the export
DEFAULT_TEST_ARTICLES = PROJECT_DIR / "ocr_training" / "splits" / "test_articles.json"
DEFAULT_OUT_DIR = PROJECT_DIR / "ocr_training" / "results" / "page_eval_v1"
IOU_THRESHOLD = 0.5
MIN_BOX_PX = 3  # the dataset builder skips boxes smaller than this


def load_test_pages(export_path, test_articles, exclude_pages=td.DEFAULT_EXCLUDE_PAGES) -> list[dict]:
    """Ground-truth pages of the test articles, as returned by yolo_prefill.load_pages."""
    pages, _report = yp.load_pages(export_path, TASK_OVERRIDES)
    wanted = set(test_articles)
    pages = [p for p in pages if p["article_id"] in wanted and (p["article_id"], p["page_index"]) not in set(exclude_pages)]
    incomplete = [(p["article_id"], p["page_index"]) for p in pages if p["incomplete"]]
    if incomplete:
        raise ValueError(f"Test pages with boxes but no text, finish them first: {incomplete}")
    return pages


def gold_lines(page: dict, width: int, height: int) -> list[tuple]:
    """((x1, y1, x2, y2), text) in page pixels, in export order, filtered like the dataset builder."""
    lines = []
    for (x, y, bw, bh), text in zip(page["boxes"], page["texts"]):
        text = td.normalise_text(text)
        x1, y1 = x / 100 * width, y / 100 * height
        x2, y2 = x1 + bw / 100 * width, y1 + bh / 100 * height
        if not text or (y2 - y1) < MIN_BOX_PX or (x2 - x1) < MIN_BOX_PX:
            continue
        lines.append(((x1, y1, x2, y2), text))
    return lines


def evaluate_page(provider, png: bytes, gold: list[tuple]) -> tuple[dict, list]:
    started = time.perf_counter()
    detected = provider.read_page(png)
    seconds = time.perf_counter() - started
    pred = [((*line.xyxy, line.detector_confidence), line.text) for line in detected]
    line_metrics = stage_metrics.matched_line_cer(pred, gold, IOU_THRESHOLD)
    pred_text = "\n".join(line.text for line in detected if line.text)  # what the product passes on
    gold_text = "\n".join(text for _, text in gold)
    return {"seconds": round(seconds, 3), "line": line_metrics, "page": stage_metrics.page_cer(pred_text, gold_text)}, detected


def overall(rows: list[dict]) -> dict:
    total = {k: sum(r["line"][k] for r in rows) for k in
             ("edits", "matched_edits", "missed_chars", "extra_chars", "gold_chars",
              "gold_lines", "pred_lines", "matched_lines")}
    page_edits = sum(r["page"]["edits"] for r in rows)
    page_chars = sum(r["page"]["gold_chars"] for r in rows)
    seconds = [r["seconds"] for r in rows]
    return {
        "pages": len(rows),
        "matched_line_cer": stage_metrics._ratio(total["edits"], total["gold_chars"]),
        "matched_line_char_accuracy": None if not total["gold_chars"] else round(1 - total["edits"] / total["gold_chars"], 6),
        "page_cer": stage_metrics._ratio(page_edits, page_chars),
        "detector_precision": stage_metrics._ratio(total["matched_lines"], total["pred_lines"]),
        "detector_recall": stage_metrics._ratio(total["matched_lines"], total["gold_lines"]),
        "seconds_per_page_mean": round(statistics.mean(seconds), 3) if seconds else None,
        "seconds_per_page_median": round(statistics.median(seconds), 3) if seconds else None,
        **total,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Page-level CER of the TrOCR provider on the locked test articles.")
    parser.add_argument("--export", required=True, help="Label Studio JSON export")
    parser.add_argument("--pages-base", required=True, help="folder with <article>/pages/page_NN.png")
    parser.add_argument("--test-articles", default=str(DEFAULT_TEST_ARTICLES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", help="TrOCR folder (default: OCR_MODEL_PATH)")
    parser.add_argument("--detector", help="YOLO weights (default: LINE_DETECTOR_PATH)")
    parser.add_argument("--imgsz", type=int, help="detector image size (default: LINE_DETECTOR_IMGSZ or 960)")
    parser.add_argument("--beams", type=int, help="1 = greedy (default: OCR_NUM_BEAMS or 1)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int, help="torch CPU threads (default: OCR_TORCH_THREADS)")
    parser.add_argument("--device", default="cpu", help="cpu (default, for the timing) or cuda")
    parser.add_argument("--limit", type=int, help="evaluate only the first N pages, for a quick check")
    args = parser.parse_args()

    from ocr_trocr import TrOCROCRProvider, decode_page

    test_articles = json.loads(Path(args.test_articles).read_text(encoding="utf-8"))
    pages = load_test_pages(args.export, test_articles)[: args.limit]
    provider = TrOCROCRProvider(args.model, args.detector, imgsz=args.imgsz, num_beams=args.beams,
                                batch_size=args.batch_size, device=args.device, threads=args.threads)
    started = time.perf_counter()
    provider.load()
    load_seconds = round(time.perf_counter() - started, 2)
    decoding = "greedy" if provider.num_beams == 1 else f"beam{provider.num_beams}"
    print(f"Models loaded in {load_seconds}s. {len(pages)} test pages, decoding {decoding}.")

    rows, predictions = [], []
    for p in pages:
        png_path = Path(args.pages_base) / p["article_id"] / "pages" / f"page_{p['page_index']:02d}.png"
        png = png_path.read_bytes()
        height, width = decode_page(png).shape[:2]
        gold = gold_lines(p, width, height)
        result, detected = evaluate_page(provider, png, gold)
        rows.append({"article_id": p["article_id"], "page_index": p["page_index"], **result})
        predictions.append({
            "article_id": p["article_id"], "page_index": p["page_index"],
            "pred": [{"order": d.order, "xyxy": [round(v, 1) for v in d.xyxy], "text": d.text,
                      "confidence": d.confidence, "detector_confidence": round(d.detector_confidence, 4)}
                     for d in detected],
            "gold": [{"xyxy": [round(v, 1) for v in box], "text": text} for box, text in gold],
        })
        line = result["line"]
        print(f"{p['article_id']} p{p['page_index']:02d}: matched-line CER {line['cer']}, page CER "
              f"{result['page']['cer']}, P {line['precision']} R {line['recall']}, {result['seconds']}s")

    summary = overall(rows)
    if summary["gold_lines"] != 641:
        print(f"Note: {summary['gold_lines']} gold lines; the line-level test set has 641.")
    import torch
    import transformers

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {
            "model": provider.model_path.name, "detector": provider.detector_path.name,
            "detector_imgsz": provider.imgsz, "detector_conf": 0.35, "iou_threshold": IOU_THRESHOLD,
            "decoding": decoding, "num_beams": provider.num_beams, "batch_size": provider.batch_size,
            "device": args.device, "torch_threads": torch.get_num_threads(), "test_articles": test_articles,
            "torch": torch.__version__, "transformers": transformers.__version__, "cpu": platform.processor(),
        },
        "model_load_seconds": load_seconds,
        "overall": summary,
        "per_page": rows,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"page_eval_{decoding}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(out / f"page_eval_{decoding}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["article_id", "page_index", "gold_lines", "pred_lines", "matched_lines", "precision",
                         "recall", "matched_line_cer", "page_cer", "gold_chars", "seconds"])
        for r in rows:
            line = r["line"]
            writer.writerow([r["article_id"], r["page_index"], line["gold_lines"], line["pred_lines"],
                             line["matched_lines"], line["precision"], line["recall"], line["cer"],
                             r["page"]["cer"], line["gold_chars"], r["seconds"]])
    with open(out / f"predictions_{decoding}.jsonl", "w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=1))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
