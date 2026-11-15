"""
Train the YOLO line detector locally on a CUDA GPU (detector v2).

Builds the YOLO dataset from a Label Studio export with yolo_prefill.export_to_yolo
(the locked test articles are left out, task 1 is mapped to 1950_11_002 page 0),
trains with yolo_prefill.train, then validates best.pt on the validation articles.

Outputs:
  models/yolo/qalam_lines_v2.pt              best.pt (gitignored)
  ocr_training/results/yolo_v2/metrics.json  settings, dataset summary, validation metrics

Usage:
    python ocr_training/train_yolo.py --export exports/label_studio_export.json
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

import yolo_prefill as yp  # noqa: E402

TASK_OVERRIDES = {1: ("1950_11_002", 0)}  # task 1 has no metadata in the export
DEFAULT_TEST_ARTICLES = TRAINING_DIR / "splits" / "test_articles.json"
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "yolo_ds_v2"
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs" / "yolo_v2"
DEFAULT_WEIGHTS = PROJECT_DIR / "models" / "yolo" / "qalam_lines_v2.pt"
DEFAULT_RESULTS_DIR = TRAINING_DIR / "results" / "yolo_v2"
RUN_NAME = "qalam_lines_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="Label Studio JSON export")
    parser.add_argument("--test-articles", default=str(DEFAULT_TEST_ARTICLES), help="JSON list of locked test articles")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device index (default 0)")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="YOLO dataset folder (rebuilt each run)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="ultralytics run folder")
    parser.add_argument("--weights-out", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def epochs_run(results_csv: Path) -> int | None:
    """Number of epochs ultralytics logged (fewer than --epochs when patience stopped it)."""
    if not results_csv.exists():
        return None
    with open(results_csv, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def validation_metrics(results) -> dict:
    """Plain floats from an ultralytics validation result, with short keys."""
    names = {"metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
             "metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95"}
    return {names[k]: round(float(v), 6) for k, v in results.results_dict.items() if k in names}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script trains on a CUDA GPU only.")
    import ultralytics
    from ultralytics import YOLO

    test_articles = json.loads(Path(args.test_articles).read_text(encoding="utf-8"))
    summary = yp.export_to_yolo(args.export, args.dataset_dir, exclude_articles=test_articles, overrides=TASK_OVERRIDES)
    print(json.dumps({k: summary[k] for k in ("train_pages", "train_boxes", "val_pages", "val_boxes", "val_articles")}))
    if summary["incomplete_pages"]:
        print(f"Skipped {len(summary['incomplete_pages'])} page(s) with boxes but no text: {summary['incomplete_pages']}")

    data_yaml = Path(args.dataset_dir) / "data.yaml"
    best = Path(yp.train(data_yaml, project=Path(args.runs_dir).resolve(), name=RUN_NAME, model=args.model,
                         epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                         device=args.device))
    weights_out = Path(args.weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights_out)
    print("Saved", weights_out)

    val = YOLO(str(weights_out)).val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
                                     project=str(Path(args.runs_dir).resolve()), name=f"{RUN_NAME}_val", exist_ok=True)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": weights_out.name,
        "settings": {"model": args.model, "imgsz": args.imgsz, "epochs": args.epochs, "patience": args.patience,
                     "batch": args.batch, "device": args.device},
        "epochs_run": epochs_run(best.parent.parent / "results.csv"),
        "validation": validation_metrics(val),   # best.pt on the validation articles, not the test set
        "dataset": summary,
        "export": Path(args.export).name,
        "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(int(str(args.device).split(",")[0]))},
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("Wrote", out_json)
    print(json.dumps(metrics["validation"]))
    return 0


if __name__ == "__main__":   # the guard is required: ultralytics starts dataloader workers on Windows
    raise SystemExit(main())
