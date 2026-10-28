# OCR training pipeline (RO1, RO2)

Dataset building, line detection and TrOCR fine-tuning for Majalah Qalam (1950_11, 1950_12).
These scripts were run in Google Colab from `MyDrive/jawi_dataset/`, so they use flat imports
(`import yolo_prefill`, `import generate_tasks`). Keep them in one folder.

## Files

| File | Purpose |
|---|---|
| `prepare_jawi_dataset_v2.py` | Discover articles, render PDF pages at 400 DPI |
| `ground_truth.py`, `ground_truth_gsheets.py`, `sheet_discovery_fix.py` | Load corrected Jawi lines from the Rumi/Jawi sheets |
| `segment_lines.py` | Projection-based line segmentation (first pre-fill, superseded by YOLO) |
| `generate_tasks.py` | Build Label Studio tasks (compressed page preview + boxes + text) |
| `box_review.py`, `convert_export.py` | Review boxes, convert Label Studio exports |
| `yolo_prefill.py` | YOLOv8 line detector: export to YOLO format, train, predict, XY-cut reading order, pre-fill tasks |
| `trocr_dataset.py` | Build the TrOCR line dataset from the Label Studio export (label normalisation, article-level split, crop jitter) |
| `notebooks/yolo_prefill_colab.ipynb` | Detector training and 1951_01 pre-fill, as run |
| `notebooks/trocr_v1_colab.ipynb` | TrOCR v1 training and evaluation |
| `splits/test_articles.json` | Locked test articles, never used to train the detector or TrOCR |
| `results/trocr_v1/` | Metrics only (no weights, no predictions) |

## TrOCR v1 results (test set, 641 lines, 6 held-out articles)

| System | Line CER |
|---|---|
| Tesseract (`ara`, `--psm 7`) | 64.31% |
| TrOCR v1 (`microsoft/trocr-base-stage1`, greedy decoding) | **3.78%** |

Training data: 3,173 train / 366 val / 641 test lines from 77 annotated pages.
Validation articles: 1950_11_015, 1950_12_005, 1950_12_008 (seed 42, 12%).

## Not in the repo (on Drive only)

Model weights (`best/`, `*.pt`), Label Studio exports, page images, and the prediction CSVs.
The prediction CSVs contain Qalam text, and this repository is public.
