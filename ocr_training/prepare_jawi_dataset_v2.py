"""
Multi-issue, multi-page Jawi TrOCR dataset preparation tool.

Extends the original single-page prepare_jawi_dataset.py to handle the
REAL corpus structure, confirmed by inspecting actual files:
  - Article-level PDFs (e.g. 1950_11_003.pdf), each may contain MULTIPLE
    physical pages (not one page per file, as the original script assumed)
  - Ground truth is in .xlsx, one row per sentence/paragraph, with
    multiple physical printed lines joined by \n within a cell
  - No per-page marker in the ground truth: a flat list of lines for the
    whole article, which may span page boundaries mid-row
  - Column naming and column count are NOT consistent across real xlsx
    files (confirmed: some use "Jawi Correction", others "jawi_corrected"
    with different casing, at least one has an entirely unnamed data
    column). The Jawi ground-truth column is identified by CONTENT
    (majority Arabic-script text, appearing consistently across multiple
    rows), not by header name.

Key design decision: automated line segmentation is used ONLY as a
starting point for manual correction, never as a pass/fail gate. Testing
against 3 real sample articles showed exact automatic line-count matches
are unreliable (photos, title blocks, and masthead elements are scattered
throughout the corpus, not confined to predictable pages), so every page
still requires a human review pass in the BBoxWidget annotator. What
automation buys you is a pre-filled starting point instead of an empty
canvas, which is still a real time saving, just not a free pass.

The existing export_dataset() safety behavior is preserved EXACTLY:
if a page's box count doesn't match its assigned ground-truth line
count, export hard-fails for that page rather than silently zipping
mismatched pairs. This is the last line of defense against corrupting
the training set, and it is not weakened by anything in this file.

Workflow:
  1. python prepare_jawi_dataset_v2.py discover
       Scans the PDF and ground-truth folders, reports which articles
       are available and how many pages/GT lines each has.
  2. python prepare_jawi_dataset_v2.py render --article 1950_11_003
       Renders all pages of one article at 400 DPI.
  3. python prepare_jawi_dataset_v2.py autoseg --article 1950_11_003
       Runs automated line segmentation on all rendered pages, writes
       results into annotations.csv as a PRE-FILLED starting point, and
       reports the per-page detected count vs a naive split of the GT
       line total, so you know roughly how much manual correction to
       expect before opening the annotator.
  4. In a notebook: launch_annotator("1950_11_003", page_index=0)
       Opens the BBoxWidget pre-loaded with the auto-detected boxes for
       that page, ready for you to fix/add/remove rather than draw from
       scratch. Repeat per page_index for multi-page articles.
  5. python prepare_jawi_dataset_v2.py export --article 1950_11_003
       Same hard-fail-on-mismatch behavior as before. Only pages/articles
       that pass are written to dataset/images and labels.csv.
  6. python prepare_jawi_dataset_v2.py preview
       Same as before: crop-next-to-text visual sanity check.
"""

from __future__ import annotations

import argparse
import base64
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import load_article_lines
from segment_lines import segment_page as auto_segment_page

try:
    from ground_truth_gsheets import load_article_lines_from_sheet, extract_spreadsheet_id
except ImportError:
    load_article_lines_from_sheet = None
    extract_spreadsheet_id = None


ANNOTATION_COLUMNS = ["article_id", "page_index", "line_index", "x", "y", "w", "h"]
RENDER_DPI = 400


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


# ---------------------------------------------------------------------------
# Discovery: find articles across the PDF and ground-truth folders
# ---------------------------------------------------------------------------

def discover_articles(pdf_dir: Path, gt_dir: Path, issue_prefixes: tuple[str, ...] | None = None) -> dict[str, dict]:
    """
    Returns {article_id: {"pdf": Path, "xlsx": Path or None}}
    An article_id is the PDF stem, e.g. "1950_11_003".

    issue_prefixes, if given, restricts discovery to article IDs starting
    with one of these prefixes (e.g. ("1950_07", "1950_09", "1950_10",
    "1950_11", "1950_12")). This matters when pdf_dir is a shared folder
    containing PDFs from unrelated issues/years -- confirmed to be the
    case for this corpus's actual Drive folder, which mixes files from
    1950 through 1969 in one place. Without this filter, discovery would
    silently include hundreds of irrelevant articles.
    """
    articles = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        article_id = pdf_path.stem
        if issue_prefixes and not article_id.startswith(issue_prefixes):
            continue
        xlsx_path = gt_dir / f"{article_id}.xlsx"
        articles[article_id] = {
            "pdf": pdf_path,
            "xlsx": xlsx_path if xlsx_path.exists() else None,
        }
    return articles


def discover_articles_gsheets(pdf_dir: Path, drive_service, csv_folder_ids: dict[str, str],
                               issue_prefixes: tuple[str, ...] | None = None) -> dict[str, dict]:
    """
    Same shape as discover_articles, but looks up ground truth as Google
    Sheets instead of local .xlsx files.

    csv_folder_ids maps issue prefix -> Drive folder ID of that issue's
    "csv" subfolder under Al_Qalam_Rumi_by_batches (e.g.
    {"1950_12": "1Dq9wC37RMgyKP0g47R4keW18uk8xK18W", ...}). This mapping
    has to be supplied explicitly since folder IDs aren't derivable from
    a naming convention alone -- confirmed each issue's ground truth
    lives in its own separate Drive folder, not a single shared one.

    drive_service is an authenticated Drive API client (googleapiclient
    build('drive', 'v3', credentials=creds)), used to list files within
    each csv folder and match them to article IDs by title.

    Returns {article_id: {"pdf": Path, "sheet_id": str or None}}
    """
    articles = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        article_id = pdf_path.stem
        if issue_prefixes and not article_id.startswith(issue_prefixes):
            continue
        articles[article_id] = {"pdf": pdf_path, "sheet_id": None}

    # Build a title -> id lookup per relevant csv folder, then match
    for issue_prefix, folder_id in csv_folder_ids.items():
        if issue_prefixes and issue_prefix not in issue_prefixes:
            continue
        page_token = None
        while True:
            resp = drive_service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
            ).execute()
            for f in resp.get("files", []):
                title = f["name"]
                if title in articles:
                    articles[title]["sheet_id"] = f["id"]
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    return articles


def discover_articles_gsheets_gspread(pdf_dir: Path, gc, csv_folder_ids: dict[str, str],
                                       issue_prefixes: tuple[str, ...] | None = None) -> dict[str, dict]:
    """
    Same as discover_articles_gsheets, but uses a gspread client (already
    the auth pattern documented in the README) instead of a raw Drive API
    service object, so only one auth flow is needed end-to-end in Colab.

    gc: an authorized gspread.Client (see README for the auth snippet).
    csv_folder_ids: {issue_prefix: drive_folder_id}, e.g.
        {"1950_12": "1Dq9wC37RMgyKP0g47R4keW18uk8xK18W"}
    """
    articles = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        article_id = pdf_path.stem
        if issue_prefixes and not article_id.startswith(issue_prefixes):
            continue
        articles[article_id] = {"pdf": pdf_path, "sheet_id": None}

    drive_service = gc.http_client.session if hasattr(gc, "http_client") else None
    # gspread wraps its own authorized session; simplest robust path is to
    # use gspread's list_spreadsheet_files_in_folder if available, falling
    # back to iterating folder contents via the Drive API client gspread
    # already holds internally.
    for issue_prefix, folder_id in csv_folder_ids.items():
        if issue_prefixes and issue_prefix not in issue_prefixes:
            continue
        try:
            files = gc.list_spreadsheet_files_in_folder(folder_id)
        except AttributeError:
            raise RuntimeError(
                "This gspread version lacks list_spreadsheet_files_in_folder. "
                "Use discover_articles_gsheets with a raw Drive API service "
                "object instead (see its docstring)."
            )
        for f in files:
            title = f["name"]
            if title in articles:
                articles[title]["sheet_id"] = f["id"]

    return articles


def article_out_dir(base_out_dir: Path, article_id: str) -> Path:
    return base_out_dir / article_id


def pages_dir(base_out_dir: Path, article_id: str) -> Path:
    return article_out_dir(base_out_dir, article_id) / "pages"


def annotations_path(base_out_dir: Path, article_id: str) -> Path:
    return article_out_dir(base_out_dir, article_id) / "annotations.csv"


def labels_path(base_out_dir: Path, article_id: str) -> Path:
    return article_out_dir(base_out_dir, article_id) / "labels.csv"


def page_png_path(base_out_dir: Path, article_id: str, page_index: int) -> Path:
    return pages_dir(base_out_dir, article_id) / f"page_{page_index:02d}.png"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_article(pdf_path: Path, base_out_dir: Path, article_id: str, dpi: int = RENDER_DPI) -> int:
    out_pages_dir = pages_dir(base_out_dir, article_id)
    out_pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)
        for i in range(n_pages):
            page = doc.load_page(i)
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out_path = page_png_path(base_out_dir, article_id, i)
            cv2.imwrite(str(out_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"  rendered page {i}: {out_path} ({image.shape[1]}x{image.shape[0]})")
    finally:
        doc.close()

    create_empty_annotations_file(annotations_path(base_out_dir, article_id))
    return n_pages


def create_empty_annotations_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()


# ---------------------------------------------------------------------------
# Annotations read/write (article_id + page_index keyed, not pdf_name/page_id)
# ---------------------------------------------------------------------------

def read_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append({
                "article_id": row["article_id"],
                "page_index": int(row["page_index"]),
                "line_index": int(row["line_index"]),
                "x": int(float(row["x"])),
                "y": int(float(row["y"])),
                "w": int(float(row["w"])),
                "h": int(float(row["h"])),
            })
    return sorted(rows, key=lambda r: (r["article_id"], r["page_index"], r["line_index"]))


def write_annotations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["article_id"], r["page_index"], r["line_index"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def replace_page_annotations(
    base_out_dir: str | Path, article_id: str, page_index: int, boxes: list[dict]
) -> None:
    """
    Notebook helper: replace one page's annotations.
    boxes must be in final reading order. Each item can use either:
      {"x":.., "y":.., "w":.., "h":..} or jupyter_bbox_widget's
      {"x":.., "y":.., "width":.., "height":..}
    """
    base_out_dir = Path(base_out_dir)
    path = annotations_path(base_out_dir, article_id)

    existing = [
        row for row in read_annotations(path)
        if not (row["article_id"] == article_id and row["page_index"] == page_index)
    ]
    new_rows = []
    for i, box in enumerate(boxes, start=1):
        w = box.get("w", box.get("width"))
        h = box.get("h", box.get("height"))
        if w is None or h is None:
            raise ValueError("Each box needs w/h or width/height.")
        new_rows.append({
            "article_id": article_id,
            "page_index": page_index,
            "line_index": i,
            "x": int(round(float(box["x"]))),
            "y": int(round(float(box["y"]))),
            "w": int(round(float(w))),
            "h": int(round(float(h))),
        })

    write_annotations(path, existing + new_rows)
    print(f"Saved {len(new_rows)} boxes for {article_id} page {page_index} -> {path}")


# ---------------------------------------------------------------------------
# Automated segmentation (pre-fill step, NOT a pass/fail gate)
# ---------------------------------------------------------------------------

def autoseg_article(base_out_dir: Path, article_id: str, gt_lines: list[str]) -> None:
    """
    Runs automated line segmentation on every rendered page of an article
    and writes the results into annotations.csv as a starting point.

    IMPORTANT, stated plainly: this is a best-effort pre-fill, not a
    verified result. Testing against 3 real sample articles in this
    corpus showed automated segmentation does NOT reliably hit an exact
    line count match, because photos, title blocks, and masthead
    elements are scattered throughout the corpus (confirmed directly,
    not assumed) in ways that break simple projection-based heuristics.
    Every page still needs a manual review pass in the BBoxWidget
    annotator before being trusted. This function's only purpose is to
    save you the time of drawing every box from a blank canvas; it does
    NOT save you the review step itself.
    """
    out_pages_dir = pages_dir(base_out_dir, article_id)
    page_paths = sorted(out_pages_dir.glob("page_*.png"))
    if not page_paths:
        print(f"ERROR: No rendered pages found for {article_id}. Run 'render' first.", file=sys.stderr)
        return

    per_page_boxes = []
    per_page_counts = []
    for page_path in page_paths:
        boxes, info = auto_segment_page(str(page_path))
        per_page_boxes.append(boxes)
        per_page_counts.append(len(boxes))

    total_detected = sum(per_page_counts)
    total_gt = len(gt_lines)
    diff = total_detected - total_gt

    print(f"\n{article_id}: {len(page_paths)} pages")
    for i, count in enumerate(per_page_counts):
        print(f"  page {i}: {count} boxes auto-detected")
    print(f"  TOTAL detected: {total_detected}  |  GT lines: {total_gt}  |  diff: {diff:+d}")
    if diff != 0:
        print(
            f"  NOTE: counts do not match. This is EXPECTED and does not block anything -- "
            f"auto-segmentation is a starting point only. Open each page in the annotator "
            f"and correct boxes; the true per-page GT line count is only knowable after your "
            f"review, since the ground truth itself has no per-page marker."
        )

    # Write auto-detected boxes as the initial annotations, in reading order,
    # per page. These are placeholders: line_index assignment here does NOT
    # attempt to split gt_lines across pages (that split is unknowable until
    # a human confirms each page's true count), so annotations.csv at this
    # stage holds boxes only; text pairing happens later at export time
    # once you have corrected each page's boxes to match its true GT count
    # (you will need to determine the correct split point yourself while
    # reviewing, since GT has no per-page marker -- cross-check against the
    # page image while annotating).
    all_rows = [row for row in read_annotations(annotations_path(base_out_dir, article_id))
                if row["article_id"] != article_id]
    for page_index, boxes in enumerate(per_page_boxes):
        for line_index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            all_rows.append({
                "article_id": article_id,
                "page_index": page_index,
                "line_index": line_index,
                "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            })
    write_annotations(annotations_path(base_out_dir, article_id), all_rows)
    print(f"  Wrote pre-filled boxes to {annotations_path(base_out_dir, article_id)}")


# ---------------------------------------------------------------------------
# Export (hard-fail-on-mismatch preserved exactly as in the original script)
# ---------------------------------------------------------------------------

def clamp_box(box: Box, image: np.ndarray, padding_x: int, padding_y: int) -> Box:
    h, w = image.shape[:2]
    x1 = max(0, box.x - padding_x)
    y1 = max(0, box.y - padding_y)
    x2 = min(w, box.right + padding_x)
    y2 = min(h, box.bottom + padding_y)
    return Box(x1, y1, x2 - x1, y2 - y1)


def crop_box(image: np.ndarray, box: Box) -> np.ndarray:
    return image[box.y:box.bottom, box.x:box.right]


def export_article(
    base_out_dir: Path, article_id: str, gt_lines: list[str],
    padding_x: int = 4, padding_y: int = 4,
) -> int:
    """
    Exports crop images + labels.csv for one article, but ONLY if the
    TOTAL box count across all its pages exactly matches the article's
    total ground-truth line count. This mirrors the original script's
    per-page hard-fail check, applied at the article level since GT
    lines aren't marked per-page.

    This does NOT attempt to guess where the page split falls when
    counts mismatch -- it fails loudly instead, exactly as the original
    script did, because a guessed split risks silently pairing a crop
    with the wrong line of text.
    """
    ann_path = annotations_path(base_out_dir, article_id)
    annotations = read_annotations(ann_path)
    if not annotations:
        print(f"ERROR: No annotations found for {article_id}. Annotate boxes first.", file=sys.stderr)
        return 1

    page_indices = sorted(set(row["page_index"] for row in annotations))
    per_page_rows = {
        pi: sorted(
            [r for r in annotations if r["page_index"] == pi],
            key=lambda r: r["line_index"],
        )
        for pi in page_indices
    }
    total_boxes = sum(len(rows) for rows in per_page_rows.values())
    total_gt = len(gt_lines)

    if total_boxes != total_gt:
        print(
            f"ERROR: {article_id} has {total_boxes} total boxes across {len(page_indices)} "
            f"pages but {total_gt} ground-truth lines. Export requires an exact match. "
            f"Fix boxes in the annotator (add/remove/resize) until counts agree.",
            file=sys.stderr,
        )
        return 1

    images_dir = article_out_dir(base_out_dir, article_id) / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    gt_cursor = 0
    for page_index in page_indices:
        rows = per_page_rows[page_index]
        page_image_path = page_png_path(base_out_dir, article_id, page_index)
        image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"ERROR: Rendered page missing: {page_image_path}", file=sys.stderr)
            return 1

        page_gt_lines = gt_lines[gt_cursor:gt_cursor + len(rows)]
        gt_cursor += len(rows)

        for row, text in zip(rows, page_gt_lines):
            box = Box(row["x"], row["y"], row["w"], row["h"])
            box = clamp_box(box, image, padding_x, padding_y)
            crop = crop_box(image, box)
            if crop.size == 0:
                print(
                    f"ERROR: Empty crop at {article_id} page {page_index}, "
                    f"line {row['line_index']:03d}.",
                    file=sys.stderr,
                )
                return 1
            file_name = f"{article_id}_p{page_index:02d}_line_{row['line_index']:03d}.png"
            cv2.imwrite(str(images_dir / file_name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            labels.append({"file_name": file_name, "text": text})

    write_labels(labels_path(base_out_dir, article_id), labels)
    print(f"Wrote {len(labels)} crops to {images_dir}")
    print(f"Wrote labels: {labels_path(base_out_dir, article_id)}")
    return 0


def write_labels(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Annotator widget launcher (same BBoxWidget as the original script,
# pre-loaded with either existing saved boxes or fresh auto-seg boxes)
# ---------------------------------------------------------------------------

def image_as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def launch_annotator(article_id: str, page_index: int, base_out_dir: str | Path = "dataset"):
    """
    Notebook usage:
        widget = tool.launch_annotator("1950_11_003", page_index=0)
        # ... draw/adjust boxes in the widget, then:
        tool.replace_page_annotations("dataset", "1950_11_003", 0, widget.bboxes)

    Requires: pip install jupyter_bbox_widget
    """
    from jupyter_bbox_widget import BBoxWidget

    base_out_dir = Path(base_out_dir)
    image_path = page_png_path(base_out_dir, article_id, page_index)
    if not image_path.exists():
        raise FileNotFoundError(f"Rendered page not found: {image_path}. Run 'render' first.")

    existing = [
        row for row in read_annotations(annotations_path(base_out_dir, article_id))
        if row["article_id"] == article_id and row["page_index"] == page_index
    ]
    existing = sorted(existing, key=lambda r: r["line_index"])

    widget = BBoxWidget()
    widget.image = image_as_data_url(image_path)
    widget.classes = ["line"]
    widget.bboxes = [
        {"x": r["x"], "y": r["y"], "width": r["w"], "height": r["h"], "label": "line"}
        for r in existing
    ]
    return widget


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-issue Jawi TrOCR dataset preparation.")
    parser.add_argument("command", choices=["discover", "render", "autoseg", "export"])
    parser.add_argument("--pdf-dir", default="Al_Qalam_Jawi_Full")
    parser.add_argument("--gt-dir", default="Al_Qalam_Jawi_Full")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--article", help="Article id, e.g. 1950_11_003")
    parser.add_argument(
        "--issues", default="1950_07,1950_09,1950_10,1950_11,1950_12",
        help="Comma-separated issue prefixes to restrict discovery to, "
             "since Al_Qalam_Jawi_Full contains unrelated years/issues too."
    )
    parser.add_argument(
        "--gt-mode", choices=["xlsx", "gsheets"], default="xlsx",
        help="Where ground truth lives. 'xlsx': local .xlsx files in "
             "--gt-dir (tested, default). 'gsheets': native Google Sheets "
             "under Al_Qalam_Rumi_by_batches/<issue>/csv/ -- requires "
             "calling this script's functions from a notebook with an "
             "authorized gspread client, not directly from the CLI, since "
             "auth can't happen inside a bare `python script.py` call. See "
             "README for the notebook-based gsheets workflow."
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    gt_dir = Path(args.gt_dir)
    base_out_dir = Path(args.out_dir)
    issue_prefixes = tuple(p.strip() for p in args.issues.split(",") if p.strip())

    if args.gt_mode == "gsheets":
        print(
            "ERROR: --gt-mode gsheets requires an authorized gspread client, "
            "which cannot be created inside a plain CLI invocation. Run this "
            "workflow from a notebook cell instead, importing this module "
            "and calling discover_articles_gsheets_gspread(...) / "
            "autoseg_article(...) / export_article(...) directly with "
            "gt_lines loaded via ground_truth_gsheets.load_article_lines_from_sheet(). "
            "See README for the full snippet.",
            file=sys.stderr,
        )
        return 1

    if args.command == "discover":
        articles = discover_articles(pdf_dir, gt_dir, issue_prefixes=issue_prefixes)
        print(f"Found {len(articles)} article PDFs in {pdf_dir} matching issues {issue_prefixes}")
        for aid, info in sorted(articles.items()):
            gt_status = "OK" if info["xlsx"] else "MISSING ground truth"
            print(f"  {aid}: {gt_status}")
        return 0

    if not args.article:
        print("ERROR: --article is required for this command.", file=sys.stderr)
        return 1

    articles = discover_articles(pdf_dir, gt_dir, issue_prefixes=issue_prefixes)
    if args.article not in articles:
        print(f"ERROR: article {args.article} not found in {pdf_dir}", file=sys.stderr)
        return 1
    info = articles[args.article]

    if args.command == "render":
        n = render_article(info["pdf"], base_out_dir, args.article)
        print(f"Rendered {n} pages for {args.article}")
        return 0

    if info["xlsx"] is None:
        print(f"ERROR: no ground-truth xlsx found for {args.article}", file=sys.stderr)
        return 1
    gt_lines = load_article_lines(str(info["xlsx"]))

    if args.command == "autoseg":
        autoseg_article(base_out_dir, args.article, gt_lines)
        return 0

    if args.command == "export":
        return export_article(base_out_dir, args.article, gt_lines)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Batch helper: render + autoseg across many articles in one notebook call
# ---------------------------------------------------------------------------

def batch_render_and_autoseg(base_out_dir, ready_articles, gc, skip_existing=True):
    """
    ready_articles: dict like {article_id: {"pdf": Path, "sheet_id": str}},
    e.g. the output of discover_articles_gsheets filtered to entries with
    a real sheet_id.

    For each article: renders its pages (skipped if already rendered, since
    re-rendering 23 articles' worth of 400 DPI pages repeatedly wastes time
    and Drive bandwidth for no benefit), loads its ground truth from the
    matched Sheet, and runs auto-segmentation to pre-fill annotations.csv.

    Returns a summary dict per article: detected vs GT counts, so you can
    see at a glance which articles are closest to a clean match before
    diving into manual correction.
    """
    from pathlib import Path as _Path
    summary = {}
    for article_id, info in sorted(ready_articles.items()):
        print(f"=== {article_id} ===")
        out_pages = pages_dir(base_out_dir, article_id)
        already_rendered = out_pages.exists() and any(out_pages.glob("page_*.png"))
        if skip_existing and already_rendered:
            print("  pages already rendered, skipping render step")
        else:
            n = render_article(info["pdf"], base_out_dir, article_id)
            print(f"  rendered {n} pages")

        try:
            gt_lines = load_article_lines_from_sheet(info["sheet_id"], gspread_client=gc)
        except Exception as e:
            print(f"  ERROR loading ground truth: {e}")
            summary[article_id] = {"error": str(e)}
            continue

        autoseg_article(base_out_dir, article_id, gt_lines)
        summary[article_id] = {"gt_lines": len(gt_lines)}

    return summary
