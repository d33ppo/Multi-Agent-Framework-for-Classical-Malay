"""
Fast visual review helper: renders a page with its auto-detected boxes
overlaid and numbered, and flags boxes that are statistical outliers in
height (likely merged multiple lines) so you know exactly where to look
first, instead of scanning every box by eye with equal attention.

This does NOT try to fix anything automatically -- it only helps you see
the problem fast. Actual correction still happens by editing
annotations.csv directly (coordinates), which is more reliable than the
BBoxWidget for this corpus given the state we found it in.
"""
import cv2
import numpy as np
import csv
from pathlib import Path


def load_page_boxes(annotations_path, article_id, page_index):
    """Returns list of (line_index, x, y, w, h) sorted by line_index."""
    boxes = []
    with open(annotations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["article_id"] == article_id and int(row["page_index"]) == page_index:
                boxes.append((
                    int(row["line_index"]), int(row["x"]), int(row["y"]),
                    int(row["w"]), int(row["h"]),
                ))
    return sorted(boxes, key=lambda b: b[0])


def flag_outlier_boxes(boxes, height_factor=1.6, page_image=None, empty_ink_threshold=0.01):
    """
    Returns set of line_index values that are likely wrong, for two
    distinct reasons found in real testing on this corpus:

    1. Unusually tall boxes (height_factor times the page's median line
       height) -- a signal of a merged multi-line box.
    2. Boxes with almost no ink inside them (if page_image is given) --
       a signal of a phantom box over blank whitespace near a page edge,
       which was found to be the actual dominant false-positive type in
       practice, not merged lines, on pages tested so far. Both checks
       are kept since either failure mode can occur.
    """
    if len(boxes) < 3:
        return set()

    flagged = set()

    heights = sorted(b[4] for b in boxes)
    median_h = heights[len(heights) // 2]
    if median_h > 0:
        flagged |= {b[0] for b in boxes if b[4] > median_h * height_factor}

    if page_image is not None:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if page_image.ndim == 3 else page_image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for (line_index, x, y, w, h) in boxes:
            if w <= 0 or h <= 0:
                flagged.add(line_index)
                continue
            region = binary[y:y + h, x:x + w]
            ink_ratio = np.count_nonzero(region) / (w * h)
            if ink_ratio < empty_ink_threshold:
                flagged.add(line_index)

    return flagged


def render_review_image(page_image_path, annotations_path, article_id, page_index,
                         out_path=None):
    """
    Draws all boxes for one page, numbered in reading order, with likely-
    merged boxes highlighted in a different color so they stand out
    immediately. Also prints a summary: total boxes, flagged count, and
    the median/max height for quick sanity-checking.

    Returns (out_path, summary_dict).
    """
    img = cv2.imread(str(page_image_path))
    if img is None:
        raise FileNotFoundError(page_image_path)

    boxes = load_page_boxes(annotations_path, article_id, page_index)
    flagged = flag_outlier_boxes(boxes, page_image=img)

    for (line_index, x, y, w, h) in boxes:
        color = (0, 0, 255) if line_index in flagged else (0, 200, 0)
        thickness = 5 if line_index in flagged else 3
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        label_pos = (max(0, x - 10), max(20, y + 30))
        cv2.putText(img, str(line_index), label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA)

    if out_path is None:
        out_path = Path(page_image_path).with_name(
            f"{Path(page_image_path).stem}_review.png"
        )
    cv2.imwrite(str(out_path), img)

    heights = [b[4] for b in boxes]
    summary = {
        "total_boxes": len(boxes),
        "flagged_likely_merged": sorted(flagged),
        "median_height": int(np.median(heights)) if heights else 0,
        "max_height": max(heights) if heights else 0,
    }
    return str(out_path), summary


def review_article(base_out_dir, article_id):
    """
    Convenience: runs render_review_image for every page of an article
    and prints a one-line summary per page. Returns list of image paths
    for display in a notebook (e.g. via IPython.display.Image).
    """
    base_out_dir = Path(base_out_dir)
    article_dir = base_out_dir / article_id
    pages_dir = article_dir / "pages"
    ann_path = article_dir / "annotations.csv"

    if not ann_path.exists():
        print(f"No annotations.csv found for {article_id}. Run autoseg first.")
        return []

    review_dir = article_dir / "review"
    review_dir.mkdir(exist_ok=True)

    out_paths = []
    page_paths = sorted(pages_dir.glob("page_*.png"))
    for page_path in page_paths:
        page_index = int(page_path.stem.split("_")[1])
        out_path = review_dir / f"page_{page_index:02d}_review.png"
        _, summary = render_review_image(page_path, ann_path, article_id, page_index, out_path)
        flag_str = f"FLAGGED: {summary['flagged_likely_merged']}" if summary['flagged_likely_merged'] else "no flags"
        print(
            f"{article_id} page {page_index}: {summary['total_boxes']} boxes, "
            f"median height {summary['median_height']}px, max {summary['max_height']}px -- {flag_str}"
        )
        out_paths.append(str(out_path))

    return out_paths
