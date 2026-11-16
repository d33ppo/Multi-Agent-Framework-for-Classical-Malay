"""
Render review sheets for the gold reading order of the test pages.

The page-level evaluation (scripts/eval_page_cer.py) takes the Label Studio region
order as the gold reading order. That order is wrong on some pages: boxes added
late during annotation are appended at the end. This script draws every test page,
downscaled, with the gold boxes numbered in their current order, so the true
order can be written down by hand.

The numbers are the 0-based gold line indices of eval_page_cer.gold_lines (lines
with empty text or a box under 3 px are dropped first, exactly as in the eval).

Outputs in --out-dir (default ocr_training/review/, gitignored: page images):
  <article>_p<NN>.jpg                 the page with numbered boxes
  test_reading_order.template.json    {"<article>_p<NN>": [0, 1, 2, ...]} in the current order

To fix the order: copy the template to ocr_training/splits/test_reading_order.json,
reorder the indices of the pages that are wrong (pages left out keep the export
order) and re-run eval_page_cer.py. With --reading-order the sheets also show the
reviewed order as a line through the box centres, to check it.

Usage:
    python scripts/make_order_sheets.py --export exports/label_studio_export.json --pages-base data/dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "ocr_training" / "review"


def _load_eval_script():
    spec = importlib.util.spec_from_file_location("eval_page_cer", Path(__file__).resolve().parent / "eval_page_cer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_eval_script()


def draw_sheet(image, gold: list[tuple], max_side: int = 1800, path_order: list[int] | None = None):
    """
    Return the page scaled so its longer side is at most max_side, with each gold box
    outlined and numbered with its index. path_order draws a line through the box
    centres in that order (the reviewed reading order).
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    sheet = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else image.copy()
    boxes = [tuple(round(v * scale) for v in box) for box, _ in gold]
    if path_order:
        centres = [((boxes[j][0] + boxes[j][2]) // 2, (boxes[j][1] + boxes[j][3]) // 2) for j in path_order]
        for a, b in zip(centres, centres[1:]):
            cv2.arrowedLine(sheet, a, b, (0, 160, 0), 1, cv2.LINE_AA, tipLength=0.03)
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(sheet, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label = str(index)
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        # A filled tag at the right end of the box (lines start on the right in Jawi).
        tag_x = min(x2 + 2, sheet.shape[1] - text_w - 4)
        tag_y = max(y1, 0)
        cv2.rectangle(sheet, (tag_x, tag_y), (tag_x + text_w + 4, tag_y + text_h + 6), (255, 255, 255), -1)
        cv2.rectangle(sheet, (tag_x, tag_y), (tag_x + text_w + 4, tag_y + text_h + 6), (255, 0, 0), 1)
        cv2.putText(sheet, label, (tag_x + 2, tag_y + text_h + 3), font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
    return sheet


def make_sheets(export, pages_base, test_articles: list[str], out_dir, max_side: int = 1800,
                reading_orders: dict | None = None) -> dict:
    """Write one sheet per test page and the order template. Returns the template dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    template = {}
    for page in ev.load_test_pages(export, test_articles):
        key = ev.page_key(page["article_id"], page["page_index"])
        png_path = Path(pages_base) / page["article_id"] / "pages" / f"page_{page['page_index']:02d}.png"
        image = cv2.imread(str(png_path))
        if image is None:
            raise FileNotFoundError(png_path)
        gold = ev.gold_lines(page, image.shape[1], image.shape[0])
        reviewed = ev.gold_order_for(reading_orders or {}, key, len(gold))
        cv2.imwrite(str(out / f"{key}.jpg"), draw_sheet(image, gold, max_side, reviewed),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        template[key] = list(range(len(gold)))
    lines = [f' {json.dumps(key)}: {json.dumps(order)}' for key, order in template.items()]  # one page per line
    (out / "test_reading_order.template.json").write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return template


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Numbered gold-box sheets of the test pages, to review the reading order.")
    parser.add_argument("--export", required=True, help="Label Studio JSON export")
    parser.add_argument("--pages-base", required=True, help="folder with <article>/pages/page_NN.png")
    parser.add_argument("--test-articles", default=str(ev.DEFAULT_TEST_ARTICLES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-side", type=int, default=1800, help="longer side of the sheet in pixels")
    parser.add_argument("--reading-order", default=str(ev.DEFAULT_READING_ORDER),
                        help="reviewed orders to draw as a path, when the file exists")
    args = parser.parse_args(argv)

    test_articles = json.loads(Path(args.test_articles).read_text(encoding="utf-8"))
    template = make_sheets(args.export, args.pages_base, test_articles, args.out_dir, args.max_side,
                           ev.load_reading_order(args.reading_order))
    print(f"Wrote {len(template)} sheets and test_reading_order.template.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
