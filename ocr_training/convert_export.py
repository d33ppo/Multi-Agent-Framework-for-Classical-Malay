"""
Converts a Label Studio export (JSON, downloaded after annotating) back
into cropped line images + labels.csv, matching this pipeline's existing
output shape.

Label Studio's exported JSON structure (per task) is documented as:
  {
    "data": {"image": ...},
    "annotations": [
      {
        "result": [
          {"id": "...", "type": "rectanglelabels", "value": {x,y,width,height,...}},
          {"id": "...", "type": "textarea", "value": {x,y,width,height,text:[...]}}
        ]
      }
    ]
  }

Box and text entries sharing the same "id" belong to the same region --
this mirrors exactly how they were generated in generate_tasks.py, so
pairing them back up is a straightforward id-match, not a guess.

IMPORTANT CAVEAT: I have not been able to test this against a REAL
Label Studio export from the live Cloud UI -- only against the documented
schema and against synthetic JSON built to match it. If your actual
export looks different in some way (a renamed field, a different
nesting), this script's `parse_ls_export_task` function is the one place
to adjust; the crop/label-writing logic downstream doesn't need to change.
Treat the first real export you run through this as a test: check a few
output crops against their labels before trusting the whole batch.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2


def parse_ls_export_task(task, source_image_path):
    """
    Extracts (boxes_px, texts) in region-id order from one Label Studio
    export task dict. Requires the ORIGINAL page image on disk (not the
    one embedded in the export, since we need exact pixel dimensions for
    the percent-to-pixel conversion, and re-reading our own saved render
    is simpler and more reliable than re-decoding a base64 blob from the
    export file).
    """
    img = cv2.imread(str(source_image_path))
    if img is None:
        raise FileNotFoundError(source_image_path)
    h, w = img.shape[:2]

    annotations = task.get("annotations", [])
    if not annotations:
        return [], []
    # Use the first (or most recent) completed annotation for this task.
    result = annotations[0].get("result", [])

    boxes_by_id = {}
    texts_by_id = {}
    order_by_id = {}
    order_counter = 0

    for item in result:
        region_id = item.get("id")
        item_type = item.get("type")
        value = item.get("value", {})
        if region_id not in order_by_id:
            order_by_id[region_id] = order_counter
            order_counter += 1

        if item_type == "rectanglelabels":
            x1 = value["x"] / 100 * w
            y1 = value["y"] / 100 * h
            x2 = x1 + value["width"] / 100 * w
            y2 = y1 + value["height"] / 100 * h
            boxes_by_id[region_id] = (round(x1), round(y1), round(x2), round(y2))
        elif item_type == "textarea":
            text_list = value.get("text", [])
            texts_by_id[region_id] = text_list[0] if text_list else ""

    ordered_ids = sorted(order_by_id, key=lambda rid: order_by_id[rid])

    boxes_px = []
    texts = []
    for rid in ordered_ids:
        if rid not in boxes_by_id:
            print(f"WARNING: region {rid} has text but no box, skipping.", file=sys.stderr)
            continue
        boxes_px.append(boxes_by_id[rid])
        texts.append(texts_by_id.get(rid, ""))

    return boxes_px, texts


def export_from_ls_json_batch(ls_export_path, base_out_dir, padding=4):
    """
    Handles a SINGLE export file containing tasks from MANY articles and
    pages mixed together (the realistic case once you've annotated
    everything in one project before exporting once at the end).

    Matches each task back to its original page by the article_id and
    page_index fields embedded in task["data"] at generation time (see
    generate_tasks.build_task), NOT by task order -- Label Studio does
    not guarantee export order matches import order once multiple
    articles and sessions are involved, so relying on order would risk
    silently pairing crops with the wrong article/page.

    base_out_dir: the same "dataset" folder used throughout this
    pipeline; page images are expected at
    {base_out_dir}/{article_id}/pages/page_{page_index:02d}.png
    exactly as render_article already writes them.

    Writes labels.csv and images/ PER ARTICLE, same as the rest of the
    pipeline, grouping tasks by their article_id automatically.
    """
    with open(ls_export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    base_out_dir = Path(base_out_dir)
    by_article = {}
    skipped = 0
    for task in tasks:
        data = task.get("data", {})
        article_id = data.get("article_id")
        page_index = data.get("page_index")
        if article_id is None or page_index is None:
            skipped += 1
            continue
        by_article.setdefault(article_id, {})[page_index] = task

    if skipped:
        print(
            f"WARNING: {skipped} task(s) had no article_id/page_index metadata "
            f"and were skipped -- these were likely generated before this "
            f"metadata field was added. Re-generate their tasks if needed."
        )

    results = {}
    for article_id, pages_by_index in sorted(by_article.items()):
        page_indices = sorted(pages_by_index.keys())
        images_dir = base_out_dir / article_id / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        labels = []
        for page_index in page_indices:
            task = pages_by_index[page_index]
            image_path = base_out_dir / article_id / "pages" / f"page_{page_index:02d}.png"
            if not image_path.exists():
                print(f"ERROR: {image_path} not found, skipping {article_id} page {page_index}.", file=sys.stderr)
                continue

            boxes_px, texts = parse_ls_export_task(task, image_path)
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]

            for line_index, (box, text) in enumerate(zip(boxes_px, texts), start=1):
                if not text.strip():
                    print(f"WARNING: empty text at {article_id} page {page_index} line {line_index}, skipping.")
                    continue
                x1, y1, x2, y2 = box
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    print(f"ERROR: empty crop at {article_id} page {page_index} line {line_index}, skipping.", file=sys.stderr)
                    continue
                file_name = f"{article_id}_p{page_index:02d}_line_{line_index:03d}.png"
                cv2.imwrite(str(images_dir / file_name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                labels.append({"file_name": file_name, "text": text})

        labels_csv_path = base_out_dir / article_id / "labels.csv"
        with labels_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(labels)

        print(f"{article_id}: wrote {len(labels)} crops to {images_dir}")
        results[article_id] = len(labels)

    total = sum(results.values())
    print(f"\nTOTAL: {total} pairs across {len(results)} articles.")
    return results


def export_from_ls_json(ls_export_path, page_image_paths, article_id, out_dir, padding=4):
    """
    ls_export_path: path to the JSON file exported from Label Studio.
    page_image_paths: list of paths, ORDERED to match the tasks in the
      export file (task 0 -> page_image_paths[0], etc). You are
      responsible for this ordering matching how you uploaded pages --
      Label Studio does not guarantee task order is preserved unless you
      set it up that way, so double-check page numbering against the
      images shown in the export before trusting this.
    """
    with open(ls_export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if len(tasks) != len(page_image_paths):
        print(
            f"WARNING: {len(tasks)} tasks in export but {len(page_image_paths)} "
            f"page image paths given. Extra/missing pages will be silently "
            f"skipped/ignored -- check this before trusting the output.",
            file=sys.stderr,
        )

    out_dir = Path(out_dir)
    images_dir = out_dir / article_id / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for page_index, (task, image_path) in enumerate(zip(tasks, page_image_paths)):
        boxes_px, texts = parse_ls_export_task(task, image_path)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        for line_index, (box, text) in enumerate(zip(boxes_px, texts), start=1):
            if not text.strip():
                print(f"WARNING: empty text at page {page_index} line {line_index}, skipping.")
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"ERROR: empty crop at page {page_index} line {line_index}, skipping.", file=sys.stderr)
                continue
            file_name = f"{article_id}_p{page_index:02d}_line_{line_index:03d}.png"
            cv2.imwrite(str(images_dir / file_name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            labels.append({"file_name": file_name, "text": text})

    labels_csv_path = out_dir / article_id / "labels.csv"
    with labels_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Wrote {len(labels)} crops to {images_dir}")
    print(f"Wrote labels: {labels_csv_path}")
    return len(labels)
