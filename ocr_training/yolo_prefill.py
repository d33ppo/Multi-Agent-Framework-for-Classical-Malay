"""
YOLO line-detector pre-fill for Label Studio.

Trains a text-line detector on the boxes you already annotated in Label Studio,
then uses it to pre-fill boxes (plus ground-truth text in reading order) for
new articles, e.g. 1951_01. You still correct every page in Label Studio; this
only replaces the projection-profile heuristic with a model trained on Qalam.

The same detector is the product's line detector (plan task 4.2), so this is not
throwaway work. Test articles are excluded from detector training so the
detector can be evaluated honestly later.

Typical Colab use (see the notebook):
    import yolo_prefill as yp
    stats = yp.export_to_yolo(EXPORT_JSON, "yolo_ds", exclude_articles=TEST_ARTICLES,
                              overrides={1: ("1950_11_002", 0)})
    model_path = yp.train("yolo_ds/data.yaml", project="/content/drive/MyDrive/jawi_dataset/yolo_runs")
    tasks = yp.prefill_article(model_path, page_paths, gt_lines, article_id)
"""
from __future__ import annotations

import base64
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Label Studio export -> YOLO dataset
# ---------------------------------------------------------------------------


def _decode_data_url(data_url: str):
    """Decode the base64 preview image embedded in a Label Studio task."""
    if not data_url or not data_url.startswith("data:"):
        return None
    b64 = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _latest_annotation(task):
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    return max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def load_pages(ls_export_path, overrides=None):
    """
    Read a Label Studio JSON export and return one record per (article_id, page_index):
    {"task_id", "article_id", "page_index", "image" (BGR preview), "boxes" [(x,y,w,h) in %],
     "texts" [str], "updated_at"}.
    Duplicate pages keep the most recently updated annotation. Tasks without
    metadata are kept only if listed in `overrides` {task_id: (article_id, page_index)}.
    Returns (pages, report) where report lists what was dropped and why.
    """
    overrides = overrides or {}
    tasks = json.loads(Path(ls_export_path).read_text(encoding="utf-8"))
    report = {"no_metadata": [], "duplicates_dropped": [], "incomplete_pages": [], "no_image": []}
    by_key = {}
    for task in tasks:
        data = task.get("data", {})
        art, page = data.get("article_id"), data.get("page_index")
        if (art is None or page is None) and task.get("id") in overrides:
            art, page = overrides[task["id"]]
        if art is None or page is None:
            report["no_metadata"].append(task.get("id"))
            continue
        ann = _latest_annotation(task)
        if ann is None:
            continue
        boxes, texts = {}, {}
        for r in ann.get("result", []):
            v = r.get("value", {})
            if r.get("type") == "rectanglelabels":
                boxes[r["id"]] = (v["x"], v["y"], v["width"], v["height"])
            elif r.get("type") == "textarea":
                texts[r["id"]] = (v.get("text") or [""])[0]
        ids = list(boxes)
        rec = {
            "task_id": task.get("id"),
            "article_id": art,
            "page_index": int(page),
            "boxes": [boxes[i] for i in ids],
            "texts": [texts.get(i, "") for i in ids],
            "updated_at": ann.get("updated_at") or "",
            "image_url": data.get("image", ""),
        }
        key = (art, int(page))
        if key in by_key:
            old = by_key[key]
            keep, drop = (rec, old) if rec["updated_at"] >= old["updated_at"] else (old, rec)
            report["duplicates_dropped"].append({"key": key, "kept_task": keep["task_id"], "dropped_task": drop["task_id"]})
            by_key[key] = keep
        else:
            by_key[key] = rec
    pages = []
    for key, rec in sorted(by_key.items()):
        empty = sum(1 for t in rec["texts"] if not t.strip())
        if empty:
            report["incomplete_pages"].append({"key": key, "task_id": rec["task_id"], "boxes_without_text": empty})
            rec["incomplete"] = True
        else:
            rec["incomplete"] = False
        pages.append(rec)
    return pages, report


def export_to_yolo(ls_export_path, out_dir, exclude_articles=(), overrides=None,
                   val_frac=0.12, seed=42, skip_incomplete=True):
    """
    Write a YOLO detection dataset (single class 'text_line') from a Label Studio export.
    Uses the preview images embedded in the export, so it needs nothing else from Drive.
    Splits by ARTICLE (never by page). Articles in `exclude_articles` (your locked test
    set) are left out entirely.
    """
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    pages, report = load_pages(ls_export_path, overrides)
    exclude = set(exclude_articles)
    usable = [p for p in pages if p["article_id"] not in exclude and not (skip_incomplete and p["incomplete"])]
    articles = sorted({p["article_id"] for p in usable})
    rng = random.Random(seed)
    shuffled = articles[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    val_articles = set(shuffled[:n_val])

    counts = {"train": [0, 0], "val": [0, 0]}
    for p in usable:
        split = "val" if p["article_id"] in val_articles else "train"
        img = _decode_data_url(p["image_url"])
        if img is None:
            report["no_image"].append(p["task_id"])
            continue
        name = f"{p['article_id']}_p{p['page_index']:02d}"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for (x, y, w, h) in p["boxes"]:
            cx, cy = (x + w / 2) / 100, (y + h / 2) / 100
            bw, bh = w / 100, h / 100
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            lines.append(f"0 {cx:.6f} {cy:.6f} {min(bw, 1):.6f} {min(bh, 1):.6f}")
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split][0] += 1
        counts[split][1] += len(lines)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: text_line\n", encoding="utf-8")
    summary = {
        "train_pages": counts["train"][0], "train_boxes": counts["train"][1],
        "val_pages": counts["val"][0], "val_boxes": counts["val"][1],
        "val_articles": sorted(val_articles), "excluded_test_articles": sorted(exclude),
        **report,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 2. Training (run on a Colab GPU)
# ---------------------------------------------------------------------------


def train(data_yaml, project, name="qalam_lines", model="yolov8n.pt", epochs=80, imgsz=1280,
          batch=8, patience=20, device=None):
    """Train a single-class line detector. Returns the path to best.pt."""
    from ultralytics import YOLO

    kwargs = dict(data=str(data_yaml), project=str(project), name=name, epochs=epochs, imgsz=imgsz,
                  batch=batch, patience=patience, exist_ok=True,
                  fliplr=0.0,        # keep layout direction (RTL columns)
                  mosaic=0.3,        # mostly whole pages; layout context matters
                  degrees=1.0, translate=0.05, scale=0.25, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3)
    if device is not None:
        kwargs["device"] = device
    yolo = YOLO(model)
    yolo.train(**kwargs)
    # Ultralytics may nest relative project paths (e.g. runs/detect/<project>), so ask the trainer.
    return str(Path(yolo.trainer.save_dir) / "weights" / "best.pt")


# ---------------------------------------------------------------------------
# 3. Prediction + reading order
# ---------------------------------------------------------------------------


def _suppress_contained(boxes, max_contained=0.6):
    """
    Drop a box when most of it lies inside a higher-confidence box. YOLO's NMS keeps
    fragments of one line (low IoU with the full line box); this removes them.
    """
    kept = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k in kept:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            karea = max(1e-6, (k[2] - k[0]) * (k[3] - k[1]))
            if ix * iy / min(area, karea) > max_contained:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def predict_boxes(model, image_bgr, conf=0.35, imgsz=1280, suppress=True):
    """Return [(x1, y1, x2, y2, conf)] in pixels of `image_bgr`."""
    res = model.predict(image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    return _suppress_contained(out) if suppress else out


def _overlap_frac(a, b):
    """Horizontal overlap as a fraction of the narrower box."""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def _order_columns(boxes):
    """Group boxes into columns by horizontal overlap; columns right->left, lines top->bottom."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap_frac(boxes[i], boxes[j]) >= 0.5:
                parent[find(i)] = find(j)
    cols = {}
    for i in range(n):
        cols.setdefault(find(i), []).append(boxes[i])
    ordered_cols = sorted(cols.values(), key=lambda c: -np.mean([(b[0] + b[2]) / 2 for b in c]))
    out = []
    for col in ordered_cols:
        out.extend(sorted(col, key=lambda b: (b[1] + b[3]) / 2))
    return out


def _split_by_gap(boxes, lo, hi, min_gap):
    """Split boxes into groups separated by an empty band of at least min_gap along one axis."""
    order = sorted(boxes, key=lambda b: b[lo])
    groups, current, reach = [], [order[0]], order[0][hi]
    for b in order[1:]:
        if b[lo] - reach >= min_gap:
            groups.append(current)
            current, reach = [b], b[hi]
        else:
            current.append(b)
            reach = max(reach, b[hi])
    groups.append(current)
    return groups


def _rows_right_to_left(boxes):
    """Leaf case: group boxes into rows by vertical overlap; rows top->bottom, boxes right->left."""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        for row in rows:
            if row["y1"] <= cy <= row["y2"]:
                row["boxes"].append(b)
                break
        else:
            rows.append({"y1": b[1], "y2": b[3], "boxes": [b]})
    out = []
    for row in rows:
        out.extend(sorted(row["boxes"], key=lambda b: -(b[0] + b[2]) / 2))
    return out


def _column_cut(boxes, min_gap_x):
    """
    Find a vertical cut between columns. Tolerates a couple of small boxes crossing
    the gutter (page numbers, stray marks), which would block a strict XY-cut.
    Returns (left, right) or None. Each side must keep at least 2 boxes.
    """
    n = len(boxes)
    if n < 4:
        return None
    x0 = int(min(b[0] for b in boxes))
    x1 = int(max(b[2] for b in boxes)) + 1
    cover = np.zeros(x1 - x0 + 2, dtype=np.int32)
    for b in boxes:
        cover[int(b[0]) - x0] += 1
        cover[int(b[2]) - x0 + 1] -= 1
    cover = np.cumsum(cover)[: x1 - x0]
    allowance = 1 if n < 40 else 2
    best = None
    i = 0
    while i < len(cover):
        if cover[i] <= allowance:
            j = i
            while j < len(cover) and cover[j] <= allowance:
                j += 1
            run_len, run_min = j - i, int(cover[i:j].min())
            if run_len >= (1 if run_min == 0 else min_gap_x):
                cut = x0 + (i + j) / 2
                left = [b for b in boxes if (b[0] + b[2]) / 2 < cut]
                right = [b for b in boxes if (b[0] + b[2]) / 2 >= cut]
                if len(left) >= 2 and len(right) >= 2:
                    key = (run_min, -run_len)
                    if best is None or key < best[0]:
                        best = (key, left, right)
            i = j
        else:
            i += 1
    return None if best is None else (best[1], best[2], best[0][1] * -1)


def _band_cut(boxes):
    """Widest fully empty horizontal gap across the region. Returns (top, bottom, gap) or None."""
    order = sorted(boxes, key=lambda b: b[1])
    best, reach = None, order[0][3]
    for k in range(1, len(order)):
        gap = order[k][1] - reach
        if gap > 0 and (best is None or gap > best[0]):
            best = (gap, k)
        reach = max(reach, order[k][3])
    if best is None:
        return None
    gap, k = best
    return order[:k], order[k:], gap


def _xy_cut(boxes, min_gap_x, min_gap_y, depth=0):
    if len(boxes) <= 1 or depth > 60:
        return _rows_right_to_left(boxes)
    col = _column_cut(boxes, min_gap_x)
    band = _band_cut(boxes)
    # Like classical XY-cut, take the wider gap first; a band gap must be clearly
    # wider than the column gutter, so aligned text rows never split columns.
    if band is not None and (col is None or band[2] > 1.5 * col[2]):
        top, bottom, _ = band
        return [b for g in (top, bottom) for b in _xy_cut(g, min_gap_x, min_gap_y, depth + 1)]
    if col is not None:
        left, right, _ = col
        return [b for g in (right, left) for b in _xy_cut(g, min_gap_x, min_gap_y, depth + 1)]  # right first
    bands = _split_by_gap(boxes, 1, 3, min_gap_y)         # horizontal cut = bands
    if len(bands) > 1:
        return [b for g in bands for b in _xy_cut(g, min_gap_x, min_gap_y, depth + 1)]
    return _rows_right_to_left(boxes)


def reading_order(boxes, page_width, min_gap_frac=0.008):
    """
    Jawi reading order via recursive XY-cut: wherever an empty vertical gap runs
    through a region, it is split into columns read right to left; otherwise it is
    split at horizontal gaps into bands read top to bottom. Pieces of the same
    printed line are read right to left. Headlines spanning columns block a
    vertical cut, so they come out in their own band first.
    """
    if not boxes:
        return []
    return _xy_cut(list(boxes), min_gap_frac * page_width, 1.0)


# ---------------------------------------------------------------------------
# 4. Build Label Studio tasks for a new article
# ---------------------------------------------------------------------------


def prefill_article(model, page_paths, gt_lines, article_id, conf=0.25, imgsz=1280,
                    model_version="yolo_prefill_v1"):
    """
    model: a loaded ultralytics YOLO model.
    page_paths: full-resolution page PNGs in page order.
    gt_lines: the article's ground-truth Jawi lines in reading order.
    Returns (tasks, stats). Text is assigned to boxes in reading order across pages;
    if the detector misses or adds a line, the text after it shifts, which you fix
    in Label Studio as before.
    """
    from generate_tasks import build_task  # the compressed-preview version

    tasks, cursor, per_page = [], 0, []
    for page_index, page_path in enumerate(page_paths):
        img = cv2.imread(str(page_path))
        h, w = img.shape[:2]
        boxes = reading_order(predict_boxes(model, img, conf=conf, imgsz=imgsz), w)
        boxes_px = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
        texts = (gt_lines[cursor:cursor + len(boxes_px)] + [""] * len(boxes_px))[:len(boxes_px)]
        cursor += len(boxes_px)
        task = build_task(str(page_path), w, h, boxes_px, texts, article_id=article_id, page_index=page_index)
        for pred in task.get("predictions", []):
            pred["model_version"] = model_version
        tasks.append(task)
        per_page.append(len(boxes_px))
    stats = {"article_id": article_id, "pages": len(page_paths), "boxes_per_page": per_page,
             "total_boxes": sum(per_page), "gt_lines": len(gt_lines)}
    return tasks, stats


# ---------------------------------------------------------------------------
# 5. Quick quality check against your own annotations
# ---------------------------------------------------------------------------


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_stats(pred, gold, thr=0.5):
    """Greedy one-to-one matching at IoU >= thr. Returns precision, recall, matched count."""
    used, tp = set(), 0
    for p in sorted(pred, key=lambda b: -(b[4] if len(b) > 4 else 1)):
        best, best_j = 0.0, None
        for j, g in enumerate(gold):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "tp": tp,
            "pred": len(pred), "gold": len(gold)}


def draw_order(image_bgr, boxes, out_path):
    """Save a debug image with boxes numbered in reading order."""
    img = image_bgr.copy()
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, str(i), (x2 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    cv2.imwrite(str(out_path), img)
