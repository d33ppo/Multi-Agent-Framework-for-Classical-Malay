"""
Build the TrOCR line dataset (image crop + Jawi text) from a Label Studio export.

- Crops from the ORIGINAL full-resolution page PNGs (dataset/<article>/pages/page_NN.png)
  when they exist, otherwise from the preview image embedded in the export (logged).
- Each crop is saved with an extra margin; the true box position inside the crop is
  stored in the CSV so training can jitter the box (detector boxes at inference are
  never pixel-identical to hand-drawn ones) and evaluation can use a fixed margin.
- Splits by ARTICLE: the locked test articles, then the same validation articles the
  YOLO detector uses (same seed and fraction), the rest is train.

Usage (Colab):
    import trocr_dataset as td
    stats = td.build_line_dataset(EXPORT_JSON, pages_base="dataset", out_dir="/content/trocr_ds",
                                  test_articles=TEST_ARTICLES, overrides={1: ("1950_11_002", 0)})
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import cv2

import yolo_prefill as yp

JAWI_SPECIAL = "ڠڽڬچۏةف"  # Jawi letters standard Arabic OCR lacks, plus ta marbuta and fa (used for /p/)
MARGIN_FRAC = 0.30          # stored margin, as a fraction of the box height
EVAL_MARGIN_FRAC = 0.12     # margin used at evaluation / inference


# Label normalisation, decided by checking crops against labels (23 Oct 2026):
# - The 1950 print uses ف for /p/; the 1950_11 corrector wrote ڤ (~1,500 times) while
#   1950_12 wrote ف. Crops confirm the print shows a single dot (ف).
# - ک (Persian keheh) and ݢ are encoding variants of ك and Jawi ڬ.
# - پ appears only as a typo for ڽ (e.g. "باپق" for banyak).
# - إ is used by one corrector only; the print shows a plain alif.
# Digits are NOT changed: the print mixes Western digits and the Jawi ٢ repetition mark.
LABEL_MAP = {"ڤ": "ف", "ک": "ك", "ݢ": "ڬ", "پ": "ڽ", "إ": "ا"}
LABEL_FIX_COUNTS: Counter = Counter()

# Pages whose labels are the uncorrected auto-transliteration (pre-filled from the wrong sheet).
# Excluded from v1; fix them in Label Studio from the _DONE sheet for v2.
DEFAULT_EXCLUDE_PAGES = {("1950_12_016", 0), ("1950_12_017", 0)}


def normalise_text(text: str) -> str:
    """NFC, LABEL_MAP, trim, collapse whitespace. Spelling is otherwise kept as printed."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in LABEL_MAP.items():
        n = text.count(src)
        if n:
            LABEL_FIX_COUNTS[f"{src}->{dst}"] += n
            text = text.replace(src, dst)
    return " ".join(text.split())


def split_articles(articles, test_articles, val_frac=0.12, seed=42):
    """Same rule as yolo_prefill.export_to_yolo, so detector and recogniser share val articles."""
    rest = sorted(a for a in articles if a not in set(test_articles))
    shuffled = rest[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    val = set(shuffled[:n_val])
    return {a: ("test" if a in set(test_articles) else "val" if a in val else "train") for a in articles}


def build_line_dataset(ls_export_path, pages_base, out_dir, test_articles, overrides=None,
                       val_frac=0.12, seed=42, allow_preview_fallback=True,
                       exclude_pages=DEFAULT_EXCLUDE_PAGES):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    pages, report = yp.load_pages(ls_export_path, overrides)
    incomplete = [p for p in pages if p["incomplete"]]
    if incomplete:
        raise ValueError(f"Pages with boxes but no text, finish them first: {[(p['article_id'], p['page_index']) for p in incomplete]}")

    split_of = split_articles({p["article_id"] for p in pages}, test_articles, val_frac, seed)
    LABEL_FIX_COUNTS.clear()
    rows, sources, excluded = [], Counter(), []
    for p in pages:
        art, pi = p["article_id"], p["page_index"]
        if (art, pi) in set(exclude_pages):
            excluded.append((art, pi, len(p["boxes"])))
            continue
        page_png = Path(pages_base) / art / "pages" / f"page_{pi:02d}.png"
        if page_png.exists():
            img, src = cv2.imread(str(page_png)), "page_png"
        elif allow_preview_fallback:
            img, src = yp._decode_data_url(p["image_url"]), "preview"
        else:
            raise FileNotFoundError(page_png)
        h, w = img.shape[:2]
        for li, ((x, y, bw, bh), text) in enumerate(zip(p["boxes"], p["texts"]), start=1):
            text = normalise_text(text)
            if not text:
                continue
            x1, y1 = x / 100 * w, y / 100 * h
            x2, y2 = x1 + bw / 100 * w, y1 + bh / 100 * h
            m = MARGIN_FRAC * (y2 - y1)
            cx1, cy1 = max(0, int(x1 - m)), max(0, int(y1 - m))
            cx2, cy2 = min(w, int(x2 + m)), min(h, int(y2 + m))
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or (y2 - y1) < 3 or (x2 - x1) < 3:
                continue
            name = f"{art}_p{pi:02d}_l{li:03d}.png"
            cv2.imwrite(str(out / "images" / name), crop)
            rows.append({
                "file_name": name, "text": text, "article_id": art, "page_index": pi, "line_index": li,
                "split": split_of[art], "source": src,
                # true box inside the crop, in crop pixels
                "bx1": round(x1 - cx1, 1), "by1": round(y1 - cy1, 1),
                "bx2": round(x2 - cx1, 1), "by2": round(y2 - cy1, 1),
            })
            sources[src] += 1

    for split in ("train", "val", "test"):
        with open(out / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(r for r in rows if r["split"] == split)

    chars = Counter(c for r in rows for c in r["text"])
    lengths = sorted(len(r["text"]) for r in rows)
    stats = {
        "lines": {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")},
        "articles": {s: sorted({r["article_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
        "pages": len(pages),
        "crop_source": dict(sources),
        "char_set_size": len(chars),
        "jawi_special_counts": {c: chars.get(c, 0) for c in JAWI_SPECIAL},
        "line_length_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                              "p95": lengths[int(len(lengths) * 0.95)], "max": lengths[-1]},
        "charset": "".join(sorted(chars)),
        "duplicates_dropped": report["duplicates_dropped"],
        "excluded_pages": excluded,
        "label_fixes": dict(LABEL_FIX_COUNTS),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def crop_for(image, row, train: bool, rng=random):
    """
    Return the line image to feed the model. Train: random margin and small box jitter.
    Eval: fixed EVAL_MARGIN_FRAC margin. `image` is the stored crop (PIL or numpy HxWxC).
    """
    import numpy as np
    arr = np.asarray(image)
    H, W = arr.shape[:2]
    bx1, by1, bx2, by2 = (float(row[k]) for k in ("bx1", "by1", "bx2", "by2"))
    bh = max(1.0, by2 - by1)
    if train:
        def m():
            return rng.uniform(0.0, MARGIN_FRAC) * bh
        jitter = 0.08 * bh
        x1 = bx1 - m() + rng.uniform(-jitter, jitter)
        x2 = bx2 + m() + rng.uniform(-jitter, jitter)
        y1 = by1 - m() + rng.uniform(-jitter, jitter)
        y2 = by2 + m() + rng.uniform(-jitter, jitter)
    else:
        e = EVAL_MARGIN_FRAC * bh
        x1, y1, x2, y2 = bx1 - e, by1 - e, bx2 + e, by2 + e
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return arr
    return arr[y1:y2, x1:x2]
