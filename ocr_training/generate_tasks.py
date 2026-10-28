"""
Generates Label Studio import tasks (with pre-filled box+text predictions)
from our auto-segmented boxes and a best-effort ground-truth text guess.

Label Studio represents region coordinates as PERCENTAGES (0-100) of the
image's width/height, not raw pixels -- this is Label Studio's documented
format, confirmed against their schema, not an assumption. Getting this
conversion right is critical: a pixel value used where a percentage is
expected would silently misplace every box.

Each task's predictions link a box and its transcription text together
via a shared region "id" -- this is how Label Studio's OCR-style config
(RectangleLabels + per-region TextArea) associates a text answer with
its specific box.
"""
import json
import base64
from pathlib import Path


def _pct(value, dimension):
    return round((value / dimension) * 100, 4)


def boxes_and_text_to_ls_predictions(boxes_px, texts, image_width, image_height):
    """
    boxes_px: list of (x1, y1, x2, y2) in pixel coordinates, reading order.
    texts: list of strings, same length and order as boxes_px.
    Returns a Label Studio "predictions" list (single prediction, one
    result item per box + one per its paired text).
    """
    if len(boxes_px) != len(texts):
        raise ValueError(f"boxes ({len(boxes_px)}) and texts ({len(texts)}) must be same length")

    result = []
    for i, ((x1, y1, x2, y2), text) in enumerate(zip(boxes_px, texts)):
        region_id = f"region_{i+1}"
        x_pct = _pct(x1, image_width)
        y_pct = _pct(y1, image_height)
        w_pct = _pct(x2 - x1, image_width)
        h_pct = _pct(y2 - y1, image_height)

        result.append({
            "id": region_id,
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
                "rotation": 0,
                "rectanglelabels": ["line"],
            },
        })
        result.append({
            "id": region_id,
            "type": "textarea",
            "from_name": "transcription",
            "to_name": "image",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
                "rotation": 0,
                "text": [text],
            },
        })

    return [{"model_version": "auto_segment_v1", "result": result}]


def image_to_data_url(image_path, max_dimension=1600, jpeg_quality=85):
    """
    Encodes an image as a data URL, downscaling and JPEG-compressing it
    first if it's large.

    This embedded image is ONLY used for Label Studio's UI so you can see
    the page while drawing/correcting boxes -- it has no effect on final
    crop quality, since export_from_ls_json_batch always re-reads the
    ORIGINAL full-resolution page PNG from disk to produce the actual
    training crops, never the embedded preview. Confirmed by inspecting
    that function: it opens base_out_dir/article_id/pages/page_NN.png
    directly, not anything from the task JSON.

    Embedding full 400 DPI PNGs as base64 was found in practice to
    produce import files well over 100MB for multi-page articles, which
    Label Studio's import failed on ("Unexpected end of JSON input" --
    confirmed the JSON itself was valid, so this was a transfer/size
    limit issue, not file corruption). Downscaling to a reasonable
    viewing resolution and using JPEG instead of PNG cuts file size
    dramatically while remaining perfectly legible for annotation.
    """
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    scale = min(1.0, max_dimension / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not success:
        raise RuntimeError(f"Failed to encode {image_path}")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_task(image_path, image_width, image_height, boxes_px, texts, article_id=None, page_index=None, meta=None):
    """
    Returns one Label Studio task dict, ready to be placed in a list and
    written as JSON for import.

    article_id and page_index are embedded directly into the task's data
    payload (as plain visible fields, not hidden metadata) so that once
    many articles' tasks are mixed together in one project, the export
    step can match each annotated task back to the correct original page
    by these fields rather than by trusting task order -- which Label
    Studio does not guarantee is preserved once multiple import batches
    and sessions are involved.
    """
    task = {
        "data": {
            "image": image_to_data_url(image_path),
            "article_id": article_id,
            "page_index": page_index,
        },
        "predictions": boxes_and_text_to_ls_predictions(boxes_px, texts, image_width, image_height),
    }
    if meta:
        task["data"].update(meta)
    return task


def write_tasks_json(tasks, out_path):
    Path(out_path).write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {out_path}")
