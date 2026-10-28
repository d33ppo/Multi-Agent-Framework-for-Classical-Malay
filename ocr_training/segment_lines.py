"""
Line segmentation for Jawi Qalam scans via horizontal projection profiling.

This is intentionally simple: it does not try to classify page layout.
It always attempts 2-column segmentation (this corpus has no confirmed
single-column full pages so far), and returns whatever boxes it finds.
Correctness is NOT assumed; the caller is responsible for checking the
detected count against ground truth and flagging mismatches for manual
correction. This module's only job is to produce a reasonable first-pass
guess, fast, not to guarantee accuracy.
"""
import cv2
import numpy as np


def binarize(gray_img):
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def find_gutter(binary, search_frac=(0.35, 0.65)):
    """Whole-page vertical projection to locate the column gutter."""
    h, w = binary.shape
    col_sums = np.sum(binary, axis=0)
    mid_start, mid_end = int(w * search_frac[0]), int(w * search_frac[1])
    window = col_sums[mid_start:mid_end]
    local_min_idx = np.argmin(window)
    return mid_start + local_min_idx


def segment_lines_in_column(col_binary, min_height=10, noise_height=3):
    """
    Horizontal projection profile line segmentation within one column.

    Uses a fixed-pitch peak-detection approach rather than a purely
    threshold-based band-merging approach. Direct inspection of real
    scans in this corpus showed that inter-band gaps do NOT separate
    cleanly into "within-line" vs "between-line" clusters (both a
    "biggest jump in sorted gaps" heuristic and a fixed gap threshold
    were tested and failed on real pages, either merging most of a
    column into a few giant blocks or fragmenting it), because Jawi text
    has visible word-spacing gaps that overlap in size with the smallest
    inter-line gaps.

    Instead: detect peaks in a smoothed row-ink profile, which correspond
    to line centers directly, and are far more robust to the gap-size
    ambiguity above since they look at where ink DENSITY is highest, not
    where it's absent.
    """
    from scipy.signal import find_peaks

    row_sums = np.sum(col_binary, axis=1)
    if row_sums.max() == 0:
        return []

    smoothed = np.convolve(row_sums, np.ones(5) / 5, mode='same')
    peaks, properties = find_peaks(
        smoothed,
        distance=40,
        prominence=row_sums.max() * 0.04,
    )

    if len(peaks) == 0:
        return []

    # Convert line-center peaks into (start, end) boxes by taking the
    # midpoint between consecutive peaks as the boundary between lines.
    lines = []
    for i, peak in enumerate(peaks):
        prev_boundary = 0 if i == 0 else (peaks[i - 1] + peak) // 2
        next_boundary = len(row_sums) if i == len(peaks) - 1 else (peak + peaks[i + 1]) // 2
        # trim the box to the actual ink extent within this slice, so
        # boxes hug the text rather than spanning the full midpoint range
        slice_sums = row_sums[prev_boundary:next_boundary]
        ink_rows = np.where(slice_sums > 0)[0]
        if len(ink_rows) == 0:
            continue
        y1 = prev_boundary + ink_rows[0]
        y2 = prev_boundary + ink_rows[-1] + 1
        if y2 - y1 >= min_height:
            lines.append((y1, y2))

    return lines


def segment_page(image_path, margin_frac=0.02):
    """
    Segments a rendered page image into line boxes, reading order
    (right column top-to-bottom, then left column top-to-bottom, matching
    Jawi/Arabic-script right-to-left column order).

    Returns list of boxes: [(x1, y1, x2, y2), ...] in original image
    pixel coordinates, in reading order.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    binary = binarize(gray)
    h, w = binary.shape
    margin = int(w * margin_frac)

    gutter_x = find_gutter(binary)

    right_col = binary[:, gutter_x:w - margin]
    left_col = binary[:, margin:gutter_x]

    right_lines = segment_lines_in_column(right_col)
    left_lines = segment_lines_in_column(left_col)

    boxes = []
    for (y1, y2) in right_lines:
        boxes.append((gutter_x, y1, w - margin, y2))
    for (y1, y2) in left_lines:
        boxes.append((margin, y1, gutter_x, y2))

    return boxes, {"gutter_x": gutter_x, "right_count": len(right_lines), "left_count": len(left_lines)}


if __name__ == "__main__":
    import sys
    boxes, info = segment_page(sys.argv[1])
    print(f"Detected {len(boxes)} boxes: {info}")
