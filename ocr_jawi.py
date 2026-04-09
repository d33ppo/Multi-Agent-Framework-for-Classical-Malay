"""
OCR module for extracting Jawi text from Classical Malay manuscript images.

Jawi is the Arabic-script writing system used for Malay.
This module uses Tesseract OCR with Arabic language data to extract Jawi text.
"""

import os
import sys
import time
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Point to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use local tessdata directory (contains ara.traineddata)
TESSDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess manuscript image to improve OCR accuracy.

    Steps:
    - Convert to grayscale
    - Denoise
    - Binarize with Otsu thresholding
    - Deskew (straighten the image)
    """
    img = cv2.imread(image_path)
    if img is None:
        # Try with PIL for GIF and other formats
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Binarize using Otsu's thresholding
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew: find the angle of the text and rotate to correct
    binary = _deskew(binary)

    return binary


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew in a binarized image."""
    coords = np.column_stack(np.where(image < 128))  # dark pixels
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:  # skip if nearly straight
        return image
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def extract_jawi_text(image_path: str) -> dict:
    """
    Extract Jawi text from a manuscript image using Tesseract OCR.

    Args:
        image_path: Path to the manuscript image (GIF, PNG, JPG, PDF).

    Returns:
        dict with keys:
            - raw_text: extracted Jawi text (str)
            - confidence: average OCR confidence score (float)
            - processing_time: time taken in seconds (float)
            - language: OCR language used (str)
    """
    start_time = time.time()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Preprocess
    processed = preprocess_image(image_path)

    # Set TESSDATA_PREFIX so Tesseract finds ara.traineddata from our local tessdata dir
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

    # Tesseract config:
    # -l ara         -> Arabic language (covers Jawi script)
    # --oem 1        -> LSTM neural net engine
    # --psm 6        -> Assume a single uniform block of text
    # -c preserve_interword_spaces=1 -> keep spacing for RTL text
    custom_config = "-l ara --oem 1 --psm 6 -c preserve_interword_spaces=1"

    # Run OCR and get detailed output for confidence
    data = pytesseract.image_to_data(
        processed, config=custom_config, output_type=pytesseract.Output.DICT
    )

    # Extract text
    raw_text = pytesseract.image_to_string(processed, config=custom_config)

    # Compute mean confidence (ignore -1 values which mean no text)
    confidences = [int(c) for c in data["conf"] if int(c) != -1]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    processing_time = time.time() - start_time

    return {
        "raw_text": raw_text.strip(),
        "confidence": round(avg_confidence, 2),
        "processing_time": round(processing_time, 3),
        "language": "ara (Jawi)",
    }


def save_output(result: dict, output_path: str) -> None:
    """Save OCR result to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Jawi OCR Result ===\n\n")
        f.write(result["raw_text"])
        f.write(f"\n\n=== Metadata ===\n")
        f.write(f"Language     : {result['language']}\n")
        f.write(f"Confidence   : {result['confidence']}%\n")
        f.write(f"Process Time : {result['processing_time']}s\n")
    print(f"Output saved to: {output_path}")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test-data/jawi-manuscript-2.gif"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/jawi_ocr_output-2.txt"

    print(f"Processing: {image_path}")
    print(f"Tessdata dir: {TESSDATA_DIR}")

    result = extract_jawi_text(image_path)

    # Set stdout to UTF-8 for Windows consoles that default to cp1252
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("\n=== Extracted Jawi Text ===")
    print(result["raw_text"])
    print(f"\n=== Metadata ===")
    print(f"Language     : {result['language']}")
    print(f"Confidence   : {result['confidence']}%")
    print(f"Process Time : {result['processing_time']}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_output(result, output_path)


if __name__ == "__main__":
    main()
