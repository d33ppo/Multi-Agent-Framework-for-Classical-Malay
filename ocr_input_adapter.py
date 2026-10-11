"""
Adapter for normalizing OCR inputs into a fixed downstream contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline_types import OCRInputRecord


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def load_ocr_input(input_path: str, ocr_provider: str = "auto") -> OCRInputRecord:
    """
    Load OCR input from:
      - raw/plain text fixtures
      - saved OCR output .txt files
      - canonical .json input records
      - local image files via Tesseract
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_record(path)
    if suffix == ".txt":
        return _load_text_record(path)
    if suffix in IMAGE_SUFFIXES:
        return _load_image_with_tesseract(path, requested_provider=ocr_provider)
    if suffix == ".pdf":
        raise ValueError(
            "Direct PDF OCR is not enabled in this lightweight pipeline. "
            "Use a pre-generated OCR text fixture or export page images first."
        )

    raise ValueError(f"Unsupported input type: {path.suffix}")


def _load_json_record(path: Path) -> OCRInputRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    return OCRInputRecord(
        source_file=data["source_file"],
        ocr_provider=data["ocr_provider"],
        ocr_jawi_text=data["ocr_jawi_text"].strip(),
        metadata=data.get("metadata", {}),
    )


def _load_text_record(path: Path) -> OCRInputRecord:
    content = path.read_text(encoding="utf-8")
    body, metadata = _parse_ocr_output_text(content)
    provider = _infer_provider(path, metadata, content)
    source_file = metadata.get("source_file") or str(path)
    return OCRInputRecord(
        source_file=source_file,
        ocr_provider=provider,
        ocr_jawi_text=body.strip(),
        metadata=metadata,
    )


def _load_image_with_tesseract(path: Path, requested_provider: str) -> OCRInputRecord:
    from ocr_jawi import extract_jawi_text

    if requested_provider not in {"auto", "tesseract"}:
        raise ValueError(
            f"Image input currently supports only local Tesseract. "
            f"Got ocr_provider='{requested_provider}'."
        )
    result = extract_jawi_text(str(path))
    metadata = {
        "language": result.get("language"),
        "confidence": result.get("confidence"),
        "processing_time": result.get("processing_time"),
    }
    return OCRInputRecord(
        source_file=str(path),
        ocr_provider="tesseract",
        ocr_jawi_text=result["raw_text"].strip(),
        metadata=metadata,
    )


def _parse_ocr_output_text(content: str) -> tuple[str, dict[str, str]]:
    metadata: dict[str, str] = {}
    if "=== Metadata ===" not in content:
        return content.strip(), metadata

    main_text, metadata_block = content.split("=== Metadata ===", 1)
    lines = [line.rstrip() for line in main_text.strip().splitlines()]
    while lines and lines[0].startswith("==="):
        lines.pop(0)
    body = "\n".join(lines).strip()

    for raw_line in metadata_block.strip().splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        metadata[_normalize_meta_key(key)] = value.strip()

    return body, metadata


def _normalize_meta_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower().strip()).strip("_")


def _infer_provider(path: Path, metadata: dict[str, str], content: str) -> str:
    if "provider" in metadata:
        return metadata["provider"].lower()

    lower_name = path.name.lower()
    if "qwen" in lower_name:
        return "qwen-fixture"
    if "kraken" in lower_name:
        return "kraken-fixture"
    if "tesseract" in lower_name or "ocr_output" in lower_name:
        return "tesseract"

    lowered = content.lower()
    if "jawi llm-ocr result (qwen)" in lowered:
        return "qwen-fixture"
    if "jawi llm-ocr result (kraken)" in lowered:
        return "kraken-fixture"
    if "jawi ocr result" in lowered:
        return "tesseract"

    return "text-fixture"
