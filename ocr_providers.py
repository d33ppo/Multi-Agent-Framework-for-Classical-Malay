"""
OCR providers for the upload pipeline.

Every provider implements one method, recognise_page(page), which returns the
text lines of one rendered page. The fine-tuned TrOCR provider lives in
ocr_trocr.py and is imported only when it is chosen, so torch, transformers and
ultralytics are not needed for the other providers.

All three providers run locally. Nothing is sent to an external service.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from document_storage import PROJECT_DIR
from llm_text_runtime import env_value
from ocr_input_adapter import parse_ocr_output_text

DEFAULT_OCR_PROVIDER = "tesseract"
DEFAULT_FIXTURE_DIR = "data/ocr_fixtures"


@dataclass(frozen=True)
class PageImage:
    """One rendered page handed to an OCR provider."""

    document_name: str  # upload file name without extension, safe characters only
    page_number: int  # 1-based
    page_count: int
    png: bytes


@dataclass
class OCRLine:
    text: str
    box: tuple[int, int, int, int] | None = None  # left, top, width, height in page pixels
    confidence: float | None = None  # 0 to 1, None when the provider has no score
    order: int | None = None  # 1-based reading-order position, None when the provider does not number lines
    low_confidence: bool | None = None  # confidence below OCR_CONFIDENCE_THRESHOLD, None when not checked

    def to_dict(self) -> dict:
        return asdict(self)


class OCRProvider(Protocol):
    name: str

    def recognise_page(self, page: PageImage) -> list[OCRLine]: ...


def safe_document_name(file_name: str | None) -> str:
    """Upload file name without folder or extension, reduced to safe characters."""
    stem = Path((file_name or "").replace("\\", "/")).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "document"


class FixtureOCRProvider:
    """
    Reads saved text instead of running a model. Used for tests and demos.

    Page N of an upload named `article.pdf` is read from `article_N.txt` or
    `article_pN.txt` in the fixture folder. A one-page upload may also use
    `article.txt`. Saved OCR output files with a metadata block are accepted.
    """

    name = "fixture"

    def __init__(self, fixture_dir: str | Path | None = None):
        folder = Path(fixture_dir or env_value("OCR_FIXTURE_DIR", DEFAULT_FIXTURE_DIR))
        self.fixture_dir = folder if folder.is_absolute() else PROJECT_DIR / folder

    def fixture_path(self, page: PageImage) -> Path:
        names = [f"{page.document_name}_{page.page_number}.txt", f"{page.document_name}_p{page.page_number}.txt"]
        if page.page_count == 1:
            names.append(f"{page.document_name}.txt")
        for name in names:
            path = self.fixture_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"No OCR fixture for page {page.page_number} of '{page.document_name}'. "
            f"Looked for {', '.join(names)} in {self.fixture_dir}."
        )

    def recognise_page(self, page: PageImage) -> list[OCRLine]:
        content = self.fixture_path(page).read_text(encoding="utf-8-sig")
        body, _metadata = parse_ocr_output_text(content)
        return [OCRLine(text=line.strip()) for line in body.splitlines() if line.strip()]


class TesseractOCRProvider:
    """Local Tesseract baseline (Arabic language data). Weak on Jawi-specific letters."""

    name = "tesseract"

    def recognise_page(self, page: PageImage) -> list[OCRLine]:
        from ocr_jawi import extract_jawi_lines  # imported here so OpenCV and pytesseract are only needed for this provider

        return [OCRLine(**line) for line in extract_jawi_lines(page.png)]


def _trocr_provider() -> OCRProvider:
    from ocr_trocr import TrOCROCRProvider

    return TrOCROCRProvider()


PROVIDERS = {"fixture": FixtureOCRProvider, "tesseract": TesseractOCRProvider, "trocr": _trocr_provider}


def default_provider_name() -> str:
    """OCR_PROVIDER when set; otherwise trocr once OCR_MODEL_PATH is set, else the Tesseract baseline."""
    return env_value("OCR_PROVIDER") or ("trocr" if env_value("OCR_MODEL_PATH") else DEFAULT_OCR_PROVIDER)


def build_ocr_provider(name: str | None = None) -> OCRProvider:
    """Create the configured provider. The server chooses it, not the client."""
    name = (name or default_provider_name()).lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown OCR_PROVIDER '{name}'. Choose one of: {', '.join(sorted(PROVIDERS))}.")
    return PROVIDERS[name]()
