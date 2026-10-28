"""
Upload type detection and page rendering.

The file type is decided by the first bytes of the content, never by the file
name or the Content-Type header. A PDF is rendered to one PNG per page with
PyMuPDF. A PNG or JPEG upload is treated as a one-page document.
"""

from __future__ import annotations

from typing import Iterator

import pymupdf

from llm_text_runtime import _get_int_env

DEFAULT_RENDER_DPI = 300
# The TrOCR model and the YOLO line detector were trained on pages rendered at 400 DPI.
TROCR_RENDER_DPI = 400

# Leading bytes that identify each accepted type.
MAGIC_BYTES = {
    "pdf": b"%PDF-",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
}


class UnreadableDocumentError(ValueError):
    """The upload has an accepted type but cannot be opened or has no pages."""


def detect_file_type(data: bytes) -> str | None:
    """Return "pdf", "png" or "jpg" from the content, or None for anything else."""
    for file_type, magic in MAGIC_BYTES.items():
        if data.startswith(magic):
            return file_type
    return None


def render_dpi(ocr_provider: str | None = None) -> int:
    """PDF_RENDER_DPI when set; otherwise the DPI the chosen OCR provider expects."""
    default = TROCR_RENDER_DPI if ocr_provider == "trocr" else DEFAULT_RENDER_DPI
    return _get_int_env("PDF_RENDER_DPI") or default


def _open_pdf(data: bytes) -> pymupdf.Document:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several error types for bad files
        raise UnreadableDocumentError(f"The PDF could not be opened: {exc}") from exc
    if document.needs_pass:
        raise UnreadableDocumentError("The PDF is password protected.")
    if document.page_count == 0:
        raise UnreadableDocumentError("The PDF has no pages.")
    return document


def _open_image(data: bytes) -> pymupdf.Pixmap:
    try:
        pixmap = pymupdf.Pixmap(data)
    except Exception as exc:  # noqa: BLE001
        raise UnreadableDocumentError(f"The image could not be read: {exc}") from exc
    if pixmap.colorspace is None or pixmap.colorspace.n > 3:  # for example CMYK JPEG
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    return pixmap


def count_pages(data: bytes, file_type: str) -> int:
    """Open the upload to check it is readable and return its page count."""
    if file_type == "pdf":
        with _open_pdf(data) as document:
            return document.page_count
    _open_image(data)
    return 1


def iter_page_pngs(data: bytes, file_type: str, dpi: int | None = None) -> Iterator[bytes]:
    """Yield one PNG per page, in page order. Pages are produced one at a time to limit memory use."""
    if file_type != "pdf":
        yield _open_image(data).tobytes("png")
        return

    dpi = dpi or render_dpi()
    with _open_pdf(data) as document:
        for page in document:
            yield page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
