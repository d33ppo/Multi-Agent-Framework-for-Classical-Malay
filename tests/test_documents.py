"""
Tests for document storage, upload type detection, page rendering and the OCR
providers. Sample files are generated in memory or in a temporary folder, and
Tesseract is mocked.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pymupdf

from document_storage import LocalStorage
from pdf_render import UnreadableDocumentError, count_pages, detect_file_type, iter_page_pngs, render_dpi
from tests.support import EnvTestCase


def make_pdf(pages=3):
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page(width=200, height=100)
        page.insert_text((20, 50), f"page {number}")
    return document.tobytes()


def make_image(image_format="png"):
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20), False)
    pixmap.clear_with(255)
    return pixmap.tobytes(image_format)


class TempDirTestCase(EnvTestCase):
    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)


class StorageTests(TempDirTestCase):
    def test_save_load_and_exists(self):
        storage = LocalStorage(self.tmp)
        self.assertFalse(storage.exists("documents/a/original.pdf"))
        storage.save("documents/a/original.pdf", b"content")
        self.assertTrue(storage.exists("documents/a/original.pdf"))
        self.assertEqual(storage.load("documents/a/original.pdf"), b"content")

    def test_saves_are_write_once_unless_overwrite_is_requested(self):
        storage = LocalStorage(self.tmp)
        storage.save("runs/r/ocr/ocr_text.txt", b"original ocr")
        with self.assertRaises(FileExistsError):
            storage.save("runs/r/ocr/ocr_text.txt", b"changed")
        self.assertEqual(storage.load("runs/r/ocr/ocr_text.txt"), b"original ocr")

        storage.save("runs/r/run.json", b"1")
        storage.save("runs/r/run.json", b"2", overwrite=True)
        self.assertEqual(storage.load("runs/r/run.json"), b"2")

    def test_keys_cannot_leave_the_storage_folder(self):
        storage = LocalStorage(self.tmp / "root")
        for key in ("../outside.txt", "a/../../outside.txt", str(self.tmp / "outside.txt"), ""):
            with self.subTest(key=key), self.assertRaises(ValueError):
                storage.save(key, b"x")
        self.assertFalse((self.tmp / "outside.txt").exists())


class FileTypeTests(unittest.TestCase):
    def test_type_comes_from_the_content(self):
        self.assertEqual(detect_file_type(make_pdf(1)), "pdf")
        self.assertEqual(detect_file_type(make_image("png")), "png")
        self.assertEqual(detect_file_type(make_image("jpeg")), "jpg")

    def test_other_content_is_not_accepted(self):
        for data in (b"", b"plain text", b"GIF89a....", b"PK\x03\x04 a zip or docx", b"MZ an executable"):
            with self.subTest(data=data):
                self.assertIsNone(detect_file_type(data))


class RenderTests(EnvTestCase):
    def test_multi_page_pdf_renders_one_png_per_page(self):
        pdf = make_pdf(pages=3)
        self.assertEqual(count_pages(pdf, "pdf"), 3)
        pages = list(iter_page_pngs(pdf, "pdf", dpi=72))
        self.assertEqual(len(pages), 3)
        for page in pages:
            self.assertEqual(detect_file_type(page), "png")

    def test_dpi_sets_the_pixel_size(self):
        pdf = make_pdf(pages=1)
        low = pymupdf.Pixmap(next(iter_page_pngs(pdf, "pdf", dpi=72)))
        high = pymupdf.Pixmap(next(iter_page_pngs(pdf, "pdf", dpi=144)))
        self.assertEqual((low.width, low.height), (200, 100))
        self.assertEqual((high.width, high.height), (400, 200))

    def test_dpi_default_and_env(self):
        self.assertEqual(render_dpi(), 300)
        os.environ["PDF_RENDER_DPI"] = "150"
        self.assertEqual(render_dpi(), 150)

    def test_image_upload_is_a_one_page_document(self):
        for image_format, file_type in (("png", "png"), ("jpeg", "jpg")):
            image = make_image(image_format)
            self.assertEqual(count_pages(image, file_type), 1)
            pages = list(iter_page_pngs(image, file_type))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pymupdf.Pixmap(pages[0]).width, 40)

    def test_unreadable_files_raise_a_clear_error(self):
        with self.assertRaises(UnreadableDocumentError):
            count_pages(b"%PDF-1.7 not really a pdf", "pdf")
        with self.assertRaises(UnreadableDocumentError):
            count_pages(b"\x89PNG\r\n\x1a\n not really a png", "png")


if __name__ == "__main__":
    unittest.main()
