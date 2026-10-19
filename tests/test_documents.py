"""
Tests for document storage, upload type detection, page rendering and the OCR
providers. Sample files are generated in memory or in a temporary folder, and
Tesseract is mocked.
"""

import os
import unittest
from unittest import mock

import pymupdf

from document_storage import LocalStorage
from ocr_providers import (
    FixtureOCRProvider,
    OCRLine,
    PageImage,
    TesseractOCRProvider,
    build_ocr_provider,
    safe_document_name,
)
from pdf_render import UnreadableDocumentError, count_pages, detect_file_type, iter_page_pngs, render_dpi
from tests.support import EnvTestCase, TempDirTestCase, make_image, make_pdf

try:
    import ocr_jawi
except ImportError:  # OCR extras (pytesseract, opencv) not installed
    ocr_jawi = None


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


def page(name="article", number=1, count=1, png=b""):
    return PageImage(document_name=name, page_number=number, page_count=count, png=png)


class FixtureProviderTests(TempDirTestCase):
    def write(self, name, text):
        (self.tmp / name).write_text(text, encoding="utf-8")

    def test_reads_one_file_per_page(self):
        self.write("article_1.txt", "ايبو ڤد\n\n  ماس دهولو  \n")
        self.write("article_p2.txt", "هيدوڤ")
        provider = FixtureOCRProvider(self.tmp)
        first = provider.recognise_page(page(number=1, count=2))
        self.assertEqual([line.text for line in first], ["ايبو ڤد", "ماس دهولو"])
        self.assertEqual((first[0].box, first[0].confidence), (None, None))
        self.assertEqual([line.text for line in provider.recognise_page(page(number=2, count=2))], ["هيدوڤ"])

    def test_one_page_upload_may_use_the_plain_file_name(self):
        self.write("scan.txt", "ايبو")
        provider = FixtureOCRProvider(self.tmp)
        self.assertEqual(provider.recognise_page(page(name="scan"))[0].text, "ايبو")
        with self.assertRaises(FileNotFoundError):  # the plain name is not used for a multi-page upload
            provider.recognise_page(page(name="scan", number=1, count=2))

    def test_saved_ocr_output_with_metadata_block_is_accepted(self):
        self.write("scan.txt", "=== Jawi OCR Result ===\n\nايبو ڤد\n\n=== Metadata ===\nConfidence   : 61.2%\n")
        lines = FixtureOCRProvider(self.tmp).recognise_page(page(name="scan"))
        self.assertEqual([line.text for line in lines], ["ايبو ڤد"])

    def test_missing_fixture_names_the_files_it_looked_for(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            FixtureOCRProvider(self.tmp).recognise_page(page(name="unknown", number=3, count=4))
        self.assertIn("unknown_3.txt", str(ctx.exception))

    def test_document_name_is_reduced_to_safe_characters(self):
        self.assertEqual(safe_document_name("1950_07_002.pdf"), "1950_07_002")
        self.assertEqual(safe_document_name("..\\..\\secret/../My Scan (1).PDF"), "My_Scan_1")
        self.assertEqual(safe_document_name("ڤد.png"), "document")
        self.assertEqual(safe_document_name(None), "document")


class ProviderSelectionTests(EnvTestCase):
    def test_provider_comes_from_the_environment(self):
        self.assertIsInstance(build_ocr_provider(), TesseractOCRProvider)
        os.environ["OCR_PROVIDER"] = "Fixture"
        self.assertIsInstance(build_ocr_provider(), FixtureOCRProvider)

    def test_unknown_provider_is_rejected(self):
        os.environ["OCR_PROVIDER"] = "magic"
        with self.assertRaises(ValueError) as ctx:
            build_ocr_provider()
        self.assertIn("fixture, tesseract", str(ctx.exception))


@unittest.skipUnless(ocr_jawi, "OCR dependencies not installed")
class TesseractProviderTests(EnvTestCase):
    # Two words on line 1, one word on line 2, plus the layout rows Tesseract adds (confidence -1).
    TESSERACT_ROWS = {
        "text": ["", "ايبو", "ڤد", "", "ماس"],
        "conf": ["-1", "90", "70.0", "-1", 50],
        "block_num": [1, 1, 1, 1, 1],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 2, 2],
        "left": [0, 120, 40, 0, 100],
        "top": [0, 10, 12, 0, 60],
        "width": [200, 60, 50, 200, 80],
        "height": [100, 30, 26, 100, 30],
    }

    def test_words_are_grouped_into_lines_with_boxes_and_confidence(self):
        lines = ocr_jawi.group_words_into_lines(self.TESSERACT_ROWS)
        self.assertEqual(
            lines,
            [
                {"text": "ايبو ڤد", "box": (40, 10, 140, 30), "confidence": 0.8},
                {"text": "ماس", "box": (100, 60, 80, 30), "confidence": 0.5},
            ],
        )

    def test_provider_returns_lines_without_running_tesseract(self):
        self.patch("ocr_jawi.resolve_tesseract_cmd", lambda: "tesseract")
        run = self.patch("ocr_jawi.pytesseract.image_to_data", mock.Mock(return_value=self.TESSERACT_ROWS))
        lines = TesseractOCRProvider().recognise_page(page(png=make_image("png")))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(lines[0], OCRLine(text="ايبو ڤد", box=(40, 10, 140, 30), confidence=0.8))
        self.assertEqual(lines[0].to_dict()["confidence"], 0.8)

    def test_undecodable_page_image_is_an_error(self):
        with self.assertRaises(ValueError):
            TesseractOCRProvider().recognise_page(page(png=b"not an image"))


if __name__ == "__main__":
    unittest.main()
