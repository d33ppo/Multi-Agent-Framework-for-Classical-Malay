"""
Tests for the TrOCR OCR provider, its preprocessing and the page-level OCR metrics.

The YOLO detector and the TrOCR recogniser are fakes, and the one test that runs
real generation builds a tiny random model in memory. No model is downloaded and
nothing touches the network.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import ocr_trocr
import stage_metrics
from jawi_dictionary import JawiRumiDictionary
from ocr_providers import FixtureOCRProvider, PageImage, TesseractOCRProvider, build_ocr_provider
from ocr_trocr import TrOCROCRProvider, TrOCRRecogniser, line_crop, mean_logprob_confidence, normalise_output
from pdf_render import render_dpi
from tests.support import TempDirTestCase, make_pdf
from tests.test_upload_jobs import UploadTestCase

td, yp = ocr_trocr.td, ocr_trocr.yp
PAGE_W, PAGE_H = 1100, 800


def load_eval_script():
    path = ocr_trocr.PROJECT_DIR / "scripts" / "eval_page_cer.py"
    spec = importlib.util.spec_from_file_location("eval_page_cer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def page_png(fill=(255, 255, 255), width=PAGE_W, height=PAGE_H, patches=()):
    """A BGR page encoded as PNG. `patches` are (x1, y1, x2, y2, bgr) rectangles."""
    image = np.full((height, width, 3), fill, dtype=np.uint8)
    for x1, y1, x2, y2, colour in patches:
        image[y1:y2, x1:x2] = colour
    return cv2.imencode(".png", image)[1].tobytes()


def page(png):
    return PageImage("article", 1, 1, png)


class _Array:
    """Stands in for a torch tensor: .cpu().numpy()."""

    def __init__(self, values, shape):
        self.values = np.asarray(values, dtype=np.float32).reshape(shape)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeYOLO:
    """Returns fixed boxes (x1, y1, x2, y2, conf) the way ultralytics does."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, image, conf, imgsz, verbose):
        self.calls.append({"conf": conf, "imgsz": imgsz, "image": image})
        boxes = SimpleNamespace(xyxy=_Array([b[:4] for b in self.boxes], (-1, 4)),
                                conf=_Array([b[4] for b in self.boxes], (-1,)))
        return [SimpleNamespace(boxes=boxes)]


class FakeRecogniser:
    """Answers line N with its own text, or with the given (text, confidence) pairs."""

    def __init__(self, answers=None):
        self.answers = answers
        self.crops = []

    def __call__(self, crops):
        self.crops.extend(crops)
        if self.answers is not None:
            return self.answers[: len(crops)]
        return [(f"line {i}", 0.9) for i in range(1, len(crops) + 1)]


def provider_with(boxes, answers=None, detector=None, **kwargs):
    detector, recogniser = detector or FakeYOLO(boxes), FakeRecogniser(answers)
    return TrOCROCRProvider(detector=detector, recogniser=recogniser, **kwargs), detector, recogniser


def column(x1, x2, ys, height=30, conf=0.9):
    return [(x1, y, x2, y + height, conf) for y in ys]


class ReadingOrderTests(TempDirTestCase):
    """yolo_prefill.reading_order on synthetic Qalam-like layouts."""

    ROWS = [100, 150, 200, 250]

    def order(self, named):
        boxes = [(*box, name) for box, name in named]
        return [b[-1] for b in yp.reading_order(boxes, PAGE_W)]

    def columns(self):
        left = [((100, y, 540, y + 30), f"L{i}") for i, y in enumerate(self.ROWS, 1)]
        right = [((560, y, 1000, y + 30), f"R{i}") for i, y in enumerate(self.ROWS, 1)]
        return left + right

    def test_two_columns_are_read_right_column_first(self):
        self.assertEqual(self.order(self.columns()), ["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"])

    def test_headline_band_across_columns_comes_first(self):
        named = [((100, 20, 1000, 70), "H")] + self.columns()
        self.assertEqual(self.order(named), ["H", "R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4"])

    def test_pieces_of_one_printed_line_are_read_right_to_left(self):
        named = [((100, 100, 480, 130), "piece 2"), ((520, 100, 1000, 130), "piece 1"),
                 ((100, 150, 1000, 180), "line 2"), ((100, 200, 1000, 230), "line 3")]
        self.assertEqual(self.order(named), ["piece 1", "piece 2", "line 2", "line 3"])

    def test_adverts_below_the_text_stay_in_their_columns(self):
        # No heading spans the gutter at the gap above the adverts, so there is no band cut:
        # each advert is read at the foot of its own column. The text lines keep their order.
        named = self.columns() + [((100, 700, 500, 760), "advert L"), ((600, 700, 1000, 760), "advert R")]
        self.assertEqual(self.order(named),
                         ["R1", "R2", "R3", "R4", "advert R", "L1", "L2", "L3", "L4", "advert L"])

    def test_photo_opposite_a_section_break_does_not_split_the_column(self):
        # Right column: text, a section break, more text. Left column: text, a photo (no boxes)
        # at the height of the break, more text. The empty band crosses the page, but no
        # heading spans the gutter there, so each column is read whole.
        right = [((560, y, 1000, y + 30), f"R{i}") for i, y in enumerate([100, 150, 200, 250, 400, 450, 500], 1)]
        left = [((100, y, 540, y + 30), f"L{i}") for i, y in enumerate([100, 150, 400, 450], 1)]
        self.assertEqual(self.order(left + right),
                         ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "L1", "L2", "L3", "L4"])

    def test_heading_across_the_gutter_cuts_the_page_into_bands(self):
        # The same gap, but a full-width heading sits in it: the columns above the heading
        # are read first, then the heading, then the columns below.
        right = [((560, y, 1000, y + 30), f"R{i}") for i, y in enumerate([100, 150, 200, 250, 430, 480, 530], 1)]
        left = [((100, y, 540, y + 30), f"L{i}") for i, y in enumerate([100, 150, 200, 250, 430, 480], 1)]
        named = left + right + [((100, 330, 1000, 380), "H")]
        self.assertEqual(self.order(named), ["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4", "H",
                                             "R5", "R6", "R7", "L5", "L6"])

    def test_page_number_below_everything_is_read_last(self):
        # A page number just right of the gutter, under the columns: XY-cut alone puts it at the foot
        # of the right column, before the left column.
        named = [((552, 600, 592, 630), "page")] + self.columns()
        self.assertEqual(self.order(named), ["R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4", "page"])

    def test_a_wide_box_at_the_bottom_is_not_a_page_number(self):
        named = self.columns() + [((560, 600, 1000, 630), "foot R"), ((100, 650, 540, 680), "foot L")]
        self.assertIsNone(yp._page_number([box for box, _ in named]))
        self.assertEqual(self.order(named), ["R1", "R2", "R3", "R4", "foot R", "L1", "L2", "L3", "L4", "foot L"])
        # A short box that is not below everything is not a page number either.
        self.assertIsNone(yp._page_number([box for box, _ in self.columns()] + [(530, 200, 570, 230)]))

    def test_tall_headline_at_the_top_of_the_left_column_is_read_first(self):
        # A masthead (over 1.8 line heights tall) above the left column: XY-cut alone reads the
        # whole right column before it.
        left = [((100, y, 540, y + 30), f"L{i}") for i, y in enumerate([200, 250, 300, 350], 1)]
        right = [((560, y, 1000, y + 30), f"R{i}") for i, y in enumerate([100, 150, 200, 250, 300, 350], 1)]
        named = left + right + [((150, 100, 500, 170), "masthead")]
        self.assertEqual(self.order(named),
                         ["masthead", "R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3", "L4"])

    def test_gutter_spanning_title_in_the_top_band_is_read_first(self):
        named = self.columns() + [((150, 40, 950, 75), "title")]
        self.assertEqual(self.order(named)[0], "title")

    def test_a_heading_with_text_above_it_stays_in_place(self):
        # A full-width heading in the top band with column lines above it is not "before the columns".
        ys = [100, 150, 300, 350, 400, 450, 500, 550]
        right = [((560, y, 1000, y + 30), f"R{i}") for i, y in enumerate(ys, 1)]
        left = [((100, y, 540, y + 30), f"L{i}") for i, y in enumerate(ys, 1)]
        named = left + right + [((100, 210, 1000, 260), "H")]
        self.assertEqual(yp._headlines([box for box, _ in named], 8.8), set())
        self.assertEqual(self.order(named)[:5], ["R1", "R2", "L1", "L2", "H"])

    def test_a_small_box_in_the_gutter_is_not_a_heading(self):
        # A page number in the gutter crosses it but does not reach into the columns.
        self.assertTrue(yp._spans_gutter((100, 0, 1000, 30), (541, 560), 100, 1000))
        self.assertFalse(yp._spans_gutter((520, 0, 580, 30), (541, 560), 100, 1000))
        self.assertFalse(yp._spans_gutter((560, 0, 1000, 30), (541, 560), 100, 1000))


class CropTests(TempDirTestCase):
    def test_crop_margin_is_0_12_of_the_box_height_from_the_full_page(self):
        image = np.random.default_rng(0).integers(0, 255, (PAGE_H, PAGE_W, 3), dtype=np.uint8)
        box = (200.0, 300.0, 700.0, 340.0)  # 40 px high, so the margin is 4.8 px
        crop = line_crop(image, box)
        self.assertEqual(td.EVAL_MARGIN_FRAC, 0.12)
        np.testing.assert_array_equal(crop, image[int(300 - 4.8):int(340 + 4.8), int(200 - 4.8):int(700 + 4.8)])
        np.testing.assert_array_equal(
            crop, td.crop_for(image, {"bx1": 200.0, "by1": 300.0, "bx2": 700.0, "by2": 340.0}, train=False))

    def test_crop_is_clamped_at_the_page_edge(self):
        image = np.zeros((PAGE_H, PAGE_W, 3), dtype=np.uint8)
        self.assertEqual(line_crop(image, (0.0, 2.0, 50.0, 52.0)).shape, (int(52 + 6), int(50 + 6), 3))

    def test_provider_hands_the_recogniser_rgb_crops_of_each_box(self):
        blue_bgr = (255, 0, 0)
        png = page_png(patches=[(200, 300, 700, 340, blue_bgr)])
        provider, detector, recogniser = provider_with([(200.0, 300.0, 700.0, 340.0, 0.9)])
        provider.recognise_page(page(png))

        self.assertEqual(recogniser.crops[0].shape, (344 - 295, 704 - 195, 3))
        self.assertEqual(tuple(recogniser.crops[0][20, 100]), (0, 0, 255))  # RGB
        self.assertEqual(tuple(detector.calls[0]["image"][320, 400]), blue_bgr)  # the detector gets BGR


class ProviderTests(TempDirTestCase):
    def test_page_with_zero_boxes_gives_no_lines_and_skips_the_recogniser(self):
        provider, _detector, recogniser = provider_with([])
        self.assertEqual(provider.recognise_page(page(page_png())), [])
        self.assertEqual(recogniser.crops, [])

    def test_detector_uses_the_training_settings(self):
        provider, detector, _ = provider_with([])
        provider.recognise_page(page(page_png()))
        self.assertEqual((detector.calls[0]["conf"], detector.calls[0]["imgsz"]), (0.35, 960))

        os.environ["LINE_DETECTOR_IMGSZ"] = "1280"
        provider, detector, _ = provider_with([])
        provider.recognise_page(page(page_png()))
        self.assertEqual(detector.calls[0]["imgsz"], 1280)

    def test_lines_come_back_in_reading_order_with_box_and_confidence(self):
        boxes = column(100, 540, [100, 150]) + column(560, 1000, [100, 150])
        provider, _, _ = provider_with(boxes)
        lines = provider.recognise_page(page(page_png()))

        self.assertEqual([line.order for line in lines], [1, 2, 3, 4])
        self.assertEqual(lines[0].box, (560, 100, 440, 30))  # right column first, left/top/width/height
        self.assertEqual(lines[2].box, (100, 100, 440, 30))
        self.assertEqual(lines[0].to_dict(), {"text": "line 1", "box": (560, 100, 440, 30), "confidence": 0.9,
                                              "order": 1, "low_confidence": False})

    def test_fragment_inside_a_line_box_is_suppressed(self):
        boxes = [(100, 100, 1000, 130, 0.95), (400, 102, 600, 128, 0.5)]
        provider, _, recogniser = provider_with(boxes)
        self.assertEqual(len(provider.recognise_page(page(page_png()))), 1)
        self.assertEqual(len(recogniser.crops), 1)

    def test_low_confidence_flag_follows_the_threshold(self):
        boxes = column(100, 1000, [100, 150, 200])
        answers = [("ا", 0.95), ("ب", 0.55), ("ت", None)]
        provider, _, _ = provider_with(boxes, answers)
        lines = provider.recognise_page(page(page_png()))
        self.assertEqual([line.low_confidence for line in lines], [False, True, None])  # default 0.60

        os.environ["OCR_CONFIDENCE_THRESHOLD"] = "0.99"
        provider, _, _ = provider_with(boxes, answers)
        self.assertEqual([line.low_confidence for line in provider.recognise_page(page(page_png()))],
                         [True, True, None])

    def test_invalid_threshold_is_rejected(self):
        os.environ["OCR_CONFIDENCE_THRESHOLD"] = "high"
        with self.assertRaises(ValueError):
            provider_with([])

    def test_empty_recognised_lines_are_left_out_of_the_page_but_kept_for_evaluation(self):
        boxes = column(100, 1000, [100, 150])
        provider, _, _ = provider_with(boxes, [("", 0.2), ("ب", 0.9)])
        self.assertEqual([line.text for line in provider.recognise_page(page(page_png()))], ["ب"])
        self.assertEqual([line.text for line in provider.read_page(page_png())], ["", "ب"])

    def test_box_padding_is_off_by_default(self):
        provider, _, recogniser = provider_with(column(300, 700, [100]))
        self.assertEqual((provider.pad_x, provider.pad_y), (0.0, 0.0))
        provider.read_page(page_png())
        self.assertEqual(recogniser.crops[0].shape,
                         line_crop(np.zeros((PAGE_H, PAGE_W, 3)), (300, 100, 700, 130)).shape)

    def test_box_padding_widens_only_the_crop(self):
        # A 30 px high box: pad_x 0.2 adds 6 px left and right, pad_y 0.1 adds 3 px top and bottom.
        os.environ["OCR_BOX_PAD_X"], os.environ["OCR_BOX_PAD_Y"] = "0.2", "0.1"
        provider, _, recogniser = provider_with(column(300, 700, [100]))
        self.assertEqual((provider.pad_x, provider.pad_y), (0.2, 0.1))
        lines = provider.read_page(page_png())
        expected = line_crop(np.zeros((PAGE_H, PAGE_W, 3)), (294, 97, 706, 133))
        self.assertEqual(recogniser.crops[0].shape, expected.shape)
        self.assertEqual(lines[0].xyxy, (300.0, 100.0, 700.0, 130.0))  # the line keeps the detected box
        self.assertAlmostEqual(lines[0].detector_confidence, 0.9, places=5)

    def test_given_boxes_are_not_padded(self):
        provider, _, recogniser = provider_with([], pad_x=0.3, pad_y=0.05, detector=RaisingDetector())
        provider.read_boxes(page_png(), [(300, 100, 700, 130)])
        self.assertEqual(recogniser.crops[0].shape,
                         line_crop(np.zeros((PAGE_H, PAGE_W, 3)), (300, 100, 700, 130)).shape)

    def test_pad_box_clips_to_the_page_and_keeps_extra_items(self):
        self.assertEqual(ocr_trocr.pad_box((5, 2, 95, 22, 0.8), 0.5, 0.5, 100, 30), (0.0, 0.0, 100.0, 30.0, 0.8))
        self.assertEqual(ocr_trocr.pad_box((10, 10, 50, 30, 0.8), 0.25, 0.0, 100, 100), (5.0, 10.0, 55.0, 30.0, 0.8))
        box = (10, 10, 50, 30, 0.8)
        self.assertIs(ocr_trocr.pad_box(box, 0.0, 0.0, 100, 100), box)

    def test_invalid_box_padding_is_rejected(self):
        for value in ("wide", "-0.1", "nan", "inf"):
            os.environ["OCR_BOX_PAD_X"] = value
            with self.assertRaises(ValueError, msg=value):
                provider_with([])
        del os.environ["OCR_BOX_PAD_X"]
        with self.assertRaises(ValueError):
            provider_with([], pad_y=-0.05)

    def test_undecodable_page_raises(self):
        provider, _, _ = provider_with([])
        with self.assertRaises(ValueError):
            provider.recognise_page(page(b"not an image"))


class NormalisationAndConfidenceTests(TempDirTestCase):
    def test_output_gets_nfc_and_whitespace_collapse_but_not_the_label_map(self):
        decomposed = "أ"  # alef + hamza above, NFC gives U+0623
        self.assertEqual(normalise_output(f"  ڤد  {decomposed}\n ک "), "ڤد أ ک")  # ڤ and ک stay as printed

    def test_confidence_stops_at_the_first_eos(self):
        logprobs = [np.log(0.5), np.log(0.5), np.log(1.0), -50.0, -50.0]  # padding after EOS has junk scores
        self.assertAlmostEqual(mean_logprob_confidence([5, 6, 2, 1, 1], logprobs, 2), round(np.exp(np.mean(logprobs[:3])), 4))
        self.assertAlmostEqual(mean_logprob_confidence([5, 6], logprobs[:2], [2]), 0.5)
        self.assertIsNone(mean_logprob_confidence([], [], 2))


class TinyModelTests(TempDirTestCase):
    """Real generate() and score handling on a tiny random VisionEncoderDecoder built in memory."""

    @staticmethod
    def tiny_model():
        import torch
        from transformers import TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

        encoder = ViTConfig(image_size=32, patch_size=16, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)
        decoder = TrOCRConfig(vocab_size=12, d_model=16, decoder_layers=1, decoder_attention_heads=2,
                              decoder_ffn_dim=32, max_position_embeddings=160, pad_token_id=1,
                              bos_token_id=0, eos_token_id=2, decoder_start_token_id=0, use_cache=False)
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
        config.decoder_start_token_id, config.pad_token_id, config.eos_token_id = 0, 1, 2
        torch.manual_seed(0)
        model = VisionEncoderDecoderModel(config)
        model.generation_config.decoder_start_token_id = 0
        model.generation_config.pad_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.use_cache = False
        return model

    class Processor:
        def __init__(self):
            self.batches = []

        def __call__(self, images, return_tensors):
            import torch

            self.batches.append(len(images))
            return SimpleNamespace(pixel_values=torch.rand(len(images), 3, 32, 32, generator=torch.Generator().manual_seed(1)))

        def batch_decode(self, sequences, skip_special_tokens):
            return [" ".join(str(int(t)) for t in seq if int(t) > 2) + "   " for seq in sequences]

    def run_recogniser(self, num_beams):
        processor = self.Processor()
        recogniser = TrOCRRecogniser(self.tiny_model(), processor, num_beams=num_beams, batch_size=2)
        crops = [np.full((20, 60, 3), 200, dtype=np.uint8) for _ in range(5)]
        return recogniser, processor, recogniser(crops)

    def test_greedy_decoding_returns_text_and_confidence_per_crop(self):
        recogniser, processor, results = self.run_recogniser(num_beams=1)
        self.assertTrue(recogniser.model.config.decoder.use_cache)
        self.assertTrue(recogniser.model.generation_config.use_cache)
        self.assertEqual(processor.batches, [2, 2, 1])
        self.assertEqual(len(results), 5)
        for text, confidence in results:
            self.assertEqual(text, text.strip())
            self.assertGreater(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_beam_decoding_uses_the_sequence_scores(self):
        _, _, results = self.run_recogniser(num_beams=2)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(0.0 < confidence <= 1.0 for _, confidence in results))


class ProviderSelectionTests(TempDirTestCase):
    def make_models(self):
        model_dir = self.tmp / "trocr_v1"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        detector = self.tmp / "lines.pt"
        detector.write_bytes(b"")
        return model_dir, detector

    def test_trocr_is_the_default_once_the_model_path_is_set(self):
        self.assertIsInstance(build_ocr_provider(), TesseractOCRProvider)
        model_dir, detector = self.make_models()
        os.environ.update(OCR_MODEL_PATH=str(model_dir), LINE_DETECTOR_PATH=str(detector))
        provider = build_ocr_provider()
        self.assertIsInstance(provider, TrOCROCRProvider)
        self.assertIsNone(provider._recogniser)  # nothing is loaded until the first page

    def test_ocr_provider_setting_still_wins(self):
        model_dir, detector = self.make_models()
        os.environ.update(OCR_MODEL_PATH=str(model_dir), LINE_DETECTOR_PATH=str(detector), OCR_PROVIDER="fixture")
        self.assertIsInstance(build_ocr_provider(), FixtureOCRProvider)

    def test_missing_model_or_detector_is_a_configuration_error(self):
        model_dir, detector = self.make_models()
        os.environ.update(OCR_MODEL_PATH=str(self.tmp / "missing"), LINE_DETECTOR_PATH=str(detector))
        with self.assertRaisesRegex(ValueError, "TrOCR model not found"):
            build_ocr_provider()
        os.environ.update(OCR_MODEL_PATH=str(model_dir), LINE_DETECTOR_PATH=str(self.tmp / "missing.pt"))
        with self.assertRaisesRegex(ValueError, "Line detector not found"):
            build_ocr_provider()
        os.environ.update(OCR_PROVIDER="trocr", OCR_MODEL_PATH="")
        with self.assertRaisesRegex(ValueError, "TrOCR model not found"):
            build_ocr_provider()


class UploadWithTrOCRTests(UploadTestCase):
    def test_upload_runs_with_the_trocr_provider_and_counts_low_confidence_lines(self):
        provider, _, _ = provider_with(column(100, 1000, [100, 150]), [("ايبو ڤد ايبو ڤد ايبو", 0.9),
                                                                       ("ڤد ايبو ڤد ايبو ڤد", 0.3)])
        self.patch("backend.main.build_ocr_provider", lambda: provider)
        run = self.finished_run(self.upload(page_png(), name="page.png"))

        self.assertEqual(run["ocr_provider"], "trocr")
        self.assertEqual(run["stages"]["ocr"]["low_confidence_lines"], 1)
        self.assertEqual(run["outputs"]["ocr_text"], "ايبو ڤد ايبو ڤد ايبو\nڤد ايبو ڤد ايبو ڤد")
        lines = run["outputs"]["ocr_pages"][0]["lines"]
        self.assertEqual([(line["order"], line["low_confidence"]) for line in lines], [(1, False), (2, True)])
        self.assertEqual(run["status"], "completed")  # the uncalibrated flag does not change the run status

    def test_pdf_upload_is_rendered_at_400_dpi_for_trocr(self):
        # TrOCR writes the 1950 print's ف for /p/, never ڤ; Method B must still read it.
        provider, detector, _ = provider_with(column(100, 1000, [100, 150]), [("ايبو فد ايبو فد ايبو", 0.9),
                                                                              ("فد ايبو فد ايبو فد", 0.9)])
        self.patch("backend.main.build_ocr_provider", lambda: provider)
        run = self.finished_run(self.upload(make_pdf(pages=1)))

        self.assertEqual(run["stages"]["render"]["dpi"], 400)
        self.assertEqual(run["stages"]["ocr"]["render_dpi"], 400)
        height, width = detector.calls[0]["image"].shape[:2]
        self.assertAlmostEqual(width, 200 * 400 / 72, delta=1)  # a 200 x 100 pt page at 400 DPI
        self.assertAlmostEqual(height, 100 * 400 / 72, delta=1)
        self.assertNotIn("ڤ", run["outputs"]["ocr_text"])
        self.assertEqual(run["outputs"]["romanisation"]["text"], "ibu pada ibu pada ibu\npada ibu pada ibu pada")
        self.assertEqual(run["status"], "completed")

    def test_pdf_render_dpi_setting_still_wins_and_images_keep_their_resolution(self):
        os.environ["PDF_RENDER_DPI"] = "300"
        provider, _, _ = provider_with(column(100, 1000, [100]), [("ايبو فد ايبو فد ايبو", 0.9)])
        self.patch("backend.main.build_ocr_provider", lambda: provider)
        run = self.finished_run(self.upload(make_pdf(pages=1)))
        self.assertEqual((run["stages"]["render"]["dpi"], run["stages"]["ocr"]["render_dpi"]), (300, 300))

        run = self.finished_run(self.upload(page_png(), name="page.png"))
        self.assertEqual((run["stages"]["render"]["dpi"], run["stages"]["ocr"]["render_dpi"]), (None, None))


class RenderDpiTests(TempDirTestCase):
    def test_trocr_pages_are_rendered_at_the_training_dpi_unless_configured(self):
        self.assertEqual((render_dpi(), render_dpi("tesseract"), render_dpi("trocr")), (300, 300, 400))
        os.environ["PDF_RENDER_DPI"] = "250"
        self.assertEqual(render_dpi("trocr"), 250)


class TrOCRToMethodBTests(TempDirTestCase):
    """TrOCR output keeps the 1950 spelling (ف for /p/), which modern-Jawi Method B data writes as ڤ."""

    def test_trocr_style_jawi_romanises_through_the_old_spelling_retry(self):
        dictionary = JawiRumiDictionary({"ڤد": {"pada": 9000}, "هيدوڤ": {"hidup": 600}, "ايبو": {"ibu": 500}})
        provider, _, _ = provider_with(column(100, 1000, [100, 150]), [(normalise_output("ايبو  فد\n"), 0.9),
                                                                       (normalise_output("هيدوف"), 0.9)])
        lines = provider.recognise_page(page(page_png()))
        ocr_text = "\n".join(line.text for line in lines)
        self.assertEqual(ocr_text, "ايبو فد\nهيدوف")
        self.assertNotIn("ڤ", ocr_text)

        output = dictionary.romanise(ocr_text)
        self.assertEqual(output.text, "ibu pada\nhidup")
        rules = {word.jawi: (word.rumi, word.status, word.rule) for word in output.words}
        self.assertEqual(rules["فد"], ("pada", "known", "fa_to_pa"))
        self.assertEqual(rules["هيدوف"], ("hidup", "known", "fa_to_pa"))
        self.assertEqual(rules["ايبو"], ("ibu", "known", "exact"))
        self.assertEqual(output.unknown_words, [])


class PageMetricTests(TempDirTestCase):
    def test_matched_line_cer_counts_misses_and_extras_as_full_edits(self):
        gold = [((0, 0, 100, 20), "abcd"), ((0, 30, 100, 50), "efgh"), ((0, 60, 100, 80), "ijkl")]
        pred = [((1, 0, 100, 20, 0.9), "abXd"),     # matched, 1 edit
                ((0, 31, 100, 50, 0.8), "efgh"),    # matched, 0 edits
                ((0, 200, 100, 220, 0.7), "zz")]    # extra, 2 insertions; "ijkl" missed, 4 deletions
        result = stage_metrics.matched_line_cer(pred, gold)
        self.assertEqual((result["matched_edits"], result["missed_chars"], result["extra_chars"]), (1, 4, 2))
        self.assertEqual((result["edits"], result["gold_chars"]), (7, 12))
        self.assertAlmostEqual(result["cer"], 7 / 12, places=6)
        self.assertAlmostEqual(result["precision"], 2 / 3, places=6)
        self.assertAlmostEqual(result["recall"], 2 / 3, places=6)

    def test_perfect_page_and_empty_prediction(self):
        gold = [((0, 0, 100, 20), "abcd")]
        self.assertEqual(stage_metrics.matched_line_cer([((0, 0, 100, 20, 1.0), "abcd")], gold)["cer"], 0.0)
        empty = stage_metrics.matched_line_cer([], gold)
        self.assertEqual((empty["cer"], empty["recall"], empty["precision"]), (1.0, 0.0, None))

    def test_matched_line_cer_ignores_order_but_page_cer_does_not(self):
        gold = [((0, 0, 100, 20), "abcd"), ((0, 30, 100, 50), "efgh")]
        pred = [((0, 30, 100, 50, 0.9), "efgh"), ((0, 0, 100, 20, 0.9), "abcd")]
        self.assertEqual(stage_metrics.matched_line_cer(pred, gold)["cer"], 0.0)
        self.assertGreater(stage_metrics.page_cer("efgh\nabcd", "abcd\nefgh")["cer"], 0.0)

    def test_kendall_tau_of_the_reading_order(self):
        self.assertEqual(stage_metrics.kendall_tau([0, 1, 2, 3])["kendall_tau"], 1.0)
        self.assertEqual(stage_metrics.kendall_tau([3, 2, 1, 0])["kendall_tau"], -1.0)
        swapped = stage_metrics.kendall_tau([0, 2, 1, 3])
        self.assertEqual((swapped["concordant"], swapped["discordant"]), (5, 1))
        self.assertAlmostEqual(swapped["kendall_tau"], 4 / 6, places=6)
        self.assertIsNone(stage_metrics.kendall_tau([0])["kendall_tau"])

    def test_reading_order_agreement_uses_matched_lines_and_the_gold_order(self):
        gold = [((0, y, 100, y + 20), text) for y, text in ((0, "a"), (30, "b"), (60, "c"), (90, "d"))]
        pred = [((0, 60, 100, 80, 0.9), "c"), ((0, 0, 100, 20, 0.9), "a"), ((0, 30, 100, 50, 0.9), "b"),
                ((0, 500, 100, 520, 0.9), "extra")]  # "d" is missed, the extra line is ignored
        export_order = stage_metrics.reading_order_agreement(pred, gold)
        self.assertEqual((export_order["matched_lines"], export_order["concordant"], export_order["discordant"]),
                         (3, 1, 2))
        self.assertAlmostEqual(export_order["kendall_tau"], -1 / 3, places=6)
        # The true order is c, a, b, d: the prediction follows it exactly.
        self.assertEqual(stage_metrics.reading_order_agreement(pred, gold, [2, 0, 1, 3])["kendall_tau"], 1.0)

    def test_one_to_one_matching_agrees_with_the_detector_check(self):
        rng = np.random.default_rng(3)
        gold = [(x, y, x + 300, y + 30) for x in (100, 500) for y in range(50, 700, 45)]
        pred = [(x + rng.uniform(-20, 20), y + rng.uniform(-10, 10), x2 + rng.uniform(-20, 20), y2, rng.uniform())
                for x, y, x2, y2 in gold[::2]] + [(900, 900, 1000, 930, 0.99)]
        pairs = stage_metrics.match_lines(pred, gold)
        self.assertEqual(len(pairs), yp.match_stats(pred, gold)["tp"])
        self.assertEqual(len({j for _, j in pairs}), len(pairs))


class EvalScriptTests(TempDirTestCase):
    def write_export(self):
        def result(box_id, box, text):
            return [{"id": box_id, "type": "rectanglelabels", "value": dict(zip(("x", "y", "width", "height"), box))},
                    {"id": box_id, "type": "textarea", "value": {"text": [text]}}]

        def task(task_id, data, annotations):
            return {"id": task_id, "data": data, "annotations": annotations}

        export = [
            # Task 1 has no metadata; the override maps it to 1950_11_002 page 0.
            task(1, {"image": ""}, [{"updated_at": "2026-01-01", "result": result("a", (10, 10, 50, 5), "ڤد")}]),
            # The same test page twice: the later annotation wins.
            task(2, {"article_id": "1950_11_009", "page_index": 0, "image": ""},
                 [{"updated_at": "2026-01-01", "result": result("b", (10, 10, 50, 5), "old")}]),
            task(3, {"article_id": "1950_11_009", "page_index": 0, "image": ""},
                 [{"updated_at": "2026-02-01",
                   "result": result("c", (10, 10, 50, 5), " ايبو   ڤد ") + result("d", (10, 30, 0.1, 5), "tiny")}]),
        ]
        path = self.tmp / "export.json"
        path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
        return path

    def test_ground_truth_uses_the_override_latest_annotation_and_label_normalisation(self):
        script = load_eval_script()
        export = self.write_export()

        pages = script.load_test_pages(export, ["1950_11_009", "1950_11_002"])
        self.assertEqual([(p["article_id"], p["page_index"]) for p in pages], [("1950_11_002", 0), ("1950_11_009", 0)])
        self.assertEqual(script.gold_lines(pages[0], 1000, 2000), [((100.0, 200.0, 600.0, 300.0), "فد")])
        # Latest annotation, NFC + LABEL_MAP + whitespace, and the 1 px wide box is dropped like the dataset builder.
        self.assertEqual([text for _, text in script.gold_lines(pages[1], 1000, 2000)], ["ايبو فد"])
        self.assertEqual(script.load_test_pages(export, ["1950_11_009"])[0]["article_id"], "1950_11_009")

    def test_val_split_uses_the_training_rule(self):
        script = load_eval_script()
        export = self.write_export()  # articles 1950_11_002 (via the override) and 1950_11_009
        self.assertEqual(script.split_article_ids(export, ["1950_11_009"], "test"), ["1950_11_009"])
        # One non-test article left: split_articles always keeps at least one validation article.
        self.assertEqual(script.split_article_ids(export, ["1950_11_009"], "val"), ["1950_11_002"])
        # The same call the dataset builder makes, so the validation articles match training.
        with mock.patch.object(td, "split_articles", wraps=td.split_articles) as split:
            script.split_article_ids(export, ["1950_11_009"], "val")
        split.assert_called_once_with({"1950_11_002", "1950_11_009"}, ["1950_11_009"])
        with self.assertRaises(ValueError):
            script.split_article_ids(export, ["1950_11_009"], "train")

    def test_page_evaluation_with_fake_models(self):
        script = load_eval_script()
        gold = [((100.0, 100.0, 1000.0, 130.0), "ايبو"), ((100.0, 150.0, 1000.0, 180.0), "ڤد")]
        provider, _, _ = provider_with(column(100, 1000, [100, 150]), [("ايبو", 0.9), ("ڤ", 0.8)])
        result, detected = script.evaluate_page(provider, page_png(), gold)

        self.assertEqual(len(detected), 2)
        self.assertEqual(result["line"]["edits"], 1)
        self.assertEqual((result["line"]["precision"], result["line"]["recall"]), (1.0, 1.0))
        self.assertEqual(result["page"]["edits"], 1)
        summary = script.overall([{"article_id": "a", "page_index": 0, **result}])
        self.assertAlmostEqual(summary["matched_line_cer"], 1 / 6, places=6)
        self.assertEqual(summary["pages"], 1)

    def test_reviewed_reading_order_replaces_the_export_order(self):
        script = load_eval_script()
        # The export lists the left box first; the true order (and the provider's) is right first.
        gold = [((100.0, 100.0, 540.0, 130.0), "ايبو"), ((560.0, 100.0, 1000.0, 130.0), "فد")]
        boxes = [(100, 100, 540, 130, 0.9), (560, 100, 1000, 130, 0.9)]
        answers = [("فد", 0.9), ("ايبو", 0.9)]  # read right to left

        by_export, _ = script.evaluate_page(provider_with(boxes, answers)[0], page_png(), gold)
        self.assertGreater(by_export["page"]["cer"], 0.0)
        self.assertEqual((by_export["order"]["kendall_tau"], by_export["order"]["gold_order"]), (-1.0, "export"))

        reviewed, _ = script.evaluate_page(provider_with(boxes, answers)[0], page_png(), gold, gold_order=[1, 0])
        self.assertEqual(reviewed["page"]["cer"], 0.0)
        self.assertEqual((reviewed["order"]["kendall_tau"], reviewed["order"]["gold_order"]), (1.0, "reviewed"))
        self.assertEqual(reviewed["line"], by_export["line"])  # the matched-line CER does not depend on order

        summary = script.overall([{"article_id": "a", "page_index": 0, **by_export},
                                  {"article_id": "a", "page_index": 1, **reviewed}])
        self.assertEqual((summary["order_kendall_tau"], summary["order_kendall_tau_page_mean"]), (0.0, 0.0))
        self.assertEqual(summary["pages_with_reviewed_order"], 1)

    def test_gold_boxes_follow_the_reviewed_order(self):
        script = load_eval_script()
        gold = [((100.0, 100.0, 540.0, 130.0), "ايبو"), ((560.0, 100.0, 1000.0, 130.0), "فد")]
        provider, _, recogniser = provider_with([], [("فد", 0.9), ("ايبو", 0.9)], detector=RaisingDetector())
        result, detected = script.evaluate_page(provider, page_png(), gold, gold_boxes=True, gold_order=[1, 0])
        self.assertEqual([d.xyxy for d in detected], [gold[1][0], gold[0][0]])  # right box sent first
        self.assertEqual((result["page"]["cer"], result["line"]["cer"]), (0.0, 0.0))
        self.assertEqual(result["order"]["kendall_tau"], 1.0)

    def test_reading_order_file_is_optional_and_validated(self):
        script = load_eval_script()
        self.assertEqual(script.load_reading_order(self.tmp / "missing.json"), {})
        path = self.tmp / "order.json"
        path.write_text(json.dumps({"1950_11_009_p00": [1, 0]}), encoding="utf-8")
        orders = script.load_reading_order(path)
        self.assertEqual(script.page_key("1950_11_009", 0), "1950_11_009_p00")
        self.assertEqual(script.gold_order_for(orders, "1950_11_009_p00", 2), [1, 0])
        self.assertIsNone(script.gold_order_for(orders, "1950_11_009_p01", 2))  # not reviewed: export order
        for bad_count in (1, 3):  # an index out of range, or a gold line left out
            with self.assertRaises(ValueError):
                script.gold_order_for(orders, "1950_11_009_p00", bad_count)
        with self.assertRaises(ValueError):
            script.gold_order_for({"k": [0, 0]}, "k", 2)

    def test_order_sheets_number_the_gold_boxes_and_write_a_template(self):
        export = self.write_export()
        folder = self.tmp / "pages" / "1950_11_009" / "pages"
        folder.mkdir(parents=True)
        (folder / "page_00.png").write_bytes(page_png(width=1000, height=2000))
        spec = importlib.util.spec_from_file_location(
            "make_order_sheets", ocr_trocr.PROJECT_DIR / "scripts" / "make_order_sheets.py")
        sheets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sheets)

        out = self.tmp / "review"
        template = sheets.make_sheets(export, self.tmp / "pages", ["1950_11_009"], out, max_side=500,
                                      reading_orders={"1950_11_009_p00": [0]})
        self.assertEqual(template, {"1950_11_009_p00": [0]})  # the 1 px box is dropped, as in the eval
        self.assertEqual(json.loads((out / "test_reading_order.template.json").read_text(encoding="utf-8")), template)
        sheet = cv2.imread(str(out / "1950_11_009_p00.jpg"))
        self.assertEqual(sheet.shape[:2], (500, 250))  # downscaled, aspect kept
        self.assertLess(sheet.min(), 200)  # something was drawn on the white page

        blank = np.full((200, 400, 3), 255, dtype=np.uint8)
        drawn = sheets.draw_sheet(blank, [((10, 10, 300, 40), "a"), ((10, 60, 300, 90), "b")], path_order=[1, 0])
        self.assertEqual(drawn.shape, blank.shape)  # small pages are not upscaled
        self.assertTrue((blank == 255).all())  # the input image is left untouched

    def test_gold_boxes_skip_the_detector_and_keep_annotation_order(self):
        script = load_eval_script()
        gold = [((100.0, 100.0, 540.0, 130.0), "ايبو"), ((560.0, 100.0, 1000.0, 130.0), "فد")]  # left box first
        provider, _, recogniser = provider_with([], [("ايبو", 0.9), ("فد", 0.8)], detector=RaisingDetector())
        result, detected = script.evaluate_page(provider, page_png(), gold, gold_boxes=True)

        self.assertEqual([(d.order, d.xyxy, d.detector_confidence) for d in detected],
                         [(1, gold[0][0], None), (2, gold[1][0], None)])
        self.assertEqual(recogniser.crops[0].shape, line_crop(np.zeros((PAGE_H, PAGE_W, 3)), gold[0][0]).shape)
        self.assertEqual((result["line"]["cer"], result["line"]["precision"], result["line"]["recall"]), (0.0, 1.0, 1.0))
        self.assertEqual(result["page"]["cer"], 0.0)  # both texts follow the export order
        self.assertEqual(result["errors"]["extra"]["no_gold_overlap"]["lines"], 0)

    def test_main_on_the_val_split_writes_val_files_with_the_padding(self):
        script = load_eval_script()
        export = self.write_export()
        folder = self.tmp / "pages" / "1950_11_002" / "pages"
        folder.mkdir(parents=True)
        (folder / "page_00.png").write_bytes(page_png(width=1000, height=2000))
        articles = self.tmp / "test_articles.json"
        articles.write_text(json.dumps(["1950_11_009"]), encoding="utf-8")

        built = {}

        def build(*args, **kwargs):
            built.update(kwargs)
            provider, _, _ = provider_with([(100, 200, 600, 300, 0.9)], pad_x=kwargs["pad_x"], pad_y=kwargs["pad_y"])
            provider.model_path, provider.detector_path = Path("trocr_v2"), Path("lines.pt")
            return provider

        self.patch("ocr_trocr.TrOCROCRProvider", build)
        out = self.tmp / "out"
        argv = ["eval_page_cer.py", "--export", str(export), "--pages-base", str(self.tmp / "pages"),
                "--test-articles", str(articles), "--out-dir", str(out), "--split", "val",
                "--pad-x", "0.2", "--pad-y", "0.05"]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            script.main()

        self.assertEqual((built["pad_x"], built["pad_y"]), (0.2, 0.05))
        report = json.loads((out / "page_eval_greedy_val.json").read_text(encoding="utf-8"))
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["config"]["articles"], ["1950_11_002"])  # the test article is never evaluated
        self.assertEqual((report["config"]["box_pad_x"], report["config"]["box_pad_y"]), (0.2, 0.05))
        self.assertEqual([(p["article_id"], p["page_index"]) for p in report["per_page"]], [("1950_11_002", 0)])
        self.assertTrue((out / "predictions_greedy_val.jsonl").is_file())
        self.assertFalse((out / "page_eval_greedy.json").exists())  # the test-split report is not overwritten

    def test_main_with_gold_boxes_writes_a_gold_report(self):
        script = load_eval_script()
        export = self.write_export()
        for article in ("1950_11_002", "1950_11_009"):
            folder = self.tmp / "pages" / article / "pages"
            folder.mkdir(parents=True)
            (folder / "page_00.png").write_bytes(page_png(width=1000, height=2000))
        articles = self.tmp / "test_articles.json"
        articles.write_text(json.dumps(["1950_11_002", "1950_11_009"]), encoding="utf-8")

        provider, _, _ = provider_with([], detector=RaisingDetector())
        provider.model_path, provider.detector_path = Path("trocr_v1"), Path("lines.pt")
        self.patch("ocr_trocr.TrOCROCRProvider", lambda *args, **kwargs: provider)
        out = self.tmp / "out"
        argv = ["eval_page_cer.py", "--export", str(export), "--pages-base", str(self.tmp / "pages"),
                "--test-articles", str(articles), "--out-dir", str(out), "--gold-boxes",
                "--reading-order", str(self.tmp / "no_reading_order.json")]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            script.main()

        report = json.loads((out / "page_eval_greedy_gold_boxes.json").read_text(encoding="utf-8"))
        self.assertEqual((report["boxes"], report["config"]["detector"]), ("gold", None))
        self.assertEqual(report["overall"]["gold_lines"], 2)
        self.assertEqual((report["overall"]["detector_precision"], report["overall"]["detector_recall"]), (1.0, 1.0))
        self.assertIn("recognition", report["overall"]["errors"])
        self.assertEqual(report["overall"]["errors"]["recognition"]["lines"], 2)
        self.assertEqual((report["split"], report["config"]["box_pad_x"], report["config"]["box_pad_y"]),
                         ("test", 0.0, 0.0))
        self.assertIsNone(report["config"]["reading_order_file"])
        self.assertEqual(report["overall"]["pages_with_reviewed_order"], 0)
        self.assertEqual([p["order"]["gold_order"] for p in report["per_page"]], ["export", "export"])
        self.assertTrue((out / "page_eval_greedy_gold_boxes.csv").is_file())
        self.assertFalse((out / "page_eval_greedy.json").exists())  # the detector run is not overwritten


class RaisingDetector:
    def predict(self, *args, **kwargs):
        raise AssertionError("the detector must not run")


class GoldBoxProviderTests(TempDirTestCase):
    def test_read_boxes_loads_only_trocr(self):
        detector_file = self.tmp / "lines.pt"
        detector_file.write_bytes(b"")
        self.patch("ocr_trocr.load_detector", mock.Mock(side_effect=AssertionError("detector loaded")))
        provider = TrOCROCRProvider(detector_path=detector_file, recogniser=FakeRecogniser())
        lines = provider.read_boxes(page_png(), [(100.0, 100.0, 1000.0, 130.0)])
        self.assertEqual([(line.text, line.detector_confidence) for line in lines], [("line 1", None)])
        self.assertEqual(provider.read_boxes(page_png(), []), [])

    def test_processor_is_loaded_with_the_slow_tokenizer(self):
        model = mock.MagicMock()
        model.to.return_value = model
        with mock.patch("transformers.TrOCRProcessor.from_pretrained") as processor_load, \
                mock.patch("transformers.VisionEncoderDecoderModel.from_pretrained", return_value=model) as model_load:
            TrOCRRecogniser.load(self.tmp, device="cpu")
        self.assertEqual(processor_load.call_args.kwargs, {"local_files_only": True, "use_fast": False})
        self.assertEqual(model_load.call_args.kwargs, {"local_files_only": True})


class ErrorBreakdownTests(TempDirTestCase):
    GOLD = [((0, 0, 100, 20), "abcdefghij"),
            ((0, 30, 100, 50), "abcdefghij"),
            ((0, 60, 100, 80), "abcdefghij"),
            ((0, 90, 100, 110), "klmn"),      # missed, but a small prediction touches it
            ((0, 500, 100, 520), "opqr")]     # missed, nothing near it
    PRED = [((0, 0, 100, 20, 0.9), "abcdefghij"),           # exact
            ((0, 30, 100, 50, 0.9), "abcdefghijbbbbbb"),    # 60% longer, new run of six b, 6 edits
            ((0, 60, 100, 80, 0.9), "abcde"),               # 50% shorter, 5 edits
            ((0, 95, 30, 105, 0.9), "zz"),                  # extra, touches "klmn" (IoU 0.15)
            ((500, 500, 600, 520, 0.9), "yyyyy")]           # extra, touches nothing, run of five y

    def test_breakdown_categories(self):
        errors = stage_metrics.line_error_breakdown(self.PRED, self.GOLD)
        self.assertEqual(errors["recognition"], {"lines": 3, "edits": 11, "gold_chars": 30, "cer": round(11 / 30, 6)})
        self.assertEqual(errors["length_mismatch"], {"longer": {"lines": 1, "edits": 6}, "shorter": {"lines": 1, "edits": 5}})
        self.assertEqual(errors["repeated_run"], {"lines": 2, "edits": 11})
        self.assertEqual(errors["extra"], {"overlaps_gold": {"lines": 1, "edits": 2}, "no_gold_overlap": {"lines": 1, "edits": 5}})
        self.assertEqual(errors["missed"], {"partly_covered": {"lines": 1, "edits": 4}, "not_detected": {"lines": 1, "edits": 4}})

    def test_recognition_extra_and_missed_add_up_to_the_matched_line_cer(self):
        errors = stage_metrics.line_error_breakdown(self.PRED, self.GOLD)
        total = (errors["recognition"]["edits"]
                 + sum(b["edits"] for b in errors["extra"].values())
                 + sum(b["edits"] for b in errors["missed"].values()))
        self.assertEqual(total, stage_metrics.matched_line_cer(self.PRED, self.GOLD)["edits"])

    def test_repeat_run_already_in_gold_and_small_length_changes_are_not_flagged(self):
        gold = [((0, 0, 100, 20), "aaaaa bcdefghij")]
        pred = [((0, 0, 100, 20, 0.9), "aaaaa bcdefghi")]   # 7% shorter, run is in gold
        errors = stage_metrics.line_error_breakdown(pred, gold)
        self.assertEqual(errors["repeated_run"]["lines"], 0)
        self.assertEqual(errors["length_mismatch"]["shorter"]["lines"], 0)
        self.assertEqual(errors["recognition"]["edits"], 1)

    def test_page_breakdowns_add_up(self):
        page = stage_metrics.line_error_breakdown(self.PRED, self.GOLD)
        total = stage_metrics.sum_error_breakdowns([page, page])
        self.assertEqual(total["recognition"]["edits"], 22)
        self.assertEqual(total["recognition"]["cer"], page["recognition"]["cer"])
        self.assertEqual(total["missed"]["not_detected"], {"lines": 2, "edits": 8})
        self.assertEqual(stage_metrics.sum_error_breakdowns([]), {})
