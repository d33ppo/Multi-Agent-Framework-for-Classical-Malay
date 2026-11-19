"""
Tests for the local OCR training scripts (ocr_training/train_yolo.py, train_trocr.py)
and the label normalisation they rely on.

Nothing is trained and nothing is downloaded: the tests cover the argument defaults,
the dataset checks, the metric functions and the decoding settings, with a tiny
random model built in memory for the one test that runs real generation.
"""

import importlib.util
import json
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import ocr_trocr
from tests.support import TempDirTestCase
import tests.test_ocr_trocr as trocr_tests

td = ocr_trocr.td
TRAINING_DIR = ocr_trocr.PROJECT_DIR / "ocr_training"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, TRAINING_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LabelMapTests(TempDirTestCase):
    def test_reversed_comma_becomes_the_arabic_comma(self):
        self.assertEqual(td.LABEL_MAP["⹁"], "،")
        td.LABEL_FIX_COUNTS.clear()
        self.assertEqual(td.normalise_text(" ايبو⹁  باف⹁ "), "ايبو، باف،")
        self.assertEqual(td.LABEL_FIX_COUNTS["⹁->،"], 2)
        self.assertEqual(td.normalise_text("ايبو، باف"), "ايبو، باف")  # the Arabic comma itself is untouched


class TrainYoloScriptTests(TempDirTestCase):
    def test_defaults_match_the_v2_plan(self):
        script = load_script("train_yolo")
        args = script.build_parser().parse_args(["--export", "e.json"])
        self.assertEqual((args.model, args.imgsz, args.epochs, args.patience, args.batch),
                         ("yolov8s.pt", 1280, 100, 20, 8))
        self.assertEqual(script.DEFAULT_WEIGHTS, ocr_trocr.PROJECT_DIR / "models" / "yolo" / "qalam_lines_v2.pt")
        self.assertEqual(script.DEFAULT_RESULTS_DIR, TRAINING_DIR / "results" / "yolo_v2")
        self.assertEqual(script.TASK_OVERRIDES, {1: ("1950_11_002", 0)})
        self.assertEqual(json.loads(open(args.test_articles, encoding="utf-8").read())[0], "1950_11_009")

    def test_metrics_helpers(self):
        script = load_script("train_yolo")
        results = SimpleNamespace(results_dict={"metrics/precision(B)": np.float64(0.91), "metrics/recall(B)": 0.9,
                                                "metrics/mAP50(B)": 0.95, "metrics/mAP50-95(B)": 0.7, "fitness": 0.72})
        self.assertEqual(script.validation_metrics(results),
                         {"precision": 0.91, "recall": 0.9, "map50": 0.95, "map50_95": 0.7})
        csv_path = self.tmp / "results.csv"
        csv_path.write_text("epoch,loss\n1,0.5\n2,0.4\n3,0.3\n", encoding="utf-8")
        self.assertEqual(script.epochs_run(csv_path), 3)
        self.assertIsNone(script.epochs_run(self.tmp / "missing.csv"))

    def test_stops_without_cuda_before_touching_the_dataset(self):
        script = load_script("train_yolo")
        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(script.yp, "export_to_yolo") as export:
            with self.assertRaises(SystemExit):
                script.main(["--export", "e.json"])
        export.assert_not_called()


class TrainTrocrScriptTests(TempDirTestCase):
    def test_defaults_keep_the_effective_batch_at_8(self):
        script = load_script("train_trocr")
        args = script.build_parser().parse_args(["--export", "e.json"])
        self.assertEqual((args.batch, args.grad_accum, args.batch * args.grad_accum), (4, 2, 8))
        self.assertEqual((args.epochs, args.lr), (25, 5e-5))
        self.assertEqual(script.CKPT, "microsoft/trocr-base-stage1")
        self.assertEqual(script.DEFAULT_MODEL_DIR, ocr_trocr.PROJECT_DIR / "models" / "trocr_v2")
        self.assertEqual(script.DEFAULT_RESULTS_DIR, TRAINING_DIR / "results" / "trocr_v2")

    def test_cer_is_total_edits_over_total_reference_characters(self):
        script = load_script("train_trocr")
        self.assertEqual(script.cer(["abcd", "ef"], ["abxd", "ef"]), 1 / 6)
        self.assertEqual(script.cer(["ab"], ["ab"]), 0.0)
        self.assertEqual(script.norm("  ا  ب\n"), "ا ب")

    def test_missing_pages_lists_pages_without_a_full_resolution_png(self):
        script = load_script("train_trocr")
        pages = [{"article_id": "1950_11_002", "page_index": 0}, {"article_id": "1950_11_002", "page_index": 1},
                 {"article_id": "1950_12_016", "page_index": 0}]  # the last one is an excluded page
        folder = self.tmp / "1950_11_002" / "pages"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "page_00.png"), np.full((4, 4, 3), 255, dtype=np.uint8))
        self.assertEqual(script.missing_pages(pages, self.tmp), [("1950_11_002", 1)])

    def test_max_label_length_is_the_longest_line_plus_8_capped_at_512(self):
        script = load_script("train_trocr")

        def tokenizer(text, add_special_tokens):
            return SimpleNamespace(input_ids=list(text))

        self.assertEqual(script.max_label_length(tokenizer, ["ab", "abcde"]), 13)
        self.assertEqual(script.max_label_length(tokenizer, ["a" * 600]), 512)

    def test_test_set_decoding_is_greedy_on_any_device(self):
        script = load_script("train_trocr")
        self.assertEqual((script.NUM_BEAMS, script.DECODING), (1, "greedy"))

        class Model:
            calls = []

            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, batch, **kwargs):
                self.calls.append((len(batch), kwargs))
                return [[5, 6]] * len(batch)

        import torch
        dataset = SimpleNamespace(pixel_values=lambda i: torch.zeros(3, 4, 4))
        dataset = type("DS", (), {"__len__": lambda self: 5, "pixel_values": staticmethod(dataset.pixel_values)})()
        tokenizer = SimpleNamespace(batch_decode=lambda seqs, skip_special_tokens: [" x  y "] * len(seqs))
        for device in ("cpu", "cuda"):  # the fake model ignores .to(), so "cuda" needs no GPU
            Model.calls = []
            with mock.patch.object(torch.Tensor, "to", lambda tensor, *a, **k: tensor):
                preds = script.predict_greedy(Model(), tokenizer, dataset, max_length=128, batch_size=2, device=device)
            self.assertEqual(preds, ["x y"] * 5)
            self.assertEqual([n for n, _ in Model.calls], [2, 2, 1])
            for _, kwargs in Model.calls:
                self.assertEqual(kwargs, {"num_beams": 1, "do_sample": False, "max_length": 128})

    def test_greedy_prediction_runs_real_generation_and_model_config_is_greedy(self):
        script = load_script("train_trocr")
        import torch
        model = trocr_tests.TinyModelTests.tiny_model()
        tokenizer = SimpleNamespace(bos_token_id=0, pad_token_id=1, eos_token_id=2,
                                    batch_decode=lambda seqs, skip_special_tokens: [str(len(s)) for s in seqs])
        script.configure_model(model, tokenizer, 12)
        gen = model.generation_config
        self.assertEqual((gen.num_beams, gen.do_sample, gen.max_length, gen.no_repeat_ngram_size), (1, False, 12, 0))
        dataset = type("DS", (), {"__len__": lambda self: 3,
                                  "pixel_values": lambda self, i: torch.rand(3, 32, 32)})()
        self.assertEqual(len(script.predict_greedy(model, tokenizer, dataset, max_length=12, batch_size=2)), 3)

    def test_test_report_scores_per_article_and_jawi_letters(self):
        script = load_script("train_trocr")
        rows = [{"article_id": "a"}, {"article_id": "b"}]
        report = script.score_test_set(rows, ["ڠا", "فد"], ["نا", "فد"])
        self.assertEqual(report["decoding"], "greedy")
        self.assertEqual(report["num_beams"], 1)
        self.assertEqual(report["test_cer"], 0.25)
        self.assertEqual(report["per_article"], {"a": 0.5, "b": 0.0})
        self.assertEqual(report["jawi_letter_error_rate"]["ڠ"], {"error_rate": 1.0, "occurrences": 1})
        self.assertEqual(report["jawi_letter_error_rate"]["ف"], {"error_rate": 0.0, "occurrences": 1})

    def test_line_dataset_items_have_padded_labels(self):
        script = load_script("train_trocr")
        import torch
        (self.tmp / "images").mkdir()
        cv2.imwrite(str(self.tmp / "images" / "l.png"), np.full((30, 90, 3), 255, dtype=np.uint8))
        (self.tmp / "val.csv").write_text("file_name,text,article_id,bx1,by1,bx2,by2\nl.png,abc,a,5,5,85,25\n",
                                          encoding="utf-8")

        class Tokenizer:
            eos_token_id = 2

            def __call__(self, text, add_special_tokens, truncation, max_length):
                return SimpleNamespace(input_ids=[10 + i for i, _ in enumerate(text)][:max_length])

        def processor(images, return_tensors):
            return SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8))

        processor.tokenizer = Tokenizer()
        item = script.LineDS(self.tmp, "val", False, processor, max_length=6)[0]
        self.assertEqual(item["labels"].tolist(), [10, 11, 12, 2, -100, -100])
        self.assertEqual(tuple(item["pixel_values"].shape), (3, 8, 8))
