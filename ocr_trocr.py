"""
Fine-tuned TrOCR provider for the upload pipeline.

    page PNG -> YOLO line boxes -> reading order -> eval crops -> batched TrOCR -> lines

The preprocessing is the training code itself, imported from ocr_training/:
yolo_prefill.predict_boxes (conf 0.35, contained-box suppression),
yolo_prefill.reading_order (XY-cut, columns right to left) and
trocr_dataset.crop_for(..., train=False) (0.12 x box height margin, cut from
the full-resolution page). Model outputs only get NFC and whitespace collapse;
LABEL_MAP is a label fix and is not applied to outputs.

Both models load from local files only. Ultralytics is put in offline mode with
settings sync off, and transformers uses local_files_only, so no page image or
text leaves the machine.

Line confidence is exp(mean token log-prob). The low-confidence threshold
(OCR_CONFIDENCE_THRESHOLD) is UNCALIBRATED until task 3.6: treat the flag as a
hint, not a measured error rate.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from document_storage import PROJECT_DIR
from llm_text_runtime import _get_int_env, env_value
from ocr_providers import OCRLine, PageImage

# The training helpers import each other as top-level modules, so their folder goes on the path.
OCR_TRAINING_DIR = PROJECT_DIR / "ocr_training"
if str(OCR_TRAINING_DIR) not in sys.path:
    sys.path.append(str(OCR_TRAINING_DIR))

import trocr_dataset as td  # noqa: E402
import yolo_prefill as yp  # noqa: E402

DEFAULT_LINE_DETECTOR_PATH = "models/yolo/qalam_lines_cpu25.pt"
DEFAULT_LINE_DETECTOR_IMGSZ = 960  # the size the detector was trained at
DETECTOR_CONF = 0.35
MAX_LENGTH = 128
DEFAULT_NUM_BEAMS = 1  # greedy, the decoding behind the reported test CER
DEFAULT_BATCH_SIZE = 4  # what the Colab CPU evaluation used without running out of RAM
DEFAULT_CONFIDENCE_THRESHOLD = 0.60  # UNCALIBRATED placeholder until task 3.6


def normalise_output(text: str) -> str:
    """NFC and collapsed whitespace, the same normalisation the reported CER used."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def confidence_threshold() -> float:
    raw = env_value("OCR_CONFIDENCE_THRESHOLD")
    if not raw:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"OCR_CONFIDENCE_THRESHOLD must be a number between 0 and 1, got '{raw}'.") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"OCR_CONFIDENCE_THRESHOLD must be between 0 and 1, got {value}.")
    return value


def decode_page(png: bytes) -> np.ndarray:
    """BGR page image, as cv2.imread gave the detector during training and evaluation."""
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The page image could not be decoded.")
    return image


def line_crop(page_rgb: np.ndarray, box) -> np.ndarray:
    """The evaluation crop from training: EVAL_MARGIN_FRAC x box height around the box, from the full page."""
    x1, y1, x2, y2 = box[:4]
    crop = td.crop_for(page_rgb, {"bx1": x1, "by1": y1, "bx2": x2, "by2": y2}, train=False)
    return np.ascontiguousarray(crop)


def mean_logprob_confidence(token_ids, token_logprobs, eos_token_ids) -> float | None:
    """exp(mean log-prob) of the generated tokens up to and including the first EOS. Padding after it is ignored."""
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or ())
    total, count = 0.0, 0
    for token, logprob in zip(token_ids, token_logprobs):
        total += float(logprob)
        count += 1
        if token in eos:
            break
    return _clamp(math.exp(total / count)) if count else None


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def choose_device(name: str) -> str:
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError(f"OCR_DEVICE must be auto, cpu or cuda, got '{name}'.")
    return name


def load_detector(path: Path, device: str):
    """The YOLO line detector, with every Ultralytics network feature off."""
    os.environ["YOLO_OFFLINE"] = "true"  # skips the connectivity check and downloads
    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no usage analytics
    detector = YOLO(str(path))
    detector.overrides["device"] = device
    return detector


class TrOCRRecogniser:
    """Batched TrOCR line recognition. Called with RGB line crops, returns (text, confidence) per crop."""

    def __init__(self, model, processor, *, device: str = "cpu", num_beams: int = DEFAULT_NUM_BEAMS,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        # Without the cache every decoding step re-runs the whole prefix, which is very slow on CPU.
        model.config.decoder.use_cache = True
        model.generation_config.use_cache = True
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.num_beams = num_beams
        self.batch_size = batch_size

    @classmethod
    def load(cls, model_dir: Path, *, device: str, threads: int | None = None, **kwargs) -> "TrOCRRecogniser":
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if threads:
            torch.set_num_threads(threads)
        processor = TrOCRProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(str(model_dir), local_files_only=True)
        return cls(model, processor, device=device, **kwargs)

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float | None]]:
        results = []
        for start in range(0, len(crops), self.batch_size):
            results.extend(self._batch(crops[start:start + self.batch_size]))
        return results

    def _batch(self, crops: list[np.ndarray]) -> list[tuple[str, float | None]]:
        import torch
        from PIL import Image

        images = [Image.fromarray(crop) for crop in crops]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values, num_beams=self.num_beams, max_length=MAX_LENGTH,
                return_dict_in_generate=True, output_scores=True,
            )
            if self.num_beams > 1:
                confidences = [_clamp(math.exp(float(score))) for score in output.sequences_scores]
            else:
                logprobs = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                eos = self.model.generation_config.eos_token_id
                # For an encoder-decoder model, position 0 of each sequence is the decoder start token.
                confidences = [
                    mean_logprob_confidence(tokens.tolist(), scores.tolist(), eos)
                    for tokens, scores in zip(output.sequences[:, 1:], logprobs)
                ]
                del logprobs
        texts = [normalise_output(t) for t in self.processor.batch_decode(output.sequences, skip_special_tokens=True)]
        del output, pixel_values  # the per-step scores are large; keep none past this batch
        return list(zip(texts, confidences))


@dataclass
class DetectedLine:
    order: int  # 1-based position in reading order among the detected boxes
    xyxy: tuple[float, float, float, float]  # page pixels
    detector_confidence: float
    text: str
    confidence: float | None


class TrOCROCRProvider:
    """
    Fine-tuned TrOCR with the YOLO line detector. Both models load once, on first
    use, and pages are processed one at a time so memory stays bounded.
    `detector` and `recogniser` may be passed in already built (tests do this).
    """

    name = "trocr"

    def __init__(self, model_path: str | Path | None = None, detector_path: str | Path | None = None, *,
                 imgsz: int | None = None, num_beams: int | None = None, batch_size: int | None = None,
                 device: str | None = None, threads: int | None = None, threshold: float | None = None,
                 detector=None, recogniser=None):
        self.imgsz = imgsz or _get_int_env("LINE_DETECTOR_IMGSZ") or DEFAULT_LINE_DETECTOR_IMGSZ
        self.num_beams = num_beams or _get_int_env("OCR_NUM_BEAMS") or DEFAULT_NUM_BEAMS
        self.batch_size = batch_size or _get_int_env("OCR_BATCH_SIZE") or DEFAULT_BATCH_SIZE
        self.threads = threads or _get_int_env("OCR_TORCH_THREADS")
        self.device_setting = device or env_value("OCR_DEVICE", "auto")
        self.threshold = confidence_threshold() if threshold is None else threshold

        model_path = model_path or env_value("OCR_MODEL_PATH")
        detector_path = detector_path or env_value("LINE_DETECTOR_PATH", DEFAULT_LINE_DETECTOR_PATH)
        self.model_path = project_path(model_path) if model_path else None
        self.detector_path = project_path(detector_path)
        if recogniser is None and not (self.model_path and (self.model_path / "config.json").is_file()):
            raise ValueError(
                f"TrOCR model not found at '{self.model_path}'. Set OCR_MODEL_PATH to the folder with "
                "config.json, the weights and the processor files."
            )
        if detector is None and not self.detector_path.is_file():
            raise ValueError(f"Line detector not found at '{self.detector_path}'. Set LINE_DETECTOR_PATH.")

        self._detector = detector
        self._recogniser = recogniser
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load both models now instead of on the first page."""
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._detector is not None and self._recogniser is not None:
            return
        device = choose_device(self.device_setting)
        if self._detector is None:
            self._detector = load_detector(self.detector_path, device)
        if self._recogniser is None:
            self._recogniser = TrOCRRecogniser.load(
                self.model_path, device=device, threads=self.threads,
                num_beams=self.num_beams, batch_size=self.batch_size,
            )

    def read_page(self, png: bytes) -> list[DetectedLine]:
        """Every detected line in reading order, including lines the recogniser left empty."""
        page_bgr = decode_page(png)
        with self._lock:
            self._load()
            boxes = yp.reading_order(
                yp.predict_boxes(self._detector, page_bgr, conf=DETECTOR_CONF, imgsz=self.imgsz),
                page_bgr.shape[1],
            )
            if not boxes:
                return []
            # The recogniser was trained on crops read with PIL, which are RGB.
            page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
            results = self._recogniser([line_crop(page_rgb, box) for box in boxes])
        return [
            DetectedLine(order, tuple(float(v) for v in box[:4]), float(box[4]), text, confidence)
            for order, (box, (text, confidence)) in enumerate(zip(boxes, results), start=1)
        ]

    def recognise_page(self, page: PageImage) -> list[OCRLine]:
        lines = []
        for line in self.read_page(page.png):
            if not line.text:
                continue
            x1, y1, x2, y2 = line.xyxy
            left, top = round(x1), round(y1)
            lines.append(OCRLine(
                text=line.text,
                box=(left, top, round(x2) - left, round(y2) - top),
                confidence=line.confidence,
                order=line.order,
                low_confidence=None if line.confidence is None else line.confidence < self.threshold,
            ))
        return lines
