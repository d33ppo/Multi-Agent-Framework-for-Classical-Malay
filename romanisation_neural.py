"""
Method A romanisation: pretrained neural Jawi -> Rumi transliteration model.

Model: mesolitica/jawi-nanot5-small-malaysian-cased (T5, ~358MB). Reported by
Malaya as chrF2++ 98.01 Jawi->Rumi on a 10k held-out set of modern Jawi.
The model is used as-is; this project does not fine-tune it.

The model is loaded with `transformers` directly using the same input prefix
as Malaya's `malaya.jawi.huggingface(...).generate(..., to_lang='rumi')`,
which avoids adding the full `malaya` package as a dependency.
Inference runs locally: manuscript text is not sent to any external service.
The first run downloads the model weights from HuggingFace.
"""

from __future__ import annotations

import os
import threading

DEFAULT_NEURAL_MODEL = "mesolitica/jawi-nanot5-small-malaysian-cased"
MALAYA_PREFIX = "terjemah ke Rumi: "  # matches malaya.torch_model.huggingface.Translation
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16


class NeuralRomaniser:
    """Lazy wrapper around the nanot5 model. Romanises one line at a time."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.environ.get("JAWI_NEURAL_MODEL", DEFAULT_NEURAL_MODEL)
        self.device = device
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def romanise(self, jawi_text: str) -> str:
        """Romanise Jawi text, keeping the original line breaks."""
        lines = jawi_text.splitlines()
        non_empty = [line.strip() for line in lines if line.strip()]
        if not non_empty:
            return ""

        romanised: list[str] = []
        for start in range(0, len(non_empty), BATCH_SIZE):
            romanised.extend(self._generate(non_empty[start : start + BATCH_SIZE]))
        outputs = iter(romanised)
        return "\n".join(next(outputs) if line.strip() else "" for line in lines)

    def _generate(self, lines: list[str]) -> list[str]:
        self._load()
        import torch

        inputs = self._tokenizer(
            [f"{MALAYA_PREFIX}{line}" for line in lines],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=1)
        return [text.strip() for text in self._tokenizer.batch_decode(generated, skip_special_tokens=True)]
