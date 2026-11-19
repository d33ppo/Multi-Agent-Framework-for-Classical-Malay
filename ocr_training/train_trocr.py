"""
Fine-tune TrOCR locally on a CUDA GPU (TrOCR v2).

Same setup as notebooks/trocr_v1_colab.ipynb: microsoft/trocr-base-stage1, lr 5e-5,
25 epochs, fp16, greedy validation each epoch, early stopping after 5 epochs without
a better validation CER, same augmentation. The only training change is memory:
per-device batch 4 with gradient_accumulation_steps 2, so the effective batch stays 8
on an 8 GB GPU.

Steps:
  1. trocr_dataset.build_line_dataset from the Label Studio export and the local
     400 DPI page PNGs (<pages-base>/<article>/pages/page_NN.png). Low-resolution
     previews are never used: missing pages stop the run.
  2. Train with Seq2SeqTrainer. Re-running resumes from the last checkpoint.
  3. Evaluate the saved model on the locked test articles. Decoding is always
     greedy (num_beams=1), whatever the device.

Outputs:
  models/trocr_v2/                     model + processor (gitignored), checkpoints/ inside
  ocr_training/results/trocr_v2/       dataset_stats.json, log_history.json, test_results_greedy.json,
                                       test_predictions_greedy.csv (gitignored, contains Qalam text)

Usage:
    python ocr_training/train_trocr.py --export exports/label_studio_export.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

import trocr_dataset as td  # noqa: E402

CKPT = "microsoft/trocr-base-stage1"
TASK_OVERRIDES = {1: ("1950_11_002", 0)}  # task 1 has no metadata in the export
SEED = 42
NUM_BEAMS = 1          # greedy, for validation and for the test set, on any device
DECODING = "greedy"
DEFAULT_TEST_ARTICLES = TRAINING_DIR / "splits" / "test_articles.json"
DEFAULT_PAGES_BASE = PROJECT_DIR / "jawi_dataset" / "dataset"
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "trocr_ds_v2"
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "trocr_v2"
DEFAULT_RESULTS_DIR = TRAINING_DIR / "results" / "trocr_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="Label Studio JSON export")
    parser.add_argument("--pages-base", default=str(DEFAULT_PAGES_BASE), help="folder with <article>/pages/page_NN.png at 400 DPI")
    parser.add_argument("--test-articles", default=str(DEFAULT_TEST_ARTICLES), help="JSON list of locked test articles")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="line dataset folder (rebuilt each run)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=4, help="per-device train batch")
    parser.add_argument("--grad-accum", type=int, default=2, help="gradient accumulation steps (batch x this = 8)")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--skip-test", action="store_true", help="train only, do not touch the test set")
    return parser


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def cer(refs: list[str], hyps: list[str]) -> float:
    """Total character edits over total reference characters (what jiwer.cer gives for lists)."""
    from rapidfuzz.distance import Levenshtein
    edits = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    return edits / max(1, sum(len(r) for r in refs))


def missing_pages(pages: list[dict], pages_base, exclude_pages=td.DEFAULT_EXCLUDE_PAGES) -> list[tuple]:
    """(article, page_index) of annotated pages with no full-resolution PNG under pages_base."""
    exclude = set(exclude_pages)
    keys = sorted({(p["article_id"], p["page_index"]) for p in pages} - exclude)
    return [(a, i) for a, i in keys if not (Path(pages_base) / a / "pages" / f"page_{i:02d}.png").exists()]


def max_label_length(tokenizer, texts: list[str]) -> int:
    """Same rule as the v1 notebook: longest training line in tokens plus 8, capped at 512."""
    longest = max(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)
    return min(512, longest + 8)


def augment(img):
    """Same augmentation as v1. No flips: they would break right-to-left script."""
    from PIL import Image, ImageEnhance, ImageFilter
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(40, 85))
        img = Image.open(buf).convert("RGB")
    return img


def read_rows(dataset_dir, split: str) -> list[dict]:
    with open(Path(dataset_dir) / f"{split}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class LineDS:
    """One split of the line dataset. Train: jittered crop + augmentation. Eval: fixed crop."""

    def __init__(self, dataset_dir, split: str, train: bool, processor, max_length: int):
        self.dir = Path(dataset_dir)
        self.rows = read_rows(dataset_dir, split)
        self.train = train
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def pixel_values(self, i: int):
        from PIL import Image
        row = self.rows[i]
        with Image.open(self.dir / "images" / row["file_name"]) as im:
            img = Image.fromarray(td.crop_for(im.convert("RGB"), row, self.train))
        if self.train:
            img = augment(img)
        return self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

    def __getitem__(self, i: int):
        import torch
        tok = self.processor.tokenizer
        ids = tok(self.rows[i]["text"], add_special_tokens=False, truncation=True,
                  max_length=self.max_length - 1).input_ids + [tok.eos_token_id]
        labels = ids + [-100] * (self.max_length - len(ids))
        return {"pixel_values": self.pixel_values(i), "labels": torch.tensor(labels)}


def training_arguments(args, checkpoint_dir, max_length: int):
    from transformers import Seq2SeqTrainingArguments
    return Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr, warmup_ratio=0.05, weight_decay=0.01, num_train_epochs=args.epochs, fp16=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="cer", greater_is_better=False, predict_with_generate=True,
        generation_max_length=max_length, generation_num_beams=NUM_BEAMS, logging_steps=50, report_to="none",
        dataloader_num_workers=args.workers, seed=SEED)


def configure_model(model, tokenizer, max_length: int) -> None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = model.generation_config
    gen.decoder_start_token_id = tokenizer.bos_token_id
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    gen.max_length = max_length
    gen.no_repeat_ngram_size = 0   # OCR text legitimately repeats characters
    gen.num_beams = NUM_BEAMS
    gen.do_sample = False


def predict_greedy(model, tokenizer, dataset, max_length: int, batch_size: int = 16, device: str = "cpu") -> list[str]:
    """Greedy predictions for every line of `dataset`, normalised. The decoding never depends on the device."""
    import torch
    model = model.to(device).eval()
    preds = []
    for start in range(0, len(dataset), batch_size):
        batch = torch.stack([dataset.pixel_values(j) for j in range(start, min(start + batch_size, len(dataset)))])
        with torch.inference_mode():
            seqs = model.generate(batch.to(device), num_beams=NUM_BEAMS, do_sample=False, max_length=max_length)
        preds.extend(norm(s) for s in tokenizer.batch_decode(seqs, skip_special_tokens=True))
    return preds


def score_test_set(rows: list[dict], refs: list[str], preds: list[str]) -> dict:
    """Line CER, character accuracy, per-article CER and the error rate of the Jawi-specific letters."""
    from rapidfuzz.distance import Levenshtein
    per_article: dict[str, list[int]] = {}
    ref_count, err_count = Counter(), Counter()
    for row, ref, hyp in zip(rows, refs, preds):
        totals = per_article.setdefault(row["article_id"], [0, 0])
        totals[0] += Levenshtein.distance(ref, hyp)
        totals[1] += len(ref)
        ref_count.update(ref)
        for op in Levenshtein.editops(ref, hyp):
            if op.tag in ("replace", "delete"):
                err_count[ref[op.src_pos]] += 1
    score = cer(refs, preds)
    return {
        "test_cer": score, "char_accuracy": 1 - score, "decoding": DECODING, "num_beams": NUM_BEAMS,
        "per_article": {a: e / max(1, n) for a, (e, n) in sorted(per_article.items())},
        "jawi_letter_error_rate": {ch: {"error_rate": round(err_count[ch] / ref_count[ch], 4), "occurrences": ref_count[ch]}
                                   for ch in td.JAWI_SPECIAL if ref_count[ch]},
        "lines": len(rows),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script trains on a CUDA GPU only (fp16).")
    try:
        import accelerate  # noqa: F401  (the transformers Trainer needs it)
    except ImportError:
        raise SystemExit("Missing package: pip install accelerate")
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, TrOCRProcessor, VisionEncoderDecoderModel,
                              default_data_collator)
    from transformers.trainer_utils import get_last_checkpoint

    random.seed(SEED)
    model_dir, results_dir = Path(args.model_dir), Path(args.results_dir)
    checkpoint_dir = model_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    test_articles = json.loads(Path(args.test_articles).read_text(encoding="utf-8"))

    # 1. Dataset, from full-resolution pages only
    pages, _report = td.yp.load_pages(args.export, TASK_OVERRIDES)
    missing = missing_pages(pages, args.pages_base)
    if missing:
        raise SystemExit(f"{len(missing)} annotated page(s) have no 400 DPI PNG under {args.pages_base}: {missing}")
    stats = td.build_line_dataset(args.export, args.pages_base, args.dataset_dir, test_articles,
                                  overrides=TASK_OVERRIDES, allow_preview_fallback=False)
    (results_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("lines", stats["lines"], "| pages", stats["pages"], "| label fixes", stats["label_fixes"])

    # 2. Train
    processor = TrOCRProcessor.from_pretrained(CKPT)
    tok = processor.tokenizer
    max_length = max_label_length(tok, [r["text"] for r in read_rows(args.dataset_dir, "train")])
    print("max label length", max_length)
    train_ds = LineDS(args.dataset_dir, "train", True, processor, max_length)
    val_ds = LineDS(args.dataset_dir, "val", False, processor, max_length)

    model = VisionEncoderDecoderModel.from_pretrained(CKPT)
    configure_model(model, tok, max_length)

    def compute_metrics(pred):
        labels = pred.label_ids.copy()
        labels[labels == -100] = tok.pad_token_id
        hyp = [norm(s) for s in tok.batch_decode(pred.predictions, skip_special_tokens=True)]
        ref = [norm(s) for s in tok.batch_decode(labels, skip_special_tokens=True)]
        return {"cer": cer(ref, hyp)}

    trainer = Seq2SeqTrainer(model=model, args=training_arguments(args, checkpoint_dir, max_length),
                             train_dataset=train_ds, eval_dataset=val_ds, data_collator=default_data_collator,
                             compute_metrics=compute_metrics,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    last = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    print("resume from", last)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(model_dir))          # the best checkpoint (load_best_model_at_end)
    processor.save_pretrained(str(model_dir))
    (results_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    print("best val CER", trainer.state.best_metric)

    # 3. Test set, greedy
    if args.skip_test:
        return 0
    del trainer, model
    torch.cuda.empty_cache()
    test_ds = LineDS(args.dataset_dir, "test", False, processor, max_length)
    best = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
    preds = predict_greedy(best, tok, test_ds, max_length, batch_size=args.eval_batch, device="cuda")
    refs = [norm(r["text"]) for r in test_ds.rows]
    report = score_test_set(test_ds.rows, refs, preds)
    (results_dir / f"test_results_{DECODING}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(results_dir / f"test_predictions_{DECODING}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "article_id", "reference", "prediction"])
        writer.writerows([r["file_name"], r["article_id"], ref, hyp] for r, ref, hyp in zip(test_ds.rows, refs, preds))
    print(f"TEST line CER {report['test_cer']:.4f}  character accuracy {report['char_accuracy']:.4f}  (decoding: {DECODING})")
    for article, value in report["per_article"].items():
        print(f"  {article}: CER {value:.4f}")
    return 0


if __name__ == "__main__":   # the guard is required: dataloader workers are spawned on Windows
    raise SystemExit(main())
