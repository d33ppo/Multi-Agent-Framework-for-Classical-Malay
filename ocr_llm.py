"""
LLM Vision-based OCR for extracting Jawi text from Classical Malay manuscripts.

Uses OpenRouter (https://openrouter.ai) with any vision-capable model as an
alternative to Tesseract, encoding the manuscript image as base64 and prompting
the model to extract Arabic-script (Jawi) text.

Results are returned in the same dict format as ocr_jawi.extract_jawi_text()
for easy comparison.

Usage:
    python ocr_llm.py data/input/jawi_image_data/1950_07_002_3.png data/output/jawi_text_llm_&_tesseract/vlm_1950_07_002_3.txt

Configure via .env:
    OPENROUTER_API_KEY      — required
    OPENROUTER_VISION_MODEL — model ID (default: google/gemini-2.5-flash-preview)
"""

import os
import sys
import io
import time
import json
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv

from ocr_jawi import compute_cer

load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Jawi-specific prompt
# ---------------------------------------------------------------------------

JAWI_EXTRACTION_PROMPT = """\
You are an expert palaeographer specialising in Classical Malay manuscripts written in Jawi script (Arabic-based writing system for the Malay language).

Examine the attached manuscript image carefully and extract **all visible Jawi text** exactly as it appears. Follow these rules:

1. **Script direction**: Jawi is written right-to-left. Preserve the original line order.
2. **Faithfulness**: Transcribe every character you can read. Do not paraphrase, translate, or summarise.
3. **Uncertain characters**: If a character is unclear, provide your best reading. Do NOT insert placeholders like "[?]" or "...".
4. **Line breaks**: Preserve the line structure of the original manuscript — each visual line should be a separate line in your output.
5. **Diacritics**: Include all diacritical marks (dots, vowel marks) you can identify.
6. **No commentary**: Output ONLY the extracted Jawi text. Do not include any English explanation, translation, metadata, or preamble.

Begin transcription:
"""

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        ext_map = {
            ".gif": "image/gif", ".png": "image/png",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".tif": "image/tiff", ".tiff": "image/tiff",
            ".bmp": "image/bmp", ".webp": "image/webp",
        }
        mime_type = ext_map.get(path.suffix.lower(), "image/png")

    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")

    return b64, mime_type


def _estimate_cost(model: str, b64_data: str, output_text: str) -> float | None:
    """Rough cost estimate in USD for Gemini 2.5 Flash pricing."""
    if "gemini-2.5-flash" in model:
        image_tokens = len(b64_data) / 750
        prompt_tokens = len(JAWI_EXTRACTION_PROMPT) / 4
        input_tokens = image_tokens + prompt_tokens
        output_tokens = len(output_text) / 4
        cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60
        return round(cost, 6)
    return None


# ---------------------------------------------------------------------------
# OpenRouter provider
# ---------------------------------------------------------------------------


def _extract_openrouter(image_path: str) -> dict:
    """Extract Jawi text using a vision-capable model via OpenRouter."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )

    model_name = os.environ.get("OPENROUTER_VISION_MODEL", DEFAULT_VISION_MODEL)
    b64_data, media_type = _encode_image(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/malay-fyp/classical-malay-pipeline",
        "X-Title": "Classical Malay Multi-Agent Pipeline",
    }

    payload = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": JAWI_EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    }

    start = time.time()
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})

    return {
        "raw_text": raw_text,
        "confidence": "N/A (LLM)",
        "processing_time": round(elapsed, 3),
        "language": "Jawi (via OpenRouter)",
        "provider": "openrouter",
        "model": data.get("model", model_name),
        "cost_estimate_usd": _estimate_cost(model_name, b64_data, raw_text),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", "N/A"),
            "output_tokens": usage.get("completion_tokens", "N/A"),
        },
    }


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": _extract_openrouter,
}


def extract_jawi_text_llm(image_path: str, provider: str = "openrouter") -> dict:
    """
    Extract Jawi text from a manuscript image using a vision LLM via OpenRouter.

    Args:
        image_path: Path to the manuscript image.
        provider:   "openrouter" (only supported provider).

    Returns:
        dict with keys: raw_text, confidence, processing_time,
                        language, provider, model, cost_estimate_usd, usage
    """
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS.keys())}"
        )
    return PROVIDERS[provider](image_path)


def save_output(result: dict, output_path: str, cer_results: dict | None = None) -> None:
    """Save LLM OCR result to a text file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== Jawi LLM-OCR Result (OpenRouter / {result['model']}) ===\n\n")
        f.write(result["raw_text"])
        f.write("\n\n=== Metadata ===\n")
        f.write(f"Provider     : {result['provider']}\n")
        f.write(f"Model        : {result['model']}\n")
        f.write(f"Language     : {result['language']}\n")
        f.write(f"Confidence   : {result['confidence']}\n")
        f.write(f"Process Time : {result['processing_time']}s\n")
        if result.get("cost_estimate_usd") is not None:
            f.write(f"Est. Cost    : ${result['cost_estimate_usd']}\n")
        if result.get("usage"):
            f.write(f"Tokens In    : {result['usage'].get('input_tokens', 'N/A')}\n")
            f.write(f"Tokens Out   : {result['usage'].get('output_tokens', 'N/A')}\n")
        if cer_results:
            f.write("\n=== CER Evaluation ===\n")
            for gt_path, cer in cer_results.items():
                f.write(f"Ground Truth : {gt_path}\n")
                f.write(f"CER          : {cer:.4f} ({cer * 100:.2f}%)\n\n")
    print(f"Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/input/jawi_image_data/1950_07_002_4.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/output/jawi_text_llm_&_tesseract/vlm_1950_07_002_4.txt"

    # Derive ground truth path from input filename, e.g. 1950_07_002_3.png -> 1950_07_002_3.txt
    input_base = os.path.splitext(os.path.basename(image_path))[0]
    gt_path = os.path.join("data/ground_truth", f"{input_base}.txt")

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print(f"Processing : {image_path}")
    print(f"Model      : {os.environ.get('OPENROUTER_VISION_MODEL', DEFAULT_VISION_MODEL)}")

    result = extract_jawi_text_llm(image_path)

    print(f"\n=== Extracted Jawi Text ===")
    print(result["raw_text"])
    print(f"\n=== Metadata ===")
    print(f"Provider     : {result['provider']}")
    print(f"Model        : {result['model']}")
    print(f"Language     : {result['language']}")
    print(f"Confidence   : {result['confidence']}")
    print(f"Process Time : {result['processing_time']}s")
    if result.get("cost_estimate_usd") is not None:
        print(f"Est. Cost    : ${result['cost_estimate_usd']}")

    cer_results = {}
    print("\n=== CER Evaluation ===")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            reference = f.read()
        cer = compute_cer(reference, result["raw_text"])
        cer_results[gt_path] = cer
        print(f"Ground Truth : {gt_path}")
        print(f"CER          : {cer:.4f} ({cer * 100:.2f}%)")
    else:
        print(f"Ground truth not found, skipping: {gt_path}")

    save_output(result, output_path, cer_results)


if __name__ == "__main__":
    main()
