"""
LLM Vision-based OCR for extracting Jawi text from Classical Malay manuscripts.

Provides two vision-LLM providers as alternatives to Tesseract:
  - Claude (Anthropic) — uses the anthropic Python SDK with vision capabilities
  - ILMU (YTL AI Labs) — uses an OpenAI-compatible REST endpoint

Both providers encode the manuscript image as base64 and prompt the model
to extract Arabic-script (Jawi) text, returning results in the same dict
format as ocr_jawi.extract_jawi_text() for easy comparison.

Usage:
    python ocr_llm.py <image_path> [provider] [output_path]

Providers: claude, ilmu, zai  (default: claude)
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

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Jawi-specific prompt shared by both providers
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    # Fallback mapping for common manuscript formats
    if mime_type is None:
        ext_map = {".gif": "image/gif", ".png": "image/png",
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".tif": "image/tiff", ".tiff": "image/tiff",
                    ".bmp": "image/bmp", ".webp": "image/webp"}
        mime_type = ext_map.get(path.suffix.lower(), "image/png")

    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")

    return b64, mime_type


def _estimate_cost(provider: str, b64_data: str, output_text: str) -> float | None:
    """
    Rough cost estimate in USD.

    Claude pricing (claude-sonnet-4-20250514, as of 2025-Q2):
        Input : $3.00 / 1M tokens   (images ≈ 1 token per 750 bytes of base64)
        Output: $15.00 / 1M tokens

    ILMU pricing is not publicly documented — return None.
    """
    if provider == "claude":
        # Approximate token counts
        image_tokens = len(b64_data) / 750
        prompt_tokens = len(JAWI_EXTRACTION_PROMPT) / 4  # ~4 chars/token
        input_tokens = image_tokens + prompt_tokens
        output_tokens = len(output_text) / 4
        cost = (input_tokens / 1_000_000) * 3.00 + (output_tokens / 1_000_000) * 15.00
        return round(cost, 6)
    return None


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------


def _extract_claude(image_path: str) -> dict:
    """Extract Jawi text using Anthropic Claude's vision API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for the Claude provider.\n"
            "Install it with: pip install anthropic>=0.40.0"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
        )

    b64_data, media_type = _encode_image(image_path)

    client = anthropic.Anthropic(api_key=api_key)

    start = time.time()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": JAWI_EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    )
    elapsed = time.time() - start

    raw_text = message.content[0].text.strip()

    return {
        "raw_text": raw_text,
        "confidence": "N/A (LLM)",
        "processing_time": round(elapsed, 3),
        "language": "Jawi (via Claude)",
        "provider": "claude",
        "model": message.model,
        "cost_estimate_usd": _estimate_cost("claude", b64_data, raw_text),
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# ILMU provider
# ---------------------------------------------------------------------------


def _extract_ilmu(image_path: str) -> dict:
    """
    Extract Jawi text using ILMU (YTL AI Labs) vision API.

    Assumes an OpenAI-compatible /v1/chat/completions endpoint.
    Configure the endpoint URL via ILMU_API_URL env var.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' package is required for the ILMU provider.\n"
            "Install it with: pip install requests>=2.31.0"
        )

    api_key = os.environ.get("ILMU_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ILMU_API_KEY not set. Add it to your .env file or environment."
        )

    api_url = os.environ.get(
        "ILMU_API_URL", "https://api.ilmu.ai/v1/chat/completions"
    )
    model_name = os.environ.get("ILMU_MODEL", "ilmu-vision")

    b64_data, media_type = _encode_image(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start

    if response.status_code != 200:
        raise RuntimeError(
            f"ILMU API error {response.status_code}: {response.text}"
        )

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()

    usage = data.get("usage", {})

    return {
        "raw_text": raw_text,
        "confidence": "N/A (LLM)",
        "processing_time": round(elapsed, 3),
        "language": "Jawi (via ILMU)",
        "provider": "ilmu",
        "model": data.get("model", model_name),
        "cost_estimate_usd": _estimate_cost("ilmu", b64_data, raw_text),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", "N/A"),
            "output_tokens": usage.get("completion_tokens", "N/A"),
        },
    }


# ---------------------------------------------------------------------------
# z.AI provider
# ---------------------------------------------------------------------------


def _extract_zai(image_path: str) -> dict:
    """
    Extract Jawi text using z.AI vision API.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' package is required for the z.AI provider.\n"
            "Install it with: pip install requests>=2.31.0"
        )

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ZAI_API_KEY not set. Add it to your .env file or environment."
        )

    api_url = os.environ.get(
        "ZAI_API_URL", "https://api.z.ai/api/paas/v4/chat/completions"
    )
    model_name = os.environ.get("ZAI_MODEL", "glm-4.5")

    b64_data, media_type = _encode_image(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start

    if response.status_code != 200:
        raise RuntimeError(
            f"z.AI API error {response.status_code}: {response.text}"
        )

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()

    usage = data.get("usage", {})

    return {
        "raw_text": raw_text,
        "confidence": "N/A (LLM)",
        "processing_time": round(elapsed, 3),
        "language": "Jawi (via z.AI)",
        "provider": "zai",
        "model": data.get("model", model_name),
        "cost_estimate_usd": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", "N/A"),
            "output_tokens": usage.get("completion_tokens", "N/A"),
        },
    }


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

PROVIDERS = {
    "claude": _extract_claude,
    "ilmu": _extract_ilmu,
    "zai": _extract_zai,
}


def extract_jawi_text_llm(image_path: str, provider: str = "claude") -> dict:
    """
    Extract Jawi text from a manuscript image using an LLM vision model.

    Args:
        image_path: Path to the manuscript image.
        provider:   "claude", "ilmu", or "zai".

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


def save_output(result: dict, output_path: str) -> None:
    """Save LLM OCR result to a text file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== Jawi LLM-OCR Result ({result['provider'].upper()}) ===\n\n")
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
    print(f"Output saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test-data/jawi-manuscript-4.png"
    provider = sys.argv[2] if len(sys.argv) > 2 else "claude"

    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/jawi_llm_{provider}_{base_name}_{timestamp}_output.txt"

    # Ensure UTF-8 output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print(f"Processing : {image_path}")
    print(f"Provider   : {provider}")

    result = extract_jawi_text_llm(image_path, provider)

    print(f"\n=== Extracted Jawi Text ({provider.upper()}) ===")
    print(result["raw_text"])
    print(f"\n=== Metadata ===")
    print(f"Provider     : {result['provider']}")
    print(f"Model        : {result['model']}")
    print(f"Language     : {result['language']}")
    print(f"Confidence   : {result['confidence']}")
    print(f"Process Time : {result['processing_time']}s")
    if result.get("cost_estimate_usd") is not None:
        print(f"Est. Cost    : ${result['cost_estimate_usd']}")

    save_output(result, output_path)


if __name__ == "__main__":
    main()
