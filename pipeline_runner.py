"""
Backend-first runner for the post-OCR multi-agent pipeline.

Usage examples:
    python pipeline_runner.py fixtures/pipeline_inputs/good_qwen_sample.txt
    python pipeline_runner.py output/jawi_ocr_output-4.txt --provider auto
    python pipeline_runner.py test-data/jawi-manuscript-4.png --ocr-provider tesseract
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from knowledge_agent import run_knowledge_agent
from llm_text_runtime import LLMTextRuntime
from ocr_input_adapter import load_ocr_input
from pipeline_types import PipelineResult
from result_exporter import export_pipeline_run
from romanisation_agent import run_romanisation_agent
from summary_agent import run_summary_agent
from translation_agent import run_translation_agent


OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pipeline"


def run_pipeline(
    input_path: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    ocr_provider: str = "auto",
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, str]:
    ocr_input = load_ocr_input(input_path, ocr_provider=ocr_provider)
    runtime = LLMTextRuntime(provider=provider, model=model, allow_mock_fallback=True)

    started = time.time()
    romanized_text = run_romanisation_agent(ocr_input.ocr_jawi_text, runtime)
    modern_malay_translation = run_translation_agent(romanized_text, runtime)
    summary = run_summary_agent(modern_malay_translation, runtime)
    knowledge_extraction = run_knowledge_agent(modern_malay_translation, runtime)
    elapsed = round(time.time() - started, 3)

    result = PipelineResult(
        source_file=ocr_input.source_file,
        ocr_provider=ocr_input.ocr_provider,
        ocr_jawi_text=ocr_input.ocr_jawi_text,
        romanized_text=romanized_text.strip(),
        modern_malay_translation=modern_malay_translation.strip(),
        summary=summary.strip(),
        knowledge_extraction=knowledge_extraction,
        metadata={
            "agent_runtime": runtime.provider,
            "agent_mode": runtime.mode,
            "pipeline_version": "v0.1-post-ocr",
            "processing_time_seconds": elapsed,
            "ocr_metadata": ocr_input.metadata,
        },
    )
    return export_pipeline_run(ocr_input=ocr_input, result=result, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the post-OCR multi-agent pipeline on OCR text fixtures or local image input."
    )
    parser.add_argument("input_path", help="Path to a text fixture, OCR result file, JSON record, or image file.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model ID, e.g. google/gemini-2.5-flash-preview. Overrides OPENROUTER_MODEL env var.",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help="Use 'mock' for offline testing, 'auto' (default) for OpenRouter.",
    )
    parser.add_argument(
        "--ocr-provider",
        default="auto",
        help="OCR provider for image input. Currently supports auto/tesseract. Default: auto",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for pipeline run outputs. Default: output/pipeline",
    )

    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    artifacts = run_pipeline(
        args.input_path,
        provider=args.provider,
        model=args.model,
        ocr_provider=args.ocr_provider,
        output_dir=args.output_dir,
    )

    print("Pipeline run complete.")
    print(f"Run directory : {artifacts['run_dir']}")
    print(f"JSON output   : {artifacts['json_path']}")
    print(f"Markdown report: {artifacts['markdown_path']}")


if __name__ == "__main__":
    main()
