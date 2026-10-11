"""
Backend-first runner for the post-OCR multi-agent pipeline.

Flow:
  Romanisation (Method A + Method B in parallel)
    -> Inter-Method Agreement Gate
      -> Translation
        -> Summarisation and Knowledge Extraction (concurrently)
Every stage is wrapped by the Runtime Fault Handler (fault_handler.run_stage).

Usage examples:
    python pipeline_runner.py data/input/pipeline_inputs/good_qwen_sample.txt --provider mock
    python pipeline_runner.py data/ground_truth/1950_07_002_1.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from fault_handler import StageOutcome, run_stage
from knowledge_agent import KnowledgeValidationError, run_knowledge_agent
from llm_text_runtime import LLMNotConfiguredError, LLMTextRuntime, live_config_problem
from ocr_input_adapter import load_ocr_input
from pipeline_types import PipelineResult
from result_exporter import export_pipeline_run
from romanisation_agent import (
    DictionaryRomaniser,
    MockNeuralRomaniser,
    Romaniser,
    run_romanisation_agent,
)
from summary_agent import run_summary_agent
from translation_agent import run_translation_agent


OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pipeline"
PIPELINE_VERSION = "v0.2-agreement-gate"

# The three LLM stages call an external API, so a failed call is retried. Local romanisation is not.
LLM_STAGE_MAX_RETRIES = 2
# Failures a retry cannot fix. Knowledge extraction already made its one repair call.
LLM_NO_RETRY_ON = (LLMNotConfiguredError, KnowledgeValidationError)
LLM_STAGE_RETRY = {"max_retries": LLM_STAGE_MAX_RETRIES, "no_retry_on": LLM_NO_RETRY_ON}


def iter_agent_pipeline(
    jawi_text: str,
    runtime: LLMTextRuntime,
    *,
    neural: Romaniser | None = None,
    dictionary: DictionaryRomaniser | None = None,
    skip_summary: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Run the agent stages and yield progress events.

    The final event has type "pipeline_result" and carries every stage output.
    """
    if neural is None and runtime.is_mock:
        neural = MockNeuralRomaniser()
    stages: dict[str, StageOutcome] = {}

    # SubAgent 1: Romanisation + Inter-Method Agreement Gate
    yield {"type": "agent_start", "agent": "romanisation", "label": "Romanisation Agent"}
    romanisation = run_romanisation_agent(jawi_text, neural=neural, dictionary=dictionary)
    stages["romanisation_method_a"] = romanisation.method_a
    stages["romanisation_method_b"] = romanisation.method_b
    yield {
        "type": "agent_done",
        "agent": "romanisation",
        "label": "Romanisation Agent",
        "result": romanisation.text,
        "duration": max(romanisation.method_a.duration, romanisation.method_b.duration),
        "incomplete": romanisation.method_a.incomplete,
        "method_b_text": romanisation.method_b.output.text if romanisation.method_b_details() else None,
    }
    yield {"type": "agreement_gate_done", **romanisation.to_dict()["agreement_gate"],
           "needs_human_review": romanisation.needs_human_review}

    # SubAgent 2: Translation (falls back to the romanised text)
    yield {"type": "agent_start", "agent": "translation", "label": "Translation Agent"}
    translation = run_stage(
        "translation", run_translation_agent, romanisation.text, runtime,
        fallback=romanisation.text, **LLM_STAGE_RETRY,
    )
    stages["translation"] = translation
    translation_text = translation.output.strip()
    yield _done_event("translation", "Translation Agent", translation, translation_text)

    # SubAgents 3 + 4: Summarisation and Knowledge Extraction run concurrently on the translation
    jobs = {"knowledge": ("Knowledge Extraction", run_knowledge_agent, {})}
    if skip_summary:
        yield {"type": "agent_skipped", "agent": "summary", "label": "Summary Agent",
               "reason": "Short text — skipped by orchestrator"}
    else:
        jobs["summary"] = ("Summary Agent", run_summary_agent, "")

    for agent, (label, _, _) in jobs.items():
        yield {"type": "agent_start", "agent": agent, "label": label}

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(
                run_stage, agent, func, translation_text, runtime, fallback=fallback, **LLM_STAGE_RETRY
            ): (agent, label)
            for agent, (label, func, fallback) in jobs.items()
        }
        for future in as_completed(futures):
            agent, label = futures[future]
            outcome = future.result()
            stages[agent] = outcome
            result = outcome.output.strip() if isinstance(outcome.output, str) else outcome.output
            yield _done_event(agent, label, outcome, result)

    summary = stages["summary"].output.strip() if "summary" in stages else ""
    yield {
        "type": "pipeline_result",
        "romanisation": romanisation.to_dict(),
        "translation": translation_text,
        "summary": summary,
        "knowledge": stages["knowledge"].output,
        "stage_status": {name: outcome.to_dict() for name, outcome in stages.items()},
        "incomplete_stages": [name for name, outcome in stages.items() if outcome.incomplete],
        "needs_human_review": romanisation.needs_human_review,
    }


def _done_event(agent: str, label: str, outcome: StageOutcome, result: Any) -> dict[str, Any]:
    return {
        "type": "agent_done",
        "agent": agent,
        "label": label,
        "result": result,
        "duration": outcome.duration,
        "incomplete": outcome.incomplete,
        "error": outcome.error,
        "attempts": outcome.attempts,
    }


def run_pipeline(
    input_path: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    ocr_provider: str = "auto",
    output_dir: str | Path = OUTPUT_DIR,
    neural: Romaniser | None = None,
    dictionary: DictionaryRomaniser | None = None,
) -> dict[str, str]:
    ocr_input = load_ocr_input(input_path, ocr_provider=ocr_provider)
    runtime = LLMTextRuntime(provider=provider, model=model, allow_mock_fallback=True)

    started = time.time()
    final: dict[str, Any] = {}
    for event in iter_agent_pipeline(ocr_input.ocr_jawi_text, runtime, neural=neural, dictionary=dictionary):
        if event["type"] == "pipeline_result":
            final = event
    elapsed = round(time.time() - started, 3)

    romanisation = final["romanisation"]
    result = PipelineResult(
        source_file=ocr_input.source_file,
        ocr_provider=ocr_input.ocr_provider,
        ocr_jawi_text=ocr_input.ocr_jawi_text,
        romanized_text=romanisation["text"],
        modern_malay_translation=final["translation"],
        summary=final["summary"],
        knowledge_extraction=final["knowledge"],
        romanisation_method_b=(romanisation["method_b"]["details"] or {}).get("text", ""),
        agreement_gate=romanisation["agreement_gate"],
        stage_status=final["stage_status"],
        needs_human_review=final["needs_human_review"],
        metadata={
            "agent_runtime": runtime.provider,
            "agent_mode": runtime.mode,
            "pipeline_version": PIPELINE_VERSION,
            "processing_time_seconds": elapsed,
            "incomplete_stages": final["incomplete_stages"],
            "method_b_cases": (romanisation["method_b"]["details"] or {}).get("cases", []),
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
        help="OpenRouter model ID, e.g. google/gemini-2.5-flash. Overrides OPENROUTER_MODEL env var.",
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

    if args.provider != "mock":
        problem = live_config_problem()
        if problem:
            print(f"Running the agents in MOCK mode: {problem}")

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
