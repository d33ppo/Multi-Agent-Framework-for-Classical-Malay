"""
Helpers for exporting multi-agent pipeline results.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pipeline_types import OCRInputRecord, PipelineResult


def export_pipeline_run(
    *,
    ocr_input: OCRInputRecord,
    result: PipelineResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, Markdown, and stage text outputs for a pipeline run."""
    base_dir = Path(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(Path(ocr_input.source_file).stem or "pipeline_input")
    run_dir = base_dir / f"{slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "pipeline_result.json"
    md_path = run_dir / "pipeline_report.md"
    jawi_path = run_dir / "01_ocr_jawi.txt"
    romanized_path = run_dir / "02_romanized.txt"
    translation_path = run_dir / "03_modern_malay_translation.txt"
    summary_path = run_dir / "04_summary.txt"
    knowledge_path = run_dir / "05_knowledge_extraction.json"

    jawi_path.write_text(result.ocr_jawi_text, encoding="utf-8")
    romanized_path.write_text(result.romanized_text, encoding="utf-8")
    translation_path.write_text(result.modern_malay_translation, encoding="utf-8")
    summary_path.write_text(result.summary, encoding="utf-8")
    knowledge_path.write_text(
        json.dumps(result.knowledge_extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(_build_markdown_report(ocr_input, result), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


def _build_markdown_report(ocr_input: OCRInputRecord, result: PipelineResult) -> str:
    return "\n".join(
        [
            "# Multi-Agent Pipeline Report",
            "",
            f"**Source file:** `{ocr_input.source_file}`",
            f"**OCR provider:** `{ocr_input.ocr_provider}`",
            f"**Agent runtime:** `{result.metadata.get('agent_runtime', 'unknown')}`",
            "",
            "## OCR Jawi Text",
            "",
            "```text",
            result.ocr_jawi_text,
            "```",
            "",
            "## Romanised Classical Malay",
            "",
            "```text",
            result.romanized_text,
            "```",
            "",
            "## Modern Malay Translation",
            "",
            "```text",
            result.modern_malay_translation,
            "```",
            "",
            "## Summary",
            "",
            result.summary,
            "",
            "## Knowledge Extraction",
            "",
            "```json",
            json.dumps(result.knowledge_extraction, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(result.metadata, indent=2, ensure_ascii=False),
            "```",
        ]
    )


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "pipeline-input"
