"""
Shared data models for the post-OCR multi-agent pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OCRInputRecord:
    """Canonical input contract for the downstream agent pipeline."""

    source_file: str
    ocr_provider: str
    ocr_jawi_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Final result contract for the downstream agent pipeline."""

    source_file: str
    ocr_provider: str
    ocr_jawi_text: str
    romanized_text: str
    modern_malay_translation: str
    summary: str
    knowledge_extraction: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
