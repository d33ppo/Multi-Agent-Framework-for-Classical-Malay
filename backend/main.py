"""
FastAPI backend for the Classical Malay Multi-Agent Pipeline.

Run with:
    python run_server.py
or:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow importing the root-level pipeline modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowledge_agent import run_knowledge_agent
from llm_text_runtime import LLMTextRuntime
from romanisation_agent import run_romanisation_agent
from summary_agent import run_summary_agent
from translation_agent import run_translation_agent

app = FastAPI(title="Classical Malay Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    jawi_text: str
    model: str = "google/gemini-2.5-flash-preview"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict:
    if not request.jawi_text.strip():
        raise HTTPException(status_code=400, detail="jawi_text cannot be empty")

    try:
        is_mock = request.model == "mock"
        runtime = LLMTextRuntime(
            model=None if is_mock else request.model,
            provider="mock" if is_mock else "auto",
            allow_mock_fallback=True,
        )
        started = time.time()

        romanized = run_romanisation_agent(request.jawi_text.strip(), runtime)
        translation = run_translation_agent(romanized, runtime)
        summary = run_summary_agent(translation, runtime)
        knowledge = run_knowledge_agent(translation, runtime)

        elapsed = round(time.time() - started, 2)

        return {
            "success": True,
            "result": {
                "ocr_jawi_text": request.jawi_text.strip(),
                "romanized_text": romanized.strip(),
                "modern_malay_translation": translation.strip(),
                "summary": summary.strip(),
                "knowledge_extraction": knowledge,
            },
            "metadata": {
                "provider": runtime.provider,
                "model": runtime.model,
                "mode": runtime.mode,
                "processing_time_seconds": elapsed,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Serve the React/HTML frontend
_frontend = ROOT / "frontend"
if _frontend.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend)), name="assets")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(str(_frontend / "index.html"))
