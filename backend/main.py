"""
FastAPI backend for the Classical Malay Multi-Agent Pipeline.

Run with:
    python run_server.py
or:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_text_runtime import DEFAULT_MODEL, LLMTextRuntime
from pipeline_runner import iter_agent_pipeline

app = FastAPI(title="Classical Malay Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHORT_TEXT_THRESHOLD = 10  # words — below this, summary is skipped


class PipelineRequest(BaseModel):
    jawi_text: str
    model: str = DEFAULT_MODEL


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _orchestrate(jawi_text: str) -> dict:
    """Inspect input and decide which agents to run."""
    word_count = len(jawi_text.split())
    skip_summary = word_count < SHORT_TEXT_THRESHOLD
    if skip_summary:
        label = f"Short snippet ({word_count} words) — skipping summary agent"
    else:
        label = f"Full article ({word_count} words) — running all 4 agents"
    return {"label": label, "skip_summary": skip_summary, "word_count": word_count}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> StreamingResponse:
    if not request.jawi_text.strip():
        raise HTTPException(status_code=400, detail="jawi_text cannot be empty")

    def generate():
        try:
            is_mock = request.model == "mock"
            runtime = LLMTextRuntime(
                model=None if is_mock else request.model,
                provider="mock" if is_mock else "auto",
                allow_mock_fallback=True,
            )
            started = time.time()

            # ── Orchestrator ──────────────────────────────────────
            routing = _orchestrate(request.jawi_text.strip())
            yield _sse({
                "type":         "orchestrator_decision",
                "label":        routing["label"],
                "skip_summary": routing["skip_summary"],
                "word_count":   routing["word_count"],
                "duration":     round(time.time() - started, 3),
            })

            # ── Agent stages (romanisation A+B → agreement gate → translation → summary ∥ knowledge)
            final: dict = {}
            for event in iter_agent_pipeline(
                request.jawi_text.strip(), runtime, skip_summary=routing["skip_summary"]
            ):
                if event["type"] == "pipeline_result":
                    final = event
                else:
                    yield _sse(event)

            yield _sse({"type": "pipeline_done",
                        "total_duration":     round(time.time() - started, 2),
                        "model":              runtime.model,
                        "mode":               runtime.mode,
                        "incomplete_stages":  final.get("incomplete_stages", []),
                        "needs_human_review": final.get("needs_human_review", True)})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the React/HTML frontend
_frontend = ROOT / "frontend"
if _frontend.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend)), name="assets")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(str(_frontend / "index.html"))
