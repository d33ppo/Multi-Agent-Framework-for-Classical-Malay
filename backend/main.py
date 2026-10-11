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

from llm_text_runtime import (
    LLMNotConfiguredError,
    LLMTextRuntime,
    _get_int_env,
    allowed_models,
    default_model,
    demo_mock_enabled,
    env_value,
)
from pipeline_runner import iter_agent_pipeline

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:8000"
DEFAULT_MAX_TEXT_CHARS = 50000
MOCK_MODEL = "mock"


def cors_origins() -> list[str]:
    """Browser origins allowed to call the API, from CORS_ALLOWED_ORIGINS (comma-separated)."""
    raw = env_value("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def max_text_chars() -> int:
    return _get_int_env("MAX_TEXT_CHARS") or DEFAULT_MAX_TEXT_CHARS


app = FastAPI(title="Classical Malay Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

SHORT_TEXT_THRESHOLD = 10  # words — below this, summary is skipped


class PipelineRequest(BaseModel):
    jawi_text: str
    model: str | None = None  # None means the server default


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _server_default_model() -> str:
    """OPENROUTER_MODEL when it is on the allowlist, otherwise the first allowlist entry."""
    allowed = allowed_models()
    configured = default_model()
    return configured if configured in allowed else allowed[0]


def _resolve_model(requested: str | None) -> str:
    """Return the model to run, or raise 400 when the client asks for one that is not allowed."""
    model = (requested or "").strip() or _server_default_model()
    if model == MOCK_MODEL:
        if demo_mock_enabled():
            return model
        raise HTTPException(
            status_code=400,
            detail="Mock mode is only available when the server runs with APP_MODE=demo_mock.",
        )
    allowed = allowed_models()
    if model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model[:100]}' is not allowed. Allowed models: {', '.join(allowed)}",
        )
    return model


def _build_runtime(model: str) -> LLMTextRuntime:
    """Create the agent runtime, or raise 503 instead of silently serving mock text."""
    if model == MOCK_MODEL:
        return LLMTextRuntime(provider="mock")
    try:
        return LLMTextRuntime(model=model, allow_mock_fallback=demo_mock_enabled())
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider is not available: {exc}") from exc


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


@app.get("/api/config")
def config() -> dict:
    """Settings the frontend needs to build its model list. Contains no secrets."""
    return {
        "allowed_models": allowed_models(),
        "default_model": _server_default_model(),
        "demo_mock": demo_mock_enabled(),
        "max_text_chars": max_text_chars(),
    }


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> StreamingResponse:
    limit = max_text_chars()
    if len(request.jawi_text) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"jawi_text is too long ({len(request.jawi_text)} characters). The limit is {limit}.",
        )
    jawi_text = request.jawi_text.strip()
    if not jawi_text:
        raise HTTPException(status_code=400, detail="jawi_text cannot be empty")

    # Checked before the stream starts so the client gets a real HTTP error status.
    runtime = _build_runtime(_resolve_model(request.model))

    def generate():
        try:
            started = time.time()

            # ── Orchestrator ──────────────────────────────────────
            routing = _orchestrate(jawi_text)
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
                jawi_text, runtime, skip_summary=routing["skip_summary"]
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
