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
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.datastructures import Headers

from document_storage import LocalStorage
from job_runner import JobRunner, Run
from llm_text_runtime import (
    LLMNotConfiguredError,
    LLMTextRuntime,
    _get_int_env,
    allowed_models,
    default_model,
    demo_mock_enabled,
    env_value,
)
from ocr_providers import build_ocr_provider
from pdf_render import UnreadableDocumentError, count_pages, detect_file_type
from pipeline_runner import iter_agent_pipeline, plan_agents

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:8000"
DEFAULT_MAX_TEXT_CHARS = 50000
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_UPLOAD_PAGES = 30
UPLOAD_FORM_OVERHEAD_BYTES = 64 * 1024  # multipart boundaries and the other form fields
UPLOAD_ROUTE = "/api/documents"
SSE_KEEPALIVE_SECONDS = 15
MOCK_MODEL = "mock"


def cors_origins() -> list[str]:
    """Browser origins allowed to call the API, from CORS_ALLOWED_ORIGINS (comma-separated)."""
    raw = env_value("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def max_text_chars() -> int:
    return _get_int_env("MAX_TEXT_CHARS") or DEFAULT_MAX_TEXT_CHARS


def max_upload_bytes() -> int:
    return (_get_int_env("MAX_UPLOAD_MB") or DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024


def max_upload_pages() -> int:
    return _get_int_env("MAX_UPLOAD_PAGES") or DEFAULT_MAX_UPLOAD_PAGES


def _file_too_large() -> str:
    return f"File is too large. The limit is {max_upload_bytes() // (1024 * 1024)} MB."


class UploadSizeLimit:
    """Reject an oversized upload from its Content-Length header, before the body is read."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == UPLOAD_ROUTE:
            length = Headers(scope=scope).get("content-length", "")
            if length.isdigit() and int(length) > max_upload_bytes() + UPLOAD_FORM_OVERHEAD_BYTES:
                response = JSONResponse({"detail": _file_too_large()}, status_code=413)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = FastAPI(title="Classical Malay Pipeline API", version="0.1.0")

# The last middleware added runs first, so CORS headers are also set on the 413 above.
app.add_middleware(UploadSizeLimit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_job_runner: JobRunner | None = None
_job_runner_lock = threading.Lock()


def get_job_runner() -> JobRunner:
    """The one job runner of this process, created on first use."""
    global _job_runner
    with _job_runner_lock:
        if _job_runner is None:
            _job_runner = JobRunner(LocalStorage())
        return _job_runner


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


def _get_run(run_id: str) -> Run:
    run = get_job_runner().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


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
            routing = plan_agents(jawi_text)
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


@app.post(UPLOAD_ROUTE, status_code=202)
def upload_document(file: UploadFile = File(...), model: str | None = Form(None)) -> dict:
    """Store one article (PDF, PNG or JPEG) and start its pipeline run in the background."""
    limit = max_upload_bytes()
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=_file_too_large())

    file_type = detect_file_type(data)  # from the content, not the file name or Content-Type
    if file_type is None:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF, PNG or JPEG file.")
    try:
        page_count = count_pages(data, file_type)
    except UnreadableDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    page_limit = max_upload_pages()
    if page_count > page_limit:
        raise HTTPException(
            status_code=413, detail=f"The document has {page_count} pages. The limit is {page_limit}."
        )

    # Same model and key checks as the text endpoint, before anything is stored.
    runtime = _build_runtime(_resolve_model(model))
    try:
        provider = build_ocr_provider()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"OCR is not available: {exc}") from exc

    run = get_job_runner().submit(
        data=data, file_name=file.filename, file_type=file_type, page_count=page_count,
        runtime=runtime, provider=provider,
    )
    record = run.to_dict()
    return {
        "document_id": record["document_id"],
        "run_id": record["run_id"],
        "status": record["status"],
        "file_type": file_type,
        "page_count": page_count,
        "status_url": f"/api/runs/{record['run_id']}",
        "events_url": f"/api/runs/{record['run_id']}/events",
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Status, stage statuses and outputs of one run."""
    return _get_run(run_id).to_dict()


@app.get("/api/runs/{run_id}/events")
def stream_run_events(run_id: str) -> StreamingResponse:
    """Replay the run's stage events from the start, then follow it until it finishes."""
    run = _get_run(run_id)

    def generate():
        index = 0
        while True:
            events, finished = run.events_since(index, timeout=SSE_KEEPALIVE_SECONDS)
            for event in events:
                yield _sse(event)
            index += len(events)
            if finished:
                break
            if not events:
                yield ": keep-alive\n\n"

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
