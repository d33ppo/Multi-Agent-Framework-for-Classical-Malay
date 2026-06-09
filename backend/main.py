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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowledge_agent import run_knowledge_agent
from llm_text_runtime import LLMTextRuntime
from quality_gate_agent import QUALITY_THRESHOLD, run_romanisation_enhanced, score_romanisation
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

SHORT_TEXT_THRESHOLD = 10  # words — below this, summary is skipped


class PipelineRequest(BaseModel):
    jawi_text: str
    model: str = "anthropic/claude-sonnet-4-6"


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

            # ── Romanisation ──────────────────────────────────────
            yield _sse({"type": "agent_start", "agent": "romanisation", "label": "Romanisation Agent"})
            t0 = time.time()
            romanized = run_romanisation_agent(request.jawi_text.strip(), runtime)
            yield _sse({"type": "agent_done", "agent": "romanisation", "label": "Romanisation Agent",
                        "result": romanized.strip(), "duration": round(time.time() - t0, 2)})

            # ── Quality Gate ──────────────────────────────────────
            yield _sse({"type": "quality_gate_check"})
            t0 = time.time()
            gate = score_romanisation(romanized.strip(), runtime)
            retried = False

            if gate["score"] < QUALITY_THRESHOLD:
                yield _sse({"type": "quality_gate_retry",
                            "score": gate["score"],
                            "reason": gate.get("reason", "")})
                romanized = run_romanisation_enhanced(request.jawi_text.strip(), runtime)
                gate = score_romanisation(romanized.strip(), runtime)
                retried = True

            yield _sse({"type": "quality_gate_done",
                        "score":    gate["score"],
                        "reason":   gate.get("reason", ""),
                        "retried":  retried,
                        "duration": round(time.time() - t0, 2)})

            # ── Translation ───────────────────────────────────────
            yield _sse({"type": "agent_start", "agent": "translation", "label": "Translation Agent"})
            t0 = time.time()
            translation = run_translation_agent(romanized.strip(), runtime)
            yield _sse({"type": "agent_done", "agent": "translation", "label": "Translation Agent",
                        "result": translation.strip(), "duration": round(time.time() - t0, 2)})

            # ── Summary + Knowledge (parallel) ────────────────────
            if routing["skip_summary"]:
                yield _sse({"type": "agent_skipped", "agent": "summary", "label": "Summary Agent",
                            "reason": "Short text — skipped by orchestrator"})
                yield _sse({"type": "agent_start", "agent": "knowledge", "label": "Knowledge Extraction"})
                t0 = time.time()
                knowledge = run_knowledge_agent(translation.strip(), runtime)
                yield _sse({"type": "agent_done", "agent": "knowledge", "label": "Knowledge Extraction",
                            "result": knowledge, "duration": round(time.time() - t0, 2)})
                summary = ""
            else:
                yield _sse({"type": "agent_start", "agent": "summary",   "label": "Summary Agent"})
                yield _sse({"type": "agent_start", "agent": "knowledge", "label": "Knowledge Extraction"})

                t_parallel = time.time()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {
                        pool.submit(run_summary_agent,   translation.strip(), runtime): ("summary",   "Summary Agent"),
                        pool.submit(run_knowledge_agent, translation.strip(), runtime): ("knowledge", "Knowledge Extraction"),
                    }
                    summary = knowledge = None
                    for future in as_completed(futures):
                        agent_id, label = futures[future]
                        result   = future.result()
                        duration = round(time.time() - t_parallel, 2)
                        if agent_id == "summary":
                            summary = result.strip()
                            yield _sse({"type": "agent_done", "agent": "summary", "label": "Summary Agent",
                                        "result": summary, "duration": duration})
                        else:
                            knowledge = result
                            yield _sse({"type": "agent_done", "agent": "knowledge", "label": "Knowledge Extraction",
                                        "result": knowledge, "duration": duration})

            yield _sse({"type": "pipeline_done",
                        "total_duration": round(time.time() - started, 2),
                        "model":          runtime.model,
                        "mode":           runtime.mode})

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
