"""
Background job runner for uploaded documents.

One upload is one article. A run takes it through these stages:
  render -> ocr -> romanisation -> agreement_gate -> translation -> summary + knowledge
Render and OCR are handled here. The agent stages reuse
pipeline_runner.iter_agent_pipeline, so the stage order and event shapes are
the same as for the text endpoint. Every stage goes through the Runtime Fault
Handler and its status is recorded on the run.

Runs execute on a small thread pool inside the API process. There is no
external queue. Each run keeps its progress events so a client can replay and
follow them, and the run record is saved to storage after every stage.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from document_storage import Storage
from fault_handler import run_stage
from llm_text_runtime import LLMTextRuntime, _get_int_env
from ocr_providers import OCRProvider, PageImage, safe_document_name
from pdf_render import iter_page_pngs, render_dpi
from pipeline_runner import iter_agent_pipeline, plan_agents

logger = logging.getLogger(__name__)

STAGES = ("render", "ocr", "romanisation", "agreement_gate", "translation", "summary", "knowledge")
AGENT_LABELS = {
    "romanisation": "Romanisation Agent",
    "translation": "Translation Agent",
    "summary": "Summary Agent",
    "knowledge": "Knowledge Extraction",
}
TERMINAL_STATUSES = {"completed", "needs_review", "failed"}
GATE_STAGE_STATUS = {"passed": "completed", "flagged": "flagged", "skipped": "skipped"}
DEFAULT_JOB_WORKERS = 1
MAX_RUNS_IN_MEMORY = 50
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def job_workers() -> int:
    return _get_int_env("JOB_WORKERS") or DEFAULT_JOB_WORKERS


def document_key(document_id: str, file_type: str) -> str:
    return f"documents/{document_id}/original.{file_type}"


def run_key(run_id: str) -> str:
    return f"runs/{run_id}/run.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


class Run:
    """State of one run. Request threads may read it while the job thread updates it."""

    def __init__(self, record: dict[str, Any], events: list[dict[str, Any]] | None = None):
        self._changed = threading.Condition()
        self.record = record
        self.events = events or []

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    @property
    def finished(self) -> bool:
        return self.record["status"] in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        with self._changed:
            return copy.deepcopy(self.record)

    def snapshot(self) -> bytes:
        with self._changed:
            return json.dumps({"record": self.record, "events": self.events}, ensure_ascii=False).encode("utf-8")

    def update(self, **fields: Any) -> None:
        with self._changed:
            self.record.update(fields)

    def set_stage(self, name: str, **fields: Any) -> None:
        with self._changed:
            self.record["stages"][name].update(fields)

    def emit(self, event: dict[str, Any]) -> None:
        with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    def finish(self, status: str, final_event: dict[str, Any], **fields: Any) -> None:
        """Set the final status and add the last event in one step, so a reader sees both or neither."""
        with self._changed:
            self.record.update(status=status, finished_at=_now(), **fields)
            self.events.append(final_event)
            self._changed.notify_all()

    def events_since(self, index: int, timeout: float | None = None) -> tuple[list[dict[str, Any]], bool]:
        """Wait for events after `index`. Returns the new events and whether the run has finished."""
        with self._changed:
            self._changed.wait_for(lambda: len(self.events) > index or self.finished, timeout)
            return self.events[index:], self.finished

    def wait_finished(self, timeout: float | None = None) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self.finished, timeout)


class JobRunner:
    def __init__(self, storage: Storage, workers: int | None = None):
        self.storage = storage
        self._pool = ThreadPoolExecutor(max_workers=workers or job_workers(), thread_name_prefix="pipeline-job")
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    # ── Creating and finding runs ─────────────────────────────────

    def submit(
        self,
        *,
        data: bytes,
        file_name: str | None,
        file_type: str,
        page_count: int,
        runtime: LLMTextRuntime,
        provider: OCRProvider,
    ) -> Run:
        """Store the upload, create its run, and queue the run. Returns at once."""
        document_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.storage.save(document_key(document_id, file_type), data)

        run = Run({
            "run_id": run_id,
            "document_id": document_id,
            "file_name": file_name or "",
            "file_type": file_type,
            "page_count": page_count,
            "status": "pending",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "model": runtime.model,
            "mode": runtime.mode,
            "ocr_provider": provider.name,
            "stages": {name: {"status": "pending"} for name in STAGES},
            "incomplete_stages": [],
            "needs_human_review": False,
            "outputs": {},
            "error": None,
        })
        with self._lock:
            self._runs[run_id] = run
            self._drop_old_finished_runs()
        self._save(run)
        self._pool.submit(self._execute, run, runtime, provider)
        return run

    def get(self, run_id: str) -> Run | None:
        if not RUN_ID_PATTERN.match(run_id):
            return None
        with self._lock:
            run = self._runs.get(run_id)
        return run or self._load(run_id)

    def _load(self, run_id: str) -> Run | None:
        """Read a run that is no longer in memory from its saved record."""
        if not self.storage.exists(run_key(run_id)):
            return None
        saved = json.loads(self.storage.load(run_key(run_id)))
        run = Run(saved["record"], saved["events"])
        if not run.finished:
            # Only finished runs are dropped from memory, so the process that owned this one is gone.
            error = "The server restarted before this run finished."
            run.finish("failed", self._done_event(run, "failed", error, None), error=error, needs_human_review=True)
        return run

    def _save(self, run: Run) -> None:
        self.storage.save(run_key(run.run_id), run.snapshot(), overwrite=True)

    def _drop_old_finished_runs(self) -> None:
        finished = [run_id for run_id, run in self._runs.items() if run.finished]
        for run_id in finished[: max(0, len(self._runs) - MAX_RUNS_IN_MEMORY)]:
            del self._runs[run_id]

    # ── Running the stages ────────────────────────────────────────

    def _execute(self, run: Run, runtime: LLMTextRuntime, provider: OCRProvider) -> None:
        started = time.time()
        error = None
        try:
            run.update(status="running", started_at=_now())
            self._save(run)
            page_keys = self._render(run)
            ocr_text = self._ocr(run, page_keys, provider)
            if ocr_text.strip():
                self._agents(run, ocr_text, runtime)
            else:
                error = "No OCR text was produced, so the agent stages were skipped."
                self._skip_agents(run, error)
                stages = run.to_dict()["stages"]
                cause = stages["render"].get("error") or stages["ocr"].get("error")
                if cause:
                    error += f" Cause: {cause}"
        except Exception as exc:  # noqa: BLE001 - a run must always reach a final status
            logger.exception("Run %s failed outside the fault handler.", run.run_id)
            error = f"{type(exc).__name__}: {exc}"
        self._finish(run, error, started)

    def _finish(self, run: Run, error: str | None, started: float) -> None:
        record = run.to_dict()
        for name, stage in record["stages"].items():
            if stage["status"] == "running":  # only possible after a crash outside the fault handler
                run.set_stage(name, status="incomplete")

        gate_flagged = record["stages"]["agreement_gate"]["status"] in ("flagged", "skipped")
        if error:
            status = "failed"
        elif record["incomplete_stages"] or gate_flagged:
            status = "needs_review"
        else:
            status = "completed"
        run.finish(
            status,
            self._done_event(run, status, error, round(time.time() - started, 2)),
            error=error,
            needs_human_review=status != "completed",
        )
        self._save(run)

    @staticmethod
    def _done_event(run: Run, status: str, error: str | None, total_duration: float | None) -> dict[str, Any]:
        record = run.to_dict()
        return {
            "type": "pipeline_done",
            "run_id": record["run_id"],
            "status": status,
            "total_duration": total_duration,
            "model": record["model"],
            "mode": record["mode"],
            "incomplete_stages": record["incomplete_stages"],
            "needs_human_review": status != "completed",
            "error": error,
        }

    def _start_stage(self, run: Run, name: str, label: str) -> None:
        run.set_stage(name, status="running")
        run.emit({"type": "agent_start", "agent": name, "label": label})
        self._save(run)

    def _mark_incomplete(self, run: Run, names: list[str]) -> None:
        run.update(incomplete_stages=run.to_dict()["incomplete_stages"] + names)

    def _render(self, run: Run) -> list[str]:
        """Stage 1: turn the upload into one stored PNG per page. Returns the page keys."""
        record = run.to_dict()
        self._start_stage(run, "render", "Page Rendering")
        dpi = render_dpi(record["ocr_provider"]) if record["file_type"] == "pdf" else None
        outcome = run_stage(
            "render", self._render_pages, record["document_id"], record["file_type"], dpi, fallback=[]
        )

        run.set_stage(
            "render",
            status="incomplete" if outcome.incomplete else "completed",
            duration=outcome.duration, error=outcome.error, attempts=outcome.attempts,
            pages=len(outcome.output), dpi=dpi,
        )
        if outcome.incomplete:
            self._mark_incomplete(run, ["render"])
        run.emit({
            "type": "agent_done", "agent": "render", "label": "Page Rendering",
            "result": {"pages": len(outcome.output), "dpi": dpi},
            "duration": outcome.duration, "incomplete": outcome.incomplete,
            "error": outcome.error, "attempts": outcome.attempts,
        })
        self._save(run)
        return outcome.output

    def _render_pages(self, document_id: str, file_type: str, dpi: int | None) -> list[str]:
        data = self.storage.load(document_key(document_id, file_type))
        keys = []
        for number, png in enumerate(iter_page_pngs(data, file_type, dpi), start=1):
            key = f"documents/{document_id}/pages/page_{number:04d}.png"
            self.storage.save(key, png)
            keys.append(key)
        return keys

    def _ocr(self, run: Run, page_keys: list[str], provider: OCRProvider) -> str:
        """
        Stage 2: OCR every page and return the article text (pages joined in order).

        A page that fails is skipped and marks the stage incomplete; the other pages
        still count. The provider output is saved once per page and never changed.
        """
        if not page_keys:
            reason = "No pages were rendered."
            run.set_stage("ocr", status="skipped", reason=reason)
            run.emit({"type": "agent_skipped", "agent": "ocr", "label": "OCR", "reason": reason})
            return ""

        record = run.to_dict()
        self._start_stage(run, "ocr", "OCR")
        started = time.time()
        document_name = safe_document_name(record["file_name"])

        def recognise(number: int, key: str) -> list:
            page = PageImage(document_name, number, len(page_keys), self.storage.load(key))
            return provider.recognise_page(page)

        pages, texts = [], []
        for number, key in enumerate(page_keys, start=1):
            outcome = run_stage(f"ocr_page_{number}", recognise, number, key, fallback=[])
            lines = [line.to_dict() for line in outcome.output]
            page = {
                "page": number,
                "status": outcome.status,
                "error": outcome.error,
                "duration": outcome.duration,
                "mean_confidence": _mean([line["confidence"] for line in lines if line["confidence"] is not None]),
                "lines": lines,
            }
            self.storage.save(
                f"runs/{run.run_id}/ocr/page_{number:04d}.json",
                json.dumps({"provider": provider.name, **page}, ensure_ascii=False).encode("utf-8"),
            )
            pages.append(page)
            if lines:
                texts.append("\n".join(line["text"] for line in lines))

        ocr_text = "\n".join(texts)
        self.storage.save(f"runs/{run.run_id}/ocr/ocr_text.txt", ocr_text.encode("utf-8"))

        failed = [page for page in pages if page["status"] != "completed"]
        error = None
        if failed:
            numbers = ", ".join(str(page["page"]) for page in failed)
            error = f"OCR failed on page(s) {numbers}: {failed[0]['error']}"
            self._mark_incomplete(run, ["ocr"])
        confidences = [line["confidence"] for page in pages for line in page["lines"] if line["confidence"] is not None]
        mean_confidence = _mean(confidences)
        # Recorded for the reviewer only: the threshold is not calibrated yet, so it does not change the run status.
        low_confidence_lines = sum(1 for page in pages for line in page["lines"] if line.get("low_confidence"))
        duration = round(time.time() - started, 3)

        run.set_stage(
            "ocr",
            status="incomplete" if failed else "completed",
            duration=duration, error=error, provider=provider.name,
            pages=len(pages), failed_pages=[page["page"] for page in failed], mean_confidence=mean_confidence,
            low_confidence_lines=low_confidence_lines,
            # The resolution the provider saw; None for an image upload, which keeps its own resolution.
            render_dpi=record["stages"]["render"].get("dpi"),
        )
        run.update(outputs={"ocr_text": ocr_text, "ocr_mean_confidence": mean_confidence, "ocr_pages": pages})
        run.emit({
            "type": "agent_done", "agent": "ocr", "label": "OCR",
            "result": ocr_text,
            "duration": duration, "incomplete": bool(failed), "error": error,
            "provider": provider.name, "mean_confidence": mean_confidence,
            "pages": [
                {"page": page["page"], "status": page["status"], "lines": len(page["lines"]),
                 "mean_confidence": page["mean_confidence"]}
                for page in pages
            ],
        })
        self._save(run)
        return ocr_text

    def _agents(self, run: Run, ocr_text: str, runtime: LLMTextRuntime) -> None:
        """Stages 3 to 7: the same agent pipeline as the text endpoint, on the whole article text."""
        started = time.time()
        routing = plan_agents(ocr_text)
        run.emit({
            "type": "orchestrator_decision",
            "label": routing["label"],
            "skip_summary": routing["skip_summary"],
            "word_count": routing["word_count"],
            "duration": round(time.time() - started, 3),
        })

        final: dict[str, Any] = {}
        for event in iter_agent_pipeline(ocr_text, runtime, skip_summary=routing["skip_summary"]):
            if event["type"] == "pipeline_result":
                final = event
                continue
            self._record_agent_event(run, event)
            run.emit(event)
            if event["type"] != "agent_start":
                self._save(run)

        stage_status = final["stage_status"]
        run.set_stage(
            "romanisation",
            method_a=stage_status["romanisation_method_a"],
            method_b=stage_status["romanisation_method_b"],
        )
        self._mark_incomplete(run, final["incomplete_stages"])
        run.update(outputs={
            **run.to_dict()["outputs"],
            "romanisation": final["romanisation"],
            "translation": final["translation"],
            "summary": final["summary"],
            "knowledge": final["knowledge"],
        })

    @staticmethod
    def _record_agent_event(run: Run, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "agent_start":
            run.set_stage(event["agent"], status="running")
        elif kind == "agent_done":
            run.set_stage(
                event["agent"],
                status="incomplete" if event["incomplete"] else "completed",
                duration=event["duration"], error=event.get("error"), attempts=event.get("attempts"),
            )
        elif kind == "agent_skipped":
            run.set_stage(event["agent"], status="skipped", reason=event["reason"])
        elif kind == "agreement_gate_done":
            details = {key: value for key, value in event.items() if key not in ("type", "status")}
            run.set_stage("agreement_gate", status=GATE_STAGE_STATUS[event["status"]], gate_status=event["status"], **details)

    @staticmethod
    def _skip_agents(run: Run, reason: str) -> None:
        run.set_stage("agreement_gate", status="skipped", reason=reason)
        for agent, label in AGENT_LABELS.items():
            run.set_stage(agent, status="skipped", reason=reason)
            run.emit({"type": "agent_skipped", "agent": agent, "label": label, "reason": reason})
