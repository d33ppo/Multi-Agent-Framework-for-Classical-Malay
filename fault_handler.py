"""
Runtime Fault Handler.

Wraps every pipeline stage. On an exception, a timeout, or empty output it:
  1. logs the incident,
  2. passes through the last valid output instead of halting the pipeline,
  3. marks the stage as incomplete so the user/report can see it.

This is fault-tolerance infrastructure only. Quality scoring lives in
agreement_gate.py (runtime) and stage_metrics.py (offline evaluation).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_STAGE_TIMEOUT_SECONDS = 180.0


@dataclass
class StageOutcome:
    stage: str
    status: str  # "completed" | "failed_exception" | "failed_timeout" | "failed_empty_output"
    output: Any
    duration: float
    error: str | None = None
    used_fallback: bool = False

    @property
    def incomplete(self) -> bool:
        return self.status != "completed"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "incomplete": self.incomplete,
            "used_fallback": self.used_fallback,
            "error": self.error,
            "duration": self.duration,
        }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False


def run_stage(
    stage: str,
    func: Callable[..., Any],
    *args: Any,
    fallback: Any = "",
    timeout: float | None = DEFAULT_STAGE_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> StageOutcome:
    """
    Run one pipeline stage under fault handling.

    `fallback` is the last valid output to pass through if the stage fails.
    Note: on timeout the worker thread cannot be killed; its late result is discarded.
    """
    started = time.time()
    status, output, error = "completed", None, None

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        output = pool.submit(func, *args, **kwargs).result(timeout=timeout)
        if _is_empty(output):
            status, error = "failed_empty_output", "Stage returned empty output."
    except FutureTimeout:
        status, error = "failed_timeout", f"Stage exceeded {timeout}s timeout."
    except Exception as exc:  # noqa: BLE001 - the handler must catch every stage failure
        status, error = "failed_exception", f"{type(exc).__name__}: {exc}"
    finally:
        pool.shutdown(wait=False)

    duration = round(time.time() - started, 3)
    if status == "completed":
        return StageOutcome(stage, status, output, duration)

    logger.warning("Stage '%s' %s: %s — passing through last valid output.", stage, status, error)
    return StageOutcome(stage, status, fallback, duration, error=error, used_fallback=True)
