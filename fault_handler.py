"""
Runtime Fault Handler.

Wraps every pipeline stage. On an exception, a timeout, or empty output it:
  1. logs the incident,
  2. passes through the last valid output instead of halting the pipeline,
  3. marks the stage as incomplete so the user/report can see it.

A stage may also be retried before it is given up on (max_retries, default 0).
Only exceptions and timeouts are retried, with exponential backoff and jitter.
Empty output is never retried: the stage ran and had nothing to say.

This is fault-tolerance infrastructure only. Quality scoring lives in
agreement_gate.py (runtime) and stage_metrics.py (offline evaluation).
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_STAGE_TIMEOUT_SECONDS = 180.0
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 10.0

_sleep = time.sleep  # indirection so tests can skip the real wait


@dataclass
class StageOutcome:
    stage: str
    status: str  # "completed" | "failed_exception" | "failed_timeout" | "failed_empty_output"
    output: Any
    duration: float
    error: str | None = None
    used_fallback: bool = False
    attempts: int = 1  # how many times the stage function was called

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
            "attempts": self.attempts,
        }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False


def backoff_delay(retry: int) -> float:
    """Seconds to wait before retry number `retry` (1-based): exponential with jitter."""
    ceiling = min(RETRY_BACKOFF_MAX_SECONDS, RETRY_BACKOFF_BASE_SECONDS * 2 ** (retry - 1))
    return random.uniform(ceiling / 2, ceiling)


def _attempt(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    timeout: float | None,
    no_retry_on: tuple[type[BaseException], ...],
) -> tuple[str, Any, str | None, bool]:
    """Call the stage function once. Returns (status, output, error, retryable)."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        output = pool.submit(func, *args, **kwargs).result(timeout=timeout)
    except FutureTimeout:
        return "failed_timeout", None, f"Stage exceeded {timeout}s timeout.", True
    except Exception as exc:  # noqa: BLE001 - the handler must catch every stage failure
        return "failed_exception", None, f"{type(exc).__name__}: {exc}", not isinstance(exc, no_retry_on)
    finally:
        pool.shutdown(wait=False)

    if _is_empty(output):
        return "failed_empty_output", None, "Stage returned empty output.", False
    return "completed", output, None, False


def run_stage(
    stage: str,
    func: Callable[..., Any],
    *args: Any,
    fallback: Any = "",
    timeout: float | None = DEFAULT_STAGE_TIMEOUT_SECONDS,
    max_retries: int = 0,
    no_retry_on: tuple[type[BaseException], ...] = (),
    **kwargs: Any,
) -> StageOutcome:
    """
    Run one pipeline stage under fault handling.

    `fallback` is the last valid output to pass through if the stage fails.
    `max_retries` extra attempts are made after an exception or a timeout, unless
    the exception type is listed in `no_retry_on`. `timeout` applies to each attempt.
    Note: on timeout the worker thread cannot be killed; its late result is discarded,
    so a retry after a timeout can overlap with the abandoned call.
    """
    started = time.time()
    attempts = 0
    while True:
        attempts += 1
        status, output, error, retryable = _attempt(func, args, kwargs, timeout, no_retry_on)
        if status == "completed" or not retryable or attempts > max_retries:
            break
        delay = backoff_delay(attempts)
        logger.warning(
            "Stage '%s' attempt %d of %d %s: %s Retrying in %.1fs.",
            stage, attempts, max_retries + 1, status, error, delay,
        )
        _sleep(delay)

    duration = round(time.time() - started, 3)
    if status == "completed":
        return StageOutcome(stage, status, output, duration, attempts=attempts)

    logger.warning("Stage '%s' %s: %s — passing through last valid output.", stage, status, error)
    return StageOutcome(stage, status, fallback, duration, error=error, used_fallback=True, attempts=attempts)
