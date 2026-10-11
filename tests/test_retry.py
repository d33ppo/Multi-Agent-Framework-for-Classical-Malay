"""
Tests for retries in the Runtime Fault Handler and their use by the LLM stages.

The backoff wait is replaced in every test, so nothing here sleeps for real
except the one deliberate timeout case. No network call is made: the agents
are replaced by fakes and the runtime is in mock mode.
"""

import time
import unittest
from unittest import mock

import fault_handler
import pipeline_runner
from fault_handler import StageOutcome, backoff_delay, run_stage
from jawi_dictionary import JawiRumiDictionary
from llm_text_runtime import LLMTextRuntime
from pipeline_runner import iter_agent_pipeline


class Flaky:
    """Fails a set number of times, then returns its result. Counts its calls."""

    def __init__(self, failures, result="ok", error=None):
        self.failures = failures
        self.result = result
        self.error = error or RuntimeError("API down")
        self.calls = 0

    def __call__(self, *_args):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.result


class NoWaitTestCase(unittest.TestCase):
    def setUp(self):
        sleep = mock.patch("fault_handler._sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)


class RetryTests(NoWaitTestCase):
    def test_default_is_a_single_attempt(self):
        func = Flaky(failures=1)
        outcome = run_stage("demo", func, "x", fallback="prev")
        self.assertEqual((outcome.status, outcome.attempts, func.calls), ("failed_exception", 1, 1))
        self.sleep.assert_not_called()

    def test_succeeds_on_second_attempt(self):
        func = Flaky(failures=1, result="translated")
        outcome = run_stage("demo", func, "x", fallback="prev", max_retries=2)
        self.assertEqual((outcome.status, outcome.output, outcome.attempts), ("completed", "translated", 2))
        self.assertFalse(outcome.incomplete or outcome.used_fallback)
        self.assertIsNone(outcome.error)
        self.assertEqual(self.sleep.call_count, 1)

    def test_gives_up_and_passes_through_fallback(self):
        func = Flaky(failures=99)
        outcome = run_stage("demo", func, "x", fallback="last valid", max_retries=2)
        self.assertEqual((outcome.status, outcome.output), ("failed_exception", "last valid"))
        self.assertEqual((outcome.attempts, func.calls), (3, 3))
        self.assertTrue(outcome.incomplete and outcome.used_fallback)
        self.assertIn("API down", outcome.error)
        self.assertEqual(self.sleep.call_count, 2)

    def test_empty_output_is_never_retried(self):
        func = Flaky(failures=0, result="   ")
        outcome = run_stage("demo", func, "x", fallback="prev", max_retries=2)
        self.assertEqual((outcome.status, outcome.output), ("failed_empty_output", "prev"))
        self.assertEqual((outcome.attempts, func.calls), (1, 1))
        self.sleep.assert_not_called()

    def test_timeout_is_retried(self):
        calls = []

        def slow_then_fast(_):
            calls.append(1)
            if len(calls) == 1:
                time.sleep(0.5)
            return "on time"

        outcome = run_stage("demo", slow_then_fast, "x", fallback="prev", timeout=0.1, max_retries=1)
        self.assertEqual((outcome.status, outcome.output, outcome.attempts), ("completed", "on time", 2))

    def test_timeout_on_every_attempt_gives_up(self):
        outcome = run_stage(
            "demo", lambda _: time.sleep(0.3) or "late", "x", fallback="prev", timeout=0.05, max_retries=1
        )
        self.assertEqual((outcome.status, outcome.output, outcome.attempts), ("failed_timeout", "prev", 2))

    def test_listed_exception_types_are_not_retried(self):
        func = Flaky(failures=99, error=ValueError("output still invalid"))
        outcome = run_stage("demo", func, "x", fallback="prev", max_retries=2, no_retry_on=(ValueError,))
        self.assertEqual((outcome.status, outcome.attempts, func.calls), ("failed_exception", 1, 1))
        self.sleep.assert_not_called()

    def test_backoff_is_exponential_with_jitter_and_capped(self):
        for _ in range(50):
            self.assertTrue(0.5 <= backoff_delay(1) <= 1.0)
            self.assertTrue(1.0 <= backoff_delay(2) <= 2.0)
            self.assertTrue(2.0 <= backoff_delay(3) <= 4.0)
            self.assertTrue(5.0 <= backoff_delay(20) <= fault_handler.RETRY_BACKOFF_MAX_SECONDS)
        self.assertGreater(len({backoff_delay(1) for _ in range(20)}), 1)  # jitter: not a fixed value

    def test_waits_grow_between_attempts(self):
        run_stage("demo", Flaky(failures=99), "x", max_retries=2)
        first, second = (call.args[0] for call in self.sleep.call_args_list)
        self.assertTrue(0.5 <= first <= 1.0 and 1.0 <= second <= 2.0)

    def test_to_dict_reports_attempts(self):
        self.assertEqual(StageOutcome("demo", "completed", "x", 0.1).to_dict()["attempts"], 1)
        outcome = run_stage("demo", Flaky(failures=1), "x", max_retries=1)
        self.assertEqual(outcome.to_dict()["attempts"], 2)


class CountingNeural:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def romanise(self, jawi_text):
        self.calls += 1
        if self.error:
            raise self.error
        return "ibu pada"


class PipelineRetryTests(NoWaitTestCase):
    def _events(self, neural=None):
        dictionary = JawiRumiDictionary({"ايبو": {"ibu": 500}, "ڤد": {"pada": 9000}})
        return list(iter_agent_pipeline(
            "ايبو ڤد", LLMTextRuntime(provider="mock"), neural=neural or CountingNeural(), dictionary=dictionary
        ))

    def test_translation_recovers_on_second_attempt(self):
        translate = Flaky(failures=1, result="ibu pada masa dahulu")
        with mock.patch.object(pipeline_runner, "run_translation_agent", translate):
            events = self._events()
        final = events[-1]
        self.assertEqual(final["translation"], "ibu pada masa dahulu")
        self.assertEqual(final["incomplete_stages"], [])
        self.assertEqual(final["stage_status"]["translation"]["attempts"], 2)
        done = next(e for e in events if e["type"] == "agent_done" and e["agent"] == "translation")
        self.assertEqual((done["attempts"], done["incomplete"]), (2, False))

    def test_llm_stage_gives_up_after_three_attempts(self):
        summarise = Flaky(failures=99)
        with mock.patch.object(pipeline_runner, "run_summary_agent", summarise):
            final = self._events()[-1]
        self.assertEqual(final["incomplete_stages"], ["summary"])
        self.assertEqual((final["stage_status"]["summary"]["attempts"], summarise.calls), (3, 3))
        self.assertEqual(final["stage_status"]["knowledge"]["status"], "completed")

    def test_local_romanisation_is_not_retried(self):
        neural = CountingNeural(error=RuntimeError("model missing"))
        final = self._events(neural=neural)[-1]
        self.assertEqual(neural.calls, 1)
        self.assertEqual(final["stage_status"]["romanisation_method_a"]["attempts"], 1)
        self.assertEqual(final["stage_status"]["romanisation_method_b"]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
