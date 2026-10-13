"""
Tests for the LangChain parts of the pipeline: the ChatOpenAI call to
OpenRouter and the runnable chain Translation -> (Knowledge || Summary).

The real LangChain and OpenAI SDK code runs. Only the HTTP request is faked
(see tests/support.py), so nothing here touches the network.
"""

import os
import threading
import unittest
from unittest import mock

from langchain_core.runnables import Runnable, RunnableParallel

import pipeline_runner
from llm_text_runtime import OPENROUTER_BASE_URL, LLMTextRuntime
from pipeline_runner import build_llm_chain, iter_agent_pipeline
from tests.support import LIVE, EnvTestCase, FakeNeural, FakeResponse, fake_openrouter, make_dictionary

JAWI = "ايبو ڤد"


def run_events(runtime, **kwargs):
    return list(iter_agent_pipeline(JAWI, runtime, neural=FakeNeural(), dictionary=make_dictionary(), **kwargs))


def sequence(events):
    return [(event["type"], event.get("agent")) for event in events]


class ChatOpenAITests(EnvTestCase):
    def setUp(self):
        super().setUp()
        os.environ.update(LIVE)

    def test_chat_model_points_at_openrouter_and_leaves_retries_to_the_fault_handler(self):
        chat = LLMTextRuntime(model="vendor/model")._chat_model()
        self.assertEqual(type(chat).__name__, "ChatOpenAI")
        self.assertEqual((chat.model_name, chat.openai_api_base), ("vendor/model", OPENROUTER_BASE_URL))
        self.assertEqual((chat.max_retries, chat.request_timeout), (0, 120))

    def test_request_sent_to_openrouter(self):
        self.answer_with(lambda _url, **kwargs: FakeResponse("hello"))
        text = LLMTextRuntime(model="vendor/model").generate_text(
            system_prompt="s", user_prompt="u", max_tokens=321, temperature=0.15
        )
        self.assertEqual(text, "hello")

        url = self.post.call_args.args[0]
        body = self.post.call_args.kwargs["json"]
        headers = self.post.call_args.kwargs["headers"]
        self.assertEqual(url, f"{OPENROUTER_BASE_URL}/chat/completions")
        self.assertEqual(
            (body["model"], body["temperature"], body["max_tokens"]), ("vendor/model", 0.15, 321)
        )
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(
            body["messages"], [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        )
        self.assertEqual(headers["authorization"], "Bearer test-key-not-real")
        self.assertEqual(headers["x-title"], "Classical Malay Multi-Agent Pipeline")
        self.assertIn("http-referer", headers)

    def test_max_tokens_can_be_set_from_the_environment(self):
        os.environ["OPENROUTER_MAX_TOKENS"] = "777"
        self.answer_with(lambda _url, **kwargs: FakeResponse("hello"))
        LLMTextRuntime().generate_text(system_prompt="s", user_prompt="u", max_tokens=321)
        self.assertEqual(self.post.call_args.kwargs["json"]["max_tokens"], 777)

    def test_http_error_is_raised_once_with_the_status_and_body(self):
        self.answer_with(lambda _url, **kwargs: FakeResponse("upstream unavailable", status_code=503))
        with self.assertRaises(RuntimeError) as ctx:
            LLMTextRuntime().generate_text(system_prompt="s", user_prompt="u")
        self.assertEqual(str(ctx.exception), "OpenRouter API error 503: upstream unavailable")
        self.assertEqual(self.post.call_count, 1)  # no hidden retry inside the SDK

    def test_empty_content_is_an_error(self):
        self.answer_with(lambda _url, **kwargs: FakeResponse("   "))
        with self.assertRaises(RuntimeError) as ctx:
            LLMTextRuntime().generate_text(system_prompt="s", user_prompt="u")
        self.assertIn("empty content", str(ctx.exception))

    def test_mock_runtime_never_builds_a_chat_model(self):
        runtime = LLMTextRuntime(provider="mock")
        final = run_events(runtime)[-1]
        self.assertEqual(final["incomplete_stages"], [])
        self.assertIn("MOCK TRANSLATION", final["translation"])
        self.assertIsNone(runtime._chat)
        self.post.assert_not_called()


class ChainTests(EnvTestCase):
    def setUp(self):
        super().setUp()
        sleep = mock.patch("fault_handler._sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.mock_runtime = LLMTextRuntime(provider="mock")

    def test_chain_is_a_runnable_with_a_parallel_pair(self):
        chain = build_llm_chain(self.mock_runtime)
        self.assertIsInstance(chain, Runnable)
        pair = chain.last.mapper
        self.assertIsInstance(pair, RunnableParallel)
        self.assertEqual(set(pair.steps__), {"knowledge", "summary"})

        short = build_llm_chain(self.mock_runtime, skip_summary=True)
        self.assertEqual(set(short.last.mapper.steps__), {"knowledge"})

    def test_chain_can_be_invoked_directly(self):
        state = build_llm_chain(self.mock_runtime).invoke({"romanised": "ibu pada"})
        self.assertEqual(set(state), {"romanised", "translation", "knowledge", "summary"})
        self.assertEqual({state[name].status for name in ("translation", "knowledge", "summary")}, {"completed"})
        self.assertIn("ibu pada", state["translation"].output)

    def test_event_order(self):
        events = sequence(run_events(self.mock_runtime))
        self.assertEqual(events[:7], [
            ("agent_start", "romanisation"),
            ("agent_done", "romanisation"),
            ("agreement_gate_done", None),
            ("agent_start", "translation"),
            ("agent_done", "translation"),
            ("agent_start", "knowledge"),
            ("agent_start", "summary"),
        ])
        self.assertEqual(set(events[7:9]), {("agent_done", "knowledge"), ("agent_done", "summary")})
        self.assertEqual(events[9:], [("pipeline_result", None)])

    def test_event_order_when_summary_is_skipped(self):
        events = run_events(self.mock_runtime, skip_summary=True)
        self.assertEqual(sequence(events)[3:], [
            ("agent_start", "translation"),
            ("agent_done", "translation"),
            ("agent_skipped", "summary"),
            ("agent_start", "knowledge"),
            ("agent_done", "knowledge"),
            ("pipeline_result", None),
        ])
        self.assertEqual(events[-1]["summary"], "")
        self.assertNotIn("summary", events[-1]["stage_status"])

    def test_translation_event_arrives_before_the_pair_finishes(self):
        release = threading.Event()
        released_in_time = []

        def summary(_text, _runtime):
            released_in_time.append(release.wait(timeout=10))
            return "Ringkasan."

        def knowledge(_text, _runtime):
            released_in_time.append(release.wait(timeout=10))
            return {"themes": ["ujian"]}

        seen = []
        with mock.patch.object(pipeline_runner, "run_summary_agent", summary), \
                mock.patch.object(pipeline_runner, "run_knowledge_agent", knowledge):
            for event in iter_agent_pipeline(
                JAWI, self.mock_runtime, neural=FakeNeural(), dictionary=make_dictionary()
            ):
                seen.append((event["type"], event.get("agent")))
                if seen[-1] == ("agent_start", "summary"):
                    # The translation result is already out while both of the pair are still waiting.
                    self.assertIn(("agent_done", "translation"), seen)
                    self.assertFalse(release.is_set())
                    release.set()
        self.assertEqual(released_in_time, [True, True])
        self.assertEqual(seen[-1], ("pipeline_result", None))

    def test_summary_and_knowledge_run_at_the_same_time(self):
        together = threading.Barrier(2, timeout=10)  # passes only if both are inside at once

        def summary(_text, _runtime):
            together.wait()
            return "Ringkasan."

        def knowledge(_text, _runtime):
            together.wait()
            return {"themes": ["ujian"]}

        with mock.patch.object(pipeline_runner, "run_summary_agent", summary), \
                mock.patch.object(pipeline_runner, "run_knowledge_agent", knowledge):
            final = run_events(self.mock_runtime)[-1]
        self.assertEqual(final["incomplete_stages"], [])
        self.assertEqual((final["summary"], final["knowledge"]), ("Ringkasan.", {"themes": ["ujian"]}))

    def test_pair_receives_the_translation_and_translation_receives_the_romanised_text(self):
        received = {}

        def record(name, result):
            def agent(text, _runtime):
                received[name] = text
                return result
            return agent

        with mock.patch.object(pipeline_runner, "run_translation_agent", record("translation", " ibu dahulu \n")), \
                mock.patch.object(pipeline_runner, "run_summary_agent", record("summary", "Ringkasan.")), \
                mock.patch.object(pipeline_runner, "run_knowledge_agent", record("knowledge", {"themes": ["x"]})):
            final = run_events(self.mock_runtime)[-1]
        self.assertEqual(received, {"translation": "ibu pada", "summary": "ibu dahulu", "knowledge": "ibu dahulu"})
        self.assertEqual(final["translation"], "ibu dahulu")

    def test_live_chain_retries_through_the_fault_handler_only(self):
        os.environ.update(LIVE)

        def translation_down(url, **kwargs):
            if "translator" in kwargs["json"]["messages"][0]["content"]:
                return FakeResponse("upstream unavailable", status_code=503)
            return fake_openrouter(url, **kwargs)

        self.answer_with(translation_down)
        events = run_events(LLMTextRuntime())
        final = events[-1]

        translation_calls = [
            call for call in self.post.call_args_list
            if "translator" in call.kwargs["json"]["messages"][0]["content"]
        ]
        self.assertEqual(len(translation_calls), 3)  # 1 attempt + 2 fault handler retries, none from the SDK
        status = final["stage_status"]["translation"]
        self.assertEqual((status["status"], status["attempts"], status["used_fallback"]), ("failed_exception", 3, True))
        self.assertEqual(status["error"], "RuntimeError: OpenRouter API error 503: upstream unavailable")
        self.assertEqual(final["translation"], "ibu pada")  # the romanised text, passed through
        self.assertEqual(final["incomplete_stages"], ["translation"])
        self.assertEqual(final["summary"], "Ringkasan ujian.")
        self.assertEqual(final["knowledge"]["themes"], ["ujian"])

    def test_live_chain_completes_all_three_llm_stages(self):
        os.environ.update(LIVE)
        self.answer_with(fake_openrouter)
        final = run_events(LLMTextRuntime(model="vendor/other"))[-1]
        self.assertEqual(final["incomplete_stages"], [])
        self.assertEqual((final["translation"], final["summary"]), ("ibu pada masa dahulu", "Ringkasan ujian."))
        self.assertEqual(self.post.call_count, 3)
        self.assertEqual({call.kwargs["json"]["model"] for call in self.post.call_args_list}, {"vendor/other"})


if __name__ == "__main__":
    unittest.main()
