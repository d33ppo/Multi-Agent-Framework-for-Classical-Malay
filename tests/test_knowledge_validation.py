"""
Tests for knowledge extraction output validation, the single repair call, and
evidence quote filtering. The LLM is a scripted fake, so no network call is made.
"""

import json
import unittest
from unittest import mock

import pipeline_runner
from jawi_dictionary import JawiRumiDictionary
from knowledge_agent import (
    SCHEMA_KEYS,
    KnowledgeValidationError,
    run_knowledge_agent,
    validate_knowledge,
)
from llm_text_runtime import LLMTextRuntime
from pipeline_runner import iter_agent_pipeline

SOURCE = "Sukarno melawat Kuala Lumpur pada tahun 1955.\nBeliau bertemu Tunku Abdul Rahman."


def knowledge(**overrides):
    data = {
        "people": ["Sukarno"],
        "places": ["Kuala Lumpur"],
        "dates": ["1955"],
        "organizations": [],
        "themes": ["lawatan"],
        "key_claims": [],
    }
    data.update(overrides)
    return data


def as_json(**overrides):
    return json.dumps(knowledge(**overrides), ensure_ascii=False)


class ScriptedRuntime:
    """Looks like a live runtime: returns queued responses and records the prompts it was sent."""

    is_mock = False

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_text(self, *, system_prompt, user_prompt, max_tokens=0, temperature=0.0):
        self.prompts.append(user_prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ValidationTests(unittest.TestCase):
    def test_valid_output_has_schema_keys_and_evidence_only(self):
        runtime = ScriptedRuntime(as_json())
        result = run_knowledge_agent(SOURCE, runtime)
        self.assertEqual(set(result), set(SCHEMA_KEYS) | {"evidence"})
        self.assertEqual((result["people"], result["evidence"]), (["Sukarno"], []))
        self.assertEqual(len(runtime.prompts), 1)

    def test_unknown_keys_are_dropped_and_single_string_becomes_a_list(self):
        result = validate_knowledge(
            knowledge(people="Sukarno", relationships=[{"a": "b"}], confidence=0.9), SOURCE
        )
        self.assertEqual(result["people"], ["Sukarno"])
        self.assertNotIn("relationships", result)
        self.assertNotIn("confidence", result)

    def test_items_are_stripped_and_blank_items_removed(self):
        result = validate_knowledge(knowledge(places=["  Kuala Lumpur ", "", "   "]), SOURCE)
        self.assertEqual(result["places"], ["Kuala Lumpur"])

    def test_code_fenced_json_is_accepted(self):
        runtime = ScriptedRuntime(f"```json\n{as_json()}\n```")
        self.assertEqual(run_knowledge_agent(SOURCE, runtime)["dates"], ["1955"])

    def test_missing_key_is_invalid(self):
        data = knowledge()
        del data["themes"]
        with self.assertRaises(KnowledgeValidationError) as ctx:
            validate_knowledge(data, SOURCE)
        self.assertIn("themes", str(ctx.exception))

    def test_values_that_are_not_lists_of_strings_are_invalid(self):
        for bad in ([1955], [{"name": "Sukarno"}], {"name": "Sukarno"}, None, 7):
            with self.subTest(bad=bad), self.assertRaises(KnowledgeValidationError):
                validate_knowledge(knowledge(dates=bad), SOURCE)

    def test_top_level_must_be_an_object(self):
        with self.assertRaises(KnowledgeValidationError):
            validate_knowledge(["Sukarno"], SOURCE)


class RepairTests(unittest.TestCase):
    def test_unparseable_output_is_repaired_with_one_extra_call(self):
        runtime = ScriptedRuntime("Maaf, berikut ialah hasilnya: people = Sukarno", as_json())
        result = run_knowledge_agent(SOURCE, runtime)
        self.assertEqual(result["people"], ["Sukarno"])
        self.assertEqual(len(runtime.prompts), 2)
        repair_prompt = runtime.prompts[1]
        self.assertIn("people = Sukarno", repair_prompt)  # the bad output is sent back
        self.assertIn("valid JSON only", repair_prompt)

    def test_schema_error_is_repaired_too(self):
        runtime = ScriptedRuntime(json.dumps({"people": ["Sukarno"]}), as_json())
        self.assertEqual(run_knowledge_agent(SOURCE, runtime)["places"], ["Kuala Lumpur"])
        self.assertIn("Missing schema keys", runtime.prompts[1])

    def test_invalid_twice_raises_after_exactly_one_repair_call(self):
        runtime = ScriptedRuntime("not json", "still not json", as_json())
        with self.assertRaises(KnowledgeValidationError) as ctx:
            run_knowledge_agent(SOURCE, runtime)
        self.assertIn("after one repair call", str(ctx.exception))
        self.assertEqual(len(runtime.prompts), 2)

    def test_network_error_during_repair_is_not_a_validation_error(self):
        runtime = ScriptedRuntime("not json", RuntimeError("OpenRouter API error 503"))
        with self.assertRaises(RuntimeError) as ctx:
            run_knowledge_agent(SOURCE, runtime)
        self.assertNotIsInstance(ctx.exception, KnowledgeValidationError)


class EvidenceTests(unittest.TestCase):
    def test_only_quotes_found_in_the_input_text_are_kept(self):
        evidence = [
            {"entity": "Sukarno", "quote": "Sukarno melawat Kuala Lumpur"},  # word for word
            {"entity": "1955", "quote": "tahun 1955. Beliau bertemu"},  # spans a line break in the source
            {"entity": "Sukarno", "quote": "Sukarno pergi ke Kuala Lumpur"},  # paraphrase
            {"entity": "Sukarno", "quote": "sukarno melawat"},  # wrong case
            {"entity": "Jakarta", "quote": "Jakarta"},  # not in the text at all
        ]
        result = validate_knowledge(knowledge(evidence=evidence), SOURCE)
        self.assertEqual(
            result["evidence"],
            [
                {"entity": "Sukarno", "quote": "Sukarno melawat Kuala Lumpur"},
                {"entity": "1955", "quote": "tahun 1955. Beliau bertemu"},
            ],
        )

    def test_malformed_evidence_is_dropped_without_failing(self):
        evidence = [
            "Sukarno melawat",
            {"entity": "Sukarno"},
            {"entity": "Sukarno", "quote": 1955},
            {"entity": "", "quote": "Sukarno melawat"},
            {"entity": "Sukarno", "quote": "   "},
        ]
        self.assertEqual(validate_knowledge(knowledge(evidence=evidence), SOURCE)["evidence"], [])
        self.assertEqual(validate_knowledge(knowledge(evidence="Sukarno melawat"), SOURCE)["evidence"], [])

    def test_mock_mode_has_the_same_shape(self):
        result = run_knowledge_agent(SOURCE, LLMTextRuntime(provider="mock"))
        self.assertEqual(validate_knowledge(result, SOURCE), result)


class PipelineKnowledgeTests(unittest.TestCase):
    def setUp(self):
        for target, replacement in (
            ("fault_handler._sleep", mock.Mock()),
            ("pipeline_runner.run_translation_agent", lambda *_: SOURCE),
            ("pipeline_runner.run_summary_agent", lambda *_: "Ringkasan."),
        ):
            patcher = mock.patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _final(self, runtime):
        class Neural:
            def romanise(self, jawi_text):
                return "ibu pada"

        dictionary = JawiRumiDictionary({"ايبو": {"ibu": 500}, "ڤد": {"pada": 9000}})
        return list(iter_agent_pipeline("ايبو ڤد", runtime, neural=Neural(), dictionary=dictionary))[-1]

    def test_invalid_twice_marks_the_stage_incomplete_without_stage_retries(self):
        runtime = ScriptedRuntime("not json", "still not json", as_json(), as_json())
        final = self._final(runtime)
        status = final["stage_status"]["knowledge"]
        self.assertEqual(final["incomplete_stages"], ["knowledge"])
        self.assertEqual(final["knowledge"], {})
        self.assertEqual((status["status"], status["attempts"]), ("failed_exception", 1))
        self.assertIn("KnowledgeValidationError", status["error"])
        self.assertEqual(len(runtime.prompts), 2)  # one extraction call and one repair call, nothing more
        self.assertEqual(final["summary"], "Ringkasan.")

    def test_network_error_is_still_retried_by_the_fault_handler(self):
        runtime = ScriptedRuntime(RuntimeError("OpenRouter API error 503"), as_json())
        final = self._final(runtime)
        self.assertEqual(final["incomplete_stages"], [])
        self.assertEqual(final["stage_status"]["knowledge"]["attempts"], 2)
        self.assertEqual(final["knowledge"]["people"], ["Sukarno"])

    def test_evidence_is_checked_against_the_translation(self):
        evidence = [
            {"entity": "Tunku Abdul Rahman", "quote": "Beliau bertemu Tunku Abdul Rahman"},
            {"entity": "Tunku Abdul Rahman", "quote": "Tunku menyambut beliau"},
        ]
        final = self._final(ScriptedRuntime(as_json(evidence=evidence)))
        self.assertEqual(
            final["knowledge"]["evidence"],
            [{"entity": "Tunku Abdul Rahman", "quote": "Beliau bertemu Tunku Abdul Rahman"}],
        )


if __name__ == "__main__":
    unittest.main()
