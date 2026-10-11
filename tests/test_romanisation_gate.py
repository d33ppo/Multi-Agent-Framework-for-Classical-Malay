"""
Tests for SubAgent 1 (Method A + Method B), the Inter-Method Agreement Gate,
the Runtime Fault Handler, the concurrent pipeline stages, and the offline
stage metrics. All external models and LLM calls are faked.
"""

import importlib.util
import time
import unittest

from agreement_gate import AGREEMENT_THRESHOLD, character_agreement, run_agreement_gate
from fault_handler import run_stage
from jawi_dictionary import JawiRumiDictionary, build_lookup, clean_pair
from jawi_normalise import consonant_skeleton, normalise_jawi, spelling_variants, tokenise_jawi
from llm_text_runtime import LLMTextRuntime
from pipeline_runner import iter_agent_pipeline
from romanisation_agent import run_romanisation_agent
from stage_metrics import entity_f1, rubric_summary

# Small modern-Jawi lookup table used instead of the 170k-entry build.
ENTRIES = {
    "ڤد": {"pada": 9000},
    "هيدوڤ": {"hidup": 600},
    "سماڠت": {"semangat": 300},
    "ايبو": {"ibu": 500},
    "تركنل": {"terkenal": 296},
    "بوليه": {"boleh": 2000, "bulih": 5},
    "بوك": {"buka": 50, "buku": 40},
    "ساورڠ": {"seorang": 858},
    "فد": {"fdi": 8},  # noisy exact entry that must not beat ڤد -> pada
}


def make_dictionary() -> JawiRumiDictionary:
    return JawiRumiDictionary(ENTRIES)


class FakeNeural:
    def __init__(self, output="pada", delay=0.0, error=None):
        self.output, self.delay, self.error = output, delay, error

    def romanise(self, jawi_text):
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.output


class FailingDictionary:
    def romanise(self, jawi_text):
        raise FileNotFoundError("lookup table missing")


class NormalisationTests(unittest.TestCase):
    def test_keheh_and_presentation_forms_are_unified(self):
        self.assertEqual(normalise_jawi("ک"), "ك")  # keheh -> kaf
        self.assertEqual(normalise_jawi("ﭺ"), "چ")  # ca presentation form -> tcheh
        self.assertEqual(normalise_jawi("أ"), "ا")  # alef hamza -> alef

    def test_tokeniser_marks_reduplication_and_keeps_separators(self):
        tokens = tokenise_jawi("ايبو٢، ڤد")
        self.assertEqual([t.text for t in tokens], ["ايبو", "، ", "ڤد"])
        self.assertTrue(tokens[0].reduplicated)
        self.assertFalse(tokens[1].is_word)

    def test_old_spelling_variants(self):
        variants = dict((rule, v) for v, rule in spelling_variants("سماڠة"))
        self.assertEqual(variants["teh_marbuta_to_teh"], "سماڠت")
        self.assertEqual(spelling_variants("ايبو"), [("ايبو", "exact")])

    def test_consonant_skeleton_matches_old_and_modern_boleh(self):
        self.assertEqual(consonant_skeleton("بوله"), consonant_skeleton("بوليه"))


class DictionaryBuildTests(unittest.TestCase):
    def test_clean_pair_strips_wiki_quote_hamza(self):
        self.assertEqual(clean_pair("'''hukum-hakam", "ءءءحكوم-حكام"), ("hukum-hakam", "حكوم-حكام"))

    def test_clean_pair_rejects_non_alphabetic(self):
        self.assertEqual(clean_pair("covid19", "كوۏيد١٩"), "rumi_not_alphabetic")

    def test_build_inverts_and_counts_with_rejection_report(self):
        entries, report = build_lookup(
            [["pada", "ڤد"], ["Pada", "ڤد"], ["x1", "س"]],
            phrase_pairs=[("ibu pada", "ايبو ڤد"), ("satu dua", "ساتو")],
        )
        self.assertEqual(entries["ڤد"], {"pada": 3})
        self.assertEqual(report["rejected"]["phrase_token_count_mismatch"], 1)
        self.assertEqual(report["rejected"]["rumi_not_alphabetic"], 1)


class DictionaryLookupTests(unittest.TestCase):
    def test_known_word(self):
        result = make_dictionary().lookup_word("ايبو")
        self.assertEqual((result.rumi, result.status), ("ibu", "known"))

    def test_old_spelling_pooling_beats_noisy_exact_entry(self):
        result = make_dictionary().lookup_word("فد")
        self.assertEqual(result.rumi, "pada")
        self.assertEqual(result.rule, "fa_to_pa")

    def test_ambiguous_word_picks_most_frequent_and_flags(self):
        result = make_dictionary().lookup_word("بوك")
        self.assertEqual((result.rumi, result.status), ("buka", "ambiguous"))
        self.assertEqual([c[0] for c in result.candidates], ["buka", "buku"])

    def test_joined_yang_prefix_is_split(self):
        result = make_dictionary().lookup_word("يڠتركنل")
        self.assertEqual(result.rumi, "yang terkenal")
        self.assertTrue(result.rule.startswith("prefix_split"))

    def test_skeleton_match_is_always_ambiguous(self):
        result = make_dictionary().lookup_word("بوله")
        self.assertEqual((result.rumi, result.status, result.rule), ("boleh", "ambiguous", "consonant_skeleton"))

    def test_unknown_word_is_flagged_and_kept_visible(self):
        output = make_dictionary().romanise("ايبو زززز")
        self.assertEqual(output.text, "ibu ⟦زززز⟧")
        self.assertEqual(len(output.unknown_words), 1)
        self.assertAlmostEqual(output.coverage, 0.5)
        self.assertEqual(output.case_log()[0]["status"], "unknown")

    def test_romanise_handles_reduplication_punctuation_and_empty_input(self):
        self.assertEqual(make_dictionary().romanise("ايبو٢، هيدوف").text, "ibu-ibu, hidup")
        self.assertEqual(make_dictionary().romanise("").text, "")


class AgreementGateTests(unittest.TestCase):
    def test_identical_texts_pass(self):
        gate = run_agreement_gate("Ibu pada, hidup.", "ibu pada hidup")
        self.assertEqual(gate.agreement, 1.0)
        self.assertTrue(gate.passed)

    def test_low_agreement_is_flagged_with_coverage_note(self):
        gate = run_agreement_gate("ibu pada hidup", "⟦زززز⟧ pada", method_b_coverage=0.5)
        self.assertLess(gate.agreement, AGREEMENT_THRESHOLD)
        self.assertEqual(gate.status, "flagged")
        self.assertIn("dictionary covered 50.0%", gate.reason)

    def test_empty_texts_score_zero(self):
        self.assertEqual(character_agreement("", ""), 0.0)


class FaultHandlerTests(unittest.TestCase):
    def test_success(self):
        outcome = run_stage("demo", lambda x: x.upper(), "ok")
        self.assertEqual((outcome.status, outcome.output, outcome.incomplete), ("completed", "OK", False))

    def test_exception_passes_through_fallback(self):
        def boom(_):
            raise RuntimeError("API down")

        outcome = run_stage("demo", boom, "x", fallback="last valid")
        self.assertEqual(outcome.status, "failed_exception")
        self.assertEqual(outcome.output, "last valid")
        self.assertIn("API down", outcome.error)
        self.assertTrue(outcome.incomplete and outcome.used_fallback)

    def test_empty_output_is_a_failure(self):
        outcome = run_stage("demo", lambda _: "   ", "x", fallback="prev")
        self.assertEqual((outcome.status, outcome.output), ("failed_empty_output", "prev"))

    def test_timeout(self):
        outcome = run_stage("demo", lambda _: time.sleep(1) or "late", "x", fallback="prev", timeout=0.1)
        self.assertEqual((outcome.status, outcome.output), ("failed_timeout", "prev"))


class RomanisationAgentTests(unittest.TestCase):
    def test_methods_agree_and_gate_passes(self):
        outcome = run_romanisation_agent("ايبو ڤد", neural=FakeNeural("ibu pada"), dictionary=make_dictionary())
        self.assertEqual(outcome.text, "ibu pada")
        self.assertTrue(outcome.gate.passed)
        self.assertFalse(outcome.needs_human_review)

    def test_methods_disagree_and_item_needs_review(self):
        outcome = run_romanisation_agent("ايبو ڤد", neural=FakeNeural("xyz qrs"), dictionary=make_dictionary())
        self.assertEqual(outcome.gate.status, "flagged")
        self.assertTrue(outcome.needs_human_review)
        self.assertEqual(outcome.text, "xyz qrs")  # Method A remains the primary output

    def test_methods_run_in_parallel(self):
        started = time.time()
        run_romanisation_agent("ايبو", neural=FakeNeural("ibu", delay=0.3), dictionary=make_dictionary())
        self.assertLess(time.time() - started, 0.6)

    def test_method_a_failure_falls_back_to_method_b(self):
        outcome = run_romanisation_agent(
            "ايبو ڤد", neural=FakeNeural(error=RuntimeError("model missing")), dictionary=make_dictionary()
        )
        self.assertEqual(outcome.text, "ibu pada")
        self.assertEqual(outcome.method_a.status, "failed_exception")
        self.assertIsNone(outcome.gate)
        self.assertTrue(outcome.needs_human_review)
        self.assertEqual(outcome.to_dict()["agreement_gate"]["status"], "skipped")

    def test_both_methods_fail_falls_back_to_ocr_text(self):
        outcome = run_romanisation_agent(
            "ايبو", neural=FakeNeural(error=RuntimeError("x")), dictionary=FailingDictionary()
        )
        self.assertEqual(outcome.text, "ايبو")
        self.assertTrue(outcome.method_b.incomplete)


class PipelineStageTests(unittest.TestCase):
    def _events(self, runtime, **kwargs):
        return list(iter_agent_pipeline(
            "ايبو ڤد", runtime, neural=FakeNeural("ibu pada"), dictionary=make_dictionary(), **kwargs
        ))

    def test_stage_order_and_result(self):
        events = self._events(LLMTextRuntime(provider="mock"))
        starts = [e["agent"] for e in events if e["type"] == "agent_start"]
        self.assertEqual(starts[:2], ["romanisation", "translation"])
        self.assertEqual(set(starts[2:]), {"summary", "knowledge"})
        gate_index = next(i for i, e in enumerate(events) if e["type"] == "agreement_gate_done")
        translation_index = next(i for i, e in enumerate(events) if e.get("agent") == "translation")
        self.assertLess(gate_index, translation_index)
        final = events[-1]
        self.assertEqual(final["type"], "pipeline_result")
        self.assertEqual(final["incomplete_stages"], [])

    def test_skip_summary(self):
        events = self._events(LLMTextRuntime(provider="mock"), skip_summary=True)
        self.assertTrue(any(e["type"] == "agent_skipped" and e["agent"] == "summary" for e in events))
        self.assertEqual(events[-1]["summary"], "")

    def test_translation_failure_passes_romanised_text_downstream(self):
        runtime = LLMTextRuntime(provider="mock")

        import pipeline_runner

        original = pipeline_runner.run_translation_agent
        pipeline_runner.run_translation_agent = lambda *_: (_ for _ in ()).throw(RuntimeError("OpenRouter 503"))
        try:
            final = self._events(runtime)[-1]
        finally:
            pipeline_runner.run_translation_agent = original

        self.assertEqual(final["translation"], "ibu pada")
        self.assertIn("translation", final["incomplete_stages"])
        self.assertIn("OpenRouter 503", final["stage_status"]["translation"]["error"])

    def test_summary_and_knowledge_run_concurrently(self):
        import pipeline_runner

        originals = pipeline_runner.run_summary_agent, pipeline_runner.run_knowledge_agent
        pipeline_runner.run_summary_agent = lambda *_: time.sleep(0.3) or "ringkasan"
        pipeline_runner.run_knowledge_agent = lambda *_: time.sleep(0.3) or {"people": []}
        try:
            started = time.time()
            self._events(LLMTextRuntime(provider="mock"))
            elapsed = time.time() - started
        finally:
            pipeline_runner.run_summary_agent, pipeline_runner.run_knowledge_agent = originals
        self.assertLess(elapsed, 0.55)


class PipelineApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import romanisation_agent
        from backend.main import app

        self._original = romanisation_agent.default_dictionary
        romanisation_agent.default_dictionary = make_dictionary
        self.client = TestClient(app)

    def tearDown(self):
        import romanisation_agent

        romanisation_agent.default_dictionary = self._original

    def _stream(self, text):
        import json

        response = self.client.post("/api/pipeline/run", json={"jawi_text": text, "model": "mock"})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_sse_stream_has_gate_and_final_status(self):
        events = self._stream("ايبو ڤد")
        types = [e["type"] for e in events]
        self.assertIn("agreement_gate_done", types)
        self.assertNotIn("error", types)
        done = events[-1]
        self.assertEqual(done["type"], "pipeline_done")
        self.assertIn("needs_human_review", done)
        self.assertIn("incomplete_stages", done)

    def test_empty_text_is_rejected(self):
        response = self.client.post("/api/pipeline/run", json={"jawi_text": "   ", "model": "mock"})
        self.assertEqual(response.status_code, 400)


class StageMetricTests(unittest.TestCase):
    def test_entity_f1(self):
        result = entity_f1(
            [{"people": ["Sukarno", "Fatimah"], "places": ["Jakarta"]}],
            [{"people": ["sukarno"], "places": ["Jakarta", "Malaya"]}],
        )
        self.assertEqual(result["per_field"]["people"], {"precision": 0.5, "recall": 1.0, "f1": 0.6667})
        self.assertEqual(result["micro"]["precision"], 0.6667)

    def test_rubric_summary_and_validation(self):
        rows = [
            {"item_id": "1", "rater": "student", "adequacy": "4", "fluency": "5"},
            {"item_id": "1", "rater": "supervisor", "adequacy": "3", "fluency": "4"},
        ]
        result = rubric_summary(rows)
        self.assertEqual(result["adequacy"]["mean"], 3.5)
        self.assertEqual(result["items"], 1)
        with self.assertRaises(ValueError):
            rubric_summary([{"item_id": "1", "rater": "a", "adequacy": "7", "fluency": "3"}])

    @unittest.skipUnless(importlib.util.find_spec("sacrebleu"), "sacrebleu not installed")
    def test_romanisation_scores_identical_text(self):
        from stage_metrics import romanisation_scores

        self.assertEqual(romanisation_scores(["ibu pada masa"], ["ibu pada masa"])["chrf"], 100.0)

    @unittest.skipUnless(importlib.util.find_spec("rouge_score"), "rouge-score not installed")
    def test_summary_scores_identical_text(self):
        from stage_metrics import summary_scores

        self.assertEqual(summary_scores(["ibu pada masa"], ["ibu pada masa"])["rouge1"], 1.0)


if __name__ == "__main__":
    unittest.main()
