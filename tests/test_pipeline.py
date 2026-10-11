import os
import unittest
from pathlib import Path
import shutil

from llm_text_runtime import (
    LLMTextRuntime,
    _extract_openai_compatible_text,
    _get_int_env,
    _load_env_file,
)
from ocr_input_adapter import load_ocr_input
from pipeline_runner import run_pipeline
from jawi_dictionary import JawiRumiDictionary
from romanisation_agent import MockNeuralRomaniser, run_romanisation_agent


ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_get_int_env_reads_first_valid_positive_value(self):
        os.environ["PIPELINE_TEST_INT_A"] = "invalid"
        os.environ["PIPELINE_TEST_INT_B"] = "3000"
        try:
            self.assertEqual(_get_int_env("PIPELINE_TEST_INT_A", "PIPELINE_TEST_INT_B"), 3000)
        finally:
            os.environ.pop("PIPELINE_TEST_INT_A", None)
            os.environ.pop("PIPELINE_TEST_INT_B", None)

    def test_extract_openai_compatible_text_from_string(self):
        data = {
            "model": "demo-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "  hello world  "},
                }
            ],
        }
        self.assertEqual(_extract_openai_compatible_text(data), "hello world")

    def test_extract_openai_compatible_text_from_content_parts(self):
        data = {
            "model": "demo-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {"type": "output_text", "text": "first line"},
                            {"type": "output_text", "text": "second line"},
                        ]
                    },
                }
            ],
        }
        self.assertEqual(
            _extract_openai_compatible_text(data),
            "first line\nsecond line",
        )

    def test_extract_openai_compatible_text_length_error_is_actionable(self):
        data = {
            "model": "demo-model",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None},
                }
            ],
        }
        with self.assertRaises(RuntimeError) as ctx:
            _extract_openai_compatible_text(data)
        self.assertIn("max_tokens", str(ctx.exception))

    def test_load_env_file_fallback(self):
        env_name = "PIPELINE_TEST_ENV_FALLBACK"
        os.environ.pop(env_name, None)

        temp_dir = ROOT / "output" / "test_env_loader_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            env_path = temp_dir / ".env"
            env_path.write_text(f"{env_name}=fallback-ok\n", encoding="utf-8")
            _load_env_file(env_path)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        self.assertEqual(os.environ.get(env_name), "fallback-ok")
        os.environ.pop(env_name, None)

    def test_load_plain_text_fixture(self):
        record = load_ocr_input(str(ROOT / "data" / "input" / "pipeline_inputs" / "good_qwen_sample.txt"))
        self.assertEqual(record.ocr_provider, "qwen-fixture")
        self.assertIn("حکايت", record.ocr_jawi_text)

    def test_parse_saved_ocr_output(self):
        record = load_ocr_input(str(ROOT / "data" / "output" / "jawi_text_llm_&_tesseract" / "jawi_ocr_output-4.txt"))
        self.assertEqual(record.ocr_provider, "tesseract")
        self.assertIn("confidence", record.metadata)

    def test_mock_romanisation(self):
        outcome = run_romanisation_agent(
            "حکايت",
            neural=MockNeuralRomaniser(),
            dictionary=JawiRumiDictionary({"حكايت": {"hikayat": 10}}),
        )
        self.assertIn("MOCK", outcome.text)
        self.assertEqual(outcome.method_b.output.text, "hikayat")

    def test_mock_pipeline_generates_artifacts(self):
        tmpdir = ROOT / "output" / "test_pipeline_temp"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        try:
            artifacts = run_pipeline(
                str(ROOT / "data" / "input" / "pipeline_inputs" / "good_qwen_sample.txt"),
                provider="mock",
                output_dir=str(tmpdir),
                dictionary=JawiRumiDictionary({}),
            )
            json_path = Path(artifacts["json_path"])
            md_path = Path(artifacts["markdown_path"])
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertTrue((json_path.parent / "02b_romanized_method_b.txt").exists())
            self.assertIn("Inter-Method Agreement Gate", md_path.read_text(encoding="utf-8"))
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
