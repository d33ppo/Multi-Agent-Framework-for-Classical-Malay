"""
Tests for the API and configuration hardening: no silent mock output, the
external-API switch, the model allowlist, CORS origins, the text size limit,
and the OCR script settings.

Nothing here touches the network. Outbound HTTP is replaced in every test
(see tests/support.py), and each test starts from a clean set of configuration variables so a real
.env on the developer machine cannot leak in.
"""

import json
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend.main import DEFAULT_MAX_TEXT_CHARS, app, cors_origins
from llm_text_runtime import (
    DEFAULT_MODEL,
    LLMNotConfiguredError,
    LLMTextRuntime,
    allowed_models,
    default_model,
)
from tests.support import LIVE, EnvTestCase, FakeNeural, FakeResponse, fake_openrouter, make_dictionary

try:  # imported here, not inside tests, so their .env loading happens before setUp cleans the environment
    import ocr_jawi
    import ocr_llm
except ImportError:  # OCR extras (pytesseract, opencv) not installed
    ocr_jawi = ocr_llm = None

JAWI = "ايبو ڤد"


class RuntimeConfigTests(EnvTestCase):
    def test_missing_key_raises_instead_of_falling_back_to_mock(self):
        with self.assertRaises(LLMNotConfiguredError) as ctx:
            LLMTextRuntime()
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_key_without_external_api_flag_raises(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
        with self.assertRaises(LLMNotConfiguredError) as ctx:
            LLMTextRuntime()
        self.assertIn("ALLOW_EXTERNAL_MANUSCRIPT_API", str(ctx.exception))

    def test_key_and_flag_give_live_runtime(self):
        os.environ.update(LIVE)
        runtime = LLMTextRuntime()
        self.assertEqual((runtime.mode, runtime.model), ("live", DEFAULT_MODEL))

    def test_mock_is_available_only_on_request(self):
        self.assertTrue(LLMTextRuntime(provider="mock").is_mock)
        self.assertTrue(LLMTextRuntime(allow_mock_fallback=True).is_mock)

    def test_generate_text_refuses_once_the_flag_is_off(self):
        os.environ.update(LIVE)
        runtime = LLMTextRuntime()
        os.environ["ALLOW_EXTERNAL_MANUSCRIPT_API"] = "false"
        with self.assertRaises(LLMNotConfiguredError):
            runtime.generate_text(system_prompt="s", user_prompt="manuscript text")
        self.post.assert_not_called()

    def test_generate_text_sends_the_chosen_model(self):
        os.environ.update(LIVE)
        self.answer_with(lambda _url, **kwargs: FakeResponse("hello"))
        text = LLMTextRuntime(model="vendor/model").generate_text(system_prompt="s", user_prompt="u")
        self.assertEqual(text, "hello")
        self.assertEqual(self.post.call_args.kwargs["json"]["model"], "vendor/model")

    def test_allowed_models_default_and_env(self):
        self.assertEqual(allowed_models(), [DEFAULT_MODEL])
        os.environ["OPENROUTER_ALLOWED_MODELS"] = " a/b , c/d ,, a/b "
        self.assertEqual(allowed_models(), ["a/b", "c/d"])

    def test_empty_model_variable_counts_as_unset(self):
        os.environ["OPENROUTER_MODEL"] = "  "
        self.assertEqual(default_model(), DEFAULT_MODEL)


class ApiHardeningTests(EnvTestCase):
    def setUp(self):
        super().setUp()
        # Keep romanisation local and tiny: no dictionary file, no HuggingFace download.
        self.patch("romanisation_agent.default_dictionary", make_dictionary)
        self.patch("romanisation_agent.default_neural_romaniser", FakeNeural)
        self.client = TestClient(app)

    def run_pipeline(self, text=JAWI, **body):
        return self.client.post("/api/pipeline/run", json={"jawi_text": text, **body})

    @staticmethod
    def events(response):
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_missing_key_returns_503_not_mock_text(self):
        response = self.run_pipeline()
        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENROUTER_API_KEY", response.json()["detail"])
        self.assertNotIn("MOCK", response.text)
        self.assertNotIn("data: ", response.text)

    def test_external_api_flag_off_returns_503(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
        response = self.run_pipeline()
        self.assertEqual(response.status_code, 503)
        self.assertIn("ALLOW_EXTERNAL_MANUSCRIPT_API", response.json()["detail"])
        self.post.assert_not_called()

    def test_demo_mock_mode_serves_output_labelled_as_mock(self):
        os.environ["APP_MODE"] = "demo_mock"
        response = self.run_pipeline()
        self.assertEqual(response.status_code, 200)
        done = self.events(response)[-1]
        self.assertEqual((done["type"], done["mode"]), ("pipeline_done", "mock"))
        self.post.assert_not_called()

    def test_explicit_mock_model_needs_demo_mode(self):
        self.assertEqual(self.run_pipeline(model="mock").status_code, 400)
        os.environ["APP_MODE"] = "demo_mock"
        self.assertEqual(self.run_pipeline(model="mock").status_code, 200)

    def test_model_not_in_allowlist_returns_400(self):
        os.environ.update(LIVE)
        response = self.run_pipeline(model="openai/gpt-4o")
        self.assertEqual(response.status_code, 400)
        self.assertIn("not allowed", response.json()["detail"])
        self.post.assert_not_called()

    def test_model_from_env_allowlist_runs_live(self):
        os.environ.update(LIVE)
        os.environ["OPENROUTER_ALLOWED_MODELS"] = f"{DEFAULT_MODEL}, vendor/other"
        self.answer_with(fake_openrouter)
        response = self.run_pipeline(model="vendor/other")
        self.assertEqual(response.status_code, 200)
        done = self.events(response)[-1]
        self.assertEqual((done["mode"], done["model"], done["incomplete_stages"]), ("live", "vendor/other", []))
        sent_models = {call.kwargs["json"]["model"] for call in self.post.call_args_list}
        self.assertEqual(sent_models, {"vendor/other"})

    def test_omitted_model_uses_server_default(self):
        os.environ.update(LIVE)
        self.answer_with(fake_openrouter)
        done = self.events(self.run_pipeline())[-1]
        self.assertEqual(done["model"], DEFAULT_MODEL)

    def test_text_over_the_limit_returns_413(self):
        self.assertEqual(self.run_pipeline(text="ا" * (DEFAULT_MAX_TEXT_CHARS + 1)).status_code, 413)
        os.environ["MAX_TEXT_CHARS"] = "5"
        response = self.run_pipeline(text="ايبو ڤد ماس")
        self.assertEqual(response.status_code, 413)
        self.assertIn("limit is 5", response.json()["detail"])

    def test_config_endpoint_lists_allowed_models(self):
        os.environ["OPENROUTER_ALLOWED_MODELS"] = "a/b,c/d"
        os.environ["OPENROUTER_MODEL"] = "c/d"
        self.assertEqual(
            self.client.get("/api/config").json(),
            {"allowed_models": ["a/b", "c/d"], "default_model": "c/d", "demo_mock": False,
             "max_text_chars": DEFAULT_MAX_TEXT_CHARS},
        )
        os.environ["OPENROUTER_MODEL"] = "not/allowed"
        os.environ["APP_MODE"] = "demo_mock"
        config = self.client.get("/api/config").json()
        self.assertEqual((config["default_model"], config["demo_mock"]), ("a/b", True))

    def test_cors_origins_default_and_env(self):
        self.assertEqual(cors_origins(), ["http://localhost:5173", "http://localhost:8000"])
        os.environ["CORS_ALLOWED_ORIGINS"] = "https://qalam.example , http://localhost:3000"
        self.assertEqual(cors_origins(), ["https://qalam.example", "http://localhost:3000"])


@unittest.skipUnless(ocr_llm, "OCR dependencies not installed")
class OcrLlmGateTests(EnvTestCase):
    def test_disabled_by_default_even_with_a_key(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
        with mock.patch.object(ocr_llm, "_encode_image") as encode:
            with self.assertRaises(LLMNotConfiguredError) as ctx:
                ocr_llm.extract_jawi_text_llm("page.png")
        self.assertIn("ALLOW_EXTERNAL_MANUSCRIPT_API", str(ctx.exception))
        encode.assert_not_called()
        self.post.assert_not_called()

    def test_enabled_uses_vision_model_from_env(self):
        os.environ.update(LIVE)
        os.environ["OPENROUTER_VISION_MODEL"] = "vendor/vision"
        self.answer_with(lambda _url, **kwargs: FakeResponse("جاوي"))
        with mock.patch.object(ocr_llm, "_encode_image", return_value=("aGk=", "image/png")):
            result = ocr_llm.extract_jawi_text_llm("page.png")
        self.assertEqual(result["raw_text"], "جاوي")
        self.assertEqual(self.post.call_args.kwargs["json"]["model"], "vendor/vision")

    def test_empty_vision_model_falls_back_to_default(self):
        os.environ["OPENROUTER_VISION_MODEL"] = ""
        self.assertEqual(ocr_llm.vision_model(), ocr_llm.DEFAULT_VISION_MODEL)


@unittest.skipUnless(ocr_jawi, "OCR dependencies not installed")
class TesseractCommandTests(EnvTestCase):
    def test_env_variable_wins_over_path(self):
        os.environ["TESSERACT_CMD"] = '"C:\\Tools\\tesseract.exe"'
        with mock.patch("shutil.which", return_value="/usr/bin/tesseract"):
            self.assertEqual(ocr_jawi.resolve_tesseract_cmd(), "C:\\Tools\\tesseract.exe")

    def test_falls_back_to_path(self):
        with mock.patch("shutil.which", return_value="/usr/bin/tesseract"):
            self.assertEqual(ocr_jawi.resolve_tesseract_cmd(), "/usr/bin/tesseract")

    def test_missing_everywhere_is_a_clear_error(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError) as ctx:
                ocr_jawi.resolve_tesseract_cmd()
        self.assertIn("TESSERACT_CMD", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
