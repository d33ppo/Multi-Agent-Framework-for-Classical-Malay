"""
Shared test helpers: a clean configuration for every test, and fakes that
stand in for OpenRouter and the romanisation models. Nothing here touches the
network.
"""

import json
import os
import unittest
from unittest import mock

from jawi_dictionary import JawiRumiDictionary

# Every variable the code reads. Each test starts with all of them unset, so a
# real .env on the developer machine cannot leak into a test.
CONTROLLED_ENV = (
    "OPENROUTER_API_KEY",
    "ALLOW_EXTERNAL_MANUSCRIPT_API",
    "APP_MODE",
    "OPENROUTER_ALLOWED_MODELS",
    "OPENROUTER_MODEL",
    "OPENROUTER_MAX_TOKENS",
    "OPENROUTER_VISION_MODEL",
    "MAX_TEXT_CHARS",
    "CORS_ALLOWED_ORIGINS",
    "TESSERACT_CMD",
    "MAX_UPLOAD_MB",
    "MAX_UPLOAD_PAGES",
    "PDF_RENDER_DPI",
    "OCR_PROVIDER",
    "OCR_FIXTURE_DIR",
    "STORAGE_DIR",
    "JOB_WORKERS",
)
LIVE = {"OPENROUTER_API_KEY": "test-key-not-real", "ALLOW_EXTERNAL_MANUSCRIPT_API": "true"}
KNOWLEDGE_JSON = json.dumps(
    {"people": [], "places": [], "dates": [], "organizations": [], "themes": ["ujian"], "key_claims": []}
)


class FakeResponse:
    """Stands in for an OpenRouter HTTP response."""

    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def json(self):
        return {"model": "fake", "choices": [{"finish_reason": "stop", "message": {"content": self.text}}]}


def fake_openrouter(_url, **kwargs):
    """Answer each agent by looking at its system prompt (summary and knowledge run concurrently)."""
    system_prompt = kwargs["json"]["messages"][0]["content"]
    if "translator" in system_prompt:
        return FakeResponse("ibu pada masa dahulu")
    if "summarise" in system_prompt:
        return FakeResponse("Ringkasan ujian.")
    return FakeResponse(KNOWLEDGE_JSON)


def make_dictionary():
    return JawiRumiDictionary({"ايبو": {"ibu": 500}, "ڤد": {"pada": 9000}})


class FakeNeural:
    """Method A stand-in that always answers "ibu pada"."""

    def romanise(self, jawi_text):
        return "ibu pada"


class EnvTestCase(unittest.TestCase):
    """Clean configuration for every test, and a network call that fails loudly if reached."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in CONTROLLED_ENV:
            os.environ.pop(name, None)

        post = mock.patch("requests.post", side_effect=AssertionError("unexpected network call"))
        self.post = post.start()
        self.addCleanup(post.stop)

    def answer_with(self, responder):
        self.post.side_effect = responder

    def patch(self, target, replacement):
        """Replace `target` for the length of the test."""
        patcher = mock.patch(target, replacement)
        patcher.start()
        self.addCleanup(patcher.stop)
        return replacement
