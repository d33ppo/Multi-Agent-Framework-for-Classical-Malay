"""
Shared test helpers: a clean configuration for every test, and fakes that
stand in for OpenRouter and the romanisation models. Nothing here touches the
network.

The agents call OpenRouter through LangChain's ChatOpenAI, which sends with
httpx. The VLM OCR baseline sends with requests. Both are replaced by the same
fake, `self.post`, so the real LangChain and OpenAI SDK code runs in tests and
only the HTTP request is faked.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import pymupdf

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
    "OCR_MODEL_PATH",
    "LINE_DETECTOR_PATH",
    "LINE_DETECTOR_IMGSZ",
    "OCR_NUM_BEAMS",
    "OCR_BATCH_SIZE",
    "OCR_TORCH_THREADS",
    "OCR_DEVICE",
    "OCR_CONFIDENCE_THRESHOLD",
    "STORAGE_DIR",
    "JOB_WORKERS",
    "LANGSMITH_TRACING",  # LangChain tracing would upload chain inputs, so it is always off in tests
    "LANGCHAIN_TRACING_V2",
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
        message = {"role": "assistant", "content": self.text}
        return {"model": "fake", "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}


def fake_openrouter(_url, **kwargs):
    """Answer each agent by looking at its system prompt (summary and knowledge run concurrently)."""
    system_prompt = kwargs["json"]["messages"][0]["content"]
    if "translator" in system_prompt:
        return FakeResponse("ibu pada masa dahulu")
    if "summarise" in system_prompt:
        return FakeResponse("Ringkasan ujian.")
    return FakeResponse(KNOWLEDGE_JSON)


def send_through(post):
    """A replacement for httpx.Client.send that hands each outbound request to `post`."""
    real_send = httpx.Client.send

    def send(client, request, **kwargs):
        if request.url.host == "testserver":  # FastAPI's TestClient, which never leaves the process
            return real_send(client, request, **kwargs)
        fake = post(str(request.url), headers=dict(request.headers), json=json.loads(request.content))
        if fake.status_code != 200:
            return httpx.Response(fake.status_code, text=fake.text, request=request)
        return httpx.Response(200, json=fake.json(), request=request)

    return send


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
        self.patch("httpx.Client.send", send_through(self.post))

    def answer_with(self, responder):
        self.post.side_effect = responder

    def patch(self, target, replacement):
        """Replace `target` for the length of the test."""
        patcher = mock.patch(target, replacement)
        patcher.start()
        self.addCleanup(patcher.stop)
        return replacement


class TempDirTestCase(EnvTestCase):
    """Adds `self.tmp`, a folder that is removed after the test."""

    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)


def make_pdf(pages=3):
    """A small PDF with the given number of pages (200 x 100 points each)."""
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page(width=200, height=100)
        page.insert_text((20, 50), f"page {number}")
    return document.tobytes()


def make_image(image_format="png"):
    """A small blank image, 40 x 20 pixels."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20), False)
    pixmap.clear_with(255)
    return pixmap.tobytes(image_format)
