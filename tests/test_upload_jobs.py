"""
Tests for document upload and background pipeline runs.

Uploads are generated in memory, storage and OCR fixtures live in a temporary
folder, OpenRouter is replaced by a fake, and romanisation uses a tiny
dictionary. Nothing touches the network or the real storage folder.
"""

import json
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import job_runner
from backend.main import app
from document_storage import LocalStorage
from job_runner import STAGES, JobRunner, run_key
from llm_text_runtime import DEFAULT_MODEL
from tests.support import (
    LIVE,
    FakeNeural,
    FakeResponse,
    TempDirTestCase,
    fake_openrouter,
    make_dictionary,
    make_image,
    make_pdf,
)

# Ten words over two pages, so the orchestrator runs the summary stage too.
PAGE_1 = "ايبو ڤد ايبو ڤد ايبو"
PAGE_2 = "ڤد ايبو ڤد ايبو ڤد"
AGENTS = ("romanisation", "translation", "summary", "knowledge")


class EchoNeural:
    """Method A stand-in that agrees with the test dictionary, so the agreement gate passes."""

    def romanise(self, jawi_text):
        return make_dictionary().romanise(jawi_text).text


def openrouter_with_failing_translation(url, **kwargs):
    if "translator" in kwargs["json"]["messages"][0]["content"]:
        return FakeResponse("upstream unavailable", status_code=503)
    return fake_openrouter(url, **kwargs)


class UploadTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.fixtures = self.tmp / "fixtures"
        self.fixtures.mkdir()
        self.storage = LocalStorage(self.tmp / "storage")
        self.runner = JobRunner(self.storage, workers=1)

        self.patch("backend.main._job_runner", self.runner)
        self.patch("romanisation_agent.default_dictionary", make_dictionary)
        self.patch("romanisation_agent.default_neural_romaniser", EchoNeural)
        self.patch("fault_handler._sleep", mock.Mock())  # no real backoff wait
        os.environ.update(LIVE, OCR_PROVIDER="fixture", OCR_FIXTURE_DIR=str(self.fixtures))
        self.answer_with(fake_openrouter)
        self.client = TestClient(app)
        # Registered last so it runs first: background jobs end before any fake above is removed.
        self.addCleanup(self.runner.shutdown)

    def write_fixture(self, name, text):
        (self.fixtures / name).write_text(text, encoding="utf-8")

    def upload(self, data, name="article.pdf", **form):
        files = {"file": (name, data, "application/octet-stream")}
        return self.client.post("/api/documents", files=files, data=form)

    def finished_run(self, response):
        """Wait for the uploaded run to end and return its record from the API."""
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()["run_id"]
        self.assertTrue(self.runner.get(run_id).wait_finished(timeout=30), "run did not finish")
        return self.client.get(f"/api/runs/{run_id}").json()

    def stream(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def stored_files(self):
        root = self.storage.root
        return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())

    @staticmethod
    def stage_statuses(run):
        return {name: stage["status"] for name, stage in run["stages"].items()}


class UploadValidationTests(UploadTestCase):
    def test_unsupported_file_type_is_rejected(self):
        uploads = (
            ("notes.pdf", b"plain text with a pdf file name"),
            ("scan.png", b"GIF89a not an accepted image type"),
            ("empty.pdf", b""),
        )
        for name, data in uploads:
            with self.subTest(name=name):
                response = self.upload(data, name)
                self.assertEqual(response.status_code, 400)
                self.assertIn("Unsupported file type", response.json()["detail"])
        self.assertEqual(self.stored_files(), [])

    def test_type_comes_from_the_content_not_the_file_name(self):
        self.write_fixture("scan.txt", PAGE_1)
        response = self.upload(make_image("png"), name="scan.pdf")
        run = self.finished_run(response)
        self.assertEqual((response.json()["file_type"], response.json()["page_count"]), ("png", 1))
        self.assertIn(f"documents/{run['document_id']}/original.png", self.stored_files())

    def test_oversize_file_is_rejected(self):
        os.environ["MAX_UPLOAD_MB"] = "1"
        megabyte = 1024 * 1024
        uploads = {
            "just over, found when the file is read": b"%PDF-" + b"0" * megabyte,
            "far over, refused from the Content-Length header": b"%PDF-" + b"0" * (2 * megabyte),
        }
        for case, data in uploads.items():
            with self.subTest(case=case):
                response = self.upload(data)
                self.assertEqual(response.status_code, 413)
                self.assertIn("limit is 1 MB", response.json()["detail"])
        self.assertEqual(self.stored_files(), [])

    def test_document_with_too_many_pages_is_rejected(self):
        os.environ["MAX_UPLOAD_PAGES"] = "2"
        response = self.upload(make_pdf(pages=3))
        self.assertEqual(response.status_code, 413)
        self.assertIn("3 pages", response.json()["detail"])
        self.assertEqual(self.stored_files(), [])

    def test_unreadable_pdf_is_rejected(self):
        response = self.upload(b"%PDF-1.7 this is not a real pdf")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stored_files(), [])

    def test_missing_file_field_is_rejected(self):
        self.assertEqual(self.client.post("/api/documents", data={"model": DEFAULT_MODEL}).status_code, 422)

    def test_missing_key_returns_503_and_stores_nothing(self):
        del os.environ["OPENROUTER_API_KEY"]
        response = self.upload(make_pdf(pages=1))
        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENROUTER_API_KEY", response.json()["detail"])
        self.assertEqual(self.stored_files(), [])
        self.post.assert_not_called()

    def test_model_not_on_the_allowlist_is_rejected(self):
        response = self.upload(make_pdf(pages=1), model="openai/gpt-4o")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.stored_files(), [])

    def test_unknown_ocr_provider_setting_returns_503(self):
        os.environ["OCR_PROVIDER"] = "magic"
        response = self.upload(make_pdf(pages=1))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.stored_files(), [])


class UploadRunTests(UploadTestCase):
    def upload_two_page_article(self):
        self.write_fixture("article_1.txt", PAGE_1)
        self.write_fixture("article_2.txt", PAGE_2)
        return self.upload(make_pdf(pages=2))

    def test_job_completes_with_fixture_ocr_and_mocked_llm(self):
        response = self.upload_two_page_article()
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual((body["file_type"], body["page_count"]), ("pdf", 2))

        events = self.stream(body["events_url"])  # returns when the run has finished
        run = self.client.get(body["status_url"]).json()

        self.assertEqual(run["status"], "completed")
        self.assertEqual((run["mode"], run["model"], run["ocr_provider"]), ("live", DEFAULT_MODEL, "fixture"))
        self.assertEqual(self.stage_statuses(run), dict.fromkeys(STAGES, "completed"))
        self.assertEqual((run["incomplete_stages"], run["needs_human_review"], run["error"]), ([], False, None))
        self.assertIsNotNone(run["finished_at"])
        self.assertGreaterEqual(run["stages"]["agreement_gate"]["agreement"], 0.85)

        outputs = run["outputs"]
        self.assertEqual(outputs["ocr_text"], f"{PAGE_1}\n{PAGE_2}")
        self.assertEqual(outputs["ocr_pages"][0]["lines"], [{"text": PAGE_1, "box": None, "confidence": None}])
        self.assertEqual(outputs["romanisation"]["text"], "ibu pada ibu pada ibu\npada ibu pada ibu pada")
        self.assertEqual(outputs["translation"], "ibu pada masa dahulu")
        self.assertEqual(outputs["summary"], "Ringkasan ujian.")
        self.assertEqual(outputs["knowledge"]["themes"], ["ujian"])

        order = [(event["type"], event.get("agent")) for event in events]
        self.assertEqual(
            order[:6],
            [
                ("agent_start", "render"), ("agent_done", "render"),
                ("agent_start", "ocr"), ("agent_done", "ocr"),
                ("orchestrator_decision", None), ("agent_start", "romanisation"),
            ],
        )
        self.assertLess(order.index(("agreement_gate_done", None)), order.index(("agent_start", "translation")))
        self.assertEqual(events[1]["result"], {"pages": 2, "dpi": 300})
        self.assertEqual(events[3]["result"], outputs["ocr_text"])
        done = events[-1]
        self.assertEqual((done["type"], done["status"], done["run_id"]), ("pipeline_done", "completed", run["run_id"]))

    def test_original_ocr_output_is_stored_once_and_cannot_be_overwritten(self):
        run = self.finished_run(self.upload_two_page_article())
        document, run_folder = f"documents/{run['document_id']}", f"runs/{run['run_id']}"
        self.assertEqual(
            self.stored_files(),
            [
                f"{document}/original.pdf",
                f"{document}/pages/page_0001.png",
                f"{document}/pages/page_0002.png",
                f"{run_folder}/ocr/ocr_text.txt",
                f"{run_folder}/ocr/page_0001.json",
                f"{run_folder}/ocr/page_0002.json",
                f"{run_folder}/run.json",
            ],
        )
        ocr_key = f"{run_folder}/ocr/ocr_text.txt"
        self.assertEqual(self.storage.load(ocr_key).decode("utf-8"), f"{PAGE_1}\n{PAGE_2}")
        with self.assertRaises(FileExistsError):
            self.storage.save(ocr_key, "edited".encode("utf-8"))
        page = json.loads(self.storage.load(f"{run_folder}/ocr/page_0002.json"))
        self.assertEqual((page["provider"], page["page"], page["lines"][0]["text"]), ("fixture", 2, PAGE_2))

    def test_failing_stage_marks_the_run_for_review_but_the_run_still_finishes(self):
        self.answer_with(openrouter_with_failing_translation)
        response = self.upload_two_page_article()
        run = self.finished_run(response)

        self.assertEqual(run["status"], "needs_review")
        self.assertEqual(run["incomplete_stages"], ["translation"])
        self.assertTrue(run["needs_human_review"])
        translation = run["stages"]["translation"]
        self.assertEqual((translation["status"], translation["attempts"]), ("incomplete", 3))
        self.assertIn("503", translation["error"])
        statuses = self.stage_statuses(run)
        self.assertEqual((statuses["summary"], statuses["knowledge"]), ("completed", "completed"))
        # The romanised text was passed through as the last valid output.
        self.assertEqual(run["outputs"]["translation"], run["outputs"]["romanisation"]["text"])
        self.assertEqual(run["outputs"]["summary"], "Ringkasan ujian.")

        done = self.stream(response.json()["events_url"])[-1]
        self.assertEqual((done["type"], done["status"], done["incomplete_stages"]),
                         ("pipeline_done", "needs_review", ["translation"]))

    def test_flagged_agreement_gate_gives_needs_review(self):
        self.patch("romanisation_agent.default_neural_romaniser", FakeNeural)  # always "ibu pada"
        run = self.finished_run(self.upload_two_page_article())
        self.assertEqual((run["status"], run["incomplete_stages"]), ("needs_review", []))
        self.assertEqual(run["stages"]["agreement_gate"]["status"], "flagged")

    def test_one_failed_page_does_not_stop_the_run(self):
        self.write_fixture("article_1.txt", PAGE_1)  # no fixture for page 2
        run = self.finished_run(self.upload(make_pdf(pages=2)))

        self.assertEqual(run["status"], "needs_review")
        self.assertEqual(run["incomplete_stages"], ["ocr"])
        ocr = run["stages"]["ocr"]
        self.assertEqual((ocr["status"], ocr["failed_pages"]), ("incomplete", [2]))
        self.assertIn("article_2.txt", ocr["error"])
        self.assertEqual(run["outputs"]["ocr_text"], PAGE_1)
        self.assertEqual(run["stages"]["translation"]["status"], "completed")

    def test_no_ocr_text_fails_the_run_and_skips_the_agents(self):
        response = self.upload(make_pdf(pages=2))  # no fixtures at all
        run = self.finished_run(response)

        self.assertEqual(run["status"], "failed")
        self.assertIn("No OCR text was produced", run["error"])
        statuses = self.stage_statuses(run)
        self.assertEqual((statuses["render"], statuses["ocr"]), ("completed", "incomplete"))
        for stage in ("agreement_gate", *AGENTS):
            self.assertEqual(statuses[stage], "skipped")
        self.post.assert_not_called()

        events = self.stream(response.json()["events_url"])
        skipped = [event["agent"] for event in events if event["type"] == "agent_skipped"]
        self.assertEqual(skipped, list(AGENTS))
        self.assertEqual((events[-1]["type"], events[-1]["status"]), ("pipeline_done", "failed"))

    def test_render_failure_fails_the_run(self):
        def broken_renderer(*_args, **_kwargs):
            raise RuntimeError("renderer crashed")

        self.patch("job_runner.iter_page_pngs", broken_renderer)
        run = self.finished_run(self.upload_two_page_article())
        statuses = self.stage_statuses(run)
        self.assertEqual((run["status"], statuses["render"], statuses["ocr"]), ("failed", "incomplete", "skipped"))
        self.assertIn("renderer crashed", run["error"])
        self.assertEqual(run["incomplete_stages"], ["render"])

    def test_demo_mock_mode_runs_without_a_key(self):
        del os.environ["OPENROUTER_API_KEY"]
        os.environ["APP_MODE"] = "demo_mock"
        run = self.finished_run(self.upload_two_page_article())
        self.assertEqual(run["mode"], "mock")
        self.assertIn(run["status"], ("completed", "needs_review"))
        self.assertIn("MOCK", run["outputs"]["translation"])
        self.post.assert_not_called()

    def test_text_endpoint_still_works(self):
        response = self.client.post("/api/pipeline/run", json={"jawi_text": f"{PAGE_1} {PAGE_2}"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[0]["type"], "orchestrator_decision")
        self.assertFalse(events[0]["skip_summary"])
        self.assertEqual((events[-1]["type"], events[-1]["incomplete_stages"]), ("pipeline_done", []))


class RunLookupTests(UploadTestCase):
    def test_unknown_run_is_404(self):
        for run_id in ("0" * 32, "not-a-run-id", "..%2F..%2Fsecret"):
            with self.subTest(run_id=run_id):
                self.assertEqual(self.client.get(f"/api/runs/{run_id}").status_code, 404)
                self.assertEqual(self.client.get(f"/api/runs/{run_id}/events").status_code, 404)

    def test_finished_run_can_be_read_after_a_restart(self):
        self.write_fixture("article_1.txt", PAGE_1)
        self.write_fixture("article_2.txt", PAGE_2)
        run = self.finished_run(self.upload(make_pdf(pages=2)))

        restarted = JobRunner(self.storage, workers=1)
        self.addCleanup(restarted.shutdown)
        reloaded = restarted.get(run["run_id"])
        self.assertEqual(reloaded.to_dict(), run)
        self.assertEqual(reloaded.events[-1]["type"], "pipeline_done")

    def test_run_interrupted_by_a_restart_is_reported_as_failed(self):
        run_id = "a" * 32
        record = {
            "run_id": run_id, "status": "running", "model": DEFAULT_MODEL, "mode": "live",
            "incomplete_stages": [], "stages": {name: {"status": "pending"} for name in STAGES},
        }
        saved = json.dumps({"record": record, "events": []}).encode("utf-8")
        self.storage.save(run_key(run_id), saved)

        response = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("restarted", response.json()["error"])
        events = self.stream(f"/api/runs/{run_id}/events")
        self.assertEqual((events[-1]["type"], events[-1]["status"]), ("pipeline_done", "failed"))

    def test_only_recent_finished_runs_are_kept_in_memory(self):
        self.patch("job_runner.MAX_RUNS_IN_MEMORY", 1)
        self.write_fixture("scan.txt", PAGE_1)
        first = self.finished_run(self.upload(make_image("png"), name="scan.png"))
        second = self.finished_run(self.upload(make_image("png"), name="scan.png"))
        self.finished_run(self.upload(make_image("png"), name="scan.png"))

        self.assertNotIn(first["run_id"], self.runner._runs)
        self.assertLessEqual(len(self.runner._runs), 2)
        # A run dropped from memory is still served from its saved record.
        self.assertEqual(self.client.get(f"/api/runs/{first['run_id']}").json(), first)
        self.assertEqual(self.client.get(f"/api/runs/{second['run_id']}").json()["status"], second["status"])


if __name__ == "__main__":
    unittest.main()
