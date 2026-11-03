"""Offline API tests: feedback, provider-error mapping, progress, and health."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from groq import RateLimitError

import main


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        for name, value in (
            ("FEEDBACK_PATH", tmp / "feedback.json"),
            ("QUIZ_HISTORY_PATH", tmp / "history.json"),
        ):
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_feedback_accepts_json_body(self) -> None:
        response = self.client.post(
            "/api/chat/feedback", json={"message": "An answer", "feedback": "up", "query": "q"}
        )
        self.assertEqual(response.status_code, 200)
        saved = json.loads(main.FEEDBACK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["feedback"], "up")

    def test_feedback_rejects_unknown_value(self) -> None:
        response = self.client.post("/api/chat/feedback", json={"message": "x", "feedback": "meh"})
        self.assertEqual(response.status_code, 422)

    def test_rate_limit_becomes_readable_429(self) -> None:
        error = RateLimitError(
            "limited",
            response=__import__("httpx").Response(429, request=__import__("httpx").Request("POST", "http://x")),
            body=None,
        )
        with (
            patch("main.list_pdf_names", return_value=[]),
            patch("main.rewrite_followup", side_effect=lambda q, h: q),
            patch("main.retrieve_for_api", return_value=[{"filename": "a.pdf", "citation": "a.pdf, p. 1", "text": "t", "score": 0.5}]),
            patch("main.call_llm", side_effect=error),
        ):
            response = self.client.post("/api/chat", json={"query": "hello"})
        self.assertEqual(response.status_code, 429)
        self.assertIn("rate limited", response.json()["detail"])

    def _stream(self, **patches):
        retrieved = [{"filename": "a.pdf", "citation": "a.pdf, p. 1", "text": "t", "score": 0.5}]
        with (
            patch("main.list_pdf_names", return_value=["a.pdf"]),
            patch("main.rewrite_followup", side_effect=lambda q, h: q),
            patch("main.retrieve_for_api", return_value=patches.pop("retrieved", retrieved)),
            patch("main.assess", return_value={"label": "High", "reason": "ok"}),
            patch("main.call_llm_stream", **patches),
        ):
            response = self.client.post("/api/chat/stream", json={"query": "hello"})
        return response, [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def test_stream_sends_meta_tokens_then_done(self) -> None:
        response, events = self._stream(return_value=iter(["Hel", "lo (a.pdf, p. 1)"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual([event["type"] for event in events], ["meta", "token", "token", "done"])
        self.assertEqual(events[0]["context"][0]["citation"], "a.pdf, p. 1")
        self.assertEqual(events[-1]["answer"], "Hello (a.pdf, p. 1)")
        self.assertEqual(events[-1]["trust"]["label"], "High")

    def test_stream_reports_provider_error_in_band(self) -> None:
        def failing(*_, **__):
            yield "partial"
            raise ConnectionError("boom")
        _, events = self._stream(side_effect=failing)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("could not be generated", events[-1]["detail"])

    def test_stream_without_passages_skips_the_llm(self) -> None:
        _, events = self._stream(retrieved=[], side_effect=AssertionError("must not call"))
        self.assertEqual([event["type"] for event in events], ["meta", "done"])
        self.assertIn("Not found", events[-1]["answer"])

    def test_pdf_parse_cache_reuses_unchanged_files(self) -> None:
        from langchain_core.documents import Document
        main._parse_pdf.cache_clear()
        pdf = Path(self.tmp.name) / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        with patch("main.PyPDFLoader") as loader:
            loader.return_value.load.return_value = [Document(page_content="x", metadata={"page": 0})]
            first = main.load_pdf_pages(pdf)
            first[0].metadata["mutated"] = True  # callers must not corrupt the cache
            second = main.load_pdf_pages(pdf)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(second[0].metadata["filename"], "paper.pdf")
        self.assertNotIn("mutated", second[0].metadata)
        main._parse_pdf.cache_clear()

    def test_chat_top_k_is_validated(self) -> None:
        response = self.client.post("/api/chat", json={"query": "hello", "top_k": 99})
        self.assertEqual(response.status_code, 422)

    def test_progress_reports_recent_attempts_and_weak_topics(self) -> None:
        for score, topic in ((40, "attention"), (90, "scaling"), (50, "attention")):
            self.client.post(
                "/api/progress/quiz",
                json={"pdf": "a.pdf", "difficulty": "Medium", "score": score, "topics": [topic]},
            )
        progress = self.client.get("/api/progress").json()
        self.assertEqual(progress["quizzes_taken"], 3)
        self.assertEqual(progress["most_missed_topics"], ["attention"])
        self.assertEqual(progress["recent"][0]["score"], 50)
        self.assertEqual(progress["overall_average"], 60.0)

    def test_health_reports_missing_index(self) -> None:
        with patch("main.get_hybrid_resources", side_effect=RuntimeError("index mismatch")):
            data = self.client.get("/api/health").json()
        self.assertFalse(data["index_ready"])
        self.assertIn("mismatch", data["index_error"])


if __name__ == "__main__":
    unittest.main()
