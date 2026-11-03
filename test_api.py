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
