"""Daily LLM token allowance per user."""

import unittest
from unittest.mock import patch

import main
import quota
from auth_testing import authed_client

RETRIEVED = [{"filename": "a.pdf", "citation": "a.pdf, p. 1", "text": "t", "score": 0.5}]


def fake_llm(prompt, temperature=0.0, label="", usage=None, **_):
    if usage is not None:
        usage["total_tokens"] = usage.get("total_tokens", 0) + 1200
    return "An answer (a.pdf, p. 1)"


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.client = authed_client(main.app)
        self.user_id = self.client.get("/api/auth/me").json()["user"]["id"]
        patcher = patch.object(quota, "DAILY_TOKEN_LIMIT", 2000)
        patcher.start()
        self.addCleanup(patcher.stop)

    def chat(self):
        with patch("main.list_pdf_names", return_value=["a.pdf"]), \
                patch("main.rewrite_followup", side_effect=lambda q, h: q), \
                patch("main.retrieve_for_api", return_value=RETRIEVED), \
                patch("main.assess", return_value={"label": "High", "reason": "ok"}), \
                patch("main.call_llm", side_effect=fake_llm) as llm:
            return self.client.post("/api/chat", json={"query": "q"}), llm

    def test_chat_records_the_providers_token_count(self):
        response, _ = self.chat()
        self.assertEqual(response.status_code, 200)
        usage = self.client.get("/api/usage").json()
        self.assertEqual(usage["tokens_today"], 1200)
        self.assertEqual(usage["requests_today"], 1)
        self.assertEqual(usage["daily_limit"], 2000)

    def test_requests_are_refused_once_the_allowance_is_used_without_calling_the_llm(self):
        quota.record(self.user_id, 2000)
        response, llm = self.chat()
        self.assertEqual(response.status_code, 429)
        self.assertIn("allowance", response.json()["detail"])
        llm.assert_not_called()
        stream = self.client.post("/api/chat/stream", json={"query": "q"})
        self.assertEqual(stream.status_code, 429)   # refused before the stream opens

    def test_allowances_are_per_user_and_reset_each_day(self):
        quota.record(self.user_id, 5000)
        with self.assertRaises(quota.QuotaExceeded):
            quota.check(self.user_id)
        other = authed_client(main.app, new_database=False)
        other_id = other.get("/api/auth/me").json()["user"]["id"]
        quota.check(other_id)   # someone else's usage does not count against you
        with patch.object(quota, "_today", return_value="2999-01-01"):
            quota.check(self.user_id)

    def test_retrieval_only_inspection_still_works_over_the_allowance(self):
        quota.record(self.user_id, 5000)
        with patch("main.list_pdf_names", return_value=[]), \
                patch("main.retrieve_for_api", return_value=[]):
            response = self.client.post("/api/retrieval/inspect", json={"query": "q"})
        self.assertEqual(response.status_code, 200)

    def test_zero_means_unlimited(self):
        with patch.object(quota, "DAILY_TOKEN_LIMIT", 0):
            quota.record(self.user_id, 10**9)
            quota.check(self.user_id)
            self.assertIsNone(quota.usage(self.user_id)["daily_limit"])


if __name__ == "__main__":
    unittest.main()
