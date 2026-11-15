"""Multi-paper chat: every selected paper is addressed, from its own evidence only."""

import unittest
from unittest.mock import patch

import main
from auth_testing import authed_client
from config import PROMPT_TEMPLATE

A = {"filename": "a.pdf", "citation": "a.pdf, p. 2", "text": "Paper A uses BLEU.", "score": 0.5}
B = {"filename": "b.pdf", "citation": "b.pdf, p. 7", "text": "Paper B uses exam scores.", "score": 0.4}


class MultiPaperPromptTests(unittest.TestCase):
    def test_zero_or_one_paper_keeps_the_evaluated_prompt_exactly(self):
        flat = PROMPT_TEMPLATE.format(context="Source: a.pdf, p. 2\nPaper A uses BLEU.", question="q")
        self.assertEqual(main._chat_prompt("q", [A]), flat)
        self.assertEqual(main._chat_prompt("q", [A], ["a.pdf"]), flat)

    def test_two_papers_get_their_own_sections_in_order(self):
        prompt = main._chat_prompt("How is evaluation done?", [B, A], ["a.pdf", "b.pdf"])
        a_at, b_at = prompt.index("=== Paper: a.pdf ==="), prompt.index("=== Paper: b.pdf ===")
        self.assertLess(a_at, b_at)
        self.assertIn("Paper A uses BLEU.", prompt[a_at:b_at])        # A's evidence under A only
        self.assertNotIn("Paper B uses exam scores.", prompt[a_at:b_at])
        self.assertIn("selected 2 papers", prompt)
        self.assertIn("say so in its section", prompt)

    def test_a_paper_with_no_passages_is_still_listed(self):
        prompt = main._chat_prompt("q", [A], ["a.pdf", "b.pdf"])
        self.assertIn("=== Paper: b.pdf ===\n[No passages were retrieved from this paper.]", prompt)

    def test_chat_sends_the_grouped_prompt_when_two_papers_are_selected(self):
        client = authed_client(main.app)
        prompts = []

        def fake_llm(prompt, **_):
            prompts.append(prompt)
            return "## a.pdf\nBLEU (a.pdf, p. 2)\n## b.pdf\nExam scores (b.pdf, p. 7)"

        with patch("main.list_pdf_names", return_value=["a.pdf", "b.pdf"]), \
                patch("main.rewrite_followup", side_effect=lambda q, h: q), \
                patch("main.retrieve_for_api", return_value=[A, B]), \
                patch("main.assess", return_value={"label": "High", "reason": "ok"}), \
                patch("main.call_llm", side_effect=fake_llm):
            response = client.post("/api/chat", json={"query": "How is evaluation done?",
                                                       "papers": ["a.pdf", "b.pdf"]})
        self.assertEqual(response.status_code, 200)
        self.assertIn("=== Paper: b.pdf ===", prompts[0])
        self.assertEqual(response.json()["trace"]["papers"], ["a.pdf", "b.pdf"])


if __name__ == "__main__":
    unittest.main()
