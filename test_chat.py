"""Offline tests for enhanced chat follow-up handling and trust-friendly errors."""

import unittest
from unittest.mock import patch

from main import rewrite_followup


class ChatEnhancementsTests(unittest.TestCase):
    def test_standalone_query_rewrite_uses_mocked_llm(self) -> None:
        history = [
            {"role": "user", "content": "What architecture does GPT-4 use?"},
            {"role": "assistant", "content": "The report describes a transformer architecture."},
        ]
        prompts = []

        def mock_call(prompt, temperature=0.0, label="llm"):
            prompts.append(prompt)
            return "What architecture and training method does GPT-4 use?"

        with patch("main.call_llm", side_effect=mock_call):
            result = rewrite_followup("How does it differ from the baseline?", history)

        self.assertEqual(result, "What architecture and training method does GPT-4 use?")
        self.assertEqual(len(prompts), 1)
        self.assertIn("GPT-4", prompts[0])

    def test_non_contextual_followup_is_not_rewritten(self) -> None:
        history = [{"role": "user", "content": "What architecture is used?"}]

        with patch("main.call_llm") as mocked_call:
            result = rewrite_followup("List the reported limitations", history)

        self.assertEqual(result, "List the reported limitations")
        mocked_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
