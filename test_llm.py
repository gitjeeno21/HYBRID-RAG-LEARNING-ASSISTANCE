"""Offline tests for llm.py and the answer prompt contract (no network)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import AuthenticationError, RateLimitError

import config
import llm
from trust import is_refusal


def reply(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens),
    )


def stream_chunk(text: str | None, usage=None):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if text is not None else [],
                           x_groq=SimpleNamespace(usage=usage) if usage else None)


def provider_error(cls, status: int):
    request = httpx.Request("POST", "https://api.groq.com")
    return cls("error", response=httpx.Response(status, request=request), body=None)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class LlmTests(unittest.TestCase):
    def test_call_records_token_usage(self):
        usage = {}
        with patch("llm.get_client", return_value=FakeClient([reply("hi", 12, 3)])):
            self.assertEqual(llm.call_llm("p", usage=usage), "hi")
        self.assertEqual(usage, {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})

    def test_rate_limit_is_retried_then_succeeds(self):
        client = FakeClient([provider_error(RateLimitError, 429), reply("ok")])
        with patch("llm.get_client", return_value=client), patch("llm.time.sleep") as sleep:
            self.assertEqual(llm.call_llm("p"), "ok")
        self.assertEqual(client.calls, 2)
        sleep.assert_called_once()

    def test_bad_key_is_not_retried(self):
        client = FakeClient([provider_error(AuthenticationError, 401)])
        with patch("llm.get_client", return_value=client), self.assertRaises(AuthenticationError):
            llm.call_llm("p")
        self.assertEqual(client.calls, 1)

    def test_stream_yields_text_and_records_final_usage(self):
        final = SimpleNamespace(prompt_tokens=20, completion_tokens=4, total_tokens=24)
        chunks = iter([stream_chunk("Hel"), stream_chunk("lo"), stream_chunk(None, final)])
        usage = {}
        with patch("llm.get_client", return_value=FakeClient([chunks])):
            self.assertEqual("".join(llm.call_llm_stream("p", usage=usage)), "Hello")
        self.assertEqual(usage["total_tokens"], 24)

    def test_reasoning_models_get_a_larger_budget(self):
        self.assertEqual(llm.generation_options("openai/gpt-oss-120b")["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", llm.generation_options("llama-3.1-8b-instant"))


class PromptContractTests(unittest.TestCase):
    def test_full_refusal_sentence_is_still_required_and_detected(self):
        prompt = config.PROMPT_TEMPLATE.format(context="c", question="q")
        self.assertIn(f'"{config.NO_ANSWER_TEXT}"', prompt)
        self.assertTrue(is_refusal(config.NO_ANSWER_TEXT))

    def test_partial_answers_are_permitted_without_outside_knowledge(self):
        prompt = config.PROMPT_TEMPLATE.format(context="c", question="q")
        self.assertIn("answers only part of the question", prompt)
        self.assertIn("Never fill a gap with outside knowledge", prompt)
        # A partial answer that names the gap is not a refusal.
        self.assertFalse(is_refusal("GPT-4 scored 163 (source.pdf, p. 5). The documents do not "
                                    "cover GPT-3.5's percentile."))


if __name__ == "__main__":
    unittest.main()
