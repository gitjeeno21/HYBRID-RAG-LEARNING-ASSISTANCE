"""The one Groq client used by the app and the evaluation scripts.

WHY a module of its own: the app used to import call_llm from evaluator.py,
an offline experiment script, so the web server depended on research code and
built a Groq client as an import side effect (importing the app failed without
an API key). The client is now created on first use.

Every call can report token usage: pass a dict as `usage` and it is filled with
prompt/completion/total token counts (left empty if the provider sends none).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from config import LLM_MODEL

load_dotenv()

# Free-tier Groq throttles aggressively: wait and retry rather than fail.
MAX_ATTEMPTS = 4
BASE_SLEEP_SECONDS = 5  # 5s, 10s, 15s between attempts


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Create the client on first use (reads GROQ_API_KEY from the environment)."""
    return Groq()


def generation_options(model: str = LLM_MODEL) -> dict:
    """Extra request options that keep reasoning models from truncating replies.

    Groq caps completions at 2048 tokens by default, and gpt-oss models spend
    hidden reasoning tokens from that same budget, so the visible reply can come
    back empty or cut off mid-JSON. A larger cap plus low reasoning effort
    leaves room for the answer.
    """
    options: dict = {"max_completion_tokens": 8192}
    if model.startswith("openai/gpt-oss"):
        options["reasoning_effort"] = "low"
    return options


# Collects usage from every call made inside `with track_usage():`, for code
# that cannot pass a dict down (e.g. helpers whose signature tests rely on).
_usage_sink: ContextVar[dict | None] = ContextVar("llm_usage_sink", default=None)


@contextmanager
def track_usage():
    """Yield a dict that accumulates token usage of all calls in this block."""
    totals: dict = {}
    token = _usage_sink.set(totals)
    try:
        yield totals
    finally:
        _usage_sink.reset(token)


def _record_usage(usage_out: dict | None, usage) -> None:
    if usage is None:
        return
    for target in {id(t): t for t in (usage_out, _usage_sink.get()) if t is not None}.values():
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, field, None)
            if isinstance(value, int):
                target[field] = target.get(field, 0) + value


def _wait(label: str, exc: Exception, attempt: int) -> None:
    wait = BASE_SLEEP_SECONDS * attempt
    print(f"    [{type(exc).__name__}] {label}: retrying in {wait}s "
          f"(attempt {attempt}/{MAX_ATTEMPTS})", file=sys.stderr, flush=True)
    time.sleep(wait)


def call_llm(prompt: str, temperature: float = 0.0, label: str = "llm",
             usage: dict | None = None, model: str = LLM_MODEL) -> str:
    """Send one prompt and return the reply text, retrying transient errors.

    An authentication error is raised at once: a bad key never fixes itself.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **generation_options(model),
            )
            _record_usage(usage, getattr(response, "usage", None))
            return response.choices[0].message.content or ""
        except AuthenticationError:
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            _wait(label, exc, attempt)
    raise RuntimeError("unreachable")


def call_llm_stream(prompt: str, temperature: float = 0.0, label: str = "llm",
                    usage: dict | None = None, model: str = LLM_MODEL) -> Iterator[str]:
    """Yield the reply as text fragments.

    Retries apply only while opening the stream: once text has reached the
    browser, a retry would duplicate it, so a mid-stream failure is raised.
    Groq reports usage on the final chunk (`x_groq.usage`); it is recorded when
    the stream ends.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            stream = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                **generation_options(model),
            )
            break
        except AuthenticationError:
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            _wait(label, exc, attempt)
    for chunk in stream:
        extra = getattr(chunk, "x_groq", None)
        _record_usage(usage, getattr(extra, "usage", None) or getattr(chunk, "usage", None))
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
