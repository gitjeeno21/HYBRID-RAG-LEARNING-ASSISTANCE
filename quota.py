"""A daily LLM token allowance per user.

WHY: every account's questions are answered with the operator's single Groq
key, whose free tier has a daily token limit shared by everyone. Without a
per-user allowance, one heavy user (or a runaway script) exhausts the key for
all. Usage is counted from the provider's own token counts (llm.py), checked
before any LLM-backed request starts, and reset at midnight UTC.

APP_DAILY_TOKEN_LIMIT sets the allowance (default 100,000; 0 = unlimited).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import db

DAILY_TOKEN_LIMIT = int(os.getenv("APP_DAILY_TOKEN_LIMIT", "100000"))


class QuotaExceeded(Exception):
    pass


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def usage(user_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT tokens, requests FROM llm_usage WHERE user_id = ? AND day = ?",
                           (user_id, _today())).fetchone()
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    return {
        "tokens_today": row["tokens"] if row else 0,
        "requests_today": row["requests"] if row else 0,
        "daily_limit": DAILY_TOKEN_LIMIT or None,
        "resets_at": tomorrow,
    }


def check(user_id: int) -> None:
    """Raise QuotaExceeded when the user has used up today's allowance."""
    if DAILY_TOKEN_LIMIT and usage(user_id)["tokens_today"] >= DAILY_TOKEN_LIMIT:
        raise QuotaExceeded(
            f"You have used today's AI allowance ({DAILY_TOKEN_LIMIT:,} tokens). "
            "It resets at midnight UTC. Retrieval (Inspect) still works."
        )


def record(user_id: int, tokens: int) -> None:
    """Add one LLM-backed request's tokens to today's total."""
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO llm_usage (user_id, day, tokens, requests) VALUES (?, ?, ?, 1)
               ON CONFLICT (user_id, day) DO UPDATE SET tokens = tokens + excluded.tokens,
                                                       requests = requests + 1""",
            (user_id, _today(), max(0, int(tokens or 0))),
        )
