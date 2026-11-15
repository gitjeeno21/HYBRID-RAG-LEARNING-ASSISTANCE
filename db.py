"""SQLite storage for accounts, sessions and per-user records.

WHY SQLite and the standard library (no ORM): the app runs locally for one
machine, and sqlite3 ships with Python, so accounts add no server and no
dependency. Every query below is parameterised (never string-formatted), and
the schema is created by numbered migrations so a future change is one more
entry in MIGRATIONS, applied once and in order.

The database path comes from APP_DB_PATH (default: app.db next to this file).
Tests point it at a temporary file with use_database().
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from config import BASE_DIR

_db_path = Path(os.getenv("APP_DB_PATH", str(BASE_DIR / "app.db")))
_migrate_lock = threading.Lock()
_migrated: set[str] = set()

# Each entry runs once, in order; schema_version records how far a database has got.
MIGRATIONS: list[str] = [
    # 1: accounts and sessions
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
        name          TEXT NOT NULL DEFAULT '',
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE TABLE sessions (
        token_hash  TEXT PRIMARY KEY,           -- sha256 of the cookie value; the raw token is never stored
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        csrf_token  TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        expires_at  TEXT NOT NULL
    );
    CREATE INDEX sessions_user ON sessions(user_id);
    """,
    # 2: per-user records that used to live in shared JSON files
    """
    CREATE TABLE quiz_history (
        id          INTEGER PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        pdf         TEXT NOT NULL,
        difficulty  TEXT NOT NULL,
        score       REAL NOT NULL,
        topics      TEXT NOT NULL DEFAULT '[]',  -- JSON list
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE INDEX quiz_history_user ON quiz_history(user_id);
    CREATE TABLE chat_feedback (
        id          INTEGER PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        query       TEXT NOT NULL DEFAULT '',
        message     TEXT NOT NULL,
        feedback    TEXT NOT NULL CHECK (feedback IN ('up', 'down')),
        created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    """,
]


def use_database(path: Path | str) -> None:
    """Point every later connection at `path` (tests use a temporary file)."""
    global _db_path
    _db_path = Path(path)


def database_path() -> Path:
    return _db_path


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] or 0
    for version, script in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        # executescript commits first, so each migration and its version row
        # are applied together inside one explicit transaction.
        conn.executescript(f"BEGIN; {script}; INSERT INTO schema_version VALUES ({version}); COMMIT;")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A connection with foreign keys on, rows as dicts, committed on success.

    One short-lived connection per call: FastAPI runs sync endpoints in a
    thread pool and sqlite3 connections must not be shared across threads.
    """
    path = _db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        key = str(path.resolve())
        if key not in _migrated:
            with _migrate_lock:
                if key not in _migrated:
                    conn.execute("PRAGMA journal_mode = WAL")  # readers never block the writer
                    _migrate(conn)
                    _migrated.add(key)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
