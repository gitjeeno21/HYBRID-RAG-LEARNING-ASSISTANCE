"""Test helper: a TestClient signed in as a fresh user on a throwaway database."""

import tempfile
from itertools import count
from pathlib import Path

from fastapi.testclient import TestClient

import auth
import db

_ids = count(1)
_keep_alive: list[tempfile.TemporaryDirectory] = []
PASSWORD = "correct horse battery staple"


def fresh_database() -> Path:
    """Point the app at a new empty SQLite file and return its path."""
    folder = tempfile.TemporaryDirectory()
    _keep_alive.append(folder)
    path = Path(folder.name) / "test.db"
    db.use_database(path)
    auth._throttle.clear()
    return path


def signed_in(client: TestClient, email: str) -> TestClient:
    """Register `email` on this client's app and attach its CSRF header."""
    response = client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers[auth.CSRF_HEADER] = client.cookies.get(auth.CSRF_COOKIE)
    return client


def authed_client(app, email: str | None = None, new_database: bool = True) -> TestClient:
    if new_database:
        fresh_database()
    return signed_in(TestClient(app), email or f"user{next(_ids)}@example.com")
