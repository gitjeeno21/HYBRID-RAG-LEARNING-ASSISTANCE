"""Test helper: a TestClient signed in as a fresh, email-verified user on a throwaway database.

Importing this module installs a CaptureMailer, so tests never send real email
and can read the codes an account was sent (MAIL.last_code(email)).
"""

import tempfile
from itertools import count
from pathlib import Path

from fastapi.testclient import TestClient

import auth
import db
import mailer

_ids = count(1)
_keep_alive: list[tempfile.TemporaryDirectory] = []
PASSWORD = "correct horse battery staple"
MAIL = mailer.CaptureMailer()
mailer.set_mailer(MAIL)


def fresh_database() -> Path:
    """Point the app at a new empty SQLite file and return its path."""
    folder = tempfile.TemporaryDirectory()
    _keep_alive.append(folder)
    path = Path(folder.name) / "test.db"
    db.use_database(path)
    auth._throttle.clear()
    MAIL.sent.clear()
    return path


def signed_in(client: TestClient, email: str, password: str = PASSWORD) -> TestClient:
    """Register `email`, verify it with the emailed code, and attach the CSRF header."""
    response = client.post("/api/auth/register", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    response = client.post("/api/auth/verify-email", json={"email": email, "code": MAIL.last_code(email)})
    assert response.status_code == 200, response.text
    client.headers[auth.CSRF_HEADER] = client.cookies.get(auth.CSRF_COOKIE)
    return client


def authed_client(app, email: str | None = None, new_database: bool = True) -> TestClient:
    if new_database:
        fresh_database()
    return signed_in(TestClient(app), email or f"user{next(_ids)}@example.com")
