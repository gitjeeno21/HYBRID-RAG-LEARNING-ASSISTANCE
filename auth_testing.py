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


def make_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF whose text pypdf can extract."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)
