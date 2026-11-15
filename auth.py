"""Accounts, password hashing, sessions and login throttling.

Security decisions, each deliberate:
- Passwords are hashed with Argon2id (argon2-cffi defaults, the OWASP
  recommendation); the plain password is never stored or logged. A hash made
  with older parameters is upgraded on the next successful login.
- A session is a random 256-bit token in an HttpOnly cookie. The database
  stores only its SHA-256, so a leaked database cannot be replayed as cookies.
- Every session has its own CSRF token (double-submit: a readable cookie the
  page echoes in the X-CSRF-Token header on every write request).
- Login failures are throttled per email and per client address, and the
  error message never reveals whether the email exists. A failed lookup still
  runs one Argon2 verification so response time does not reveal it either.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

import db

SESSION_COOKIE = "rag_session"
CSRF_COOKIE = "rag_csrf"
CSRF_HEADER = "X-CSRF-Token"
SESSION_DAYS = 7
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256          # bounds hashing work for absurd inputs
MAX_FAILURES = 5                   # failed logins allowed ...
FAILURE_WINDOW_SECONDS = 15 * 60   # ... per key in this window

_hasher = PasswordHasher()
# Verified against when the email is unknown, so both paths cost the same.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """A user-facing authentication problem (safe to show as-is)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: str

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name}


@dataclass(frozen=True)
class Session:
    token: str        # raw value for the cookie; only returned at creation
    csrf_token: str
    expires_at: datetime


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_registration(email: str, password: str, name: str) -> None:
    if not _EMAIL_RE.fullmatch(normalize_email(email)) or len(email) > 254:
        raise AuthError("Enter a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError("That password is too long.")
    if password.strip().lower() == normalize_email(email):
        raise AuthError("Your password must not be your email address.")
    if len(name) > 100:
        raise AuthError("That name is too long.")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def create_user(email: str, password: str, name: str = "") -> User:
    validate_registration(email, password, name)
    email = normalize_email(email)
    password_hash = _hasher.hash(password)
    try:
        with db.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                (email, name.strip(), password_hash),
            )
            user_id = cursor.lastrowid
    except db.sqlite3.IntegrityError as exc:
        raise AuthError("An account with this email already exists.", status=409) from exc
    return User(id=user_id, email=email, name=name.strip())


def get_user(user_id: int) -> User | None:
    with db.connect() as conn:
        row = conn.execute("SELECT id, email, name FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(**dict(row)) if row else None


def get_user_by_email(email: str) -> User | None:
    with db.connect() as conn:
        row = conn.execute("SELECT id, email, name FROM users WHERE email = ?",
                           (normalize_email(email),)).fetchone()
    return User(**dict(row)) if row else None


def authenticate(email: str, password: str, client: str = "") -> User:
    """Return the user for valid credentials, else raise one generic AuthError."""
    email = normalize_email(email)
    keys = [f"email:{email}", f"client:{client}"] if client else [f"email:{email}"]
    if any(_throttle.blocked(key) for key in keys):
        raise AuthError("Too many failed attempts. Wait a few minutes and try again.", status=429)
    with db.connect() as conn:
        row = conn.execute("SELECT id, email, name, password_hash FROM users WHERE email = ?",
                           (email,)).fetchone()
    stored = row["password_hash"] if row else _DUMMY_HASH
    try:
        _hasher.verify(stored, password[:MAX_PASSWORD_LENGTH])
        ok = row is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        ok = False
    if not ok:
        for key in keys:
            _throttle.fail(key)
        raise AuthError("Incorrect email or password.", status=401)
    for key in keys:
        _throttle.reset(key)
    if _hasher.check_needs_rehash(stored):
        with db.connect() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (_hasher.hash(password), row["id"]))
    return User(id=row["id"], email=row["email"], name=row["name"])


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def create_session(user: User) -> Session:
    token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(days=SESSION_DAYS)
    with db.connect() as conn:
        # Housekeeping: expired sessions are useless; drop them as we go.
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(_now()),))
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, csrf_token, expires_at) VALUES (?, ?, ?, ?)",
            (_hash_token(token), user.id, csrf_token, _iso(expires)),
        )
    return Session(token=token, csrf_token=csrf_token, expires_at=expires)


def resolve_session(token: str | None) -> tuple[User, str] | None:
    """Return (user, csrf_token) for a live session cookie, else None."""
    if not token or len(token) > 200:
        return None
    with db.connect() as conn:
        row = conn.execute(
            """SELECT u.id, u.email, u.name, s.csrf_token, s.expires_at
               FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ?""",
            (_hash_token(token),),
        ).fetchone()
    if not row or row["expires_at"] < _iso(_now()):
        return None
    return User(id=row["id"], email=row["email"], name=row["name"]), row["csrf_token"]


def revoke_session(token: str | None) -> None:
    if token:
        with db.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


def csrf_matches(expected: str, supplied: str | None) -> bool:
    return bool(supplied) and secrets.compare_digest(expected, supplied)


# ---------------------------------------------------------------------------
# Login throttling (in memory: this app runs as one local process)
# ---------------------------------------------------------------------------
class _Throttle:
    def __init__(self) -> None:
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> list[float]:
        cutoff = time.monotonic() - FAILURE_WINDOW_SECONDS
        recent = [t for t in self._failures.get(key, []) if t > cutoff]
        self._failures[key] = recent
        return recent

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= MAX_FAILURES

    def fail(self, key: str) -> None:
        with self._lock:
            self._recent(key).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._failures.clear()


_throttle = _Throttle()
