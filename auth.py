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
- One-time codes (verify email, reset password) are 6 random digits, stored
  only as a hash, valid for 10 minutes and for 5 guesses; sending is limited
  to one per minute and five per hour per account. Endpoints that take only an
  email answer the same way whether or not an account exists.
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
import mailer

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


CODE_TTL_MINUTES = 10
MAX_CODE_ATTEMPTS = 5
RESEND_COOLDOWN_SECONDS = 60
MAX_CODES_PER_HOUR = 5
PURPOSE_TEXT = {"verify": "verify your email address", "reset": "reset your password"}


class AuthError(Exception):
    """A user-facing authentication problem (safe to show as-is)."""

    def __init__(self, message: str, status: int = 400, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code      # machine-readable reason for the frontend, e.g. "email_unverified"


@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: str
    email_verified: bool = True

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name,
                "email_verified": self.email_verified}


_USER_COLUMNS = "id, email, name, email_verified"


def _user(row) -> User:
    return User(id=row["id"], email=row["email"], name=row["name"],
                email_verified=bool(row["email_verified"]))


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
    validate_password(password, email)
    if len(name) > 100:
        raise AuthError("That name is too long.")


def validate_password(password: str, email: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError("That password is too long.")
    if password.strip().lower() == normalize_email(email):
        raise AuthError("Your password must not be your email address.")


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
                "INSERT INTO users (email, name, password_hash, email_verified) VALUES (?, ?, ?, 0)",
                (email, name.strip(), password_hash),
            )
            user_id = cursor.lastrowid
    except db.sqlite3.IntegrityError as exc:
        raise AuthError("An account with this email already exists.", status=409) from exc
    return User(id=user_id, email=email, name=name.strip(), email_verified=False)


def get_user(user_id: int) -> User | None:
    with db.connect() as conn:
        row = conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user(row) if row else None


def get_user_by_email(email: str) -> User | None:
    with db.connect() as conn:
        row = conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE email = ?",
                           (normalize_email(email),)).fetchone()
    return _user(row) if row else None


def authenticate(email: str, password: str, client: str = "") -> User:
    """Return the user for valid credentials, else raise one generic AuthError."""
    email = normalize_email(email)
    keys = [f"email:{email}", f"client:{client}"] if client else [f"email:{email}"]
    if any(_throttle.blocked(key) for key in keys):
        raise AuthError("Too many failed attempts. Wait a few minutes and try again.", status=429)
    with db.connect() as conn:
        row = conn.execute(f"SELECT {_USER_COLUMNS}, password_hash FROM users WHERE email = ?",
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
    user = _user(row)
    # Revealed only after the correct password, so it does not leak which
    # emails have accounts.
    if not user.email_verified:
        raise AuthError("Verify your email address first. We can send you a new code.",
                        status=403, code="email_unverified")
    return user


def _set_password(user_id: int, password: str) -> None:
    with db.connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hasher.hash(password), user_id))


def change_password(user: User, current_password: str, new_password: str,
                    keep_token: str | None = None) -> None:
    """Change a signed-in user's password; every other session is signed out."""
    key = f"change:{user.id}"
    if _throttle.blocked(key):
        raise AuthError("Too many failed attempts. Wait a few minutes and try again.", status=429)
    with db.connect() as conn:
        stored = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()[0]
    try:
        _hasher.verify(stored, current_password[:MAX_PASSWORD_LENGTH])
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        _throttle.fail(key)
        raise AuthError("Your current password is incorrect.", status=401) from None
    _throttle.reset(key)
    validate_password(new_password, user.email)
    if new_password == current_password:
        raise AuthError("Choose a password different from your current one.")
    _set_password(user.id, new_password)
    revoke_all_sessions(user.id, except_token=keep_token)


def confirm_password(user: User, password: str) -> None:
    """Re-check a signed-in user's password before a sensitive action (throttled)."""
    key = f"confirm:{user.id}"
    if _throttle.blocked(key):
        raise AuthError("Too many failed attempts. Wait a few minutes and try again.", status=429)
    with db.connect() as conn:
        stored = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()[0]
    try:
        _hasher.verify(stored, password[:MAX_PASSWORD_LENGTH])
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        _throttle.fail(key)
        raise AuthError("That password is incorrect.", status=401) from None
    _throttle.reset(key)


def delete_user_row(user_id: int) -> None:
    """Remove the account row; sessions, history, feedback, usage and codes cascade."""
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def update_name(user: User, name: str) -> User:
    name = (name or "").strip()
    if len(name) > 100:
        raise AuthError("That name is too long.")
    with db.connect() as conn:
        conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user.id))
    return get_user(user.id)


# ---------------------------------------------------------------------------
# One-time codes: email verification and password reset
# ---------------------------------------------------------------------------
def _code_hash(user_id: int, purpose: str, code: str) -> str:
    return hashlib.sha256(f"{user_id}:{purpose}:{code}".encode()).hexdigest()


def issue_code(user: User, purpose: str) -> None:
    """Create a fresh code (superseding older ones) and email it."""
    now = _now()
    with db.connect() as conn:
        conn.execute("DELETE FROM email_codes WHERE created_at < ?", (_iso(now - timedelta(days=1)),))
        last = conn.execute(
            "SELECT MAX(created_at) FROM email_codes WHERE user_id = ? AND purpose = ?",
            (user.id, purpose)).fetchone()[0]
        recent = conn.execute(
            "SELECT COUNT(*) FROM email_codes WHERE user_id = ? AND created_at > ?",
            (user.id, _iso(now - timedelta(hours=1)))).fetchone()[0]
    if last and last > _iso(now - timedelta(seconds=RESEND_COOLDOWN_SECONDS)):
        raise AuthError("A code was just sent. Wait a minute before asking for another.",
                        status=429, code="cooldown")
    if recent >= MAX_CODES_PER_HOUR:
        raise AuthError("Too many codes requested. Try again in an hour.", status=429, code="cooldown")
    code = f"{secrets.randbelow(10**6):06d}"
    with db.connect() as conn:
        conn.execute("UPDATE email_codes SET used = 1 WHERE user_id = ? AND purpose = ?", (user.id, purpose))
        cursor = conn.execute(
            "INSERT INTO email_codes (user_id, purpose, code_hash, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (user.id, purpose, _code_hash(user.id, purpose, code), _iso(now),
             _iso(now + timedelta(minutes=CODE_TTL_MINUTES))))
        code_id = cursor.lastrowid
    try:
        mailer.get_mailer().send(mailer.Message(
            to_email=user.email, to_name=user.name, code=code,
            purpose=PURPOSE_TEXT[purpose], expires_minutes=CODE_TTL_MINUTES))
    except mailer.MailError as exc:
        with db.connect() as conn:   # an unsent code must not count or linger
            conn.execute("DELETE FROM email_codes WHERE id = ?", (code_id,))
        raise AuthError(str(exc), status=503, code="mail_failed") from exc


def check_code(user: User, purpose: str, code: str) -> None:
    """Accept the current code once; count wrong guesses against it."""
    invalid = AuthError("That code is incorrect or has expired.", status=400, code="bad_code")
    code = (code or "").strip()
    with db.connect() as conn:
        row = conn.execute(
            """SELECT id, code_hash, attempts, expires_at FROM email_codes
               WHERE user_id = ? AND purpose = ? AND used = 0 ORDER BY id DESC LIMIT 1""",
            (user.id, purpose)).fetchone()
    if row is None or row["expires_at"] < _iso(_now()) or row["attempts"] >= MAX_CODE_ATTEMPTS:
        raise invalid
    if not (code.isdigit() and len(code) == 6
            and secrets.compare_digest(row["code_hash"], _code_hash(user.id, purpose, code))):
        with db.connect() as conn:
            conn.execute("UPDATE email_codes SET attempts = attempts + 1 WHERE id = ?", (row["id"],))
        raise invalid
    with db.connect() as conn:
        conn.execute("UPDATE email_codes SET used = 1 WHERE id = ?", (row["id"],))


def send_verification(email: str) -> None:
    """Email a verification code if an unverified account exists (silent otherwise)."""
    user = get_user_by_email(email)
    if user and not user.email_verified:
        try:
            issue_code(user, "verify")
        except AuthError as exc:
            if exc.code != "cooldown":    # a cooldown is not reported: it would confirm the account
                raise


def verify_email(email: str, code: str) -> User:
    user = get_user_by_email(email)
    if user is None:
        raise AuthError("That code is incorrect or has expired.", status=400, code="bad_code")
    if user.email_verified:
        return user
    check_code(user, "verify", code)
    with db.connect() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user.id,))
    return get_user(user.id)


def request_password_reset(email: str) -> None:
    """Email a reset code if the account exists; the reply is identical either way."""
    user = get_user_by_email(email)
    if user:
        try:
            issue_code(user, "reset")
        except AuthError as exc:
            if exc.code != "cooldown":
                raise


def reset_password(email: str, code: str, new_password: str) -> None:
    user = get_user_by_email(email)
    if user is None:
        raise AuthError("That code is incorrect or has expired.", status=400, code="bad_code")
    validate_password(new_password, user.email)
    check_code(user, "reset", code)
    _set_password(user.id, new_password)
    with db.connect() as conn:   # receiving the code proves the address belongs to them
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user.id,))
    revoke_all_sessions(user.id)
    _throttle.reset(f"email:{user.email}")


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
            """SELECT u.id, u.email, u.name, u.email_verified, s.csrf_token, s.expires_at
               FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ?""",
            (_hash_token(token),),
        ).fetchone()
    if not row or row["expires_at"] < _iso(_now()):
        return None
    return _user(row), row["csrf_token"]


def revoke_all_sessions(user_id: int, except_token: str | None = None) -> None:
    """Sign a user out everywhere (optionally keeping the session making the request)."""
    with db.connect() as conn:
        if except_token:
            conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
                         (user_id, _hash_token(except_token)))
        else:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


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
