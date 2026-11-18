"""Sending account emails (verification and password-reset codes).

Codes are generated and checked on the server; this module only delivers
them. It sends through EmailJS's REST API from the server, never from the
browser: a page that sent the email itself would know the code, and anyone
could then skip verification.

Configuration (.env, never committed):
    EMAILJS_SERVICE_ID     the email service connected in the EmailJS dashboard
    EMAILJS_TEMPLATE_ID    a template that uses the parameters listed below
    EMAILJS_PUBLIC_KEY     Account > API keys > Public key
    EMAILJS_PRIVATE_KEY    Account > API keys > Private key (server-side auth)
    APP_EMAIL_MODE         emailjs | console   (default: emailjs when the keys
                           above are set, otherwise console)

Template parameters: to_email, to_name, otp_code, purpose, expires_minutes,
app_name. In the template, set "To Email" to {{to_email}}.

console mode prints codes to the server's terminal instead of emailing them.
It is for local development only; never use it on a server others can reach.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

EMAILJS_URL = "https://api.emailjs.com/api/v1.0/email/send"
APP_NAME = "Hybrid RAG"
log = logging.getLogger("rag.mail")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


class MailError(Exception):
    """The email could not be sent (shown to the user in general terms)."""


@dataclass
class Message:
    to_email: str
    to_name: str
    code: str
    purpose: str            # human wording, e.g. "verify your email address"
    expires_minutes: int


class EmailJSMailer:
    def __init__(self, service_id: str, template_id: str, public_key: str, private_key: str):
        self.service_id, self.template_id = service_id, template_id
        self.public_key, self.private_key = public_key, private_key

    def send(self, message: Message) -> None:
        payload = {
            "service_id": self.service_id,
            "template_id": self.template_id,
            "user_id": self.public_key,
            "accessToken": self.private_key,
            "template_params": {
                "to_email": message.to_email,
                "to_name": message.to_name or message.to_email,
                "otp_code": message.code,
                "purpose": message.purpose,
                "expires_minutes": message.expires_minutes,
                "app_name": APP_NAME,
            },
        }
        try:
            response = httpx.post(EMAILJS_URL, json=payload, timeout=15)
        except httpx.HTTPError as exc:
            log.error("EmailJS unreachable: %s", type(exc).__name__)
            raise MailError("The email service could not be reached. Try again shortly.") from exc
        if response.status_code != 200:
            # The body explains configuration problems (wrong IDs, API access
            # for non-browser apps disabled); it holds no secret, so log it.
            log.error("EmailJS rejected the request: %s %s", response.status_code, response.text[:300])
            raise MailError("The email could not be sent. Please try again later.")


class ConsoleMailer:
    """Development only: prints the code to the server terminal."""

    def send(self, message: Message) -> None:
        log.warning("EMAIL NOT CONFIGURED (console mode). Code for %s to %s: %s",
                    message.to_email, message.purpose, message.code)


@dataclass
class CaptureMailer:
    """Test double: keeps every message."""

    sent: list[Message] = field(default_factory=list)

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def last_code(self, email: str) -> str:
        return next(m.code for m in reversed(self.sent) if m.to_email == email)


def _from_environment():
    keys = [os.getenv(name, "").strip() for name in
            ("EMAILJS_SERVICE_ID", "EMAILJS_TEMPLATE_ID", "EMAILJS_PUBLIC_KEY", "EMAILJS_PRIVATE_KEY")]
    mode = os.getenv("APP_EMAIL_MODE", "").strip().lower() or ("emailjs" if all(keys) else "console")
    if mode == "emailjs":
        if not all(keys):
            raise RuntimeError("APP_EMAIL_MODE=emailjs needs EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, "
                               "EMAILJS_PUBLIC_KEY and EMAILJS_PRIVATE_KEY in .env")
        return EmailJSMailer(*keys)
    if mode == "console":
        log.warning("Email is in console mode: verification codes are printed here, not emailed. "
                    "Set the EMAILJS_* keys in .env to send real email.")
        return ConsoleMailer()
    raise RuntimeError(f"Unknown APP_EMAIL_MODE {mode!r} (use emailjs or console)")


_mailer = None


def get_mailer():
    global _mailer
    if _mailer is None:
        _mailer = _from_environment()
    return _mailer


def set_mailer(mailer) -> None:
    """Replace the mailer (tests install a CaptureMailer)."""
    global _mailer
    _mailer = mailer


def mode() -> str:
    mailer = get_mailer()
    return {EmailJSMailer: "emailjs", ConsoleMailer: "console"}.get(type(mailer), "test")
