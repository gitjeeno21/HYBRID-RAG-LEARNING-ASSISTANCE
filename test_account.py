"""Email verification codes, password reset, password change and account settings."""

import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import auth
import db
import mailer
import main
from auth_testing import MAIL, PASSWORD, authed_client, fresh_database, signed_in

NEW_PASSWORD = "an entirely new passphrase"


def register(client, email):
    return client.post("/api/auth/register", json={"email": email, "password": PASSWORD})


def verify(client, email, code):
    return client.post("/api/auth/verify-email", json={"email": email, "code": code})


def age_codes(seconds):
    """Pretend every code was sent `seconds` earlier (to step past cooldowns)."""
    with db.connect() as conn:
        for row in conn.execute("SELECT id, created_at FROM email_codes").fetchall():
            from datetime import datetime, timedelta
            moved = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ") - timedelta(seconds=seconds)
            conn.execute("UPDATE email_codes SET created_at = ? WHERE id = ?",
                         (moved.strftime("%Y-%m-%dT%H:%M:%SZ"), row["id"]))


class VerificationCodeTests(unittest.TestCase):
    def setUp(self):
        fresh_database()
        self.client = TestClient(main.app)
        register(self.client, "v@example.com")

    def test_code_is_six_digits_and_stored_only_as_a_hash(self):
        code = MAIL.last_code("v@example.com")
        self.assertRegex(code, r"^\d{6}$")
        self.assertEqual(MAIL.sent[-1].purpose, "verify your email address")
        with db.connect() as conn:
            stored = conn.execute("SELECT code_hash FROM email_codes").fetchone()[0]
        self.assertNotIn(code, stored)

    def test_five_wrong_guesses_burn_the_code(self):
        code = MAIL.last_code("v@example.com")
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(auth.MAX_CODE_ATTEMPTS):
            self.assertEqual(verify(self.client, "v@example.com", wrong).status_code, 400)
        self.assertEqual(verify(self.client, "v@example.com", code).status_code, 400)

    def test_expired_codes_are_rejected(self):
        with db.connect() as conn:
            conn.execute("UPDATE email_codes SET expires_at = '2000-01-01T00:00:00Z'")
        self.assertEqual(verify(self.client, "v@example.com", MAIL.last_code("v@example.com")).status_code, 400)

    def test_a_new_code_supersedes_the_old_one(self):
        old = MAIL.last_code("v@example.com")
        age_codes(120)
        self.client.post("/api/auth/resend-verification", json={"email": "v@example.com"})
        new = MAIL.last_code("v@example.com")
        if old != new:
            self.assertEqual(verify(self.client, "v@example.com", old).status_code, 400)
        self.assertEqual(verify(self.client, "v@example.com", new).status_code, 200)

    def test_resend_is_rate_limited_without_revealing_the_account(self):
        sent = len(MAIL.sent)
        known = self.client.post("/api/auth/resend-verification", json={"email": "v@example.com"})
        unknown = self.client.post("/api/auth/resend-verification", json={"email": "nobody@example.com"})
        self.assertEqual(len(MAIL.sent), sent)                # inside the 60 s cooldown: nothing sent
        self.assertEqual((known.status_code, known.json()), (unknown.status_code, unknown.json()))

    def test_at_most_five_codes_per_hour(self):
        for _ in range(10):
            age_codes(120)
            self.client.post("/api/auth/resend-verification", json={"email": "v@example.com"})
        self.assertEqual(len([m for m in MAIL.sent if m.to_email == "v@example.com"]), auth.MAX_CODES_PER_HOUR)

    def test_a_failed_email_send_is_reported_and_not_counted(self):
        class Broken:
            def send(self, message):
                raise mailer.MailError("The email service could not be reached. Try again shortly.")

        mailer.set_mailer(Broken())
        try:
            response = register(TestClient(main.app), "w@example.com")
        finally:
            mailer.set_mailer(MAIL)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "mail_failed")
        with db.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM email_codes e JOIN users u ON u.id = e.user_id "
                "WHERE u.email = 'w@example.com'").fetchone()[0], 0)


class PasswordResetTests(unittest.TestCase):
    def setUp(self):
        fresh_database()
        self.client = signed_in(TestClient(main.app), "r@example.com")

    def forgot(self, email):
        return TestClient(main.app).post("/api/auth/password/forgot", json={"email": email})

    def reset(self, code, password=NEW_PASSWORD):
        return TestClient(main.app).post("/api/auth/password/reset",
                                         json={"email": "r@example.com", "code": code, "new_password": password})

    def login(self, password):
        return TestClient(main.app).post("/api/auth/login", json={"email": "r@example.com", "password": password})

    def test_forgot_answers_the_same_for_unknown_emails_and_sends_nothing(self):
        sent = len(MAIL.sent)
        unknown = self.forgot("nobody@example.com")
        self.assertEqual(len(MAIL.sent), sent)
        known = self.forgot("r@example.com")
        self.assertEqual(len(MAIL.sent), sent + 1)
        self.assertEqual(MAIL.sent[-1].purpose, "reset your password")
        self.assertEqual((known.status_code, known.json()), (unknown.status_code, unknown.json()))

    def test_reset_changes_the_password_signs_out_everywhere_and_is_single_use(self):
        self.forgot("r@example.com")
        code = MAIL.last_code("r@example.com")
        self.assertEqual(self.reset(code).status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)   # old session revoked
        self.assertEqual(self.login(PASSWORD).status_code, 401)
        self.assertEqual(self.login(NEW_PASSWORD).status_code, 200)
        self.assertEqual(self.reset(code, "yet another passphrase").status_code, 400)  # used up

    def test_a_wrong_code_leaves_the_password_unchanged(self):
        self.forgot("r@example.com")
        code = MAIL.last_code("r@example.com")
        self.assertEqual(self.reset("000000" if code != "000000" else "111111").status_code, 400)
        self.assertEqual(self.login(PASSWORD).status_code, 200)

    def test_weak_new_password_is_rejected_before_the_code_is_spent(self):
        self.forgot("r@example.com")
        code = MAIL.last_code("r@example.com")
        self.assertEqual(self.reset(code, "short").status_code, 400)
        self.assertEqual(self.reset(code).status_code, 200)

    def test_reset_also_verifies_an_unverified_email(self):
        auth.create_user("u@example.com", PASSWORD)
        TestClient(main.app).post("/api/auth/password/forgot", json={"email": "u@example.com"})
        TestClient(main.app).post("/api/auth/password/reset", json={
            "email": "u@example.com", "code": MAIL.last_code("u@example.com"), "new_password": NEW_PASSWORD})
        self.assertTrue(auth.get_user_by_email("u@example.com").email_verified)


class AccountSettingsTests(unittest.TestCase):
    def setUp(self):
        fresh_database()
        self.client = signed_in(TestClient(main.app), "s@example.com")
        self.other_device = TestClient(main.app)
        self.other_device.post("/api/auth/login", json={"email": "s@example.com", "password": PASSWORD})

    def change(self, current, new):
        return self.client.post("/api/account/password",
                                json={"current_password": current, "new_password": new})

    def test_change_password_needs_the_current_one(self):
        self.assertEqual(self.change("wrong password!!", NEW_PASSWORD).status_code, 401)
        self.assertEqual(self.change(PASSWORD, "short").status_code, 400)
        self.assertEqual(self.change(PASSWORD, PASSWORD).status_code, 400)

    def test_changing_password_keeps_this_session_and_ends_the_others(self):
        self.assertEqual(self.change(PASSWORD, NEW_PASSWORD).status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.other_device.get("/api/auth/me").status_code, 401)
        login = TestClient(main.app).post("/api/auth/login", json={"email": "s@example.com", "password": NEW_PASSWORD})
        self.assertEqual(login.status_code, 200)

    def test_profile_name_can_be_changed(self):
        response = self.client.post("/api/account/profile", json={"name": "Sam"})
        self.assertEqual(response.json()["user"]["name"], "Sam")
        self.assertEqual(self.client.get("/api/auth/me").json()["user"]["name"], "Sam")

    def test_sign_out_everywhere_ends_every_session(self):
        self.assertEqual(self.client.post("/api/account/sign-out-everywhere").status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.other_device.get("/api/auth/me").status_code, 401)

    def test_account_endpoints_need_csrf(self):
        token = self.client.headers.pop(auth.CSRF_HEADER)
        self.assertEqual(self.client.post("/api/account/profile", json={"name": "x"}).status_code, 403)
        self.client.headers[auth.CSRF_HEADER] = token


class EmailJSTests(unittest.TestCase):
    MESSAGE = mailer.Message(to_email="to@example.com", to_name="To", code="123456",
                             purpose="verify your email address", expires_minutes=10)

    def test_request_shape_uses_the_private_key_and_template_params(self):
        sender = mailer.EmailJSMailer("service_x", "template_y", "public_z", "private_w")
        with patch("mailer.httpx.post", return_value=httpx.Response(200, text="OK")) as post:
            sender.send(self.MESSAGE)
        url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
        self.assertEqual(url, "https://api.emailjs.com/api/v1.0/email/send")
        self.assertEqual((payload["service_id"], payload["template_id"], payload["user_id"], payload["accessToken"]),
                         ("service_x", "template_y", "public_z", "private_w"))
        self.assertEqual(payload["template_params"]["to_email"], "to@example.com")
        self.assertEqual(payload["template_params"]["otp_code"], "123456")

    def test_rejections_and_network_errors_become_mail_errors(self):
        sender = mailer.EmailJSMailer("s", "t", "p", "k")
        with patch("mailer.httpx.post", return_value=httpx.Response(403, text="API calls are disabled")):
            with self.assertRaises(mailer.MailError):
                sender.send(self.MESSAGE)
        with patch("mailer.httpx.post", side_effect=httpx.ConnectError("down")):
            with self.assertRaises(mailer.MailError):
                sender.send(self.MESSAGE)

    def test_mode_follows_the_environment(self):
        keys = {"EMAILJS_SERVICE_ID": "s", "EMAILJS_TEMPLATE_ID": "t",
                "EMAILJS_PUBLIC_KEY": "p", "EMAILJS_PRIVATE_KEY": "k", "APP_EMAIL_MODE": ""}
        with patch.dict(os.environ, keys):
            self.assertIsInstance(mailer._from_environment(), mailer.EmailJSMailer)
        empty = {name: "" for name in keys}
        with patch.dict(os.environ, empty):
            self.assertIsInstance(mailer._from_environment(), mailer.ConsoleMailer)
        with patch.dict(os.environ, {**empty, "APP_EMAIL_MODE": "emailjs"}):
            with self.assertRaises(RuntimeError):
                mailer._from_environment()


class ExistingUsersTests(unittest.TestCase):
    def test_signed_in_clients_report_a_verified_email(self):
        client = authed_client(main.app)
        self.assertTrue(client.get("/api/auth/me").json()["user"]["email_verified"])


if __name__ == "__main__":
    unittest.main()
