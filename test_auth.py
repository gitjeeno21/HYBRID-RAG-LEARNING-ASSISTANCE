"""Accounts and sessions: each test pins down one security guarantee."""

import unittest

from fastapi.testclient import TestClient

import auth
import db
import main
from auth_testing import PASSWORD, authed_client, fresh_database, signed_in


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        fresh_database()
        self.client = TestClient(main.app)

    def test_register_signs_in_and_never_stores_the_plain_password(self):
        response = self.client.post("/api/auth/register",
                                    json={"email": "Ada@Example.com", "password": PASSWORD, "name": "Ada"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user"]["email"], "ada@example.com")
        self.assertEqual(self.client.get("/api/auth/me").json()["user"]["name"], "Ada")
        with db.connect() as conn:
            stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertTrue(stored.startswith("$argon2id$"))
        self.assertNotIn(PASSWORD, stored)

    def test_duplicate_email_is_rejected_case_insensitively(self):
        self.client.post("/api/auth/register", json={"email": "a@example.com", "password": PASSWORD})
        response = TestClient(main.app).post("/api/auth/register",
                                             json={"email": "A@EXAMPLE.com", "password": PASSWORD})
        self.assertEqual(response.status_code, 409)

    def test_weak_password_and_bad_email_are_rejected(self):
        for payload in ({"email": "a@example.com", "password": "short"},
                        {"email": "not-an-email", "password": PASSWORD},
                        {"email": "a@example.com", "password": "a@example.com"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/auth/register", json=payload).status_code, 400)

    def test_session_cookie_is_httponly_and_csrf_cookie_is_readable(self):
        response = self.client.post("/api/auth/register", json={"email": "c@example.com", "password": PASSWORD})
        cookies = response.headers.get_list("set-cookie")
        session = next(c for c in cookies if c.startswith(auth.SESSION_COOKIE + "="))
        csrf = next(c for c in cookies if c.startswith(auth.CSRF_COOKIE + "="))
        self.assertIn("HttpOnly", session)
        self.assertIn("samesite=lax", session.lower())
        self.assertNotIn("HttpOnly", csrf)


class LoginTests(unittest.TestCase):
    def setUp(self):
        fresh_database()
        auth.create_user("user@example.com", PASSWORD)

    def login(self, email="user@example.com", password=PASSWORD):
        return TestClient(main.app).post("/api/auth/login", json={"email": email, "password": password})

    def test_correct_credentials_sign_in(self):
        self.assertEqual(self.login().status_code, 200)

    def test_wrong_password_and_unknown_email_get_the_same_answer(self):
        wrong, unknown = self.login(password="wrong password!!"), self.login(email="nobody@example.com")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())

    def test_repeated_failures_lock_the_account_even_for_the_right_password(self):
        for _ in range(auth.MAX_FAILURES):
            self.login(password="wrong password!!")
        self.assertEqual(self.login().status_code, 429)

    def test_cross_site_login_is_blocked(self):
        response = TestClient(main.app).post(
            "/api/auth/login", json={"email": "user@example.com", "password": PASSWORD},
            headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status_code, 403)


class SessionTests(unittest.TestCase):
    def test_every_api_route_except_sign_in_requires_a_session(self):
        fresh_database()
        anonymous = TestClient(main.app)
        protected = [route for route in main.app.routes
                     if getattr(route, "path", "").startswith("/api/")
                     and route.path not in main.OPEN_API_PATHS]
        self.assertGreater(len(protected), 15)
        for route in protected:
            method = sorted(route.methods - {"HEAD"})[0]
            path = route.path.replace("{filename:path}", "x.pdf")
            with self.subTest(route=f"{method} {path}"):
                response = anonymous.request(method, path, json={})
                self.assertEqual(response.status_code, 401)

    def test_writes_need_the_matching_csrf_token(self):
        client = authed_client(main.app)
        good = client.headers.pop(auth.CSRF_HEADER)
        payload = {"message": "m", "feedback": "up"}
        self.assertEqual(client.post("/api/chat/feedback", json=payload).status_code, 403)
        self.assertEqual(client.post("/api/chat/feedback", json=payload,
                                     headers={auth.CSRF_HEADER: "forged"}).status_code, 403)
        self.assertEqual(client.post("/api/chat/feedback", json=payload,
                                     headers={auth.CSRF_HEADER: good}).status_code, 200)

    def test_logout_revokes_the_session_server_side(self):
        client = authed_client(main.app)
        stolen = client.cookies.get(auth.SESSION_COOKIE)
        client.post("/api/auth/logout")
        replay = TestClient(main.app, cookies={auth.SESSION_COOKIE: stolen})
        self.assertEqual(replay.get("/api/auth/me").status_code, 401)

    def test_database_stores_only_a_hash_of_the_session_token(self):
        client = authed_client(main.app)
        token = client.cookies.get(auth.SESSION_COOKIE)
        with db.connect() as conn:
            stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertNotEqual(stored, token)
        self.assertIsNone(auth.resolve_session(stored))  # the hash itself is not a valid cookie

    def test_expired_sessions_are_rejected(self):
        client = authed_client(main.app)
        with db.connect() as conn:
            conn.execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00Z'")
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_two_users_get_separate_sessions(self):
        fresh_database()
        alice = signed_in(TestClient(main.app), "alice@example.com")
        bob = signed_in(TestClient(main.app), "bob@example.com")
        self.assertEqual(alice.get("/api/auth/me").json()["user"]["email"], "alice@example.com")
        self.assertEqual(bob.get("/api/auth/me").json()["user"]["email"], "bob@example.com")


if __name__ == "__main__":
    unittest.main()
