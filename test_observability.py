"""Request logging, request ids, security headers and the public liveness probe."""

import json
import unittest

from fastapi.testclient import TestClient

import main
from auth_testing import PASSWORD, authed_client, fresh_database


class ObservabilityTests(unittest.TestCase):
    def test_liveness_is_public_and_reveals_nothing(self):
        response = TestClient(main.app).get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_every_response_carries_security_headers_and_a_request_id(self):
        response = TestClient(main.app).get("/")
        for header in ("Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options"):
            self.assertIn(header, response.headers)
        self.assertRegex(response.headers["X-Request-ID"], r"^[0-9a-f]{16}$")

    def test_a_well_formed_request_id_is_echoed_and_a_malformed_one_replaced(self):
        client = TestClient(main.app)
        self.assertEqual(client.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"], "abc-123")
        replaced = client.get("/healthz", headers={"X-Request-ID": "bad id\nInjected: yes"}).headers["X-Request-ID"]
        self.assertRegex(replaced, r"^[0-9a-f]{16}$")

    def test_api_requests_are_logged_with_user_but_never_with_bodies(self):
        client = authed_client(main.app)
        with self.assertLogs("rag.access", level="INFO") as logs:
            client.post("/api/chat/feedback", json={"message": "my private question", "feedback": "up"})
        entry = json.loads(logs.output[-1].split(":", 2)[2])
        self.assertEqual(entry["path"], "/api/chat/feedback")
        self.assertEqual(entry["status"], 200)
        self.assertIsInstance(entry["user_id"], int)
        self.assertNotIn("my private question", logs.output[-1])

    def test_passwords_never_reach_the_log(self):
        fresh_database()
        with self.assertLogs("rag.access", level="INFO") as logs:
            TestClient(main.app).post("/api/auth/register", json={"email": "z@example.com", "password": PASSWORD})
        self.assertFalse([line for line in logs.output if PASSWORD in line])


if __name__ == "__main__":
    unittest.main()
