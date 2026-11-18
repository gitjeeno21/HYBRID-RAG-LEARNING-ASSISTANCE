"""Deleting an account removes every trace of that user, and nothing of anyone else."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the embedding model is already cached locally

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import db
import main
import workspace
from auth_testing import PASSWORD, fresh_database, make_pdf, signed_in


class AccountDeletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)   # Chroma holds files open on Windows
        self.addCleanup(self.tmp.cleanup)
        workspace.use_storage(Path(self.tmp.name) / "storage")
        self.addCleanup(workspace.use_storage, main.BASE_DIR / "storage")
        fresh_database()
        self.leaver = signed_in(TestClient(main.app), "leaver@example.com")
        self.stayer = signed_in(TestClient(main.app), "stayer@example.com")
        for client, text in ((self.leaver, "Leaver notes mention PELICAN77."),
                             (self.stayer, "Stayer notes mention OSPREY42.")):
            response = client.post("/api/upload", files={"file": ("notes.pdf", make_pdf(text), "application/pdf")})
            self.assertEqual(response.status_code, 200, response.text)
        self.leaver.post("/api/progress/quiz", json={"pdf": "notes.pdf", "difficulty": "Easy", "score": 80, "topics": []})
        self.leaver_id = self.leaver.get("/api/auth/me").json()["user"]["id"]

    def delete(self, password=PASSWORD):
        return self.leaver.post("/api/account/delete", json={"password": password})

    def vectors(self, user_id):
        return len(workspace.Workspace(user_id).store.get()["metadatas"])

    def test_a_wrong_password_deletes_nothing(self):
        self.assertEqual(self.delete("not my password!!").status_code, 401)
        self.assertEqual(self.leaver.get("/api/pdfs").json()["pdfs"], ["notes.pdf"])
        self.assertGreater(self.vectors(self.leaver_id), 0)

    def test_deletion_removes_account_files_vectors_and_records(self):
        root = workspace.Workspace(self.leaver_id).root
        self.assertTrue(root.exists())
        self.assertEqual(self.delete().status_code, 200)

        self.assertFalse(root.exists())                                   # PDFs, cache, archive
        self.assertEqual(self.vectors(self.leaver_id), 0)                 # search index
        with db.connect() as conn:                                        # account and every record
            for table in ("users", "sessions", "quiz_history", "chat_feedback", "llm_usage", "email_codes"):
                column = "id" if table == "users" else "user_id"
                count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (self.leaver_id,)).fetchone()[0]
                self.assertEqual(count, 0, table)
        self.assertEqual(self.leaver.get("/api/auth/me").status_code, 401)  # signed out
        login = TestClient(main.app).post("/api/auth/login", json={"email": "leaver@example.com", "password": PASSWORD})
        self.assertEqual(login.status_code, 401)

    def test_another_users_data_is_untouched(self):
        self.delete()
        texts = self.stayer.post("/api/retrieval/inspect", json={"query": "notes mention", "top_k": 5}).json()
        joined = " ".join(item["text"] for item in texts["selected"])
        self.assertIn("OSPREY42", joined)
        self.assertNotIn("PELICAN77", joined)
        self.assertEqual(self.stayer.get("/api/pdfs").json()["pdfs"], ["notes.pdf"])

    def test_the_email_can_register_again_afterwards(self):
        self.delete()
        again = signed_in(TestClient(main.app), "leaver@example.com")
        self.assertEqual(again.get("/api/pdfs").json()["pdfs"], [])        # a fresh, empty library


if __name__ == "__main__":
    unittest.main()
