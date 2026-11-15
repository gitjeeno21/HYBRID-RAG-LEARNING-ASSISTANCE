"""Per-user isolation, end to end: real PDFs, real embeddings, real Chroma.

Two users upload papers containing unique secret tokens. Every way one user
could reach the other's data is attempted through the public API, and each
must fail or return nothing. Nothing on the data path is mocked; only the LLM
is replaced, so the test can inspect exactly what would be sent to it.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the embedding model is already cached locally

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import auth
import main
import workspace
from auth_testing import fresh_database, signed_in
from workspace import ScopedStore, scoped


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


ALICE_SECRET = "Alice private launch code is ZEBRA4471 for the orbital project."
BOB_TEXT = "Bob studies gradient descent and learning rate schedules for neural networks."


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Chroma keeps its files open on Windows, so a locked temp file is left for the OS.
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        workspace.use_storage(Path(cls.tmp.name) / "storage")
        fresh_database()
        cls.alice = signed_in(TestClient(main.app), "alice@example.com")
        cls.bob = signed_in(TestClient(main.app), "bob@example.com")
        for client, name, text in ((cls.alice, "secret.pdf", ALICE_SECRET),
                                   (cls.alice, "shared.pdf", "Alice shared name document about KESTREL9 birds."),
                                   (cls.bob, "notes.pdf", BOB_TEXT),
                                   (cls.bob, "shared.pdf", "Bob shared name document about optimisers.")):
            response = client.post("/api/upload", files={"file": (name, make_pdf(text), "application/pdf")})
            assert response.status_code == 200, response.text

    @classmethod
    def tearDownClass(cls):
        workspace.use_storage(main.BASE_DIR / "storage")
        cls.tmp.cleanup()

    def inspect(self, client, query, papers=()):
        response = client.post("/api/retrieval/inspect", json={"query": query, "papers": list(papers), "top_k": 10})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        texts = " ".join(item["text"] for item in body["selected"])
        candidates = [c["citation"] for s in body["searches"] for c in s["candidates"]]
        return texts, candidates

    # --- library ---------------------------------------------------------
    def test_each_user_lists_only_their_own_papers(self):
        self.assertEqual(self.alice.get("/api/pdfs").json()["pdfs"], ["secret.pdf", "shared.pdf"])
        self.assertEqual(self.bob.get("/api/pdfs").json()["pdfs"], ["notes.pdf", "shared.pdf"])
        self.assertEqual(self.bob.get("/api/health").json()["papers"], 2)

    def test_another_users_file_cannot_be_previewed_opened_or_deleted(self):
        for method, path in (("GET", "/api/library/secret.pdf/preview"),
                             ("GET", "/api/library/secret.pdf/file"),
                             ("DELETE", "/api/library/secret.pdf")):
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(self.bob.request(method, path).status_code, 404)
        self.assertIn("secret.pdf", self.alice.get("/api/pdfs").json()["pdfs"])

    def test_path_traversal_is_rejected(self):
        for name in ("../1/pdfs/secret.pdf", "..%2F1%2Fpdfs%2Fsecret.pdf"):
            with self.subTest(name=name):
                self.assertIn(self.bob.get(f"/api/library/{name}/file").status_code, {404, 422})

    # --- retrieval -------------------------------------------------------
    def test_search_never_returns_another_users_passages(self):
        texts, candidates = self.inspect(self.bob, "launch code ZEBRA4471 orbital project")
        self.assertNotIn("ZEBRA4471", texts)
        self.assertFalse([c for c in candidates if c.startswith("secret.pdf")])
        texts, _ = self.inspect(self.alice, "launch code ZEBRA4471 orbital project")
        self.assertIn("ZEBRA4471", texts)   # the owner still finds it

    def test_naming_another_users_paper_in_a_scoped_search_is_refused(self):
        response = self.bob.post("/api/retrieval/inspect", json={"query": "launch code", "papers": ["secret.pdf"]})
        self.assertEqual(response.status_code, 404)

    def test_same_filename_in_two_libraries_stays_separate(self):
        alice_text, _ = self.inspect(self.alice, "shared name document", papers=["shared.pdf"])
        bob_text, _ = self.inspect(self.bob, "shared name document", papers=["shared.pdf"])
        self.assertIn("KESTREL9", alice_text)
        self.assertNotIn("KESTREL9", bob_text)
        self.assertIn("optimisers", bob_text)

    def test_the_llm_prompt_contains_only_the_askers_passages(self):
        prompts = []

        def fake_llm(prompt, **_):
            prompts.append(prompt)
            return "I don't know based on the provided documents."

        with patch("main.call_llm", side_effect=fake_llm), \
                patch("main.assess", return_value={"label": "n/a", "reason": "test"}):
            response = self.bob.post("/api/chat", json={"query": "What is the launch code ZEBRA4471?"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(prompts)
        self.assertNotIn("ZEBRA4471", prompts[0].split("Question:")[0])   # context part only
        self.assertFalse([c for c in response.json()["citations"] if c["filename"] == "secret.pdf"])

    def test_quiz_and_compare_refuse_another_users_paper(self):
        quiz = self.bob.post("/api/quiz/generate", json={"pdf": "secret.pdf", "topic": "launch"})
        compare = self.bob.post("/api/compare", json={"pdfs": ["notes.pdf", "secret.pdf"], "question": "x"})
        self.assertEqual(quiz.status_code, 404)
        self.assertEqual(compare.status_code, 404)

    # --- records -----------------------------------------------------------
    def test_quiz_history_and_progress_are_private(self):
        self.alice.post("/api/progress/quiz", json={"pdf": "secret.pdf", "difficulty": "Hard",
                                                    "score": 30, "topics": ["launch codes"]})
        bob_progress = self.bob.get("/api/progress").json()
        self.assertNotIn("launch codes", bob_progress["most_missed_topics"])
        self.assertFalse([r for r in bob_progress["recent"] if r["pdf"] == "secret.pdf"])

    # --- the store wrapper itself -------------------------------------------
    def test_scoped_store_adds_the_user_filter_to_every_operation(self):
        calls = []

        class Recorder:
            def similarity_search_with_score(self, query, k, filter):
                calls.append(("search", filter)); return []

            def delete(self, where):
                calls.append(("delete", where))

            def get(self, where, include):
                calls.append(("get", where)); return {"documents": [], "metadatas": []}

            def add_documents(self, documents):
                calls.append(("add", [d.metadata["user_id"] for d in documents])); return []

        from langchain_core.documents import Document
        store = ScopedStore(Recorder(), 7)
        store.similarity_search_with_score("q", k=3, filter={"filename": "a.pdf"})
        store.delete(where={"filename": "a.pdf"})
        store.delete()
        store.get()
        store.add_documents([Document(page_content="x", metadata={"user_id": 99, "filename": "a.pdf"})])
        self.assertEqual(calls, [
            ("search", {"$and": [{"user_id": 7}, {"filename": "a.pdf"}]}),
            ("delete", {"$and": [{"user_id": 7}, {"filename": "a.pdf"}]}),
            ("delete", {"user_id": 7}),
            ("get", {"user_id": 7}),
            ("add", [7]),   # a forged user_id in metadata is overwritten
        ])
        self.assertEqual(scoped({"user_id": 1}, {"$or": [{"a": 1}, {"b": 2}]}),
                         {"$and": [{"user_id": 1}, {"$or": [{"a": 1}, {"b": 2}]}]})

    def test_signing_out_ends_access(self):
        client = signed_in(TestClient(main.app), "carol@example.com")
        client.post("/api/auth/logout")
        self.assertEqual(client.get("/api/pdfs").status_code, 401)
        self.assertIsNotNone(auth.get_user_by_email("carol@example.com"))


if __name__ == "__main__":
    unittest.main()
