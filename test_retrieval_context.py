"""Stage 1 regressions: passages sent to the model, index verification, upload."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import main
from hybrid import unique_chunks


def doc(text: str, page: int = 0, name: str = "a.pdf") -> Document:
    return Document(page_content=text, metadata={"filename": name, "page": page})


class FakeStore:
    """Only the Chroma `get` call that verify_index_matches uses."""

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks

    def get(self, include: list[str]) -> dict:
        return {
            "documents": [chunk.page_content for chunk in self.chunks],
            "metadatas": [chunk.metadata for chunk in self.chunks],
        }


class UniqueChunksTests(unittest.TestCase):
    def test_two_passages_from_one_page_are_both_kept(self) -> None:
        first, second = doc("first half of page 1"), doc("second half of page 1")
        kept = unique_chunks([(first, 0.5), (second, 0.4)])
        self.assertEqual([d.page_content for d, _ in kept],
                         ["first half of page 1", "second half of page 1"])

    def test_an_exactly_repeated_chunk_is_dropped_keeping_rank_order(self) -> None:
        a, b = doc("alpha"), doc("beta", page=1)
        kept = unique_chunks([(a, 0.9), (b, 0.8), (doc("alpha"), 0.7)])
        self.assertEqual([(d.page_content, s) for d, s in kept], [("alpha", 0.9), ("beta", 0.8)])


class RetrieveForApiTests(unittest.TestCase):
    def test_same_page_passages_all_reach_the_prompt(self) -> None:
        hits = [(doc("part one", page=2), 0.6), (doc("part two", page=2), 0.5),
                (doc("other page", page=3), 0.4)]
        with patch("main.get_hybrid_resources", return_value=([doc("x")], object())), \
                patch("main.get_vector_store"), \
                patch("main.hybrid_search_with_scores", return_value=hits):
            retrieved = main.retrieve_for_api("q", [], 3)
        self.assertEqual(len(retrieved), 3)
        self.assertEqual([item["citation"] for item in retrieved],
                         ["a.pdf, p. 3", "a.pdf, p. 3", "a.pdf, p. 4"])
        # The citation list shown with the answer names each page once.
        self.assertEqual([c["citation"] for c in main._citations(retrieved)],
                         ["a.pdf, p. 3", "a.pdf, p. 4"])


class VerifyIndexTests(unittest.TestCase):
    def test_identical_corpora_pass(self) -> None:
        chunks = [doc("one"), doc("two", page=1)]
        main.verify_index_matches(FakeStore(list(reversed(chunks))), chunks)  # order may differ

    def test_same_count_but_different_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "1 chunk\\(s\\) missing.*1 stale"):
            main.verify_index_matches(FakeStore([doc("one"), doc("old text")]),
                                      [doc("one"), doc("new text")])

    def test_a_duplicated_chunk_in_the_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale"):
            main.verify_index_matches(FakeStore([doc("one"), doc("one")]), [doc("one")])


class UploadEndpointTests(unittest.TestCase):
    def test_upload_runs_in_the_thread_pool_not_on_the_event_loop(self) -> None:
        self.assertFalse(inspect.iscoroutinefunction(main.upload_pdf))

    def test_upload_still_indexes_and_reports_chunks(self) -> None:
        add = MagicMock(return_value=7)
        with patch("main.add_pdf_to_library", add), patch("main.get_vector_store"), \
                patch("main.clear_indexes"), patch("main.get_hybrid_resources"):
            response = TestClient(main.app).post(
                "/api/upload", files={"file": ("paper.pdf", b"%PDF-1.4 test", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks_indexed"], 7)
        self.assertEqual(add.call_args.args[1], b"%PDF-1.4 test")

    def test_oversized_upload_is_rejected(self) -> None:
        with patch("main.MAX_UPLOAD_BYTES", 4):
            response = TestClient(main.app).post(
                "/api/upload", files={"file": ("big.pdf", b"%PDF-1.4", "application/pdf")})
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
