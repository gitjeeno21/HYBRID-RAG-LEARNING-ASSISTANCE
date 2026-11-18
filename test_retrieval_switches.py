"""Offline tests for the Phase 3 retrieval switches (no models downloaded)."""

import unittest
from unittest import mock

from langchain_core.documents import Document

import hybrid
from ingest import split_documents_cross_page


def page(filename: str, number: int, text: str) -> Document:
    return Document(page_content=text, metadata={
        "filename": filename, "page": number - 1, "page_label": str(number),
    })


class CrossPageChunkTests(unittest.TestCase):
    def test_chunks_cross_page_breaks_and_keep_start_page(self):
        # PDF page text is made of lines, so use multi-line pages.
        pages = [page("a.pdf", 1, "alpha line\n" * 20), page("a.pdf", 2, "beta line\n" * 20),
                 page("b.pdf", 1, "gamma line\n" * 5)]
        chunks = split_documents_cross_page(pages, chunk_size=150, chunk_overlap=0)
        spanning = [c for c in chunks if c.metadata["page_label"] != c.metadata["page_end_label"]]
        self.assertTrue(spanning, "expected at least one chunk to cross the page break")
        chunk = spanning[0]
        self.assertIn("alpha", chunk.page_content)
        self.assertIn("beta", chunk.page_content)
        self.assertEqual(chunk.metadata["page_label"], "1")      # cited by start page
        self.assertEqual(chunk.metadata["page_end_label"], "2")
        # Files are never merged into one chunk.
        self.assertTrue(all(
            ("gamma" in c.page_content) == (c.metadata["filename"] == "b.pdf") for c in chunks
        ))

    def test_metadata_has_no_none_values_for_chroma(self):
        pages = [Document(page_content="x " * 50, metadata={"filename": "c.pdf", "page": 0})]
        for chunk in split_documents_cross_page(pages, chunk_size=40, chunk_overlap=0):
            self.assertTrue(all(value is not None for value in chunk.metadata.values()))


class RerankTests(unittest.TestCase):
    def test_rerank_reorders_only_the_pool_and_keeps_the_tail(self):
        chunks = [Document(page_content=f"c{i}") for i in range(5)]
        fake = mock.Mock()
        # Scores for the first 3 fused results (indices 4, 2, 0): 2 scores highest.
        fake.predict.return_value = [0.1, 0.9, 0.5]
        with mock.patch.object(hybrid, "_cross_encoder", return_value=fake):
            order = hybrid.rerank_indices("q", chunks, [4, 2, 0, 1, 3], pool=3)
        self.assertEqual(order, [2, 0, 4, 1, 3])
        pairs = fake.predict.call_args[0][0]
        self.assertEqual(pairs, [("q", "c4"), ("q", "c2"), ("q", "c0")])

    def test_switches_default_to_the_evaluated_configuration(self):
        import os

        import config
        if "RAG_TOP_K" not in os.environ:
            self.assertEqual(config.TOP_K, 5)            # adopted in Phase 3
        if "RAG_RERANK" not in os.environ:
            self.assertFalse(config.RERANK_ENABLED)      # rejected in Phase 3
        if "RAG_CROSS_PAGE" not in os.environ:
            self.assertFalse(config.CROSS_PAGE_CHUNKS)   # rejected in Phase 3
        self.assertEqual(hybrid.CANDIDATES_PER_METHOD, config.RETRIEVAL_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
