"""Offline tests for isolated comparison retrieval and prompt construction."""

import unittest

from langchain_core.documents import Document

from compare import build_compare_prompt, generate_comparison, retrieve_paper_contexts
from hybrid import (
    CANDIDATES_PER_METHOD,
    build_bm25,
    hybrid_search_with_scores,
)
from quiz import retrieve_quiz_chunks


class FakeVectorStore:
    """Return distinguishable evidence and record each PDF-scoped search."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict]] = []
        self.documents = [
            Document(
                page_content="Paper one uses a recurrent approach.",
                metadata={"filename": "paper_one.pdf", "page": 1},
            ),
            Document(
                page_content="Paper two uses a self-attention approach.",
                metadata={"filename": "paper_two.pdf", "page": 4},
            ),
        ]

    def similarity_search(
        self,
        question: str,
        k: int,
        filter: dict | None = None,
    ) -> list[Document]:
        self.calls.append((question, k, filter or {}))
        documents = self.documents
        if filter:
            documents = [
                document
                for document in documents
                if all(
                    document.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]
        return documents[:k]

    def similarity_search_with_relevance_scores(
        self,
        question: str,
        k: int,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        return [(document, 0.8) for document in self.similarity_search(question, k, filter)]


class CompareModeTests(unittest.TestCase):
    def test_prompt_explicitly_handles_a_paper_without_retrieved_evidence(self) -> None:
        contexts = {
            "paper_one.pdf": [],
            "paper_two.pdf": [
                Document(
                    page_content="Evidence from the second paper.",
                    metadata={"filename": "paper_two.pdf", "page": 0},
                )
            ],
        }

        prompt = build_compare_prompt("What is the approach?", contexts)

        self.assertIn("[No passages were retrieved from this paper.]", prompt)
        self.assertIn("explicitly state that the paper does not cover the topic", prompt)
        self.assertIn("Evidence from the second paper.", prompt)

    def test_isolated_retrieval_and_single_mocked_generation(self) -> None:
        store = FakeVectorStore()
        bm25 = build_bm25(store.documents)
        names = ["paper_one.pdf", "paper_two.pdf"]
        contexts = retrieve_paper_contexts(
            store, bm25, store.documents, "What differs?", names
        )

        self.assertEqual(
            store.calls,
            [
                (
                    "What differs?",
                    CANDIDATES_PER_METHOD,
                    {"filename": "paper_one.pdf"},
                ),
                (
                    "What differs?",
                    CANDIDATES_PER_METHOD,
                    {"filename": "paper_two.pdf"},
                ),
            ],
        )
        self.assertEqual(
            [doc.metadata["filename"] for doc in contexts["paper_one.pdf"]],
            ["paper_one.pdf"],
        )
        self.assertEqual(
            [doc.metadata["filename"] for doc in contexts["paper_two.pdf"]],
            ["paper_two.pdf"],
        )

        prompts: list[str] = []

        def mock_llm(prompt: str) -> str:
            prompts.append(prompt)
            return "Mocked two-paper comparison."

        result = generate_comparison("What differs?", contexts, ask=mock_llm)
        self.assertEqual(result, "Mocked two-paper comparison.")
        self.assertEqual(len(prompts), 1)
        prompt = prompts[0]
        self.assertIn("paper_one.pdf, p. 2", prompt)
        self.assertIn("paper_two.pdf, p. 5", prompt)
        self.assertIn("Paper one uses a recurrent approach.", prompt)
        self.assertIn("Paper two uses a self-attention approach.", prompt)
        self.assertIn("exactly two sentences", prompt)
        paper_one, paper_two = prompt.split("Paper 2: paper_two.pdf", maxsplit=1)
        self.assertIn("Paper 1: paper_one.pdf", paper_one)
        self.assertNotIn("Paper two uses a self-attention approach.", paper_one)
        self.assertNotIn("Paper one uses a recurrent approach.", paper_two)

    def test_quiz_retrieval_uses_filtered_hybrid_search(self) -> None:
        store = FakeVectorStore()
        bm25 = build_bm25(store.documents)

        results = retrieve_quiz_chunks(
            store,
            bm25,
            store.documents,
            "paper_two.pdf",
            "What differs?",
        )

        self.assertEqual(
            [document.metadata["filename"] for document in results],
            ["paper_two.pdf"],
        )
        self.assertEqual(
            store.calls,
            [
                (
                    "What differs?",
                    CANDIDATES_PER_METHOD,
                    {"filename": "paper_two.pdf"},
                )
            ],
        )

    def test_chat_hybrid_results_keep_dense_scores_for_trust(self) -> None:
        store = FakeVectorStore()
        bm25 = build_bm25(store.documents)

        results = hybrid_search_with_scores(
            "What differs?",
            store,
            bm25,
            store.documents,
        )

        self.assertTrue(results)
        self.assertTrue(all(score == 0.8 for _, score in results))


if __name__ == "__main__":
    unittest.main()
