"""Offline tests for isolated comparison retrieval and prompt construction."""

import unittest

from langchain_core.documents import Document

from compare import build_compare_prompt, generate_comparison, retrieve_paper_contexts
from config import TOP_K


class FakeVectorStore:
    """Return distinguishable evidence and record each PDF-scoped search."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict]] = []

    def similarity_search(self, question: str, k: int, filter: dict) -> list[Document]:
        self.calls.append((question, k, filter))
        pdf_name = filter["filename"]
        if pdf_name == "paper_one.pdf":
            page_content = "Paper one uses a recurrent approach."
            page = 2
        else:
            page_content = "Paper two uses a self-attention approach."
            page = 5
        return [
            Document(
                page_content=page_content,
                metadata={"filename": pdf_name, "page": page - 1},
            )
        ]


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
        names = ["paper_one.pdf", "paper_two.pdf"]
        contexts = retrieve_paper_contexts(store, "What differs?", names)

        self.assertEqual(
            store.calls,
            [
                ("What differs?", TOP_K, {"filename": "paper_one.pdf"}),
                ("What differs?", TOP_K, {"filename": "paper_two.pdf"}),
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


if __name__ == "__main__":
    unittest.main()
