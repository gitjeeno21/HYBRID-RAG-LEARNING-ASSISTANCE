"""Self-reflective RAG agent using hybrid retrieval and a bounded retry loop."""

import os
from collections.abc import Sequence

from langchain_core.documents import Document

from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PROMPT_TEMPLATE,
    TOP_K,
    citation,
)

MAX_RETRIES = 2
REFUSAL = "I don't know the answer based on the provided documents."
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 50
EXPECTED_CHUNK_COUNT = 281
INDEX_PATH = BASE_DIR / "chroma_db_eval_1500_50"
COLLECTION_NAME = "eval_1500_50"


class AutonomousRAGAgent:
    """Retrieve, grade, and answer with up to two rewritten-query retries."""

    def __init__(self) -> None:
        self._store = None
        self._chunks: list[Document] | None = None
        self._bm25 = None
        self._llm_client = None

    def answer(self, question: str) -> str:
        """Return a cited answer or the fixed refusal after exhausted retries."""
        if not question.strip():
            raise ValueError("question must not be empty")

        query = question
        retries = 0
        while True:
            chunks = self._retrieve(query)
            relevant = self._grade_relevance(chunks, question)
            if relevant:
                return self._generate_answer(chunks, question)

            if retries >= MAX_RETRIES:
                return REFUSAL

            query = self._rewrite_query(question, query)
            retries += 1

    def _retrieve(self, query: str) -> list[Document]:
        """Retrieve top-k documents using the existing fixed hybrid search."""
        self._ensure_retriever()
        from hybrid import hybrid_search

        assert self._store is not None
        assert self._chunks is not None
        return hybrid_search(
            query,
            self._store,
            self._bm25,
            self._chunks,
            n_results=TOP_K,
        )

    def _ensure_retriever(self) -> None:
        if self._store is not None:
            return
        if not INDEX_PATH.is_dir():
            raise FileNotFoundError(f"Existing 1500/50 index not found: {INDEX_PATH}")

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        from ingest import load_documents, split_documents

        documents = load_documents()
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=str(INDEX_PATH),
        )
        indexed_count = store._collection.count()
        if indexed_count != EXPECTED_CHUNK_COUNT or indexed_count != len(chunks):
            raise RuntimeError(
                "The existing 1500/50 index does not match the expected corpus: "
                f"index={indexed_count}, chunks={len(chunks)}, "
                f"expected={EXPECTED_CHUNK_COUNT}"
            )

        hybrid = self._load_hybrid_module()
        self._chunks = chunks
        self._bm25 = hybrid.build_bm25(chunks)
        self._store = store

    @staticmethod
    def _load_hybrid_module():
        """Import hybrid without letting evaluator.py read .env at import time."""
        env_names = ("PYTHON_DOTENV_DISABLED", "GROQ_API_KEY")
        previous = {name: os.environ.get(name) for name in env_names}
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ["GROQ_API_KEY"] = "hybrid-import-placeholder"
        try:
            import hybrid
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return hybrid

    def _grade_relevance(self, chunks: Sequence[Document], question: str) -> bool:
        """Ask the LLM whether the retrieved chunks answer the question."""
        context = self._format_context(chunks)
        prompt = (
            "You are a relevance grader for a retrieval-augmented question "
            "answering system. Decide whether the retrieved context contains "
            "information relevant to answering the question. Reply with exactly "
            "YES if relevant or NO if irrelevant.\n\n"
            f"Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
            "Answer only YES or NO."
        )
        response = self._call_llm(prompt).strip().upper().rstrip(".!")
        if response not in {"YES", "NO"}:
            raise ValueError(f"Relevance grader returned an invalid response: {response!r}")
        return response == "YES"

    def _rewrite_query(self, question: str, previous_query: str) -> str:
        """Ask the LLM to rewrite a failed search query."""
        prompt = (
            "Rewrite this question to retrieve better search results. Preserve "
            "its intent and do not answer it.\n\n"
            f"Original question:\n{question}\n\n"
            f"Previous search query:\n{previous_query}\n\n"
            "Return only the rewritten search query."
        )
        rewritten = self._call_llm(prompt).strip()
        if not rewritten:
            raise ValueError("Query rewriting returned an empty search query")
        return rewritten

    def _generate_answer(
        self, chunks: Sequence[Document], question: str
    ) -> str:
        """Generate the final answer from relevant chunks with page citations."""
        context = self._format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._call_llm(prompt).strip()
        if not answer:
            raise RuntimeError("Answer generation returned an empty response")
        return answer

    @staticmethod
    def _format_context(chunks: Sequence[Document]) -> str:
        return "\n\n".join(
            f"[Chunk {rank}] Source: {citation(chunk.metadata)}\n"
            f"{chunk.page_content}"
            for rank, chunk in enumerate(chunks, start=1)
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Groq only when an actual, non-mocked agent method needs it."""
        if self._llm_client is None:
            from dotenv import load_dotenv
            from groq import Groq

            load_dotenv()
            self._llm_client = Groq()
        response = self._llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The LLM returned an empty response")
        return content


def _run_mocked_loop_test() -> None:
    """Exercise the success and refusal paths without retrieval or API calls."""

    class MockedAgent(AutonomousRAGAgent):
        def __init__(self, grades: list[bool]) -> None:
            super().__init__()
            self.grades = iter(grades)
            self.queries: list[str] = []
            self.rewrite_count = 0
            self.generation_count = 0

        def _retrieve(self, query: str) -> list[Document]:
            self.queries.append(query)
            return [Document(page_content="mock retrieved context", metadata={})]

        def _grade_relevance(
            self, chunks: Sequence[Document], question: str
        ) -> bool:
            return next(self.grades)

        def _rewrite_query(self, question: str, previous_query: str) -> str:
            self.rewrite_count += 1
            return f"rewritten query {self.rewrite_count}"

        def _generate_answer(
            self, chunks: Sequence[Document], question: str
        ) -> str:
            self.generation_count += 1
            return "Mock answer with citations."

    success = MockedAgent([False, True])
    assert success.answer("sample question") == "Mock answer with citations."
    assert success.queries == ["sample question", "rewritten query 1"]
    assert success.rewrite_count == 1
    assert success.generation_count == 1

    exhausted = MockedAgent([False, False, False])
    assert exhausted.answer("unanswerable question") == REFUSAL
    assert exhausted.queries == [
        "unanswerable question",
        "rewritten query 1",
        "rewritten query 2",
    ]
    assert exhausted.rewrite_count == MAX_RETRIES
    assert exhausted.generation_count == 0
    print("Mocked agent loop checks passed: rewrite/retry, generation, and refusal.")


if __name__ == "__main__":
    _run_mocked_loop_test()
