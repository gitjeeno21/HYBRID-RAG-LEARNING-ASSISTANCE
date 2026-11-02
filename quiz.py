"""Grounded five-question quizzes built only from retrieved PDF passages."""

from collections.abc import Callable, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import citation
from hybrid import VectorSearchStore, hybrid_search


def retrieve_quiz_chunks(
    vector_store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: Sequence[Document],
    pdf_name: str,
    topic: str,
) -> list[Document]:
    """Retrieve quiz evidence with hybrid search restricted to the chosen PDF."""
    return hybrid_search(
        topic,
        vector_store,
        bm25,
        list(chunks),
        metadata_filter={"filename": pdf_name},
    )

QUIZ_PROMPT_TEMPLATE = (
    "Create a practice quiz using ONLY the retrieved text below. Do not use "
    "outside knowledge, assumptions, or facts from memory.\n"
    "The selected PDF is {pdf_name}. The requested topic is: {topic}\n\n"
    "If the retrieved text does not cover the topic well enough to support "
    "questions and answers, do not invent a quiz. Reply with exactly: "
    "\"The retrieved text does not cover this topic, so I can't generate a "
    "grounded quiz.\"\n"
    "Otherwise, generate exactly 5 distinct practice questions. Give each a "
    "short answer supported by the retrieved text and cite its source page "
    "using the exact citation labels below. Do not cite pages or make claims "
    "that are absent from the retrieved text.\n\n"
    "Retrieved text:\n{context}\n\n"
    "Use this format:\n"
    "1. **Question:** ...\n"
    "   **Short answer:** ...\n"
    "   **Source:** <filename, p. X>\n"
    "... continue through 5."
)


def build_quiz_prompt(
    pdf_name: str,
    topic: str,
    chunks: Sequence[Document],
) -> str:
    """Create one prompt whose only evidence is the selected PDF's retrieved text."""
    if chunks:
        # Add citation labels before prompting so answers can cite exact source
        # pages without having to infer metadata from the passage itself.
        context = "\n\n".join(
            f"Source: {citation(chunk.metadata)}\n{chunk.page_content}"
            for chunk in chunks
        )
    else:
        # An empty evidence block tells the model it must report no coverage,
        # rather than making up questions to fill the requested five slots.
        context = "[No text was retrieved from the selected PDF.]"
    return QUIZ_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        topic=topic,
        context=context,
    )


def generate_quiz(
    pdf_name: str,
    topic: str,
    chunks: Sequence[Document],
    ask: Callable[[str], str] | None = None,
) -> str:
    """Generate a quiz with one LLM call; `ask` is injectable for offline tests."""
    prompt = build_quiz_prompt(pdf_name, topic, chunks)
    if ask is None:
        # Reuse the project's retry/backoff helper instead of adding another
        # Groq client or a second retry policy. This is one prompt/call on the
        # successful path, with retries only for transient API failures.
        from evaluator import call_llm

        answer = call_llm(prompt, temperature=0.2, label="quiz")
    else:
        # Tests can supply a deterministic response without contacting Groq.
        answer = ask(prompt)
    if not answer or not answer.strip():
        raise RuntimeError("Quiz generation returned an empty response")
    return answer.strip()
