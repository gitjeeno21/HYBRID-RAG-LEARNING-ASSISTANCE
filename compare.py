"""Retrieve and compare evidence from two PDFs in one LLM request."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from langchain_core.documents import Document

from config import TOP_K, citation


class SearchableVectorStore(Protocol):
    """The small vector-store surface needed for independently filtered search."""

    def similarity_search(
        self,
        query: str,
        k: int,
        filter: dict[str, str],
    ) -> list[Document]: ...


def retrieve_paper_contexts(
    vector_store: SearchableVectorStore,
    question: str,
    pdf_names: Sequence[str],
) -> dict[str, list[Document]]:
    """Run isolated filename-filtered retrieval for each of two selected PDFs."""
    if len(pdf_names) != 2 or pdf_names[0] == pdf_names[1]:
        raise ValueError("Compare Mode requires two distinct PDF filenames")

    contexts: dict[str, list[Document]] = {}
    for pdf_name in pdf_names:
        # A separate filtered search prevents passages from one PDF entering
        # the other paper's evidence, even when both discuss the same topic.
        contexts[pdf_name] = vector_store.similarity_search(
            question,
            k=TOP_K,
            filter={"filename": pdf_name},
        )
    return contexts


def build_compare_prompt(
    question: str,
    contexts: Mapping[str, Sequence[Document]],
) -> str:
    """Build one evidence-separated prompt with citations for both papers."""
    if len(contexts) != 2:
        raise ValueError("Compare Mode requires context from exactly two PDFs")

    paper_sections = []
    for paper_number, (pdf_name, chunks) in enumerate(contexts.items(), start=1):
        if chunks:
            evidence = "\n\n".join(
                f"Source: {citation(chunk.metadata)}\n{chunk.page_content}"
                for chunk in chunks
            )
        else:
            # Keep each paper's empty result explicit, so the model cannot
            # silently borrow evidence from the other paper.
            evidence = "[No passages were retrieved from this paper.]"
        paper_sections.append(
            f"Paper {paper_number}: {pdf_name}\n"
            f"Evidence for this paper only:\n{evidence}"
        )

    return (
        "Answer the user's question using only the supplied evidence. Treat each "
        "paper's evidence as isolated: never use one paper to answer for the "
        "other. For each paper, either answer with citations in the exact "
        "'filename, p. X' form shown in its evidence, or explicitly state that "
        "the paper does not cover the topic. Do not invent claims or citations.\n\n"
        f"Question: {question}\n\n"
        f"{paper_sections[0]}\n\n"
        f"{paper_sections[1]}\n\n"
        "Use this format:\n"
        f"## {list(contexts)[0]}\n"
        "Answer: ...\n"
        f"## {list(contexts)[1]}\n"
        "Answer: ...\n"
        "## Difference\n"
        "Summarize how the papers' approaches or answers differ in exactly "
        "two sentences. Base the comparison only on the evidence above."
    )


def generate_comparison(
    question: str,
    contexts: Mapping[str, Sequence[Document]],
    ask: Callable[[str], str] | None = None,
) -> str:
    """Generate the two-paper comparison with exactly one prompt/call."""
    prompt = build_compare_prompt(question, contexts)
    if ask is None:
        # Reuse the app's transient-error/rate-limit retry behavior without
        # creating another client or issuing separate per-paper LLM requests.
        from evaluator import call_llm

        answer = call_llm(prompt, temperature=0.2, label="compare")
    else:
        # Injected test callbacks verify the full prompt without network access.
        answer = ask(prompt)
    if not answer or not answer.strip():
        raise RuntimeError("Comparison generation returned an empty response")
    return answer.strip()
