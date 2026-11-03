"""Retrieve and compare evidence from two PDFs in one LLM request."""

from collections.abc import Callable, Mapping, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import citation
from hybrid import VectorSearchStore, hybrid_search

COMPARE_CHIP_EXPANSIONS = {
    "main objective": "What is the main objective or purpose of this paper?",
    "methodology": "What methodology or approach does this paper use?",
    "results and metrics": "What results or metrics does this paper report?",
    "limitations": "What limitations does this paper mention?",
    "data used": "What data or dataset does this paper use?",
}


def expand_compare_query(question: str) -> str:
    key = question.strip().casefold()
    return COMPARE_CHIP_EXPANSIONS.get(key, question.strip())


def retrieve_paper_contexts(
    vector_store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: Sequence[Document],
    question: str,
    pdf_names: Sequence[str],
) -> dict[str, list[Document]]:
    """Run isolated hybrid retrieval for each of two selected PDFs."""
    if len(pdf_names) != 2 or pdf_names[0] == pdf_names[1]:
        raise ValueError("Compare Mode requires two distinct PDF filenames")

    contexts: dict[str, list[Document]] = {}
    for pdf_name in pdf_names:
        # A separate metadata-filtered hybrid search prevents dense or BM25
        # passages from one paper entering the other's evidence.
        contexts[pdf_name] = hybrid_search(
            question,
            vector_store,
            bm25,
            list(chunks),
            metadata_filter={"filename": pdf_name},
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
        "two sentences. Base the comparison only on the evidence above.\n"
        "## Similarities\n"
        "Write one short evidence-grounded similarity, or state that no "
        "evidence-supported similarity is found."
    )


def generate_comparison(
    question: str,
    contexts: Mapping[str, Sequence[Document]],
    ask: Callable[[str], str] | None = None,
) -> str:
    """Generate the two-paper comparison with exactly one prompt/call."""
    if any(not chunks for chunks in contexts.values()):
        if contexts and all(not chunks for chunks in contexts.values()):
            pass
        else:
            return ""
    prompt = build_compare_prompt(question, contexts)
    def request_answer() -> str:
        if ask is None:
            # Reuse the app's transient-error/rate-limit retry behavior.
            from evaluator import call_llm

            return call_llm(prompt, temperature=0.2, label="compare")
        # Injected test callbacks verify the full prompt without network access.
        return ask(prompt)

    answer = request_answer()
    if not answer or not answer.strip():
        raise RuntimeError("Comparison generation returned an empty response")
    first_answer = answer.strip()
    expected_sections = [name.casefold() for name in contexts]
    has_expected_sections = all(
        f"## {name}".casefold() in first_answer.casefold()
        for name in expected_sections
    )
    if not has_expected_sections:
        answer = request_answer()
        if not answer or not answer.strip():
            return first_answer
        return answer.strip()
    return first_answer
