"""Diagnose dense and hybrid retrieval for held-out questions q35, q38-q40.

This is a read-only diagnostic: it opens the existing 1500/50 Chroma
collection directly, never rebuilds an index, and uses local embeddings only.
"""

import os

# Prevent imports of evaluator.py (via hybrid.py) from reading .env or using a
# real API key. The diagnostic does not call any hosted API.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["GROQ_API_KEY"] = "retrieval-only-disabled"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import re
from pathlib import Path

from langchain_chroma import Chroma

from config import BASE_DIR, page_number, source_filename
from evaluator import build_chunks, get_embeddings
from hybrid import (
    CANDIDATES_PER_METHOD,
    DENSE_CHUNK_OVERLAP,
    DENSE_CHUNK_SIZE,
    RRF_K,
    bm25_ranking,
    build_bm25,
    hybrid_search,
)
from ingest import load_documents

QUESTION_IDS = ("q35", "q38", "q39", "q40")
INDEX_PATH = BASE_DIR / "chroma_db_eval_1500_50"
COLLECTION_NAME = "eval_1500_50"
OUTPUT_PATH = BASE_DIR / "diagnose_misses_output.md"
SCRATCH_PATH = BASE_DIR / "scratch_step4.py"


def expected_page_match(doc: object, question: dict) -> bool:
    metadata = doc.metadata
    return (
        source_filename(metadata) == question["expected_source"]
        and page_number(metadata) == str(question["source_page"])
    )


def keyword_matches(doc: object, keywords: list[str]) -> list[str]:
    text = doc.page_content.lower()
    return [keyword for keyword in keywords if keyword.lower() in text]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:250].replace("|", r"\|")


def source_page_rank(docs: list, question: dict) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if expected_page_match(doc, question):
            return rank
    return None


def rank_text(rank: int | None) -> str:
    return str(rank) if rank is not None else "not in top 10"


def classify(docs: list, question: dict) -> tuple[str, str]:
    top_three = docs[:3]
    right_page_in_top_three = any(
        expected_page_match(doc, question) for doc in top_three
    )
    expected_keyword_in_top_three = any(
        expected_page_match(doc, question)
        and keyword_matches(doc, question["expected_keywords"])
        for doc in top_three
    )
    if expected_keyword_in_top_three:
        return "HIT", "The top three include the expected page and at least one expected keyword."
    if right_page_in_top_three:
        return (
            "KEYWORD ISSUE",
            "The expected source and page are in the top three, but none of the expected keywords appears there.",
        )
    rank = source_page_rank(docs, question)
    if rank is not None:
        return (
            "RANKING FAILURE",
            f"The expected source and page rank {rank} in the top 10 but fall below the top three.",
        )
    return (
        "RETRIEVAL FAILURE",
        "No chunk from the expected source and page appears in the top 10.",
    )


def render_top_three(method: str, docs: list, question: dict) -> list[str]:
    lines = [f"**{method} top 3**"]
    for rank, doc in enumerate(docs[:3], start=1):
        keywords = keyword_matches(doc, question["expected_keywords"])
        shown_keywords = ", ".join(keywords) if keywords else "none"
        lines.append(
            f"{rank}. `{source_filename(doc.metadata)}`, p. {page_number(doc.metadata)}"
            f" — keywords: {shown_keywords} — `{compact_text(doc.page_content)}`"
        )
    return lines


def run_diagnostic() -> str:
    questions = json.loads(
        (BASE_DIR / "eval_questions.json").read_text(encoding="utf-8")
    )
    by_id = {question["id"]: question for question in questions}
    missing = [question_id for question_id in QUESTION_IDS if question_id not in by_id]
    if missing:
        raise ValueError(f"Required questions are missing: {', '.join(missing)}")
    selected = [by_id[question_id] for question_id in QUESTION_IDS]

    if not INDEX_PATH.is_dir():
        raise FileNotFoundError(f"Existing index not found: {INDEX_PATH}")
    documents = load_documents()
    chunks = build_chunks(documents, DENSE_CHUNK_SIZE, DENSE_CHUNK_OVERLAP)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_PATH),
    )
    indexed_count = store._collection.count()
    if indexed_count != len(chunks) or indexed_count != 281:
        raise RuntimeError(
            "Refusing to diagnose against a mismatched index: "
            f"index={indexed_count}, local 1500/50 chunks={len(chunks)}, expected=281"
        )
    if (
        DENSE_CHUNK_SIZE != 1500
        or DENSE_CHUNK_OVERLAP != 50
        or RRF_K != 60
        or CANDIDATES_PER_METHOD != 10
    ):
        raise RuntimeError("The fixed hybrid retrieval settings do not match the request.")

    bm25 = build_bm25(chunks)
    scratch_lines = (
        SCRATCH_PATH.read_text(encoding="utf-8").splitlines()[:10]
        if SCRATCH_PATH.exists()
        else []
    )
    output = [
        "# Retrieval miss diagnostic: q35, q38, q39, q40",
        "",
        "Read-only retrieval using the existing 1500/50 index (281 chunks), "
        "local embeddings, BM25 over those same chunks, RRF k=60, 10 candidates "
        "per method, and top 3. No questions or retrieval settings were changed; "
        "no hosted API was called.",
        "",
    ]
    method_classifications: dict[str, dict[str, tuple[str, str]]] = {}
    rank_rows = []

    for question in selected:
        dense_docs = store.similarity_search(
            question["question"], k=CANDIDATES_PER_METHOD
        )
        lexical_indices = bm25_ranking(
            bm25, question["question"], CANDIDATES_PER_METHOD
        )
        bm25_docs = [chunks[index] for index in lexical_indices]
        # hybrid_search applies the existing 10+10 candidate RRF and returns
        # its fused order; requesting 10 exposes the full candidate ranking.
        hybrid_docs = hybrid_search(
            question["question"],
            store,
            bm25,
            chunks,
            n_results=CANDIDATES_PER_METHOD,
        )

        output.extend(
            [
                f"## {question['id']}",
                "",
                f"**Question:** {question['question']}",
                f"**Expected source/page:** `{question['expected_source']}`, p. {question['source_page']}",
                f"**Expected keywords:** {', '.join(question['expected_keywords'])}",
                "",
            ]
        )
        output.extend(render_top_three("Dense", dense_docs, question))
        output.append("")
        output.extend(render_top_three("Hybrid", hybrid_docs, question))
        output.append("")

        classifications = {}
        for method, docs in (
            ("Dense", dense_docs),
            ("BM25", bm25_docs),
            ("Hybrid", hybrid_docs),
        ):
            rank = source_page_rank(docs, question)
            label, reason = classify(docs, question)
            classifications[method] = (label, reason)
            output.append(
                f"- **{method}:** first expected source/page chunk rank "
                f"{rank_text(rank)}; {label} — {reason}"
            )
        method_classifications[question["id"]] = classifications
        ranks = {
            method: rank_text(source_page_rank(docs, question))
            for method, docs in (
                ("Dense", dense_docs),
                ("BM25", bm25_docs),
                ("Hybrid", hybrid_docs),
            )
        }
        rank_rows.append((question, classifications, ranks))
        output.append("")

    output.extend(
        [
            "## Summary table",
            "",
            "| Question | Dense label | Hybrid label | Right chunk rank (dense / BM25 / hybrid) |",
            "|---|---|---|---|",
        ]
    )
    for question, classifications, ranks in rank_rows:
        output.append(
            f"| {question['id']} | {classifications['Dense'][0]} "
            f"| {classifications['Hybrid'][0]} "
            f"| {ranks['Dense']} / {ranks['BM25']} / {ranks['Hybrid']} |"
        )

    failure_labels = ("RANKING FAILURE", "RETRIEVAL FAILURE", "KEYWORD ISSUE")
    missed_method_classifications = [
        label
        for classifications in method_classifications.values()
        for method in ("Dense", "Hybrid")
        for label, _ in [classifications[method]]
        if label in failure_labels
    ]
    counts = {
        label: missed_method_classifications.count(label)
        for label in failure_labels
    }
    output.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Across the dense and hybrid method-level misses, {counts['RANKING FAILURE']} are ranking failures, so a re-ranker could help in those cases.",
            f"{counts['RETRIEVAL FAILURE']} {'is a retrieval failure' if counts['RETRIEVAL FAILURE'] == 1 else 'are retrieval failures'}, where the expected source/page chunk is absent from the top 10, pointing to chunking or embedding retrieval rather than ranking.",
            f"{counts['KEYWORD ISSUE']} {'is a keyword issue' if counts['KEYWORD ISSUE'] == 1 else 'are keyword issues'}, where the expected page is present in the top three but the evaluator's expected keywords are absent.",
            "",
            "## scratch_step4.py: first 10 lines",
            "",
        ]
    )
    if scratch_lines:
        output.extend(["```python", *scratch_lines, "```", ""])
        output.append(
            "This is a scratch helper that prints judging summaries from "
            "`eval_results.json`; it is not application code."
        )
    else:
        output.append("The scratch script was not present.")
    output.append("")
    return "\n".join(output)


def main() -> None:
    report = run_diagnostic()
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nSaved diagnostic to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
