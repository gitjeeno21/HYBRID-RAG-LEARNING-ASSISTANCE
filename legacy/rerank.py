"""Evaluate fixed cross-encoder reranking over the existing local retrieval index."""

import os

# Prevent evaluator.py's dotenv loading and ensure no real API key is used.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["GROQ_API_KEY"] = "retrieval-only-disabled"

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import BASE_DIR, page_number, source_filename
from config import LEGACY_EVAL_TOP_K as TOP_K  # pinned: this experiment ran at k=3
from evaluator import build_chunks, get_embeddings
from hybrid import (
    CANDIDATES_PER_METHOD,
    DENSE_CHUNK_OVERLAP,
    DENSE_CHUNK_SIZE,
    RRF_K,
    build_bm25,
    docs_hit,
    hybrid_search,
)
from ingest import load_documents

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH = BASE_DIR / "chroma_db_eval_1500_50"
COLLECTION_NAME = "eval_1500_50"
QUESTIONS_PATH = BASE_DIR / "eval_questions.json"
RESULTS_PATH = BASE_DIR / "rerank_results.json"
METHODS = ("dense", "hybrid", "dense_rerank", "hybrid_rerank")
SET_NAMES = ("q1-q20", "q21-q30", "q31-q40", "q41-q50")


def rerank_candidates(
    question: str,
    candidates: list[Document],
    cross_encoder: CrossEncoder,
    top_k: int = TOP_K,
) -> list[Document]:
    """Score the provided candidate list and return its highest-scoring items."""
    pairs = [(question, document.page_content) for document in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [candidates[index] for index, _ in ranked[:top_k]]


def question_set(question: dict) -> str:
    number = int(question["id"][1:])
    if 1 <= number <= 20:
        return "q1-q20"
    if 21 <= number <= 30:
        return "q21-q30"
    if 31 <= number <= 40:
        return "q31-q40"
    if 41 <= number <= 50:
        return "q41-q50"
    raise ValueError(f"Question is outside the evaluation sets: {question['id']}")


def page_hit(documents: list[Document], question: dict) -> bool:
    expected_page = question["source_page"]
    for document in documents:
        if source_filename(document.metadata) != question["expected_source"]:
            continue
        retrieved_page = page_number(document.metadata)
        if retrieved_page.isdecimal() and abs(int(retrieved_page) - expected_page) <= 1:
            return True
    return False


def summarize_method(rows: dict[str, dict], method: str) -> dict:
    total = len(rows)
    primary_hits = sum(row[method]["primary"] for row in rows.values())
    page_hits = sum(row[method]["page_hit"] for row in rows.values())
    return {
        "primary_hits": primary_hits,
        "total": total,
        "primary_hit_rate": round(primary_hits / total, 4) if total else None,
        "page_hits": page_hits,
        "page_hit_rate": round(page_hits / total, 4) if total else None,
    }


def flip_lists(
    rows: dict[str, dict],
    original_method: str,
    reranked_method: str,
) -> dict[str, list[str]]:
    miss_to_hit = []
    hit_to_miss = []
    for question_id, row in rows.items():
        original_hit = row[original_method]["primary"]
        reranked_hit = row[reranked_method]["primary"]
        if reranked_hit and not original_hit:
            miss_to_hit.append(question_id)
        elif original_hit and not reranked_hit:
            hit_to_miss.append(question_id)
    return {"miss_to_hit": miss_to_hit, "hit_to_miss": hit_to_miss}


def format_rate(metric: dict, rate_key: str, hits_key: str) -> str:
    rate = metric[rate_key]
    return f"{metric[hits_key]}/{metric['total']} ({rate:.0%})" if rate is not None else "-"


def print_report(payload: dict) -> None:
    print("| Method | Set | Primary hit rate | Secondary page hit rate |")
    print("|---|---|---:|---:|")
    for set_name in SET_NAMES:
        for method in METHODS:
            metric = payload["results"][set_name][method]
            primary = format_rate(metric, "primary_hit_rate", "primary_hits")
            secondary = format_rate(metric, "page_hit_rate", "page_hits")
            print(f"| {method} | {set_name} | {primary} | {secondary} |")

    print("\nq41-q50 primary hit/miss by method")
    print("| Question | Type | Dense | Hybrid | Dense + rerank | Hybrid + rerank |")
    print("|---|---|---|---|---|---|")
    for question in payload["questions"]["q41-q50"]:
        row = payload["per_question"][question["id"]]
        statuses = [
            "hit" if row[method]["primary"] else "miss" for method in METHODS
        ]
        print(
            f"| {question['id']} | {question['type']} | "
            f"{statuses[0]} | {statuses[1]} | {statuses[2]} | {statuses[3]} |"
        )

    print("\nPrimary hit flips vs the corresponding original method")
    for set_name in ("q31-q40", "q41-q50"):
        print(f"\n{set_name}")
        for comparison, flip in payload["flips"][set_name].items():
            gains = ", ".join(flip["miss_to_hit"]) or "none"
            losses = ", ".join(flip["hit_to_miss"]) or "none"
            print(f"- {comparison}: miss -> hit: {gains}; hit -> miss: {losses}")

    new_set = payload["results"]["q41-q50"]
    dense_rerank = new_set["dense_rerank"]["primary_hits"]
    hybrid_rerank = new_set["hybrid_rerank"]["primary_hits"]
    dense = new_set["dense"]["primary_hits"]
    hybrid = new_set["hybrid"]["primary_hits"]
    if dense_rerank == dense and hybrid_rerank == hybrid:
        benefit_sentence = "On q41-q50, reranking ties both original methods: no evidence of benefit."
    else:
        improvements = (dense_rerank - dense, hybrid_rerank - hybrid)
        if max(improvements) > 0:
            benefit_sentence = (
                "On q41-q50, the only clean held-out set, reranking improves "
                f"dense by {improvements[0]} question(s) and hybrid by "
                f"{improvements[1]} question(s)."
            )
        else:
            benefit_sentence = (
                "On q41-q50, the only clean held-out set, reranking does not "
                "improve either original method."
            )
    largest_gain = max(dense_rerank - dense, hybrid_rerank - hybrid)
    size_sentence = (
        "No gain exceeds one question, so no gain is larger than the 10-point "
        "noise of a single question."
        if largest_gain <= 1
        else f"The largest gain is {largest_gain} questions, exceeding one question."
    )
    seen_sentence = (
        "q31-q40 was seen before choosing the re-ranker, so it is not a clean test."
    )
    print(f"\n{benefit_sentence} {size_sentence} {seen_sentence}")


def run_evaluation() -> dict:
    all_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions_by_set = {
        set_name: [
            question
            for question in all_questions
            if question["id"].startswith("q")
            and question_set(question) == set_name
        ]
        for set_name in SET_NAMES
    }
    if [len(questions_by_set[name]) for name in SET_NAMES] != [20, 10, 10, 10]:
        raise ValueError("Expected complete q1-q50 question sets with sizes 20/10/10/10.")
    if not INDEX_PATH.is_dir():
        raise FileNotFoundError(f"Existing 1500/50 index not found: {INDEX_PATH}")
    if (
        DENSE_CHUNK_SIZE != 1500
        or DENSE_CHUNK_OVERLAP != 50
        or RRF_K != 60
        or CANDIDATES_PER_METHOD != 10
        or TOP_K != 3
    ):
        raise RuntimeError("The fixed retrieval settings do not match the request.")

    # Load/download the requested model before retrieval starts. Do not silently
    # fall back to another cross-encoder if this model is unavailable.
    cross_encoder = CrossEncoder(RERANKER_MODEL)

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
            "Refusing to evaluate against a mismatched index: "
            f"index={indexed_count}, local chunks={len(chunks)}, expected=281"
        )
    bm25 = build_bm25(chunks)

    per_question: dict[str, dict] = {}
    for set_name in SET_NAMES:
        for question in questions_by_set[set_name]:
            query = question["question"]
            dense_candidates = store.similarity_search(
                query, k=CANDIDATES_PER_METHOD
            )
            hybrid_candidates = hybrid_search(
                query,
                store,
                bm25,
                chunks,
                n_results=CANDIDATES_PER_METHOD,
            )
            retrieved = {
                "dense": dense_candidates[:TOP_K],
                "hybrid": hybrid_candidates[:TOP_K],
                "dense_rerank": rerank_candidates(
                    query, dense_candidates, cross_encoder
                ),
                "hybrid_rerank": rerank_candidates(
                    query, hybrid_candidates, cross_encoder
                ),
            }
            per_question[question["id"]] = {
                method: {
                    "primary": docs_hit(retrieved_docs, question),
                    "page_hit": page_hit(retrieved_docs, question),
                }
                for method, retrieved_docs in retrieved.items()
            }

    results = {
        set_name: {
            method: summarize_method(
                {
                    question["id"]: per_question[question["id"]]
                    for question in questions_by_set[set_name]
                },
                method,
            )
            for method in METHODS
        }
        for set_name in SET_NAMES
    }
    flips = {
        set_name: {
            "dense_rerank vs dense": flip_lists(
                {
                    question["id"]: per_question[question["id"]]
                    for question in questions_by_set[set_name]
                },
                "dense",
                "dense_rerank",
            ),
            "hybrid_rerank vs hybrid": flip_lists(
                {
                    question["id"]: per_question[question["id"]]
                    for question in questions_by_set[set_name]
                },
                "hybrid",
                "hybrid_rerank",
            ),
        }
        for set_name in ("q31-q40", "q41-q50")
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "index": "chroma_db_eval_1500_50",
            "indexed_chunks": indexed_count,
            "dense_chunk_size": DENSE_CHUNK_SIZE,
            "dense_chunk_overlap": DENSE_CHUNK_OVERLAP,
            "rrf_k": RRF_K,
            "candidates_per_method": CANDIDATES_PER_METHOD,
            "top_k": TOP_K,
            "reranker_model": RERANKER_MODEL,
            "retrieval_only": True,
            "local_models_only": True,
        },
        "metric_definitions": {
            "primary": "Top 3 contains a chunk from the expected PDF with any expected keyword.",
            "secondary": "Top 3 contains a chunk from the expected PDF whose page is within 1 of the expected page.",
        },
        "question_counts": {
            set_name: len(questions) for set_name, questions in questions_by_set.items()
        },
        "questions": questions_by_set,
        "results": results,
        "per_question": per_question,
        "flips": flips,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_report(payload)
    print(f"\nSaved to {RESULTS_PATH.name}")
    return payload


if __name__ == "__main__":
    run_evaluation()
