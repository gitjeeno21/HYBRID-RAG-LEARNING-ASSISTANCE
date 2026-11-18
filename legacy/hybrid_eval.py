"""The original dense-vs-hybrid experiment (q1-q30, keyword hit, top 3).

Moved out of hybrid.py so the live retrieval module contains only live code.
Superseded by eval/run_eval.py (page-level gold labels, more metrics, CIs).
Kept so legacy/results/hybrid_results.json stays reproducible:

    python -m legacy.hybrid_eval
"""

import json
from datetime import datetime, timezone

from config import BASE_DIR, LEGACY_EVAL_TOP_K, source_filename
from hybrid import CANDIDATES_PER_METHOD, RRF_K, build_bm25, hybrid_search

QUESTIONS_PATH = BASE_DIR / "eval_questions.json"
RESULTS_PATH = BASE_DIR / "legacy" / "results" / "hybrid_results.json"
DENSE_CHUNK_SIZE = 1500
DENSE_CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def docs_hit(docs: list, question: dict) -> bool:
    """True if one of `docs` is from the expected PDF and contains a keyword.

    This mirrors evaluator.question_is_hit exactly (top-k, right PDF, any
    expected keyword). It is duplicated rather than imported because that
    function queries a Chroma store, while hybrid results are already a list of
    Documents. evaluate() cross-checks the two definitions on the dense side so
    a drift would be visible, not silent.
    """
    for doc in docs:
        if source_filename(doc.metadata) != question["expected_source"]:
            continue
        text = doc.page_content.lower()
        if any(kw.lower() in text for kw in question["expected_keywords"]):
            return True
    return False


def split_question_groups(questions: list) -> dict:
    """Split into the original set (q01-q20) and the held-out set (q21-q30).

    should_refuse questions (t01-t06) are excluded: hit rate is undefined for a
    question that has no expected source or keywords.
    """
    groups = {"q1-q20": [], "q21-q30": []}
    for q in questions:
        qid = q.get("id", "")
        if not qid.startswith("q") or not qid[1:].isdigit():
            continue
        number = int(qid[1:])
        if 1 <= number <= 20:
            groups["q1-q20"].append(q)
        elif 21 <= number <= 30:
            groups["q21-q30"].append(q)
    return groups


def evaluate() -> dict:
    """Measure dense-only and hybrid hit rate on both question groups.

    Runs once, writes hybrid_results.json, and prints the comparison. No LLM
    calls: every number here is deterministic retrieval.
    """
    # Keep evaluation dependencies lazy: importing hybrid search from the
    # Streamlit app should not construct evaluator.py's Groq client.
    from legacy.evaluator import build_chunks, get_vector_store, question_is_hit
    from ingest import load_documents

    documents = load_documents()
    # Reuses the persisted 1500/50 index when present, so this does not
    # re-embed the corpus.
    store = get_vector_store(documents, DENSE_CHUNK_SIZE, DENSE_CHUNK_OVERLAP)
    chunks = build_chunks(documents, DENSE_CHUNK_SIZE, DENSE_CHUNK_OVERLAP)

    # BM25 must see the same chunks as the dense index; if these differ, the
    # comparison would be between two different corpora rather than two
    # retrieval methods.
    indexed = store._collection.count()
    if indexed != len(chunks):
        print(f"WARNING: index holds {indexed} chunks but the splitter produced "
              f"{len(chunks)}; fusion may drop chunks.")

    bm25 = build_bm25(chunks)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    groups = split_question_groups(questions)

    results = {}
    per_question = {}
    drift = 0
    for group_name, group_questions in groups.items():
        block = {
            "dense": {"hits": 0, "total": len(group_questions), "missed": []},
            "hybrid": {"hits": 0, "total": len(group_questions), "missed": []},
        }
        for q in group_questions:
            # Existing dense path, unchanged.
            dense_hit = question_is_hit(store, q)
            dense_docs = store.similarity_search(q["question"], k=LEGACY_EVAL_TOP_K)
            # Guard against the two hit definitions drifting apart.
            if docs_hit(dense_docs, q) != dense_hit:
                drift += 1
                dense_hit = docs_hit(dense_docs, q)

            hybrid_docs = hybrid_search(q["question"], store, bm25, chunks,
                                        n_results=LEGACY_EVAL_TOP_K)
            hybrid_hit = docs_hit(hybrid_docs, q)

            per_question[q["id"]] = {"dense": dense_hit, "hybrid": hybrid_hit}
            for method, hit in (("dense", dense_hit), ("hybrid", hybrid_hit)):
                if hit:
                    block[method]["hits"] += 1
                else:
                    block[method]["missed"].append(q["id"])
        for method in ("dense", "hybrid"):
            total = block[method]["total"]
            block[method]["hit_rate"] = (
                round(block[method]["hits"] / total, 4) if total else None
            )
        results[group_name] = block

    # Which questions flipped between the two methods, per group. This is the
    # substance of the comparison: a net rate can hide one gain and one loss.
    changed = {}
    for group_name, group_questions in groups.items():
        gains = [q["id"] for q in group_questions
                 if per_question[q["id"]]["hybrid"] and not per_question[q["id"]]["dense"]]
        losses = [q["id"] for q in group_questions
                  if per_question[q["id"]]["dense"] and not per_question[q["id"]]["hybrid"]]
        changed[group_name] = {"miss_to_hit": gains, "hit_to_miss": losses}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "dense_chunk_size": DENSE_CHUNK_SIZE,
            "dense_chunk_overlap": DENSE_CHUNK_OVERLAP,
            "rrf_k": RRF_K,
            "candidates_per_method": CANDIDATES_PER_METHOD,
            "top_k": LEGACY_EVAL_TOP_K,
            "retrieval_only": True,
            "fixed_before_measurement": True,
        },
        "question_counts": {name: len(group) for name, group in groups.items()},
        "results": results,
        "per_question": per_question,
        "changed": changed,
        "dense_definition_mismatches": drift,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # --- report ------------------------------------------------------------
    print("\n| method | q1-q20 hit rate | q21-q30 hit rate |")
    print("| --- | ---: | ---: |")
    for method in ("dense", "hybrid"):
        cells = []
        for group_name in ("q1-q20", "q21-q30"):
            block = results[group_name][method]
            cells.append(
                f"{block['hits']}/{block['total']} ({block['hit_rate']:.0%})"
                if block["hit_rate"] is not None else "-"
            )
        print(f"| {method} | {cells[0]} | {cells[1]} |")

    for group_name in ("q1-q20", "q21-q30"):
        gain = changed[group_name]["miss_to_hit"]
        loss = changed[group_name]["hit_to_miss"]
        print(f"\n{group_name}: dense -> hybrid")
        print(f"  miss -> hit: {', '.join(gain) if gain else 'none'}")
        print(f"  hit -> miss: {', '.join(loss) if loss else 'none'}")

    print(f"\nSaved to {RESULTS_PATH.name}")
    print(f"(dense hit-definition mismatches: {drift})")
    return payload


if __name__ == "__main__":
    evaluate()
