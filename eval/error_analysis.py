"""Phase 2 error analysis: where did the correct passage go? (zero LLM calls)

For every answerable question, records the 1-based rank of the gold page in the
dense, BM25 and final (live hybrid RRF) rankings, looking 20 deep, and assigns
one class from the FINAL list:

  HIT_TOP3        gold page is in the final top 3
  PAGE_SPLIT      gold page is not in the top 3, but a neighbouring page (±1)
                  of the same file is - the system found the right spot and
                  the page boundary cost the hit
  RANKED_4_TO_20  gold page is in the final list at rank 4-20
  NOT_RETRIEVED   gold page is not in the final top 20

Diagnostics added per question:
  - in_dense_top20 / in_bm25_top20: which retriever could have supplied it
    (the live fusion only takes 10 candidates from each)
  - evidence_in_one_chunk: False when the gold quote is cut by a chunk boundary
  - evidence_token_start: token offset of the quote inside its chunk; a value
    >= 256 means MiniLM never embedded it (dense search is blind to it)
  - rerank flips: rank under hybrid + cross-encoder (top 10) vs hybrid
  - legacy set only: label_ok = does the labelled page contain an expected
    keyword? A miss on a page that contains none suggests a wrong label.

Usage: python eval/error_analysis.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as H  # noqa: E402  (also installs the zero-LLM guard)

TOP_N = 20
OUT = H.EVAL_DIR / "error_analysis.json"
SETS = {
    "heldout_v2": H.EVAL_DIR / "heldout_v2_questions.json",
    "legacy_dev": H.ROOT / "eval_questions.json",
}
EMBED_WINDOW = 256


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_rank(ranking, chunks, question, pages) -> int | None:
    for rank, index in enumerate(ranking[:TOP_N], start=1):
        meta = chunks[index].metadata
        label = H.page_number(meta)
        if (H.source_filename(meta) == question["file"] and label.isdecimal()
                and int(label) in pages):
            return rank
    return None


def classify(final_rank: int | None, neighbour_top3: bool) -> str:
    if final_rank is not None and final_rank <= 3:
        return "HIT_TOP3"
    if neighbour_top3:
        return "PAGE_SPLIT"
    if final_rank is not None:
        return "RANKED_4_TO_20"
    return "NOT_RETRIEVED"


def evidence_diagnostics(question, chunks, tokenizer) -> dict:
    """Is the gold quote inside one chunk, and is it inside MiniLM's window?"""
    quotes = question.get("evidence") or []
    if not quotes:
        return {}
    quote, page = collapse(quotes[0]), question["page"][0]
    for chunk in chunks:
        meta = chunk.metadata
        if H.source_filename(meta) != question["file"] or H.page_number(meta) != str(page):
            continue
        text = collapse(chunk.page_content)
        position = text.find(quote)
        if position >= 0:
            before = tokenizer(text[:position], add_special_tokens=False)["input_ids"]
            start = len(before) + 1  # +1 for the [CLS] token
            return {"evidence_in_one_chunk": True, "evidence_token_start": start,
                    "evidence_beyond_embedding_window": start >= EMBED_WINDOW}
    return {"evidence_in_one_chunk": False}


def legacy_label_ok(question, raw, pages_text) -> bool | None:
    keywords = raw.get("expected_keywords") or []
    text = pages_text.get((question["file"], question["page"][0]), "").lower()
    if not keywords or not text:
        return None
    return any(k.lower() in text for k in keywords)


def main() -> None:
    from sentence_transformers import SentenceTransformer

    settings = H.IndexSettings()
    files = H.corpus_files(None)
    pages = H.load_pages(files)
    pages_text = {(p.metadata["filename"], int(p.metadata["page_label"])): p.page_content
                  for p in pages}
    chunks = H.make_chunks(pages, settings)
    store, fingerprint = H.get_store(chunks, files, settings)
    retriever = H.Retriever(store, chunks)
    tokenizer = SentenceTransformer(settings.embedding_model).tokenizer

    dense_cfg = H.RetrievalConfig("dense", "dense")
    bm25_cfg = H.RetrievalConfig("bm25", "bm25")
    final_cfg = H.RetrievalConfig("hybrid_rrf60", "hybrid")
    rerank_cfg = H.RetrievalConfig("hybrid_rerank_top10", "hybrid_rerank", rerank_pool=10)

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "top_n": TOP_N, "index": fingerprint, "llm_calls": 0, "sets": {}}

    for set_name, path in SETS.items():
        raw_by_id = {q["id"]: q for q in json.loads(path.read_text(encoding="utf-8"))}
        questions = [q for q in H.load_questions(path) if q.get("file") and q.get("page")]
        rows = []
        for q in questions:
            gold = set(q["page"])
            near = {p + d for p in gold for d in (-1, 1)} - gold
            dense = retriever.ranking(q["q"], dense_cfg)
            lexical = retriever.ranking(q["q"], bm25_cfg)
            final = retriever.ranking(q["q"], final_cfg)
            reranked = retriever.ranking(q["q"], rerank_cfg)
            final_rank = first_rank(final, chunks, q, gold)
            row = {
                "id": q["id"], "type": q["type"], "file": q["file"], "page": q["page"],
                "rank_dense": first_rank(dense, chunks, q, gold),
                "rank_bm25": first_rank(lexical, chunks, q, gold),
                "rank_final": final_rank,
                "rank_reranked": first_rank(reranked, chunks, q, gold),
                "class": classify(final_rank, first_rank(final[:3], chunks, q, near) is not None),
                **evidence_diagnostics(q, chunks, tokenizer),
            }
            row["in_dense_top20"] = row["rank_dense"] is not None
            row["in_bm25_top20"] = row["rank_bm25"] is not None
            if set_name == "legacy_dev":
                row["label_ok"] = legacy_label_ok(q, raw_by_id[q["id"]], pages_text)
            rows.append(row)

        misses = [r for r in rows if r["class"] != "HIT_TOP3"]
        dense_blind = [r for r in rows if r.get("evidence_beyond_embedding_window")]
        rerank_gain = [r["id"] for r in rows if r["class"] != "HIT_TOP3"
                       and r["rank_reranked"] is not None and r["rank_reranked"] <= 3]
        rerank_loss = [r["id"] for r in rows if r["class"] == "HIT_TOP3"
                       and (r["rank_reranked"] is None or r["rank_reranked"] > 3)]
        summary = {
            "n": len(rows),
            "classes": dict(Counter(r["class"] for r in rows)),
            "miss_classes_by_type": {
                t: dict(Counter(r["class"] for r in misses if r["type"] == t))
                for t in sorted({r["type"] for r in misses})
            },
            "misses_in_dense_top20": sum(r["in_dense_top20"] for r in misses),
            "misses_in_bm25_top20": sum(r["in_bm25_top20"] for r in misses),
            "misses_in_neither_top20": sum(not r["in_dense_top20"] and not r["in_bm25_top20"]
                                           for r in misses),
            "evidence_split_across_chunks": sum(r.get("evidence_in_one_chunk") is False for r in rows),
            "evidence_beyond_embedding_window": {
                "all": len(dense_blind),
                "dense_rank_when_blind": [r["rank_dense"] for r in dense_blind],
                "dense_top3_when_blind": sum((r["rank_dense"] or 99) <= 3 for r in dense_blind),
                "dense_top3_when_visible": sum((r["rank_dense"] or 99) <= 3 for r in rows
                                               if r.get("evidence_beyond_embedding_window") is False),
                "n_visible": sum(r.get("evidence_beyond_embedding_window") is False for r in rows),
            },
            "rerank_top10_vs_live": {"miss_to_hit": rerank_gain, "hit_to_miss": rerank_loss},
        }
        if set_name == "legacy_dev":
            summary["label_suspect"] = [r["id"] for r in rows if r.get("label_ok") is False]
            summary["misses_with_suspect_label"] = [r["id"] for r in misses if r.get("label_ok") is False]
            summary["rerank_losses_with_suspect_label"] = [
                i for i in rerank_loss if next(r for r in rows if r["id"] == i).get("label_ok") is False
            ]
        report["sets"][set_name] = {"summary": summary, "questions": rows}

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for set_name, block in report["sets"].items():
        s = block["summary"]
        print(f"\n=== {set_name} ({s['n']} answerable) ===")
        print("classes:", s["classes"])
        print("miss classes by type:", s["miss_classes_by_type"])
        print(f"misses with gold in dense top-20: {s['misses_in_dense_top20']}, "
              f"in BM25 top-20: {s['misses_in_bm25_top20']}, in neither: {s['misses_in_neither_top20']}")
        b = s["evidence_beyond_embedding_window"]
        print(f"gold quote beyond MiniLM's 256 tokens: {b['all']} questions; dense top-3 for those: "
              f"{b['dense_top3_when_blind']}/{b['all']} vs {b['dense_top3_when_visible']}/{b['n_visible']} when visible")
        print("gold quote split across two chunks:", s["evidence_split_across_chunks"])
        print("rerank top10 vs live:", s["rerank_top10_vs_live"])
        if "label_suspect" in s:
            print("legacy labels with no expected keyword on the labelled page:", s["label_suspect"])
            print("  ...of which live misses:", s["misses_with_suspect_label"])
            print("  ...of which rerank losses:", s["rerank_losses_with_suspect_label"])
        print("\nmisses:")
        for r in block["questions"]:
            if r["class"] != "HIT_TOP3":
                extra = f" label_ok={r['label_ok']}" if "label_ok" in r else ""
                print(f"  {r['id']:5} {r['type']:20} {r['class']:15} dense={r['rank_dense']} "
                      f"bm25={r['rank_bm25']} final={r['rank_final']} rerank={r['rank_reranked']} "
                      f"tok={r.get('evidence_token_start')}{extra}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
