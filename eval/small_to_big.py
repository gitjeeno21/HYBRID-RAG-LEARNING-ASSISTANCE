"""Small-to-big retrieval ablation (zero LLM calls).

PROBLEM (eval/ERROR_ANALYSIS.md): all-MiniLM-L6-v2 embeds only the first 256
tokens (~1000 characters) of each 1500-character chunk. In 12 of 47 held-out
questions the gold quote starts beyond that window, so dense search never saw
it. Smaller chunks would fit the window but would give the model less context.

IDEA (small-to-big / parent-child retrieval): search small CHILD chunks that
fit the embedding window, but return their 1500-character PARENT chunk. The
model receives exactly the same unit of context as today; only how parents
are ranked by dense search changes.

PRE-REGISTERED DESIGN (fixed before measuring; not tuned on either set):
  children   each parent chunk split with RecursiveCharacterTextSplitter
             (400 characters, 50 overlap), so every child has one parent
  dense_s2b  dense search over children; a parent ranks by its best child
  hybrid_s2b RRF (k = 60) of the dense_s2b parent ranking and the unchanged
             BM25 parent ranking, 10 candidates each (live settings)
  rule       adopt only if hybrid_s2b does not lose on dev (hit@5 and nDCG@5)
             AND improves the held-out test set (Phase 0 adoption rule)

Usage: python eval/small_to_big.py
Output: eval/SMALL_TO_BIG.md, eval/small_to_big_results.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as H  # noqa: E402  (installs the zero-LLM guard)

CHILD_SIZE = 400
CHILD_OVERLAP = 50
CHILD_POOL = 200          # child hits fetched, enough to rank POOL distinct parents
TOP_K = 5
SETS = {"dev": H.EVAL_DIR / "dev_questions_multipage.json",
        "test": H.EVAL_DIR / "heldout_v2_questions.json"}
OUT_MD = H.EVAL_DIR / "SMALL_TO_BIG.md"
OUT_JSON = H.EVAL_DIR / "small_to_big_results.json"


def make_children(parents: list) -> list:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP)
    children = []
    for index, parent in enumerate(parents):
        for piece in splitter.split_text(parent.page_content):
            children.append(Document(page_content=piece,
                                     metadata={**parent.metadata, "parent": index}))
    return children


def child_store(children: list, parent_fingerprint: str):
    """A separate eval index for the children, keyed by parent index + child settings."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    digest = hashlib.sha1(f"{parent_fingerprint}:{CHILD_SIZE}:{CHILD_OVERLAP}".encode()).hexdigest()[:12]
    path = H.INDEX_ROOT / f"children_{digest}"
    embeddings = HuggingFaceEmbeddings(model=H.IndexSettings().embedding_model,
                                       encode_kwargs={"normalize_embeddings": True})
    store = Chroma(collection_name="eval_children", embedding_function=embeddings,
                   persist_directory=str(path))
    if store._collection.count() != len(children):
        if store._collection.count():
            store.delete_collection()
            store = Chroma(collection_name="eval_children", embedding_function=embeddings,
                           persist_directory=str(path))
        print(f"Building child index {digest} ({len(children)} children)...", flush=True)
        # Chroma caps a single add; insert in batches.
        for start in range(0, len(children), 1000):
            store.add_documents(children[start:start + 1000])
    return store


class SmallToBigRetriever(H.Retriever):
    """The harness retriever plus two parent-child methods."""

    def __init__(self, store, chunks: list, children_store):
        super().__init__(store, chunks)
        self.children = children_store
        self._parent_cache: dict[str, list[int]] = {}

    def dense_parents(self, query: str) -> list[int]:
        if query not in self._parent_cache:
            hits = self.children.similarity_search(query, k=CHILD_POOL)
            order, seen = [], set()
            for hit in hits:
                parent = int(hit.metadata["parent"])
                if parent not in seen:
                    seen.add(parent)
                    order.append(parent)
            self._parent_cache[query] = order
        return self._parent_cache[query]

    def ranking(self, query: str, config: H.RetrievalConfig) -> list[int]:
        if config.method == "dense_s2b":
            return self.dense_parents(query)
        if config.method == "hybrid_s2b":
            _, lexical = self.base(query)
            n = config.candidates_per_method
            return H.reciprocal_rank_fusion([self.dense_parents(query)[:n], lexical[:n]], k=config.rrf_k)
        return super().ranking(query, config)

    def timed_run(self, query: str, config: H.RetrievalConfig) -> dict[str, float]:
        if config.method not in {"dense_s2b", "hybrid_s2b"}:
            return super().timed_run(query, config)
        clock, timings = time.perf_counter, {}
        started = clock()
        hits = self.children.similarity_search(query, k=CHILD_POOL)
        parents = list(dict.fromkeys(int(hit.metadata["parent"]) for hit in hits))
        timings["dense_ms"] = (clock() - started) * 1e3
        if config.method == "hybrid_s2b":
            started = clock()
            lexical = H.bm25_ranking(self.bm25, query, config.candidates_per_method)
            timings["bm25_ms"] = (clock() - started) * 1e3
            started = clock()
            H.reciprocal_rank_fusion([parents[:config.candidates_per_method], lexical], k=config.rrf_k)
            timings["fusion_ms"] = (clock() - started) * 1e3
        timings["total_ms"] = sum(timings.values())
        return timings


def evidence_in_context(questions: list[dict], retriever, config) -> tuple[int, int]:
    """Answerable questions whose every evidence quote is inside the top-k parents."""
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().lower()  # noqa: E731
    covered = total = 0
    for q in questions:
        if not q.get("file") or not q.get("evidence"):
            continue
        total += 1
        context = normalize(" ".join(retriever.chunks[i].page_content
                                     for i in retriever.ranking(q["q"], config)[:config.top_k]))
        covered += all(normalize(quote) in context for quote in q["evidence"])
    return covered, total


def main() -> None:
    files = H.corpus_files(None)
    settings = H.IndexSettings()
    parents = H.make_chunks(H.load_pages(files), settings)
    store, fingerprint = H.get_store(parents, files, settings)
    children = make_children(parents)
    retriever = SmallToBigRetriever(store, parents, child_store(children, fingerprint))
    configs = [H.RetrievalConfig(name, method, top_k=TOP_K) for name, method in (
        ("dense", "dense"), ("dense_s2b", "dense_s2b"),
        ("hybrid_rrf60", "hybrid"), ("hybrid_s2b", "hybrid_s2b"))]

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_commit": H.git_commit(), "parents": len(parents), "children": len(children),
               "child_size": CHILD_SIZE, "child_overlap": CHILD_OVERLAP, "top_k": TOP_K, "sets": {}}
    sections = []
    for name, path in SETS.items():
        questions = H.load_questions(path)
        names = {f.name for f in files}
        questions = [q for q in questions if q.get("file") is None or q["file"] in names]
        result = H.evaluate(questions, retriever, configs)
        coverage = {c.name: evidence_in_context(questions, retriever, c) for c in configs}
        payload["sets"][name] = {**result, "evidence_in_context": coverage}
        lines = [f"### {name} ({result['n_answerable_scored']} answerable questions)", "",
                 H.format_report(result)]
        if any(total for _, total in coverage.values()):
            lines += ["", "Every gold evidence quote inside the top-5 passages sent to the model:", "",
                      "| config | questions |", "|---|---|"]
            lines += [f"| {c} | {k}/{n} |" for c, (k, n) in coverage.items()]
        sections.append("\n".join(lines))
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    header = (f"# Small-to-big retrieval ablation\n\nGenerated by `python eval/small_to_big.py` "
              f"(zero LLM calls) at commit {payload['git_commit']}. {len(parents)} parent chunks, "
              f"{len(children)} children ({CHILD_SIZE} chars, {CHILD_OVERLAP} overlap). "
              "The design and adoption rule are fixed in the script's docstring.\n\n")
    OUT_MD.write_text(header + "\n\n".join(sections) + "\n", encoding="utf-8")
    print("\n\n".join(sections))
    print(f"\nSaved {OUT_MD}")


if __name__ == "__main__":
    main()
