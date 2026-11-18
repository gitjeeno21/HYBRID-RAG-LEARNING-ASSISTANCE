"""Hybrid retrieval: BM25 (lexical) + dense embeddings, merged with RRF.

WHY hybrid at all: the dense half of this project embeds a *question* and
compares it with *chunk* embeddings. That works for paraphrase ("how long did
training take") but is weak on exact tokens -- numbers, model names, symbols
like "N = 6" or "32000". BM25 is the opposite: it only matches literal terms,
but it matches them precisely. Merging the two covers each one's blind spot.

WHY Reciprocal Rank Fusion (RRF): the two methods produce scores on completely
different scales (cosine similarity in [-1, 1] vs an unbounded BM25 score), so
averaging or normalising them would be arbitrary. RRF only uses each method's
RANK, score = sum(1 / (k + rank)), which needs no calibration and is robust to
one list being much more confident than the other. k = 60 is the value from the
original RRF paper and it damps the influence of the very top ranks so one
method cannot dominate on a single lucky hit.

All settings below are FIXED before any measurement and were not tuned against
the held-out q21-q30 questions:
  - BM25 index: the same 1500/50 chunks the current dense index uses (the
    best_config recorded in eval_results.json),
  - RRF k = 60,
  - 10 candidates from each method before fusing, then the usual top 3.

The existing dense path is untouched: dense results here come from
evaluator.question_is_hit / the same persisted Chroma collection, so the
dense-only numbers are directly comparable with eval_results.json.

Retrieval only -- this module makes no LLM calls.

The original dense-vs-hybrid experiment that justified this module lives in
legacy/hybrid_eval.py; current measurements come from eval/run_eval.py.
"""

import re
import time
from collections import Counter
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    RERANK_ENABLED,
    RERANK_POOL,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATES,
    TOP_K,
    page_number,
    source_filename,
)


# --- Fixed settings (do not tune) ------------------------------------------
RRF_K = 60
# Retrieve N per method before fusing: fusing only 3+3 would let a single
# disagreement between the lists decide the result, while a deep candidate pool
# gives RRF something to actually re-rank. Set in config.RETRIEVAL_CANDIDATES
# (default 10) so ablations can change it behind a switch.
CANDIDATES_PER_METHOD = RETRIEVAL_CANDIDATES


@lru_cache(maxsize=1)
def _cross_encoder():
    """Load the reranker once, only when reranking is switched on."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank_indices(query: str, chunks: list[Document], fused: list[int],
                   pool: int = RERANK_POOL, return_scores: bool = False):
    """Re-order the first `pool` fused results by cross-encoder score.

    Mirrors eval/run_eval.py's "hybrid_rerank" method exactly, so what the
    evaluation measured is what the app runs. Results beyond the pool keep
    their RRF order after the reranked ones.
    """
    head = fused[:pool]
    if not head:
        return (fused, {}) if return_scores else fused
    scores = _cross_encoder().predict(
        [(query, chunks[index].page_content) for index in head], show_progress_bar=False
    )
    order = sorted(range(len(head)), key=lambda j: (-float(scores[j]), j))
    reordered = [head[j] for j in order] + fused[pool:]
    if return_scores:
        return reordered, {head[j]: float(scores[j]) for j in range(len(head))}
    return reordered


class VectorSearchStore(Protocol):
    """The Chroma search methods needed by hybrid retrieval."""

    def similarity_search(
        self,
        query: str,
        k: int,
        filter: dict[str, str] | None = None,
    ) -> list[Document]: ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
        filter: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]: ...  # (document, distance)


# ---------------------------------------------------------------------------
# Tokenisation and BM25
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    WHY not whitespace split: the PDFs contain punctuation glued to numbers
    ("28.4", "(91.3", "N = 6"), and punctuation-only tokens would inflate the
    BM25 vocabulary without ever matching. Keeping digits intact matters
    because many eval questions turn on a number.
    """
    return _TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """Build a BM25 index over the chunk texts.

    BM25Okapi is the standard variant (term frequency saturation + document
    length normalisation) and needs no training, so it is rebuilt in memory per
    process instead of being persisted like the dense index.
    """
    return BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])


def bm25_ranking(bm25: BM25Okapi, query: str, n: int) -> list[int]:
    """Return the indices of the n best BM25 chunks, best first.

    WHY sort indices rather than call bm25.get_top_n: get_top_n returns the
    document texts, and we need positions to fuse with the dense list and to
    map back to Document objects (which carry the filename metadata the hit
    check needs).
    """
    scores = bm25.get_scores(tokenize(query))
    # Stable, deterministic ordering: score desc, then index asc so equal
    # scores (very common for short queries) never shuffle between runs.
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:n]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge ranked index lists into one ordering using RRF.

    score(chunk) = sum over lists of 1 / (k + rank), rank starting at 1.

    WHY ranks and not scores: see the module docstring -- the two scales are
    incomparable. WHY the tie-break on best_rank then index: chunks that appear
    in only one list, or not at all, must still come out in a reproducible
    order, otherwise two identical runs could report different hit rates.
    """
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            if rank < best_rank.get(idx, 10**9):
                best_rank[idx] = rank
    return sorted(scores, key=lambda i: (-scores[i], best_rank[i], i))


def hybrid_search(
    query: str,
    store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: list[Document],
    n_results: int = TOP_K,
    metadata_filter: dict[str, str] | None = None,
) -> list[Document]:
    """Return top-n chunks from BM25 + dense via RRF.

    The optional metadata filter is applied to both retrieval methods so
    paper-specific queries cannot mix evidence across PDFs.
    """
    return [document for document, _ in _hybrid_search(
        query, store, bm25, chunks, n_results, metadata_filter)]


def hybrid_search_with_scores(
    query: str,
    store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: list[Document],
    n_results: int = TOP_K,
    metadata_filter: dict[str, str] | None = None,
    trace: dict | None = None,
) -> list[tuple[Document, float | None]]:
    """Return fused chunks with their dense cosine similarity (None if dense missed them).

    Pass a dict as `trace` to receive every stage of the ranking (see _hybrid_search).
    """
    return _hybrid_search(query, store, bm25, chunks, n_results, metadata_filter, trace)


def cosine_from_distance(distance: float) -> float:
    """Convert Chroma's default distance to cosine similarity.

    Chroma's default space is SQUARED L2, and all-MiniLM-L6-v2 vectors are unit
    length, so distance = 2 - 2 cos, i.e. cos = 1 - distance / 2 (checked
    against directly computed cosines). WHY not LangChain's relevance score:
    it maps this distance to 1 - distance / sqrt(2) = sqrt(2) cos - 0.414,
    which is not a cosine and goes negative; trust.py's cut-offs were set on
    that scale while being documented as cosine.
    """
    return 1.0 - float(distance) / 2.0


def _hybrid_search(
    query: str,
    store: VectorSearchStore,
    bm25: BM25Okapi,
    chunks: list[Document],
    n_results: int,
    metadata_filter: dict[str, str] | None,
    trace: dict | None = None,
) -> list[tuple[Document, float | None]]:
    """The one live hybrid pipeline: BM25 + dense -> RRF -> optional rerank.

    When `trace` is a dict it is filled with what each stage did, for the
    retrieval inspector: per-stage timings in ms, the candidates each retriever
    returned with rank and score, and the fused order with RRF (and rerank)
    scores. Tracing only records values computed anyway, so it cannot change
    the result.
    """
    clock = time.perf_counter
    timings: dict[str, float] = {}

    started = clock()
    if metadata_filter:
        # Rebuild BM25 over just the selected paper so its lexical ranking and
        # dense ranking both operate on an isolated candidate corpus.
        eligible = [
            (index, chunk)
            for index, chunk in enumerate(chunks)
            if all(chunk.metadata.get(key) == value
                   for key, value in metadata_filter.items())
        ]
        if not eligible:
            return []
        scoring_bm25 = build_bm25([chunk for _, chunk in eligible])
        to_global = [index for index, _ in eligible]
    else:
        scoring_bm25, to_global = bm25, None
    bm25_scores = scoring_bm25.get_scores(tokenize(query))
    local = sorted(range(len(bm25_scores)), key=lambda i: (-bm25_scores[i], i))[:CANDIDATES_PER_METHOD]
    lexical_idx = [to_global[i] for i in local] if to_global else local
    lexical_score = {index: float(bm25_scores[i]) for index, i in zip(lexical_idx, local)}
    timings["bm25_ms"] = (clock() - started) * 1e3

    started = clock()
    search_filter = {"filter": metadata_filter} if metadata_filter else {}
    dense_results = store.similarity_search_with_score(
        query, k=CANDIDATES_PER_METHOD, **search_filter
    )
    timings["dense_ms"] = (clock() - started) * 1e3
    dense_scores = {
        _document_key(document): cosine_from_distance(distance)
        for document, distance in dense_results
    }
    # Include source and page in the identity so duplicate text from separate
    # documents cannot be assigned to the wrong BM25 position.
    position = {_document_key(chunk): index for index, chunk in enumerate(chunks)}
    dense_idx = [
        position[_document_key(document)]
        for document, _ in dense_results
        if _document_key(document) in position
    ]

    started = clock()
    fused = reciprocal_rank_fusion([dense_idx, lexical_idx])
    timings["fusion_ms"] = (clock() - started) * 1e3
    rerank_scores: dict[int, float] = {}
    if RERANK_ENABLED:
        started = clock()
        fused, rerank_scores = rerank_indices(query, chunks, fused, return_scores=True)
        timings["rerank_ms"] = (clock() - started) * 1e3

    if trace is not None:
        dense_rank = {index: rank for rank, index in enumerate(dense_idx, start=1)}
        lexical_rank = {index: rank for rank, index in enumerate(lexical_idx, start=1)}

        def describe(index: int) -> dict:
            key = _document_key(chunks[index])
            return {
                "citation": f"{key[0]}, p. {key[1]}",
                "dense_rank": dense_rank.get(index),
                "dense_cosine": _round(dense_scores.get(key)),
                "bm25_rank": lexical_rank.get(index),
                "bm25_score": _round(lexical_score.get(index)),
                "rrf_score": _round(sum(1.0 / (RRF_K + ranks[index])
                                        for ranks in (dense_rank, lexical_rank) if index in ranks)),
                "rerank_score": _round(rerank_scores.get(index)),
                "selected": index in fused[:n_results],
                "preview": chunks[index].page_content[:240],
            }

        trace.update({
            "settings": {"candidates_per_method": CANDIDATES_PER_METHOD, "rrf_k": RRF_K,
                         "rerank": RERANK_ENABLED, "top_k": n_results,
                         "filter": metadata_filter or None},
            "timings_ms": {name: round(value, 2) for name, value in timings.items()},
            "candidates": [describe(index) for index in fused],
        })

    return [
        (chunks[index], dense_scores.get(_document_key(chunks[index])))
        for index in fused[:n_results]
    ]


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _document_key(document: Document) -> tuple[str, str, str]:
    """Identify a chunk by source, page, and text across Chroma and BM25."""
    return (
        source_filename(document.metadata),
        page_number(document.metadata),
        document.page_content,
    )


def chunk_keys(chunks: list[Document]) -> Counter:
    """Multiset of chunk identities, for checking two corpora hold the same chunks."""
    return Counter(_document_key(chunk) for chunk in chunks)


def unique_chunks(
    results: list[tuple[Document, float | None]],
) -> list[tuple[Document, float | None]]:
    """Drop repeated chunks while keeping rank order.

    WHY by chunk identity and not by citation: a 1500-character chunk is
    smaller than most pages (117 of 140 pages in the corpus produce two or
    more chunks), so two different passages often share one "file, p. X"
    label. Removing duplicates by label silently discarded the lower-ranked
    passage -- 33 passages across the 60 held-out queries -- even when it held
    the answer. Only a truly repeated chunk (possible when several per-paper
    searches are merged) is removed here.
    """
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for document, score in results:
        key = _document_key(document)
        if key not in seen:
            seen.add(key)
            unique.append((document, score))
    return unique
