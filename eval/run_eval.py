"""Retrieval-only evaluation harness (zero LLM calls).

Scores several retrieval configurations on a question file and reports, per
configuration: hit@k (count and %), MRR@10, a per-question-type breakdown and
95% bootstrap confidence intervals, plus PAIRED bootstrap intervals for the
difference between each configuration and the live baseline (hybrid RRF).

WHY a separate index: the live app's chroma_db/ is never read or written here.
The harness builds (or reuses) its own Chroma index under eval/.index/, keyed by
a fingerprint of the corpus files, chunk settings and embedding model, so an
experiment can never change what the app serves.

WHY a page-based hit: the old harness counted a hit when a chunk from the right
PDF contained ANY expected keyword, which is lenient ("BLEU" appears on many
pages). Here a retrieved chunk is "gold" only if it comes from the right file
AND from one of the question's gold pages.

Zero-LLM guarantee: the `groq` package is blocked before any project import, so
any code path that tries to reach the model raises ImportError instead of
spending quota.

Usage:
    python eval/run_eval.py                                  # held-out v2 set
    python eval/run_eval.py --questions eval_questions.json  # legacy dev set
    python eval/run_eval.py --min-hit-rate 0.6               # CI gate (exit 1 below)
    python eval/run_eval.py --only-files source_doc.pdf gpt4_technical_report.pdf
"""

from __future__ import annotations

import os
import sys

# --- Zero-LLM guard: must run before any project module is imported ---------
sys.modules["groq"] = None  # type: ignore[assignment]  # any `import groq` now fails
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.pop("GROQ_API_KEY", None)

import argparse
import fnmatch
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import PDF_DIR, page_number, source_filename  # noqa: E402
from hybrid import bm25_ranking, build_bm25, reciprocal_rank_fusion  # noqa: E402

EVAL_DIR = ROOT / "eval"
INDEX_ROOT = EVAL_DIR / ".index"
DEFAULT_QUESTIONS = EVAL_DIR / "heldout_v2_questions.json"
DEFAULT_RESULTS = EVAL_DIR / "results_v2.json"
# Personal files are never part of the evaluation corpus.
EXCLUDE_PATTERNS = ["*Resume*"]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL = 50          # how deep each base ranking is computed (for ranks and MRR)
MRR_CUTOFF = 10
BOOTSTRAP_SAMPLES = 10_000
SEED = 0


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexSettings:
    """Everything that changes the chunks or their embeddings."""
    chunk_size: int = 1500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_page: bool = False   # Phase 3 option; False = live behaviour (split per page)


@dataclass(frozen=True)
class RetrievalConfig:
    """One scored configuration. Defaults reproduce the live app."""
    name: str
    method: str                      # dense | bm25 | hybrid | hybrid_rerank
    top_k: int = 3
    candidates_per_method: int = 10  # live hybrid.CANDIDATES_PER_METHOD
    rrf_k: int = 60                  # live hybrid.RRF_K
    rerank_pool: int = 0             # rerank the first N of the fused list
    reranker: str = RERANKER_MODEL   # cross-encoder used when rerank_pool > 0


DEFAULT_CONFIGS = [
    RetrievalConfig("dense", "dense"),
    RetrievalConfig("bm25", "bm25"),
    RetrievalConfig("hybrid_rrf60", "hybrid"),
    RetrievalConfig("hybrid_rerank_top10", "hybrid_rerank", rerank_pool=10),
    RetrievalConfig("hybrid_rerank_top20", "hybrid_rerank", rerank_pool=20),
]
# Not in the default run (1.1 GB download, slow on CPU); select with
# --configs hybrid_rrf60 hybrid_bge_rerank_top10
EXTRA_CONFIGS = [
    RetrievalConfig("hybrid_bge_rerank_top10", "hybrid_rerank", rerank_pool=10,
                    reranker="BAAI/bge-reranker-base"),
]
BASELINE = "hybrid_rrf60"


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    """Load the v2 schema, or adapt the legacy eval_questions.json schema.

    v2 item: {"id", "q", "file", "page": [int] | null, "type", ...}
    legacy item: {"id", "question", "expected_source", "source_page", "type"}
    Legacy should_refuse items become type "unanswerable".
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for item in raw:
        if "q" in item:
            questions.append(item)
            continue
        if item.get("should_refuse"):
            questions.append({"id": item["id"], "q": item["question"], "file": None,
                              "page": None, "type": "unanswerable"})
        else:
            questions.append({"id": item["id"], "q": item["question"],
                              "file": item["expected_source"],
                              "page": [int(item["source_page"])],
                              "type": f"legacy_{item.get('type', 'unknown')}"})
    return questions


# ---------------------------------------------------------------------------
# Corpus and index
# ---------------------------------------------------------------------------
def corpus_files(only_files: list[str] | None) -> list[Path]:
    files = []
    for path in sorted(Path(PDF_DIR).glob("*.pdf")):
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_PATTERNS):
            continue
        if only_files and path.name not in only_files:
            continue
        files.append(path)
    if not files:
        raise SystemExit("No PDFs to evaluate on.")
    return files


def load_pages(files: list[Path]) -> list:
    from langchain_community.document_loaders import PyPDFLoader

    pages = []
    for path in files:
        for page in PyPDFLoader(str(path)).load():
            page.metadata["filename"] = path.name
            pages.append(page)
    return pages


def make_chunks(pages: list, settings: IndexSettings) -> list:
    """Split exactly like the live app (per page), unless cross_page is on."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not settings.cross_page:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        return splitter.split_documents(pages)
    from ingest import split_documents_cross_page
    return split_documents_cross_page(pages, settings.chunk_size, settings.chunk_overlap)


def index_fingerprint(files: list[Path], settings: IndexSettings) -> str:
    digest = hashlib.sha1()
    for path in files:
        digest.update(path.name.encode())
        digest.update(hashlib.sha1(path.read_bytes()).digest())
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode())
    if settings.cross_page:
        # The splitter's code is part of the index identity, so an edited
        # splitter never silently reuses chunks built by the old one.
        import inspect

        from ingest import split_documents_cross_page
        digest.update(inspect.getsource(split_documents_cross_page).encode())
    return digest.hexdigest()[:12]


def get_store(chunks: list, files: list[Path], settings: IndexSettings):
    """Build the eval index once per fingerprint, then reuse it."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    fingerprint = index_fingerprint(files, settings)
    path = INDEX_ROOT / fingerprint
    # Normalised vectors: a no-op for MiniLM (its pipeline already ends with a
    # Normalize layer, so the baseline is unchanged) and required for bge.
    embeddings = HuggingFaceEmbeddings(model=settings.embedding_model,
                                       encode_kwargs={"normalize_embeddings": True})
    store = Chroma(collection_name="eval", embedding_function=embeddings,
                   persist_directory=str(path))
    count = store._collection.count()
    if count != len(chunks):
        if count:
            store.delete_collection()
            store = Chroma(collection_name="eval", embedding_function=embeddings,
                           persist_directory=str(path))
        print(f"Building eval index {fingerprint} ({len(chunks)} chunks)...", flush=True)
        store.add_documents(chunks)
    return store, fingerprint


def chunk_key(document) -> tuple[str, str, str]:
    return (source_filename(document.metadata), page_number(document.metadata),
            document.page_content)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class Retriever:
    """Computes base rankings once per question and derives every config."""

    def __init__(self, store, chunks: list):
        self.store = store
        self.chunks = chunks
        self.bm25 = build_bm25(chunks)
        self.position: dict = {}
        for index, chunk in enumerate(chunks):
            self.position.setdefault(chunk_key(chunk), index)
        self._cross_encoders: dict = {}
        self._cache: dict[str, tuple[list[int], list[int]]] = {}

    def base(self, query: str) -> tuple[list[int], list[int]]:
        if query not in self._cache:
            dense_docs = self.store.similarity_search(query, k=POOL)
            dense = [self.position[chunk_key(d)] for d in dense_docs
                     if chunk_key(d) in self.position]
            lexical = bm25_ranking(self.bm25, query, POOL)
            self._cache[query] = (dense, lexical)
        return self._cache[query]

    def cross_encoder(self, model: str = RERANKER_MODEL):
        if model not in self._cross_encoders:
            from sentence_transformers import CrossEncoder
            self._cross_encoders[model] = CrossEncoder(model)
        return self._cross_encoders[model]

    def timed_run(self, query: str, config: RetrievalConfig) -> dict[str, float]:
        """Run one config the way the live app does, uncached, and time each stage (ms).

        WHY separate from ranking(): ranking() reuses cached base lists so that
        scoring many configs is cheap, which makes its timings meaningless.
        Here every stage runs from scratch with the live candidate depth.
        """
        timings: dict[str, float] = {}
        clock = time.perf_counter
        if config.method in {"dense", "hybrid", "hybrid_rerank"}:
            depth = config.top_k if config.method == "dense" else config.candidates_per_method
            started = clock()
            dense_docs = self.store.similarity_search(query, k=depth)
            timings["dense_ms"] = (clock() - started) * 1e3
            dense = [self.position[chunk_key(d)] for d in dense_docs if chunk_key(d) in self.position]
        if config.method in {"bm25", "hybrid", "hybrid_rerank"}:
            depth = config.top_k if config.method == "bm25" else config.candidates_per_method
            started = clock()
            lexical = bm25_ranking(self.bm25, query, depth)
            timings["bm25_ms"] = (clock() - started) * 1e3
        if config.method in {"hybrid", "hybrid_rerank"}:
            started = clock()
            fused = reciprocal_rank_fusion([dense, lexical], k=config.rrf_k)
            timings["fusion_ms"] = (clock() - started) * 1e3
        if config.method == "hybrid_rerank":
            pool = fused[:config.rerank_pool]
            started = clock()
            self.cross_encoder(config.reranker).predict([(query, self.chunks[i].page_content) for i in pool],
                                         show_progress_bar=False)
            timings["rerank_ms"] = (clock() - started) * 1e3
        timings["total_ms"] = sum(timings.values())
        return timings

    def ranking(self, query: str, config: RetrievalConfig) -> list[int]:
        """Full ordered list of chunk indices for one config (best first)."""
        dense, lexical = self.base(query)
        if config.method == "dense":
            return dense
        if config.method == "bm25":
            return lexical
        n = config.candidates_per_method
        fused = reciprocal_rank_fusion([dense[:n], lexical[:n]], k=config.rrf_k)
        if config.method == "hybrid":
            return fused
        if config.method == "hybrid_rerank":
            pool = fused[:config.rerank_pool]
            scores = self.cross_encoder(config.reranker).predict(
                [(query, self.chunks[i].page_content) for i in pool],
                show_progress_bar=False,
            )
            order = sorted(range(len(pool)), key=lambda j: (-float(scores[j]), j))
            return [pool[j] for j in order] + fused[config.rerank_pool:]
        raise ValueError(f"unknown method {config.method}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def chunk_pages(meta: dict) -> list[int]:
    """Pages a chunk covers: its own page, or start..end for cross-page chunks."""
    start = page_number(meta)
    if not start.isdecimal():
        return []
    end = str(meta.get("page_end_label") or start)
    return list(range(int(start), int(end) + 1)) if end.isdecimal() else [int(start)]


def gold_ranks(ranking: list[int], chunks: list, question: dict) -> dict[int, int | None]:
    """For each gold page, the 1-based rank of its first chunk (None if absent)."""
    ranks: dict[int, int | None] = {page: None for page in question["page"]}
    for rank, index in enumerate(ranking, start=1):
        meta = chunks[index].metadata
        if source_filename(meta) != question["file"]:
            continue
        for page in chunk_pages(meta):
            if page in ranks and ranks[page] is None:
                ranks[page] = rank
    return ranks


def near_page_hit(ranking: list[int], chunks: list, question: dict, top_k: int) -> bool:
    for index in ranking[:top_k]:
        meta = chunks[index].metadata
        if source_filename(meta) == question["file"] and any(
            abs(covered - page) <= 1 for covered in chunk_pages(meta) for page in question["page"]
        ):
            return True
    return False


def is_relevant(meta: dict, question: dict) -> bool:
    """A chunk is relevant when it comes from the gold file and covers a gold page.

    Relevance is page-level, like the hit metric: when a page is split into
    several chunks, all of them count, even though only one may hold the
    answer. Precision and nDCG therefore measure "right page", not "right
    sentence"; eval/context_coverage.py measures the sentence level.
    """
    return (source_filename(meta) == question["file"]
            and any(page in question["page"] for page in chunk_pages(meta)))


def ranking_metrics(ranking: list[int], chunks: list, question: dict, top_k: int,
                    relevant_total: int, hops: list[list[int]],
                    ranks: dict[int, int | None]) -> dict[str, float]:
    """Precision@k, Recall@k and nDCG@k for one question (binary relevance).

    precision@k = relevant chunks in the top k / k
    recall@k    = facts (hops) with at least one gold page in the top k / facts.
                  A single-fact question has one hop per listed page unless
                  `hops` groups them, so recall equals hit for most questions
                  and differs only for multi-hop ones.
    nDCG@k      = DCG / ideal DCG, with DCG = sum(rel_i / log2(i + 1)); the
                  ideal ranking puts min(k, relevant chunks in corpus) first.
    """
    relevance = [1.0 if is_relevant(chunks[i].metadata, question) else 0.0
                 for i in ranking[:top_k]]
    dcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(top_k, relevant_total) + 1))
    covered = sum(
        any(ranks.get(page) is not None and ranks[page] <= top_k for page in hop) for hop in hops
    )
    return {
        "precision": sum(relevance) / top_k,
        "recall": covered / len(hops) if hops else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def summarize(rows: list[dict], key: str, rng: np.random.Generator) -> dict:
    values = np.array([row[key] for row in rows], dtype=float)
    return {"count": round(float(values.sum()), 4), "n": len(values),
            "mean": round(float(values.mean()), 4) if len(values) else None,
            "ci95": bootstrap_ci(values, rng)}


def git_commit() -> str | None:
    try:
        # --dirty marks results produced by uncommitted code, so a result file
        # never claims a commit that did not contain the code that made it.
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def latency(questions: list[dict], retriever: Retriever, config: RetrievalConfig) -> dict:
    """Median and 95th-percentile stage timings over the question set (ms).

    One untimed warm-up query first, so model loading is not counted.
    """
    if not questions:
        return {}
    retriever.timed_run(questions[0]["q"], config)
    runs = [retriever.timed_run(q["q"], config) for q in questions]
    return {
        stage: {"p50": round(float(np.percentile(values, 50)), 1),
                "p95": round(float(np.percentile(values, 95)), 1)}
        for stage in runs[0]
        for values in [[run[stage] for run in runs]]
    }


def evaluate(questions: list[dict], retriever: Retriever,
             configs: list[RetrievalConfig], measure_latency: bool = True) -> dict:
    answerable = [q for q in questions if q.get("file") and q.get("page")]
    rng = np.random.default_rng(SEED)
    relevant_totals = {
        q["id"]: sum(is_relevant(chunk.metadata, q) for chunk in retriever.chunks)
        for q in answerable
    }
    per_config: dict = {}
    per_question: dict = {q["id"]: {"type": q["type"], "file": q["file"], "page": q["page"]}
                          for q in answerable}

    for config in configs:
        started = time.perf_counter()
        rows = []
        for q in answerable:
            ranking = retriever.ranking(q["q"], config)
            ranks = gold_ranks(ranking, retriever.chunks, q)
            found = [r for r in ranks.values() if r is not None]
            first = min(found) if found else None
            # A multi-hop question is fully covered when EVERY hop (one fact)
            # has at least one of its pages in the top k. A fact printed on two
            # pages is one hop, so finding either page is enough.
            hops = q.get("hops") or [[page] for page in q["page"]]
            covered = all(
                any(ranks.get(page) is not None and ranks[page] <= config.top_k for page in hop)
                for hop in hops
            )
            row = {
                "hit": float(first is not None and first <= config.top_k),
                "hit_all": float(covered),
                "near_hit": float(near_page_hit(ranking, retriever.chunks, q, config.top_k)),
                "rr": 1.0 / first if first is not None and first <= MRR_CUTOFF else 0.0,
                "type": q["type"],
                **ranking_metrics(ranking, retriever.chunks, q, config.top_k,
                                  relevant_totals[q["id"]], hops, ranks),
            }
            rows.append(row)
            per_question[q["id"]][config.name] = {
                "first_gold_rank": first, "gold_page_ranks": {str(k): v for k, v in ranks.items()},
                "hit": bool(row["hit"]),
            }
        block = {
            "config": asdict(config),
            f"hit@{config.top_k}": summarize(rows, "hit", rng),
            "mrr@10": summarize(rows, "rr", rng),
            f"precision@{config.top_k}": summarize(rows, "precision", rng),
            f"recall@{config.top_k}": summarize(rows, "recall", rng),
            f"ndcg@{config.top_k}": summarize(rows, "ndcg", rng),
            "near_page_hit": summarize(rows, "near_hit", rng),
            "by_type": {},
            "seconds": round(time.perf_counter() - started, 2),
        }
        if measure_latency:
            block["latency_ms"] = latency(answerable, retriever, config)
        for qtype in sorted({row["type"] for row in rows}):
            subset = [row for row in rows if row["type"] == qtype]
            block["by_type"][qtype] = {
                "hits": int(sum(row["hit"] for row in subset)), "n": len(subset),
                "mrr": round(float(np.mean([row["rr"] for row in subset])), 4),
            }
            if qtype == "multi_hop":
                block["by_type"][qtype]["all_pages_hits"] = int(sum(row["hit_all"] for row in subset))
        block["_rows"] = rows
        per_config[config.name] = block

    # Paired bootstrap: every config vs the live baseline, on the same questions.
    if BASELINE in per_config:
        base_rows = per_config[BASELINE]["_rows"]
        for name, block in per_config.items():
            if name == BASELINE:
                continue
            for metric in ("hit", "rr", "precision", "recall", "ndcg"):
                diff = np.array([a[metric] - b[metric] for a, b in zip(block["_rows"], base_rows)])
                block.setdefault("vs_baseline", {})[metric] = {
                    "delta": round(float(diff.mean()), 4),
                    "ci95": bootstrap_ci(diff, rng),
                    "better_on": int(sum(d > 0 for d in diff)),
                    "worse_on": int(sum(d < 0 for d in diff)),
                }
    for block in per_config.values():
        block.pop("_rows")

    return {
        "n_questions": len(questions),
        "n_answerable_scored": len(answerable),
        "n_unanswerable_skipped": len(questions) - len(answerable),
        "configs": per_config,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{x:.0%}"


def _delta(block: dict, metric: str, as_pct: bool) -> str:
    delta = block.get("vs_baseline", {}).get(metric)
    if not delta:
        return "baseline"
    fmt = (lambda x: f"{x:+.0%}") if as_pct else (lambda x: f"{x:+.3f}")
    return (f"{fmt(delta['delta'])} [{fmt(delta['ci95'][0])}, {fmt(delta['ci95'][1])}] "
            f"+{delta['better_on']}/-{delta['worse_on']}")


def format_report(result: dict) -> str:
    configs = result["configs"]
    out = [f"Scored {result['n_answerable_scored']} answerable questions "
           f"({result['n_unanswerable_skipped']} unanswerable skipped: no gold page).", ""]
    out.append("| config | hit@k | 95% CI | MRR@10 | P@k | R@k | nDCG@k | ±1 page hit | "
               "Δ hit vs baseline [95% CI] | Δ nDCG vs baseline |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, block in configs.items():
        k = block["config"]["top_k"]
        hit = block[f"hit@{k}"]
        near = block["near_page_hit"]
        out.append(
            f"| {name} | {int(hit['count'])}/{hit['n']} ({pct(hit['mean'])}) | "
            f"{pct(hit['ci95'][0])}–{pct(hit['ci95'][1])} | {block['mrr@10']['mean']:.3f} | "
            f"{block[f'precision@{k}']['mean']:.3f} | {block[f'recall@{k}']['mean']:.3f} | "
            f"{block[f'ndcg@{k}']['mean']:.3f} | {int(near['count'])}/{near['n']} | "
            f"{_delta(block, 'hit', True)} | {_delta(block, 'ndcg', False)} |")

    types = sorted({t for block in configs.values() for t in block["by_type"]})
    out += ["", "Hits by question type:", ""]
    out.append("| config | " + " | ".join(types) + " |")
    out.append("|---|" + "---|" * len(types))
    for name, block in configs.items():
        cells = []
        for t in types:
            cell = block["by_type"].get(t)
            if not cell:
                cells.append("-")
                continue
            text = f"{cell['hits']}/{cell['n']}"
            if "all_pages_hits" in cell:
                text += f" (all pages {cell['all_pages_hits']}/{cell['n']})"
            cells.append(text)
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    timed = {name: block["latency_ms"] for name, block in configs.items() if block.get("latency_ms")}
    if timed:
        stages = ["dense_ms", "bm25_ms", "fusion_ms", "rerank_ms", "total_ms"]
        out += ["", "Retrieval latency per query, CPU, ms (median / 95th percentile; "
                "one warm-up query excluded, LLM time not included):", ""]
        out.append("| config | " + " | ".join(s.removesuffix("_ms") for s in stages) + " |")
        out.append("|---|" + "---|" * len(stages))
        for name, lat in timed.items():
            cells = [f"{lat[s]['p50']:.1f} / {lat[s]['p95']:.1f}" if s in lat else "–" for s in stages]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def print_report(result: dict) -> None:
    print("\n" + format_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--only-files", nargs="*", default=None,
                        help="restrict corpus AND questions to these PDFs (e.g. for CI)")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="subset of config names to run")
    parser.add_argument("--min-hit-rate", type=float, default=None,
                        help="exit 1 if the gate config's hit rate is below this (0-1)")
    parser.add_argument("--gate-config", default=BASELINE)
    parser.add_argument("--top-k", type=int, default=None,
                        help="score every config at this k (default: 3, the legacy setting; the app uses config.TOP_K)")
    parser.add_argument("--no-latency", action="store_true", help="skip the timed runs")
    parser.add_argument("--md", type=Path, default=None, help="also write the report as Markdown")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    files = corpus_files(args.only_files)
    names = {path.name for path in files}
    questions = [q for q in questions if q.get("file") is None or q["file"] in names]

    settings = IndexSettings()
    pages = load_pages(files)
    chunks = make_chunks(pages, settings)
    store, fingerprint = get_store(chunks, files, settings)
    retriever = Retriever(store, chunks)

    configs = ([c for c in DEFAULT_CONFIGS + EXTRA_CONFIGS if c.name in args.configs]
               if args.configs else list(DEFAULT_CONFIGS))
    if args.top_k is not None:
        configs = [replace(c, top_k=args.top_k) for c in configs]
    result = evaluate(questions, retriever, configs, measure_latency=not args.no_latency)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "questions_file": Path(os.path.relpath(args.questions.resolve(), ROOT)).as_posix(),
        "corpus": {"files": sorted(names), "chunks": len(chunks), "index": fingerprint},
        "index_settings": asdict(settings),
        "retrieval_only": True,
        "llm_calls": 0,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED, "method": "percentile"},
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_report(result)
    print(f"\nSaved {args.out}")
    if args.md:
        header = (f"Generated by `python eval/run_eval.py` (zero LLM calls) at commit "
                  f"{payload['git_commit']}. Questions: `{payload['questions_file']}`. "
                  f"Corpus: {len(names)} PDFs, {len(chunks)} chunks.\n\n")
        args.md.write_text(header + format_report(result) + "\n", encoding="utf-8")
        print(f"Saved {args.md}")

    if args.min_hit_rate is not None:
        block = result["configs"].get(args.gate_config)
        if block is None:
            raise SystemExit(f"Gate config {args.gate_config!r} was not run.")
        k = block["config"]["top_k"]
        rate = block[f"hit@{k}"]["mean"]
        if rate < args.min_hit_rate:
            print(f"\nFAIL: {args.gate_config} hit@{k} = {rate:.1%} < {args.min_hit_rate:.1%}")
            raise SystemExit(1)
        print(f"\nPASS: {args.gate_config} hit@{k} = {rate:.1%} >= {args.min_hit_rate:.1%}")


if __name__ == "__main__":
    main()
