"""Production-style FastAPI entry point for the Hybrid RAG application."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from citations import check_citations
from compare import (
    expand_compare_query,
    generate_comparison,
    retrieve_paper_contexts,
)
from config import (
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PDF_DIR,
    PROMPT_TEMPLATE,
    REMOVED_PDF_DIR,
    RERANK_ENABLED,
    TOP_K,
    citation,
    source_filename,
)
from llm import call_llm, call_llm_stream, track_usage
from hybrid import build_bm25, chunk_keys, hybrid_search_with_scores, unique_chunks
from ingest import best_chunk_config, split_documents
from library import remove_pdf_from_library, rename_pdf_in_library
from quiz import (
    build_quiz_context,
    generate_quiz,
    grade_quiz,
    retrieve_quiz_chunks,
)
from trust import assess
from upload import add_pdf_to_library

load_dotenv()

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
QUIZ_HISTORY_PATH = BASE_DIR / "quiz_history.json"
FEEDBACK_PATH = BASE_DIR / "chat_feedback.json"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TOP_K = 10

# One lock for the small JSON stores: FastAPI runs sync endpoints in a thread
# pool, so two simultaneous saves could otherwise interleave and lose an entry.
_json_lock = threading.Lock()
# lru_cache does not stop two threads from running the first (expensive) call
# at once; concurrent Chroma client creation crashes ("Could not connect to
# tenant default_tenant"), so index construction is serialised.
_index_lock = threading.RLock()

app = FastAPI(title="Hybrid RAG Knowledge Assistant", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request models. These must be defined before the routes that use them:
# with `from __future__ import annotations`, FastAPI resolves annotations by
# name at registration time, and an undefined model silently becomes a
# required query parameter (which made every feedback request fail with 422).
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    papers: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class ChatFeedbackRequest(BaseModel):
    message: str = Field(min_length=1)
    feedback: str
    query: str = ""


class RenamePdfRequest(BaseModel):
    filename: str = Field(min_length=1)
    new_filename: str = Field(min_length=1)


class QuizHistoryRequest(BaseModel):
    pdf: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    score: float
    topics: list[str] = Field(default_factory=list)


class QuizGenerateRequest(BaseModel):
    pdf: str = Field(min_length=1)
    topic: str = ""
    difficulty: str = "Medium"
    question_count: int = 5
    question_type: str = "Mixed"
    page_range: str | None = None


class QuizGradeRequest(BaseModel):
    pdf: str = Field(min_length=1)
    topic: str = ""
    questions: list[dict[str, Any]]
    answers: list[str] = Field(min_length=1)
    page_range: str | None = None
    # Must match the values used for generation so grading retrieves the same
    # passages that the questions cite.
    difficulty: str = "Medium"
    question_count: int = 5


class InspectRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    papers: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class CompareRequest(BaseModel):
    pdfs: list[str] = Field(min_length=2, max_length=2)
    question: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# AI-provider errors -> readable HTTP errors (instead of a bare 500)
# ---------------------------------------------------------------------------
@app.exception_handler(RateLimitError)
async def _rate_limited(_: Request, __: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "The AI service is rate limited. Wait a moment, then try again."},
    )


@app.exception_handler(AuthenticationError)
async def _unauthenticated(_: Request, __: AuthenticationError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "The AI service is not configured. Add a valid GROQ_API_KEY to .env."},
    )


@app.exception_handler(APIConnectionError)
@app.exception_handler(APITimeoutError)
async def _unreachable(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "The AI service could not be reached. Check your internet connection."},
    )


# ---------------------------------------------------------------------------
# Index resources
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _parse_pdf(path: str, mtime_ns: int, size: int) -> tuple:
    """Parse one PDF; cached by (path, mtime, size) so unchanged files are
    never re-read when the library changes. The stat values are only part of
    the key: an edited or replaced file gets a new key and is parsed again."""
    return tuple(PyPDFLoader(path).load())


def load_pdf_pages(path: Path) -> list:
    """Return fresh copies of a PDF's pages tagged with their filename."""
    stat = path.stat()
    pages = []
    for page in _parse_pdf(str(path), stat.st_mtime_ns, stat.st_size):
        copy = page.model_copy(deep=True)
        copy.metadata["filename"] = path.name
        pages.append(copy)
    return pages


def load_documents() -> list:
    """Load every PDF in the library (same result as ingest.load_documents)."""
    return [page for name in list_pdf_names() for page in load_pdf_pages(Path(PDF_DIR) / name)]


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> Chroma:
    with _index_lock:
        return _get_vector_store()


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0 and list_pdf_names():
        store.add_documents(split_documents(load_documents(), *best_chunk_config()))
    return store


def get_hybrid_resources():
    with _index_lock:
        return _get_hybrid_resources()


@lru_cache(maxsize=1)
def _get_hybrid_resources():
    store = get_vector_store()
    if not list_pdf_names():
        return [], None
    # The BM25 corpus is re-split from the PDFs (not read back from Chroma)
    # because Chroma returns chunks in storage order, and BM25/RRF break ties
    # by position: re-splitting keeps the exact order eval/run_eval.py measured.
    chunks = split_documents(load_documents(), *best_chunk_config())
    verify_index_matches(store, chunks)
    return chunks, build_bm25(chunks)


def verify_index_matches(store: Chroma, chunks: list) -> None:
    """Fail loudly unless Chroma holds exactly the chunks BM25 will search.

    WHY compare contents, not counts: equal counts can hide different chunks,
    e.g. after changing RAG_CROSS_PAGE or the chunk size without re-ingesting.
    Dense hits whose text is not in the BM25 corpus are then dropped from
    fusion without any error, and hybrid search quietly becomes BM25-only.
    """
    stored = store.get(include=["documents", "metadatas"])
    indexed = chunk_keys([
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ])
    expected = chunk_keys(chunks)
    if indexed != expected:
        missing = sum((expected - indexed).values())
        extra = sum((indexed - expected).values())
        raise RuntimeError(
            "The dense index and BM25 corpus do not match: "
            f"{missing} chunk(s) missing from the index, {extra} stale chunk(s) in it "
            f"(index={sum(indexed.values())}, local corpus={len(chunks)}). "
            "Run `python ingest.py` to rebuild the index."
        )


def clear_indexes() -> None:
    with _index_lock:
        _get_hybrid_resources.cache_clear()
        _get_vector_store.cache_clear()


def list_pdf_names() -> list[str]:
    return sorted(path.name for path in Path(PDF_DIR).glob("*.pdf"))


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _append_json(path: Path, entry: dict[str, Any]) -> None:
    with _json_lock:
        entries = _read_json_list(path)
        entries.append(entry)
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _llm_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key) and key != "your_groq_api_key_here"


def retrieve_for_api(
    query: str,
    papers: list[str] | None = None,
    n_results: int = TOP_K,
    trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval over all papers, or n_results per selected paper.

    `score` is the dense cosine similarity (None when only BM25 found the
    chunk). Pass a list as `trace` to receive one stage-by-stage record per
    search (one per selected paper), as shown by the retrieval inspector.
    """
    chunks, bm25 = get_hybrid_resources()
    if not chunks:
        return []
    retrieved = []
    # _hybrid_search rebuilds BM25 over the filtered subset itself, so the
    # global index is passed through unchanged for per-paper searches.
    for paper in (papers or [None]):
        search_trace: dict[str, Any] | None = {} if trace is not None else None
        retrieved.extend(hybrid_search_with_scores(
            query,
            get_vector_store(),
            bm25,
            chunks,
            n_results=n_results,
            metadata_filter={"filename": paper} if paper else None,
            trace=search_trace,
        ))
        if trace is not None:
            trace.append({"paper": paper, **(search_trace or {})})
    return [
        {
            "filename": source_filename(document.metadata),
            "citation": citation(document.metadata),
            "text": document.page_content,
            "score": round(score, 4) if score is not None else None,
        }
        for document, score in unique_chunks(retrieved)
    ]


def rewrite_followup(query: str, history: list[dict[str, str]]) -> str:
    """Make a short contextual follow-up suitable for retrieval without changing its meaning."""
    if not history:
        return query
    recent = history[-3:]
    words = set(re.findall(r"[a-z]+", query.lower()))
    if not words.intersection({"it", "they", "this", "that", "those", "them", "its", "their"}):
        return query
    prior = " ".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in recent
    )
    prompt = (
        "Rewrite the follow-up as one standalone search query using the prior "
        "conversation for context. Return only the query.\n"
        f"Conversation:\n{prior}\nFollow-up: {query}"
    )
    try:
        rewritten = call_llm(prompt, temperature=0.0, label="chat_followup").strip()
        return rewritten or query
    except Exception:
        return query


def _paper_title(filename: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(filename).stem).strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Real status for the header indicator instead of a hard-coded label."""
    try:
        chunks, _ = get_hybrid_resources()
        index_error = None
    except Exception as exc:  # noqa: BLE001 - reported to the UI, not raised
        chunks, index_error = [], str(exc)
    return {
        "index_ready": index_error is None and bool(chunks),
        "index_error": index_error,
        "papers": len(list_pdf_names()),
        "chunks": len(chunks),
        "llm_configured": _llm_configured(),
    }


@app.get("/api/pdfs")
def get_pdfs() -> dict[str, list[str]]:
    return {"pdfs": list_pdf_names()}


@app.get("/api/settings")
def get_app_settings() -> dict[str, Any]:
    chunk_size, chunk_overlap = best_chunk_config()
    return {
        "top_k": TOP_K,
        "max_top_k": MAX_TOP_K,
        "model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


@app.get("/api/chat/starters")
def chat_starters() -> dict[str, list[str]]:
    names = list_pdf_names()
    starters = [f"Summarize the main contribution of {_paper_title(name)}." for name in names[:2]]
    if len(names) >= 2:
        starters.append("How do the methodologies in my papers differ?")
    starters += [
        "What limitations are discussed?",
        "What datasets or evaluation metrics are reported?",
        "Explain the key terms a beginner should know.",
    ]
    return {"starters": starters[:4]}


@app.post("/api/chat/feedback")
def save_chat_feedback(request: ChatFeedbackRequest) -> dict[str, bool]:
    if request.feedback not in {"up", "down"}:
        raise HTTPException(status_code=422, detail="Feedback must be up or down")
    _append_json(FEEDBACK_PATH, {
        "query": request.query,
        "message": request.message,
        "feedback": request.feedback,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"saved": True}


@app.post("/api/progress/quiz")
def save_quiz_progress(request: QuizHistoryRequest) -> dict[str, bool]:
    _append_json(QUIZ_HISTORY_PATH, {
        "pdf": request.pdf,
        "difficulty": request.difficulty,
        "score": request.score,
        "topics": request.topics,
        "date": datetime.now(timezone.utc).isoformat(),
    })
    return {"saved": True}


@app.get("/api/progress")
def get_progress() -> dict[str, Any]:
    history = _read_json_list(QUIZ_HISTORY_PATH)
    averages: dict[str, float] = {}
    for difficulty in ("Easy", "Medium", "Hard", "Mixed"):
        scores = [
            float(item["score"]) for item in history
            if item.get("difficulty") == difficulty and isinstance(item.get("score"), (int, float))
        ]
        if scores:
            averages[difficulty] = round(sum(scores) / len(scores), 1)
    # "Weak" topics are those with the lowest average score, not merely the
    # most-quizzed ones: a topic you ace every time is not a topic you miss.
    topic_scores: dict[str, list[float]] = {}
    for item in history:
        for topic in item.get("topics", []):
            topic_scores.setdefault(topic, []).append(float(item.get("score", 0)))
    weak = sorted(
        ((topic, sum(s) / len(s)) for topic, s in topic_scores.items() if sum(s) / len(s) < 70),
        key=lambda pair: (pair[1], pair[0]),
    )[:5]
    all_scores = [float(item["score"]) for item in history if isinstance(item.get("score"), (int, float))]
    return {
        "quizzes_taken": len(history),
        "overall_average": round(sum(all_scores) / len(all_scores), 1) if all_scores else None,
        "averages": averages,
        "most_missed_topics": [topic for topic, _ in weak],
        "recent": list(reversed(history[-10:])),
    }


@app.get("/api/library")
def get_library() -> dict[str, Any]:
    chunks, _ = get_hybrid_resources()
    papers = []
    for filename in list_pdf_names():
        paper_chunks = [c for c in chunks if source_filename(c.metadata) == filename]
        pages = {
            chunk.metadata.get("page_label")
            or (chunk.metadata.get("page") + 1 if isinstance(chunk.metadata.get("page"), int) else None)
            for chunk in paper_chunks
        }
        papers.append({
            "filename": filename,
            "pages": len({page for page in pages if page is not None}),
            "chunks": len(paper_chunks),
            "indexed": bool(paper_chunks),
            "size_kb": round((Path(PDF_DIR) / filename).stat().st_size / 1024),
        })
    return {"papers": papers}


@app.get("/api/library/{filename:path}/preview")
def preview_pdf(filename: str) -> dict[str, Any]:
    path = Path(PDF_DIR) / Path(filename).name
    if Path(filename).name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        pages = load_pdf_pages(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="PDF preview could not be parsed") from exc
    first = pages[0].page_content.strip() if pages else ""
    metadata = pages[0].metadata if pages else {}
    return {
        "filename": filename,
        "pages": len(pages),
        "preview": first[:1500],
        "metadata": {
            key: value for key, value in metadata.items()
            if key in {"title", "author", "creator", "producer", "creationdate"}
            and isinstance(value, (str, int, float)) and str(value).strip()
        },
    }


@app.get("/api/library/{filename:path}/file", include_in_schema=False)
def open_pdf(filename: str) -> FileResponse:
    path = Path(PDF_DIR) / Path(filename).name
    if Path(filename).name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf")


@app.delete("/api/library/{filename:path}")
def delete_pdf(filename: str) -> dict[str, Any]:
    if Path(filename).name != filename:
        raise HTTPException(status_code=422, detail="Invalid filename")
    try:
        remove_pdf_from_library(
            filename, get_vector_store(), Path(PDF_DIR), Path(REMOVED_PDF_DIR), keep_copy=False
        )
        clear_indexes()
        get_hybrid_resources()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"deleted": filename}


@app.post("/api/library/rename")
def rename_pdf(request: RenamePdfRequest) -> dict[str, Any]:
    old = Path(request.filename).name
    new = Path(request.new_filename).name
    if old != request.filename or new != request.new_filename:
        raise HTTPException(status_code=422, detail="Use a simple PDF filename")
    if not new.lower().endswith(".pdf"):
        new += ".pdf"
    if not (Path(PDF_DIR) / old).is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    if (Path(PDF_DIR) / new).exists():
        raise HTTPException(status_code=409, detail="That filename already exists")
    try:
        new_path = rename_pdf_in_library(old, new, Path(PDF_DIR))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Re-index only the renamed paper instead of re-parsing the whole library.
    pages = load_pdf_pages(new_path)
    store = get_vector_store()
    store.delete(where={"filename": old})
    store.add_documents(split_documents(pages, *best_chunk_config()))
    clear_indexes()
    get_hybrid_resources()
    return {"filename": new}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1e3, 1)


def _prepare_chat(request: ChatRequest) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Shared by /api/chat and /api/chat/stream: rewrite, then retrieve.

    Also returns the start of the pipeline trace: how long each step took, the
    tokens the follow-up rewrite used, and what every retrieval stage did.
    """
    available = set(list_pdf_names())
    papers = [paper for paper in request.papers if paper in available]
    trace: dict[str, Any] = {"timings_ms": {}, "tokens": {}}
    started = time.perf_counter()
    with track_usage() as rewrite_usage:
        standalone_query = rewrite_followup(request.query.strip(), request.history)
    trace["timings_ms"]["rewrite"] = _ms(started)
    if rewrite_usage:
        trace["tokens"]["rewrite"] = rewrite_usage
    started = time.perf_counter()
    searches: list[dict[str, Any]] = []
    retrieved = retrieve_for_api(standalone_query, papers, request.top_k or TOP_K, trace=searches)
    trace["timings_ms"]["retrieval"] = _ms(started)
    trace["retrieval"] = searches
    return standalone_query, retrieved, trace


def _assess(query: str, answer: str, retrieved: list[dict[str, Any]], trace: dict[str, Any]) -> dict:
    """Run the trust check, recording its time and tokens in the trace."""
    usage: dict[str, int] = {}
    started = time.perf_counter()
    result = assess(query, answer, retrieved,
                    ask=lambda prompt: call_llm(prompt, temperature=0.0, label="trust", usage=usage))
    trace["timings_ms"]["support_check"] = _ms(started)
    if usage:
        trace["tokens"]["support_check"] = usage
    return result


def _finish_trace(trace: dict[str, Any], request_started: float) -> dict[str, Any]:
    trace["timings_ms"]["total"] = _ms(request_started)
    trace["tokens"]["total"] = sum(part.get("total_tokens", 0) for part in trace["tokens"].values()
                                   if isinstance(part, dict))
    return trace


def _chat_prompt(query: str, retrieved: list[dict[str, Any]]) -> str:
    context = "\n\n".join(f"Source: {item['citation']}\n{item['text']}" for item in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=query)


NOT_FOUND_TRUST = {"label": "Low", "reason": "no passages were retrieved"}


def _citations(retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One entry per cited page: several passages can come from the same page."""
    unique = {item["citation"]: item["filename"] for item in retrieved}
    return [{"filename": filename, "citation": label} for label, filename in unique.items()]


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    request_started = time.perf_counter()
    standalone_query, retrieved, trace = _prepare_chat(request)
    if not retrieved:
        return {
            "answer": "Not found in the selected papers.",
            "citations": [], "context": [],
            "trust": NOT_FOUND_TRUST, "query_used": standalone_query,
            "trace": _finish_trace(trace, request_started),
        }
    usage: dict[str, int] = {}
    started = time.perf_counter()
    answer = call_llm(_chat_prompt(standalone_query, retrieved), temperature=0.2,
                      label="api_chat", usage=usage).strip()
    trace["timings_ms"]["generation"] = _ms(started)
    trace["tokens"]["generation"] = usage
    trust = _assess(standalone_query, answer, retrieved, trace)
    return {
        "answer": answer,
        "citations": _citations(retrieved),
        "context": retrieved,
        "trust": trust,
        "citation_check": check_citations(answer, retrieved, list_pdf_names()),
        "query_used": standalone_query,
        "trace": _finish_trace(trace, request_started),
    }


def _provider_error_detail(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "The AI service is rate limited. Wait a moment, then try again."
    if isinstance(exc, AuthenticationError):
        return "The AI service is not configured. Add a valid GROQ_API_KEY to .env."
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return "The AI service could not be reached. Check your internet connection."
    return "The answer could not be generated. Please try again."


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Newline-delimited JSON events: meta, then token*, then done (or error).

    Retrieval runs before the response starts, so its failures still become
    ordinary HTTP errors; only generation happens inside the stream.
    """
    request_started = time.perf_counter()
    standalone_query, retrieved, trace = _prepare_chat(request)

    def events():
        def line(payload: dict[str, Any]) -> str:
            return json.dumps(payload, ensure_ascii=False) + "\n"

        yield line({
            "type": "meta", "query_used": standalone_query,
            "citations": _citations(retrieved), "context": retrieved,
        })
        if not retrieved:
            yield line({"type": "done", "answer": "Not found in the selected papers.",
                        "trust": NOT_FOUND_TRUST, "trace": _finish_trace(trace, request_started)})
            return
        parts: list[str] = []
        usage: dict[str, int] = {}
        started = time.perf_counter()
        first_token_ms = None
        try:
            for piece in call_llm_stream(_chat_prompt(standalone_query, retrieved), temperature=0.2,
                                         label="api_chat", usage=usage):
                if first_token_ms is None:
                    first_token_ms = _ms(started)
                parts.append(piece)
                yield line({"type": "token", "text": piece})
        except Exception as exc:  # noqa: BLE001 - reported in-stream; headers are already sent
            yield line({"type": "error", "detail": _provider_error_detail(exc)})
            return
        trace["timings_ms"]["generation"] = _ms(started)
        trace["timings_ms"]["first_token"] = first_token_ms
        trace["tokens"]["generation"] = usage
        answer = "".join(parts).strip()
        trust = _assess(standalone_query, answer, retrieved, trace)
        yield line({
            "type": "done", "answer": answer,
            "trust": trust,
            "citation_check": check_citations(answer, retrieved, list_pdf_names()),
            "trace": _finish_trace(trace, request_started),
        })

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/retrieval/inspect")
def inspect_retrieval(request: InspectRequest) -> dict[str, Any]:
    """Run only retrieval and return every stage, for the retrieval inspector.

    No LLM call and no follow-up rewrite: the query is searched exactly as typed,
    so what you see is what the live pipeline ranks for that string.
    """
    available = set(list_pdf_names())
    unknown = [paper for paper in request.papers if paper not in available]
    if unknown:
        raise HTTPException(status_code=404, detail=f"PDF not found: {unknown[0]}")
    searches: list[dict[str, Any]] = []
    started = time.perf_counter()
    retrieved = retrieve_for_api(request.query.strip(), request.papers,
                                 request.top_k or TOP_K, trace=searches)
    return {"query": request.query.strip(), "retrieval_ms": _ms(started),
            "searches": searches, "selected": retrieved}


EVAL_RESULT_FILES = {
    "test": ("Held-out test (v2)", BASE_DIR / "eval" / "comparison_test_k5.json"),
    "dev": ("Dev (legacy, multi-page labels)", BASE_DIR / "eval" / "comparison_dev_k5.json"),
}
EVAL_METRICS = ("hit", "mrr", "ndcg", "precision", "recall")


@app.get("/api/eval/summary")
def eval_summary() -> dict[str, Any]:
    """Recorded retrieval benchmark results for the evaluation dashboard.

    Serves what eval/run_eval.py last MEASURED (with commit and timestamp); it
    never computes or estimates numbers itself. Sets that have not been run
    are reported as missing rather than filled in.
    """
    sets = []
    for key, (label, path) in EVAL_RESULT_FILES.items():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sets.append({"key": key, "label": label, "available": False})
            continue
        configs = []
        for name, block in data.get("configs", {}).items():
            k = block["config"]["top_k"]
            source = {"hit": f"hit@{k}", "mrr": "mrr@10", "ndcg": f"ndcg@{k}",
                      "precision": f"precision@{k}", "recall": f"recall@{k}"}
            configs.append({
                "name": name,
                "method": block["config"]["method"],
                "top_k": k,
                "metrics": {metric: {"mean": block[field]["mean"], "ci95": block[field]["ci95"],
                                     "count": block[field].get("count"), "n": block[field].get("n")}
                            for metric, field in source.items() if field in block},
                "vs_baseline": block.get("vs_baseline", {}),
                "latency_ms": block.get("latency_ms", {}),
            })
        sets.append({
            "key": key, "label": label, "available": True,
            "generated_at": data.get("generated_at"), "git_commit": data.get("git_commit"),
            "questions": data.get("n_answerable_scored"), "corpus": data.get("corpus", {}),
            "configs": configs,
        })
    live = "hybrid_rerank_top10" if RERANK_ENABLED else "hybrid_rrf60"
    return {"live_config": live, "sets": sets}


def _quiz_evidence(pdf: str, topic: str, difficulty: str, page_range: str | None, question_count: int):
    chunks, bm25 = get_hybrid_resources()
    try:
        return retrieve_quiz_chunks(
            get_vector_store(), bm25, chunks, pdf,
            topic.strip() or "main concepts",
            difficulty, page_range, question_count,
        )
    except ValueError as exc:  # invalid page range
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/quiz/generate")
def generate_api_quiz(request: QuizGenerateRequest) -> dict[str, Any]:
    if request.pdf not in list_pdf_names():
        raise HTTPException(status_code=404, detail="Selected PDF was not found")
    evidence = _quiz_evidence(
        request.pdf, request.topic, request.difficulty, request.page_range, request.question_count
    )
    try:
        quiz = generate_quiz(
            request.pdf, request.topic.strip() or "main concepts", evidence,
            difficulty=request.difficulty,
            question_count=request.question_count,
            question_type=request.question_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"The quiz could not be generated: {exc}") from exc
    return {
        "pdf": request.pdf,
        "topic": request.topic.strip(),
        **quiz,
        "context": [
            {"citation": citation(chunk.metadata), "text": chunk.page_content}
            for chunk in evidence
        ],
    }


@app.post("/api/quiz/grade")
def grade_api_quiz(request: QuizGradeRequest) -> dict[str, Any]:
    if request.pdf not in list_pdf_names():
        raise HTTPException(status_code=404, detail="Selected PDF was not found")
    if len(request.questions) != len(request.answers):
        raise HTTPException(status_code=422, detail="Questions and answers must have the same length")
    unanswered = [
        index + 1
        for index, answer in enumerate(request.answers)
        if not isinstance(answer, str) or not answer.strip()
    ]
    if unanswered:
        raise HTTPException(
            status_code=422,
            detail=f"Please answer all questions before grading ({len(unanswered)} unanswered)",
        )
    evidence = _quiz_evidence(
        request.pdf, request.topic, request.difficulty, request.page_range, request.question_count
    )
    try:
        return grade_quiz(request.questions, request.answers, build_quiz_context(evidence))
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Grading failed, please retry: {exc}") from exc


@app.post("/api/compare")
def compare_papers(request: CompareRequest) -> dict[str, Any]:
    pdf_names = request.pdfs
    available = set(list_pdf_names())
    missing = [name for name in pdf_names if name not in available]
    if missing:
        raise HTTPException(status_code=404, detail=f"PDF not found: {missing[0]}")
    if pdf_names[0] == pdf_names[1]:
        raise HTTPException(status_code=422, detail="Select two distinct PDF files")

    question = expand_compare_query(request.question.strip())
    chunks, bm25 = get_hybrid_resources()
    contexts = retrieve_paper_contexts(get_vector_store(), bm25, chunks, question, pdf_names)
    missing_context = [name for name in pdf_names if not contexts[name]]
    if missing_context:
        summary = "\n\n".join(
            f"## {name}\nNot found in {name} for this question. Try a more specific question."
            for name in missing_context
        )
        summary += (
            "\n\n## Difference\nNo difference summary can be grounded while a paper has no retrieved passages."
            "\n\n## Similarities\nNo evidence-supported similarity can be determined from empty evidence."
        )
    else:
        try:
            summary = generate_comparison(question, contexts)
        except RuntimeError as exc:
            if "empty response" in str(exc).lower():
                raise HTTPException(
                    status_code=502,
                    detail="The comparison returned no results. Try a broader question or another paper.",
                ) from exc
            raise HTTPException(status_code=502, detail="Comparison generation failed. Please try again.") from exc
    return {
        "summary": summary,
        "papers": {
            filename: [
                {"citation": citation(chunk.metadata), "text": chunk.page_content}
                for chunk in paper_chunks
            ]
            for filename, paper_chunks in contexts.items()
        },
    }


@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    # A plain `def`, not `async def`: parsing and embedding a PDF takes seconds
    # of CPU, and inside an async endpoint that work ran on the event loop and
    # froze every other request (including streaming chat) until it finished.
    # FastAPI runs sync endpoints in its thread pool instead.
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")
    file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDFs larger than 50 MB are not supported")
    try:
        chunk_size, chunk_overlap = best_chunk_config()
        count = add_pdf_to_library(
            filename, file_bytes, get_vector_store(), Path(PDF_DIR), chunk_size, chunk_overlap,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clear_indexes()
    # Rebuild BM25 and verify the dense/BM25 corpora immediately.
    get_hybrid_resources()
    return {"filename": filename, "chunks_indexed": count, "status": "indexed"}
