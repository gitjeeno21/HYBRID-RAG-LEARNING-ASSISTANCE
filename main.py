"""Production-style FastAPI entry point for the Hybrid RAG application."""

from __future__ import annotations

import json
import logging
import secrets
import os
import time
from pathlib import Path
from typing import Any

import re

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from groq import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import BaseModel, Field

import auth
import db
import quota
from citations import check_citations
from compare import (
    expand_compare_query,
    generate_comparison,
    retrieve_paper_contexts,
)
from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PROMPT_TEMPLATE,
    RERANK_ENABLED,
    TOP_K,
    citation,
    source_filename,
)
from llm import call_llm, call_llm_stream, track_usage
from hybrid import hybrid_search_with_scores, unique_chunks
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
from workspace import Workspace, load_pdf_pages, verify_index_matches  # noqa: F401

load_dotenv()

STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TOP_K = 10

app = FastAPI(title="Hybrid RAG Knowledge Assistant", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request models. These must be defined before the routes that use them:
# with `from __future__ import annotations`, FastAPI resolves annotations by
# name at registration time, and an undefined model silently becomes a
# required query parameter (which made every feedback request fail with 422).
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD_LENGTH)
    name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD_LENGTH)


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
# Authentication: every /api route requires a signed-in user, by default
# ---------------------------------------------------------------------------
# WHY a middleware instead of a dependency on each route: secure by default.
# A route added later is protected without anyone remembering to protect it;
# only the endpoints below, which create or end a session, are open.
OPEN_API_PATHS = {"/api/auth/register", "/api/auth/login", "/api/auth/logout"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Cookies are sent over plain http on localhost; set APP_SECURE_COOKIES=1 when
# the app is served over HTTPS so browsers never send them unencrypted.
SECURE_COOKIES = os.getenv("APP_SECURE_COOKIES", "0").strip().lower() in {"1", "true", "yes", "on"}


def _same_origin(request: Request) -> bool:
    """Reject cross-site writes to the open auth endpoints (login CSRF).

    Browsers always send Origin on cross-origin POSTs; requests without it
    (curl, tests) are not cross-site browser requests.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    return origin.rstrip("/") == f"{request.url.scheme}://{request.url.netloc}"


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)
    unsafe = request.method not in SAFE_METHODS
    if path in OPEN_API_PATHS:
        if unsafe and not _same_origin(request):
            return JSONResponse(status_code=403, content={"detail": "Cross-site request blocked."})
        return await call_next(request)
    resolved = await run_in_threadpool(auth.resolve_session, request.cookies.get(auth.SESSION_COOKIE))
    if resolved is None:
        return JSONResponse(status_code=401, content={"detail": "Please sign in."})
    user, csrf_token = resolved
    if unsafe and not auth.csrf_matches(csrf_token, request.headers.get(auth.CSRF_HEADER)):
        return JSONResponse(status_code=403,
                            content={"detail": "Security check failed. Reload the page and try again."})
    request.state.user = user
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request logging and security headers (outermost middleware: defined last)
# ---------------------------------------------------------------------------
access_log = logging.getLogger("rag.access")
if not access_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    access_log.addHandler(_handler)
    access_log.setLevel(logging.INFO)
    access_log.propagate = False

_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    # Scripts and data only from this origin (inline scripts are the two
    # small ones in index.html); fonts from Google Fonts; no framing.
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    ),
}


def current_user(request: Request) -> auth.User:
    """The signed-in user (set by require_login for every protected route)."""
    return request.state.user


def _start_session(response: Response, user: auth.User) -> dict[str, Any]:
    session = auth.create_session(user)
    max_age = auth.SESSION_DAYS * 24 * 3600
    response.set_cookie(auth.SESSION_COOKIE, session.token, max_age=max_age, httponly=True,
                        samesite="lax", secure=SECURE_COOKIES, path="/")
    # Readable by the page on purpose: it is echoed back in X-CSRF-Token.
    response.set_cookie(auth.CSRF_COOKIE, session.csrf_token, max_age=max_age, httponly=False,
                        samesite="lax", secure=SECURE_COOKIES, path="/")
    return {"user": user.public()}


@app.middleware("http")
async def log_and_secure(request: Request, call_next):
    """One JSON log line per API request, plus a request id and security headers.

    Logged: method, path, status, duration, user id, request id. Never logged:
    bodies, query strings, cookies or headers (questions and passwords live there).
    """
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if _REQUEST_ID.fullmatch(supplied) else secrets.token_hex(8)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        if request.url.path.startswith("/api/") or status >= 500:
            user = getattr(request.state, "user", None)
            access_log.info(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": status, "ms": round((time.perf_counter() - started) * 1e3, 1),
                "user_id": user.id if user else None,
            }))
    response.headers["X-Request-ID"] = request_id
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


@app.get("/healthz", include_in_schema=False)
def liveness() -> dict[str, str]:
    """Public liveness probe for containers: reveals nothing about any user."""
    return {"status": "ok"}


@app.exception_handler(quota.QuotaExceeded)
async def _quota_exceeded(_: Request, exc: quota.QuotaExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.get("/api/usage")
def get_usage(http: Request) -> dict[str, Any]:
    """Today's LLM token use against the daily allowance."""
    return quota.usage(current_user(http).id)


@app.exception_handler(auth.AuthError)
async def _auth_error(_: Request, exc: auth.AuthError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


@app.post("/api/auth/register")
def register(payload: RegisterRequest, response: Response) -> dict[str, Any]:
    user = auth.create_user(payload.email, payload.password, payload.name)
    return _start_session(response, user)


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    client = request.client.host if request.client else ""
    user = auth.authenticate(payload.email, payload.password, client=client)
    return _start_session(response, user)


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    auth.revoke_session(request.cookies.get(auth.SESSION_COOKIE))
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    response.delete_cookie(auth.CSRF_COOKIE, path="/")
    return {"signed_out": True}


@app.get("/api/auth/me")
def me(request: Request) -> dict[str, Any]:
    return {"user": current_user(request).public()}


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
def workspace_for(http: Request) -> Workspace:
    """The signed-in user's library. Every data endpoint goes through this."""
    return Workspace(current_user(http).id)


# Thin per-workspace accessors, kept as module functions so tests can patch them.
def list_pdf_names(ws: Workspace) -> list[str]:
    return ws.pdf_names()


def get_vector_store(ws: Workspace):
    return ws.store


def get_hybrid_resources(ws: Workspace):
    return ws.resources()


def _llm_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key) and key != "your_groq_api_key_here"


def retrieve_for_api(
    ws: Workspace,
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
    chunks, bm25 = get_hybrid_resources(ws)
    if not chunks:
        return []
    store = get_vector_store(ws)
    retrieved = []
    # _hybrid_search rebuilds BM25 over the filtered subset itself, so the
    # global index is passed through unchanged for per-paper searches.
    for paper in (papers or [None]):
        search_trace: dict[str, Any] | None = {} if trace is not None else None
        retrieved.extend(hybrid_search_with_scores(
            query,
            store,
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
def health(http: Request) -> dict[str, Any]:
    """Real status for the header indicator instead of a hard-coded label."""
    ws = workspace_for(http)
    try:
        chunks, _ = get_hybrid_resources(ws)
        index_error = None
    except Exception as exc:  # noqa: BLE001 - reported to the UI, not raised
        chunks, index_error = [], str(exc)
    return {
        # An empty library is a ready (if empty) index, not an error.
        "index_ready": index_error is None,
        "index_error": index_error,
        "papers": len(list_pdf_names(ws)),
        "chunks": len(chunks),
        "llm_configured": _llm_configured(),
    }


@app.get("/api/pdfs")
def get_pdfs(http: Request) -> dict[str, list[str]]:
    return {"pdfs": list_pdf_names(workspace_for(http))}


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
def chat_starters(http: Request) -> dict[str, list[str]]:
    names = list_pdf_names(workspace_for(http))
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
def save_chat_feedback(request: ChatFeedbackRequest, http: Request) -> dict[str, bool]:
    if request.feedback not in {"up", "down"}:
        raise HTTPException(status_code=422, detail="Feedback must be up or down")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO chat_feedback (user_id, query, message, feedback) VALUES (?, ?, ?, ?)",
            (current_user(http).id, request.query, request.message, request.feedback),
        )
    return {"saved": True}


@app.post("/api/progress/quiz")
def save_quiz_progress(request: QuizHistoryRequest, http: Request) -> dict[str, bool]:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO quiz_history (user_id, pdf, difficulty, score, topics) VALUES (?, ?, ?, ?, ?)",
            (current_user(http).id, request.pdf, request.difficulty, request.score,
             json.dumps(request.topics, ensure_ascii=False)),
        )
    return {"saved": True}


def _quiz_history(user_id: int) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT pdf, difficulty, score, topics, created_at FROM quiz_history "
            "WHERE user_id = ? ORDER BY id", (user_id,),
        ).fetchall()
    return [{"pdf": r["pdf"], "difficulty": r["difficulty"], "score": r["score"],
             "topics": json.loads(r["topics"] or "[]"), "date": r["created_at"]} for r in rows]


@app.get("/api/progress")
def get_progress(http: Request) -> dict[str, Any]:
    history = _quiz_history(current_user(http).id)
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
def get_library(http: Request) -> dict[str, Any]:
    ws = workspace_for(http)
    chunks, _ = get_hybrid_resources(ws)
    papers = []
    for filename in list_pdf_names(ws):
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
            "size_kb": round((ws.pdf_dir / filename).stat().st_size / 1024),
        })
    return {"papers": papers}


@app.get("/api/library/{filename:path}/preview")
def preview_pdf(filename: str, http: Request) -> dict[str, Any]:
    path = workspace_for(http).pdf_path(filename)
    if path is None:
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
def open_pdf(filename: str, http: Request) -> FileResponse:
    path = workspace_for(http).pdf_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf")


@app.delete("/api/library/{filename:path}")
def delete_pdf(filename: str, http: Request) -> dict[str, Any]:
    if Path(filename).name != filename:
        raise HTTPException(status_code=422, detail="Invalid filename")
    ws = workspace_for(http)
    try:
        remove_pdf_from_library(filename, get_vector_store(ws), ws.pdf_dir, ws.removed_dir, keep_copy=False)
        ws.invalidate()
        get_hybrid_resources(ws)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"deleted": filename}


@app.post("/api/library/rename")
def rename_pdf(request: RenamePdfRequest, http: Request) -> dict[str, Any]:
    ws = workspace_for(http)
    old = Path(request.filename).name
    new = Path(request.new_filename).name
    if old != request.filename or new != request.new_filename:
        raise HTTPException(status_code=422, detail="Use a simple PDF filename")
    if not new.lower().endswith(".pdf"):
        new += ".pdf"
    if ws.pdf_path(old) is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    if (ws.pdf_dir / new).exists():
        raise HTTPException(status_code=409, detail="That filename already exists")
    try:
        new_path = rename_pdf_in_library(old, new, ws.pdf_dir)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Re-index only the renamed paper instead of re-parsing the whole library.
    pages = load_pdf_pages(new_path)
    store = get_vector_store(ws)
    store.delete(where={"filename": old})
    store.add_documents(split_documents(pages, *best_chunk_config()))
    ws.invalidate()
    get_hybrid_resources(ws)
    return {"filename": new}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1e3, 1)


def _prepare_chat(request: ChatRequest, ws: Workspace) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Shared by /api/chat and /api/chat/stream: rewrite, then retrieve.

    Also returns the start of the pipeline trace: how long each step took, the
    tokens the follow-up rewrite used, and what every retrieval stage did.
    """
    available = set(list_pdf_names(ws))
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
    retrieved = retrieve_for_api(ws, standalone_query, papers, request.top_k or TOP_K, trace=searches)
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
def chat(request: ChatRequest, http: Request) -> dict[str, Any]:
    request_started = time.perf_counter()
    user_id = current_user(http).id
    quota.check(user_id)
    ws = workspace_for(http)
    standalone_query, retrieved, trace = _prepare_chat(request, ws)
    if not retrieved:
        quota.record(user_id, _finish_trace(trace, request_started)["tokens"]["total"])
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
    finished = _finish_trace(trace, request_started)
    quota.record(user_id, finished["tokens"]["total"])
    return {
        "answer": answer,
        "citations": _citations(retrieved),
        "context": retrieved,
        "trust": trust,
        "citation_check": check_citations(answer, retrieved, list_pdf_names(ws)),
        "query_used": standalone_query,
        "trace": finished,
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
def chat_stream(request: ChatRequest, http: Request) -> StreamingResponse:
    """Newline-delimited JSON events: meta, then token*, then done (or error).

    Retrieval runs before the response starts, so its failures still become
    ordinary HTTP errors; only generation happens inside the stream.
    """
    request_started = time.perf_counter()
    user_id = current_user(http).id
    quota.check(user_id)   # before the stream starts, so it is an ordinary 429
    ws = workspace_for(http)
    standalone_query, retrieved, trace = _prepare_chat(request, ws)

    def events():
        def line(payload: dict[str, Any]) -> str:
            return json.dumps(payload, ensure_ascii=False) + "\n"

        yield line({
            "type": "meta", "query_used": standalone_query,
            "citations": _citations(retrieved), "context": retrieved,
        })
        if not retrieved:
            finished = _finish_trace(trace, request_started)
            quota.record(user_id, finished["tokens"]["total"])
            yield line({"type": "done", "answer": "Not found in the selected papers.",
                        "trust": NOT_FOUND_TRUST, "trace": finished})
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
            trace["tokens"]["generation"] = usage
            quota.record(user_id, _finish_trace(trace, request_started)["tokens"]["total"])
            yield line({"type": "error", "detail": _provider_error_detail(exc)})
            return
        trace["timings_ms"]["generation"] = _ms(started)
        trace["timings_ms"]["first_token"] = first_token_ms
        trace["tokens"]["generation"] = usage
        answer = "".join(parts).strip()
        trust = _assess(standalone_query, answer, retrieved, trace)
        finished = _finish_trace(trace, request_started)
        quota.record(user_id, finished["tokens"]["total"])
        yield line({
            "type": "done", "answer": answer,
            "trust": trust,
            "citation_check": check_citations(answer, retrieved, list_pdf_names(ws)),
            "trace": finished,
        })

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/retrieval/inspect")
def inspect_retrieval(request: InspectRequest, http: Request) -> dict[str, Any]:
    """Run only retrieval and return every stage, for the retrieval inspector.

    No LLM call and no follow-up rewrite: the query is searched exactly as typed,
    so what you see is what the live pipeline ranks for that string.
    """
    ws = workspace_for(http)
    available = set(list_pdf_names(ws))
    unknown = [paper for paper in request.papers if paper not in available]
    if unknown:
        raise HTTPException(status_code=404, detail=f"PDF not found: {unknown[0]}")
    searches: list[dict[str, Any]] = []
    started = time.perf_counter()
    retrieved = retrieve_for_api(ws, request.query.strip(), request.papers,
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


def _quiz_evidence(ws: Workspace, pdf: str, topic: str, difficulty: str, page_range: str | None,
                   question_count: int):
    chunks, bm25 = get_hybrid_resources(ws)
    try:
        return retrieve_quiz_chunks(
            get_vector_store(ws), bm25, chunks, pdf,
            topic.strip() or "main concepts",
            difficulty, page_range, question_count,
        )
    except ValueError as exc:  # invalid page range
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/quiz/generate")
def generate_api_quiz(request: QuizGenerateRequest, http: Request) -> dict[str, Any]:
    ws = workspace_for(http)
    if request.pdf not in list_pdf_names(ws):
        raise HTTPException(status_code=404, detail="Selected PDF was not found")
    quota.check(current_user(http).id)
    evidence = _quiz_evidence(
        ws, request.pdf, request.topic, request.difficulty, request.page_range, request.question_count
    )
    try:
        with track_usage() as used:
            try:
                quiz = generate_quiz(
                    request.pdf, request.topic.strip() or "main concepts", evidence,
                    difficulty=request.difficulty,
                    question_count=request.question_count,
                    question_type=request.question_type,
                )
            finally:
                quota.record(current_user(http).id, used.get("total_tokens", 0))
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
def grade_api_quiz(request: QuizGradeRequest, http: Request) -> dict[str, Any]:
    ws = workspace_for(http)
    if request.pdf not in list_pdf_names(ws):
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
        ws, request.pdf, request.topic, request.difficulty, request.page_range, request.question_count
    )
    quota.check(current_user(http).id)
    try:
        with track_usage() as used:
            try:
                return grade_quiz(request.questions, request.answers, build_quiz_context(evidence))
            finally:
                if used:   # choice-only quizzes are graded without the LLM
                    quota.record(current_user(http).id, used.get("total_tokens", 0))
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Grading failed, please retry: {exc}") from exc


@app.post("/api/compare")
def compare_papers(request: CompareRequest, http: Request) -> dict[str, Any]:
    ws = workspace_for(http)
    pdf_names = request.pdfs
    available = set(list_pdf_names(ws))
    missing = [name for name in pdf_names if name not in available]
    if missing:
        raise HTTPException(status_code=404, detail=f"PDF not found: {missing[0]}")
    if pdf_names[0] == pdf_names[1]:
        raise HTTPException(status_code=422, detail="Select two distinct PDF files")

    question = expand_compare_query(request.question.strip())
    chunks, bm25 = get_hybrid_resources(ws)
    contexts = retrieve_paper_contexts(get_vector_store(ws), bm25, chunks, question, pdf_names)
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
        quota.check(current_user(http).id)
        try:
            with track_usage() as used:
                try:
                    summary = generate_comparison(question, contexts)
                finally:
                    quota.record(current_user(http).id, used.get("total_tokens", 0))
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
def upload_pdf(http: Request, file: UploadFile = File(...)) -> dict[str, Any]:
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
    ws = workspace_for(http)
    try:
        chunk_size, chunk_overlap = best_chunk_config()
        count = add_pdf_to_library(
            filename, file_bytes, get_vector_store(ws), ws.pdf_dir, chunk_size, chunk_overlap,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ws.invalidate()
    # Rebuild BM25 and verify the dense/BM25 corpora immediately.
    get_hybrid_resources(ws)
    return {"filename": filename, "chunks_indexed": count, "status": "indexed"}
