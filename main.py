"""Production-style FastAPI entry point for the Hybrid RAG application."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from groq import AuthenticationError, RateLimitError

from compare import (
    expand_compare_query,
    generate_comparison,
    retrieve_paper_contexts,
)
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_DIR,
    PROMPT_TEMPLATE,
    REMOVED_PDF_DIR,
    TOP_K,
    citation,
    source_filename,
    LLM_MODEL,
)
from evaluator import call_llm
from hybrid import build_bm25, hybrid_search_with_scores
from ingest import best_chunk_config, load_documents, split_documents
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

app = FastAPI(
    title="Hybrid RAG Knowledge Assistant",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0 and list_pdf_names():
        store.add_documents(split_documents(load_documents(), *best_chunk_config()))
    return store


@lru_cache(maxsize=1)
def get_hybrid_resources():
    store = get_vector_store()
    documents = load_documents()
    chunk_size, chunk_overlap = best_chunk_config()
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if store._collection.count() != len(chunks):
        raise RuntimeError(
            "The dense index and BM25 corpus do not match: "
            f"index={store._collection.count()} chunks, local corpus={len(chunks)} chunks"
        )
    return chunks, build_bm25(chunks)


def clear_indexes() -> None:
    get_hybrid_resources.cache_clear()
    get_vector_store.cache_clear()


def list_pdf_names() -> list[str]:
    return sorted(path.name for path in Path(PDF_DIR).glob("*.pdf"))


def retrieve_for_api(
    query: str,
    papers: list[str] | None = None,
) -> list[dict[str, Any]]:
    chunks, bm25 = get_hybrid_resources()
    selected_chunks = chunks
    if papers:
        selected_chunks = [
            chunk for chunk in chunks
            if source_filename(chunk.metadata) in papers
        ]
        if not selected_chunks:
            return []
    retrieved_chunks = []
    for paper in (papers or [None]):
        metadata_filter = {"filename": paper} if paper else None
        paper_chunks = selected_chunks if paper else chunks
        paper_bm25 = build_bm25(paper_chunks) if paper else bm25
        results = hybrid_search_with_scores(
            query,
            get_vector_store(),
            paper_bm25,
            paper_chunks,
            n_results=TOP_K,
            metadata_filter=metadata_filter,
        )
        retrieved_chunks.extend(results)
    # Per-paper retrieval may repeat the same default n_results; deduplicate
    # by citation while preserving rank order.
    results = []
    seen = set()
    for document, score in retrieved_chunks:
        key = citation(document.metadata)
        if key not in seen:
            seen.add(key)
            results.append((document, score))
    return [
        {
            "filename": source_filename(document.metadata),
            "citation": citation(document.metadata),
            "text": document.page_content,
            "score": round(score, 4) if score is not None else None,
        }
        for document, score in results
    ]


def rewrite_followup(query: str, history: list[dict[str, str]]) -> str:
    """Make a short contextual follow-up suitable for retrieval without changing its meaning."""
    if not history:
        return query
    recent = history[-3:]
    words = set(re.findall(r"[a-z]+", query.lower()))
    if not words.intersection({"it", "they", "this", "that", "those", "them"}):
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


@app.get("/api/chat/starters")
def chat_starters() -> dict[str, list[str]]:
    chunks, _ = get_hybrid_resources()
    starters: list[str] = []
    for filename in list_pdf_names():
        paper_chunk = next(
            (chunk for chunk in chunks if source_filename(chunk.metadata) == filename),
            None,
        )
        if paper_chunk:
            text = " ".join(paper_chunk.page_content.split())
            starters.append(f"What does {filename} say about {text[:55].rstrip(' .,')}?" )
    generic = [
        "Summarize the main contribution of each paper.",
        "Compare the methodologies used in the papers.",
        "What limitations are discussed?",
        "What datasets or metrics are reported?",
    ]
    return {"starters": (starters + generic)[:4]}


@app.post("/api/chat/feedback")
def save_chat_feedback(request: ChatFeedbackRequest) -> dict[str, bool]:
    if request.feedback not in {"up", "down"}:
        raise HTTPException(status_code=422, detail="Feedback must be up or down")
    feedback_path = Path(__file__).resolve().parent / "chat_feedback.json"
    entries: list[dict[str, Any]] = []
    if feedback_path.exists():
        try:
            entries = json.loads(feedback_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append({
        "query": request.query,
        "message": request.message,
        "feedback": request.feedback,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    feedback_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    papers: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)


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


class CompareRequest(BaseModel):
    pdfs: list[str] = Field(min_length=2, max_length=2)
    question: str = Field(min_length=1)


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(INDEX_FILE)


@app.get("/api/pdfs")
def get_pdfs() -> dict[str, list[str]]:
    return {"pdfs": list_pdf_names()}


@app.get("/api/settings")
def get_app_settings() -> dict[str, Any]:
    return {"top_k": TOP_K, "model": LLM_MODEL}


QUIZ_HISTORY_PATH = Path(__file__).resolve().parent / "quiz_history.json"


@app.post("/api/progress/quiz")
def save_quiz_progress(request: QuizHistoryRequest) -> dict[str, bool]:
    history: list[dict[str, Any]] = []
    if QUIZ_HISTORY_PATH.exists():
        try:
            history = json.loads(QUIZ_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    history.append({
        "pdf": request.pdf,
        "difficulty": request.difficulty,
        "score": request.score,
        "topics": request.topics,
        "date": datetime.now(timezone.utc).isoformat(),
    })
    QUIZ_HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True}


@app.get("/api/progress")
def get_progress() -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    if QUIZ_HISTORY_PATH.exists():
        try:
            history = json.loads(QUIZ_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    averages: dict[str, float] = {}
    for difficulty in ("Easy", "Medium", "Hard", "Mixed"):
        scores = [
            float(item["score"]) for item in history
            if item.get("difficulty") == difficulty
        ]
        if scores:
            averages[difficulty] = round(sum(scores) / len(scores), 1)
    topic_counts: dict[str, int] = {}
    for item in history:
        for topic in item.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    missed = sorted(topic_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    return {
        "quizzes_taken": len(history),
        "averages": averages,
        "most_missed_topics": [topic for topic, _ in missed],
    }


@app.get("/api/library")
def get_library() -> dict[str, Any]:
    chunks, _ = get_hybrid_resources()
    papers = []
    for filename in list_pdf_names():
        paper_chunks = [
            chunk for chunk in chunks
            if source_filename(chunk.metadata) == filename
        ]
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
        })
    return {"papers": papers}


@app.get("/api/library/{filename:path}/preview")
def preview_pdf(filename: str) -> dict[str, Any]:
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(PDF_DIR) / Path(filename).name
    if Path(filename).name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="PDF preview could not be parsed") from exc
    first = pages[0].page_content.strip() if pages else ""
    return {
        "filename": filename,
        "pages": len(pages),
        "preview": first[:1200],
        "metadata": {key: value for key, value in (pages[0].metadata.items() if pages else []) if isinstance(value, (str, int, float, bool))},
    }


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
        rename_pdf_in_library(old, new, Path(PDF_DIR))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = get_vector_store()
    store.delete(where={"filename": old})
    chunks = split_documents(load_documents(), *best_chunk_config())
    matching = [
        chunk for chunk in chunks
        if source_filename(chunk.metadata) == new
    ]
    store.add_documents(matching)
    clear_indexes()
    get_hybrid_resources()
    return {"filename": new}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    query = request.query.strip()
    papers = [paper for paper in request.papers if paper in list_pdf_names()]
    standalone_query = rewrite_followup(query, request.history)
    retrieved = retrieve_for_api(standalone_query, papers)
    if not retrieved:
        return {
            "answer": "Not found in the selected papers.",
            "citations": [],
            "context": [],
            "trust": {"label": "Low", "heuristic": True},
            "query_used": standalone_query,
        }
    context = "\n\n".join(
        f"Source: {item['citation']}\n{item['text']}" for item in retrieved
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    answer = call_llm(prompt, temperature=0.2, label="api_chat").strip()
    trust = assess(query, answer, retrieved)
    return {
        "answer": answer,
        "citations": [
            {"filename": item["filename"], "citation": item["citation"]}
            for item in retrieved
        ],
        "context": retrieved,
        "trust": trust,
        "query_used": standalone_query,
    }


@app.post("/api/quiz/generate")
def generate_api_quiz(request: QuizGenerateRequest) -> dict[str, Any]:
    if request.pdf not in list_pdf_names():
        raise HTTPException(status_code=404, detail="Selected PDF was not found")
    chunks, bm25 = get_hybrid_resources()
    evidence = retrieve_quiz_chunks(
        get_vector_store(), bm25, chunks, request.pdf,
        request.topic.strip() or "main concepts",
        request.difficulty, request.page_range,
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
            {
                "citation": citation(chunk.metadata),
                "text": chunk.page_content,
            }
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
        count = len(unanswered)
        raise HTTPException(
            status_code=422,
            detail=f"Please answer all questions before grading ({count} unanswered)",
        )
    chunks, bm25 = get_hybrid_resources()
    evidence = retrieve_quiz_chunks(
        get_vector_store(), bm25, chunks, request.pdf,
        request.topic.strip() or "main concepts",
        page_range=request.page_range,
    )
    context = build_quiz_context(evidence)
    result = grade_quiz(request.questions, request.answers, context)
    return result


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
    contexts = retrieve_paper_contexts(
        get_vector_store(), bm25, chunks, question, pdf_names
    )
    missing_context = [name for name in pdf_names if not contexts[name]]
    if missing_context:
        summary = "\n\n".join(
            f"## {name}\nNot found in {name} for this question. Try a more specific question."
            for name in missing_context
        )
        if len(missing_context) != len(pdf_names):
            summary += "\n\n## Difference\nNo difference summary can be grounded while a paper has no retrieved passages."
            summary += "\n\n## Similarities\nNo evidence-supported similarity can be determined from empty evidence."
        else:
            summary += "\n\n## Difference\nNo difference summary can be generated without retrieved evidence.\n\n## Similarities\nNo evidence-supported similarity can be determined from empty evidence."
    try:
        if not missing_context:
            summary = generate_comparison(question, contexts)
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The AI service is rate limited. Wait a moment, then try again.",
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI service is not configured. Add GROQ_API_KEY to .env.",
        ) from exc
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
                {
                    "citation": citation(chunk.metadata),
                    "text": chunk.page_content,
                }
                for chunk in paper_chunks
            ]
            for filename, paper_chunks in contexts.items()
        },
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")
    file_bytes = await file.read()
    try:
        chunk_size, chunk_overlap = best_chunk_config()
        count = add_pdf_to_library(
            filename,
            file_bytes,
            get_vector_store(),
            Path(PDF_DIR),
            chunk_size,
            chunk_overlap,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clear_indexes()
    # Rebuild BM25 and verify the dense/BM25 corpora immediately.
    get_hybrid_resources()
    return {"filename": filename, "chunks_indexed": count, "status": "indexed"}
