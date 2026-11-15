"""Per-user libraries: files, index scope and retrieval resources.

Isolation model (each user sees and searches only their own papers):

- Files: every user has a folder, storage/users/<id>/pdfs. Paths are always
  built from the signed-in user's id and a validated bare filename.
- Dense index: one shared Chroma collection; every chunk carries `user_id`.
  All access goes through ScopedStore, which adds the user's filter to EVERY
  search, read, write and delete. Callers (chat, quiz, compare, upload
  rollback, delete, rename) cannot forget the filter because they never see
  the unscoped store.
- Lexical index: a BM25 index per user, built from that user's chunks only.
- Defence in depth: hybrid fusion maps dense hits onto the user's own chunk
  list, so even a dense hit that escaped the filter could not be returned.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import BASE_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from hybrid import build_bm25, chunk_keys
from ingest import best_chunk_config, split_documents

STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", str(BASE_DIR / "storage")))
CACHED_WORKSPACES = 16           # BM25 indexes kept in memory (least recently used dropped)

_store_lock = threading.RLock()  # Chroma client creation is not thread-safe
_resource_lock = threading.RLock()
_resources: OrderedDict[int, tuple[list[Document], Any]] = OrderedDict()


def use_storage(path: Path | str) -> None:
    """Point user files and the shared index at `path` (tests use a temp dir)."""
    global STORAGE_DIR
    STORAGE_DIR = Path(path)
    with _store_lock:
        _shared_store.cache_clear()
    with _resource_lock:
        _resources.clear()


# ---------------------------------------------------------------------------
# PDF parsing (cached by path + mtime + size; an edited file is parsed again)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _parse_pdf(path: str, mtime_ns: int, size: int) -> tuple:
    return tuple(PyPDFLoader(path).load())


def load_pdf_pages(path: Path) -> list[Document]:
    """Fresh copies of a PDF's pages, tagged with the bare filename."""
    stat = path.stat()
    pages = []
    for page in _parse_pdf(str(path), stat.st_mtime_ns, stat.st_size):
        copy = page.model_copy(deep=True)
        copy.metadata["filename"] = path.name
        copy.metadata["source"] = path.name   # never expose the server path
        pages.append(copy)
    return pages


# ---------------------------------------------------------------------------
# Shared dense index, always accessed through ScopedStore
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _shared_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(STORAGE_DIR / "chroma"),
    )


def scoped(scope: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Combine the user scope with an optional caller filter (Chroma `where`)."""
    if not extra:
        return dict(scope)
    clauses = [dict(scope)]
    clauses += [extra] if any(key.startswith("$") for key in extra) else [{k: v} for k, v in extra.items()]
    return {"$and": clauses}


class ScopedStore:
    """A view of the shared Chroma collection limited to one user's chunks."""

    def __init__(self, store: Chroma, user_id: int):
        self._store = store
        self._scope = {"user_id": int(user_id)}
        self.user_id = int(user_id)

    def similarity_search(self, query: str, k: int, filter: dict | None = None) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=scoped(self._scope, filter))

    def similarity_search_with_score(self, query: str, k: int, filter: dict | None = None):
        return self._store.similarity_search_with_score(query, k=k, filter=scoped(self._scope, filter))

    def add_documents(self, documents: list[Document]) -> list[str]:
        stamped = [Document(page_content=d.page_content, metadata={**d.metadata, **self._scope})
                   for d in documents]
        return self._store.add_documents(stamped)

    def delete(self, where: dict | None = None) -> None:
        self._store.delete(where=scoped(self._scope, where))

    def get(self, include: list[str] | None = None, where: dict | None = None) -> dict:
        return self._store.get(where=scoped(self._scope, where), include=include or ["metadatas"])


# ---------------------------------------------------------------------------
# One user's library
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, user_id: int):
        self.user_id = int(user_id)
        self.root = STORAGE_DIR / "users" / str(self.user_id)
        self.pdf_dir = self.root / "pdfs"
        self.removed_dir = self.root / "removed"

    @property
    def store(self) -> ScopedStore:
        with _store_lock:
            return ScopedStore(_shared_store(), self.user_id)

    def pdf_names(self) -> list[str]:
        if not self.pdf_dir.is_dir():
            return []
        return sorted(path.name for path in self.pdf_dir.glob("*.pdf"))

    def pdf_path(self, filename: str) -> Path | None:
        """The path of one of this user's PDFs, or None (also for unsafe names)."""
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            return None
        path = self.pdf_dir / filename
        return path if path.is_file() else None

    def pages(self) -> list[Document]:
        return [page for name in self.pdf_names() for page in load_pdf_pages(self.pdf_dir / name)]

    def resources(self) -> tuple[list[Document], Any]:
        """(chunks, BM25) for this user, built once and verified against Chroma."""
        with _resource_lock:
            if self.user_id in _resources:
                _resources.move_to_end(self.user_id)
                return _resources[self.user_id]
            chunks = split_documents(self.pages(), *best_chunk_config()) if self.pdf_names() else []
            if chunks:
                verify_index_matches(self.store, chunks)
            built = (chunks, build_bm25(chunks) if chunks else None)
            _resources[self.user_id] = built
            while len(_resources) > CACHED_WORKSPACES:
                _resources.popitem(last=False)
            return built

    def invalidate(self) -> None:
        with _resource_lock:
            _resources.pop(self.user_id, None)

    def reindex(self) -> int:
        """Rebuild this user's dense chunks from their files; returns the chunk count."""
        store = self.store
        store.delete()
        chunks = split_documents(self.pages(), *best_chunk_config()) if self.pdf_names() else []
        if chunks:
            store.add_documents(chunks)
        self.invalidate()
        return len(chunks)


def verify_index_matches(store, chunks: list[Document]) -> None:
    """Fail loudly unless the (scoped) dense index holds exactly the chunks BM25 will search.

    WHY compare contents, not counts: equal counts can hide different chunks,
    e.g. after changing RAG_CROSS_PAGE or the chunk size without re-indexing.
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
            "Run `python manage.py reindex` to rebuild it."
        )
