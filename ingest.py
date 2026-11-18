"""Ingest every PDF in data/ into a local Chroma vector store.

Pipeline: scan data/*.pdf -> load pages and tag each chunk with its source
filename -> split into chunks (size/overlap from eval_results.json if present,
otherwise config.py defaults) -> embed with all-MiniLM-L6-v2 -> persist to
./chroma_db, which is rebuilt from scratch on every run.

WHY eval_results.json takes priority: evaluator.py measures the chunk size and
overlap that maximise both retrieval hit rate and LLM-judge score across the
real corpus. Once that experiment has run, using its winning numbers here means
the production index is built with the empirically best settings, not an
arbitrary default.
"""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# All settings come from one place so ingest.py and the query side can never
# disagree about the collection name or chunk size.
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    CROSS_PAGE_CHUNKS,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    PDF_DIR,
)


def best_chunk_config() -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) from eval_results.json, or config.py defaults.

    WHY this function exists: evaluator.py writes 'best_config' into
    eval_results.json after it finds the winning chunking parameters. Reading
    that file here means a single evaluation run automatically propagates its
    result to every future ingest — no manual config edit needed.

    If the file is missing (first run before evaluation) or malformed, we fall
    back to the CHUNK_SIZE / CHUNK_OVERLAP constants in config.py so the project
    always works out of the box.
    """
    try:
        data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        best = data.get("best_config", {})
        size = best.get("chunk_size")
        overlap = best.get("chunk_overlap")
        # Both values must be positive integers; reject partial or zero entries.
        if isinstance(size, int) and isinstance(overlap, int) and size > 0:
            return size, overlap
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # File absent or corrupt: silently fall back so the app still starts.
        pass
    return CHUNK_SIZE, CHUNK_OVERLAP


def scan_pdfs() -> list[Path]:
    """Return every .pdf file in the data/ folder, sorted by name."""
    return sorted(PDF_DIR.glob("*.pdf"))


def load_documents() -> list:
    """Load all scanned PDFs and tag every page with its source filename."""
    pdf_files = scan_pdfs()
    if not pdf_files:
        raise FileNotFoundError(f"No .pdf files found in {PDF_DIR}/")

    documents = []
    for pdf_path in pdf_files:
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            # Store the bare filename (not the full path) so citations are
            # short and stable regardless of where the project lives.
            page.metadata["filename"] = pdf_path.name
        documents.extend(pages)
        print(f"Loaded {len(pages)} page(s) from {pdf_path.name}")
    return documents


def split_documents(documents: list, chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list:
    """Split pages into chunks; metadata (filename, page) is preserved.

    chunk_size and chunk_overlap default to best_chunk_config() when not
    supplied, which reads eval_results.json if it exists.

    WHY pass them as arguments rather than always reading best_chunk_config():
    evaluator.py calls this function with explicit sizes (one per grid cell) so
    it can sweep the parameter space without mutating the module-level defaults.
    Callers that do NOT pass arguments (main.py, main()) get the best evaluated
    values automatically.

    RecursiveCharacterTextSplitter tries paragraph, then line, then sentence
    boundaries before cutting mid-word, which keeps chunks readable and
    therefore more useful as LLM context.
    """
    if chunk_size is None or chunk_overlap is None:
        chunk_size, chunk_overlap = best_chunk_config()

    if CROSS_PAGE_CHUNKS:
        return split_documents_cross_page(documents, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def split_documents_cross_page(pages: list, chunk_size: int, chunk_overlap: int) -> list:
    """Split each PDF as one continuous text so chunks can cross page breaks.

    WHY: the default split runs per page, so a sentence that continues on the
    next page is cut in two and neither half carries the whole answer
    (PAGE_SPLIT in eval/ERROR_ANALYSIS.md). Here the pages of one file are
    joined, split once, and every chunk is labelled with the page it STARTS on
    (`page`, `page_label`, so citations stay a single page) plus
    `page_end_label` for the last page it reaches.
    """
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    by_file: dict[str, list] = {}
    for page in pages:
        by_file.setdefault(page.metadata.get("filename") or page.metadata.get("source"), []).append(page)

    chunks = []
    # A single newline, not a blank line: the splitter prefers to cut at "\n\n",
    # so joining pages with one would make chunks snap to page boundaries again.
    separator = "\n"
    for file_pages in by_file.values():
        offsets, parts, position = [], [], 0
        for page in file_pages:
            # Strip page edges: a trailing newline plus the separator would form
            # "\n\n" and reintroduce a preferred split at every page break.
            content = page.page_content.strip()
            offsets.append(position)
            parts.append(content)
            position += len(content) + len(separator)
        text = separator.join(parts)
        base = dict(file_pages[0].metadata)
        for piece in splitter.split_documents([Document(page_content=text, metadata=base)]):
            start = piece.metadata.pop("start_index")
            end = start + len(piece.page_content) - 1
            first = max(i for i, offset in enumerate(offsets) if offset <= start)
            last = max(i for i, offset in enumerate(offsets) if offset <= end)
            meta = dict(file_pages[first].metadata)
            # Chroma rejects None metadata values, so fall back to the start page.
            meta["page_end_label"] = (file_pages[last].metadata.get("page_label")
                                      or meta.get("page_label") or "")
            chunks.append(Document(page_content=piece.page_content, metadata=meta))
    return chunks


def main() -> None:
    # Read winning chunk parameters (falls back to config.py if file absent).
    chunk_size, chunk_overlap = best_chunk_config()

    documents = load_documents()

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

    # Fresh build so re-ingesting never leaves stale or duplicate chunks.
    if CHROMA_DIR.is_dir():
        shutil.rmtree(CHROMA_DIR)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.add_documents(chunks)
    print(f"Saved {len(chunks)} embedded chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
