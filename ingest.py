"""Ingest every PDF in data/ into a local Chroma vector store.

Pipeline: scan data/*.pdf -> load pages and tag each chunk with its source
filename -> split into chunks (size/overlap imported from config.py) -> embed
with all-MiniLM-L6-v2 -> persist to ./chroma_db, which is rebuilt from scratch
on every run.
"""

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
    EMBEDDING_MODEL,
    PDF_DIR,
)


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


def split_documents(documents: list) -> list:
    """Split pages into chunks; metadata (filename, page) is preserved.

    RecursiveCharacterTextSplitter tries paragraph, then line, then sentence
    boundaries before cutting mid-word, which keeps chunks readable and
    therefore more useful as LLM context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def main() -> None:
    documents = load_documents()

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

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
