"""Safely save an uploaded PDF and add its chunks to the local vector store."""

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ingest import split_documents
from library import is_safe_pdf_name


def add_pdf_to_library(
    filename: str,
    file_bytes: bytes,
    vector_store,
    pdf_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Validate, persist, and index one new PDF; return its indexed chunk count."""
    safe_name = filename
    if not is_safe_pdf_name(safe_name):
        raise ValueError("Upload a PDF with a simple filename ending in .pdf")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not have a valid PDF header")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    destination = pdf_dir / safe_name
    if destination.exists():
        raise FileExistsError(
            f"{safe_name} already exists in the library; rename the upload first"
        )

    temp_path: Path | None = None
    try:
        # Parse a temporary file first so an invalid upload is never left in
        # the visible PDF library.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix=".upload-",
            dir=pdf_dir,
            delete=False,
        ) as temporary:
            temporary.write(file_bytes)
            temp_path = Path(temporary.name)

        pages = PyPDFLoader(str(temp_path)).load()
        if not pages:
            raise ValueError("The uploaded PDF contains no readable pages")
        for page in pages:
            # Use the final filename in citations and the eventual source path,
            # not the temporary parser path.
            page.metadata["filename"] = safe_name
            # The bare name, not the server path: metadata is sent to the browser.
            page.metadata["source"] = safe_name

        chunks: list[Document] = split_documents(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("The uploaded PDF did not produce any text chunks")

        vector_store.add_documents(chunks)
        os.replace(temp_path, destination)
        temp_path = None
        return len(chunks)
    except Exception:
        # If parsing, embedding, indexing, or finalizing fails, remove this
        # upload's partial vector entries and temporary/destination files so a
        # retry does not create duplicates or a phantom library entry.
        vector_store.delete(where={"filename": safe_name})
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        if destination.exists():
            destination.unlink()
        raise
