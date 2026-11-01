"""Single source of truth for paths, model names, and retrieval settings.

WHY this file exists: before it, every module carried its own copy of the
settings, and they had drifted apart -- ``query.py`` searched a Chroma
collection called "source_doc" while ``ingest.py`` wrote its chunks into
"documents", so querying silently returned nothing. Now every module imports
the same constants, and a setting is changed in exactly one place.

Import these values instead of redefining them::

    from config import CHROMA_DIR, COLLECTION_NAME, TOP_K
"""

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Resolved from this file's location, not from the shell's current directory,
# so `python query.py` works no matter where it is launched from.
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
EVAL_RESULTS_PATH = BASE_DIR / "eval_results.json"

# --- Vector store ----------------------------------------------------------
# One collection for the whole corpus. Changing this value requires a
# re-ingest, because existing chunks stay under the old collection name.
COLLECTION_NAME = "documents"

# Small (~90 MB) CPU embedding model: keeps the project runnable on 4 GB VRAM
# machines and needs no API key.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Hosted on Groq (free tier), so no local LLM memory is used at all.
LLM_MODEL = "openai/gpt-oss-120b"

# --- Chunking and retrieval ------------------------------------------------
# Defaults, used when eval_results.json does not exist yet. evaluator.py
# overwrites these at runtime with the configuration it measured as best.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
TOP_K = 3

# --- Prompting -------------------------------------------------------------
# Kept here so app.py (Streamlit) and rag.py (CLI) send the identical prompt;
# otherwise the two answer paths would drift and results would not compare.
NO_ANSWER_TEXT = "I don't know based on the provided documents."

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the context below. Do not use outside "
    "knowledge.\n"
    "If the context does not contain the answer, reply with exactly this "
    f'sentence and nothing else: "{NO_ANSWER_TEXT}"\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer in at most 120 words and cite the pages you used with the "
    "citation labels given in the context, e.g. (source_doc.pdf, p. 3). "
    "End with one final line in exactly this format:\n"
    "Sources: <filename, p. X>; <filename, p. X>"
)


# --- Small shared helpers --------------------------------------------------
# Citation formatting lives next to the settings because every module needs
# it; duplicating the filename/page fallback logic is how the collection-name
# bug happened in the first place.
def source_filename(metadata: dict) -> str:
    """Return the best available filename for a chunk.

    ``filename`` is our own tag added by ingest.py; ``source`` is the path
    PyPDFLoader records; the literal is a last resort so display code never
    crashes on a chunk with empty metadata.
    """
    return (
        metadata.get("filename")
        or os.path.basename(metadata.get("source") or "")
        or "unknown.pdf"
    )


def page_number(metadata: dict) -> str:
    """Return a 1-based page number for humans.

    PyPDFLoader stores ``page`` 0-based, which would show the first page as
    "p. 0", so we prefer ``page_label`` (already 1-based) and add 1 ourselves
    when only the raw index is available.
    """
    if metadata.get("page_label"):
        return str(metadata["page_label"])
    page = metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


def citation(metadata: dict) -> str:
    """Format a chunk's source as "filename, p. X"."""
    return f"{source_filename(metadata)}, p. {page_number(metadata)}"
