"""Query the local Chroma vector store built by ingest.py.

Loads the persisted embeddings from ./chroma_db and prints the
top-k chunks most relevant to a given question, each labelled with the
source file and page number it came from.
"""

import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    citation,
)

# The Windows console defaults to cp1252, which cannot encode some characters
# found in the PDF text (e.g. Greek letters) and would crash the print calls.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def search_database(question: str) -> None:
    """Search the vector store for the top-k chunks relevant to `question`."""
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,  # must match ingest.py, hence config.py
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    results = vector_store.similarity_search(question, k=TOP_K)

    print(f"Question: {question}\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ({citation(doc.metadata)}) ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    search_database(
        "What BLEU score did the Transformer achieve on the WMT 2014 "
        "English-to-German translation task?"
    )
