"""RAG: retrieve context from Chroma and answer a question with Groq's LLM.

Pipeline: embed question -> retrieve top-k chunks from ./chroma_db
-> build prompt -> send to Groq -> print the generated answer.

Requires GROQ_API_KEY in a .env file (loaded via python-dotenv).
"""

import sys

from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PROMPT_TEMPLATE,
    TOP_K,
    citation,
)

# Load API keys from .env before anything else uses them.
load_dotenv()

# The Windows console defaults to cp1252, which cannot encode some characters
# found in the PDF text (e.g. Greek letters) and would crash the print calls.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_context(question: str) -> str:
    """Retrieve the top-k chunks and join them into one string.

    Each chunk is labelled with "filename, p. X" so the model can cite pages
    instead of only filenames.
    """
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    docs = vector_store.similarity_search(question, k=TOP_K)
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Chunk {i}] Source: {citation(doc.metadata)}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str) -> None:
    """Answer `question` using only the retrieved context, via Groq's LLM."""
    context = get_context(question)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Groq() picks up GROQ_API_KEY from the environment (loaded from .env).
    client = Groq()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"Question: {question}\n")
    print("Answer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    answer_question(
        "What are the main advantages of the Transformer architecture "
        "over recurrent models?"
    )
