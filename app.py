"""Streamlit web interface for the Autonomous RAG Agent.

Sidebar shows the source PDFs in data/. The chat interface retrieves top
chunks from the Chroma store, labels each one with "filename, p. X", sends
them to Groq (key from .env), and streams the generated answer — with page
citations — back to the UI.

Run with: streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PDF_DIR,
    PROMPT_TEMPLATE,
    TOP_K,
    citation,
    source_filename,
)
from ingest import load_documents, split_documents

load_dotenv()


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_vector_store() -> Chroma:
    """Load the multi-document Chroma store, rebuilding it if missing."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    # Guards against a first run where `python ingest.py` was never executed
    # (or chroma_db was wiped): the app still works, just slower once.
    if store._collection.count() == 0:
        store.add_documents(split_documents(load_documents()))
    return store


@st.cache_resource
def get_llm_client() -> Groq:
    return Groq()  # reads GROQ_API_KEY from .env


def retrieve_chunks(question: str) -> list[dict]:
    """Return the top-k chunks for `question`, each with its citation."""
    docs = get_vector_store().similarity_search(question, k=TOP_K)
    return [
        {
            "filename": source_filename(doc.metadata),
            "citation": citation(doc.metadata),
            "text": doc.page_content,
        }
        for doc in docs
    ]


def format_context(chunks: list[dict]) -> str:
    """Join chunks into one context string, each labelled with its citation."""
    return "\n\n".join(
        f"Source: {chunk['citation']}\n{chunk['text']}" for chunk in chunks
    )


def stream_answer(context: str, question: str):
    """Yield the LLM's answer tokens as they arrive from Groq."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    stream = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def render_retrieved_context(chunks: list) -> None:
    """Display the raw retrieved chunks (with citations) in an expander."""
    with st.expander("View Retrieved Source Context"):
        for i, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, str):  # history written by an older version
                label, text = "unknown", chunk
            else:
                label = chunk.get("citation") or chunk["filename"]
                text = chunk["text"]
            st.markdown(f"**Chunk {i}/{len(chunks)} — {label}**")
            st.text(text)


def list_knowledge_base() -> list[str]:
    """Return the PDF filenames currently sitting in the data/ folder."""
    return sorted(p.name for p in Path(PDF_DIR).glob("*.pdf"))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous RAG Agent", page_icon="🤖")

st.sidebar.markdown("### Autonomous RAG Agent")
st.sidebar.markdown("**Optimized Configuration: 1000-character chunk size**")
st.sidebar.markdown("**Autonomous Experiment Results**")
st.sidebar.metric(label="Winning Chunk Size", value=f"{CHUNK_SIZE} chars")
st.sidebar.metric(label="AI Judge Score", value="5/5")

st.sidebar.markdown("**Active Knowledge Base**")
kb_files = list_knowledge_base()
if kb_files:
    st.sidebar.markdown("\n".join(f"- {name}" for name in kb_files))
else:
    st.sidebar.markdown("_No PDF files found in data/_")

st.title("Autonomous RAG Agent")
st.caption("Ask anything about the documents in your knowledge base.")

# 1. Initialize chat history and display it on every rerun.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chunks"):
            render_retrieved_context(message["chunks"])

if question := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 2. One spinner covers retrieval and answer generation.
        with st.spinner("Searching database & generating answer..."):
            chunks = retrieve_chunks(question)
            context = format_context(chunks)
            answer = st.write_stream(stream_answer(context, question))
        # 3. Show the raw chunks (with citations) after the answer.
        render_retrieved_context(chunks)

    # 4. Persist the answer and its chunks for history replay.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
