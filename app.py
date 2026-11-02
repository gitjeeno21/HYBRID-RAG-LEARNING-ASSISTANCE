"""Streamlit web interface for the Autonomous RAG Agent.

Sidebar shows the source PDFs in data/ and the REAL evaluation numbers read
from eval_results.json (best chunk size, overlap, hit rate, judge score).
The chat interface retrieves top chunks from the Chroma store, labels each one
with "filename, p. X" AND its similarity score, sends them to Groq (key from
.env), and streams the generated answer — with page citations — back to the UI.
Each answer then gets a trust label (High / Medium / Low) plus a one-line reason
from trust.py, computed from the retrieval scores and one extra LLM support
check. The label is a heuristic, not a measured accuracy.

Run with: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import trust
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    LLM_MODEL,
    PDF_DIR,
    PROMPT_TEMPLATE,
    TOP_K,
    citation,
    source_filename,
)
from ingest import best_chunk_config, load_documents, split_documents

load_dotenv()


# ---------------------------------------------------------------------------
# Eval results loader
# ---------------------------------------------------------------------------
@st.cache_data
def load_eval_summary() -> dict:
    """Read best_config from eval_results.json and return it as a plain dict.

    WHY cache_data (not cache_resource): the return value is a plain dict
    (JSON-serialisable), so Streamlit can hash it for cache invalidation.
    cache_resource is for objects that hold external connections (DB, model).

    WHY return {} on failure: the sidebar must render even if evaluator.py has
    never been run (e.g. a fresh clone). The caller treats missing keys as
    "not available yet" and shows a dash instead of crashing.
    """
    try:
        data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        return data.get("best_config", {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Cached heavy resources (loaded once per Streamlit process)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer model once.

    WHY cache_resource: HuggingFaceEmbeddings holds a PyTorch model in memory.
    Reloading it on every user message would add ~2 s latency and waste RAM.
    """
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_vector_store() -> Chroma:
    """Load the multi-document Chroma store, rebuilding it if missing.

    WHY rebuild on empty: guards against a first run where `python ingest.py`
    was never executed (or chroma_db was wiped). The app still works, just
    slower on that first request.
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0:
        store.add_documents(split_documents(load_documents()))
    return store


@st.cache_resource
def get_llm_client() -> Groq:
    """Return a Groq client that reads GROQ_API_KEY from .env."""
    return Groq()


# ---------------------------------------------------------------------------
# Retrieval with similarity scores
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str) -> list[dict]:
    """Return the top-k chunks for `question`, each with citation AND score.

    WHY similarity_search_with_relevance_scores instead of similarity_search:
    Chroma returns a cosine-similarity score in [0, 1] alongside each document
    when this method is used. Exposing the score in the UI lets the user judge
    how confident the retrieval step is — a score near 0 means the question is
    probably outside the corpus.
    """
    # Returns list of (Document, score) tuples; score is cosine similarity.
    results = get_vector_store().similarity_search_with_relevance_scores(
        question, k=TOP_K
    )
    chunks = []
    for doc, score in results:
        chunks.append(
            {
                "filename": source_filename(doc.metadata),
                "citation": citation(doc.metadata),
                "text": doc.page_content,
                # Round to 3 decimal places; raw float has noisy trailing digits.
                "score": round(score, 3),
            }
        )
    return chunks


def format_context(chunks: list[dict]) -> str:
    """Join chunks into one context string, each labelled with its citation.

    The score is NOT included here: it is UI metadata, not part of the prompt.
    Sending it to the LLM would waste tokens and might confuse the model.
    """
    return "\n\n".join(
        f"Source: {chunk['citation']}\n{chunk['text']}" for chunk in chunks
    )


def stream_answer(context: str, question: str):
    """Yield the LLM's answer tokens as they arrive from Groq.

    WHY stream=True: streaming lets Streamlit render each token as it arrives
    via st.write_stream, so the user sees a response immediately rather than
    waiting for the full generation to finish.
    """
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    stream = get_llm_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_trust(result: dict) -> None:
    """Show the trust label and its one-line reason directly under the answer.

    WHY a coloured emoji instead of st.metric/st.success: some Streamlit
    versions dropped the `color=` argument on st.success, and a metric box would
    take over the chat layout. Plain markdown with a traffic-light emoji works
    everywhere and reads as one line, which is what the brief asked for.

    WHY show the reason: the label comes from a heuristic (see trust.py), so a
    bare "Low" would look like a measurement. The reason names which of the two
    signals failed, letting the reader decide whether to trust the label itself.
    """
    label = result.get("label", "n/a")
    # Green / amber / red for the three real labels; grey when the system
    # refused, because a refusal is not a confidence claim in either direction.
    icon = {"High": "\U0001F7E2", "Medium": "\U0001F7E1", "Low": "\U0001F534"}.get(
        label, "\u26AA"
    )
    st.markdown(f"{icon} **Trust: {label}** — {result.get('reason', '')}")
    # The raw signals live one click away: useful in an interview, out of the way
    # in normal use.
    with st.expander("Why this trust label?"):
        st.markdown(
            f"- Support check: `{result.get('verdict') or 'not run / unparseable'}`\n"
            f"- Best chunk similarity: `{result.get('top_score')}`\n"
            f"- Mean similarity: `{result.get('mean_score')}`\n"
            "- Reminder: these two signals are a heuristic, not a measured "
            "accuracy (see trust.py)."
        )


def render_retrieved_context(chunks: list) -> None:
    """Display the raw retrieved chunks (with citations AND similarity scores).

    WHY show similarity scores here: they are debugging / transparency
    information. Displaying them inside the expander — not in the main answer
    area — keeps the primary UX clean while still letting a curious user (or
    an interviewer) inspect retrieval quality without re-running a CLI command.
    """
    with st.expander("View Retrieved Source Context"):
        for i, chunk in enumerate(chunks, start=1):
            if isinstance(chunk, str):
                # Backwards-compatibility: history written by an older version
                # stored raw strings rather than dicts.
                label, text, score = "unknown", chunk, None
            else:
                label = chunk.get("citation") or chunk["filename"]
                text = chunk["text"]
                score = chunk.get("score")

            # Show the score next to the citation so the reader can immediately
            # see how relevant the retrieval engine thought this chunk was.
            score_str = f" — similarity {score:.3f}" if score is not None else ""
            st.markdown(f"**Chunk {i}/{len(chunks)} — {label}{score_str}**")
            st.text(text)


def list_knowledge_base() -> list[str]:
    """Return the PDF filenames currently sitting in the data/ folder."""
    return sorted(p.name for p in Path(PDF_DIR).glob("*.pdf"))


# ---------------------------------------------------------------------------
# Sidebar — real numbers from eval_results.json
# ---------------------------------------------------------------------------
def render_sidebar(eval_summary: dict) -> None:
    """Populate the sidebar with live evaluation metrics.

    WHY read from eval_results.json instead of hardcoding: the evaluation grid
    in evaluator.py is the authoritative source of chunk-size and quality
    numbers. Hardcoding them in the UI would mean the display goes stale every
    time the evaluation is re-run with different parameters.
    """
    st.sidebar.markdown("### Autonomous RAG Agent")
    st.sidebar.markdown("**Best Configuration (from evaluation)**")

    chunk_size = eval_summary.get("chunk_size")
    chunk_overlap = eval_summary.get("chunk_overlap")
    hit_rate = eval_summary.get("hit_rate")
    judge_mean = eval_summary.get("judge_mean")
    judge_std = eval_summary.get("judge_std")

    # Display each metric; fall back to "—" when the key is missing so the
    # sidebar never crashes on a pre-evaluation run.
    st.sidebar.metric(
        label="Chunk size (chars)",
        value=str(chunk_size) if chunk_size is not None else "—",
    )
    st.sidebar.metric(
        label="Chunk overlap (chars)",
        value=str(chunk_overlap) if chunk_overlap is not None else "—",
    )
    st.sidebar.metric(
        label="Retrieval hit rate",
        # Format as percentage with one decimal place; e.g. 0.8 -> "80.0%".
        value=f"{hit_rate:.1%}" if hit_rate is not None else "—",
    )

    # judge_mean is on a 1–5 scale; show std in the delta slot so the user can
    # see both the central estimate and the run-to-run variability at a glance.
    judge_label = (
        f"{judge_mean:.2f} / 5"
        if judge_mean is not None
        else "—"
    )
    judge_delta = (
        f"±{judge_std:.3f} std" if judge_std is not None else None
    )
    st.sidebar.metric(
        label="LLM judge score (1–5)",
        value=judge_label,
        delta=judge_delta,
        # delta_color="off" suppresses the green/red arrow, which would be
        # misleading here (std deviation is not a directional change).
        delta_color="off",
    )

    st.sidebar.markdown("**Active Knowledge Base**")
    kb_files = list_knowledge_base()
    if kb_files:
        st.sidebar.markdown("\n".join(f"- {name}" for name in kb_files))
    else:
        st.sidebar.markdown("_No PDF files found in data/_")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous RAG Agent", page_icon="🤖")

# Load evaluation summary once; cached across reruns.
eval_summary = load_eval_summary()
render_sidebar(eval_summary)

st.title("Autonomous RAG Agent")
st.caption("Ask anything about the documents in your knowledge base.")

# 1. Initialize chat history and display it on every rerun.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Replay the stored trust label too; recomputing it would cost another
        # LLM call on every rerun just to redraw old messages.
        if message.get("trust"):
            render_trust(message["trust"])
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
            # 3. Trust meter: exactly ONE extra LLM call per answer (a support
            # check), plus the free retrieval-strength signal from the scores
            # above. It runs after the answer has streamed, so a slow or
            # rate-limited check never delays the text the user is reading.
            trust_result = trust.assess(question, answer, chunks)
        # 4. Label the answer, then show the raw chunks it was built from.
        render_trust(trust_result)
        render_retrieved_context(chunks)

    # 5. Persist the answer, its chunks and its trust label for history replay.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "trust": trust_result,
        }
    )
