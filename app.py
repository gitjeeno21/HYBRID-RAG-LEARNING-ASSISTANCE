"""Streamlit web interface for the Autonomous RAG Agent.

Sidebar shows the source PDFs in data/ and the REAL evaluation numbers read
from eval_results.json (best chunk size, overlap, hit rate, judge score).
The chat interface retrieves top chunks from the Chroma store, labels each one
with "filename, p. X" AND its similarity score, sends them to Groq (key from
.env), and streams the generated answer — with page citations — back to the UI.
Each answer then gets a trust label (High / Medium / Low) plus a one-line reason
from trust.py, computed from the retrieval scores and one extra LLM support
check. The label is a heuristic, not a measured accuracy. Quiz Mode separately
generates source-grounded study questions from one selected PDF. Compare Mode
retrieves separately from two PDFs and compares them in one LLM request. New
PDFs can be uploaded from the sidebar and indexed into the local library.

Run with: .\\venv\\Scripts\\python.exe -m streamlit run .\\app.py
"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

import trust
from compare import generate_comparison, retrieve_paper_contexts
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    LLM_MODEL,
    PDF_DIR,
    PROMPT_TEMPLATE,
    REMOVED_PDF_DIR,
    TOP_K,
    citation,
    source_filename,
)
from ingest import best_chunk_config, load_documents, split_documents
from hybrid import build_bm25, hybrid_search_with_scores
from library import remove_pdf_from_library
from quiz import (
    generate_quiz,
    make_quiz_state,
    retrieve_quiz_chunks,
    submit_quiz_state,
)
from upload import add_pdf_to_library

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
    if store._collection.count() == 0 and list_knowledge_base():
        store.add_documents(split_documents(load_documents()))
    return store


@st.cache_resource
def get_hybrid_resources() -> tuple[list[Document], BM25Okapi]:
    """Build the in-memory BM25 sidecar over the active dense index's chunks."""
    store = get_vector_store()
    documents = load_documents()
    chunk_size, chunk_overlap = best_chunk_config()
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    indexed_count = store._collection.count()
    if indexed_count != len(chunks):
        raise RuntimeError(
            "The dense index and BM25 corpus do not match: "
            f"index={indexed_count} chunks, local corpus={len(chunks)} chunks"
        )
    return chunks, build_bm25(chunks)


@st.cache_resource
def get_llm_client() -> Groq:
    """Return a Groq client that reads GROQ_API_KEY from .env."""
    return Groq()


# ---------------------------------------------------------------------------
# Retrieval with similarity scores
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str) -> list[dict]:
    """Return hybrid top-k chunks with citations and dense scores when present."""
    chunks, bm25 = get_hybrid_resources()
    results = hybrid_search_with_scores(
        question,
        get_vector_store(),
        bm25,
        chunks,
        n_results=TOP_K,
    )
    retrieved = []
    for doc, score in results:
        retrieved.append(
            {
                "filename": source_filename(doc.metadata),
                "citation": citation(doc.metadata),
                "text": doc.page_content,
                # BM25-only fused results have no dense similarity score.
                "score": round(score, 3) if score is not None else None,
            }
        )
    return retrieved


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


def _activate_quiz_mode() -> None:
    """Turn off comparison when the user selects the quiz workflow."""
    if st.session_state["quiz_mode"]:
        st.session_state["compare_mode"] = False


def _activate_compare_mode() -> None:
    """Turn off quiz mode when the user selects the comparison workflow."""
    if st.session_state["compare_mode"]:
        st.session_state["quiz_mode"] = False


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

    if kb_files:
        with st.sidebar.expander(
            f"Manage library removal  ·  {len(kb_files)} PDF{'s' if len(kb_files) != 1 else ''}",
            expanded=False,
        ):
            st.caption("Select PDFs to remove them from retrieval.")
            with st.form("remove_papers_form", clear_on_submit=True):
                selected_for_removal = st.multiselect(
                    "Papers",
                    kb_files,
                    placeholder="Choose one or more PDFs",
                    key="papers_to_remove",
                )
                keep_removed_copy = st.radio(
                    "After removal",
                    (
                        "Keep a copy in removed_pdfs",
                        "Permanently delete the PDF files",
                    ),
                    key="paper_removal_behavior",
                )
                if keep_removed_copy.startswith("Permanently"):
                    st.warning(
                        "This permanently deletes the PDF and its indexed content."
                    )
                else:
                    st.caption(
                        "The PDF can be restored from `removed_pdfs` later."
                    )

                confirmed_removal = st.checkbox(
                    "Confirm removal of selected PDFs and database entries",
                    key="confirm_paper_removal",
                )
                submitted_removal = st.form_submit_button(
                    "Remove selected papers",
                    type="primary",
                    disabled=not selected_for_removal or not confirmed_removal,
                )

        if submitted_removal:
            notices = []
            removed_any = False
            for filename in selected_for_removal:
                try:
                    archived_path = remove_pdf_from_library(
                        filename,
                        get_vector_store(),
                        Path(PDF_DIR),
                        Path(REMOVED_PDF_DIR),
                        keep_copy=keep_removed_copy.startswith("Keep a copy"),
                    )
                except Exception as exc:
                    notices.append(
                        (
                            "error",
                            f"Could not remove {filename} "
                            f"({type(exc).__name__}): {exc}",
                        )
                    )
                else:
                    removed_any = True
                    if archived_path is not None:
                        notices.append(
                            (
                                "success",
                                f"Removed {filename} from the knowledge base. "
                                f"PDF kept at {archived_path}.",
                            )
                        )
                    else:
                        notices.append(
                            (
                                "success",
                                f"Permanently removed {filename} and its indexed chunks.",
                            )
                        )

            if removed_any:
                get_hybrid_resources.clear()
            st.session_state["paper_removal_notices"] = notices
            st.rerun()

    else:
        st.sidebar.caption("There are no active papers to remove.")

    for level, message in st.session_state.pop("paper_removal_notices", []):
        if level == "success":
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)

    st.sidebar.markdown("**Add a PDF**")
    uploaded_pdf = st.sidebar.file_uploader(
        "Choose a research paper",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_library_upload",
    )
    if st.sidebar.button(
        "Add PDF to library",
        disabled=uploaded_pdf is None,
        key="add_pdf_to_library",
    ):
        if uploaded_pdf is None:
            st.sidebar.warning("Choose a PDF before adding it.")
        else:
            try:
                # Initialize the persistent collection before writing the new
                # file, so an empty-index bootstrap cannot ingest it twice.
                vector_store = get_vector_store()
                chunk_size, chunk_overlap = best_chunk_config()
                chunk_count = add_pdf_to_library(
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                    vector_store,
                    PDF_DIR,
                    chunk_size,
                    chunk_overlap,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                st.sidebar.error(f"Could not add PDF: {exc}")
            except Exception as exc:
                # PDF parsers and embedding providers raise several library-
                # specific errors; show their message instead of hiding them.
                st.sidebar.error(
                    f"Could not add PDF ({type(exc).__name__}): {exc}"
                )
            else:
                get_hybrid_resources.clear()
                st.sidebar.success(
                    f"Added {uploaded_pdf.name} ({chunk_count} chunks)."
                )
                st.rerun()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autonomous RAG Agent", page_icon="🤖")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stForm"] {
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 0.75rem;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.025);
    }
    section[data-testid="stSidebar"] div[data-testid="stForm"] button[kind="primary"] {
        width: 100%;
        border-radius: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load evaluation summary once; cached across reruns.
eval_summary = load_eval_summary()
render_sidebar(eval_summary)

st.title("Autonomous RAG Agent")

# Each workflow has its own sidebar switch; callbacks keep the two specialized
# modes mutually exclusive so one form cannot shadow the other.
quiz_mode = st.sidebar.toggle(
    "Quiz Mode",
    value=False,
    key="quiz_mode",
    on_change=_activate_quiz_mode,
)
compare_mode = st.sidebar.toggle(
    "Compare Mode",
    value=False,
    key="compare_mode",
    on_change=_activate_compare_mode,
)

if compare_mode:
    st.header("Compare Mode")
    st.caption(
        "Compare evidence from two PDFs. Each paper is retrieved separately, "
        "then both contexts are sent in one request."
    )
    pdf_files = list_knowledge_base()
    if len(pdf_files) < 2:
        st.warning("Add at least two PDFs to the knowledge base to compare them.")
    else:
        # Restrict the comparison to exactly two distinct source documents.
        selected_pdfs = st.sidebar.multiselect(
            "Papers to compare (select two)",
            pdf_files,
            default=pdf_files[:2],
            max_selections=2,
            key="compare_pdfs",
        )
        question = st.sidebar.text_input(
            "Comparison question",
            key="compare_question",
        )

        if st.button("Compare papers", key="generate_comparison"):
            if len(selected_pdfs) != 2:
                st.warning("Select exactly two PDFs before comparing.")
            elif not question.strip():
                st.warning("Enter a question before comparing the papers.")
            else:
                # The helper makes one filename-filtered search per paper; one
                # combined prompt then avoids a separate LLM call for each.
                with st.spinner("Retrieving both papers and generating comparison..."):
                    chunks, bm25 = get_hybrid_resources()
                    contexts = retrieve_paper_contexts(
                        get_vector_store(),
                        bm25,
                        chunks,
                        question.strip(),
                        selected_pdfs,
                    )
                    comparison_text = generate_comparison(
                        question.strip(),
                        contexts,
                    )
                # Keep the evidence alongside the answer for transparent replay.
                st.session_state["comparison_result"] = {
                    "pdfs": selected_pdfs,
                    "question": question.strip(),
                    "text": comparison_text,
                    "contexts": {
                        pdf_name: [
                            {
                                "citation": citation(chunk.metadata),
                                "text": chunk.page_content,
                            }
                            for chunk in chunks
                        ]
                        for pdf_name, chunks in contexts.items()
                    },
                }

        result = st.session_state.get("comparison_result")
        if (
            result
            and result["pdfs"] == selected_pdfs
            and result["question"] == question.strip()
        ):
            st.markdown(result["text"])
            with st.expander("Retrieved passages for each paper"):
                for pdf_name, passages in result["contexts"].items():
                    st.markdown(f"### {pdf_name}")
                    if not passages:
                        st.info("No passages were retrieved from this paper.")
                    for passage in passages:
                        st.markdown(f"**{passage['citation']}**")
                        st.text(passage["text"])
elif quiz_mode:
    st.header("Quiz Mode")
    st.caption(
        "Generate five practice questions grounded only in retrieved text from "
        "one selected PDF."
    )
    pdf_files = list_knowledge_base()
    if not pdf_files:
        st.warning("No PDFs are available in the knowledge base.")
    else:
        # Keep quiz inputs in the sidebar as requested, alongside the mode switch.
        selected_pdf = st.sidebar.selectbox("Quiz source PDF", pdf_files)
        topic = st.sidebar.text_input("Quiz topic")

        if st.button("Generate 5-question quiz", key="generate_quiz"):
            if not topic.strip():
                st.warning("Enter a topic before generating a quiz.")
            else:
                # Retrieval is scoped to the chosen PDF; the single LLM prompt
                # receives only these passages and their citation-helper labels.
                with st.spinner("Retrieving passages and generating your quiz..."):
                    chunks, bm25 = get_hybrid_resources()
                    quiz_chunks = retrieve_quiz_chunks(
                        get_vector_store(),
                        bm25,
                        chunks,
                        selected_pdf,
                        topic.strip(),
                    )
                    quiz = generate_quiz(
                        selected_pdf,
                        topic.strip(),
                        quiz_chunks,
                    )
                if quiz["covered"]:
                    # Preserve questions, evidence, and answers across reruns.
                    st.session_state["quiz_state"] = make_quiz_state(
                        selected_pdf,
                        topic.strip(),
                        quiz,
                        quiz_chunks,
                    )
                else:
                    st.session_state["quiz_state"] = {
                        "pdf": selected_pdf,
                        "topic": topic.strip(),
                        "questions": [],
                        "answers": [],
                        "grade": None,
                        "context": "",
                        "chunks": [],
                    }

        quiz_state = st.session_state.get("quiz_state")
        if (
            quiz_state
            and quiz_state["pdf"] == selected_pdf
            and quiz_state["topic"] == topic.strip()
        ):
            if not quiz_state["questions"]:
                st.info(
                    "The retrieved passages do not cover this topic well enough "
                    "to make a grounded quiz."
                )
            elif quiz_state["grade"] is None:
                st.subheader("Answer the questions")
                # The attempt-specific widget keys preserve typed answers over
                # reruns without mixing them into a newly generated quiz.
                with st.form(
                    f"quiz_answers_{quiz_state['attempt_id']}",
                    clear_on_submit=False,
                ):
                    answer_keys = []
                    for index, question_item in enumerate(
                        quiz_state["questions"], start=1
                    ):
                        st.markdown(f"**{index}. {question_item['question']}**")
                        answer_key = (
                            f"quiz_answer_{quiz_state['attempt_id']}_{index}"
                        )
                        answer_keys.append(answer_key)
                        st.text_area(
                            "Your answer",
                            key=answer_key,
                            height=90,
                        )
                    submitted = st.form_submit_button("Submit answers for grading")

                if submitted:
                    user_answers = [
                        st.session_state.get(answer_key, "")
                        for answer_key in answer_keys
                    ]
                    with st.spinner("Grading answers against the retrieved paper..."):
                        st.session_state["quiz_state"] = submit_quiz_state(
                            quiz_state,
                            user_answers,
                        )
                    st.rerun()
            else:
                grade = quiz_state["grade"]
                score = grade["score"]
                score_label = (
                    str(int(score)) if float(score).is_integer() else str(score)
                )
                st.subheader(f"Final score: {score_label}/{grade['maximum']}")
                st.caption("Correct = 1 point, Partial = 0.5 points, Incorrect = 0.")

                missed = []
                for index, (question_item, user_answer, result_item) in enumerate(
                    zip(
                        quiz_state["questions"],
                        quiz_state["answers"],
                        grade["results"],
                    ),
                    start=1,
                ):
                    st.markdown(
                        f"**{index}. {question_item['question']} — "
                        f"{result_item['grade']}**"
                    )
                    st.markdown(f"Your answer: {user_answer or '_No answer_'}")
                    st.markdown(result_item["feedback"])
                    if result_item["grade"] != "Correct":
                        missed.append((question_item, result_item))

                if missed:
                    st.subheader("Missed questions and source answers")
                    for question_item, result_item in missed:
                        st.markdown(f"- **{question_item['question']}**")
                        st.markdown(
                            f"  Correct answer: {result_item['correct_answer']} "
                            f"({result_item['source']})"
                        )
                else:
                    st.success("All answers were graded correct.")

            if quiz_state["chunks"]:
                with st.expander("Retrieved passages used for this quiz"):
                    for passage in quiz_state["chunks"]:
                        st.markdown(f"**{passage['citation']}**")
                        st.text(passage["text"])
else:
    st.caption("Ask anything about the documents in your knowledge base.")
    has_active_papers = bool(list_knowledge_base())
    if not has_active_papers:
        st.info("Add a PDF to the knowledge base before asking a question.")
    # 1. Initialize chat history and display it on every rerun.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Replay the stored trust label too; recomputing it would cost
            # another LLM call on every rerun just to redraw old messages.
            if message.get("trust"):
                render_trust(message["trust"])
            if message.get("chunks"):
                render_retrieved_context(message["chunks"])

    if question := st.chat_input(
        "Ask a question about your documents:",
        disabled=not has_active_papers,
    ):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # One spinner covers retrieval and answer generation.
            with st.spinner("Searching database & generating answer..."):
                chunks = retrieve_chunks(question)
                context = format_context(chunks)
                answer = st.write_stream(stream_answer(context, question))
                # Trust uses exactly one additional support-check call and the
                # already available retrieval scores.
                trust_result = trust.assess(question, answer, chunks)
            render_trust(trust_result)
            render_retrieved_context(chunks)

        # Persist the answer and diagnostics so reruns can replay them without
        # repeating generation or the trust check.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
                "trust": trust_result,
            }
        )
