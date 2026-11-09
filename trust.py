"""Heuristic "trust meter" for answers produced by the RAG pipeline.

The label this module returns is a HEURISTIC, NOT A MEASURED ACCURACY. Nothing
here has been calibrated against ground truth: the two cosine-similarity cut-offs
below were chosen by inspecting the retrieval scores of the eval questions, and
the LLM check is simply a second opinion from the same model family that wrote
the answer. A
"High" label therefore means "two cheap signals agree", not "this answer is
correct with probability X". Treat the label as a prompt to look closer, never
as evidence.

MEASURED (Phase 4, eval/ANSWER_QUALITY_k5.md, held-out v2, top-k 5): all 38
answered questions were correct, labelled High 8 and Medium 30, never Low.
Both labels were 100% correct, so on this set the label does NOT separate
right from wrong answers. The retrieval-strength cut-off (0.40) put 30 correct
answers at Medium. Do not present the label as an accuracy estimate.

Two signals are combined:

(a) RETRIEVAL STRENGTH -- how similar the retrieved chunks were to the question,
    using the cosine similarities Chroma already returns. Cheap, deterministic,
    and it catches the dominant failure mode of this system: the answer is
    confidently about the right topic but the corpus never covered it.

(b) SUPPORT CHECK -- one extra LLM call that asks whether every claim in the
    answer is supported by the retrieved context, answered SUPPORTED / PARTIAL /
    UNSUPPORTED. This catches the opposite failure: good retrieval, but the model
    added a fact that is not in the passages.

Combination rules (all four are in `combine_labels`, and they are the whole rule
set -- there is no hidden weighting):

    verdict        retrieval      label
    SUPPORTED      strong         High
    SUPPORTED      weak           Medium
    PARTIAL        strong         Medium
    PARTIAL        weak           Low
    UNSUPPORTED    either         Low
    unparseable    strong         Medium   (never High without the check)
    unparseable    weak           Low

Why this shape: an unsupported claim is a correctness problem, so it wins over
good retrieval. A supported answer with weak retrieval is still probably about
the wrong passage, so it is capped at Medium. Nothing reaches High unless both
signals are positive.

Command line, for the refusal numbers reported alongside the eval set:

    python trust.py              # refusal evaluation over eval_questions.json
    python trust.py --mock       # offline fixture check; not model performance

That run answers every question through the same prompt the app uses and counts,
for two disjoint groups of questions, how often the system refused to answer.
"""

import argparse
import json
import sys

from config import BASE_DIR, NO_ANSWER_TEXT, PROMPT_TEMPLATE, TOP_K

# The Windows console defaults to cp1252, which cannot encode some characters
# found in the PDF text (e.g. Greek letters) and would crash the print calls.
# Same guard as rag.py/query.py/evaluator.py so behaviour is consistent.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS_PATH = BASE_DIR / "eval_questions.json"

# --- Signal (a): retrieval-strength cut-offs --------------------------------
# Cosine similarity of all-MiniLM-L6-v2 embeddings. This corpus is small and the
# gaps between on-topic and off-topic scores are narrow, so these cut-offs come
# from an observed score distribution, NOT from a calibration experiment: over
# all 26 questions in eval_questions.json the off-topic ones top out at 0.23,
# while the answerable ones start at 0.28 and mostly sit in 0.35-0.65. A 0.40
# cut-off therefore separates the two groups on this set.
#
# HONESTY CAVEAT: those scores were inspected on the same questions used to
# report the refusal numbers below, so this is fitting to the test set, not a
# held-out measurement, and the 0.23 / 0.28 margin is thin enough that a new
# off-topic question could easily land above 0.40. Treat 0.40 as a reasonable
# default to re-tune on your own corpus, not as a validated boundary.
STRONG_TOP_SCORE = 0.40   # best chunk must be at least this similar ...
STRONG_MEAN_SCORE = 0.50  # ... or the average over all retrieved chunks is
# Strong retrieval means: top >= 0.40 OR mean >= 0.50. Using "or" rather than
# "and" keeps the label from collapsing to Medium when one chunk carries the
# whole answer and the other two are filler, which is common with top_k = 3.

# --- Signal (b): the support check ------------------------------------------
# Deliberately asks for exactly one of three fixed words and nothing else, so
# the reply is trivially parseable and the call stays cheap (a few dozen tokens
# of output). The prompt shows the context and the answer but NOT the question
# only -- it includes the question so the checker can notice an answer that is
# faithful but off-topic.
SUPPORT_PROMPT_TEMPLATE = (
    "You check whether a generated answer is grounded in retrieved passages.\n"
    "Is every claim in the ANSWER supported by the retrieved CONTEXT?\n"
    "- SUPPORTED: all claims are directly supported by the context.\n"
    "- PARTIAL: some claims are supported, others are missing or invented.\n"
    "- UNSUPPORTED: the claims are not supported, or the answer contradicts the "
    "context.\n"
    "Grade only grounding. Do not reward extra outside knowledge and do not "
    "judge whether the answer is useful.\n\n"
    "Question: {question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Reply with exactly one word: SUPPORTED, PARTIAL or UNSUPPORTED"
)

# NO_ANSWER_TEXT is the sentence config.PROMPT_TEMPLATE orders the model to
# return when the context does not contain the answer, so it is also the
# definition of "the system refused" used by the refusal measurement below.
_REFUSAL_MARKER = NO_ANSWER_TEXT.rstrip(".").lower()


# ---------------------------------------------------------------------------
# Signal (a): retrieval strength
# ---------------------------------------------------------------------------
def retrieval_strength(scores: list[float | None]) -> dict:
    """Summarise the similarity scores of the retrieved chunks.

    Returns {"top": float|None, "mean": float|None, "strong": bool}.

    WHY top and mean, not just top: a single chunk can be similar to the
    question by coincidence (shared boilerplate, a figure caption), so looking
    at the average too makes an accidental match less likely to be called
    "strong". WHY None-safe: main.py can replay chat history written by an older
    version that stored chunks without scores, and a missing score must not
    raise.
    """
    usable = [s for s in scores if isinstance(s, (int, float))]
    if not usable:
        # No scores at all means we cannot claim strong retrieval, so the
        # caller falls back to the verdict-only half of the rule table.
        return {"top": None, "mean": None, "strong": False}
    top = max(usable)
    mean = sum(usable) / len(usable)
    return {"top": top, "mean": mean, "strong": top >= STRONG_TOP_SCORE or mean >= STRONG_MEAN_SCORE}


# ---------------------------------------------------------------------------
# Signal (b): support check
# ---------------------------------------------------------------------------
def parse_verdict(reply: str) -> str | None:
    """Turn the checker's free text into SUPPORTED / PARTIAL / UNSUPPORTED.

    WHY check UNSUPPORTED first: the string "UNSUPPORTED" CONTAINS "SUPPORTED",
    so testing for "SUPPORTED" first would mislabel every unsupported answer as
    its exact opposite -- the single most damaging possible parse bug here.
    Returns None when the model ignored the format; the caller then degrades to
    a retrieval-only label instead of inventing a verdict.
    """
    if not reply:
        return None
    upper = reply.upper()
    if "UNSUPPORTED" in upper:
        return "UNSUPPORTED"
    if "PARTIAL" in upper:
        return "PARTIAL"
    if "SUPPORTED" in upper:
        return "SUPPORTED"
    return None


def _default_ask(prompt: str) -> str:
    """Send one prompt to Groq through evaluator.py's retrying helper.

    WHY reuse evaluator.call_llm instead of a second copy: it already handles
    Groq's free-tier 429s with the MAX_ATTEMPTS / BASE_SLEEP backoff this
    project standardised on, and duplicating it would let the two retry
    policies drift. WHY the import sits inside the function: evaluator.py
    builds a Groq client and reconfigures stdout at import time, so importing
    it lazily keeps `import trust` (and therefore the Streamlit app's start-up)
    free of that side effect until the first check actually runs.
    """
    from evaluator import call_llm

    return call_llm(prompt, temperature=0.0, label="trust")


def check_support(question: str, context: str, answer: str, ask=None) -> str | None:
    """Ask the LLM whether every claim in `answer` is supported by `context`.

    Exactly one extra LLM call per answer. `ask` is injectable so the refusal
    evaluation and any future test can run without the network.

    Returns a verdict string or None if the call failed. WHY swallow failures:
    the answer has already been shown to the user, so a rate limit at the very
    last step should downgrade the label, not crash the chat. The reason string
    says the check was unavailable, so the user is not misled into thinking a
    check happened.
    """
    ask = ask or _default_ask
    prompt = SUPPORT_PROMPT_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
    try:
        return parse_verdict(ask(prompt))
    except Exception as exc:  # noqa: BLE001 - any provider error degrades the label
        print(f"[trust] support check failed: {type(exc).__name__}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Combining the two signals
# ---------------------------------------------------------------------------
def combine_labels(strength: dict, verdict: str | None) -> tuple[str, str]:
    """Map (retrieval strength, support verdict) to (label, one-line reason).

    This literal if/elif tree IS the documented rule table from the module
    docstring. Keeping it as explicit branches rather than a score means a
    reader (or an interviewer) can trace any label in one step.
    """
    strong = strength["strong"]

    if verdict == "SUPPORTED":
        if strong:
            return "High", "the retrieved passages are closely related and every claim is backed by them"
        # Supported but only loosely retrieved: the answer is faithful to
        # passages that may not be the right passages, so it cannot be High.
        return "Medium", "every claim is backed by the passages, but the passages were only moderately related to the question"

    if verdict == "PARTIAL":
        if strong:
            return "Medium", "the passages are relevant but support only part of the answer"
        return "Low", "retrieved passages were only weakly related and supported only part of the answer"

    if verdict == "UNSUPPORTED":
        # Faithfulness beats retrieval confidence: a fluent, well-retrieved
        # answer that adds an unsupported fact is still a wrong answer.
        return "Low", "the retrieved passages do not support the answer's claims"

    # verdict is None: the check could not run or was unparseable. Cap the
    # label at Medium, because only one of the two signals is available.
    if strong:
        return "Medium", "the support check was unavailable, so this rests on strong retrieval alone"
    return "Low", "the support check was unavailable and retrieval was weak"


# ---------------------------------------------------------------------------
# Public entry point used by main.py
# ---------------------------------------------------------------------------
def is_refusal(answer: str) -> bool:
    """True if the answer is the fixed 'I don't know' sentence from config.

    Substring, case-insensitive: the model sometimes appends a Sources line, and
    matching loosely means a refusal is still recognised in that case.
    """
    return _REFUSAL_MARKER in (answer or "").lower()


def assess(question: str, answer: str, chunks: list, ask=None) -> dict:
    """Return {"label", "reason", "verdict", "top_score", "mean_score", "refused"}.

    WHY a dict and not a bare string: main.py stores the whole result in
    st.session_state so the label, its reason and the scores behind it survive a
    Streamlit rerun; recomputing would mean paying for the LLM call twice.

    Refusals get label "n/a" and NO LLM call. A refusal makes no claims, so
    there is nothing for a grounding check to verify; labelling an honest
    "I don't know" as Low (or High) would be a category error. Skipping the
    call also keeps the "one extra call per answer" budget for real answers.
    """
    scores = [
        chunk.get("score") if isinstance(chunk, dict) else None for chunk in chunks
    ]
    strength = retrieval_strength(scores)

    if is_refusal(answer):
        return {
            "label": "n/a",
            "reason": "the system declined to answer, so there are no claims to verify",
            "verdict": None,
            "top_score": strength["top"],
            "mean_score": strength["mean"],
            "refused": True,
        }

    # Retrieved context is rebuilt from the same chunk dicts the UI shows, so
    # the checker sees exactly what the answering model saw.
    context = "\n\n".join(
        f"Source: {chunk.get('citation') or chunk.get('filename', 'unknown')}\n"
        f"{chunk.get('text', '')}"
        for chunk in chunks
        if isinstance(chunk, dict)
    )
    verdict = check_support(question, context, answer, ask=ask)
    label, reason = combine_labels(strength, verdict)
    return {
        "label": label,
        "reason": reason,
        "verdict": verdict,
        "top_score": strength["top"],
        "mean_score": strength["mean"],
        "refused": False,
    }


# ---------------------------------------------------------------------------
# Refusal measurement (python trust.py)
# ---------------------------------------------------------------------------
def _answer_for_eval(question: str, store, ask) -> str:
    """Answer one question with the production index and prompt.

    Uses the same PROMPT_TEMPLATE / TOP_K as main.py so the refusal rate measures
    the system the user actually chats with, not a second prompt that happens to
    behave differently.
    """
    from config import citation

    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(
        f"Source: {citation(doc.metadata)}\n{doc.page_content}" for doc in docs
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return ask(prompt)


def run_refusal_eval(ask=None, mock: bool = False) -> dict:
    """Measure correct refusals and false refusals over eval_questions.json.

    Two numbers are reported, on two disjoint groups:
      - correct refusal rate: answers refused on questions flagged
        should_refuse (off-topic / not answerable from the corpus);
      - false refusal rate:   answers refused on the normal, answerable
        questions, where a refusal is a failure.

    Both are printed as x/y = percentage. Refusal is detected by is_refusal(),
    i.e. by the fixed NO_ANSWER_TEXT sentence -- the same definition the prompt
    instructs the model to follow. With mock=True, fixed fixture responses test
    only the metric bookkeeping and must not be reported as live model results.
    """
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    store = None
    if not mock:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=str(CHROMA_DIR),
        )
        if store._collection.count() == 0:
            raise SystemExit(
                f"{CHROMA_DIR} is empty. Run `python ingest.py` before measuring refusals."
            )
        ask = ask or _default_ask
    groups = {
        # should_refuse=True -> refusing is the correct behaviour.
        "off-topic (should refuse)": [q for q in questions if q.get("should_refuse")],
        # should_refuse=False/absent -> refusing is a false refusal.
        "answerable (must not refuse)": [q for q in questions if not q.get("should_refuse")],
    }

    if mock:
        print(
            "MOCKED EVALUATION ONLY: fixture answers follow the expected labels; "
            "these counts do not measure live model behavior.",
            flush=True,
        )

    results = {}
    for name, group in groups.items():
        # Guard against an empty group so the division below can never be /0.
        if not group:
            results[name] = {"refused": 0, "total": 0, "rate": None, "wrong_ids": []}
            continue
        print(f"\n[{name}] {len(group)} questions", flush=True)
        refused_ids = []
        for i, q in enumerate(group, start=1):
            if mock:
                # The fixture checks the counting/reporting path without
                # pretending to predict how Groq would answer these questions.
                answer = (
                    NO_ANSWER_TEXT
                    if q.get("should_refuse")
                    else "Mock grounded answer; no refusal."
                )
            else:
                assert store is not None
                assert ask is not None
                answer = _answer_for_eval(q["question"], store, ask)
            refused = is_refusal(answer)
            if refused:
                refused_ids.append(q["id"])
            print(f"  [{i}/{len(group)}] {q['id']} refused={refused}", flush=True)
        results[name] = {
            "refused": len(refused_ids),
            "total": len(group),
            "rate": round(len(refused_ids) / len(group), 4),
            "ids": refused_ids,
        }

    print("\n" + "=" * 74)
    heading = (
        "MOCKED REFUSAL METRIC CHECK (not live system performance)"
        if mock
        else "REFUSAL MEASUREMENT (n is small; treat these as rough estimates)"
    )
    print(heading)
    print("=" * 74)
    for name, block in results.items():
        rate = block["rate"]
        shown = f"{rate:.0%}" if rate is not None else "n/a"
        print(f"{name}: {block['refused']}/{block['total']} = {shown}")
    print(
        "\nNote: a 'correct' refusal only means the model emitted the fixed "
        "NO_ANSWER_TEXT sentence. It is not a judgement that refusing was the "
        "right call for that question."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # One flag only, because refusal measurement is the only thing this module
    # does offline; the label itself is exercised through the app.
    parser.add_argument(
        "--refusal-eval",
        action="store_true",
        help="measure correct refusals and false refusals (this is the default)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="exercise refusal metric bookkeeping with fixture responses; no API or retrieval",
    )
    args = parser.parse_args()
    run_refusal_eval(mock=args.mock)


if __name__ == "__main__":
    main()
