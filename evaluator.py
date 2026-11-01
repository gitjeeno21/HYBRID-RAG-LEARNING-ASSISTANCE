"""Measure retrieval and answer quality of the RAG pipeline.

Two phases, deliberately separated:

1. RETRIEVAL (deterministic, no API calls). For each chunking configuration
   (4 chunk sizes x 2 overlaps) build an index over the FULL corpus -- both
   PDFs -- and measure retrieval hit rate on the 20 questions in
   eval_questions.json. A hit means: at least one of the top-k chunks comes
   from the expected PDF AND contains at least one expected keyword.
   Because this phase is deterministic and free, it runs for all 8 configs.

2. JUDGING (LLM calls). Only the top 3 configs by hit rate are answered and
   judged, RUNS_PER_CONFIG times each. For every run we average the LLM's
   1-5 correctness/faithfulness score over the 20 questions; the config's
   score is the mean of the runs, with the standard deviation reported as a
   stability signal.

Results are written to eval_results.json after every step, so a crash or a
rate limit never loses completed work, and ingest.py / app.py read
"best_config" from that file.

Usage:
    python evaluator.py                  # both phases
    python evaluator.py --retrieval-only # free, no LLM calls
    python evaluator.py --rebuild        # discard cached indexes and re-embed

Limitations of LLM-as-judge (read this before quoting the numbers):
- The same model family both writes and grades the answer, so it tends to
  prefer its own phrasing and often cannot catch a wrong fact that is stated
  confidently and matches the retrieved context.
- A single integer per question is coarse: 3 vs 4 is frequently noise. We run
  twice and report the standard deviation, but two runs is a weak estimate of
  variance, not a confidence interval.
- Judging is non-deterministic even at temperature 0 (hardware and batching
  differences), so scores are not bit-for-bit reproducible.
- The judge only sees the retrieved chunks, not the whole document. A faithful
  answer to a bad chunk can therefore score highly.
- Reference answers are hand-written; where they are incomplete or where the
  paper contradicts itself, a correct alternative answer may be marked down.
- Hit rate measures whether the right page was retrieved, not whether the
  answer is right, and keywords are a coarse proxy for "right page".
"""

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    LLM_MODEL,
    PROMPT_TEMPLATE,
    TOP_K,
    citation,
)
from ingest import load_documents

load_dotenv()

# The Windows console defaults to cp1252, which cannot encode some characters
# found in the PDF text (e.g. Greek letters) and would crash the print calls.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS_PATH = BASE_DIR / "eval_questions.json"

# The grid from the brief. 8 configs x full corpus is the whole point: chunk
# size chosen on one paper did not necessarily generalise to two.
CHUNK_SIZES = [300, 500, 1000, 1500]
CHUNK_OVERLAPS = [50, 150]

# Only the best configs earn LLM calls, because judging is the expensive part.
TOP_CONFIGS_TO_JUDGE = 3
RUNS_PER_CONFIG = 2

# Rate-limit handling: Groq's free tier throttles aggressively, so wait and
# retry instead of throwing away a multi-minute experiment.
MAX_ATTEMPTS = 4
BASE_SLEEP_SECONDS = 5  # 5s, 10s, 15s between attempts

JUDGE_PROMPT_TEMPLATE = (
    "You are a strict grader for a retrieval-augmented question answering "
    "system. Score the ANSWER from 1 to 5.\n"
    "Judge two things together:\n"
    "(a) Is the answer factually correct for the question?\n"
    "(b) Is it supported by the RETRIEVED CONTEXT, without invented facts?\n"
    "5 = correct and fully supported; 4 = correct with a minor imprecision; "
    "3 = partly correct or missing a key part; 2 = mostly wrong but contains "
    "a relevant element; 1 = wrong, unsupported, or an unjustified refusal.\n\n"
    "Question: {question}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Reference answer (written by a human; may be less complete than a good "
    "model answer, so do not penalise extra correct detail):\n{reference}\n\n"
    "Answer to score:\n{answer}\n\n"
    "Reply with only the line: SCORE: <1-5>"
)

client = Groq()  # reads GROQ_API_KEY from .env


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once; the grid would otherwise reload it 8x."""
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def index_path(chunk_size: int, chunk_overlap: int):
    """Directory for one config's throwaway index (kept apart from chroma_db)."""
    return BASE_DIR / f"chroma_db_eval_{chunk_size}_{chunk_overlap}"


def questions_hash(questions: list) -> str:
    """Fingerprint the questions so cached hit rates cannot go stale.

    The hash covers the text, keywords and expected source, not just the ids:
    editing a question's wording changes what is retrieved, and reusing the
    old numbers after such an edit would report a measurement that no longer
    matches eval_questions.json.
    """
    blob = json.dumps(
        [
            [q["id"], q["question"], q["expected_source"], q["expected_keywords"]]
            for q in questions
        ],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def parse_score(reply: str) -> int | None:
    """Extract a 1-5 score from the judge's reply, or None if absent.

    Models sometimes wrap the number in prose ("The score is 4/5"), so we
    look for the requested label first and only then for any lone digit.
    """
    match = re.search(r"SCORE[:\s]*([1-5])", reply, re.IGNORECASE) or re.search(
        r"\b([1-5])\b", reply
    )
    return int(match.group(1)) if match else None


def call_llm(prompt: str, temperature: float = 0.0, label: str = "llm") -> str:
    """Send one prompt to Groq, retrying on rate limits and transient errors.

    WHY: a 240-call evaluation will hit free-tier throttling. Without retries
    the run aborts an hour in; with them it slows down and finishes.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            # 429: waiting is the only correct response, retrying immediately
            # just burns another unit of quota.
            if attempt == MAX_ATTEMPTS:
                raise
            wait = BASE_SLEEP_SECONDS * attempt
            print(f"    [rate limit] {label}: waiting {wait}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = BASE_SLEEP_SECONDS * attempt
            print(f"    [api error] {label}: {type(exc).__name__}, retrying in "
                  f"{wait}s (attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def build_chunks(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """Re-split the already-loaded pages for one configuration."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def get_vector_store(
    documents: list,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
) -> Chroma:
    """Return this config's index, building it only when it is missing.

    Reusing the on-disk index matters on a CPU-only machine: re-embedding all
    eight configs costs many minutes, and grading one config should not force
    re-embedding the other seven.
    """
    path = index_path(chunk_size, chunk_overlap)
    collection_name = f"eval_{chunk_size}_{chunk_overlap}"

    if not rebuild and path.is_dir():
        store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(path),
        )
        if store._collection.count() > 0:
            return store

    # A half-written or stale index would silently skew every hit rate, so
    # wipe the directory and rebuild rather than adding on top of it.
    if path.is_dir():
        shutil.rmtree(path)
    chunks = build_chunks(documents, chunk_size, chunk_overlap)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )
    store.add_documents(chunks)
    print(f"    indexed {len(chunks)} chunks -> {path.name}", flush=True)
    return store


def build_context(docs: list) -> str:
    """Label retrieved chunks with "filename, p. X" so the model can cite pages."""
    return "\n\n".join(
        f"Source: {citation(doc.metadata)}\n{doc.page_content}" for doc in docs
    )


# ---------------------------------------------------------------------------
# Phase 1: retrieval hit rate (no LLM)
# ---------------------------------------------------------------------------
def question_is_hit(store: Chroma, question: dict) -> bool:
    """True if one top-k chunk is from the expected PDF and has a keyword."""
    docs = store.similarity_search(question["question"], k=TOP_K)
    for doc in docs:
        metadata = doc.metadata
        filename = metadata.get("filename") or ""
        if filename != question["expected_source"]:
            continue
        text = doc.page_content.lower()
        if any(kw.lower() in text for kw in question["expected_keywords"]):
            return True
    return False


def evaluate_retrieval(store: Chroma, questions: list) -> dict:
    """Compute hit rate over all questions for one config."""
    hits = []
    for i, question in enumerate(questions, start=1):
        hit = question_is_hit(store, question)
        hits.append(hit)
        print(f"    [{i}/{len(questions)}] {question['id']} hit={hit}", flush=True)
    missed = [q["id"] for q, hit in zip(questions, hits) if not hit]
    return {
        "hits": sum(hits),
        "hit_rate": round(sum(hits) / len(questions), 4),
        "missed_questions": missed,
    }


# ---------------------------------------------------------------------------
# Phase 2: answer + judge (LLM)
# ---------------------------------------------------------------------------
def answer_question(store: Chroma, question: str) -> tuple[str, str]:
    """Return (retrieved context, answer) for one question."""
    docs = store.similarity_search(question, k=TOP_K)
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return context, call_llm(prompt, temperature=0.2, label="answer")


def judge_answer(question: dict, context: str, answer: str) -> int | None:
    """Score an answer 1-5 against the question, context and reference."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question["question"],
        context=context,
        reference=question["reference_answer"],
        answer=answer,
    )
    reply = call_llm(prompt, temperature=0.0, label="judge")
    score = parse_score(reply)
    if score is None:
        # Skip, don't guess: inventing a score would silently corrupt the mean.
        print(f"    could not parse judge reply: {reply[:120]!r}", flush=True)
    return score


def run_judging(store: Chroma, questions: list, run_index: int) -> dict:
    """Answer and judge all questions once; returns scores keyed by question id."""
    scores = {}
    for i, question in enumerate(questions, start=1):
        context, answer = answer_question(store, question["question"])
        score = judge_answer(question, context, answer)
        if score is not None:
            scores[question["id"]] = score
        shown = score if score is not None else "n/a"
        print(f"    [{i}/{len(questions)}] {question['id']} score={shown}", flush=True)
    values = list(scores.values())
    return {
        "run": run_index,
        "mean": round(statistics.mean(values), 3) if values else None,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Results handling
# ---------------------------------------------------------------------------
def load_results() -> dict:
    """Load previous results so an interrupted run can resume."""
    if EVAL_RESULTS_PATH.exists():
        try:
            return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Warning: eval_results.json is corrupt; starting fresh.")
    return {}


def save_results(results: dict) -> None:
    """Persist after every step; a crash must not lose finished configs."""
    EVAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def rank_key(config: dict) -> tuple:
    """Sort key: hit rate first, then judge score (the brief's ranking rule)."""
    hit = config.get("retrieval", {}).get("hit_rate", 0.0)
    judge = (config.get("judge") or {}).get("mean")
    return (
        -hit,
        -(judge if judge is not None else -1.0),
        -config["chunk_size"],
        -config["chunk_overlap"],
    )


def pick_best(configs: list) -> dict | None:
    """Best config by hit rate, then judge score."""
    judged = [c for c in configs if c.get("judge")]
    pool = judged or configs
    return sorted(pool, key=rank_key)[0] if pool else None


def summarize_configs(configs: list) -> dict:
    """Build the compact best_config block that ingest.py and app.py read."""
    best = pick_best(configs)
    if best is None:
        return {}
    return {
        "chunk_size": best["chunk_size"],
        "chunk_overlap": best["chunk_overlap"],
        "hit_rate": best.get("retrieval", {}).get("hit_rate"),
        "judge_mean": (best.get("judge") or {}).get("mean"),
        "judge_std": (best.get("judge") or {}).get("std"),
        "indexed_chunks": best.get("indexed_chunks"),
    }


def print_table(configs: list) -> None:
    """Print the results as a markdown table for pasting into the README."""
    print("\n| chunk_size | overlap | chunks | hit rate | judge mean | judge std |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in sorted(configs, key=lambda c: (c["chunk_size"], c["chunk_overlap"])):
        hit = c.get("retrieval", {}).get("hit_rate")
        judge_block = c.get("judge") or {}
        mean = judge_block.get("mean")
        std = judge_block.get("std")
        print(
            f"| {c['chunk_size']} | {c['chunk_overlap']} | {c['indexed_chunks']} "
            f"| {f'{hit:.0%}' if hit is not None else '-'} "
            f"| {f'{mean:.2f}' if mean is not None else 'not judged'} "
            f"| {f'{std:.2f}' if std is not None else '-'} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="only compute hit rates (no LLM calls, no API quota)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete cached indexes and re-embed every configuration",
    )
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    q_hash = questions_hash(questions)
    previous = load_results()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_hash": q_hash,
        "num_questions": len(questions),
        "top_k": TOP_K,
        "runs_per_config": RUNS_PER_CONFIG,
        "chunk_sizes": CHUNK_SIZES,
        "chunk_overlaps": CHUNK_OVERLAPS,
        "configs": [],
    }

    # Reuse retrieval numbers only when the question set is unchanged AND the
    # index on disk still makes the answer deterministic.
    cached = {}
    if previous.get("questions_hash") == q_hash and not args.rebuild:
        for config in previous.get("configs", []):
            key = (config["chunk_size"], config["chunk_overlap"])
            if config.get("retrieval") and index_path(*key).is_dir():
                # Keep only the free (retrieval) half: judge scores must come
                # from this run so the table always reflects the current code.
                cached[key] = {k: v for k, v in config.items() if k != "judge"}

    documents = None  # lazily loaded: --retrieval-only on cached data needs no PDFs

    # --- Phase 1: deterministic retrieval over all 8 configs ----------------
    print("=" * 74)
    print(f"PHASE 1 - retrieval hit rate (no LLM), {len(questions)} questions")
    print("=" * 74)
    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            key = (chunk_size, chunk_overlap)
            print(f"\n[{chunk_size}/{chunk_overlap}]", flush=True)

            if key in cached:
                config = dict(cached[key])
                print(f"    reusing cached hit rate {config['retrieval']['hit_rate']:.0%}",
                      flush=True)
            else:
                if documents is None:
                    documents = load_documents()
                store = get_vector_store(
                    documents, chunk_size, chunk_overlap, rebuild=args.rebuild
                )
                config = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "indexed_chunks": store._collection.count(),
                    "retrieval": evaluate_retrieval(store, questions),
                }
            results["configs"].append(config)
            results["best_config"] = summarize_configs(results["configs"])
            save_results(results)  # partial results survive a crash

    if args.retrieval_only:
        print_table(results["configs"])
        print(f"\nSaved to {EVAL_RESULTS_PATH} (retrieval only, no judge scores).")
        return

    # --- Phase 2: judge only the top configs --------------------------------
    ranked = sorted(results["configs"], key=rank_key)
    judged = ranked[:TOP_CONFIGS_TO_JUDGE]
    print("\n" + "=" * 74)
    print(f"PHASE 2 - answering + judging top {len(judged)} configs by hit rate")
    print(f"({RUNS_PER_CONFIG} runs x {len(questions)} questions each)")
    print("=" * 74)

    if documents is None:  # every retrieval number came from cache
        documents = load_documents()

    for config in judged:
        chunk_size, chunk_overlap = config["chunk_size"], config["chunk_overlap"]
        print(f"\n[{chunk_size}/{chunk_overlap}] judge phase", flush=True)
        store = get_vector_store(documents, chunk_size, chunk_overlap, rebuild=args.rebuild)

        run_means = []
        per_run = []
        for run_index in range(1, RUNS_PER_CONFIG + 1):
            print(f"  run {run_index}/{RUNS_PER_CONFIG}", flush=True)
            run = run_judging(store, questions, run_index)
            per_run.append(run)
            if run["mean"] is not None:
                run_means.append(run["mean"])
            config["judge"] = {
                "runs": per_run,
                "mean": round(statistics.mean(run_means), 3) if run_means else None,
                "std": (round(statistics.stdev(run_means), 3)
                        if len(run_means) > 1 else None),
            }
            results["best_config"] = summarize_configs(results["configs"])
            save_results(results)  # save after each run, not just each config

        print(f"  -> mean {config['judge']['mean']} std {config['judge']['std']}",
              flush=True)

    results["best_config"] = summarize_configs(results["configs"])
    results["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_results(results)

    print_table(results["configs"])
    best = results["best_config"]
    print(
        f"\nBest config: chunk_size={best['chunk_size']}, "
        f"overlap={best['chunk_overlap']} "
        f"(hit rate {best['hit_rate']:.0%}, judge {best['judge_mean']})"
    )
    print(f"Saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
