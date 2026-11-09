"""Answer + judge the selected chunking configs and persist scores.

Configs judged (top 3 by retrieval hit rate):
  - 1500 / 50   (hit rate 80%)
  - 1500 / 150  (hit rate 80%)
  -  500 / 50   (hit rate 75%)

SCOPE: judge scores are computed only over the 20 *answerable* training
questions (ids q01-q20).  The 10 held-out questions (q21-q30) and the 6
refusal probes (t01-t06) are intentionally excluded so the numbers stay
comparable across runs and cannot leak eval signal from the held-out set.

RESUME LOGIC: on startup the script reads every config's existing judge
runs from eval_results.json and skips runs that are already saved.  This
means an interrupted run (e.g. due to a rate-limit) can be restarted and
it will continue from the next missing run rather than overwriting good data.

RATE-LIMIT HANDLING: on a persistent RateLimitError the script saves
whatever it has completed so far and exits with code 1, printing which
configs were finished and which are still pending.

Usage:
    python run_judging.py           # resume / run remaining judging
    python run_judging.py --dry-run # print plan without calling any API
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    BASE_DIR,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    LLM_MODEL,
    PROMPT_TEMPLATE,
    LEGACY_EVAL_TOP_K,
    citation,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Configs to judge, in priority order.
CONFIGS_TO_JUDGE = [
    (1500, 50),
    (1500, 150),
    (500, 50),
]

# Target number of independent judging runs per config.
RUNS_PER_CONFIG = 2

# Only these question ids are used for judge scoring.
# WHY exclude q21-q30 and t01-t06: q21-q30 are held-out questions whose
# answers we deliberately do not train on, so using them for judge scores
# would leak information.  t01-t06 test the *refusal* path, not answer
# quality; the 1-5 scale is not meaningful for them.
JUDGE_QUESTION_IDS = {f"q{i:02d}" for i in range(1, 21)}  # q01 … q20

MAX_ATTEMPTS = 6
BASE_SLEEP = 8  # seconds; doubled on each successive retry

JUDGE_PROMPT = (
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

# ---------------------------------------------------------------------------
# Question helpers
# ---------------------------------------------------------------------------

def load_judge_questions() -> list[dict]:
    """Load eval_questions.json and return only the q01-q20 answerable set.

    WHY filter here rather than in eval_questions.json: the full file is also
    consumed by evaluator.py (which runs retrieval over all questions including
    held-out ones).  Filtering at load time keeps both consumers correct
    without duplicating or editing the source file.
    """
    all_qs = json.loads(
        (BASE_DIR / "eval_questions.json").read_text(encoding="utf-8")
    )
    judge_qs = [q for q in all_qs if q["id"] in JUDGE_QUESTION_IDS]
    return judge_qs


def judge_questions_hash(questions: list[dict]) -> str:
    """Fingerprint the q01-q20 question set for cache invalidation.

    WHY hash only q01-q20: the hash is stored in eval_results.json alongside
    the judge scores to detect if those questions have been edited since the
    scores were recorded.  Including q21-q30 or t01-t06 would cause a spurious
    hash mismatch whenever those questions are added or changed, even though
    the judge scores were never computed from them.
    """
    blob = json.dumps(
        [
            [q["id"], q["question"], q.get("reference_answer", "")]
            for q in sorted(questions, key=lambda q: q["id"])
        ],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results() -> dict:
    """Load eval_results.json, raising if absent."""
    if EVAL_RESULTS_PATH.exists():
        return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Cannot find {EVAL_RESULTS_PATH}")


def save_results(results: dict) -> None:
    """Persist results to disk.

    WHY save after every single run (not just after every config): if a
    rate-limit or crash interrupts a multi-run judging session, at most one
    run's worth of API calls is lost rather than all runs for that config.
    """
    EVAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  [saved {EVAL_RESULTS_PATH.name}]", flush=True)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(client: Groq, prompt: str, temperature: float = 0.0,
             label: str = "llm") -> str:
    """Send one prompt to Groq, retrying on transient errors.

    WHY exponential back-off (not a fixed delay): Groq's free tier enforces a
    token-per-minute limit.  A short fixed wait often isn't enough after a
    burst; doubling the wait on each retry gives the quota time to refill
    without wasting more time than necessary on the happy path.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == MAX_ATTEMPTS:
                raise  # caller handles persistent rate-limit
            wait = BASE_SLEEP * (2 ** (attempt - 1))
            print(
                f"    [rate-limit] {label}: sleeping {wait}s "
                f"(attempt {attempt}/{MAX_ATTEMPTS})",
                flush=True,
            )
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = BASE_SLEEP * attempt
            print(
                f"    [api-error] {label}: {type(exc).__name__}, "
                f"retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_score(reply: str) -> int | None:
    """Extract a 1-5 integer from the judge's reply, or None if absent."""
    m = (
        re.search(r"SCORE[:\s]*([1-5])", reply, re.IGNORECASE)
        or re.search(r"\b([1-5])\b", reply)
    )
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------

def get_store(chunk_size: int, chunk_overlap: int) -> Chroma:
    """Load the pre-built eval index for one config (built by evaluator.py).

    WHY not rebuild here: rebuilding takes several minutes per config and
    requires loading the PDFs again.  The eval indexes built by evaluator.py
    are stored on disk and reused here so judging is fast and the retrieval
    step is identical to what was measured during the eval grid search.
    """
    path = BASE_DIR / f"chroma_db_eval_{chunk_size}_{chunk_overlap}"
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma(
        collection_name=f"eval_{chunk_size}_{chunk_overlap}",
        embedding_function=embeddings,
        persist_directory=str(path),
    )
    count = store._collection.count()
    if count == 0:
        raise RuntimeError(
            f"Index at {path} is empty. "
            f"Run `python evaluator.py --retrieval-only` first."
        )
    print(f"  loaded index ({count} chunks) from {path.name}", flush=True)
    return store


def build_context(docs: list) -> str:
    return "\n\n".join(
        f"Source: {citation(doc.metadata)}\n{doc.page_content}" for doc in docs
    )


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def completed_run_indices(config: dict) -> set[int]:
    """Return the set of run indices already saved for this config.

    WHY read run indices rather than a simple count: if run 1 succeeded but
    run 2 failed mid-way, `len(runs) == 1` gives the same count as a case
    where run 1 was never attempted.  Checking the stored `run` field lets
    us skip exactly the runs we already have, even if they were saved out of
    order by a future parallel implementation.
    """
    judge = config.get("judge") or {}
    runs = judge.get("runs") or []
    return {r["run"] for r in runs if r.get("mean") is not None}


def runs_needed(config: dict) -> list[int]:
    """Return the list of run indices still to be executed for this config.

    WHY return a list in ascending order: the caller iterates this list and
    the run index is stored in the JSON; monotonic order makes the output
    easier to read and matches the original numbering.
    """
    done = completed_run_indices(config)
    return [r for r in range(1, RUNS_PER_CONFIG + 1) if r not in done]


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def summarize_configs(configs: list) -> dict:
    """Recompute the best_config block from all configs.

    WHY pick the judged pool first: once at least one config has a judge score,
    the best config should be the one with the highest *judge* mean (breaking
    ties by hit rate), not just the one with the best retrieval hit rate.
    """
    def rank(c: dict) -> tuple:
        hit = c.get("retrieval", {}).get("hit_rate", 0.0)
        j = (c.get("judge") or {}).get("mean")
        return (-hit, -(j if j is not None else -1.0), -c["chunk_size"], -c["chunk_overlap"])

    judged = [c for c in configs if c.get("judge")]
    pool = judged or configs
    best = sorted(pool, key=rank)[0]
    return {
        "chunk_size": best["chunk_size"],
        "chunk_overlap": best["chunk_overlap"],
        "hit_rate": best.get("retrieval", {}).get("hit_rate"),
        "judge_mean": (best.get("judge") or {}).get("mean"),
        "judge_std": (best.get("judge") or {}).get("std"),
        "indexed_chunks": best.get("indexed_chunks"),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(configs: list) -> None:
    print("\n| chunk_size | chunk_overlap | hit_rate | judge_mean | judge_std |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for c in sorted(configs, key=lambda c: (c["chunk_size"], c["chunk_overlap"])):
        hit = c.get("retrieval", {}).get("hit_rate")
        jb = c.get("judge") or {}
        mean = jb.get("mean")
        std = jb.get("std")
        n_runs = len(jb.get("runs") or [])
        run_note = f" ({n_runs}/{RUNS_PER_CONFIG} runs)" if jb else ""
        print(
            f"| {c['chunk_size']} | {c['chunk_overlap']} "
            f"| {f'{hit:.0%}' if hit is not None else '-'} "
            f"| {f'{mean:.2f}{run_note}' if mean is not None else 'not judged'} "
            f"| {f'{std:.3f}' if std is not None else '-'} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print which runs would be executed without calling any API",
    )
    parser.add_argument(
        "--only",
        metavar="SIZE/OVERLAP",
        help=(
            "restrict judging to a single config, e.g. --only 500/50. "
            "Must be one of: "
            + ", ".join(f"{s}/{o}" for s, o in CONFIGS_TO_JUDGE)
        ),
    )
    args = parser.parse_args()

    # --- Resolve --only filter --------------------------------------------
    # WHY validate here before touching the disk or the API: failing fast with
    # a clear message is better than running silently against a wrong config.
    if args.only is not None:
        # Accept "SIZE/OVERLAP" format, e.g. "500/50".
        try:
            size_str, overlap_str = args.only.split("/")
            only_key = (int(size_str), int(overlap_str))
        except ValueError:
            parser.error(
                f"--only value must be in SIZE/OVERLAP format, e.g. 500/50, "
                f"got: {args.only!r}"
            )
        if only_key not in CONFIGS_TO_JUDGE:
            valid = ", ".join(f"{s}/{o}" for s, o in CONFIGS_TO_JUDGE)
            parser.error(
                f"--only {args.only!r} is not a known config. "
                f"Valid options: {valid}"
            )
        # Restrict the working list to just the requested config.
        configs_to_run = [only_key]
        print(f"--only filter active: judging config {args.only} only", flush=True)
    else:
        configs_to_run = list(CONFIGS_TO_JUDGE)

    # --- Load and filter questions -----------------------------------------
    questions = load_judge_questions()
    q_hash = judge_questions_hash(questions)
    print(
        f"Judge question set: {len(questions)} questions "
        f"({', '.join(q['id'] for q in questions[:3])} … "
        f"{questions[-1]['id']}), hash={q_hash}",
        flush=True,
    )

    # --- Load existing results ---------------------------------------------
    results = load_results()

    # Stamp the judge scope into the top-level results dict so anyone reading
    # the file knows exactly which questions the judge scores cover.
    # WHY not overwrite existing scores: we only update metadata fields here;
    # the per-config judge blocks are only written after a run completes.
    results["judge_question_ids"] = sorted(JUDGE_QUESTION_IDS)
    results["judge_questions_hash"] = q_hash
    results["judge_num_questions"] = len(questions)

    # --- Dry-run: just print the plan and exit ----------------------------
    if args.dry_run:
        print("\n=== DRY RUN — no API calls will be made ===\n")
        any_pending = False
        for chunk_size, chunk_overlap in configs_to_run:
            config = next(
                (c for c in results["configs"]
                 if c["chunk_size"] == chunk_size and c["chunk_overlap"] == chunk_overlap),
                None,
            )
            if config is None:
                print(f"  [{chunk_size}/{chunk_overlap}] NOT FOUND in eval_results.json — skip")
                continue
            pending = runs_needed(config)
            done = completed_run_indices(config)
            if pending:
                any_pending = True
                print(
                    f"  [{chunk_size}/{chunk_overlap}] "
                    f"runs already done: {sorted(done) or 'none'}  "
                    f"→  would execute run(s): {pending}"
                )
            else:
                print(
                    f"  [{chunk_size}/{chunk_overlap}] "
                    f"all {RUNS_PER_CONFIG} runs already saved — nothing to do"
                )
        if not any_pending:
            print("\nNothing to do — all runs are complete.")
        print("\n=== END DRY RUN ===")
        return

    # --- Live judging run --------------------------------------------------
    # Only import Groq client when we are actually going to call the API so
    # a --dry-run never touches any credentials.
    groq_client = Groq()  # reads GROQ_API_KEY from .env

    for chunk_size, chunk_overlap in configs_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Judging config {chunk_size}/{chunk_overlap}", flush=True)
        print(f"{'='*60}", flush=True)

        config = next(
            (c for c in results["configs"]
             if c["chunk_size"] == chunk_size and c["chunk_overlap"] == chunk_overlap),
            None,
        )
        if config is None:
            print(
                f"  WARNING: config {chunk_size}/{chunk_overlap} not found "
                f"in eval_results.json — skipping",
                flush=True,
            )
            continue

        # Determine which runs still need to be done.
        pending = runs_needed(config)
        if not pending:
            print(f"  all {RUNS_PER_CONFIG} runs already saved — skipping", flush=True)
            continue
        done_set = completed_run_indices(config)
        print(
            f"  runs already saved: {sorted(done_set) or 'none'}  "
            f"→  will execute: {pending}",
            flush=True,
        )

        try:
            store = get_store(chunk_size, chunk_overlap)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", flush=True)
            continue

        # Initialise from whatever runs are already persisted.
        judge_block = config.get("judge") or {}
        existing_runs = list(judge_block.get("runs") or [])
        # run_means is rebuilt from all saved runs (existing + new) so the
        # aggregate mean/std across all RUNS_PER_CONFIG runs is correct.
        run_means = [
            r["mean"] for r in existing_runs if r.get("mean") is not None
        ]

        try:
            for run_index in pending:
                print(f"\n  --- run {run_index}/{RUNS_PER_CONFIG} ---", flush=True)
                scores: dict[str, int] = {}

                for i, q in enumerate(questions, start=1):
                    # Step 1: retrieve context.
                    docs = store.similarity_search(q["question"], k=LEGACY_EVAL_TOP_K)
                    context = build_context(docs)

                    # Step 2: generate an answer with the RAG prompt.
                    prompt_ans = PROMPT_TEMPLATE.format(
                        context=context, question=q["question"]
                    )
                    answer = call_llm(
                        groq_client, prompt_ans, temperature=0.2,
                        label=f"answer {q['id']}",
                    )

                    # Step 3: judge the answer on a 1-5 scale.
                    prompt_judge = JUDGE_PROMPT.format(
                        question=q["question"],
                        context=context,
                        reference=q["reference_answer"],
                        answer=answer,
                    )
                    reply = call_llm(
                        groq_client, prompt_judge, temperature=0.0,
                        label=f"judge {q['id']}",
                    )
                    score = parse_score(reply)
                    if score is not None:
                        scores[q["id"]] = score
                    shown = score if score is not None else "n/a"
                    print(
                        f"    [{i}/{len(questions)}] {q['id']} score={shown}",
                        flush=True,
                    )

                values = list(scores.values())
                run_rec = {
                    "run": run_index,
                    "mean": round(statistics.mean(values), 3) if values else None,
                    "scores": scores,
                }

                # Merge this run into the existing run list, replacing any
                # partial entry for the same run index.
                existing_runs = [r for r in existing_runs if r["run"] != run_index]
                existing_runs.append(run_rec)
                existing_runs.sort(key=lambda r: r["run"])

                if run_rec["mean"] is not None:
                    run_means = [
                        r["mean"] for r in existing_runs if r.get("mean") is not None
                    ]

                # Update the judge block with the merged run list.
                config["judge"] = {
                    "runs": existing_runs,
                    "mean": round(statistics.mean(run_means), 3) if run_means else None,
                    "std": (
                        round(statistics.stdev(run_means), 3)
                        if len(run_means) > 1
                        else None
                    ),
                }
                results["best_config"] = summarize_configs(results["configs"])

                # Save after every single run so a crash loses at most one run.
                save_results(results)
                print(f"  run {run_index} mean={run_rec['mean']}", flush=True)

        except RateLimitError:
            # Persistent rate-limit: save what we have and exit cleanly.
            print(
                f"\n[STOPPED] Persistent rate-limit on config "
                f"{chunk_size}/{chunk_overlap}. Results saved so far.",
                flush=True,
            )
            print_table(results["configs"])
            sys.exit(1)

        print(
            f"  -> mean={config['judge']['mean']} std={config['judge']['std']}",
            flush=True,
        )

    print("\n\nAll selected configs judged.")
    print_table(results["configs"])
    best = results["best_config"]
    print(
        f"\nBest config: chunk_size={best['chunk_size']}, "
        f"overlap={best['chunk_overlap']} "
        f"(hit_rate={best['hit_rate']:.0%}, judge_mean={best['judge_mean']})"
    )
    print(f"Saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
