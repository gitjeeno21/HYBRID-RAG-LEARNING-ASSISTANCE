"""One-shot judging script: answers + judges the selected configs, 2 runs each.

Configs to judge (top 3 by hit rate, which already includes 500/50):
  - 1500 / 50   (hit rate 80%)
  - 1500 / 150  (hit rate 80%)
  -  500 / 50   (hit rate 75%)

Saves eval_results.json after every run. Retries on Groq rate limits.
"""

import json
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
    TOP_K,
    citation,
)

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -----------------------------------------------------------------------
# The specific configs to judge: top 3 by retrieval hit rate.
# 500/50 is already #3 so "top-3 + 500/50" collapses to this list.
CONFIGS_TO_JUDGE = [
    (1500, 50),
    (1500, 150),
    (500, 50),
]
RUNS_PER_CONFIG = 2

MAX_ATTEMPTS = 6
BASE_SLEEP = 8  # seconds; doubles on each retry

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

client = Groq()

# -----------------------------------------------------------------------

def load_results():
    if EVAL_RESULTS_PATH.exists():
        return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Cannot find {EVAL_RESULTS_PATH}")


def save_results(results):
    EVAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  [saved {EVAL_RESULTS_PATH.name}]", flush=True)


def call_llm(prompt, temperature=0.0, label="llm"):
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
                raise
            wait = BASE_SLEEP * (2 ** (attempt - 1))
            print(f"    [rate-limit] {label}: sleeping {wait}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = BASE_SLEEP * attempt
            print(f"    [api-error] {label}: {type(exc).__name__}, "
                  f"retrying in {wait}s (attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_score(reply):
    import re
    m = re.search(r"SCORE[:\s]*([1-5])", reply, re.IGNORECASE) or re.search(r"\b([1-5])\b", reply)
    return int(m.group(1)) if m else None


def get_store(chunk_size, chunk_overlap):
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
            f"Index at {path} is empty! Run `python evaluator.py --retrieval-only` first."
        )
    print(f"  loaded index ({count} chunks) from {path.name}", flush=True)
    return store


def build_context(docs):
    return "\n\n".join(
        f"Source: {citation(doc.metadata)}\n{doc.page_content}" for doc in docs
    )


def run_one_config(store, questions, chunk_size, chunk_overlap):
    """Run RUNS_PER_CONFIG judging passes for one config, return updated config dict."""
    run_means = []
    per_run = []

    for run_index in range(1, RUNS_PER_CONFIG + 1):
        print(f"\n  --- run {run_index}/{RUNS_PER_CONFIG} ---", flush=True)
        scores = {}
        for i, q in enumerate(questions, start=1):
            # Answer
            docs = store.similarity_search(q["question"], k=TOP_K)
            context = build_context(docs)
            prompt_ans = PROMPT_TEMPLATE.format(context=context, question=q["question"])
            answer = call_llm(prompt_ans, temperature=0.2, label=f"answer q{i}")

            # Judge
            prompt_judge = JUDGE_PROMPT.format(
                question=q["question"],
                context=context,
                reference=q["reference_answer"],
                answer=answer,
            )
            reply = call_llm(prompt_judge, temperature=0.0, label=f"judge q{i}")
            score = parse_score(reply)
            if score is not None:
                scores[q["id"]] = score
            shown = score if score is not None else "n/a"
            print(f"    [{i}/{len(questions)}] {q['id']} score={shown}", flush=True)

        values = list(scores.values())
        run_rec = {
            "run": run_index,
            "mean": round(statistics.mean(values), 3) if values else None,
            "scores": scores,
        }
        per_run.append(run_rec)
        if run_rec["mean"] is not None:
            run_means.append(run_rec["mean"])
        print(f"  run {run_index} mean={run_rec['mean']}", flush=True)
        yield per_run, run_means  # caller saves after each run


def summarize_configs(configs):
    """Recompute best_config block from all configs."""
    def rank(c):
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


def print_table(configs):
    print("\n| chunk_size | chunk_overlap | hit_rate | judge_mean | judge_std |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for c in sorted(configs, key=lambda c: (c["chunk_size"], c["chunk_overlap"])):
        hit = c.get("retrieval", {}).get("hit_rate")
        jb = c.get("judge") or {}
        mean = jb.get("mean")
        std = jb.get("std")
        print(
            f"| {c['chunk_size']} | {c['chunk_overlap']} "
            f"| {f'{hit:.0%}' if hit is not None else '-'} "
            f"| {f'{mean:.2f}' if mean is not None else 'not judged'} "
            f"| {f'{std:.3f}' if std is not None else '-'} |"
        )


def main():
    questions = json.loads((BASE_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions.")

    results = load_results()

    for chunk_size, chunk_overlap in CONFIGS_TO_JUDGE:
        print(f"\n{'='*60}", flush=True)
        print(f"Judging config {chunk_size}/{chunk_overlap}", flush=True)
        print(f"{'='*60}", flush=True)

        # Find this config in the results list
        config = next(
            (c for c in results["configs"]
             if c["chunk_size"] == chunk_size and c["chunk_overlap"] == chunk_overlap),
            None,
        )
        if config is None:
            print(f"  WARNING: config {chunk_size}/{chunk_overlap} not found in "
                  f"eval_results.json — skipping", flush=True)
            continue

        try:
            store = get_store(chunk_size, chunk_overlap)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", flush=True)
            continue

        try:
            for per_run, run_means in run_one_config(store, questions, chunk_size, chunk_overlap):
                config["judge"] = {
                    "runs": per_run,
                    "mean": round(statistics.mean(run_means), 3) if run_means else None,
                    "std": (round(statistics.stdev(run_means), 3)
                            if len(run_means) > 1 else None),
                }
                results["best_config"] = summarize_configs(results["configs"])
                save_results(results)
        except RateLimitError:
            print(f"\n[STOPPED] Persistent rate-limit on config {chunk_size}/{chunk_overlap}. "
                  f"Results saved so far.", flush=True)
            print_table(results["configs"])
            sys.exit(1)

        print(f"  -> mean={config['judge']['mean']} std={config['judge']['std']}", flush=True)

    print("\n\nAll selected configs judged.")
    print_table(results["configs"])
    best = results["best_config"]
    print(f"\nBest config: chunk_size={best['chunk_size']}, "
          f"overlap={best['chunk_overlap']} "
          f"(hit_rate={best['hit_rate']:.0%}, judge_mean={best['judge_mean']})")
    print(f"Saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()

