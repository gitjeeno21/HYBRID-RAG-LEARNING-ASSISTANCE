"""Does the gold evidence actually reach the prompt? (zero LLM calls)

Retrieval hit rate (eval/run_eval.py) asks whether a chunk from the gold PAGE
is ranked in the top k. That is not the same as the model seeing the answer:
a page usually holds two or more chunks, and whatever runs between ranking and
prompt-building (de-duplication, filtering) can still drop the one that matters.
This script measures the end of that path: it builds the exact passage list the
app sends (main.retrieve_for_api), then checks whether each question's gold
evidence quotes appear in it, after whitespace/case normalisation.

  any-quote coverage: at least one evidence quote is in the prompt
  all-quote coverage: every quote is (the bar for multi-hop questions)

Quotes are matched as literal substrings, so a PDF extraction difference can
cause a false "missing"; treat the numbers as a lower bound.

Usage:
    python eval/context_coverage.py                    # the live app's behaviour
    python eval/context_coverage.py --dedup citation   # the pre-Stage-1 behaviour
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["GROQ_API_KEY"] = "context-coverage-makes-no-llm-calls"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import config  # noqa: E402
import main  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "eval" / "heldout_v2_questions.json"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def legacy_citation_dedup(retrieved: list[dict]) -> list[dict]:
    """The pre-Stage-1 rule: keep only the first passage per 'file, p. X'."""
    seen, kept = set(), []
    for item in retrieved:
        if item["citation"] not in seen:
            seen.add(item["citation"])
            kept.append(item)
    return kept


def coverage(questions: list[dict], top_k: int, dedup: str) -> dict:
    rows = []
    for question in questions:
        retrieved = main.retrieve_for_api(question["q"], [], top_k)
        if dedup == "citation":
            retrieved = legacy_citation_dedup(retrieved)
        context = normalize(" ".join(item["text"] for item in retrieved))
        found = [normalize(quote) in context for quote in question["evidence"]]
        rows.append({"id": question["id"], "type": question["type"],
                     "passages": len(retrieved), "any": any(found), "all": all(found)})
    n = len(rows)
    return {
        "questions": n,
        "top_k": top_k,
        "dedup": dedup,
        "mean_passages": round(sum(r["passages"] for r in rows) / n, 2) if n else None,
        "any_quote": sum(r["any"] for r in rows),
        "all_quotes": sum(r["all"] for r in rows),
        "missing_all": [r["id"] for r in rows if not r["all"]],
        "rows": rows,
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--dedup", choices=["chunk", "citation"], default="chunk",
                        help="chunk = live behaviour; citation = pre-Stage-1 behaviour")
    parser.add_argument("--json", type=Path, help="also write the full result here")
    args = parser.parse_args()

    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8"))
                 if q.get("type") != "unanswerable" and q.get("evidence")]
    result = coverage(questions, args.top_k, args.dedup)
    n = result["questions"]
    print(f"{n} answerable questions, top-k {args.top_k}, dedup by {args.dedup}")
    print(f"  passages sent per question: {result['mean_passages']}")
    print(f"  any evidence quote in prompt: {result['any_quote']}/{n}")
    print(f"  all evidence quotes in prompt: {result['all_quotes']}/{n}")
    print(f"  missing: {', '.join(result['missing_all']) or 'none'}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main_cli()
