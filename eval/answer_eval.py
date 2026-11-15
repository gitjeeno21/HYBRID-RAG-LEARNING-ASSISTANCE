"""Phase 4: answer quality and trust on the held-out v2 set.

What it measures, for every v2 question (answerable and unanswerable):
  1. Citation accuracy (no LLM): every (file, page) the answer cites must be one
     of the passages it was given; also how often a cited page is a gold page.
  2. Faithfulness (judge model): each answer sentence is checked against the
     retrieved passages.
  3. Correctness (judge model): the answer is graded against the reference
     answer: CORRECT / PARTIAL / INCORRECT. Unanswerable questions are correct
     only if the system refused.
  4. Calibration: how often High / Medium / Low labelled answers are correct.
  5. Refusal accuracy: correct refusals on unanswerable questions, false
     refusals on answerable ones.

Parity with the app: retrieval calls hybrid.hybrid_search_with_scores (the app's
function) and de-duplicates by chunk like main.retrieve_for_api; the prompt
is main._chat_prompt; answers use config.LLM_MODEL at temperature 0.2 like
/api/chat; the label is trust.assess with the app's checker model. The only
difference is the index: the eval index under eval/.index (no personal PDFs).

Quota: every model output is cached in eval/.cache/llm_cache.jsonl keyed by
(model, temperature, prompt), so a rerun or resume never pays twice. Calls are
paced under Groq's per-minute token limit. On the daily limit the script saves
what it has and exits with code 2; run it again later to resume.

Usage:
    python eval/answer_eval.py                 # top-k from config (5)
    python eval/answer_eval.py --top-k 3       # the pre-Phase-3 setting
    python eval/answer_eval.py --stages answer judge trust
    python eval/answer_eval.py --tag stage4   # fresh files: answers_v2_k5_stage4.json,
                                              # ANSWER_QUALITY_k5_stage4.md (no resume from
                                              # older answers made under different code)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import llm  # noqa: E402  (the Groq client + generation options the app uses)
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError  # noqa: E402

# Build the client NOW: importing run_eval below removes GROQ_API_KEY from the
# environment (its zero-LLM guard), and llm.get_client() is lazy and cached.
llm.get_client()
_groq_module = sys.modules["groq"]
import run_eval as H  # noqa: E402  (index helpers; it blocks groq for its own runs...)

sys.modules["groq"] = _groq_module  # ...so restore it: this script needs the model.

import config  # noqa: E402
from citations import check_citations, cited_pages, normalize  # noqa: E402
from hybrid import hybrid_search_with_scores, unique_chunks  # noqa: E402
from main import _chat_prompt  # noqa: E402
from trust import assess, is_refusal  # noqa: E402

QUESTIONS = H.EVAL_DIR / "heldout_v2_questions.json"
CACHE = H.EVAL_DIR / ".cache" / "llm_cache.jsonl"
ANSWER_MODEL = config.LLM_MODEL
JUDGE_MODEL = "openai/gpt-oss-20b"
TPM_BUDGET = 7_000          # stay under Groq's 8,000 tokens/minute per model


class QuotaExhausted(RuntimeError):
    """Groq's daily token or request limit was reached."""


# ---------------------------------------------------------------------------
# Cached, paced model calls
# ---------------------------------------------------------------------------
class LLM:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry
        self.window: dict[str, deque] = {}
        self.spent: Counter = Counter()

    def _pace(self, model: str, estimate: int) -> None:
        window = self.window.setdefault(model, deque())
        while True:
            now = time.time()
            while window and now - window[0][0] > 60:
                window.popleft()
            if sum(tokens for _, tokens in window) + estimate <= TPM_BUDGET or not window:
                return
            time.sleep(2)

    def ask(self, model: str, prompt: str, temperature: float, label: str) -> str:
        key = hashlib.sha256(json.dumps([model, temperature, prompt]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]["text"]
        self._pace(model, len(prompt) // 3 + 700)
        for attempt in range(8):
            try:
                response = llm.get_client().chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **llm.generation_options(model),
                )
                break
            except RateLimitError as exc:
                message = str(exc)
                if "per day" in message or "(TPD)" in message or "(RPD)" in message:
                    raise QuotaExhausted(message.split("Need more")[0][:300]) from exc
                wait = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
                seconds = (int(wait.group(1) or 0) * 60 + float(wait.group(2))) if wait else 20
                print(f"    [{label}] per-minute limit, waiting {seconds:.0f}s", flush=True)
                time.sleep(seconds + 1)
            except (APIConnectionError, APITimeoutError, InternalServerError):
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"{label}: gave up after repeated errors")
        text = response.choices[0].message.content or ""
        usage = response.usage.total_tokens if response.usage else 0
        self.window.setdefault(model, deque()).append((time.time(), usage))
        self.spent[model] += usage
        entry = {"key": key, "model": model, "label": label, "text": text, "tokens": usage,
                 "finish_reason": response.choices[0].finish_reason}
        self.cache[key] = entry
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return text


# ---------------------------------------------------------------------------
# Retrieval exactly like main.retrieve_for_api (all papers, de-dup by chunk)
# ---------------------------------------------------------------------------
def retrieve(query: str, store, bm25, chunks, top_k: int) -> list[dict]:
    pairs = unique_chunks(hybrid_search_with_scores(query, store, bm25, chunks, n_results=top_k))
    return [{"filename": config.source_filename(document.metadata),
             "citation": config.citation(document.metadata), "text": document.page_content,
             "score": round(score, 4) if score is not None else None}
            for document, score in pairs]


# ---------------------------------------------------------------------------
# 1. Citation accuracy (no LLM)
# ---------------------------------------------------------------------------
FILENAMES: list[str] = []


def citation_check(answer: str, retrieved: list[dict], question: dict) -> dict:
    """The app's own check (citations.py) plus: does it cite a gold page?"""
    result = check_citations(answer, retrieved, FILENAMES)
    cited = cited_pages(answer, FILENAMES)
    gold = {(question["file"], p) for p in question["page"]} if question.get("file") else set()
    result["cites_gold_page"] = bool(cited & gold) if gold else None
    return result


# ---------------------------------------------------------------------------
# 2-3. Judge prompts (judge model, temperature 0)
# ---------------------------------------------------------------------------
CORRECTNESS_PROMPT = """You grade a study assistant's answer against a reference answer written by a teacher.

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Grades:
- CORRECT: states the key fact(s) of the reference. Extra correct detail is fine. For a two-part question, both parts must be right.
- PARTIAL: gets some of the key facts but misses or garbles another part.
- INCORRECT: wrong, contradicts the reference, does not answer, or says it does not know.

Reply with JSON only: {{"grade": "CORRECT" | "PARTIAL" | "INCORRECT", "reason": "<one short sentence>"}}"""

FAITHFULNESS_PROMPT = """Check whether each numbered sentence of an answer is supported by the retrieved passages.
A sentence is SUPPORTED only if the passages state or directly imply everything it claims. Ignore citation markers like (file.pdf, p. 3).

Retrieved passages:
{context}

Answer sentences:
{sentences}

Reply with JSON only: {{"verdicts": [{{"n": 1, "supported": true}}, ...]}} with one entry per sentence."""


def parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


CITATION_TEXT = re.compile(r"[(\[【][^()\[\]【】]*?\.pdf[^()\[\]【】]*?[)\]】]", re.IGNORECASE)


def answer_sentences(answer: str) -> list[str]:
    """Claim-bearing sentences of an answer, for the faithfulness judge.

    Citation markers are removed first: a bare "(file.pdf, p. 7)" that the
    splitter separates from its sentence is not a claim and would otherwise be
    judged "unsupported". No split after a single capital initial ("R. Chesney").
    """
    body = "\n".join(line for line in normalize(answer).splitlines()
                     if not line.strip().lower().startswith("sources:"))
    body = CITATION_TEXT.sub("", body)
    parts = re.split(r"(?<![A-Z]\.)(?<=[.!?])\s+(?=[A-Z\"'*\-])", body.strip())
    return [p.strip() for p in parts if len(re.findall(r"[A-Za-z]{2,}", p)) >= 3]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(k: float, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def rate(k: float, n: int) -> str:
    if not n:
        return "–"
    ci = wilson(k, n)
    return f"{k:g}/{n} ({k / n:.0%}, CI {ci[0]:.0%}–{ci[1]:.0%})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--stages", nargs="*", default=["answer", "judge", "trust"])
    parser.add_argument("--tag", default="", help="suffix for output files; a new tag starts fresh")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = H.EVAL_DIR / f"answers_v2_k{args.top_k}{suffix}.json"

    llm = LLM(CACHE)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    settings = H.IndexSettings()
    files = H.corpus_files(None)
    FILENAMES[:] = [path.name for path in files]
    chunks = H.make_chunks(H.load_pages(files), settings)
    store, _ = H.get_store(chunks, files, settings)
    bm25 = H.build_bm25(chunks)

    rows: dict[str, dict] = {}
    if out_json.exists():
        rows = {r["id"]: r for r in json.loads(out_json.read_text(encoding="utf-8"))["questions"]}
    exhausted = None
    # Citation checks need no model, so they are always recomputed with the
    # current checker (e.g. after a parser fix) instead of trusting old results.
    by_id = {q["id"]: q for q in questions}
    for row in rows.values():
        if "answer" in row:
            row["citations"] = citation_check(row["answer"], row["retrieved"], by_id[row["id"]])

    def save():
        out_json.write_text(json.dumps({"questions": list(rows.values())}, indent=2,
                                       ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        for stage in ("answer", "judge", "trust"):
            if stage not in args.stages:
                continue
            print(f"\n== stage: {stage} ==", flush=True)
            for i, q in enumerate(questions, start=1):
                row = rows.setdefault(q["id"], {"id": q["id"], "type": q["type"],
                                                "file": q["file"], "page": q["page"]})
                if stage == "answer" and "answer" not in row:
                    retrieved = retrieve(q["q"], store, bm25, chunks, args.top_k)
                    answer = llm.ask(ANSWER_MODEL, _chat_prompt(q["q"], retrieved), 0.2, "answer").strip()
                    row.update({"retrieved": retrieved, "answer": answer, "refused": is_refusal(answer),
                                "citations": citation_check(answer, retrieved, q)})
                elif stage == "judge" and "answer" in row and (
                        "correctness" not in row or ("faithfulness" not in row and not row["refused"])):
                    if "correctness" in row:
                        pass
                    elif q["type"] == "unanswerable":
                        row["correctness"] = {"grade": "CORRECT" if row["refused"] else "INCORRECT",
                                              "reason": "refusal expected"}
                    elif row["refused"]:
                        row["correctness"] = {"grade": "INCORRECT", "reason": "false refusal"}
                    else:
                        verdict = parse_json(llm.ask(JUDGE_MODEL, CORRECTNESS_PROMPT.format(
                            question=q["q"], reference=q["answer"], answer=row["answer"]), 0.0, "correctness"))
                        grade = (verdict or {}).get("grade", "").upper()
                        row["correctness"] = {"grade": grade if grade in {"CORRECT", "PARTIAL", "INCORRECT"}
                                              else "UNPARSED", "reason": (verdict or {}).get("reason")}
                    if not row["refused"] and "faithfulness" not in row:
                        sentences = answer_sentences(row["answer"])
                        context = "\n\n".join(f"[{item['citation']}]\n{item['text']}" for item in row["retrieved"])
                        verdict = parse_json(llm.ask(JUDGE_MODEL, FAITHFULNESS_PROMPT.format(
                            context=context,
                            sentences="\n".join(f"{n}. {s}" for n, s in enumerate(sentences, 1))),
                            0.0, "faithfulness")) if sentences else {"verdicts": []}
                        flags = {int(v.get("n", 0)): bool(v.get("supported"))
                                 for v in (verdict or {}).get("verdicts", []) if isinstance(v, dict)}
                        row["faithfulness"] = {
                            "sentences": len(sentences),
                            "supported": sum(flags.get(n, False) for n in range(1, len(sentences) + 1)),
                            "unsupported": [s for n, s in enumerate(sentences, 1) if not flags.get(n, False)],
                            "parsed": verdict is not None,
                        }
                elif stage == "trust" and "answer" in row and "trust" not in row:
                    ask = lambda prompt: llm.ask(ANSWER_MODEL, prompt, 0.0, "trust")  # noqa: E731
                    row["trust"] = assess(q["q"], row["answer"], row["retrieved"], ask=ask)
                print(f"  [{i}/{len(questions)}] {q['id']} {stage} ok", flush=True)
                if i % 5 == 0:
                    save()
    except QuotaExhausted as exc:
        exhausted = str(exc)
        print(f"\nDAILY QUOTA REACHED: {exhausted}\nProgress is saved; run again later to resume.")
    finally:
        save()

    report(list(rows.values()), args.top_k, llm, exhausted, suffix)
    return 2 if exhausted else 0


_HOPS: dict[str, list[list[int]]] = {}


def by_hops(row: dict) -> list[list[int]]:
    """Fact groups of a question (multi-hop `hops`, else one group per page)."""
    return _HOPS.get(row["id"]) or [[page] for page in row["page"]]


def report(rows: list[dict], top_k: int, llm: LLM, exhausted: str | None, suffix: str = "") -> None:
    for q in json.loads(QUESTIONS.read_text(encoding="utf-8")):
        if q.get("hops"):
            _HOPS[q["id"]] = q["hops"]
    answered = [r for r in rows if "answer" in r]
    answerable = [r for r in answered if r["type"] != "unanswerable"]
    unanswerable = [r for r in answered if r["type"] == "unanswerable"]
    judged = [r for r in answerable if "correctness" in r]
    lines = [f"# Answer quality (held-out v2, top-k {top_k}{(', run ' + suffix[1:]) if suffix else ''})", "",
             f"Answers: {ANSWER_MODEL} (temperature 0.2, the app's settings). Judge: {JUDGE_MODEL} (temperature 0). "
             "Confidence labels: the app's own `trust.assess`. Generated by `python eval/answer_eval.py`.", ""]
    if exhausted:
        lines += [f"**Incomplete: the daily quota was reached.** {len(answered)}/60 answered, "
                  f"{len(judged)}/{len(answerable)} answerable judged, "
                  f"{sum('trust' in r for r in answered)}/{len(answered)} labelled. Re-run to resume.", ""]

    refused_u = sum(r["refused"] for r in unanswerable)
    refused_a = sum(r["refused"] for r in answerable)
    lines += ["## Refusals", "", "| | Result |", "|---|---|",
              f"| Correct refusals on unanswerable questions | {rate(refused_u, len(unanswerable))} |",
              f"| False refusals on answerable questions | {rate(refused_a, len(answerable))} |", ""]

    false_refusals = [r for r in answerable if r["refused"]]
    if false_refusals:
        lines += ["### False refusals: was the answer in the passages?", "",
                  "If the gold page was among the passages, refusing is a generation failure (the model "
                  "had the answer). If it was not, refusing was the honest response to a retrieval miss.", "",
                  "| Question | Type | Gold page retrieved? | Cause |", "|---|---|---|---|"]
        for r in false_refusals:
            gold = {f"{r['file']}, p. {p}" for p in r["page"]}
            given = {item["citation"] for item in r["retrieved"]}
            hops = by_hops(r)
            found = [any(f"{r['file']}, p. {p}" in given for p in hop) for hop in hops]
            if all(found):
                cause = "generation: every fact was in the passages"
            elif any(found):
                cause = "generation: refused instead of answering the part it had"
            else:
                cause = "retrieval miss"
            lines.append(f"| {r['id']} | {r['type']} | {'yes' if gold & given else 'no'} | {cause} |")
        lines.append("")

    grades = Counter(r["correctness"]["grade"] for r in judged)
    lines += ["## Correctness (answerable questions)", "",
              f"CORRECT {rate(grades['CORRECT'], len(judged))}; PARTIAL {grades['PARTIAL']}; "
              f"INCORRECT {grades['INCORRECT']}; unparsed {grades['UNPARSED']}.", "",
              "| Type | Correct | Partial | Incorrect |", "|---|---|---|---|"]
    for qtype in sorted({r["type"] for r in judged}):
        subset = Counter(r["correctness"]["grade"] for r in judged if r["type"] == qtype)
        n = sum(subset.values())
        lines.append(f"| {qtype} | {subset['CORRECT']}/{n} | {subset['PARTIAL']}/{n} | {subset['INCORRECT']}/{n} |")

    with_cites = [r for r in answered if not r["refused"]]
    all_cited = [c for r in with_cites for c in r["citations"]["cited"]]
    bad = [(r["id"], c) for r in with_cites for c in r["citations"]["not_in_retrieved"]]
    gold_cite = [r for r in with_cites if r["citations"]["cites_gold_page"] is not None]
    lines += ["", "## Citations (no LLM)", "",
              f"- Answers that cite at least one page: {rate(sum(bool(r['citations']['cited']) for r in with_cites), len(with_cites))}",
              f"- Citations pointing to a page that was NOT among the retrieved passages: {len(bad)} of {len(all_cited)}"
              + (f" ({', '.join(f'{i}: {c}' for i, c in bad)})" if bad else ""),
              f"- Answerable answers citing a gold page: {rate(sum(r['citations']['cites_gold_page'] for r in gold_cite), len(gold_cite))}", ""]

    faith = [r for r in with_cites if "faithfulness" in r]
    total_s = sum(r["faithfulness"]["sentences"] for r in faith)
    supported_s = sum(r["faithfulness"]["supported"] for r in faith)
    fully = sum(r["faithfulness"]["sentences"] == r["faithfulness"]["supported"] for r in faith)
    lines += ["## Faithfulness (sentence level, judge model)", "",
              f"- Sentences supported by the retrieved passages: {rate(supported_s, total_s)}",
              f"- Answers with every sentence supported: {rate(fully, len(faith))}",
              f"- Unparsed judge replies: {sum(not r['faithfulness']['parsed'] for r in faith)}", ""]

    labelled = [r for r in judged if "trust" in r]
    lines += ["## Calibration of the confidence label", "",
              "Answerable questions only. A well-calibrated label is correct more often at High than at Medium, and at Medium than at Low.", "",
              "| Label | Answers | Correct | Correct or partial |", "|---|---|---|---|"]
    for label in ("High", "Medium", "Low", "n/a"):
        subset = [r for r in labelled if r["trust"]["label"] == label]
        correct = sum(r["correctness"]["grade"] == "CORRECT" for r in subset)
        partial = sum(r["correctness"]["grade"] in {"CORRECT", "PARTIAL"} for r in subset)
        lines.append(f"| {label} | {len(subset)} | {rate(correct, len(subset))} | {rate(partial, len(subset))} |")
    answered_labels = [r for r in labelled if r["trust"]["label"] in {"High", "Medium", "Low"}]
    accuracy = {label: [r["correctness"]["grade"] == "CORRECT" for r in answered_labels
                        if r["trust"]["label"] == label] for label in ("High", "Medium", "Low")}
    used = {label: sum(v) / len(v) for label, v in accuracy.items() if v}
    if len(used) < 3 or len(set(round(v, 2) for v in used.values())) == 1:
        lines += ["", "**Verdict: the label is not shown to be calibrated on this set.** "
                  f"Labels used: {', '.join(f'{k} {len(accuracy[k])}' for k in used)}; correctness by label: "
                  + ", ".join(f"{k} {v:.0%}" for k, v in used.items())
                  + ". With no wrong answers among the answered questions, there is nothing for the label "
                  "to separate; it must not be presented as an accuracy estimate."]
    unans_labels = Counter(r["trust"]["label"] for r in unanswerable if "trust" in r and not r["refused"])
    if unans_labels:
        lines += ["", f"Labels on unanswerable questions the system wrongly answered: {dict(unans_labels)}."]
    unsupported = [(r["id"], s) for r in faith for s in r["faithfulness"]["unsupported"]]
    if unsupported:
        lines += ["", "## Sentences the judge found unsupported", "",
                  "| Question | Sentence |", "|---|---|"]
        lines += [f"| {qid} | {s.replace('|', '/')[:220]} |" for qid, s in unsupported]
    total = Counter()
    for entry in llm.cache.values():
        total[entry["model"]] += entry.get("tokens", 0)
    lines += ["", f"Groq tokens for all cached calls behind these results: {dict(total)}; "
              f"spent by this run: {dict(llm.spent) or 0}.", ""]

    md = H.EVAL_DIR / f"ANSWER_QUALITY_k{top_k}{suffix}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Saved {md}")


if __name__ == "__main__":
    raise SystemExit(main())
