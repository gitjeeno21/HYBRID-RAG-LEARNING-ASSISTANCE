"""Validate eval/heldout_v2_questions.json against the PDF text. No LLM calls.

Checks, per question:
  - schema: id, q, file, page, type, answer, evidence, reviewed
  - ids are unique; type is one of the five allowed types
  - answerable: the file exists, every listed page exists, and each evidence
    snippet appears VERBATIM (whitespace-collapsed) on its listed page
  - completeness warning: an evidence snippet that also appears on a page that
    is NOT listed (the gold page list may be incomplete)
  - unanswerable: file/page are null and none of `absent_terms_checked`
    occurs anywhere in the corpus (word-boundary, case-insensitive)
  - near-duplicate warning against the legacy eval_questions.json

Exit code 1 on any error (warnings do not fail). Re-run after editing questions.

Usage: python eval/validate_questions.py [path]
"""

from __future__ import annotations

import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import PDF_DIR  # noqa: E402

TYPES = {"lookup", "exact_term", "paraphrase", "multi_hop", "unanswerable"}
FIELDS = {"id", "q", "file", "page", "type", "answer", "evidence", "reviewed"}
EXCLUDE_PATTERNS = ["*Resume*"]


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_corpus() -> dict[str, dict[int, str]]:
    from langchain_community.document_loaders import PyPDFLoader

    corpus: dict[str, dict[int, str]] = {}
    for path in sorted(Path(PDF_DIR).glob("*.pdf")):
        if any(fnmatch.fnmatch(path.name, p) for p in EXCLUDE_PATTERNS):
            continue
        corpus[path.name] = {
            int(page.metadata["page_label"]): collapse(page.page_content)
            for page in PyPDFLoader(str(path)).load()
        }
    return corpus


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "heldout_v2_questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    corpus = load_corpus()
    whole = " ".join(text for pages in corpus.values() for text in pages.values())
    legacy = json.loads((ROOT / "eval_questions.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    ids = Counter(q.get("id") for q in questions)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]

    for q in questions:
        qid = q.get("id", "?")
        missing = FIELDS - set(q)
        if missing:
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if q["type"] not in TYPES:
            errors.append(f"{qid}: unknown type {q['type']!r}")

        if q["type"] == "unanswerable":
            if q["file"] is not None or q["page"] is not None or q["evidence"]:
                errors.append(f"{qid}: unanswerable must have file=null, page=null, evidence=[]")
            # The answer must be absent from the paper the question is about.
            # Incidental matches in OTHER papers are near-miss distractors
            # (desirable for a refusal test), so they are only reported.
            topic = q.get("topic_of")
            if topic not in corpus:
                errors.append(f"{qid}: unanswerable needs topic_of = a corpus file (got {topic!r})")
                continue
            topic_text = " ".join(corpus[topic].values()).lower()
            elsewhere = []
            for term in q.get("absent_terms_checked", []):
                pattern = r"(?<![a-z0-9])" + re.escape(term.lower())
                if re.search(pattern, topic_text):
                    errors.append(f"{qid}: 'absent' term {term!r} occurs in its topic file {topic}")
                elif re.search(pattern, whole.lower()):
                    elsewhere.append(term)
            if elsewhere:
                warnings.append(f"{qid}: terms found in other papers (check they don't answer it): {elsewhere}")
            if not q.get("absent_terms_checked"):
                warnings.append(f"{qid}: no absent_terms_checked recorded")
        else:
            pages = corpus.get(q["file"])
            if pages is None:
                errors.append(f"{qid}: file {q['file']!r} not in corpus")
                continue
            if not isinstance(q["page"], list) or not q["page"]:
                errors.append(f"{qid}: page must be a non-empty list")
                continue
            if len(q["evidence"]) != len(q["page"]):
                errors.append(f"{qid}: {len(q['page'])} pages but {len(q['evidence'])} evidence snippets")
            if q["type"] == "multi_hop" and len(set(q["page"])) < 2:
                errors.append(f"{qid}: multi_hop needs at least two distinct pages")
            for page, snippet in zip(q["page"], q["evidence"]):
                if page not in pages:
                    errors.append(f"{qid}: page {page} does not exist in {q['file']}")
                    continue
                needle = collapse(snippet)
                if needle not in pages[page]:
                    errors.append(f"{qid}: evidence not found on {q['file']} p.{page}: {snippet[:60]!r}")
                others = [p for p, text in pages.items() if p not in q["page"] and needle in text]
                if others:
                    warnings.append(f"{qid}: evidence also appears on unlisted page(s) {others}")

        q_tokens = tokens(q["q"])
        for old in legacy:
            old_tokens = tokens(old["question"])
            overlap = len(q_tokens & old_tokens) / max(1, len(q_tokens | old_tokens))
            if overlap >= 0.6:
                warnings.append(f"{qid}: similar to legacy {old['id']} (Jaccard {overlap:.2f})")

    counts = Counter(q.get("type") for q in questions)
    by_file = Counter(q.get("file") or "(unanswerable)" for q in questions)
    print(f"{len(questions)} questions | types: {dict(counts)}")
    print(f"by file: {dict(by_file)}")
    print(f"reviewed: {sum(bool(q.get('reviewed')) for q in questions)}/{len(questions)}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
