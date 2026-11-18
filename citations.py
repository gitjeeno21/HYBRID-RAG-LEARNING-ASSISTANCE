"""Check that an answer only cites pages it was actually given (no LLM).

Used by the web app (every chat answer carries a `citation_check`) and by
eval/answer_eval.py, so the app and the evaluation share one definition.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


# The model often writes typographic look-alikes inside citations: narrow
# no-break spaces ("p. 8") and non-breaking hyphens ("90‑Day"). They
# must be folded to plain characters or real filenames stop matching.
_LOOKALIKES = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
})


def normalize(text: str) -> str:
    """Fold Unicode hyphen and space look-alikes to ASCII."""
    return (text or "").translate(_LOOKALIKES)


def cited_pages(answer: str, filenames: Iterable[str]) -> set[tuple[str, int]]:
    """(file, page) pairs the answer cites, matching only known filenames.

    WHY known names only: a loose pattern like "[\\w .-]+\\.pdf" would swallow
    the words before a filename ("see gpt4_technical_report.pdf") and report a
    citation to a file that does not exist.
    """
    names = sorted({name for name in filenames if name}, key=len, reverse=True)
    answer = normalize(answer)
    if not names or not answer:
        return set()
    alternatives = "|".join(re.escape(name) for name in names)
    pattern = re.compile(rf"({alternatives})\s*,?\s*p(?:age|p)?\.?\s*(\d+)", re.IGNORECASE)
    canonical = {name.lower(): name for name in names}
    return {(canonical[name.lower()], int(page)) for name, page in pattern.findall(answer)}


def check_citations(answer: str, retrieved: list[dict], filenames: Iterable[str]) -> dict:
    """Compare the pages an answer cites with the passages it was given.

    `retrieved` items carry a "citation" of the form "file.pdf, p. 3".
    Returns {"cited": [...], "not_in_retrieved": [...], "verified": bool}.
    """
    given = set()
    for item in retrieved:
        name, _, page = str(item.get("citation", "")).rpartition(", p. ")
        if page.isdecimal():
            given.add((name, int(page)))
    cited = cited_pages(answer, filenames)
    invalid = sorted(f"{name}, p. {page}" for name, page in cited - given)
    return {
        "cited": sorted(f"{name}, p. {page}" for name, page in cited),
        "not_in_retrieved": invalid,
        "verified": not invalid,
    }
