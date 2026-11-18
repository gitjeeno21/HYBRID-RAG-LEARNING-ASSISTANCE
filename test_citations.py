"""Tests for citations.check_citations (no LLM)."""

import unittest

from citations import cited_pages, check_citations

FILES = ["gpt4_technical_report.pdf", "90-Day ML Engineer Roadmap.pdf", "ML111.pdf"]
RETRIEVED = [{"citation": "gpt4_technical_report.pdf, p. 5"},
             {"citation": "90-Day ML Engineer Roadmap.pdf, p. 3"}]


class CitationTests(unittest.TestCase):
    def test_parses_known_files_in_several_styles(self):
        text = ("GPT-4 scored 163 (see gpt4_technical_report.pdf, p. 5) and "
                "(90-Day ML Engineer Roadmap.pdf p.3).\nSources: ml111.pdf, page 2")
        self.assertEqual(cited_pages(text, FILES), {
            ("gpt4_technical_report.pdf", 5), ("90-Day ML Engineer Roadmap.pdf", 3), ("ML111.pdf", 2),
        })

    def test_flags_only_pages_that_were_not_retrieved(self):
        answer = "A (gpt4_technical_report.pdf, p. 5). B (gpt4_technical_report.pdf, p. 30)."
        result = check_citations(answer, RETRIEVED, FILES)
        self.assertEqual(result["not_in_retrieved"], ["gpt4_technical_report.pdf, p. 30"])
        self.assertFalse(result["verified"])

    def test_answer_without_citations_is_trivially_verified(self):
        result = check_citations("I don't know based on the provided documents.", RETRIEVED, FILES)
        self.assertEqual(result, {"cited": [], "not_in_retrieved": [], "verified": True})

    def test_typographic_lookalikes_in_citations_are_understood(self):
        # Real model output: narrow no-break spaces and non-breaking hyphens.
        text = "Block 4 (90‑Day ML Engineer Roadmap.pdf, p. 3)."
        self.assertEqual(cited_pages(text, FILES), {("90-Day ML Engineer Roadmap.pdf", 3)})

    def test_unknown_filenames_are_not_invented(self):
        self.assertEqual(cited_pages("see notes.pdf, p. 4", FILES), set())


if __name__ == "__main__":
    unittest.main()
