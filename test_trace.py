"""Pipeline trace and retrieval inspector, over a tiny in-memory corpus (no models, no LLM)."""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import main
from hybrid import build_bm25, cosine_from_distance, hybrid_search_with_scores

CHUNKS = [
    Document(page_content="The Transformer uses multi-head self-attention.", metadata={"filename": "a.pdf", "page": 0}),
    Document(page_content="BLEU score of 28.4 on WMT 2014 English-German.", metadata={"filename": "a.pdf", "page": 7}),
    Document(page_content="GPT-4 scored 163 on the LSAT.", metadata={"filename": "b.pdf", "page": 4}),
]


class FakeStore:
    """Dense search that returns a fixed order with squared-L2 distances."""

    def __init__(self, order_and_distance):
        self.results = order_and_distance

    def similarity_search_with_score(self, query, k, filter=None):
        rows = [(CHUNKS[i], d) for i, d in self.results]
        if filter:
            rows = [r for r in rows if all(r[0].metadata.get(key) == v for key, v in filter.items())]
        return rows[:k]


class TraceTests(unittest.TestCase):
    def test_cosine_conversion_matches_unit_vector_geometry(self):
        self.assertAlmostEqual(cosine_from_distance(0.0), 1.0)   # identical vectors
        self.assertAlmostEqual(cosine_from_distance(2.0), 0.0)   # orthogonal
        self.assertAlmostEqual(cosine_from_distance(0.852), 0.574)

    def test_trace_records_each_stage_without_changing_the_result(self):
        store, bm25 = FakeStore([(0, 0.6), (2, 1.0)]), build_bm25(CHUNKS)
        plain = hybrid_search_with_scores("BLEU 28.4", store, bm25, CHUNKS, n_results=2)
        trace = {}
        traced = hybrid_search_with_scores("BLEU 28.4", store, bm25, CHUNKS, n_results=2, trace=trace)
        self.assertEqual(plain, traced)
        by_citation = {c["citation"]: c for c in trace["candidates"]}
        bleu = by_citation["a.pdf, p. 8"]
        self.assertEqual(bleu["bm25_rank"], 1)          # lexical match
        self.assertIsNone(bleu["dense_rank"])           # dense did not return it
        self.assertAlmostEqual(by_citation["a.pdf, p. 1"]["dense_cosine"], 0.7)
        self.assertEqual(sum(c["selected"] for c in trace["candidates"]), 2)
        self.assertIn("bm25_ms", trace["timings_ms"])
        self.assertIn("dense_ms", trace["timings_ms"])
        self.assertAlmostEqual(bleu["rrf_score"], 1 / 61, places=4)

    def test_inspector_endpoint_returns_stages_and_makes_no_llm_call(self):
        with patch("main.get_hybrid_resources", return_value=(CHUNKS, build_bm25(CHUNKS))), \
                patch("main.get_vector_store", return_value=FakeStore([(1, 0.5), (0, 0.9)])), \
                patch("main.list_pdf_names", return_value=["a.pdf", "b.pdf"]), \
                patch("main.call_llm", side_effect=AssertionError("no LLM allowed")):
            response = TestClient(main.app).post("/api/retrieval/inspect",
                                                 json={"query": "BLEU score", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["searches"]), 1)
        self.assertEqual(len(body["selected"]), 2)
        self.assertEqual(body["selected"][0]["citation"], "a.pdf, p. 8")
        self.assertAlmostEqual(body["selected"][0]["score"], 0.75)

    def test_inspector_rejects_unknown_paper(self):
        with patch("main.list_pdf_names", return_value=["a.pdf"]):
            response = TestClient(main.app).post("/api/retrieval/inspect",
                                                 json={"query": "x", "papers": ["nope.pdf"]})
        self.assertEqual(response.status_code, 404)

    def test_stream_done_event_carries_timings_and_tokens(self):
        retrieved = [{"filename": "a.pdf", "citation": "a.pdf, p. 1", "text": "t", "score": 0.5}]

        def fake_stream(prompt, temperature, label, usage):
            usage["total_tokens"] = 42
            yield "Answer (a.pdf, p. 1)"

        with patch("main.list_pdf_names", return_value=["a.pdf"]), \
                patch("main.rewrite_followup", side_effect=lambda q, h: q), \
                patch("main.retrieve_for_api", return_value=retrieved), \
                patch("main.assess", return_value={"label": "High", "reason": "ok"}), \
                patch("main.call_llm_stream", side_effect=fake_stream):
            response = TestClient(main.app).post("/api/chat/stream", json={"query": "q"})
        done = [json.loads(line) for line in response.text.splitlines()][-1]
        trace = done["trace"]
        for step in ("rewrite", "retrieval", "generation", "first_token", "support_check", "total"):
            self.assertIn(step, trace["timings_ms"])
        self.assertEqual(trace["tokens"]["generation"]["total_tokens"], 42)
        self.assertEqual(trace["tokens"]["total"], 42)


if __name__ == "__main__":
    unittest.main()


class EvalSummaryTests(unittest.TestCase):
    def test_serves_recorded_numbers_and_reports_missing_sets(self):
        import tempfile
        from pathlib import Path

        block = {"config": {"top_k": 5, "method": "hybrid"},
                 "hit@5": {"mean": 0.9149, "ci95": [0.83, 0.98], "count": 43, "n": 47},
                 "mrr@10": {"mean": 0.75, "ci95": [0.6, 0.85]},
                 "latency_ms": {"total_ms": {"p50": 16.7, "p95": 20.2}}}
        with tempfile.TemporaryDirectory() as tmp:
            recorded = Path(tmp) / "test.json"
            recorded.write_text(json.dumps({"git_commit": "abc1234", "n_answerable_scored": 47,
                                            "configs": {"hybrid_rrf60": block}}), encoding="utf-8")
            files = {"test": ("Test", recorded), "dev": ("Dev", Path(tmp) / "missing.json")}
            with patch.dict(main.EVAL_RESULT_FILES, files, clear=True):
                body = TestClient(main.app).get("/api/eval/summary").json()
        test_set, dev_set = body["sets"]
        self.assertEqual(test_set["git_commit"], "abc1234")
        config = test_set["configs"][0]
        self.assertEqual(config["metrics"]["hit"]["mean"], 0.9149)
        self.assertEqual(config["metrics"]["hit"]["count"], 43)
        self.assertNotIn("ndcg", config["metrics"])     # not recorded -> not invented
        self.assertFalse(dev_set["available"])
        self.assertEqual(body["live_config"], "hybrid_rrf60")
