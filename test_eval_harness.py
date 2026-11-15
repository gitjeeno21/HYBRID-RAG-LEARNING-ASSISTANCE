"""Offline tests for eval/run_eval.py scoring logic (no models, no index, no LLM)."""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent


def load_harness():
    """Import eval/run_eval.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval"] = module  # dataclasses look the module up here
    saved = sys.modules.get("groq", "absent")
    saved_env = {key: os.environ.get(key) for key in ("GROQ_API_KEY", "PYTHON_DOTENV_DISABLED")}
    try:
        spec.loader.exec_module(module)
    finally:
        # The harness blocks `groq` and clears the API key for its own
        # process; undo both so other test modules see a normal environment.
        if saved == "absent":
            sys.modules.pop("groq", None)
        else:
            sys.modules["groq"] = saved
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


run_eval = load_harness()


def chunk(filename: str, page: int, text: str = "x") -> Document:
    return Document(page_content=f"{text}-{filename}-{page}",
                    metadata={"filename": filename, "page_label": str(page)})


class FakeRetriever:
    def __init__(self, chunks, rankings):
        self.chunks = chunks
        self.rankings = rankings

    def ranking(self, query, config):
        return self.rankings[query]


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("a.pdf", 1), chunk("a.pdf", 2), chunk("b.pdf", 2),
                       chunk("a.pdf", 5), chunk("a.pdf", 9)]

    def test_gold_rank_requires_matching_file_and_page(self):
        q = {"file": "a.pdf", "page": [2]}
        # b.pdf p.2 at rank 1 must not count: wrong file.
        ranks = run_eval.gold_ranks([2, 0, 1], self.chunks, q)
        self.assertEqual(ranks, {2: 3})

    def test_near_page_hit_accepts_neighbouring_page_only(self):
        q = {"file": "a.pdf", "page": [3]}
        self.assertTrue(run_eval.near_page_hit([1], self.chunks, q, top_k=3))   # p.2
        self.assertFalse(run_eval.near_page_hit([4], self.chunks, q, top_k=3))  # p.9

    def test_metrics_counts_mrr_and_multi_hop_groups(self):
        questions = [
            {"id": "q1", "q": "one", "file": "a.pdf", "page": [2], "type": "lookup"},
            {"id": "q2", "q": "two", "file": "a.pdf", "page": [9], "type": "lookup"},
            # Fact A on page 1 OR 5 (same fact twice), fact B on page 9.
            {"id": "q3", "q": "three", "file": "a.pdf", "page": [1, 5, 9], "type": "multi_hop",
             "hops": [[1, 5], [9]]},
            {"id": "u1", "q": "none", "file": None, "page": None, "type": "unanswerable"},
        ]
        retriever = FakeRetriever(self.chunks, {
            "one": [1, 0, 2],        # gold at rank 1
            "two": [0, 1, 2, 4],     # gold at rank 4 -> miss at k=3, rr = 0.25
            "three": [3, 4, 2],      # page 5 and page 9 -> both hops covered
        })
        config = run_eval.RetrievalConfig("t", "dense")
        result = run_eval.evaluate(questions, retriever, [config], measure_latency=False)
        block = result["configs"]["t"]
        self.assertEqual(result["n_answerable_scored"], 3)
        self.assertEqual(result["n_unanswerable_skipped"], 1)
        self.assertEqual(block["hit@3"]["count"], 2)
        self.assertAlmostEqual(block["mrr@10"]["mean"], round((1 + 0.25 + 1) / 3, 4))
        self.assertEqual(block["by_type"]["multi_hop"]["all_pages_hits"], 1)
        # Hand-computed. q1: 1 relevant of 3 shown, ideal reached. q2: nothing.
        # q3: ranks 1-2 relevant out of 3 relevant chunks in the corpus
        #     (p.1, p.5, p.9), both facts covered.
        ndcg_q3 = (1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(block["precision@3"]["mean"], round((1 / 3 + 0 + 2 / 3) / 3, 4))
        self.assertAlmostEqual(block["recall@3"]["mean"], round((1 + 0 + 1) / 3, 4))
        self.assertAlmostEqual(block["ndcg@3"]["mean"], round((1 + 0 + ndcg_q3) / 3, 4))
        low, high = block["hit@3"]["ci95"]
        self.assertLessEqual(low, block["hit@3"]["mean"])
        self.assertGreaterEqual(high, block["hit@3"]["mean"])

    def test_recall_counts_facts_not_pages(self):
        # Fact A is printed on pages 1 and 5; finding either covers it.
        q = {"file": "a.pdf", "page": [1, 5, 9], "hops": [[1, 5], [9]]}
        ranks = run_eval.gold_ranks([0, 1], self.chunks, q)
        metrics = run_eval.ranking_metrics([0, 1], self.chunks, q, 2, 3, q["hops"], ranks)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)

    def test_report_includes_new_metrics_and_latency(self):
        block = {"config": {"top_k": 5}, "by_type": {},
                 **{f"{m}@5": {"count": 1, "n": 2, "mean": 0.5, "ci95": [0.0, 1.0]}
                    for m in ("hit", "precision", "recall", "ndcg")},
                 "mrr@10": {"mean": 0.5, "ci95": [0, 1]},
                 "near_page_hit": {"count": 1, "n": 2},
                 "latency_ms": {"dense_ms": {"p50": 9.0, "p95": 12.0},
                                "total_ms": {"p50": 9.0, "p95": 12.0}}}
        text = run_eval.format_report({"configs": {"dense": block}, "n_answerable_scored": 2,
                                       "n_unanswerable_skipped": 0})
        self.assertIn("nDCG@k", text)
        self.assertIn("9.0 / 12.0", text)

    def test_legacy_questions_are_adapted(self):
        legacy = [
            {"id": "q01", "question": "Q?", "expected_source": "a.pdf", "source_page": 4,
             "expected_keywords": ["k"], "type": "numeric"},
            {"id": "t01", "question": "R?", "should_refuse": True, "type": "should_refuse"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            questions = run_eval.load_questions(path)
        self.assertEqual(questions[0]["page"], [4])
        self.assertEqual(questions[0]["type"], "legacy_numeric")
        self.assertEqual(questions[1]["type"], "unanswerable")

    def test_heldout_file_is_valid_and_not_yet_reviewed_by_default(self):
        questions = json.loads((ROOT / "eval" / "heldout_v2_questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 60)
        self.assertGreaterEqual(sum(q["type"] == "unanswerable" for q in questions), 10)
        self.assertEqual(len({q["id"] for q in questions}), 60)


if __name__ == "__main__":
    unittest.main()
