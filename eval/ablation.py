"""Phase 3 ablations: change ONE thing at a time vs the live baseline (zero LLM calls).

Every variant is scored on BOTH sets:
  - dev  = legacy eval_questions.json (used to CHOOSE changes; already tuned on)
  - test = eval/heldout_v2_questions.json (used to CONFIRM; never tuned on)
with paired bootstrap 95% CIs for the change in hit rate against the baseline on
the same questions. The dev set is also scored with its original keyword-based
hit as a cross-check, because its page labels name only one page per fact.

Decision rule (agreed in Phase 0): adopt a change only if it does not lose on
dev (page-based delta >= 0) AND it improves the held-out test set (delta > 0).
Top-k variants change how much text reaches the model, so they are judged on
coverage AND prompt cost, not hit rate alone.

Usage: python eval/ablation.py [--variants name ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as H  # noqa: E402  (installs the zero-LLM guard)

OUT_JSON = H.EVAL_DIR / "ablation_results.json"
OUT_MD = H.EVAL_DIR / "ABLATION.md"
DEV = H.ROOT / "eval_questions.json"
TEST = H.EVAL_DIR / "heldout_v2_questions.json"

BASE_INDEX = H.IndexSettings()
BASE_CFG = H.RetrievalConfig("baseline", "hybrid")

# name -> (index settings, retrieval config, what changed, Phase 2 reason)
VARIANTS = {
    "baseline": (BASE_INDEX, BASE_CFG, "live app: hybrid RRF, 10+10 candidates, top 3", "-"),
    "pool20": (BASE_INDEX, replace(BASE_CFG, name="pool20", candidates_per_method=20),
               "20 candidates per retriever before fusion", "gold in BM25 top-20 for all v2 misses"),
    "rerank_top10": (BASE_INDEX, replace(BASE_CFG, name="rerank_top10", method="hybrid_rerank",
                                         rerank_pool=10),
                     "cross-encoder reranks the fused top 10", "RANKED_4_TO_20"),
    "pool20_rerank20": (BASE_INDEX, replace(BASE_CFG, name="pool20_rerank20", method="hybrid_rerank",
                                            candidates_per_method=20, rerank_pool=20),
                        "20+20 candidates, cross-encoder reranks the fused top 20",
                        "RANKED_4_TO_20 + NOT_RETRIEVED"),
    "topk5": (BASE_INDEX, replace(BASE_CFG, name="topk5", top_k=5), "send 5 passages instead of 3",
              "multi-hop coverage 5/10"),
    "topk8": (BASE_INDEX, replace(BASE_CFG, name="topk8", top_k=8), "send 8 passages instead of 3",
              "multi-hop coverage 5/10"),
    "cross_page": (replace(BASE_INDEX, cross_page=True), replace(BASE_CFG, name="cross_page"),
                   "chunks may cross page breaks (cited by start page)", "PAGE_SPLIT 7/19 misses"),
    "bge_small": (replace(BASE_INDEX, embedding_model="BAAI/bge-small-en-v1.5"),
                  replace(BASE_CFG, name="bge_small"),
                  "bge-small-en-v1.5 embeddings (512-token window)", "truncation: 12/47 quotes unseen"),
    "chunk1000": (replace(BASE_INDEX, chunk_size=1000), replace(BASE_CFG, name="chunk1000"),
                  "1000-char chunks (fit MiniLM's window)", "truncation, no new model"),
}


NOTES = """## Decisions

Rule: adopt only if the change does not lose on dev (under BOTH page-based and keyword-based scoring) AND it improves the held-out test set.

- **Adopted: top-k 5** (`config.TOP_K = 5`). It never loses a hit (top 5 contains top 3) and gains on both sets: dev +4/-0, test +2/-0. Multi-hop questions with every fact retrieved rise from 5/10 to 8/10. Cost: about +59% context tokens per question. Top-k 8 gains a little more (multi-hop 9/10) but costs +143%, which matters on a 200k-tokens/day quota, so 5 is the chosen balance. Whether more passages also give *better answers* is checked in Phase 4; if answers get worse, this is reverted.
- **Rejected: cross-encoder reranker** (top-10 and pool-20). The clearest held-out gain (test +4/-0, CI +2% to +17%), but it loses on dev (+4/-6). Phase 2 showed 2 of those 6 losses are single-page-label artefacts, so the honest net on dev is about even. Under the agreed rule it stays off (`RAG_RERANK=1` turns it on). This is the first candidate to revisit with a larger, multi-page-labelled dev set.
- **Rejected: cross-page chunking.** The page-based dev gain (+6/-2) is inflated: a chunk spanning pages 4-5 gets credit for both. Keyword scoring, which has no such bias, shows 0 gains and 3 losses, and the test change (+3/-2) is not significant. Kept behind `RAG_CROSS_PAGE=1`.
- **Rejected: bge-small-en-v1.5.** Fixing truncation helped dense search but not the final hybrid list: dev -2%, test +2% (not significant). BM25 already covers what MiniLM cannot see.
- **Rejected: 1000-character chunks.** Worse on test (-6%, +2/-5) and multi-hop (4/10).
- **Rejected: 20 candidates per retriever.** No gain on either set.
- **Not tested: query expansion** (Phase 2 found no miss class pointing to it, and an LLM version would spend quota).

Caveat: with 47-50 questions, a 95% CI spans about ±10 points, so only large effects are distinguishable. The paired counts (+better/-worse) are the most informative column.
"""


def legacy_keyword_hit(retriever, config, raw: dict, question: dict) -> float:
    """The original hit definition: right file + any expected keyword in top k."""
    for index in retriever.ranking(question["q"], config)[:config.top_k]:
        chunk = retriever.chunks[index]
        if H.source_filename(chunk.metadata) != question["file"]:
            continue
        text = chunk.page_content.lower()
        if any(k.lower() in text for k in raw.get("expected_keywords", [])):
            return 1.0
    return 0.0


def score(retriever, config, path: Path, keyword: bool) -> dict:
    questions = [q for q in H.load_questions(path) if q.get("file") and q.get("page")]
    raw = {q["id"]: q for q in json.loads(path.read_text(encoding="utf-8"))}
    result = H.evaluate(questions, retriever, [config])
    block = result["configs"][config.name]
    hits = {qid: float(row[config.name]["hit"]) for qid, row in result["per_question"].items()}
    out = {
        "hits": hits,
        "hit": block[f"hit@{config.top_k}"],
        "mrr": block["mrr@10"]["mean"],
        "multi_hop_all": block["by_type"].get("multi_hop", {}).get("all_pages_hits"),
        "context_tokens": round(float(np.mean([
            sum(len(retriever.chunks[i].page_content) for i in retriever.ranking(q["q"], config)[:config.top_k])
            for q in questions
        ])) / 4),
    }
    if keyword:
        out["keyword_hits"] = {q["id"]: legacy_keyword_hit(retriever, config, raw[q["id"]], q)
                               for q in questions}
    return out


def paired(a: dict, b: dict, rng) -> dict:
    ids = sorted(a)
    diff = np.array([a[i] - b[i] for i in ids])
    return {"delta": round(float(diff.mean()), 4), "ci95": H.bootstrap_ci(diff, rng),
            "better": int((diff > 0).sum()), "worse": int((diff < 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    names = ["baseline"] + [n for n in args.variants if n != "baseline"]

    files = H.corpus_files(None)
    pages = H.load_pages(files)
    retrievers: dict = {}
    results: dict = {}
    for name in names:
        index_settings, config, _, _ = VARIANTS[name]
        if index_settings not in retrievers:
            chunks = H.make_chunks(pages, index_settings)
            store, fingerprint = H.get_store(chunks, files, index_settings)
            retrievers[index_settings] = (H.Retriever(store, chunks), fingerprint, len(chunks))
        retriever, fingerprint, n_chunks = retrievers[index_settings]
        print(f"scoring {name} ...", flush=True)
        results[name] = {
            "index": {**asdict(index_settings), "fingerprint": fingerprint, "chunks": n_chunks},
            "config": asdict(config),
            "dev": score(retriever, config, DEV, keyword=True),
            "test": score(retriever, config, TEST, keyword=False),
        }

    rng = np.random.default_rng(H.SEED)
    base = results["baseline"]
    for name, r in results.items():
        if name == "baseline":
            continue
        r["dev"]["vs_baseline"] = paired(r["dev"]["hits"], base["dev"]["hits"], rng)
        r["dev"]["vs_baseline_keyword"] = paired(r["dev"]["keyword_hits"], base["dev"]["keyword_hits"], rng)
        r["test"]["vs_baseline"] = paired(r["test"]["hits"], base["test"]["hits"], rng)
        top_k_change = r["config"]["top_k"] != base["config"]["top_k"]
        # "No loss on dev" must hold under BOTH scorings: page-based credit can
        # be inflated by chunks that span pages, keyword-based cannot.
        dev_ok = (r["dev"]["vs_baseline"]["delta"] >= 0
                  and r["dev"]["vs_baseline_keyword"]["delta"] >= 0)
        test_up = r["test"]["vs_baseline"]["delta"] > 0
        sig = r["test"]["vs_baseline"]["ci95"][0] > 0
        if top_k_change:
            # More passages can only add hits (top-5 contains top-3), so the
            # question is whether the coverage gain is worth the prompt cost.
            cost = r["test"]["context_tokens"] / base["test"]["context_tokens"] - 1
            r["decision"] = f"trade-off: +{cost:.0%} prompt tokens"
        elif dev_ok and test_up:
            r["decision"] = "ADOPT" + ("" if sig else " (gain not significant)")
        else:
            r["decision"] = "reject" + ("" if dev_ok else " (loses on dev)") + ("" if test_up else " (no test gain)")

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_commit": H.git_commit(), "llm_calls": 0, "variants": results}
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(block):
        return f"{int(block['hit']['count'])}/{block['hit']['n']} ({block['hit']['mean']:.0%})"

    def fmt_delta(d):
        return f"{d['delta']:+.0%} [{d['ci95'][0]:+.0%}, {d['ci95'][1]:+.0%}] +{d['better']}/-{d['worse']}"

    lines = ["| Variant | What changed | Dev hit (page) | Δ dev [95% CI] | Δ dev keyword | Test hit | Δ test [95% CI] | Test MRR | Multi-hop all facts | Context tokens | Decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        what = VARIANTS[name][2]
        if name == "baseline":
            lines.append(f"| **baseline** | {what} | {fmt(r['dev'])} | – | – | {fmt(r['test'])} | – | "
                         f"{r['test']['mrr']:.3f} | {r['test']['multi_hop_all']}/10 | {r['test']['context_tokens']} | live |")
            continue
        lines.append(
            f"| {name} | {what} | {fmt(r['dev'])} | {fmt_delta(r['dev']['vs_baseline'])} | "
            f"{fmt_delta(r['dev']['vs_baseline_keyword'])} | {fmt(r['test'])} | "
            f"{fmt_delta(r['test']['vs_baseline'])} | {r['test']['mrr']:.3f} | "
            f"{r['test']['multi_hop_all']}/10 | {r['test']['context_tokens']} | {r['decision']} |")
    table = "\n".join(lines)
    print("\n" + table)
    OUT_MD.write_text(
        "# Phase 3: Retrieval ablations\n\nGenerated by `python eval/ablation.py` (zero LLM calls). "
        "Dev = legacy eval_questions.json (50 answerable), test = held-out v2 (47 answerable). "
        "Hit = gold file AND gold page in the top k. Δ = paired difference vs baseline on the same "
        "questions, with a 95% bootstrap CI and the number of questions that got better/worse. "
        "Context tokens ≈ characters of the passages sent to the model / 4.\n\n"
        + table + "\n\n" + NOTES, encoding="utf-8")
    print(f"\nSaved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
