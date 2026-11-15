# Retrieval comparison: BM25 vs dense vs hybrid vs hybrid + reranking

All numbers are measured by `eval/run_eval.py` (zero LLM calls) at top-k 5, the app's setting. Raw output: [comparison_test_k5.md](comparison_test_k5.md), [comparison_dev_k5.md](comparison_dev_k5.md), [bge_test_k5.md](bge_test_k5.md), [bge_dev_k5.md](bge_dev_k5.md). Intervals are 95% bootstrap CIs (10,000 resamples). Δ are *paired* differences against the live configuration on the same questions, with the count of questions that got better/worse.

Reproduce:

```powershell
python eval/run_eval.py --top-k 5 --out eval/comparison_test_k5.json --md eval/comparison_test_k5.md
python eval/run_eval.py --questions eval/dev_questions_multipage.json --top-k 5 --out eval/comparison_dev_k5.json --md eval/comparison_dev_k5.md
```

Provenance: these result files record commit `7bac117`, but they were produced by the harness changes committed together with this report (P@k, R@k, nDCG@k, latency, `--top-k`). From that commit on, the harness records `--dirty` when run from uncommitted code.

## Setup

- **Corpus:** 5 PDFs, 368 chunks (1500 characters, 50 overlap, split per page). The personal résumé PDF is excluded.
- **Test:** `heldout_v2_questions.json`, 47 answerable questions with verbatim evidence quotes, checked against the PDF text by `validate_questions.py`. Not yet human-reviewed.
- **Dev:** the legacy 50 questions, with multi-page labels from [`expand_dev_labels.py`](expand_dev_labels.py). 6 questions gained a page, and every added page was read and confirmed ([DEV_LABEL_ADDITIONS.md](DEV_LABEL_ADDITIONS.md)).
- **Relevance:** a chunk is relevant if it comes from the gold file and covers a gold page. Precision and nDCG are therefore page-level. Recall counts *facts*: a multi-hop question needs every fact, and a fact printed on several pages is found by any of them.

## Results: held-out test (47 questions)

| Configuration | Hit@5 | MRR@10 | P@5 | R@5 | nDCG@5 | Δ hit vs live | Δ nDCG@5 vs live |
|---|---|---|---|---|---|---|---|
| Dense only (MiniLM) | 40/47 (85%) | 0.657 | 0.247 | 0.787 | 0.390 | −6% [−17, +4] +2/−5 | **−0.083 [−0.141, −0.029]** |
| BM25 only | 39/47 (83%) | 0.750 | 0.289 | 0.808 | 0.454 | −9% [−19, 0] +1/−5 | −0.019 [−0.075, +0.035] |
| **Hybrid RRF (live)** | **43/47 (91%)** | 0.750 | 0.311 | 0.883 | 0.473 | – | – |
| Hybrid + MiniLM rerank (top 10) | 45/47 (96%) | 0.876 | 0.328 | 0.936 | 0.540 | +4% [0, +11] +2/−0 | **+0.067 [+0.023, +0.114]** |
| Hybrid + MiniLM rerank (top 20) | 45/47 (96%) | 0.865 | 0.345 | 0.936 | 0.550 | +4% [0, +11] +2/−0 | +0.077 [+0.025, +0.130] |
| Hybrid + BGE-reranker-base (top 10) | 44/47 (94%) | 0.875 | 0.306 | 0.925 | 0.511 | +2% [0, +6] +1/−0 | +0.038 [−0.005, +0.080] |

## Results: dev (50 questions)

| Configuration | Hit@5 | MRR@10 | nDCG@5 | Δ hit vs live | Δ nDCG@5 vs live |
|---|---|---|---|---|---|
| Dense only | 33/50 (66%) | 0.566 | 0.368 | **−18% [−30, −8] +0/−9** | **−0.089 [−0.143, −0.037]** |
| BM25 only | 41/50 (82%) | 0.678 | 0.455 | −2% [−10, +6] +2/−3 | −0.002 [−0.054, +0.051] |
| **Hybrid RRF (live)** | **42/50 (84%)** | 0.653 | 0.458 | – | – |
| Hybrid + MiniLM rerank (top 10) | 41/50 (82%) | 0.601 | 0.410 | −2% [−12, +8] +3/−4 | −0.048 [−0.121, +0.022] |
| Hybrid + BGE-reranker-base (top 10) | 45/50 (90%) | 0.671 | 0.460 | +6% [0, +14] +3/−0 | +0.002 [−0.074, +0.079] |

## Why hybrid behaves differently from either retriever alone

The two retrievers fail on different questions, and the test set's question types show it:

| Test question type | Dense | BM25 | Hybrid |
|---|---|---|---|
| exact_term (numbers, names, symbols) | 9/12 | **12/12** | 12/12 |
| lookup | 9/12 | 11/12 | 11/12 |
| paraphrase (no shared wording) | **12/13** | 6/13 | 10/13 |
| multi-hop: every fact retrieved | 4/10 | 8/10 | 8/10 |

- **BM25** matches literal tokens. It never misses an exact term, but it finds only 6 of 13 paraphrased questions.
- **Dense** (MiniLM) matches meaning. It finds 12 of 13 paraphrases but misses 3 exact-term questions, because a number or symbol carries little weight in a 384-dimensional sentence embedding. Its 256-token window also cannot see the end of a 1500-character chunk (ERROR_ANALYSIS.md).
- **Reciprocal Rank Fusion** combines the two ranked lists using rank only (score = Σ 1/(60 + rank)). The two score scales (cosine vs unbounded BM25) are never compared. Hybrid keeps BM25's perfect exact-term record and finds 10 of 13 paraphrases: 4 more than BM25, 2 fewer than dense alone. RRF rewards agreement, so a chunk ranked highly by only one retriever can be outvoted by chunks both retrievers rank moderately.

### What the numbers do and do not support

- **Hybrid beats dense-only.** This holds on dev (9 questions better, 0 worse, CI excludes zero) and on ranking quality on both sets (nDCG@5 −0.083 and −0.089 for dense, both CIs exclude zero).
- **Hybrid is *not* shown to beat BM25 alone on this corpus.** The differences are +1 to +3 questions, with CIs that include zero on both sets. These are technical PDFs dense with exact terms, which is exactly where BM25 is strong. Hybrid's value here is robustness: it is never the worst on any question type. That is not a measured average gain.
- **With 47–50 questions, a CI spans about ±10 points**, so only large effects are detectable. The paired better/worse counts are the most informative column.

## Reranking decision

Rule fixed in Phase 0: adopt a change only if it does not lose on dev **and** improves the held-out set.

- **MiniLM cross-encoder: not adopted.** It gives the largest held-out gain (nDCG@5 +0.067, CI excludes zero; hit 2 better, 0 worse). But it still loses on dev with corrected multi-page labels (hit 3 better, 4 worse; nDCG −0.048), concentrated in conceptual questions (21/29 vs 23/29). The label artefact that excused the Phase 3 dev loss is fixed, and the loss remains. Cost: about 0.47 s per query on CPU, about 30× the whole hybrid pipeline.
- **BGE-reranker-base: not adopted.** It does not lose on dev (3 better, 0 worse), but its held-out gain is one question (nDCG CI includes zero). It costs about 3.0 s per query on CPU (about 200× hybrid) and needs a 1.1 GB model. It was also tested *after* the held-out set had been seen several times, so a marginal win there is weaker evidence than its nominal CI suggests.
- **Both stay available:** `RAG_RERANK=1`, plus `RAG_RERANKER_MODEL=BAAI/bge-reranker-base` for the larger model. On a GPU the latency objection largely disappears and BGE becomes the candidate to re-test on a fresh question set.

## Latency (CPU, retrieval only, per query)

Median / 95th percentile in ms, one warm-up query excluded, measured on the held-out run. Single machine (16 logical CPUs, no GPU), so these are comparative, not absolute.

| Configuration | Dense | BM25 | Fusion | Rerank | Total |
|---|---|---|---|---|---|
| Dense only | 13.6 / 16.2 | – | – | – | 13.6 / 16.2 |
| BM25 only | – | 1.3 / 2.7 | – | – | 1.3 / 2.7 |
| Hybrid RRF | 14.6 / 16.6 | 2.1 / 3.7 | <0.1 | – | 16.7 / 20.2 |
| + MiniLM rerank (top 10) | 14.4 / 18.2 | 2.1 / 3.2 | <0.1 | 485.8 / 612.5 | 500.6 / 628.6 |
| + MiniLM rerank (top 20) | 14.7 / 19.7 | 2.3 / 4.2 | <0.1 | 1015.6 / 1270.6 | 1032.0 / 1289.3 |
| + BGE rerank (top 10) | 15.0 / 21.5 | 2.2 / 4.3 | <0.1 | 2960.7 / 4249.4 | 2975.9 / 4270.6 |

End-to-end chat latency (rewrite, retrieval, generation, support check) is reported per answer in the app's pipeline trace. See [ANSWER_QUALITY_k5_stage4.md](ANSWER_QUALITY_k5_stage4.md) and the README for measured chat timings.

## Can retrieval scores decide when to abstain?

[ABSTENTION_SIGNALS.md](ABSTENTION_SIGNALS.md) (zero LLM calls) tests whether any score separates answerable from unanswerable questions well enough to refuse *before* calling the LLM.

| Held-out signal | ROC-AUC | Unanswerable caught if no answerable question may be refused |
|---|---|---|
| Top dense cosine | 0.607 | 1/13 |
| Mean dense cosine | 0.607 | 1/13 |
| Top cross-encoder score | 0.715 | 0/13 |

No. The LLM's own refusal already declines 13 of 13 unanswerable questions (Phase 4), so a score threshold could only add false refusals. Abstention stays with the grounded prompt, and no pre-LLM gate was added.
