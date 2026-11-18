# Legacy experiments

Earlier experiments and command-line scripts, moved here from the project root. None of them is used by the web app. They are kept so the results they recorded (in `results/`) stay reproducible, and so the project's history of decisions can be traced.

Run them from the project root as modules, for example `python -m legacy.rerank`.

| Script | What it did | Superseded by |
|---|---|---|
| `evaluator.py` | 8-configuration chunking grid search (keyword hit + LLM judge). Its winner, 1500/50, is still the app's chunking (`results/eval_results.json`, read by `ingest.best_chunk_config`) | `eval/run_eval.py`, `eval/ablation.py` |
| `run_judging.py` | Resumable LLM judging of the top chunking configurations | `eval/answer_eval.py` |
| `hybrid_eval.py` | First dense-vs-hybrid comparison, q1–q30, top 3 (moved out of `hybrid.py`) | `eval/RETRIEVAL_COMPARISON.md` |
| `rerank.py` | First cross-encoder reranking test on q31–q50 | `eval/RETRIEVAL_COMPARISON.md` |
| `diagnose_misses.py` | Manual diagnosis of four retrieval misses | `eval/error_analysis.py` |
| `agent.py` | Self-reflective agent loop (retrieval grading, query rewriting) prototype, mocked LLM | Not carried forward: error analysis found no miss class that query rewriting would fix |
| `rag.py`, `query.py` | Single-user dense-only command-line demos over `chroma_db/` | The web app (`main.py`) |

**Numbers in `results/` use older, more lenient metrics** (any chunk from the right PDF containing a keyword, top 3). Do not quote them next to the current page-level results in `eval/`.
