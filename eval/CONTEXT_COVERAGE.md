# Stage 1: Does the gold evidence reach the prompt?

Generated with `python eval/context_coverage.py` (zero LLM calls), held-out v2, 47 answerable questions, top-k 5, live index (372 chunks).

## The defect

`main.retrieve_for_api` removed duplicate passages by their citation label (`file, p. X`). A 1500-character chunk is smaller than most pages (117 of 140 pages produce two or more chunks), so two *different* passages often share a label, and the lower-ranked one was silently dropped. The model received fewer passages than `TOP_K`, and sometimes not the one holding the answer. `eval/answer_eval.py` copied the same rule, so Phase 4 measured the defect too.

Fix: duplicates are now removed by chunk identity (file, page, text) in `hybrid.unique_chunks`, shared by the app and the answer eval. The UI still lists each cited page once.

## Measured effect

| Dedup rule | Passages per question | Any evidence quote in prompt | All evidence quotes in prompt |
|---|---|---|---|
| by citation (before) | 4.38 | 38/47 | 35/47 |
| by chunk (after) | 5.00 | 43/47 | **40/47** |

Paired: +5 questions gain full evidence, 0 lose it (v2_01, v2_13, v2_43, v2_50, v2_53). Reproduce the "before" row with `--dedup citation`.

Quotes are matched as literal text after whitespace/case folding, so these counts are a lower bound.

## What this changes in Phase 4

[ANSWER_QUALITY_k5.md](ANSWER_QUALITY_k5.md) classified v2_01, v2_50 and v2_53 as **generation failures** ("every fact was in the passages") because a chunk from the gold *page* was retrieved. The gold *chunk* had been dropped before the prompt was built, so those three were actually caused by this defect, not by the model. Phase 4's answer-level numbers (correctness, false refusals) predate the fix and need a re-run before they are quoted again.

Spot check (one live chat, not a re-measurement): v2_01 was refused in Phase 4. After the fix it is answered correctly, "9.5 TFLOPS", citing source_doc.pdf p. 8, and the citation check passes.

Still missing after the fix: v2_08, v2_42, v2_51, v2_56 (retrieval misses) and v2_21, v2_22, v2_38 (multi-hop, partly retrieved).
