# Hybrid RAG Assistant: presentation content

Content for an 18-slide technical presentation, built from the source code and the measured results in this repository. **No number below is an estimate**; every figure traces to a file listed in the fact sheet. Where something is not implemented, not measured, or only experimental, the slide says so.

**How this document is organised:** Part A has the on-slide content, layout and visuals for every slide; Part B has the speaker notes; Parts C–G cover diagrams, charts, screenshots, judges' questions and a final checklist.

---

## Fact sheet: the source of truth for every claim

| Claim | Value | Source |
|---|---|---|
| Evaluation corpus | 5 PDFs: *Attention Is All You Need* (`source_doc.pdf`, 15 pp.), *GPT-4 Technical Report* (100 pp.), an ML-fundamentals paper (`ML111.pdf`, 9 pp.), the LIME explainability paper (`ML222.pdf`, 10 pp.), a 90-day ML roadmap (4 pp.). 368 chunks | `eval/comparison_test_k5.json` |
| Held-out test set | 60 questions: 47 answerable (12 exact-term, 12 lookup, 13 paraphrase, 10 multi-hop) + 13 unanswerable. Every answerable question has a verbatim evidence quote, machine-checked against the PDF text. **Not yet human-reviewed** | `eval/heldout_v2_questions.json`, `eval/validate_questions.py` |
| Dev set | 56 questions: 50 answerable + 6 off-topic; gold labels extended to every page that states the answer (6 questions gained a page, each checked by hand) | `eval/dev_questions_multipage.json`, `eval/DEV_LABEL_ADDITIONS.md` |
| Live retrieval settings | Chunks 1500 chars / 50 overlap, split per page; dense top 10 + BM25 top 10 → RRF (k = 60) → top 5 to the LLM | `config.py`, `hybrid.py` |
| Hybrid hit@5 (held-out) | 43/47 = 91% (95% CI 83–98%) | `eval/RETRIEVAL_COMPARISON.md` |
| Dense-only / BM25-only hit@5 (held-out) | 40/47 (85%) / 39/47 (83%) | same |
| Hybrid vs dense, dev set | 9 questions better, 0 worse (hit@5, CI excludes 0) | same |
| Hybrid vs BM25 alone | **Not significant** on either set | same |
| De-duplication bug fix | Full gold evidence reaching the prompt: 35/47 → 40/47 | `eval/CONTEXT_COVERAGE.md` |
| Answer correctness (held-out) | 38/47 → 41/47 after the upgrade (paired: 3 better, 0 worse) | `eval/ANSWER_QUALITY_k5.md` → `eval/ANSWER_QUALITY_k5_stage4.md` |
| False refusals | 9/47 (19%) → 4/47 (9%); the 4 left are retrieval misses | same |
| Sentence-level faithfulness (judge model) | 90% → 93% of answer sentences supported by the passages | same |
| LLM support check calibration | Graded 43/43 answers SUPPORTED, including the 4 with judge-flagged unsupported sentences → switched off by default | `eval/ANSWER_QUALITY_k5_stage4.md` |
| Reranker (MiniLM cross-encoder) | Held-out nDCG@5 +0.067 (significant), dev 3 better / 4 worse; ~0.5 s/query CPU → **off by default** | `eval/RETRIEVAL_COMPARISON.md` |
| Small-to-big retrieval | Dense alone 33 → 38/50 on dev; hybrid tied on dev, 43 → 42/47 held-out → **rejected** | `eval/SMALL_TO_BIG.md` |
| Abstention from retrieval scores | Held-out ROC-AUC 0.61 (cosine), 0.72 (cross-encoder) → no score-based gate | `eval/ABSTENTION_SIGNALS.md` |
| Latency (CPU, retrieval only, median) | Dense 13.6 ms, BM25 1.3 ms, hybrid 16.7 ms, + MiniLM rerank 501 ms, + BGE rerank 2,976 ms | `eval/comparison_test_k5.md`, `eval/bge_test_k5.md` |
| Chat answer (one measured trace) | Retrieval 29 ms, first token 1.19 s, generation 1.26 s; ~2,660 tokens without the support check | `ARCHITECTURE.md` |
| Cold start | First request after restart: 17 s → 0.18 s (measured: 9.1 s model load, 3.8 s PDF parsing removed from the request path) | commit `5e5e6ae` |
| Original chunking grid search | 8 configurations, 2 PDFs, 20 questions, top 3, keyword hit; 1500/50 best (80%, judge 4.15 / 5) | `legacy/results/eval_results.json` |
| Tests | 160 tests in 21 files, all offline; CI runs them plus a retrieval regression gate on every push | `.github/workflows/ci.yml` |
| CI retrieval gate | Hit@5 ≥ 89% on the 29 held-out questions whose PDFs are committed (measured 28/29) | same |
| API size | 33 REST endpoints (FastAPI) + NDJSON streaming | `main.py` |

### Implementation status (never present the bottom two groups as finished features)

| Status | Capability |
|---|---|
| **Implemented and tested** | Hybrid BM25 + dense retrieval with RRF; streaming grounded chat with follow-up rewriting (pronoun-triggered) and per-paper answers for multi-paper questions; deterministic citation verification; quiz generation (difficulty, type, count, page range) with key-based and LLM grading plus progress / weak-topic tracking; two-paper comparison with isolated evidence; retrieval inspector and benchmark dashboard; pipeline trace (time and tokens per step); PDF library (upload, preview, rename, delete); accounts with email verification, password reset/change, private per-user libraries, daily token allowance, account deletion; zero-LLM evaluation suite; CI |
| **Implemented, off by default (measured, not adopted)** | Cross-encoder reranking (MiniLM, BGE); cross-page chunking; LLM support check and High/Medium/Low confidence label |
| **Evaluated and rejected** | Small-to-big (parent-child) retrieval; bge-small embeddings; 1000-character chunks; 20 candidates per retriever |
| **Prototype only (in `legacy/`, not in the app)** | Self-reflective agent loop with retrieval grading (mocked LLM); single-user command-line demos |
| **Written but never run** | Docker image (Docker was not available while developing) |
| **Not implemented** | Semantic chunking, multi-query / HyDE query expansion, OCR or table extraction, contradiction detection, flashcards, adaptive quiz difficulty, background ingestion queue, HTTPS (needs a reverse proxy), two-factor sign-in |

---

# PART A: Slide-by-slide content

Visual direction for the whole deck: deep charcoal / midnight background (#0B0E14 range), white headings, muted grey body text (#9AA3B2), electric blue (#3B82F6 range) for retrieval and data elements, violet (#8B5CF6 range) for LLM / AI elements. One accent colour per idea. The app's own UI uses a gold accent: show screenshots in a thin dark frame so they sit inside the deck palette.

---

## Slide 1: Title

**Objective:** state what the project is in one glance.

**On-slide content**
- **HYBRID RAG ASSISTANT**
- Evidence-Grounded Knowledge Discovery & Intelligent Learning Platform
- Tagline: *Every answer from your own documents. Every citation checked. Every design decision measured.*
- `[Your name]` · `[Programme / institution]` · `[Date]`

**Key technical explanation:** none on this slide. The tagline previews the three themes of the talk: grounding, verification, measurement.

**Layout:** title left-aligned in the upper third; tagline under it in grey; presenter block bottom-left. Right half: a faint abstract visual of two ranked lists (blue = keyword, violet = semantic) converging into one, a quiet preview of hybrid fusion.

**Visual:** abstract line art only. No stock photos, no robot imagery.

**Transition:** "Let me start with why searching documents is still harder than it should be."

---

## Slide 2: The problem

**Objective:** show the gap a plain chatbot leaves.

**On-slide content (four problem cards)**
1. **Information is buried.** Answers sit somewhere in 100-page PDFs.
2. **Keyword search misses meaning.** "How much should I study on workdays?" shares few words with the page that answers it.
3. **Semantic search misses specifics.** Exact numbers, names and symbols ("9.5 TFLOPS", "P100") carry little weight in an embedding.
4. **Chatbots answer without evidence.** Fluent answers, no way to check which page they came from.

Bottom line: *A general-purpose chatbot answers from its training data. A learner needs answers from **their** documents, with proof.*

**Key technical explanation:** the two retrieval failure modes are the hook for Slide 7. Name them now, resolve them later with measured numbers.

**Layout:** 2×2 grid of cards, each with a small line icon and one sentence. The bottom line spans the width in white.

**Visual:** simple line icons (document stack, magnifier with a cross, number tag, chat bubble with a question mark).

**Transition:** "So the requirement is: find evidence reliably, and make the answer prove where it came from."

---

## Slide 3: Our solution

**Objective:** the value proposition in one slide.

**On-slide content**
- **What it is:** a web application that turns your PDF library into a searchable, quizzable, comparable knowledge base.
- **Who it's for:** students and researchers working from papers; anyone studying from a fixed set of documents.
- **How it's different:**
  - **Hybrid retrieval:** keyword search (BM25) and semantic search fused with Reciprocal Rank Fusion
  - **Grounded answers:** generated only from retrieved passages, with page citations that are verified automatically
  - **Measured, not assumed:** every retrieval setting chosen or rejected by a reproducible benchmark

Value proposition (large text): *Ask your documents. Get answers you can verify.*

**Key technical explanation:** the third point is what separates this from typical RAG demos: decisions backed by a dev/held-out benchmark with confidence intervals.

**Layout:** value proposition centred at top; three columns below (Retrieve · Ground · Measure), each with an icon, a heading and two lines.

**Visual:** three-column icon row. Retrieve in blue, Ground in violet, Measure in white/grey.

**Transition:** "Here is how a question becomes a verified answer."

---

## Slide 4: System overview

**Objective:** the end-to-end architecture in one diagram (Part C, Diagram 1).

**On-slide content (the diagram labels are the content)**
- **Ingestion:** PDF upload → validation → PyPDF text extraction (per page) → chunking (1500 chars, 50 overlap) → MiniLM embeddings → ChromaDB (+ in-memory BM25 index)
- **Query:** question → follow-up rewrite (when needed) → dense top 10 ∥ BM25 top 10 → RRF fusion → top 5 passages → grounded prompt → streamed LLM answer → citation check
- Side panel: *Per-user isolation · Pipeline trace on every answer*

**Key technical explanation:** two indexes are built from the same chunks, and startup verifies they hold *identical* content. If they drifted, fusion would silently degrade to BM25-only.

**Layout:** two horizontal swim lanes (Ingestion on top, Query below), left to right, meeting at the two indexes in the middle.

**Visual:** Diagram 1. Blue for retrieval components, violet for LLM components, grey for storage.

**Transition:** "Every box here is a deliberate technology choice. Here's the stack and why."

---

## Slide 5: Technology stack

**Objective:** the real stack, each part with its reason.

**On-slide content (table)**

| Layer | Technology | Why this one |
|---|---|---|
| Language | Python 3.11 | One language for ML, retrieval, API and evaluation |
| Backend | FastAPI + Uvicorn | Typed request validation; streaming responses (NDJSON) for token-by-token answers |
| Frontend | Vanilla JavaScript, hand-written CSS, no build step | Zero build tooling; a single-page app with Chat / Quiz / Compare / Inspect modes |
| Document processing | PyPDF (via LangChain `PyPDFLoader`), `RecursiveCharacterTextSplitter` | Page-level extraction keeps page numbers for citations |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, 384-d, CPU) | Runs locally without a GPU; no API cost for indexing |
| Vector store | ChromaDB (persistent) | Embedded, no server; metadata filters enable per-paper and per-user scoping |
| Keyword retrieval | `rank_bm25` (BM25Okapi) | Exact-term matching that embeddings miss |
| LLM | Groq API, `openai/gpt-oss-120b` | Strong open-weight model, fast hosted inference, free tier |
| Accounts | SQLite (stdlib), Argon2id (`argon2-cffi`), EmailJS (server-side REST) | No database server; standard password hashing; email codes without an SMTP setup |
| Evaluation & testing | NumPy (bootstrap CIs), `unittest` (160 tests), GitHub Actions | Reproducible, offline, runs on every push |

**Key technical explanation:** retrieval runs entirely locally on CPU; only answer generation leaves the machine.

**Layout:** a three-column table (layer · tool · reason), tool names in white, reasons in grey. Small tool logos beside the names where a logo exists.

**Visual:** the table itself; optional thin coloured bar per row (blue = retrieval, violet = LLM, grey = infrastructure).

**Transition:** "Let's follow a PDF from upload to index."

---

## Slide 6: Document ingestion pipeline

**Objective:** what happens to a PDF, with the real parameters.

**On-slide content (pipeline, 7 steps)**
1. **Upload:** filename validated (one OS-independent rule: no path separators, no hidden names, `.pdf` only), `%PDF-` header checked, 50 MB limit
2. **Extract:** PyPDF reads one document per page; parsed text is cached on disk per file version
3. **Chunk:** `RecursiveCharacterTextSplitter`, **1500 characters, 50 overlap**, split *within* each page (prefers paragraph → line → sentence boundaries)
4. **Metadata:** `filename`, `page`, `page_label`, `user_id` on every chunk
5. **Embed:** all-MiniLM-L6-v2, 384 dimensions
6. **Store:** ChromaDB vectors + per-user BM25 index over the same chunks
7. **Verify:** the dense and BM25 corpora must contain identical chunks, or the app refuses to serve

**Key technical explanation:** chunks never cross pages, so every citation points to exactly one page. If anything fails mid-upload, the partial vectors and files are rolled back.

**Honest note (small text):** there is no dedicated text-cleaning step. Text goes from PyPDF to the splitter as extracted. Scanned PDFs (images) are not supported (no OCR).

**Layout:** a horizontal 7-step pipeline with an icon per step; parameters in a callout under step 3.

**Visual:** Diagram 2 (Part C).

**Transition:** "Now the core of the system: how we find the right chunks."

---

## Slide 7: The core innovation, hybrid retrieval

**Objective:** show, with real questions, that keyword and semantic search fail on *different* questions.

**On-slide content (three columns)**

| | **BM25 (keyword)** | **Dense (semantic)** | **Hybrid (both, fused)** |
|---|---|---|---|
| Matches | exact words, numbers, names | meaning, paraphrase | both |
| Exact-term questions (12) | **12/12** | 9/12 | **12/12** |
| Paraphrase questions (13) | 6/13 | **12/13** | 10/13 |
| Overall hit@5 (47 held-out) | 83% | 85% | **91%** |

**Two real questions from the held-out set:**
- *"What sustained TFLOPS figure did the authors assume for a P100 GPU?"* → BM25 ranks the answer page **#1**; dense ranks it **#10** (a miss). Hybrid: **#3** ✓
- *"I have a full-time job; how much should I study on workdays?"* → dense ranks the answer page **#1**; BM25 ranks it **#10**. Hybrid: **#2** ✓

**Key technical explanation:** an embedding compresses a 1500-character chunk into one 384-number vector, and a token like "9.5" barely moves it. MiniLM also reads only the first 256 tokens (~1000 characters) of each chunk. BM25 has neither limit but cannot match "workdays" to "weekdays after work".

**Layout:** left two-thirds: the comparison table with the winning cell in each row highlighted. Right third: the two example questions as cards (BM25 win in blue, dense win in violet), each showing the rank badges.

**Visual:** the table plus two "rank badge" cards. Optional: the per-type grouped bar chart (Part D, Chart 2) as a build.

**Transition:** "Fusing two rankings that use completely different scores is the hard part. That's RRF."

---

## Slide 8: Reciprocal Rank Fusion

**Objective:** explain RRF precisely, with a real numerical example from the app.

**On-slide content**
- Formula (large): **RRF(d) = Σᵢ 1 / (k + rankᵢ(d))**
  - *d*: a chunk · *i*: each retriever (dense, BM25) · *rankᵢ(d)*: its position in that retriever's list (1 = best; absent → contributes 0) · *k = 60*: a damping constant
- **Why ranks, not scores:** cosine similarity lies in [−1, 1]; BM25 scores are unbounded and corpus-dependent. Averaging them would need an arbitrary weighting. RRF needs no calibration.

**Real example: "What LSAT score did GPT-4 achieve?" (from the app's retrieval inspector)**

| Page | Dense rank | BM25 rank | RRF score | Final |
|---|---|---|---|---|
| GPT-4 report p. 6 | 6 | 1 | 1/66 + 1/61 = **0.0315** | **#1** |
| GPT-4 report p. 4 | 9 | 9 | 1/69 + 1/69 = **0.0290** | #2 |
| GPT-4 report p. 5 | **1** | – | 1/61 = **0.0164** | #3 |

*Agreement beats a single strong vote: p. 5 is dense search's top hit, but BM25 didn't find it, so two moderate votes outrank it.*

- **Implementation (`hybrid.py`):** 10 candidates per retriever; ties broken by best single rank, then chunk position, so results are deterministic. Per-paper searches filter *both* retrievers.

**Key technical explanation:** k = 60 comes from the original RRF paper and was fixed before any measurement. A large k flattens the gap between rank 1 and rank 5, so no single retriever dominates.

**Layout:** formula as the hero element top-left, with variable definitions under it; the worked table on the right; implementation note as a footer strip.

**Visual:** the table, with the two retriever-rank columns coloured blue and violet and the RRF column in white. The arithmetic shown inline.

**Transition:** "Once we have the top five passages, here's how they become an answer."

---

## Slide 9: LLM-powered answer generation

**Objective:** context construction, prompt, grounding, citations, limits.

**On-slide content**
- **Context:** top 5 passages, each labelled `Source: file.pdf, p. X`; with two or more papers selected, passages are **grouped per paper** and every paper must be addressed
- **Prompt rules** (`config.PROMPT_TEMPLATE`):
  - answer **only** from the context
  - none of the answer present → reply with one fixed refusal sentence
  - only part present → answer that part and **name what is missing**
  - cite pages as `(file, p. X)`, at most 120 words
- **Generation:** `gpt-oss-120b`, temperature 0.2, streamed token by token
- **After generation (no LLM):** every cited page is checked against the pages actually retrieved → **"Citations verified"**, or a warning naming the unverified page
- **Follow-ups:** a question containing "it / they / this…" is rewritten into a standalone search using the last 3 turns

**Key technical explanation:** the citation check is deterministic, cheap and caught a real error during testing: an answer citing p. 5, which the model was never given.

**Limitations (small text):** the citation check proves the right *page* was used, not that the sentence is faithful. A checked answer once said "base model" where the paper says "big model".

**Layout:** a vertical flow on the left (Question → Retrieve → Build context → Prompt → Stream → Verify citations), with the prompt rules in a code-styled box on the right.

**Visual:** Diagram 3 (query-to-answer flow), with the prompt box in monospace on a slightly lighter panel.

**Transition:** "Retrieval and generation power four user-facing workflows."

---

## Slide 10: Key application features

**Objective:** show the product, not just the pipeline.

**On-slide content (2×3 grid, one screenshot each)**
- **Chat:** streamed answers, clickable page citations, retrieved passages, a pipeline trace (time and tokens per step)
- **Quiz:** generated from one paper; difficulty, question type, 5/10/15 questions, page range; multiple-choice graded against the key, short answers graded from the same passages; progress and weakest topics
- **Compare:** two papers answered from isolated evidence: answer per paper, difference, similarity
- **Inspect:** a retrieval debugger (every candidate's dense rank, BM25 rank, RRF score) and the benchmark dashboard
- **Library:** upload by drag-and-drop, preview, rename, delete
- **Accounts:** email-verified sign-up, password reset by emailed code, private library per user, daily AI allowance, account deletion

**Key technical explanation:** each workflow reuses the same hybrid retrieval. Quiz and Compare apply a per-paper filter to *both* retrievers so evidence can't leak between papers.

**Layout:** six tiles, each a screenshot (cropped to the relevant panel) with a one-line caption.

**Visual:** screenshots from Part E (shots 1–6). Inspect is already captured at `docs/screenshots/inspect.png`.

**Transition:** "Behind these features are a few engineering decisions worth defending."

---

## Slide 11: Technical deep dive

**Objective:** demonstrate judgement, not a list of tools.

**On-slide content (six decision cards: question → decision → evidence)**
1. **Why hybrid, not only vectors?** The retrievers fail on different questions: BM25 12/12 vs dense 9/12 on exact terms; dense 12/13 vs BM25 6/13 on paraphrases.
2. **Why RRF?** Score scales are incomparable; RRF uses ranks only, with no tuning.
3. **Why ChromaDB?** Embedded with no server; metadata filters give per-paper *and* per-user scoping. Isolation is enforced by a wrapper that adds the user filter to every operation.
4. **Why MiniLM?** CPU-friendly and free. Its 256-token window is a known limit: a 512-token model and small-to-big retrieval were both tested; neither improved the hybrid system.
5. **Why top-5, 1500-character chunks?** Measured: top-5 never loses a hit vs top-3 and lifts multi-hop coverage 5/10 → 8/10, for +59% prompt tokens.
6. **Retrieval → answer quality:** fixing one de-duplication bug raised full evidence in the prompt 35 → 40/47 and correct answers 38 → 41/47.

**Key technical explanation:** decision 6 is the strongest point of the talk. A retrieval bug showed up directly as answer quality, and was found by measuring *what the model receives*, not what the retriever ranks.

**Layout:** 3×2 grid of cards; each card has the question in white, decision in grey and evidence as a blue number.

**Visual:** cards only. Keep it text-light: one line per part.

**Transition:** "Those numbers come from a deliberate evaluation methodology. Here's how it works."

---

## Slide 12: Experiments and benchmarking

**Objective:** show the method, then each experiment with its outcome.

**On-slide content**

**Method:** zero-LLM retrieval harness (the LLM client is blocked at import) · page-level gold labels with verbatim evidence quotes · dev set to decide, held-out set to confirm · 95% bootstrap CIs · paired better/worse counts · adoption rule fixed before measuring: *don't lose on dev, improve held-out*.

| Experiment | Data | Result | Decision |
|---|---|---|---|
| Chunk size × overlap grid (8 configs) | 2 PDFs, 20 Q, top-3 | 1500/50 best: 80% hit, judge 4.15/5 | **Adopted** |
| Top-k 3 / 5 / 8 | dev 50 + held-out 47 | k=5: +4/−0 dev, +2/−0 held-out | **Adopted** k=5 |
| Cross-encoder rerank (MiniLM, BGE) | dev 50 + held-out 47 | Held-out gain, but dev loss / 3 s CPU | Off by default |
| Small-to-big retrieval | dev 50 + held-out 47 | Dense +5 on dev; hybrid no gain | **Rejected** |
| Cross-page chunks, bge-small, 1000-char chunks | dev 50 + held-out 47 | Lost on dev | **Rejected** |
| Abstention from retrieval scores | 60 + 56 Q | AUC 0.61–0.72 held-out | **Not built** |
| LLM support-check calibration | 43 answers | 43/43 SUPPORTED | Off by default |

**Key technical explanation:** half of these experiments ended in "rejected". That's the evaluation doing its job. Each rejection removed latency or complexity that wouldn't have paid for itself.

**Layout:** method as a single strip across the top (five chips); the table below with the Decision column colour-coded (adopted = blue, rejected/off = grey).

**Visual:** the table; optional small method flowchart (Diagram 5).

**Transition:** "Here are the results that matter most."

---

## Slide 13: Performance and results

**Objective:** the headline measured results, visually.

**On-slide content (three KPI tiles + one chart)**
- **91%** hybrid hit@5 on 47 held-out questions (dense 85%, BM25 83%)
- **41/47** correct answers, up from 38 (3 better, 0 worse); false refusals **19% → 9%**
- **17 ms** retrieval latency on CPU (median); about 1.3 s to a complete streamed answer

Chart: hit@5 per retrieval method on the held-out and dev sets, with 95% CI whiskers (Part D, Chart 1).

Footnote: *Measured on 47–50 questions per set; confidence intervals span about ±10 points. Hybrid's advantage over dense-only is significant; over BM25 alone it is not.*

**Key technical explanation:** say the footnote out loud. Claiming only what the statistics support is part of the result.

**Layout:** three KPI tiles across the top (big number in white, label in grey); chart below, full width.

**Visual:** Part D, Chart 1 (horizontal bars with CI whiskers, live configuration highlighted).

**Transition:** "Where would this be useful beyond studying papers?"

---

## Slide 14: Real-world applications

**Objective:** credible uses, tied to actual capabilities.

**On-slide content (five use cases, each with the capability that enables it)**
- **Academic research:** literature review across a reading list. *Multi-paper answers per paper; Compare mode.*
- **Education:** self-study and revision from assigned material. *Quiz generation and grading from the course PDFs; weak-topic tracking.*
- **Technical documentation:** exact lookups of parameters, versions and error codes. *BM25 half of hybrid retrieval.*
- **Internal knowledge bases:** policies and procedures where every answer must cite its source. *Citation verification; refusal when the documents don't cover it.*
- **Regulated, document-heavy work** (compliance, legal research support): *private libraries per user; page-level evidence.*

Note: *a prototype for these domains; production use would need the limitations on the next slide addressed.*

**Key technical explanation:** each use case is tied to an implemented feature, so nothing is promised that the code doesn't do.

**Layout:** five horizontal rows (icon · domain · use · enabling feature in blue).

**Visual:** simple line icons per domain.

**Transition:** "Every one of these depends on knowing the system's limits."

---

## Slide 15: Limitations and challenges

**Objective:** show maturity by naming real limits and the path past each.

**On-slide content (limit → mitigation)**

| Limitation | Evidence | Way forward |
|---|---|---|
| PDF extraction: no OCR, no table structure, no text cleaning | PyPDF text as extracted | OCR and layout-aware parsing for scanned / tabular PDFs |
| Embedding window: MiniLM reads ~1000 of 1500 characters | 12/47 answers past the window | Longer-context embeddings, re-tested under the same rule |
| Retrieval misses remain | 4/47 held-out; fusion can demote a page both retrievers found (v2_42) | Larger, fresh test set before tuning fusion |
| No calibrated confidence | LLM self-check: 43/43 SUPPORTED | An independent verifier, measured the same way |
| Small evaluation sets | 47–50 Q; CIs ±10 points; labels not human-reviewed | Human review; a larger fresh held-out set |
| Free-tier LLM throughput | ~8k tokens/min ≈ one answer every ~20 s | Paid tier or a self-hosted model |
| Deployment | SQLite and in-memory throttling assume one server; no HTTPS of its own; Docker image never built | Postgres, a reverse proxy, a CI-built image |

**Key technical explanation:** every limitation above was found by measurement or testing, not guessed.

**Layout:** a three-column table; the evidence column in blue to signal "measured".

**Visual:** table only.

**Transition:** "Which leads to what's next, separated honestly into what exists and what doesn't."

---

## Slide 16: Future enhancements

**Objective:** separate existing, evaluated and proposed work.

**On-slide content (three columns)**

**Already built (off by default, measured):**
- Cross-encoder reranking (MiniLM, BGE): `RAG_RERANK=1`
- LLM evidence verification: `APP_SUPPORT_CHECK=1`
- Retrieval analytics dashboard: Inspect mode *(live)*

**Evaluated and rejected (with reasons):**
- Parent-child (small-to-big) retrieval
- 512-token bge-small embeddings, cross-page chunks
- Abstention from retrieval scores

**Proposed next (not implemented):**
- Re-test BGE reranking on a GPU against a fresh question set
- Query rewriting beyond follow-ups (multi-query / HyDE), only if error analysis points to it
- OCR and table-aware document understanding
- Semantic chunking, measured against the current baseline
- An independent faithfulness verifier for answer-level confidence
- Background ingestion with progress for large uploads

**Key technical explanation:** several "future" ideas from typical RAG roadmaps are already built here *and measured*, which is a stronger position than a roadmap.

**Layout:** three columns: blue (built), grey (rejected), violet (next).

**Visual:** column headers with small status icons (check, cross, arrow).

**Transition:** "To wrap up."

---

## Slide 17: Conclusion

**Objective:** land the story in five points.

**On-slide content**
- **Problem:** answers buried in long documents, and AI answers you can't verify
- **Core technique:** hybrid BM25 + dense retrieval fused with RRF, 91% hit@5 on held-out questions
- **Engineering contribution:** an evaluation suite that decided every setting, found a bug that cost answer quality, and rejected techniques that didn't pay off
- **Practical value:** a multi-user study platform: chat, quiz, compare, inspect, with verified citations
- **Next:** stronger verification, broader document understanding, larger evaluation

Closing statement (large): *Trustworthy AI over documents isn't about a bigger model. It's about finding the right evidence, proving where answers come from, and measuring both.*

**Layout:** five short rows on the left; the closing statement large on the right or at the bottom.

**Visual:** none beyond typography.

**Transition:** "I'm happy to take questions."

---

## Slide 18: Q&A

**On-slide content**
- **Questions?**
- HYBRID RAG ASSISTANT · Evidence-grounded knowledge discovery
- *"Every answer from your documents. Every citation checked."*
- `[GitHub repository link]` · `[Contact]`

**Layout:** centred, minimal. The two-ranked-lists motif from Slide 1 returns faintly in the background.

---

# PART B: Speaker notes

**Slide 1.** "Hybrid RAG Assistant is a platform for learning from your own documents. You ask questions about your PDFs and get answers that come only from those documents, with page citations the system checks. What makes it more than a PDF chatbot is the engineering behind three words on this slide: grounded, verified, measured. I'll show you each."

**Slide 2.** "Think about studying from a 100-page technical report. The answer exists, but finding it is slow. Search tools fail in two opposite ways. Keyword search finds exact words but misses meaning: ask 'how much should I study on workdays' and it struggles, because the page says something different. Semantic search, which compares meaning, fails the other way: exact numbers like '9.5 TFLOPS' barely register. And general chatbots answer fluently from their training data, with no way to check where an answer came from. For learning, an unverifiable answer is worse than no answer."

**Slide 3.** "Our solution has three parts. Retrieve: we run keyword search and semantic search together and fuse the results. Ground: the language model may answer only from the retrieved passages, must cite the page, and the system checks that every cited page was actually provided. Measure: every retrieval setting in this project was chosen, or rejected, by a benchmark with a separate test set and confidence intervals. That third part is unusual for a student project, and it found real problems."

**Slide 4.** "Two flows. At ingestion, a PDF is validated, read page by page, split into 1500-character chunks, embedded with a small local model and stored twice: in ChromaDB for semantic search and in a BM25 index for keyword search. Startup checks that both indexes hold exactly the same chunks, because if they drift apart, fusion silently becomes keyword-only. At query time, both retrievers return their top 10, Reciprocal Rank Fusion merges them, the top 5 go into a grounded prompt, the answer streams back, and citations are checked. Everything is isolated per user, and every answer carries a trace of how long each step took."

**Slide 5.** "Everything except answer generation runs locally on CPU: extraction, embedding, both indexes and fusion. That keeps indexing free and private. FastAPI gives typed validation and streaming. The frontend is deliberately plain JavaScript with no build step. ChromaDB is embedded, so there's no database server, and its metadata filters are what make per-paper and per-user search possible. Generation uses an open-weight 120-billion-parameter model on Groq's free tier. Testing is unittest with 160 tests, run on every push by GitHub Actions."

**Slide 6.** "A PDF goes through seven steps. Filenames are validated with one rule that behaves the same on Windows and Linux; CI caught a case where they differed. Text is extracted page by page, and the parsed text is cached on disk, which cut the first request after a restart from 17 seconds to 0.18. Chunks are 1500 characters with 50 overlap, and they never cross a page boundary, so every citation is one exact page. That size won an eight-configuration grid search. To be candid: there's no text-cleaning step and no OCR, so scanned PDFs won't work. That's on my limitations slide."

**Slide 7.** "This is the core idea, and the data shows it clearly. On our 47 held-out questions, broken down by type: for exact-term questions, keyword search finds all 12 and semantic search only 9. For paraphrased questions it flips: semantic finds 12 of 13, keyword only 6. Two real examples. Asking for the P100 TFLOPS figure, keyword search ranks the right page first; semantic search ranks it tenth, outside the five we use. Asking how much to study on workdays, semantic search ranks it first and keyword search tenth. Hybrid gets both right, ranking them third and second. Why does semantic search miss numbers? An embedding squeezes 1500 characters into 384 numbers, so one token barely moves it, and this model reads only the first 256 tokens of each chunk."

**Slide 8.** "Fusing the two lists is the hard part, because their scores aren't comparable: semantic search gives a similarity between minus one and one; BM25 gives an unbounded number that depends on the corpus. Averaging them means inventing a weighting. Reciprocal Rank Fusion ignores the scores and uses only positions: each retriever contributes one over sixty plus the rank, and a chunk a retriever didn't return contributes nothing. Here's a real query from the app's inspector. Page 6 is sixth for semantic and first for keyword: 0.0315, ranked first. Page 5 is semantic's top hit, but keyword search didn't find it, so it scores only 0.0164 and lands third. RRF rewards agreement. That's usually right, and I'll show you a case where it isn't. The constant 60 comes from the original paper and was fixed before I measured anything."

**Slide 9.** "The top five passages go into the prompt, each labelled with its file and page. The rules are strict: answer only from these passages. If none of the answer is there, reply with one fixed refusal sentence, which makes refusals measurable. If part is there, answer that part and say what's missing; that rule cut wrongful refusals from 19% to 9%. When several papers are selected, the passages are grouped by paper and every paper must be addressed, because in testing the model otherwise answered from one paper and ignored the other. After generation, a deterministic check compares every cited page with the pages actually provided. It caught a real case of the model citing a page it never saw. Its limit: it proves the page, not the sentence."

**Slide 10.** "Four workflows share that retrieval. Chat, with streaming, citations and a trace showing time and tokens per step. Quiz: generated from one paper, with difficulty, question type and page range. Multiple-choice is graded instantly against the key; short answers are graded from the same passages; and it tracks your weakest topics. Compare answers the same question for two papers from strictly separated evidence. Inspect is the debugger: type any question and see every candidate's rank from both retrievers and its fused score. Around it is a full account system with email verification, password reset and private libraries."

**Slide 11.** "Six decisions, each with evidence. Hybrid because the retrievers fail on different questions; you saw the numbers. RRF because the scores don't compare. ChromaDB because its metadata filters let me enforce isolation in one place: a wrapper adds the user filter to every search, read, write and delete, so no route can forget it. MiniLM because it runs on CPU for free, and I measured its limit. Top-5 because it never loses against top-3 and raises multi-hop coverage from five to eight of ten. And the most important: a de-duplication bug was silently throwing away the second chunk from each page, often the one with the answer. Fixing it raised full evidence in the prompt from 35 to 40 of 47, and correct answers from 38 to 41, with no question getting worse. I found it by measuring what the model receives, not what the retriever ranks."

**Slide 12.** "The method matters more than any single number. Retrieval experiments make zero LLM calls; the client is blocked at import, so they're free and deterministic. Gold labels are pages with a verbatim quote, checked against the PDF text by a script. I decide on a dev set and confirm on a held-out set, with a rule written down before measuring: don't lose on dev, improve on held-out. Results get bootstrap confidence intervals and paired better/worse counts. Under that rule, top-5 was adopted, and most other ideas were rejected, including two well-known techniques: cross-encoder reranking and small-to-big retrieval. Rejections are results too: they kept half a second to three seconds of latency out of every query."

**Slide 13.** "Three headline numbers. 91% of held-out questions have the answer page in the top five passages. 41 of 47 answers are judged correct, up from 38, with three improvements and no regressions, and wrongful refusals halved. Retrieval takes about 17 milliseconds on a laptop CPU. And the honest footnote: with about 50 questions, confidence intervals are roughly ten points wide. Hybrid's advantage over semantic-only search is statistically significant. Its advantage over keyword-only search is not, on this corpus, because technical papers are full of exact terms that BM25 handles well. Hybrid's value is that it's never the worst on any question type."

**Slide 14.** "Each application is tied to something the code actually does. Research: per-paper answers and Compare for literature review. Education: quizzes from the course's own PDFs, with progress tracking. Technical documentation: exact lookups, which is precisely where BM25 earns its place. Internal knowledge bases: answers that must cite a source, and a system that says when the documents don't cover a question. These are prototypes for those domains, and the next slide says what would need to change."

**Slide 15.** "Every limitation here was found by measurement or testing. Extraction: no OCR, no table structure, no text cleaning. Embeddings: the model reads about a thousand of each chunk's fifteen hundred characters. Retrieval: four held-out questions are still missed, and one shows RRF's weakness: both retrievers found the page, at ranks 3 and 5, yet fusion pushed it to 7. Confidence: I built an LLM self-check and calibrated it, and it said 'supported' 43 times out of 43, including answers the independent judge found unsupported sentences in, so I turned it off. Evaluation: about fifty questions per set, and the labels still need human review. Operations: free-tier throughput, single-server storage, no built-in HTTPS."

**Slide 16.** "Many RAG roadmaps list reranking, evidence verification and analytics dashboards as future work. Here they already exist, and more importantly, they've been measured. Reranking and the self-check are available behind switches but off, because the data said so. The dashboard is live. Parent-child retrieval was built and rejected. What's genuinely next: re-testing the BGE reranker on a GPU against a fresh question set, document understanding with OCR and tables, and an independent faithfulness verifier, each to be measured under the same adoption rule."

**Slide 17.** "To conclude: the problem is finding evidence in long documents and trusting AI answers about them. The core technique is hybrid retrieval fused with RRF, at 91% on held-out questions. The engineering contribution is the evaluation discipline: it chose every setting, found a bug that was costing answer quality, and rejected techniques that sounded good but didn't help. The result is a working multi-user learning platform. And the lesson: trustworthy AI over documents isn't about a bigger model. It's about finding the right evidence, proving where answers come from, and measuring both."

**Slide 18.** "Thank you. I'm happy to take questions, and I can show any of this live, including the retrieval inspector for any question you'd like to try."

---

# PART C: Architecture diagrams

### Diagram 1: End-to-end system (Slide 4)

```
INGESTION
[User] → [Upload: filename + %PDF header + 50 MB check]
       → [PyPDF: one Document per page] ──(cache: parsed text per file version)
       → [RecursiveCharacterTextSplitter: 1500 / 50, within each page]
       → [Metadata: filename, page, page_label, user_id]
       ├→ [all-MiniLM-L6-v2 → 384-d vectors] → [ChromaDB (shared collection, user_id on every chunk)]
       └→ [BM25Okapi index (per user, in memory)]
       [Startup check: Chroma chunks ≡ BM25 chunks, else refuse to serve]

QUERY
[Question (+ last 3 turns)] → [Follow-up rewrite if pronoun → LLM]
       → [Dense top 10 (cosine)] ┐
       → [BM25 top 10]           ┴→ [RRF k=60] → [Top 5 passages] → [De-duplicate by chunk]
       → [Grounded prompt (per-paper sections if ≥2 papers)] → [gpt-oss-120b via Groq, streamed]
       → [Citation check: cited pages ⊆ retrieved pages] → [Answer + passages + trace]
```
Relationships: both retrievers read the same chunk set; all Chroma access goes through `ScopedStore` (user filter injected); the trace records each stage's time and tokens.

### Diagram 2: Ingestion pipeline (Slide 6)

Seven boxes left to right: Validate → Extract (PyPDF, per page) → Cache parsed text → Chunk (1500/50) → Tag metadata → Embed (MiniLM) + BM25 → Store and verify. Failure arrow from any step back to "Roll back vectors and files".

### Diagram 3: Query to answer (Slide 9)

Vertical: Question → (rewrite?) → Dense ∥ BM25 → RRF → Top 5 → Context builder (`Source: file, p. X` labels; grouped by paper if ≥2) → Prompt rules → LLM stream → `done` event → Citation check → UI (answer, passages, "Citations verified" or warning, trace).

### Diagram 4: Request lifecycle and isolation (backup / Q&A)

Browser (session cookie + CSRF header) → logging middleware (request id, JSON log, security headers) → auth middleware (session lookup by SHA-256; CSRF check on writes; 401/403) → route → Workspace(user N) → {private folder · BM25 for user N · ScopedStore adds `user_id = N` to every Chroma call} → shared ChromaDB; SQLite (users, sessions, history, usage). Matches the mermaid diagram in `ARCHITECTURE.md`.

### Diagram 5: Evaluation methodology (Slide 12)

Question sets (dev 56, held-out 60) → validator (quote on labelled page) → zero-LLM retrieval harness (own index, LLM blocked) → metrics (hit@k, MRR, P/R/nDCG, latency) → bootstrap CIs + paired deltas → adoption rule (no loss on dev AND gain on held-out) → adopt / reject. Side branch: answer evaluation (judge model, cached, quota-paced) → correctness, faithfulness, refusals, calibration. CI gate: hit@5 ≥ 89% on 29 reproducible questions.

---

# PART D: Charts from actual project data

**Chart 1: Hit@5 by retrieval method (Slide 13).** Horizontal bars with 95% CI whiskers; live configuration highlighted. Source: `eval/comparison_*_k5.md`.

| Method | Held-out hit@5 (n=47) | 95% CI | Dev hit@5 (n=50) |
|---|---|---|---|
| Dense only | 40 (85%) | 74–94% | 33 (66%) |
| BM25 only | 39 (83%) | 72–94% | 41 (82%) |
| **Hybrid RRF (live)** | **43 (91%)** | 83–98% | **42 (84%)** |
| Hybrid + MiniLM rerank | 45 (96%) | 89–100% | 41 (82%) |

**Chart 2: Hits by question type, held-out (Slide 7).** Grouped bars, three methods × four types.

| Type | Dense | BM25 | Hybrid |
|---|---|---|---|
| Exact term (12) | 9 | 12 | 12 |
| Lookup (12) | 9 | 11 | 11 |
| Paraphrase (13) | 12 | 6 | 10 |
| Multi-hop, every fact retrieved (10) | 4 | 8 | 8 |

**Chart 3: Quality vs latency (Slide 11 or backup).** Scatter: x = median retrieval latency (log scale), y = held-out nDCG@5.

| Config | Latency (ms) | nDCG@5 |
|---|---|---|
| BM25 | 1.3 | 0.454 |
| Dense | 13.6 | 0.390 |
| Hybrid (live) | 16.7 | 0.473 |
| + MiniLM rerank | 500.6 | 0.540 |
| + BGE rerank | 2,976 | 0.511 |

**Chart 4: Before/after the upgrade, answer level (Slide 13 or 11).** Paired bars. Source: `eval/ANSWER_QUALITY_k5.md` vs `_stage4.md`.

| Metric | Before | After |
|---|---|---|
| Correct answers (of 47) | 38 | 41 |
| False refusals (of 47) | 9 | 4 |
| Full evidence in prompt (of 47) | 35 | 40 |
| Sentences supported | 90% | 93% |

**Chart 5: Top-k trade-off (Slide 11 or backup).** Line chart. Source: `eval/ABLATION.md` (k=3 baseline).

| k | Dev hit (of 50) | Held-out hit (of 47) | Multi-hop all facts (of 10) | Prompt tokens |
|---|---|---|---|---|
| 3 | 37 | 41 | 5 | 1,021 |
| 5 | 41 | 43 | 8 | 1,619 |
| 8 | 44 | 44 | 9 | 2,477 |

**Chart 6: Original chunking grid (Slide 12, small).** Bar chart, 8 configurations. Source: `legacy/results/eval_results.json`. *Older lenient metric: keyword hit, top 3, 20 questions, 2 PDFs.*

| Size / overlap | 300/50 | 300/150 | 500/50 | 500/150 | 1000/50 | 1000/150 | 1500/50 | 1500/150 |
|---|---|---|---|---|---|---|---|---|
| Hit rate | 65% | 65% | 75% | 70% | 65% | 70% | **80%** | 80% |

LLM-judge score was recorded only for the winner (4.15/5 ± 0.14 over 2 runs); runner-up judge scores are **not available** in the result file.

**Chart 7: Cold start (backup).** Two bars: 17 s → 0.18 s first request after restart.

**Do not chart (insufficient or unverified data):** per-question answer latency distributions (only single traces exist); any comparison against other RAG systems (none was run); chunking-grid judge scores for configurations other than the winner.

---

# PART E: Screenshots to capture

**Privacy first:** the sidebar lists your library's PDF names. Sign in with an account that holds only the evaluation PDFs (not your résumé), or capture at a window width under 960 px, where the sidebar collapses.

| # | For slide | What to capture | How |
|---|---|---|---|
| 1 | 10 (Chat) | A finished answer with citations, the "Citations verified" badge and the **Pipeline trace** expanded | Ask *"What sustained TFLOPS figure did the authors assume for a P100 GPU?"*, then open the trace |
| 2 | 7 / 8 | **Inspect** for the LSAT query showing dense rank, BM25 rank and RRF columns | Already captured: `docs/screenshots/inspect.png`. Recapture if you want the light theme |
| 3 | 7 | Inspect for *"I have a full-time job; how much should I study on workdays?"*, showing dense #1 and BM25 lower | Inspect mode, top-k 5 |
| 4 | 10 (Quiz) | A generated quiz (mixed types) and then the graded results | Quiz on `source_doc.pdf`, topic "attention", Mixed |
| 5 | 10 (Compare) | Comparison of `source_doc.pdf` vs `gpt4_technical_report.pdf` on "Methodology" | Compare mode, the chip "Methodology" |
| 6 | 10 (Accounts) | The sign-in page, plus the "Check your email" code step | Log out; start Create account |
| 7 | 13 / backup | The benchmark dashboard switched to **nDCG@k** | Inspect mode, scroll down |
| 8 | 9 / backup | A multi-paper answer with one section per paper | Chat with two papers selected: *"How is model performance evaluated?"* |
| 9 | Q&A backup | Account settings dialog (change password, sign out everywhere, delete account) | Account menu → Account settings |

Tip: use a 1440 × 900 window, 100% zoom, and crop each shot to the panel that matters.

---

# PART F: Likely judges' questions, with answers grounded in the implementation

**1. Isn't this just a PDF chatbot?**
No, in three measurable ways. Retrieval is hybrid, and I can show per-question-type data for why. Answers are constrained to retrieved passages, and citations are verified deterministically. Every retrieval setting was chosen or rejected by a dev/held-out benchmark with confidence intervals. That process found a bug that was costing answer quality.

**2. Why not just use a bigger embedding model?**
I tested a model with a longer, 512-token window (bge-small). It helped semantic-only search but not the hybrid system, because BM25 already covers what MiniLM can't see. I also tested small-to-big retrieval for the same reason, with the same outcome.

**3. Why RRF instead of a weighted score combination?**
The scores live on different scales: cosine in [−1, 1], BM25 unbounded and corpus-dependent. A weighted sum needs normalisation and a weight, both of which would have to be tuned on the test data. RRF uses only ranks, with k = 60 fixed in advance.

**4. When does RRF fail?**
It rewards agreement, so a page ranked highly by only one retriever can be outvoted. Held-out question v2_42: dense ranked the gold page 3rd and BM25 5th, yet fusion placed it 7th, outside the top 5.

**5. Your reranker improved the held-out score. Why is it off?**
The adoption rule says: don't lose on dev, improve on held-out. The MiniLM reranker gained on held-out (nDCG +0.067) but lost on dev (3 better, 4 worse), even after I corrected the dev labels. It also adds about 0.5 s per query on CPU. The BGE reranker didn't lose on dev, but gained only one held-out question at about 3 s per query.

**6. How do you know your evaluation isn't overfitted?**
Decisions are made on dev and confirmed on held-out. The adoption rule and settings like RRF k were fixed before measuring. I stopped tuning small-to-big after one pre-registered run precisely to avoid fitting the test set. And I state the limitation: the held-out set has been looked at many times now, so the next decisions need a fresh set.

**7. How do you prevent hallucination?**
Four layers, each measured:
- The prompt restricts answers to the passages, with a fixed refusal sentence: 12/13 unanswerable questions declined, and the 13th says the documents don't cover it.
- Partial answers must name what's missing.
- Cited pages are checked against the retrieved pages.
- An LLM self-check was built, calibrated and switched off because it marked 43/43 answers as supported.

**8. What's the difference between hit rate and correctness?**
Hit@5 asks whether a chunk from the gold page is among the five passages; correctness asks whether the final answer is right (judged against a reference answer by a second model). They can diverge: the de-duplication bug kept hit rate unchanged while dropping the actual answer chunk.

**9. How did you evaluate answer quality without humans?**
A separate judge model (`gpt-oss-20b`, temperature 0) grades correctness against a reference answer and checks each sentence against the passages. I report its limits: same model family, and the labels aren't yet human-reviewed.

**10. How large is the evaluation, and is that enough?**
47 answerable + 13 unanswerable held-out questions, and 50 + 6 dev. Confidence intervals are about ±10 points, so I report paired better/worse counts, which are more sensitive than the overall rates.

**11. How does multi-user isolation work, and how do you know it holds?**
Each user has a private folder, their own BM25 index, and a `ScopedStore` that injects their user id into every Chroma search, read, write and delete. Fusion also rejects any chunk not in the user's own list. An end-to-end test uses real PDFs with secret tokens for two users and attempts every cross-user path: search, chat prompt, quiz, compare, preview, delete, identical filenames and path traversal.

**12. Why ChromaDB rather than FAISS or a hosted vector database?**
It's embedded and persistent, with metadata filtering. Isolation and per-paper search are both metadata filters, which FAISS doesn't provide natively. A hosted database would add a network dependency and cost to a local-first app.

**13. What are the latency and cost per answer?**
Retrieval is about 17 ms (median, CPU). One measured chat answer: first token at 1.19 s, about 2,660 tokens. The binding constraint is Groq's free-tier rate of about 8k tokens per minute.

**14. What was the hardest bug?**
Passages were de-duplicated by page label, so when a page produced two chunks, the second was silently dropped, often the one with the answer. Retrieval metrics didn't show it, because the right page was still "retrieved". Measuring whether the gold quote actually reached the prompt exposed it: 35/47 → 40/47 after the fix.

**15. What would you do with more time?**
Human review of the evaluation set and a fresh, larger held-out set; OCR and table-aware extraction; re-testing BGE reranking on a GPU; and an independent faithfulness verifier for answer-level confidence. Each would be measured under the same adoption rule.

**16. Did you write the code yourself?**
`[Answer truthfully about your own role and any AI assistance used. Judges respect clarity about how the work was done, and you should be able to explain every decision on slides 7–12 in your own words.]`

---

# PART G: Final quality checklist

**Accuracy**
- [ ] Every number on a slide appears in the fact sheet above, and the fact sheet's source file still shows it
- [ ] "Hybrid beats BM25" is never claimed; only "beats dense-only" and "never worst by type"
- [ ] Reranking, the LLM support check and cross-page chunking are described as **off by default**, never as live features
- [ ] Small-to-big appears only as **evaluated and rejected**
- [ ] Docker is described as **written, not built**; no OCR, table extraction or semantic chunking is claimed
- [ ] The frontend is described as vanilla JS + custom CSS (not Tailwind)
- [ ] Limitations slide includes: small eval sets, labels not human-reviewed, no OCR, no text cleaning, free-tier throughput

**Story**
- [ ] Problem → solution → how it works → what's interesting → evidence → limits → next, in that order
- [ ] Slides 7 and 8 (hybrid retrieval, RRF) get the most time
- [ ] The de-duplication bug story appears once in full (Slide 11) and is referenced in the results

**Design**
- [ ] No slide has more than ~40 words of on-slide text (details live in speaker notes)
- [ ] One accent colour per idea: blue = retrieval / data, violet = LLM / AI
- [ ] Charts have labelled axes, units and sample sizes; CI whiskers where available
- [ ] Screenshots framed consistently and contain no personal filenames or emails

**Delivery**
- [ ] Rehearse slides 7–8 until the RRF worked example takes under 90 seconds
- [ ] Have the app running locally for a live Inspect demo (with a fallback screenshot if offline)
- [ ] Prepare the honest answer to Q16 (your role)
- [ ] Time the full talk; target about 1 minute per slide, about 2 minutes for slides 7, 8 and 11
