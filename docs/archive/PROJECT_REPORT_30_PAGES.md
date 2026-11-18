<!-- PAGE 01 -->
# Hybrid RAG for Evidence-Grounded Knowledge Discovery & Learning
## Complete Project Report and Demonstration Guide

**Project:** Hybrid RAG for Evidence-Grounded Knowledge Discovery & Learning
**Application:** FastAPI backend with a Tailwind-style vanilla JavaScript web frontend
**Core techniques:** PDF ingestion, local embeddings, Chroma retrieval, BM25, reciprocal-rank fusion, grounded language-model generation, evaluation, and interactive study workflows
**Prepared:** November 2026

### Project in one sentence
This application turns a collection of research PDFs into an evidence-oriented study workspace: users can ask cited questions, generate and take quizzes, compare two papers, and add new PDFs through the interface.

### Demonstration promise
This report includes a presenter script and a repeatable demonstration sequence. The demonstration is designed to show normal product behavior without assuming that evaluation scripts or internal developer tools are part of the end-user workflow. Live model features require the project’s configured Groq access; local PDF parsing, chunking, and embeddings are separate from that hosted generation dependency.

**Audience:** a teacher, project reviewer, interviewer, or teammate who wants to understand what the project does, how information moves through it, and how to demonstrate its features.

**Document note:** this is a code-based explanation of the repository’s architecture and the recorded evaluation artifacts, not a claim that every response is guaranteed correct. The project emphasizes traceable retrieved evidence and exposes several evaluation limitations explicitly.

<!-- PAGE 02 -->
# Executive Summary

Hybrid RAG for Evidence-Grounded Knowledge Discovery & Learning is a hybrid retrieval-augmented generation application for evidence-grounded research paper analysis and interactive learning. Rather than sending a question to a language model without context, the application searches a local collection of embedded PDF chunks, attaches filename and page citations, and asks the model to answer from those passages. That retrieval step is the key architectural distinction from a general chatbot.

The FastAPI application has three workflows. **Chat** answers a question using hybrid dense-plus-lexical retrieval and shows source context, similarity information where available, and a heuristic trust label. **Quiz Mode** retrieves from one selected PDF, generates five structured questions, collects typed answers, grades them against the retrieved passages, and preserves the attempt in session state. **Compare Mode** searches two PDFs separately and asks for a cited answer for each plus a concise comparison. A sidebar uploader can save a new PDF into the library and add its chunks to the local vector store.

The local retrieval stack uses `all-MiniLM-L6-v2` embeddings and Chroma. BM25 complements semantic similarity with exact-term matching. Reciprocal Rank Fusion (RRF) combines their rankings without pretending that their raw scores share a scale. Groq provides hosted language-model generation through the project’s retry helper in workflows that use it.

The repository also contains offline and API-backed evaluation tools, a cross-encoder experiment, retrieval diagnostics, and a separate autonomous retrieval loop. These tools help explain system behavior but are not all called by the Streamlit chat path. Recorded evaluations show that retrieval methods behave differently across question sets; they are evidence about this corpus and metric, not universal performance guarantees.

<!-- PAGE 03 -->
# Contents and Reading Guide

1. Cover and project identity
2. Executive summary
3. Contents and reading guide
4. Project goals and intended users
5. Architecture overview and component map
6. Running the application and its dependencies
7. PDF ingestion from file to page objects
8. Metadata, citations, and provenance
9. Chunking strategy and the selected configuration
10. Embeddings and the Chroma vector store
11. Hybrid retrieval: dense, BM25, and RRF
12. Chat Mode end-to-end
13. Grounding, prompts, citations, and refusal behavior
14. Trust meter and its interpretation
15. Quiz generation from a selected paper
16. Interactive quiz state and answer collection
17. Grading answers from retrieved evidence
18. Compare Mode and evidence isolation
19. Adding PDFs through the interface
20. AutonomousRAGAgent loop
21. Evaluation suite and experiment design
22. Datasets and retrieval metrics
23. Recorded dense-versus-hybrid results
24. Cross-encoder re-ranking experiment
25. Retrieval miss diagnostics
26. Supporting command-line modules and tests
27. Data, privacy, and operational boundaries
28. Limitations and responsible interpretation
29. Live demonstration script
30. Presenter wrap-up, glossary, and references

### How to use this report
Pages 4–20 explain what the application does during real user interactions. Pages 21–26 describe the experiments and supporting scripts. Pages 27–28 provide operational and interpretation guidance. Pages 29–30 are written to be presented aloud during a classroom or review demonstration.

### Main source files
The architecture is chiefly visible in `config.py`, `ingest.py`, `hybrid.py`, `main.py`, `quiz.py`, `compare.py`, `trust.py`, `agent.py`, and the browser files `static/index.html` and `static/app.js`. Evaluation is implemented in `evaluator.py`, `hybrid.py`, `rerank.py`, and `diagnose_misses.py`.

<!-- PAGE 04 -->
# Project Goals and Intended Users

The main goal is to make a set of technical papers easier to study without losing the connection between an answer and its source. A reader can pose a natural-language question, request practice questions about a topic, or ask how two papers differ. The application tries to keep each task tied to passages retrieved from the chosen documents.

### Intended users
- Students learning a research topic and needing page references for follow-up reading.
- Teachers demonstrating retrieval-augmented generation and evidence-grounded prompting.
- Researchers who want a quick way to find relevant passages across a small PDF collection.
- Developers evaluating retrieval methods on a known, answerable question set.

### Product goals
1. Accept research PDFs and preserve their identity and page metadata.
2. Retrieve relevant text locally without sending the entire PDF to the hosted model.
3. Give concise answers with source citations and a refusal instruction for missing evidence.
4. Add learning-oriented workflows that reuse the same retrieval foundation.
5. Make retrieval quality measurable through reproducible question sets and explicit metrics.

### Non-goals
This repository is not a full document-management system, a scholarly search engine, a citation manager, or a formally validated grading product. It does not claim that a language-model answer is correct merely because a passage was retrieved. Nor does it replace reading the paper when the user needs a complete or legally authoritative interpretation.

The system is optimized for a manageable, local collection. Its design favors modular Python code and inspectable behavior: paths and prompts are centralized, document processing has a separate module, retrieval has named strategies, and experiments write JSON results that can be reviewed independently of the interface.

<!-- PAGE 05 -->
# Architecture Overview

The project has four logical layers:

**1. Configuration and data.** `config.py` defines corpus paths, collection names, model identifiers, top-k settings, the answer prompt, and citation formatting. PDFs live in `data/`.

**2. Document preparation and retrieval.** `ingest.py` loads and splits PDFs. `all-MiniLM-L6-v2` embeds each chunk. Chroma persists vectors and metadata. `hybrid.py` adds BM25 keyword ranking and RRF fusion.

**3. Application workflows.** `main.py` exposes FastAPI REST endpoints for Chat, Quiz, Compare, PDF uploads, settings, and history. `static/index.html` and `static/app.js` provide the browser interface, while `quiz.py` and `compare.py` keep workflow-specific prompt and validation logic out of the API layer.

**4. Evaluation and diagnostics.** `evaluator.py`, `hybrid.py`, `rerank.py`, `run_judging.py`, and `diagnose_misses.py` quantify retrieval or answer behavior. `trust.py` implements the answer-support heuristic and refusal measurement.

```text
PDF → page loader → source/page metadata → chunks → embeddings → Chroma
                                                     │
Question → dense ranking ─┐                          │
Question → BM25 ranking ──┴→ RRF → cited context → LLM → answer/UI
```

### Important separation
The FastAPI backend and static browser frontend are the user-facing product. Evaluation scripts are experiments and may use their own indexes, datasets, or model calls. The standalone `AutonomousRAGAgent` uses its own fixed evaluation index and should not be confused with the default Chat route.

<!-- PAGE 06 -->
# Running the Application

The repository uses a Python virtual environment and lists its direct dependencies in `requirements.txt`.

## Technology Stack Used

| Category | Technology used in the project | Purpose |
|---|---|---|
| Programming language | **Python 3.11** | Main implementation language for ingestion, retrieval, generation, evaluation, CLI tools, and tests. |
| Backend web framework | **FastAPI** | Provides REST endpoints for Chat, Quiz, grading, Compare, uploads, library management, settings, and progress. |
| Frontend | **HTML, Tailwind CSS CDN, and vanilla JavaScript** | Provides the responsive dark Chat, Quiz, Compare, and Paper Library interface without a Node.js build step. |
| ASGI server | **Uvicorn** | Runs the FastAPI backend locally on port 8000 and serves the static frontend. |
| Large language-model service | **Groq API** | Runs the hosted `openai/gpt-oss-120b` language model for grounded answers, quiz generation, grading support, and paper comparison. |
| Environment configuration | **python-dotenv** | Loads the Groq API key and other environment settings from `.env`. |
| PDF processing | **PyPDF**, **langchain-community/PyPDFLoader** | Reads PDF files and produces page-level documents. |
| Document splitting | **LangChain Text Splitters** | Splits loaded PDF pages into overlapping text chunks for retrieval. |
| Embedding model | **Hugging Face Sentence Transformers**, **all-MiniLM-L6-v2** | Converts chunks and questions into semantic vectors locally on the CPU. |
| Vector database | **Chroma**, **langchain-chroma** | Persists document embeddings and metadata for similarity search. |
| Lexical retrieval | **rank-bm25 / BM25Okapi** | Finds passages through exact and statistical keyword matching. |
| Hybrid search | **Python ranking logic with Reciprocal Rank Fusion (RRF)** | Combines dense vector rankings and BM25 rankings without directly comparing their different score scales. |
| Retrieval framework | **LangChain Core and integrations** | Provides shared document objects, embeddings, vector-store integration, and text-processing components. |
| Evaluation and diagnostics | **Python, JSON result files, custom evaluator scripts** | Measures retrieval hit rate, chunking configurations, answer quality, misses, and autonomous-loop behavior. |
| Testing | **Python `unittest`** | Covers library removal, PDF upload, quiz, and comparison workflows. |
| Development and execution | **PowerShell, pip, Python virtual environment** | Runs local commands, installs dependencies, starts FastAPI, and performs tests. |

### Main system architecture

- **Frontend / UI:** HTML plus Tailwind CSS CDN and vanilla JavaScript in `static/index.html` and `static/app.js`.
- **Backend / application logic:** FastAPI in `main.py`, with retrieval and workflow modules in Python.
- **Retrieval:** hybrid dense embeddings plus BM25 lexical search.
- **Vector storage:** locally persisted Chroma database.
- **Generation model:** Groq-hosted large language model accessed through the Groq Python SDK.
- **Document format:** PDF; PDF parsing uses PyPDFLoader.
- **Embeddings:** the Sentence Transformers model `all-MiniLM-L6-v2`.
- **Evaluation output:** structured JSON files.

The project does not require a separately deployed database server or frontend build server. FastAPI combines local PDF processing, vector retrieval, and hosted Groq generation with the static browser interface.

### Typical local startup
From the project directory in PowerShell:

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in the browser. FastAPI serves both the API and the frontend. If the main index has not been built, run `python ingest.py` before starting the server. Never paste secrets into a report, source file, screenshot, or public repository.

### First-run behavior
The embedding model may need to be downloaded the first time it is loaded. The local vector store is opened through a cached resource. If the main Chroma collection is empty, the app can load PDFs and populate the collection. Language-model requests happen only when a workflow is submitted; opening the interface does not itself need to answer a question.

### User actions that may call the hosted model
- Submit a normal Chat question.
- Generate a quiz and submit quiz answers for grading.
- Submit a Compare question.
- Run the non-mocked judging or refusal-evaluation commands.

### What a successful startup demonstrates
The browser page renders, the sidebar displays the current PDF library, and the Chat, Quiz, Compare, and settings controls are available. Startup alone is not proof that Groq credentials, an index, or every model request will work; those are exercised by their respective operations.

<!-- PAGE 07 -->
# PDF Ingestion: From File to Page Objects

`ingest.py` is the main batch-ingestion module. It scans the configured `data/` directory for PDFs and loads each file using `PyPDFLoader`. The loader returns page-level documents, which is useful because page boundaries can be carried into citations instead of being discarded when the text is combined.

For every loaded page, ingestion adds a stable `filename` metadata field. This is the bare PDF name rather than a machine-specific absolute path. Existing loader metadata such as the page index remains attached to the document. The tagged pages are then split into text chunks by LangChain’s `RecursiveCharacterTextSplitter`.

The normal batch command is:

```powershell
.\venv\Scripts\python.exe .\ingest.py
```

The command reads the chosen chunk size and overlap, loads the pages, creates chunks, initializes the embedding model, and writes the chunks to the configured Chroma collection. It rebuilds the main store to avoid leaving stale duplicate chunks after a re-ingestion.

### Why page-level loading matters
Retrieval operates over chunks, but users need citations at page granularity. Retaining page metadata through chunk splitting lets the application show a useful source label later.

### Why chunking follows loading
The loader handles PDF parsing and page metadata; the splitter handles chunk boundaries. Keeping those responsibilities separate makes it possible to evaluate different chunk settings without rewriting the PDF loader.

The interface also has a dedicated upload path, described on page 19. It indexes an uploaded file directly rather than rebuilding the whole collection.

<!-- PAGE 08 -->
# Metadata, Citations, and Provenance

Each chunk is more than raw text. Its metadata records which file it came from and the page that produced it. `config.py` provides `source_filename()`, `page_number()`, and `citation()` so that answer, quiz, and comparison workflows use consistent labels.

The human-readable citation format is:

```text
filename.pdf, p. 3
```

PDF loaders commonly store `page` as a zero-based integer. `page_number()` converts this to a one-based number for readers. If a loader supplies a page label, the helper prefers that label. If neither format is available, the helper falls back to `?` rather than inventing a page.

### Where citations appear
- Chat context includes a source label before each retrieved passage; the answer prompt instructs the model to cite those labels.
- Quiz-generation prompts give each passage a citation, and generated question objects are validated against the set of retrieved labels.
- Quiz grading asks for a source label from the evidence context and validates that it matches a label present there.
- Compare Mode labels both paper contexts separately and asks for each paper’s response to use its own citations.
- The UI can expand the evidence used by a response for inspection.

### Why provenance is important
A citation makes an answer auditable: the reader can return to the PDF and inspect the underlying statement. It does not guarantee that the model interpreted the statement correctly, and a citation can still be wrong if generation disobeys its prompt. That is why the app also exposes retrieved passages and why the evaluation scripts separately measure retrieval.

<!-- PAGE 09 -->
# Chunking Strategy and Selected Configuration

The text splitter uses a chunk size and overlap. Chunk size determines approximately how much text is kept together; overlap repeats a small boundary region in adjacent chunks. A short chunk may be precise but omit surrounding explanation. A long chunk can preserve context but include more unrelated material. Overlap reduces the chance that an important sentence is cut at a boundary.

`best_chunk_config()` in `ingest.py` reads `best_config` from `eval_results.json` when it contains usable values. Otherwise, it falls back to constants in `config.py`. The evaluator tests four sizes—300, 500, 1000, and 1500 characters—with overlaps of 50 and 150 characters, creating eight configurations.

The recorded winning configuration is **1500 characters with 50-character overlap**, with **281 chunks** in the evaluated two-PDF corpus. Its recorded retrieval hit rate is **0.80** on the initial 20 answerable questions. The recorded LLM-judge mean is **4.15/5**, with a run standard deviation of **0.141**.

### Interpretation
These figures summarize a particular corpus, question set, chunking grid, and evaluation implementation. They are not universal settings for all documents. Adding a PDF changes the content and the number of chunks; therefore a configuration selected on the original corpus should be re-evaluated if quality claims are being made for a materially changed library.

The main app uses the best configuration reported in its evaluation file. The specialized autonomous agent and reranker experiment explicitly target the historical 1500/50 evaluation index and validate its chunk count; they are not automatic substitutes for rebuilding an updated index.

<!-- PAGE 10 -->
# Embeddings and the Chroma Vector Store

An embedding model maps text into a numeric vector. The project uses the local sentence-transformer model `all-MiniLM-L6-v2` for chunks and user queries. At search time, a query vector is compared with stored chunk vectors so semantically similar passages can be ranked even when they do not repeat the same wording.

Chroma is the persistent vector store. The main store uses the collection name `documents` and the path defined by `CHROMA_DIR`. The FastAPI application creates the embedding function through a cached resource and opens the store through a second cached resource. This avoids reloading the model and database connection on every request.

### Persistence and rebuilds
Embeddings and chunk metadata live in Chroma’s local on-disk database. Batch ingestion rebuilds the main index. Evaluation scripts may create separate `chroma_db_eval_*` stores for experimental configurations. Those indexes are distinct from the main application index and must not be confused when interpreting results.

### Resource consistency
The app’s hybrid retrieval sidecar reconstructs chunks using the best chunk configuration and checks that the reconstructed chunk count matches the active vector collection. If the counts differ, it raises a clear error instead of fusing rankings from different corpora.

### Local does not mean entirely offline
PDF parsing, splitting, embeddings, Chroma, and BM25 run locally. Generation and some support checks use Groq’s hosted model. Thus the product has a local retrieval layer plus a remote generation dependency. The two sides have different availability, privacy, and cost implications.

<!-- PAGE 11 -->
# Hybrid Retrieval: Dense, BM25, and RRF

Dense retrieval and lexical retrieval solve related but different problems. Dense embeddings can connect paraphrases and concepts. BM25 rewards terms that occur in the query and chunk, which is often helpful for exact model names, numbers, acronyms, and symbols. Neither signal is sufficient for every question.

`hybrid.py` tokenizes text into lowercase alphanumeric terms and creates a BM25 index over the same chunk set used by dense retrieval. It obtains up to ten candidate chunks from each method, then merges the rankings using Reciprocal Rank Fusion. RRF gives each ranked item a contribution based on its position rather than averaging incomparable raw scores:

```text
RRF score(chunk) = sum over rankings of 1 / (k + rank)
```

The project fixes `k = 60`, uses ten candidates per method, and normally returns the top three fused chunks. Stable tie-breaks make ordering reproducible when scores are equal.

### Per-document filtering
Quiz and Compare calls pass a filename metadata filter. Dense search applies that filter in Chroma; the BM25 side rebuilds its candidate ranking from chunks whose metadata matches the same filter. Thus a paper’s context is isolated on both sides before fusion.

### Chat scores
Chat uses a score-bearing variant that preserves dense relevance scores for the trust meter. A chunk that enters only from the BM25 ranking may not have a dense score; that value is represented as unavailable rather than fabricated.

<!-- PAGE 12 -->
# Chat Mode: End-to-End Data Flow

The regular Chat path begins when the user enters a question in the browser and submits it to FastAPI. `retrieve_chunks()` gets the cached BM25 corpus and Chroma store, then calls the hybrid score-bearing retrieval helper. The helper fuses dense and BM25 candidate rankings and returns the top three chunks.

The API converts each result into a JSON record containing filename, citation, passage text, and a dense score when available. `format_context()` joins the text passages under their source labels. The Chat endpoint formats the shared prompt from `config.py`, asks the configured Groq model for an answer, and returns the response and citations to `static/app.js` for browser display.

After generation, the app calls `trust.assess()` with the question, answer, and retrieved chunk records. It renders a trust label and reason, then offers an expander with the raw retrieved passages and scores. The assistant message, evidence, and trust result are saved in `st.session_state["messages"]`, so reruns can redraw the conversation without regenerating the answer.

```text
User question → hybrid top 3 → cited context → streamed answer
                      │                       │
                      └── similarity scores ──┴→ trust assessment
```

### Why show the evidence?
The reader can check whether the retrieved material actually supports the answer. This is especially valuable when a fluent answer seems plausible but the question is outside the PDFs or the retrieval ranking missed the needed passage.

<!-- PAGE 13 -->
# Grounding, Prompts, Citations, and Refusal

The shared answer prompt in `config.py` instructs the model to use only the supplied context, avoid outside knowledge, and reply with a fixed refusal sentence when the answer is not present. It also asks for a short response with page citations and a final `Sources:` line.

The context is assembled from retrieved chunks, each prefixed with its source citation. The score is not included in the answer prompt: it is used as application metadata and for the trust heuristic, not as evidence about the paper’s claims.

### Refusal behavior
The prompt gives the model a refusal pattern, and `trust.py` recognizes that configured sentence when measuring refusals. The refusal evaluation distinguishes intentionally unanswerable prompts from answerable questions, so a correct refusal on one group is not confused with an undesirable refusal on the other.

### Prompting is a control, not a proof
Natural-language instructions reduce the chance of unsupported answers but cannot mathematically guarantee that a model will obey. A model may misunderstand a passage, omit a caveat, cite a nearby but insufficient page, or produce an unjustified refusal. The app’s evidence display and evaluation tools are complementary safeguards, not formal verification.

### Why context boundaries matter
Quiz and Compare prompts use explicit evidence blocks. Compare labels each paper’s evidence separately and prohibits borrowing claims across papers. Quiz generation and grading operate on the selected PDF’s retrieved passages. These design choices reduce cross-document leakage and make the context boundary visible in code.

<!-- PAGE 14 -->
# Trust Meter: Signals and Rules

`trust.py` combines two signals after a normal answer is generated:

**Retrieval strength.** It reads the similarity scores available for retrieved chunks and summarizes the top and mean score. The current strong-retrieval rule is top score at least 0.40 **or** mean score at least 0.50.

**Support check.** One extra model request asks whether every answer claim is supported by the retrieved context. The expected verdict is `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`. The parser checks `UNSUPPORTED` before `SUPPORTED`, since the first word contains the second as a substring.

| Support verdict | Strong retrieval | Label |
|---|---:|---|
| SUPPORTED | Yes | High |
| SUPPORTED | No | Medium |
| PARTIAL | Yes | Medium |
| PARTIAL | No | Low |
| UNSUPPORTED | Either | Low |
| Missing or unparseable | Yes | Medium |
| Missing or unparseable | No | Low |

A fixed refusal is labeled `n/a` and skips the check because it makes no factual claims. The result is deliberately described as a **heuristic**, not a measured probability or accuracy score. The thresholds were selected by inspecting the current evaluation-question score distribution and were not calibrated on an independent confidence dataset.

<!-- PAGE 15 -->
# Quiz Mode: Grounded Question Generation

Quiz Mode is a study workflow rather than a second general chat interface. The user selects one PDF and enters a topic. Retrieval is run through the hybrid search function with a filename filter, so both BM25 and dense candidates are restricted to the chosen paper.

`quiz.py` formats the retrieved passages with citation labels and sends a single generation prompt. The model is asked for structured JSON containing a coverage flag and five question objects. Each question includes its text, a short answer used for the later result display, and a source citation. The prompt requires the model to use only the retrieved text and return no quiz when the topic is not sufficiently covered.

The Python code validates the response rather than blindly trusting model formatting. It checks the response shape, requires exactly five questions when coverage is true, requires nonempty question/answer/citation strings, and rejects citations that do not exactly match one of the retrieved passages’ citation labels. When there are no retrieved chunks, it returns an uncovered result without making a generation request.

### User-visible sequence
1. Enable Quiz Mode.
2. Select a PDF and topic.
3. Generate the five questions.
4. Answer each question in a form.
5. Submit the answers for evidence-grounded grading.

The source passages remain available in an expander after the quiz is generated, making the evidence behind the questions inspectable.

<!-- PAGE 16 -->
# Interactive Quiz State and Answer Collection

Browser interactions issue FastAPI requests, so generated quiz state is represented explicitly in JavaScript and server-side attempt data. `make_quiz_state()` remains part of the supporting Python workflow and creates serializable state containing an attempt identifier, selected PDF, topic, generated questions, answer slots, retrieved context, displayed chunks, and an initially empty grade.

The attempt identifier is used in the form and text-area widget keys. This keeps answer inputs associated with one quiz attempt and prevents a newly generated quiz from accidentally reusing stale widget values. The `quiz_state` object is held in `st.session_state`, so it survives the reruns caused by typing and button presses.

The answer form renders one text area per question and submits all answers together. A single combined grade request is made after submission. On success, the state is replaced with a copy that includes the user’s answers and final grade. The UI then reruns and renders the result from that stored state instead of asking the model again.

### State progression

```text
No attempt
  → generated questions + evidence + blank answers
  → user-filled form
  → submitted answers + grade
  → score, per-question feedback, missed answers and citations
```

The browser form collects radio or text answers until submission. A different PDF or topic does not display a mismatched prior result. A new generation creates fresh questions and unanswered answer slots.

<!-- PAGE 17 -->
# Quiz Grading: Context Is the Authority

The grading request contains the retrieved source context, each generated question, and the user’s response. It does **not** send the generated answer key as the grading reference. Instead, the model is instructed to infer a correct answer from the original passages and compare the user response with that evidence. This reduces the risk that an unsupported generated answer becomes the grading authority.

For every question, the response must contain one of three labels: **Correct**, **Partial**, or **Incorrect**, plus feedback, an evidence-based correct answer, and a citation. The code validates that there are five results, that grades use the allowed labels, and that source labels occur in the supplied context.

Scoring is explicit: Correct is one point, Partial is half a point, and Incorrect is zero. The final score can therefore include half points, such as 3.5/5. The results page lists the status and feedback for all questions and separately highlights questions that were not fully correct with the cited source answer.

### What this does and does not guarantee
The prompt constrains grading to retrieved evidence, and code validates response structure and citations. It does not independently prove semantic correctness of every grade. The model can still misread evidence or apply an overly strict or permissive standard. The feature is best understood as an interactive, evidence-constrained study aid, not a certified examination platform.

<!-- PAGE 18 -->
# Compare Mode: Two Isolated Paper Contexts

Compare Mode asks the user to select exactly two PDFs and enter one comparison question. `retrieve_paper_contexts()` performs two separate hybrid searches. The first call filters by Paper 1’s filename and the second filters by Paper 2’s filename; their dense and BM25 candidate sets are fused independently.

The two context lists are sent together in one combined model prompt. Each context has its own heading and citation-labeled evidence. The prompt tells the model not to transfer evidence from one paper to the other. It asks for one cited response for each paper and a two-sentence summary of how their approaches or answers differ.

If a paper yields no retrieved passages, the prompt includes an explicit empty-evidence marker. The model is instructed to say that the paper does not cover the topic rather than infer an answer from the other paper.

The app persists the selected filenames, question, generated text, and both passage lists in session state. The answer is shown only while the current selections and question match the saved result. An expander exposes each paper’s retrieved passages independently.

### Why one combined request?
One prompt reduces the number of separate generation requests and gives the model both contexts for comparison. Isolation is established before generation, however: the two retrieval calls and evidence sections remain distinct. Combined generation is not the same as mixing retrieval sources.

<!-- PAGE 19 -->
# Adding a PDF Through the Interface

The Paper Library sidebar includes an **Add PDF** uploader. The user chooses or drops a PDF into the browser upload control. FastAPI opens or initializes the persistent Chroma store, reads the current best chunk configuration, and passes the uploaded bytes to `upload.py`.

The upload helper validates a simple `.pdf` filename and the PDF header, refuses to overwrite an existing filename, writes to a temporary file, and parses the pages with `PyPDFLoader`. Each page is assigned the final filename and final source path. Pages are split with the same chunking function used by ingestion, then the chunks are added to Chroma. After success, the temporary file is atomically renamed into `data/` and the cached hybrid resources are cleared so BM25 includes the new chunks on the next retrieval.

If parsing or indexing fails, the helper removes temporary files and deletes partial vector records for that filename before re-raising the error. The UI surfaces a readable error instead of showing a success message.

### Important operational details
- The current implementation does not overwrite same-named PDFs; rename the file first.
- The new document is added to the main application store, not every historical evaluation index.
- The in-memory BM25 cache refreshes after a successful upload.
- This upload route is preferable to copying a new file into `data/` while the app is running, because a non-empty existing index is not automatically reconciled against arbitrary directory changes.

<!-- PAGE 20 -->
# AutonomousRAGAgent: Bounded Self-Reflection

`agent.py` contains a separate Python class, `AutonomousRAGAgent`, that implements an explicit retrieval loop:

1. Retrieve top-k chunks using hybrid search against the fixed 1500/50 evaluation index.
2. Ask the model whether the context is relevant; it must answer YES or NO.
3. If YES, generate an answer from the retrieved context using the shared cited-answer prompt.
4. If NO, ask the model to rewrite the search query and retry.
5. After two rewrites, refuse with a fixed “I don't know” response if the context remains irrelevant.

The class lazily initializes its vector store, reconstructs the expected chunk corpus, checks that the persisted index has the expected number of chunks, and builds an in-memory BM25 index. Its mocked `__main__` test verifies both the rewrite-to-success and exhausted-retry refusal paths without network calls.

### Relationship to the app
This agent is currently a standalone class and testable module. The default FastAPI Chat workflow directly calls hybrid retrieval and answer generation; it does not instantiate `AutonomousRAGAgent`. This distinction is important when describing the architecture: the repository contains an autonomous loop, but normal UI Chat is not that loop.

### Fixed-index caveat
The agent’s historical index path and expected count are fixed in the module. If the PDF corpus changes, the index and count need corresponding maintenance; otherwise the agent intentionally fails rather than silently using an incompatible corpus.

<!-- PAGE 21 -->
# Evaluation Suite and Experiment Design

`evaluator.py` separates a deterministic retrieval phase from a more expensive answer-and-judge phase. It loads the answerable questions, excludes questions flagged as should-refuse, and tests eight configurations: four chunk sizes and two overlaps. Each configuration has its own persisted evaluation index so runs can be resumed without recomputing every embedding.

The retrieval phase uses no model-generation calls. It checks the top-k retrieved chunks and records the number of hits, rate, and missed question IDs. Results are saved incrementally in `eval_results.json`, and the best configuration is derived from retrieval and judge summaries.

For selected configurations, the evaluator answers questions from retrieved context and asks a model judge to score answer correctness and faithfulness on a 1–5 scale. It repeats judgments and records per-run means and standard deviation. `run_judging.py` provides another resumable path for a fixed subset of configurations, saving after runs and supporting a dry-run plan.

### Experimental hygiene
Question-set hashes help detect stale retrieval measurements. Held-out and refusal questions are excluded from the original judge group to reduce leakage and keep metrics meaningful. Other scripts compare dense and hybrid search or evaluate reranking on named sets.

### Cost distinction
Retrieval-only evaluation can run without Groq generation; judging requires hosted calls. A “mock” evaluation tests bookkeeping, not live model behavior. Reports should label which kind of evaluation produced a number.

<!-- PAGE 22 -->
# Datasets and Retrieval Metrics

`eval_questions.json` stores evaluation questions with fields such as ID, question text, expected source, expected keywords, reference answer, page, and question type. It contains multiple answerable sets and a group intended for refusal testing. The same file is consumed differently by different scripts, which filter to their own relevant questions.

### Primary retrieval hit
For the established metric, a question is a hit when one of the top-three retrieved chunks:
1. comes from the expected PDF, and
2. contains at least one of that question’s expected keywords.

This is a simple proxy for retrieving relevant evidence. It does not inspect the generated answer and does not prove that the correct page was found in every case.

### Secondary page hit
The reranker experiment also reports whether one of the top-three passages comes from the expected source and a page within one page of the cited expected page. It provides a page-proximity view distinct from the keyword-based primary hit.

### Refusal metrics
`trust.py` counts how often the fixed refusal appears for questions marked unanswerable and how often it appears on normal answerable questions. The mock fixture exercises only the counting/reporting logic; it must not be described as measured model performance.

### Sampling caution
Small sets make each question consequential. In a ten-question set, one answer changes the hit rate by ten percentage points. Good reporting should state set size, metric definition, and whether the set was used during method selection.

<!-- PAGE 23 -->
# Recorded Dense-versus-Hybrid Results

The stored results provide a useful example of why retrieval strategies should be measured rather than assumed:

| Question set | Dense | Hybrid |
|---|---:|---:|
| q1–q20 | 16/20 (80%) | 19/20 (95%) |
| q21–q30 | 10/10 (100%) | 10/10 (100%) |
| q31–q40, earlier hard set | 7/10 (70%) | 7/10 (70%) |

These are primary hit rates under the project’s expected-source-plus-keyword definition. The first set shows a three-question improvement for hybrid retrieval. On q21–q30, both methods reached the ceiling of this particular metric, so that set cannot distinguish them. On the q31–q40 hard held-out set, both methods had the same primary hit count, although the detailed stored report records which questions changed.

The reranker experiment’s broader saved results show the same pattern is not guaranteed to generalize: on q41–q50, dense retrieved 5/10 primary hits, hybrid 7/10. After reranking, dense was 4/10 and hybrid was 5/10. Thus the experiment does not support a claim that the cross-encoder improves the clean q41–q50 set.

### Reporting statement
These observations support using hybrid retrieval as the current user-facing strategy in this application, but do not prove that hybrid always wins. Results are specific to two original PDFs, a finite manually written question set, chosen chunking, and a coarse keyword metric.

<!-- PAGE 24 -->
# Cross-Encoder Re-Ranking Experiment

`rerank.py` evaluates the local `cross-encoder/ms-marco-MiniLM-L-6-v2` model. A cross-encoder receives a question and candidate passage together and scores their relevance. Unlike the embedding model, it can compare token interactions directly, but it is more expensive per candidate and cannot find passages that the initial retrieval stage never returned.

The experiment takes ten candidates from dense or hybrid retrieval, scores them with the cross-encoder, and returns the top three. It evaluates four methods: dense, hybrid, dense plus reranking, and hybrid plus reranking. It records primary keyword hit rate and secondary expected-page hit rate over q1–q20, q21–q30, q31–q40, and q41–q50.

On the only clean held-out set for selecting reranking, q41–q50, dense scored 5/10 and dense+rerank 4/10; hybrid scored 7/10 and hybrid+rerank 5/10. The results therefore show no measured benefit from this reranker on that set. On q31–q40, hybrid+rerank gained one primary hit versus hybrid, but that set had already been seen before reranker selection and is not a clean generalization result.

### Product architecture distinction
Reranking is an evaluation experiment, not part of the current Chat, Quiz, or Compare user-facing retrieval path. The app uses hybrid RRF. Experimental findings should not be confused with deployed behavior.

<!-- PAGE 25 -->
# Retrieval Miss Diagnostics

`diagnose_misses.py` is a read-only analysis script for selected difficult questions. It opens the historical 1500/50 index, reconstructs the same chunk corpus for BM25, and inspects dense, BM25, and hybrid rankings. The script prints the expected source and page, expected keywords, top passages, keyword matches, and the first rank at which the expected source/page appears.

It classifies misses into three diagnostic categories:

- **Retrieval failure:** no chunk from the expected source and page is in the top ten.
- **Ranking failure:** the expected chunk appears in the top ten but not the top three.
- **Keyword issue:** the expected page appears in the top three but expected keyword matching fails.

The distinction is practical. A ranking failure indicates the system found the target but placed it too low; a reranker might help. A retrieval failure suggests the target did not enter the candidate pool and points earlier in the pipeline. A keyword issue is a limitation of the evaluator’s proxy rather than necessarily a failure to retrieve useful content.

The diagnostic writes a Markdown output for inspection and does not change question definitions or index settings. Its labels depend on the exact expected-page metadata and keyword list, so they should be treated as explanations under the current diagnostic rule—not universal diagnoses of semantic correctness.

<!-- PAGE 26 -->
# Supporting Modules and Tests

The repository includes both user-facing and developer-facing entry points:

- `query.py` performs a simple dense search and prints cited chunks.
- `rag.py` demonstrates a command-line retrieval-plus-answer flow with the shared prompt.
- `ingest.py` builds the main application index.
- `evaluator.py` runs the chunking/retrieval evaluation and optional judge phase.
- `run_judging.py` resumes selected judge runs.
- `hybrid.py` implements lexical/dense fusion and its evaluation.
- `rerank.py` evaluates cross-encoder ordering.
- `diagnose_misses.py` explains selected retrieval misses.
- `agent.py` exercises a bounded self-reflective retrieval loop.

Tests use standard `unittest`, FastAPI’s `TestClient`, and mocked model calls. `test_compare.py` verifies separate paper retrieval, prompt separation, and citation behavior. `test_quiz.py` verifies structured question generation, evidence-bound grading, API validation, state progression, and score handling. `test_upload.py` verifies filename and PDF validation, metadata, persistence, duplicate protection, and indexing behavior.

### Why mocks matter
Mock callbacks make it possible to test prompt shape, result validation, session-state updates, and interface progression without spending API quota or relying on network availability. These tests establish application behavior for known fixture inputs; they do not measure actual LLM answer quality.

### Running focused tests

```powershell
.\venv\Scripts\python.exe -m unittest .\test_quiz.py .\test_compare.py .\test_upload.py -v
```

The tests are complementary to manual product demonstration, not a replacement for checking retrieval against a newly changed corpus.

<!-- PAGE 27 -->
# Data, Privacy, and Operational Boundaries

The project separates local document processing from hosted text generation. PDF parsing, chunking, local embeddings, Chroma persistence, and BM25 operate in the local environment. When a user submits a workflow that generates an answer, the application sends a prompt containing the question and retrieved passage text to the configured Groq model.

That boundary matters: “the index is local” does not mean “no document text ever leaves the machine.” Only retrieved context is intentionally placed in the LLM prompt, but this still constitutes transmission to a hosted provider. Users should follow their institution’s data-handling rules before uploading confidential, unpublished, copyrighted, or otherwise restricted documents.

### Secret handling
The API credential belongs in a local environment configuration and must never be embedded in source, reports, screenshots, or commits. The repository has an example environment file, not a reason to expose the live credential. This report does not include or inspect secret values.

### Local storage
The uploaded PDF is saved under the project’s `data/` directory. Chroma stores embeddings and document metadata locally. Evaluation artifacts include JSON files with question text and experiment outcomes. The project’s ignore rules exclude local secrets, virtual environments, and Chroma database directories from normal version control.

### Operational advice
Back up original PDFs separately. Keep evaluation and application indexes distinct. After manually changing the corpus, rebuild or synchronize the index; after using the uploader, the app adds the document to the main store and refreshes the BM25 cache.

<!-- PAGE 28 -->
# Limitations and Responsible Interpretation

**Retrieval limits.** The system retrieves only from the indexed chunks. A relevant passage may be split awkwardly, ranked below the top three, or absent from the candidate pool. Hybrid retrieval improves coverage on some recorded sets but does not eliminate misses.

**Generation limits.** Prompt instructions and citations help constrain the model, but the model can still misread evidence, omit caveats, hallucinate, or cite incorrectly. Citation validation is stronger for quiz question generation and grading than for unrestricted chat prose.

**Metric limits.** A keyword hit is a proxy, not a semantic human assessment. Page-hit proximity is also a proxy. LLM judges share model-family biases with answer generation. Tiny held-out groups have high per-question noise; in a ten-question set, one question is ten percentage points.

**Trust-label limits.** The trust meter is explicitly heuristic. Its similarity thresholds were inspected on the current question distribution and were not calibrated as probability estimates. Its support check is another model judgment, not an independent fact verifier.

**Corpus-change limits.** The autonomous agent and evaluation tools target fixed historical indexes. Adding a PDF through the UI updates the main app store, not all experiment indexes. Evaluation claims should be rerun on the corpus being discussed.

**Quiz grading limits.** Grading is constrained to retrieved text, but an LLM still decides whether an answer is correct or partial. It is suitable for practice and feedback, not official high-stakes scoring.

The strongest responsible claim is that the project provides an evidence-oriented workflow with inspectable sources and measurable retrieval experiments—not that it guarantees truth.

<!-- PAGE 29 -->
# Live Project Demonstration: Presenter Script

### Before class
1. Open a terminal in the project directory.
2. Start the interface:
   ```powershell
   .\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
3. Open `http://127.0.0.1:8000` in a browser.
4. Confirm the sidebar lists the available PDFs and the evaluation summary.
5. Use a valid local Groq configuration only if you intend to run live generation. Do not display the secret value.

### Demonstration sequence
**A. Show the library.** Point to the PDF list and uploader. Explain that uploaded PDFs are parsed, chunked, and indexed locally. Do not upload a file with a name already in the library.

**B. Demonstrate Chat.** Ask a factually answerable question about one of the papers. Show the answer citations, trust label, and retrieved-context expander. Explain that Chat uses dense-plus-BM25 retrieval fused by RRF.

**C. Demonstrate Quiz.** Enable Quiz Mode, choose a paper, and enter a covered topic. Generate five questions, answer them, submit once, and show the score, feedback, and cited source answers for partial or incorrect responses.

**D. Demonstrate Compare.** Enable Compare Mode, select two papers, ask a shared question, and show the separate paper answers and difference summary. Open the context expander to show evidence boundaries.

Live generation consumes hosted-model quota. For a quota-free rehearsal, demonstrate the interface and explain the mocked automated tests; do not claim a mock response is a live model result.

<!-- PAGE 30 -->
# Presenter Wrap-Up, Glossary, and References

### Closing statement
“This project turns research PDFs into a study workflow. It preserves source and page metadata, retrieves passages with local dense and lexical methods, and uses those passages for cited chat, quizzes, and paper comparisons. It also measures retrieval behavior and provides diagnostics so we can discuss both strengths and misses. The system is useful because it connects generated responses to inspectable evidence; it remains a study aid, not a guarantee of correctness.”

### Glossary
- **RAG:** Retrieval-Augmented Generation; retrieve evidence first, then generate from it.
- **Embedding:** Numeric representation of text used for semantic similarity.
- **Chroma:** Persistent vector database used to store and search embedded chunks.
- **BM25:** Lexical ranking method based on query-term matches and document statistics.
- **RRF:** Reciprocal Rank Fusion; merges ranked lists using rank positions.
- **Top-k:** The first k results returned by a retrieval or ranking step.
- **Cross-encoder:** A model that jointly scores a query and candidate passage.
- **Grounding:** Restricting an answer or judgment to provided evidence.
- **Browser session state:** JavaScript state and browser storage used to preserve the current interaction during a session.

### Source map
Configuration and prompts: `config.py`. PDF ingestion and splitting: `ingest.py`. FastAPI interface and upload routes: `main.py`. Browser interface: `static/index.html` and `static/app.js`. Hybrid retrieval: `hybrid.py`. Quiz logic: `quiz.py`. Compare logic: `compare.py`. Trust heuristic: `trust.py`. Autonomous loop: `agent.py`. Evaluation and diagnostics: `evaluator.py`, `run_judging.py`, `rerank.py`, and `diagnose_misses.py`.

**End of report.**
