# Hybrid RAG Study Assistant

A local study workspace for research PDFs. Ask questions and get answers that cite their pages, quiz yourself on a paper, and compare two papers side by side. Each answer is built only from passages retrieved from your own library.

- **Chat:** answers use hybrid retrieval (BM25 plus dense embeddings, fused with reciprocal-rank fusion). Each answer comes with clickable page citations, the retrieved passages, and a confidence label. Follow-up questions such as "how does *it* compare…" are rewritten into standalone searches. Answers stream in as they are written.
- **Quiz:** questions are generated from one paper, with a choice of difficulty, question type, count and page range. The same retrieved passages are used to grade your answers. You can retry missed questions and track your progress over time.
- **Compare:** two papers are compared using separate evidence for each, so passages from one paper can't leak into the other's answer.
- **Library:** add PDFs (with drag and drop), then preview, open, rename or delete them. Indexing happens straight away.

Embeddings (`all-MiniLM-L6-v2`) and the Chroma index run locally on CPU. Only answer generation calls the hosted LLM (Groq, `openai/gpt-oss-120b`).

## Quick start (Windows / PowerShell)

For detailed steps and troubleshooting, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

```powershell
cd autonomous-rag-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env      # then put your key in .env: GROQ_API_KEY=...
python ingest.py                 # optional: rebuild the index from data/*.pdf
.\run_api.bat                    # or: python -m uvicorn main:app --port 8000
```

Then open <http://127.0.0.1:8000>. You can get a free Groq key at <https://console.groq.com/keys>.

The sidebar footer shows whether the index is in sync and whether an API key is set. If it says **Index out of sync**, run `python ingest.py`.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `1` / `2` / `3` | Switch to Chat / Quiz / Compare |
| `/` | Focus the chat box |
| `Enter` / `Shift+Enter` | Send / new line |
| `Esc` | Close menus and dialogs |

## Project layout

| Path | Purpose |
| --- | --- |
| `main.py` | FastAPI app: REST API and static frontend |
| `static/` | Frontend (`index.html`, `style.css`, `app.js`), with no build step and no CDN |
| `config.py` | Paths, model names, prompt, citation helpers |
| `ingest.py` | PDF loading, chunking, full index rebuild |
| `hybrid.py` | BM25 + dense retrieval with RRF (`python hybrid.py` runs its evaluation) |
| `trust.py` | Confidence heuristic: retrieval strength + LLM support check |
| `quiz.py`, `compare.py` | Quiz generation/grading and two-paper comparison |
| `upload.py`, `library.py` | Adding, renaming and removing PDFs safely |
| `evaluator.py`, `run_judging.py`, `rerank.py`, `diagnose_misses.py`, `agent.py` | Offline evaluation and research scripts |
| `eval_questions.json`, `*_results.json` | Evaluation set and recorded results |

## Evaluation

Chunking was chosen by a grid search (`evaluator.py`). The best configuration was **1500 characters with 50 overlap**: dense retrieval found the right passage for 80% of questions, and an LLM judge scored the answers 4.15 out of 5. Hybrid retrieval raised that hit rate from **80% to 95%** on the same 20 questions (`hybrid_results.json`). `ingest.py` reads the winning settings from `eval_results.json` automatically.

## Tests

```powershell
python -m unittest discover -p "test_*.py"
```

The tests run offline. LLM calls, the vector store and the PDF loader are mocked.

## Data files created at runtime (git-ignored)

`chroma_db/` (vector index), `quiz_history.json` (quiz progress), and `chat_feedback.json` (👍/👎 ratings).

> Study aid only. Answers are grounded in retrieved passages but are not guaranteed to be correct.
