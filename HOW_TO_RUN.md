# How to Run the Hybrid RAG Study Assistant

These steps are for Windows using PowerShell.

## What you need

- **Python 3.11.** Check by running `python --version`. If it's missing, install it from <https://www.python.org/downloads/> and tick **"Add Python to PATH"** during setup.
- **A free Groq API key** from <https://console.groq.com/keys>.
- **An internet connection.** You need it to install packages, for the one-time download of the embedding model (about 90 MB), and to generate answers.
- **About 2 GB of free disk space.**

---

## First-time setup (do this once)

**1. Open PowerShell in the project folder.**

```powershell
cd path\to\autonomous-rag-agent
```

**2. Create a virtual environment.**

```powershell
python -m venv venv
```

**3. Activate it.**

```powershell
.\venv\Scripts\Activate.ps1
```

When it works, `(venv)` appears at the start of the prompt.

> If PowerShell says *"running scripts is disabled on this system"*, run this once, then try step 3 again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**4. Install the packages.** This can take a few minutes.

```powershell
python -m pip install -r requirements.txt
```

**5. Add your API key.**

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder so the line reads `GROQ_API_KEY=your_real_key_here`, then save and close Notepad.

**6. Add your PDFs (optional).** Put PDF files in the `data\` folder. You can also add them later from inside the app.

**7. Build the search index.**

```powershell
python ingest.py
```

This reads every PDF in `data\` and builds the local index in `chroma_db\`. The first run also downloads the embedding model.

---

## Running the app (every time)

**1. Start the server.** Either:

- double-click **`run_api.bat`** in the project folder, or
- in PowerShell, from the project folder:

  ```powershell
  .\venv\Scripts\Activate.ps1
  python -m uvicorn main:app --host 127.0.0.1 --port 8000
  ```

**2. Wait for this line:**

```
Uvicorn running on http://127.0.0.1:8000
```

**3. Open <http://127.0.0.1:8000> in your browser.**

**4. Check the status lights** at the bottom-left of the sidebar:

| Light | Meaning |
| --- | --- |
| 🟢 Index ready | The search index is working |
| 🟢 AI service configured | Your API key was found |
| 🟡 / 🔴 | See *Troubleshooting* below |

**5. To stop the app,** click the PowerShell window and press **Ctrl + C**.

---

## Using the app

- **Chat:** type a question and press **Enter**. Use the chips at the top to search all papers or only some of them. Click a citation to see the passage it came from.
- **Quiz:** choose a paper, a difficulty and a question type, then click **Generate**. Answer every question and click **Grade answers**.
- **Compare:** pick two papers and a topic (or click a suggestion) to see how they differ.
- **Add a PDF:** click **Add PDF** in the sidebar, or drag a PDF anywhere onto the window. It's indexed straight away, so you don't need to run `ingest.py` again.
- **Shortcuts:** `1` / `2` / `3` switch between Chat, Quiz and Compare, and `/` jumps to the chat box.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `python` is not recognized | Reinstall Python with **"Add Python to PATH"** ticked, then open a new PowerShell window. |
| `No module named ...` | Activate the venv (`.\venv\Scripts\Activate.ps1`) and run `python -m pip install -r requirements.txt` again. |
| Sidebar says **GROQ_API_KEY missing in .env** | Check that `.env` is in the project folder and contains `GROQ_API_KEY=...` with your real key. Then restart the app. |
| Sidebar says **Index out of sync** | Stop the app, run `python ingest.py`, then start the app again. This usually happens after PDFs were added to or removed from `data\` by hand. |
| "AI service is rate limited" | The free Groq tier has limits. Wait a minute and try again. |
| Port 8000 is already in use | Close the other app using it, or start on another port: `python -m uvicorn main:app --port 8001`, then open <http://127.0.0.1:8001>. |
| Warning about `HF_TOKEN` in the console | Harmless. You can ignore it. |

---

## Optional: run the tests

```powershell
python -m unittest discover -p "test_*.py"
```

The tests run offline and need no API key.
