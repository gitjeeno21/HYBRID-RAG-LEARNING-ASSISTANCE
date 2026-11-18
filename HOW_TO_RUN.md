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

**6. Start the app once and create your account** (see *Running the app* below). Open <http://127.0.0.1:8000>, choose **Create account**, and sign up with your email and a password of at least 10 characters. Then enter the 6-digit code sent to your email; until email is set up (see *Email setup*), the code is printed in the server window. Accounts live only on this computer, in `app.db`.

**7. Add your PDFs.** Either upload them in the app (**Add PDF**, or drag and drop), or import a whole folder at once. Stop the server first, then:

```powershell
python manage.py import-pdfs --email you@example.com data\*.pdf
```

Each account has its own private library in `storage\users\<id>\` and its own search index. Nobody else who signs in on this computer can see, search or delete your papers. The first import also downloads the embedding model (about 90 MB).

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
- **Inspect:** type any question to see how retrieval ranked every candidate passage (no AI call), and view the recorded retrieval benchmark. Under each chat answer, the search icon opens that question in the inspector, and **Pipeline trace** shows how long each step took and how many tokens it used.
- **Shortcuts:** `1` / `2` / `3` / `4` switch between Chat, Quiz, Compare and Inspect, and `/` jumps to the chat box.

## Email setup (verification and password-reset codes)

New accounts must verify their email with a 6-digit code, and **Forgot password?** sends a code too. The app sends these from the server through [EmailJS](https://www.emailjs.com), using the Gmail account you connect there.

Until EmailJS is set up, the app runs in **console mode**: codes are printed in the server window (look for `EMAIL NOT CONFIGURED … Code for …`) instead of being emailed. That's fine on your own computer, but never use console mode where other people sign up.

**One-time setup (about 10 minutes):**

1. Create a free account at <https://www.emailjs.com>.
2. **Email Services → Add New Service → Gmail.** Connect the Gmail account the codes should come from, and allow it to send email. Copy the **Service ID**.
3. **Email Templates → Create New Template.** Set it up like this:
   - **To Email:** `{{to_email}}`
   - **Subject:** `{{app_name}}: your code is {{otp_code}}`
   - **Content:**
     ```
     Hi {{to_name}},

     Use this code to {{purpose}}:

         {{otp_code}}

     It expires in {{expires_minutes}} minutes. If you didn't ask for it, you can ignore this email.
     ```
   Save it and copy the **Template ID**.
4. **Account → API keys:** copy the **Public Key** and the **Private Key**.
5. **Account → Security:** turn on **"Allow EmailJS API for non-browser applications"**. The app sends from the server, not the browser, so this is required.
6. Paste the four values into `.env` (never into code; `.env` is not committed):
   ```
   EMAILJS_SERVICE_ID=service_…
   EMAILJS_TEMPLATE_ID=template_…
   EMAILJS_PUBLIC_KEY=…
   EMAILJS_PRIVATE_KEY=…
   ```
7. Restart the server. The console-mode warning disappears, and codes arrive by email.

If an email doesn't arrive, the server window shows EmailJS's reason, for example a wrong template ID or the non-browser setting still turned off. EmailJS's free plan has a monthly email limit.

## Optional: retrieval switches

All off by default; each was measured before being left off (see `eval/RETRIEVAL_COMPARISON.md`). Set in PowerShell before starting the server, e.g. `$env:RAG_RERANK = "1"`.

| Variable | Effect |
| --- | --- |
| `RAG_TOP_K` | Passages sent to the model per search (default 5) |
| `RAG_RERANK=1` | Rerank the fused top 10 with a cross-encoder (~0.5 s per query on CPU) |
| `APP_SUPPORT_CHECK=1` | Re-enable the LLM support check and its High/Medium/Low label (off by default: it did not discriminate in calibration and doubles tokens per answer) |
| `RAG_RERANKER_MODEL` | Cross-encoder to use, e.g. `BAAI/bge-reranker-base` (~3 s per query on CPU, 1.1 GB download) |
| `RAG_CANDIDATES` | Candidates each retriever contributes before fusion (default 10) |
| `RAG_CROSS_PAGE=1` | Let chunks cross page breaks (then rebuild every library: stop the server, `python manage.py reindex --all`) |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `python` is not recognized | Reinstall Python with **"Add Python to PATH"** ticked, then open a new PowerShell window. |
| `No module named ...` | Activate the venv (`.\venv\Scripts\Activate.ps1`) and run `python -m pip install -r requirements.txt` again. |
| Sidebar says **GROQ_API_KEY missing in .env** | Check that `.env` is in the project folder and contains `GROQ_API_KEY=...` with your real key. Then restart the app. |
| Sidebar says **Index out of sync** | Stop the app, run `python manage.py reindex --email you@example.com`, then start it again. This happens if PDFs were added to or removed from `storage\users\<id>\pdfs\` by hand. |
| Forgot your password | Click **Forgot password?** on the sign-in page and enter the emailed code. In console mode, the code is printed in the server window. |
| Verification code never arrived | Wait a minute, then click **Send a new code**. Check spam. If it still fails, the server window shows why (see *Email setup*). |
| "Too many failed attempts" | Five wrong passwords lock sign-in for that email for 15 minutes. |
| "AI service is rate limited" | The free Groq tier has limits. Wait a minute and try again. |
| Port 8000 is already in use | Close the other app using it, or start on another port: `python -m uvicorn main:app --port 8001`, then open <http://127.0.0.1:8001>. |
| Warning about `HF_TOKEN` in the console | Harmless. You can ignore it. |

---

## Optional: run the tests

```powershell
python -m unittest discover -p "test_*.py"
```

The tests run offline and need no API key.
