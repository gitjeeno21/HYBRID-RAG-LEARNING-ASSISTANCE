const state = { questions: [], pdf: "", topic: "", settings: {}, context: [], chatHistory: [] };

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (_) {
    showAppError("Network connection failed. Check that the app server is running.", () => location.reload());
    throw new Error("Network connection failed");
  }
  const data = await response.json().catch(() => ({}));
  if (response.status === 429) {
    showAppError("AI service rate limit or quota reached. Wait a moment and retry.", () => location.reload());
  } else if (response.status === 503) {
    showAppError("GROQ_API_KEY is missing or invalid. Add it to the .env file.", () => location.reload());
  } else if (!response.ok && /no pdf|empty|no passages/i.test(data.detail || "")) {
    showAppError("The index has no matching passages. Add or re-index a PDF.", () => loadPdfs());
  }
  if (!response.ok) throw new Error(formatApiError(data.detail || data.message || data, "Request failed"));
  return data;
}

function formatApiError(error, fallback = "Request failed") {
  if (Array.isArray(error)) {
    const messages = error.map(item => {
      if (typeof item === "string") return item;
      const field = Array.isArray(item?.loc) ? item.loc.filter(part => typeof part !== "number" && part !== "body").join(".") : "";
      const message = item?.msg || item?.message || item?.detail;
      return message ? `${field ? `${field}: ` : ""}${message}` : "";
    }).filter(Boolean);
    return messages.join("; ") || fallback;
  }
  if (error && typeof error === "object") return String(error.message || error.detail || fallback);
  return typeof error === "string" && error.trim() ? error : fallback;
}

let appRetryAction = null;
function showAppError(message, retryAction = null) {
  document.getElementById("app-error-message").textContent = message;
  document.getElementById("app-error").classList.remove("hidden");
  appRetryAction = retryAction;
}
document.getElementById("app-error-retry").addEventListener("click", () => {
  document.getElementById("app-error").classList.add("hidden");
  if (appRetryAction) appRetryAction();
});

function downloadText(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.classList.remove("hidden");
  setTimeout(() => element.classList.add("hidden"), 3000);
}

async function loadPdfs() {
  const { papers } = await api("/api/library");
  const pdfs = papers.map(paper => paper.filename);
  document.getElementById("pdf-list").innerHTML = papers.length
    ? papers.map(paper => `
      <div class="group rounded-lg border border-transparent px-2 py-2 text-zinc-400 transition hover:border-white/[.07] hover:bg-white/[.025]" data-paper="${escapeHtml(paper.filename)}">
        <button type="button" class="library-select w-full text-left">
          <div class="flex items-center gap-2"><span class="h-1.5 w-1.5 rounded-full ${paper.indexed ? "bg-emerald-400" : "bg-amber-400"}"></span><span class="truncate text-zinc-300">${escapeHtml(paper.filename)}</span></div>
          <div class="mt-1 text-[11px] text-zinc-600">${paper.pages} pages · ${paper.chunks} chunks · ${paper.indexed ? "Indexed" : "Pending"}</div>
        </button>
        <details class="relative mt-1"><summary class="cursor-pointer text-right text-xs text-zinc-500">⋮</summary>
          <div class="absolute right-0 z-20 w-28 rounded-lg border border-white/10 bg-zinc-950 p-1 text-xs">
            <button type="button" class="library-preview block w-full rounded px-2 py-1.5 text-left hover:bg-white/[.05]">Preview</button>
            <button type="button" class="library-rename block w-full rounded px-2 py-1.5 text-left hover:bg-white/[.05]">Rename</button>
            <button type="button" class="library-delete block w-full rounded px-2 py-1.5 text-left text-red-300 hover:bg-white/[.05]">Delete</button>
          </div>
        </details>
      </div>`).join("")
    : '<div class="text-zinc-600">No PDFs indexed</div>';
  document.getElementById("paper-count").textContent = pdfs.length;
  wireLibraryActions();
  ["quiz-pdf", "compare-pdf-one", "compare-pdf-two"].forEach(id => {
    const select = document.getElementById(id);
    select.innerHTML = pdfs.map(pdf => `<option value="${escapeHtml(pdf)}">${escapeHtml(pdf)}</option>`).join("");
  });
  const chatPapers = document.getElementById("chat-papers");
  chatPapers.innerHTML = pdfs.map(pdf => `<option value="${escapeHtml(pdf)}">${escapeHtml(pdf)}</option>`).join("");
  renderChatEmpty();
  configureComparePapers(pdfs);
}

function showLibraryDialog(html) {
  document.getElementById("library-dialog-content").innerHTML = html;
  document.getElementById("library-dialog").showModal();
}

function wireLibraryActions() {
  document.querySelectorAll("#pdf-list [data-paper]").forEach(row => {
    const filename = row.dataset.paper;
    row.querySelector(".library-select").addEventListener("click", () => {
      state.pdf = filename;
      const quizPdf = document.getElementById("quiz-pdf");
      quizPdf.value = filename;
      [...document.getElementById("chat-papers").options].forEach(option => {
        option.selected = option.value === filename;
      });
      toast(`${filename} selected for Chat and Quiz`);
    });
    row.querySelector(".library-preview").addEventListener("click", async () => {
      const data = await api(`/api/library/${encodeURIComponent(filename)}/preview`);
      showLibraryDialog(`<h3 class="text-lg font-semibold">${escapeHtml(filename)}</h3>
        <p class="mt-1 text-sm text-zinc-500">${data.pages} pages · ${escapeHtml(JSON.stringify(data.metadata))}</p>
        <pre class="mt-4 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-black/30 p-4 text-sm text-zinc-300">${escapeHtml(data.preview)}</pre>
        <button type="button" id="ask-library-paper" class="primary-button mt-4 rounded-xl px-4 py-2 text-sm text-white">Ask about this paper</button>`);
      document.getElementById("ask-library-paper").addEventListener("click", () => {
        document.getElementById("library-dialog").close();
        document.querySelector('[data-mode="chat"]').click();
        document.getElementById("chat-input").focus();
      });
    });
    row.querySelector(".library-rename").addEventListener("click", async () => {
      const current = filename.replace(/\.pdf$/i, "");
      const next = prompt("New PDF name", current);
      if (!next || next.trim() === current) return;
      try {
        await api("/api/library/rename", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ filename, new_filename: next.trim() })
        });
        toast("Paper renamed and re-indexed");
        await loadPdfs();
      } catch (error) { toast(error.message); }
    });
    row.querySelector(".library-delete").addEventListener("click", async () => {
      if (!confirm(`Delete "${filename}" and remove its indexed chunks?`)) return;
      try {
        await api(`/api/library/${encodeURIComponent(filename)}`, { method: "DELETE" });
        toast("Paper deleted");
        await loadPdfs();
      } catch (error) { toast(error.message); }
    });
  });
}

async function renderChatEmpty() {
  const container = document.getElementById("chat-messages");
  if (state.chatHistory.length) return;
  let starters = [
    "Summarize the main contribution of each paper.",
    "How do the methods differ?",
    "What limitations are reported?",
    "What datasets and metrics are used?",
  ];
  try {
    starters = (await api("/api/chat/starters")).starters || starters;
  } catch (_) {}
  container.innerHTML = `
    <div class="flex min-h-[55vh] flex-col items-center justify-center text-center">
      <h2 class="max-w-2xl text-2xl font-semibold text-white sm:text-3xl">Hybrid RAG for Evidence-Grounded Knowledge Discovery &amp; Learning</h2>
      <p class="mt-3 text-sm text-zinc-400">Explore your research library with answers grounded in retrieved passages.</p>
      <div class="mt-7 flex max-w-2xl flex-wrap justify-center gap-2">
        ${starters.map(starter => `<button type="button" class="starter-chip rounded-full border border-white/10 bg-white/[.025] px-3 py-2 text-xs text-zinc-300 hover:bg-white/[.06]">${escapeHtml(starter)}</button>`).join("")}
      </div>
    </div>`;
  container.querySelectorAll(".starter-chip").forEach(chip => chip.addEventListener("click", () => {
    document.getElementById("chat-input").value = chip.textContent.trim();
    document.getElementById("chat-input").focus();
  }));
}

function configureComparePapers(pdfs) {
  const first = document.getElementById("compare-pdf-one");
  const second = document.getElementById("compare-pdf-two");
  const warning = document.getElementById("compare-library-warning");
  if (pdfs.length < 2) {
    warning.classList.remove("hidden");
    first.disabled = true;
    second.disabled = true;
    document.getElementById("compare-button").disabled = true;
    return;
  }
  warning.classList.add("hidden");
  first.disabled = false;
  second.disabled = false;
  const options = [...second.options];
  options.forEach(option => { option.disabled = option.value === first.value; });
  if (!second.value || second.value === first.value) {
    second.value = pdfs.find(pdf => pdf !== first.value) || "";
  }
  updateCompareButton();
}

function updateCompareButton() {
  const first = document.getElementById("compare-pdf-one").value;
  const second = document.getElementById("compare-pdf-two").value;
  const valid = first && second && first !== second;
  document.getElementById("compare-button").disabled = !valid;
}

document.getElementById("compare-pdf-one").addEventListener("change", () => configureComparePapers(
  [...document.getElementById("compare-pdf-one").options].map(option => option.value)
));
document.getElementById("compare-pdf-two").addEventListener("change", updateCompareButton);

document.querySelectorAll(".compare-chip").forEach(chip => chip.addEventListener("click", () => {
  document.getElementById("compare-question").value = chip.textContent.trim();
}));
document.getElementById("compare-add-pdf").addEventListener("click", () => document.getElementById("pdf-upload").click());

document.querySelectorAll("[data-mode]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-mode]").forEach(item => item.classList.toggle("active", item === button));
    ["chat", "quiz", "compare"].forEach(mode => {
      document.getElementById(`${mode}-panel`).classList.toggle("hidden", mode !== button.dataset.mode);
    });
    const copy = {
      chat: ["Evidence-grounded chat", "Ask questions across your research library."],
      quiz: ["Study quiz", "Generate and grade questions from one selected paper."],
      compare: ["Paper comparison", "Compare two papers using isolated evidence."]
    }[button.dataset.mode];
    document.getElementById("mode-title").textContent = copy[0];
    document.getElementById("mode-description").textContent = copy[1];
  });
});

document.getElementById("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const query = input.value.trim();
  if (!query) return;
  const container = document.getElementById("chat-messages");
  if (!state.chatHistory.length) container.innerHTML = "";
  container.insertAdjacentHTML("beforeend", `<div class="flex justify-end"><div class="max-w-xl rounded-2xl border border-indigo-400/20 bg-indigo-500/[.16] px-4 py-3 text-zinc-100 shadow-lg shadow-indigo-950/20">${escapeHtml(query)}</div></div>`);
  container.insertAdjacentHTML("beforeend", '<div id="chat-loading" class="glass mr-auto max-w-3xl animate-pulse rounded-2xl px-5 py-4 text-sm text-zinc-500">Searching retrieved passages...</div>');
  input.value = "";
  const papers = [...document.getElementById("chat-papers").selectedOptions].map(option => option.value);
  try {
    const data = await api("/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ query, papers, history: state.chatHistory.slice(-6) })
    });
    document.getElementById("chat-loading")?.remove();
    const trust = data.trust || {};
    const colors = { High: "border-emerald-400/15 bg-emerald-400/[.08] text-emerald-300", Medium: "border-amber-400/15 bg-amber-400/[.08] text-amber-300", Low: "border-red-400/15 bg-red-400/[.08] text-red-300", "n/a": "border-white/[.08] bg-white/[.03] text-zinc-400" };
    const citations = [...new Map(data.citations.map(item => [item.citation, item])).values()]
      .map(item => `<button type="button" data-citation="${escapeHtml(item.citation)}" class="citation citation-button mr-1 rounded-md px-2 py-1 text-xs">${escapeHtml(item.citation)}</button>`).join("");
    const passagePanel = (data.context || []).map(item => `
      <details data-passage="${escapeHtml(item.citation)}" class="hidden mt-3 rounded-xl border border-white/[.08] bg-black/20 p-3">
        <summary class="cursor-pointer text-xs text-zinc-300">${escapeHtml(item.citation)}</summary>
        <p class="mt-2 text-sm text-zinc-400">${escapeHtml(item.text)}</p>
      </details>`).join("");
    container.insertAdjacentHTML("beforeend",
      `<div class="glass mr-auto max-w-3xl rounded-2xl px-5 py-4 text-zinc-200">
        <div class="whitespace-pre-wrap">${escapeHtml(data.answer)}</div>
        <div class="mt-4 flex flex-wrap items-center gap-2">${citations}
          <span title="Heuristic based on retrieval similarity and a support check, not a guarantee of correctness." class="${colors[trust.label] || colors["n/a"]} rounded-full border px-3 py-1 text-xs font-semibold">Trust: ${escapeHtml(trust.label || "n/a")}</span>
        </div>
        ${passagePanel}
        <div class="mt-4 flex gap-2 text-xs text-zinc-500">
          <button type="button" class="chat-copy hover:text-zinc-200">Copy</button>
          <button type="button" class="chat-regenerate hover:text-zinc-200">Regenerate</button>
          <button type="button" class="chat-feedback hover:text-zinc-200" data-value="up">👍</button>
          <button type="button" class="chat-feedback hover:text-zinc-200" data-value="down">👎</button>
        </div>
      </div>`);
    const answerElement = container.lastElementChild;
    answerElement.querySelectorAll(".citation-button").forEach(button => button.addEventListener("click", () => {
      const passage = [...answerElement.querySelectorAll("[data-passage]")]
        .find(details => details.dataset.passage === button.dataset.citation);
      if (passage) { passage.classList.remove("hidden"); passage.open = true; }
    }));
    answerElement.querySelector(".chat-copy").addEventListener("click", () => navigator.clipboard.writeText(data.answer));
    answerElement.querySelector(".chat-regenerate").addEventListener("click", () => {
      document.getElementById("chat-input").value = query;
      document.getElementById("chat-form").requestSubmit();
    });
    answerElement.querySelectorAll(".chat-feedback").forEach(button => button.addEventListener("click", async () => {
      await api("/api/chat/feedback", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ query, message: data.answer, feedback: button.dataset.value })
      }).catch(error => toast(error.message));
      toast("Feedback saved");
    }));
    state.chatHistory.push({ role: "user", content: query }, { role: "assistant", content: data.answer });
  } catch (error) {
    document.getElementById("chat-loading")?.remove();
    toast(error.message);
  }
});

document.getElementById("clear-chat").addEventListener("click", () => {
  state.chatHistory = [];
  renderChatEmpty();
});

function renderQuizHistory() {
  const history = JSON.parse(sessionStorage.getItem("quizHistory") || "[]");
  document.getElementById("quiz-history").innerHTML = history.length
    ? history.slice(0, 5).map(item => `
      <div class="flex justify-between gap-3 rounded-lg border border-white/[.06] px-3 py-2">
        <span>${escapeHtml(item.pdf)}</span><span>${escapeHtml(item.difficulty)} · ${item.score}%</span>
        <span class="text-zinc-600">${escapeHtml(item.date)}</span>
      </div>`).join("")
    : "No attempts yet.";
}

async function generateQuiz(questionsToRetry = null) {
  state.pdf = document.getElementById("quiz-pdf").value;
  state.topic = document.getElementById("quiz-topic").value.trim();
  state.settings = {
    difficulty: document.getElementById("quiz-difficulty").value,
    question_count: Number(document.getElementById("quiz-count").value),
    question_type: document.getElementById("quiz-type").value,
    page_range: document.getElementById("quiz-pages").value.trim() || null
  };
  if (!state.pdf) return toast("Select a PDF.");
  const button = document.getElementById("generate-quiz");
  button.disabled = true;
  button.textContent = "Generating...";
  document.getElementById("quiz-form").classList.add("hidden");
  try {
    const quiz = await api("/api/quiz/generate", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ pdf: state.pdf, topic: state.topic, ...state.settings })
    });
    state.questions = quiz.questions || [];
    state.context = quiz.context || [];
    if (!quiz.covered) return toast("The retrieved passages do not cover this topic.");
    const form = document.getElementById("quiz-form");
    form.innerHTML = state.questions.map((question, index) => `
      <div class="glass rounded-2xl p-5">
        <div class="mb-2 font-medium">${index + 1}. ${escapeHtml(question.question)}</div>
        <span class="mr-1 rounded-full border border-white/10 px-2 py-0.5 text-xs text-cyan-300">${escapeHtml(question.difficulty || state.settings.difficulty)}</span>
        <span class="citation rounded-md px-2 py-1 text-xs">${escapeHtml(question.source)}</span>
        ${question.options && Array.isArray(question.options)
          ? `<fieldset class="mt-3 space-y-2" data-answer-group="${index}">${question.options.map(option => `
              <label class="flex cursor-pointer items-start gap-3 rounded-xl border border-white/[.08] bg-black/20 px-3 py-2.5 text-sm text-zinc-300 transition hover:border-cyan-400/30">
                <input type="radio" name="quiz-answer-${index}" data-answer="${index}" value="${escapeHtml(option)}" class="mt-1 accent-cyan-400">
                <span class="min-w-0 break-words">${escapeHtml(option)}</span>
              </label>`).join("")}</fieldset>`
          : `<textarea data-answer="${index}" rows="2" class="field mt-3 w-full rounded-xl p-3" aria-label="Answer for question ${index + 1}"></textarea>`}
      </div>`).join("") + `<button class="primary-button rounded-xl px-4 py-2 text-white">Grade answers</button>`;
    form.classList.remove("hidden");
    document.getElementById("quiz-results").classList.add("hidden");
  } catch (error) { toast(error.message); }
  finally {
    button.disabled = false;
    button.textContent = `Generate ${state.settings.question_count || 5} questions`;
  }
}

document.getElementById("generate-quiz").addEventListener("click", () => generateQuiz());

document.getElementById("quiz-form").addEventListener("submit", async event => {
  event.preventDefault();
  const answers = state.questions.map((_, index) => {
    const selected = document.querySelector(`input[data-answer="${index}"]:checked`);
    if (selected) return selected.value;
    const text = document.querySelector(`textarea[data-answer="${index}"]`);
    return text ? text.value.trim() : "";
  });
  const unanswered = answers.map((answer, index) => answer ? null : index).filter(index => index !== null);
  if (unanswered.length) {
    unanswered.forEach(index => {
      const card = document.querySelector(`[data-answer-group="${index}"]`)?.closest(".glass")
        || document.querySelector(`textarea[data-answer="${index}"]`)?.closest(".glass");
      card?.classList.add("ring-2", "ring-amber-400/60");
    });
    toast(`Please answer all questions before grading (${unanswered.length} unanswered)`);
    return;
  }
  try {
    const result = await api("/api/quiz/grade", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        pdf: state.pdf, topic: state.topic, questions: state.questions, answers,
        page_range: state.settings.page_range
      })
    });
    const percentage = Math.round((result.score / result.maximum) * 100);
    const suggestion = percentage >= 80 ? "You are doing well—try Hard next." : percentage < 40 ? "Try an easier level to build the foundations." : "Medium is a good fit for your current level.";
    state.lastMissed = state.questions.filter((_, index) => result.results[index].grade !== "Correct");
    const results = document.getElementById("quiz-results");
    results.innerHTML = `
      <div class="glass rounded-2xl p-5">
        <div class="text-2xl font-semibold text-white">${result.score} / ${result.maximum} <span class="text-sm text-zinc-400">(${percentage}%)</span></div>
        <div class="mt-2 text-sm text-cyan-300">${escapeHtml(suggestion)}</div>
      </div>
      ${result.results.map((item, i) => `
        <div class="glass rounded-2xl p-5">
          <div><b>${i + 1}. ${escapeHtml(item.grade)}</b> — ${escapeHtml(item.feedback)}</div>
          <div class="mt-2 text-sm text-zinc-400">Correct answer: ${escapeHtml(item.correct_answer)}</div>
          <span class="citation mt-2 inline-block rounded-md px-2 py-1 text-xs">${escapeHtml(item.source)}</span>
          <details class="mt-3 text-sm text-zinc-400"><summary class="cursor-pointer text-zinc-300">Show source passage</summary>
            <div class="mt-2 rounded-xl border border-white/[.06] bg-black/20 p-3">${escapeHtml((state.context.find(source => source.citation === item.source) || {}).text || "Source passage unavailable.")}</div>
          </details>
        </div>`).join("")}
      <div class="flex flex-wrap gap-3">
        <button type="button" id="retry-missed" class="primary-button rounded-xl px-4 py-2 text-sm text-white">Retry only missed questions</button>
        <button type="button" id="new-same" class="rounded-xl border border-white/10 px-4 py-2 text-sm text-zinc-300">New quiz, same settings</button>
      </div>`;
    results.classList.remove("hidden");
    document.getElementById("retry-missed").addEventListener("click", () => {
      if (!state.lastMissed.length) return toast("Nothing missed—great work!");
      state.questions = state.lastMissed;
      generateQuiz(state.lastMissed);
    });
    document.getElementById("new-same").addEventListener("click", () => generateQuiz());
    const history = JSON.parse(sessionStorage.getItem("quizHistory") || "[]");
    history.unshift({ pdf: state.pdf, difficulty: state.settings.difficulty, score: percentage, date: new Date().toLocaleDateString() });
    sessionStorage.setItem("quizHistory", JSON.stringify(history.slice(0, 20)));
    renderQuizHistory();
    api("/api/progress/quiz", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        pdf: state.pdf,
        difficulty: state.settings.difficulty,
        score: percentage,
        topics: state.topic ? [state.topic] : []
      })
    }).catch(() => {});
  } catch (error) { toast(error.message); }
});

document.getElementById("compare-button").addEventListener("click", async () => {
  const pdfs = [
    document.getElementById("compare-pdf-one").value,
    document.getElementById("compare-pdf-two").value
  ];
  const question = document.getElementById("compare-question").value.trim();
  if (!question || pdfs[0] === pdfs[1]) return toast("Enter a question and select two different papers.");
  const button = document.getElementById("compare-button");
  button.disabled = true;
  button.textContent = "Comparing...";
  const result = document.getElementById("compare-result");
  try {
    const data = await api("/api/compare", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ pdfs, question })
    });
    result.classList.remove("hidden");
    const getSection = heading => {
      const escapedHeading = heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const normalizedSummary = data.summary
        .replace(/\*\*##/g, "##")
        .replace(/^(##[^\n]*?)\*\*\s*$/gm, "$1");
      const match = normalizedSummary.match(new RegExp(`##\\s*${escapedHeading}\\s*\n([\\s\\S]*?)(?=\n##\\s|$)`, "i"));
      return match ? match[1].trim() : "";
    };
    const names = Object.keys(data.papers);
    const paperA = names[0] || document.getElementById("compare-pdf-one").value;
    const paperB = names[1] || document.getElementById("compare-pdf-two").value;
    const paperColumn = (name, index) => `
      <section class="glass rounded-2xl p-4">
        <h3 class="mb-3 font-semibold text-white">Paper ${index === 0 ? "A" : "B"} · ${escapeHtml(name)}</h3>
        <div class="whitespace-pre-wrap text-sm text-zinc-300">${escapeHtml(getSection(name) || getSection(name.replace(/\.pdf$/i, "")) || data.summary || "Not found in this paper for this question.")}</div>
        <details class="mt-4 text-sm text-zinc-400"><summary class="cursor-pointer text-zinc-300">Retrieved context</summary>
          <div class="mt-2 space-y-2">${data.papers[name].length
            ? data.papers[name].map(item => `<div class="rounded-xl border border-white/[.06] bg-black/20 p-3"><span class="citation text-xs">${escapeHtml(item.citation)}</span><div class="mt-2">${escapeHtml(item.text)}</div></div>`).join("")
            : '<div class="text-zinc-500">No passages retrieved.</div>'}</div>
        </details>
      </section>`;
    const difference = getSection("Difference");
    const similarities = getSection("Similarities");
    result.innerHTML = `
      <div class="grid gap-4 md:grid-cols-2">${paperColumn(paperA, 0)}${paperColumn(paperB, 1)}</div>
      <section class="mt-5 rounded-2xl border border-white/[.08] bg-white/[.025] p-4">
        <h3 class="font-semibold text-white">Key differences</h3><p class="mt-2 text-sm text-zinc-300">${escapeHtml(difference)}</p>
        <h3 class="mt-4 font-semibold text-white">Similarities</h3><p class="mt-2 text-sm text-zinc-300">${escapeHtml(similarities)}</p>
      </section>`;
  } catch (error) {
    const message = /429|rate/i.test(error.message)
      ? "The AI service is rate limited. Wait a moment, then try again."
      : /key|auth/i.test(error.message)
        ? "The AI service key is missing or invalid. Add GROQ_API_KEY to .env."
        : /network|failed to fetch/i.test(error.message)
          ? "Network connection failed. Check your connection and try again."
          : "Comparison generation failed. Please try again.";
    result.innerHTML = `<div class="rounded-xl border border-rose-400/20 bg-rose-400/[.08] p-4 text-sm text-rose-200">${escapeHtml(message)} <button id="compare-retry" type="button" class="ml-2 font-semibold underline">Retry</button></div>`;
    document.getElementById("compare-retry").addEventListener("click", () => document.getElementById("compare-button").click());
  } finally {
    button.disabled = false;
    button.textContent = "Compare papers";
    updateCompareButton();
  }
});

document.getElementById("upload-button").addEventListener("click", () => document.getElementById("pdf-upload").click());
document.getElementById("library-dialog-close").addEventListener("click", () => document.getElementById("library-dialog").close());
const dropzone = document.getElementById("upload-dropzone");
dropzone.classList.remove("hidden");
["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventEventDefault?.();
  event.preventDefault();
  dropzone.classList.add("border-indigo-400/70");
}));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.remove("border-indigo-400/70");
}));
dropzone.addEventListener("drop", event => uploadPdf(event.dataTransfer.files[0]));
document.getElementById("pdf-upload").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  await uploadPdf(file);
  event.target.value = "";
});

async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const progress = document.getElementById("upload-progress");
  try {
    progress.textContent = "Parsing → Chunking → Embedding → Indexed";
    const data = await api("/api/upload", { method: "POST", body: form });
    toast(`${data.filename}: ${data.chunks_indexed} chunks indexed`);
    progress.textContent = "Parsing ✓ → Chunking ✓ → Embedding ✓ → Indexed ✓";
    await loadPdfs();
  } catch (error) { toast(error.message); }
}

document.getElementById("chat-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close());
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebar-scrim").classList.remove("open");
  }
});
document.getElementById("menu-button").addEventListener("click", () => {
  document.getElementById("sidebar").classList.add("open");
  document.getElementById("sidebar-scrim").classList.add("open");
});
document.getElementById("sidebar-scrim").addEventListener("click", () => {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("sidebar-scrim").classList.remove("open");
});

document.getElementById("settings-button").addEventListener("click", async () => {
  const settings = await api("/api/settings");
  document.getElementById("setting-top-k").value = settings.top_k;
  document.getElementById("setting-model").value = settings.model;
  document.getElementById("setting-theme").checked = localStorage.getItem("theme") === "light";
  document.getElementById("setting-context").checked = localStorage.getItem("showContext") === "true";
  document.getElementById("settings-dialog").showModal();
});
document.getElementById("settings-close").addEventListener("click", () => {
  localStorage.setItem("showContext", document.getElementById("setting-context").checked);
  const light = document.getElementById("setting-theme").checked;
  localStorage.setItem("theme", light ? "light" : "dark");
  document.body.classList.toggle("light", light);
  document.getElementById("settings-dialog").close();
});
if (localStorage.getItem("theme") === "light") document.body.classList.add("light");

document.getElementById("dismiss-hint").addEventListener("click", () => {
  localStorage.setItem("firstRunHintDismissed", "true");
  document.getElementById("first-run-hint").classList.add("hidden");
});
if (localStorage.getItem("firstRunHintDismissed") === "true") {
  document.getElementById("first-run-hint").classList.add("hidden");
}

document.getElementById("export-chat").addEventListener("click", () => {
  const markdown = state.chatHistory.map(item =>
    item.role === "user"     ? `## You\n${item.content}` : `## Hybrid RAG for Evidence-Grounded Knowledge Discovery & Learning\n${item.content}`
  ).join("\n\n");
  downloadText("chat-export.md", `# Chat export\n\n${markdown}\n`, "text/markdown");
});

document.getElementById("progress-button").addEventListener("click", async () => {
  const progress = await api("/api/progress");
  document.getElementById("progress-content").innerHTML = `
    <p><strong>${progress.quizzes_taken}</strong> quizzes taken</p>
    <h3 class="mt-4 font-semibold text-white">Average score by difficulty</h3>
    <p>${Object.entries(progress.averages).map(([level, score]) => `${escapeHtml(level)}: ${score}%`).join("<br>") || "No completed quizzes yet."}</p>
    <h3 class="mt-4 font-semibold text-white">Most-missed topics</h3>
    <p>${progress.most_missed_topics.map(escapeHtml).join(", ") || "None yet."}</p>`;
  document.getElementById("progress-dialog").showModal();
});
document.getElementById("progress-close").addEventListener("click", () => document.getElementById("progress-dialog").close());

document.getElementById("export-quiz").addEventListener("click", () => {
  if (!state.questions.length) return toast("Generate a quiz before exporting.");
  const markdown = state.questions.map((question, index) =>
    `## ${index + 1}. ${question.question}\n- Difficulty: ${question.difficulty || state.settings.difficulty}\n- Source: ${question.source}\n`
  ).join("\n");
  downloadText("quiz-results.md", `# Quiz\n\n${markdown}`, "text/markdown");
});

function downloadQuizPdf() {
  if (!state.questions.length) return toast("Generate a quiz before exporting.");
  const lines = ["Quiz results"].concat(state.questions.map((q, i) => `${i + 1}. ${q.question}`));
  const escaped = lines.join("\n").replace(/[^\x20-\x7E]/g, "").replaceAll("\\", "\\\\").replaceAll("(", "\\(").replaceAll(")", "\\)");
  const stream = `BT /F1 11 Tf 50 790 Td 14 TL (${escaped.replaceAll("\n", ") Tj T* (")}) Tj ET`;
  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj",
    "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    `5 0 obj\n<< /Length ${stream.length} >>\nstream\n${stream}\nendstream\nendobj`,
  ];
  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += object + "\n";
  });
  const xref = pdf.length;
  pdf += `xref\n0 6\n0000000000 65535 f \n${offsets.slice(1).map(offset => `${String(offset).padStart(10, "0")} 00000 n \n`).join("")}trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
  downloadText("quiz-results.pdf", pdf, "application/pdf");
}
document.getElementById("export-quiz-pdf").addEventListener("click", downloadQuizPdf);

renderQuizHistory();
loadPdfs().catch(error => showAppError("Could not load the paper library. Check the server connection.", () => loadPdfs()));
