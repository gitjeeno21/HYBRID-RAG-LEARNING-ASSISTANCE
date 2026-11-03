"use strict";

/* ==========================================================================
   State and small helpers
   ========================================================================== */
const state = {
  mode: "chat",
  papers: [],            // [{filename, pages, chunks, indexed, size_kb}]
  scope: new Set(),      // chat paper scope; empty = all papers
  turns: [],             // [{id, query, status: "loading"|"done"|"error", data, error, feedback}]
  busy: false,
  quiz: null,            // active quiz, see generateQuiz()
  settings: { topK: null, maxTopK: 10, showContext: false, theme: "system" },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

const store = {
  get(key, fallback = null) {
    try { const value = localStorage.getItem(key); return value === null ? fallback : value; } catch (_) { return fallback; }
  },
  set(key, value) { try { localStorage.setItem(key, value); } catch (_) {} },
  session(key, value) {
    try {
      if (value === undefined) return JSON.parse(sessionStorage.getItem(key) || "null");
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch (_) { return null; }
  },
};

async function api(path, { method = "GET", json, body } = {}) {
  const options = { method };
  if (json !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(json);
  } else if (body) {
    options.body = body;
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch (_) {
    throw new Error("Network connection failed. Check that the app server is running.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 503 || response.status === 429) refreshHealth();
    throw new Error(formatApiError(data.detail || data.message || data, `Request failed (${response.status})`));
  }
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

function toast(message, type = "info", action = null) {
  const element = document.createElement("div");
  element.className = "toast";
  element.setAttribute("role", type === "error" ? "alert" : "status");
  const dot = { success: "ok", error: "bad", warn: "warn" }[type] || "";
  element.innerHTML = `<span class="dot ${dot}"></span><span>${escapeHtml(message)}</span>`;
  if (action) {
    const button = document.createElement("button");
    button.className = "btn btn-sm";
    button.textContent = action.label;
    button.addEventListener("click", () => { element.remove(); action.run(); });
    element.append(button);
  }
  $("#toasts").append(element);
  setTimeout(() => element.remove(), type === "error" ? 6500 : 3500);
}

function downloadText(filename, content, type = "text/markdown") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span><span>${escapeHtml(label)}</span>`;
    button.disabled = true;
  } else {
    if (button.dataset.label) button.innerHTML = button.dataset.label;
    button.disabled = false;
  }
}

function relativeDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`;
  return date.toLocaleDateString();
}

const paperTitle = filename => filename.replace(/\.pdf$/i, "");

/* ---------- Dialog helpers ---------- */
function openDialog(id) { const dialog = document.getElementById(id); if (!dialog.open) dialog.showModal(); return dialog; }
document.addEventListener("click", event => {
  const close = event.target.closest("[data-close]");
  if (close) close.closest("dialog")?.close();
});
$$("dialog").forEach(dialog => dialog.addEventListener("click", event => {
  if (event.target === dialog) dialog.close(); // click on backdrop
}));

function confirmDialog({ title, message, okLabel = "OK", danger = false, input = null }) {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-message").textContent = message;
  const ok = $("#confirm-ok");
  ok.textContent = okLabel;
  ok.className = `btn ${danger ? "btn-danger" : "btn-primary"}`;
  const field = $("#confirm-input");
  field.classList.toggle("hidden", input === null);
  field.value = input ?? "";
  dialog.returnValue = "";
  dialog.showModal();
  if (input !== null) { field.focus(); field.select(); }
  field.onkeydown = event => { if (event.key === "Enter") { event.preventDefault(); dialog.close("ok"); } };
  return new Promise(resolve => {
    dialog.addEventListener("close", () => {
      const confirmed = dialog.returnValue === "ok";
      resolve(input === null ? confirmed : (confirmed ? field.value.trim() : null));
    }, { once: true });
  });
}

/* ==========================================================================
   Markdown (small, safe: everything is escaped before formatting)
   ========================================================================== */
function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    // Citations appear as "(a.pdf, p. 1; p. 4)" or with full-width 【】 brackets.
    .replace(/[(【]([^()【】]*?\.pdf,\s*p\.[^()【】]*?)[)】]/gi, (_, inner) => citeButtons(inner));
}

function citeButtons(inner) {
  let lastFile = "";
  return inner.split(/\s*;\s*/).filter(Boolean).map(part => {
    let label = part.trim();
    const file = label.match(/^(.*?\.pdf),/i);
    if (file) lastFile = file[1];
    else if (/^p+\.\s*/i.test(label) && lastFile) label = `${lastFile}, ${label}`;
    return `<button type="button" class="cite" data-cite="${label}">${label}</button>`;
  }).join("");
}

function renderMarkdown(source) {
  const lines = escapeHtml(source || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let list = null;
  let paragraph = [];
  let code = null;
  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${inlineMarkdown(paragraph.join("<br>"))}</p>`);
    paragraph = [];
  };
  const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };
  for (const line of lines) {
    if (code !== null) {
      if (/^```/.test(line)) { html.push(`<pre><code>${code.join("\n")}</code></pre>`); code = null; }
      else code.push(line);
      continue;
    }
    if (/^```/.test(line)) { flushParagraph(); closeList(); code = []; continue; }
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (heading) { flushParagraph(); closeList(); html.push(`<h4>${inlineMarkdown(heading[1])}</h4>`); }
    else if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (list !== type) { closeList(); html.push(`<${type}>`); list = type; }
      html.push(`<li>${inlineMarkdown((bullet || numbered)[1])}</li>`);
    } else if (!line.trim()) { flushParagraph(); closeList(); }
    else { closeList(); paragraph.push(line); }
  }
  if (code !== null) html.push(`<pre><code>${code.join("\n")}</code></pre>`);
  flushParagraph();
  closeList();
  return html.join("");
}

/** Drop the model's trailing "Sources: ..." line; sources are shown as cards. */
function stripSourcesLine(answer) {
  return String(answer || "").replace(/\n*\s*\**Sources?\**:\s*[^\n]*\s*$/i, "").trim();
}

/* ==========================================================================
   Health / status
   ========================================================================== */
async function refreshHealth() {
  try {
    const health = await api("/api/health");
    const indexDot = $("#index-dot");
    const llmDot = $("#llm-dot");
    if (health.index_error) {
      indexDot.className = "dot bad";
      $("#index-status").textContent = "Index out of sync. Run python ingest.py";
      $("#index-status").title = health.index_error;
    } else if (!health.papers) {
      indexDot.className = "dot warn";
      $("#index-status").textContent = "No papers yet";
    } else {
      indexDot.className = "dot ok";
      $("#index-status").textContent = `Index ready · ${health.chunks} passages`;
    }
    llmDot.className = `dot ${health.llm_configured ? "ok" : "warn"}`;
    $("#llm-status").textContent = health.llm_configured ? "AI service configured" : "GROQ_API_KEY missing in .env";
  } catch (_) {
    $("#index-dot").className = "dot bad";
    $("#index-status").textContent = "Server unreachable";
    $("#llm-dot").className = "dot";
    $("#llm-status").textContent = "";
  }
}

/* ==========================================================================
   Library
   ========================================================================== */
async function loadLibrary() {
  try {
    const { papers } = await api("/api/library");
    state.papers = papers;
  } catch (error) {
    $("#pdf-list").innerHTML = `<div class="notice notice-error">${escapeHtml(error.message)}</div>`;
    return;
  }
  const names = state.papers.map(paper => paper.filename);
  state.scope = new Set([...state.scope].filter(name => names.includes(name)));
  renderLibrary();
  renderScope();
  fillPaperSelects();
  if (!state.turns.length) renderChatEmpty();
}

function renderLibrary() {
  $("#paper-count").textContent = state.papers.length;
  $("#pdf-list").innerHTML = state.papers.length
    ? state.papers.map(paper => `
      <div class="paper" data-paper="${escapeHtml(paper.filename)}">
        <div class="paper-icon">PDF</div>
        <div class="paper-body">
          <div class="paper-name" title="${escapeHtml(paper.filename)}">${escapeHtml(paperTitle(paper.filename))}</div>
          <div class="paper-meta">${paper.pages} pages · ${paper.chunks} passages${paper.indexed ? "" : " · not indexed"}</div>
        </div>
        <button type="button" class="icon-btn paper-menu" aria-label="Actions for ${escapeHtml(paper.filename)}" aria-expanded="false">${icon("dots")}</button>
      </div>`).join("")
    : '<div class="empty-library">No papers yet.<br>Add a PDF to get started.</div>';
}

function closeMenus() {
  $$(".menu").forEach(menu => menu.remove());
  $$(".paper-menu[aria-expanded='true']").forEach(button => button.setAttribute("aria-expanded", "false"));
}
document.addEventListener("click", event => {
  if (!event.target.closest(".menu") && !event.target.closest(".paper-menu")) closeMenus();
});

$("#pdf-list").addEventListener("click", event => {
  const button = event.target.closest(".paper-menu");
  if (!button) return;
  const row = button.closest(".paper");
  const open = button.getAttribute("aria-expanded") === "true";
  closeMenus();
  if (open) return;
  button.setAttribute("aria-expanded", "true");
  const filename = row.dataset.paper;
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.setAttribute("role", "menu");
  menu.innerHTML = `
    <button type="button" data-action="ask" role="menuitem">${icon("chat")}Ask about this paper</button>
    <button type="button" data-action="quiz" role="menuitem">${icon("quiz")}Quiz me on it</button>
    <button type="button" data-action="preview" role="menuitem">${icon("eye")}Preview</button>
    <button type="button" data-action="open" role="menuitem">${icon("open")}Open PDF</button>
    <button type="button" data-action="rename" role="menuitem">${icon("edit")}Rename</button>
    <button type="button" data-action="delete" class="danger" role="menuitem">${icon("trash")}Delete</button>`;
  row.append(menu);
  menu.querySelector("button").focus();
  menu.addEventListener("click", item => {
    const action = item.target.closest("[data-action]")?.dataset.action;
    if (!action) return;
    closeMenus();
    paperAction(action, filename);
  });
});

function paperAction(action, filename) {
  closeSidebar();
  if (action === "ask") {
    state.scope = new Set([filename]);
    renderScope();
    setMode("chat");
    $("#chat-input").focus();
  } else if (action === "quiz") {
    $("#quiz-pdf").value = filename;
    setMode("quiz");
  } else if (action === "open") {
    window.open(`/api/library/${encodeURIComponent(filename)}/file`, "_blank", "noopener");
  } else if (action === "preview") {
    previewPaper(filename);
  } else if (action === "rename") {
    renamePaper(filename);
  } else if (action === "delete") {
    deletePaper(filename);
  }
}

async function previewPaper(filename) {
  $("#library-dialog-title").textContent = paperTitle(filename);
  $("#library-dialog-content").innerHTML = '<div class="skeleton" style="height:18px;width:40%"></div><div class="skeleton" style="height:200px;margin-top:14px"></div>';
  $("#library-dialog-foot").innerHTML = "";
  openDialog("library-dialog");
  try {
    const data = await api(`/api/library/${encodeURIComponent(filename)}/preview`);
    const meta = Object.entries(data.metadata || {})
      .map(([key, value]) => `<span class="badge badge-neutral">${escapeHtml(key)}: ${escapeHtml(String(value).slice(0, 60))}</span>`).join(" ");
    $("#library-dialog-content").innerHTML = `
      <div class="row"><span class="badge badge-accent">${data.pages} pages</span>${meta}</div>
      <div class="hint" style="margin-top:12px">First page</div>
      <div class="preview-text">${escapeHtml(data.preview || "No extractable text on the first page.")}</div>`;
    $("#library-dialog-foot").innerHTML = `
      <button type="button" class="btn" data-preview="open">${icon("open")}Open PDF</button>
      <button type="button" class="btn" data-preview="quiz">${icon("quiz")}Quiz me</button>
      <button type="button" class="btn btn-primary" data-preview="ask">${icon("chat")}Ask about this paper</button>`;
    $$("[data-preview]", $("#library-dialog-foot")).forEach(button => button.addEventListener("click", () => {
      $("#library-dialog").close();
      paperAction(button.dataset.preview, filename);
    }));
  } catch (error) {
    $("#library-dialog-content").innerHTML = `<div class="notice notice-error">${escapeHtml(error.message)}</div>`;
  }
}

async function renamePaper(filename) {
  const next = await confirmDialog({
    title: "Rename paper", message: "Enter a new file name. The paper is re-indexed under the new name.",
    okLabel: "Rename", input: paperTitle(filename),
  });
  if (!next || next === paperTitle(filename)) return;
  try {
    const { filename: renamed } = await api("/api/library/rename", { method: "POST", json: { filename, new_filename: next } });
    if (state.scope.delete(filename)) state.scope.add(renamed);
    toast(`Renamed to ${renamed}`, "success");
    await loadLibrary();
  } catch (error) { toast(error.message, "error"); }
}

async function deletePaper(filename) {
  const confirmed = await confirmDialog({
    title: "Delete paper?", message: `"${filename}" and all of its indexed passages will be removed. This cannot be undone.`,
    okLabel: "Delete", danger: true,
  });
  if (!confirmed) return;
  try {
    await api(`/api/library/${encodeURIComponent(filename)}`, { method: "DELETE" });
    toast("Paper deleted", "success");
    await loadLibrary();
    refreshHealth();
  } catch (error) { toast(error.message, "error"); }
}

/* ---------- Upload ---------- */
async function uploadFiles(fileList) {
  const files = [...fileList].filter(file => /\.pdf$/i.test(file.name) || file.type === "application/pdf");
  if (!files.length) return toast("Only PDF files can be added.", "warn");
  const status = $("#upload-status");
  const button = $("#upload-button");
  button.disabled = true;
  status.classList.remove("hidden");
  let added = 0;
  for (const [index, file] of files.entries()) {
    status.innerHTML = `Indexing <strong>${escapeHtml(file.name)}</strong>${files.length > 1 ? ` (${index + 1}/${files.length})` : ""}…
      <div class="bar"><span></span></div>`;
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await api("/api/upload", { method: "POST", body: form });
      added += 1;
      toast(`${data.filename}: ${data.chunks_indexed} passages indexed`, "success");
    } catch (error) {
      toast(`${file.name}: ${error.message}`, "error");
    }
  }
  status.classList.add("hidden");
  button.disabled = false;
  if (added) { await loadLibrary(); refreshHealth(); }
}

$("#upload-button").addEventListener("click", () => $("#pdf-upload").click());
$("#compare-add-pdf").addEventListener("click", () => $("#pdf-upload").click());
$("#pdf-upload").addEventListener("change", async event => {
  await uploadFiles(event.target.files);
  event.target.value = "";
});

let dragDepth = 0;
const hasFiles = event => [...(event.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", event => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth += 1;
  $("#drop-overlay").classList.remove("hidden");
});
window.addEventListener("dragover", event => { if (hasFiles(event)) event.preventDefault(); });
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("#drop-overlay").classList.add("hidden");
});
window.addEventListener("drop", event => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  $("#drop-overlay").classList.add("hidden");
  uploadFiles(event.dataTransfer.files);
});

function fillPaperSelects() {
  const names = state.papers.map(paper => paper.filename);
  const options = names.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(paperTitle(name))}</option>`).join("");
  ["quiz-pdf", "compare-pdf-one", "compare-pdf-two"].forEach(id => {
    const select = document.getElementById(id);
    const previous = select.value;
    select.innerHTML = options || '<option value="">No papers yet</option>';
    if (names.includes(previous)) select.value = previous;
  });
  const second = $("#compare-pdf-two");
  if (second.value === $("#compare-pdf-one").value && names.length > 1) {
    second.value = names.find(name => name !== $("#compare-pdf-one").value);
  }
  $("#generate-quiz").disabled = !names.length;
  updateCompareControls();
}

/* ==========================================================================
   Modes
   ========================================================================== */
const MODE_COPY = {
  chat: ["Chat", "Ask questions answered only from your papers, with citations."],
  quiz: ["Quiz", "Test yourself with questions generated and graded from one paper."],
  compare: ["Compare", "Compare two papers side by side, using separate evidence for each."],
};

function setMode(mode) {
  state.mode = mode;
  $$("[data-mode]").forEach(item => {
    item.classList.toggle("active", item.dataset.mode === mode);
    item.setAttribute("aria-current", item.dataset.mode === mode ? "page" : "false");
  });
  Object.keys(MODE_COPY).forEach(name => $(`#${name}-panel`).classList.toggle("hidden", name !== mode));
  $("#mode-title").textContent = MODE_COPY[mode][0];
  $("#mode-description").textContent = MODE_COPY[mode][1];
  store.set("mode", mode);
  closeSidebar();
  if (mode === "quiz") loadQuizHistory();
}
$$("[data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#sidebar-scrim").classList.remove("open");
}
$("#menu-button").addEventListener("click", () => {
  $("#sidebar").classList.add("open");
  $("#sidebar-scrim").classList.add("open");
});
$("#sidebar-scrim").addEventListener("click", closeSidebar);

/* ==========================================================================
   Chat
   ========================================================================== */
function renderScope() {
  const all = !state.scope.size;
  $("#chat-scope").innerHTML =
    `<button type="button" class="chip" data-scope="" aria-pressed="${all}">All papers</button>` +
    state.papers.map(paper => `<button type="button" class="chip" data-scope="${escapeHtml(paper.filename)}"
      aria-pressed="${state.scope.has(paper.filename)}" title="${escapeHtml(paper.filename)}">${escapeHtml(paperTitle(paper.filename))}</button>`).join("");
}
$("#chat-scope").addEventListener("click", event => {
  const chip = event.target.closest("[data-scope]");
  if (!chip) return;
  const name = chip.dataset.scope;
  if (!name) state.scope.clear();
  else if (state.scope.has(name)) state.scope.delete(name);
  else state.scope.add(name);
  renderScope();
});

async function renderChatEmpty() {
  const container = $("#chat-messages");
  if (state.turns.length) return;
  if (!state.papers.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="orb">${icon("upload")}</div>
        <h2>Add your first paper</h2>
        <p>Upload a PDF and ask questions about it. Every answer cites the pages it came from.</p>
        <button type="button" class="btn btn-primary" style="margin-top:20px" id="empty-upload">${icon("upload")}Add PDF</button>
      </div>`;
    $("#empty-upload").addEventListener("click", () => $("#pdf-upload").click());
    return;
  }
  let starters = ["Summarize the main contribution of each paper.", "What limitations are discussed?"];
  try { starters = (await api("/api/chat/starters")).starters || starters; } catch (_) {}
  if (state.turns.length) return;
  container.innerHTML = `
    <div class="empty-state">
      <div class="orb">${icon("sparkle")}</div>
      <h2>What would you like to learn?</h2>
      <p>Ask anything about your ${state.papers.length} paper${state.papers.length === 1 ? "" : "s"}. Answers come only from retrieved passages and include page citations.</p>
      <div class="starters">${starters.map(starter => `<button type="button" class="starter">${escapeHtml(starter)}</button>`).join("")}</div>
    </div>`;
  $$(".starter", container).forEach(button => button.addEventListener("click", () => sendChat(button.textContent.trim())));
}

const historyBefore = turn => state.turns
  .slice(0, state.turns.indexOf(turn))
  .filter(item => item.status === "done")
  .flatMap(item => [{ role: "user", content: item.query }, { role: "assistant", content: item.data.answer }])
  .slice(-6);

function persistChat() {
  store.session("chatTurns", state.turns.filter(turn => turn.status === "done").slice(-30));
  $("#export-chat").disabled = !state.turns.some(turn => turn.status === "done");
  $("#clear-chat").disabled = !state.turns.length;
}

async function sendChat(query, existingTurn = null) {
  query = query.trim();
  if (!query || state.busy) return;
  const container = $("#chat-messages");
  if (!state.turns.length) container.innerHTML = "";
  const turn = existingTurn || { id: crypto.randomUUID?.() || String(Date.now()), query };
  Object.assign(turn, { status: "loading", data: null, error: null, feedback: null, papers: [...state.scope] });
  if (!existingTurn) {
    state.turns.push(turn);
    container.insertAdjacentHTML("beforeend", `<div class="turn" data-turn="${turn.id}"></div>`);
  }
  renderTurn(turn);
  scrollToBottom();
  state.busy = true;
  updateSendButton();
  try {
    const payload = { query, papers: turn.papers, history: historyBefore(turn), top_k: state.settings.topK || undefined };
    let frame = 0;
    await streamChat(payload, event => {
      if (event.type === "meta") {
        turn.data = { answer: "", trust: null, query_used: event.query_used, citations: event.citations, context: event.context };
        turn.status = "streaming";
        renderTurn(turn);
      } else if (event.type === "token") {
        turn.data.answer += event.text;
        // Re-render the answer at most once per animation frame.
        if (!frame) frame = requestAnimationFrame(() => {
          frame = 0;
          const md = $(`[data-turn="${turn.id}"] .md`);
          if (md) md.innerHTML = renderMarkdown(stripSourcesLine(turn.data.answer));
          if (nearBottom()) window.scrollTo({ top: document.body.scrollHeight });
        });
      } else if (event.type === "done") {
        cancelAnimationFrame(frame);
        Object.assign(turn.data, { answer: event.answer, trust: event.trust });
        turn.status = "done";
      } else if (event.type === "error") {
        throw new Error(event.detail);
      }
    });
    if (turn.status !== "done") throw new Error("The answer stream ended unexpectedly. Please retry.");
  } catch (error) {
    turn.status = "error";
    turn.error = error.message;
  } finally {
    state.busy = false;
    updateSendButton();
  }
  renderTurn(turn);
  persistChat();
  scrollToBottom();
}

/** POST to /api/chat/stream and call onEvent for each NDJSON line. */
async function streamChat(payload, onEvent) {
  let response;
  try {
    response = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
  } catch (_) {
    throw new Error("Network connection failed. Check that the app server is running.");
  }
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 503 || response.status === 429) refreshHealth();
    throw new Error(formatApiError(data.detail || data, `Request failed (${response.status})`));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

const nearBottom = () => window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;

function scrollToBottom() {
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

function trustBadge(trust = {}) {
  const label = trust.label || "n/a";
  const cls = { High: "badge-high", Medium: "badge-medium", Low: "badge-low" }[label] || "badge-neutral";
  const text = label === "n/a" ? "No answer found" : `${label} confidence`;
  const reason = trust.reason ? `${trust.reason}. ` : "";
  return `<span class="badge ${cls}" title="${escapeHtml(reason)}A heuristic from retrieval similarity and a support check, not a guarantee.">${escapeHtml(text)}</span>`;
}

function renderTurn(turn) {
  const wrapper = $(`[data-turn="${turn.id}"]`);
  if (!wrapper) return;
  const user = `<div class="msg-user">${escapeHtml(turn.query)}</div>`;
  let bot;
  if (turn.status === "loading") {
    bot = `<div class="typing" aria-label="Searching papers"><span></span><span></span><span></span></div>
      <div class="typing-label">Searching your papers and writing a grounded answer…</div>`;
  } else if (turn.status === "error") {
    bot = `<div class="notice notice-error"><span>${escapeHtml(turn.error)}</span>
      <button type="button" class="btn btn-sm" data-act="retry" style="margin-left:auto">${icon("refresh")}Retry</button></div>`;
  } else {
    const data = turn.data;
    const rewritten = data.query_used && data.query_used.trim().toLowerCase() !== turn.query.trim().toLowerCase()
      ? `<div class="rewritten">Searched for: “${escapeHtml(data.query_used)}”</div>` : "";
    const open = state.settings.showContext ? " open" : "";
    const sources = (data.context || []).map(item => `
      <details class="source" data-citation="${escapeHtml(item.citation)}"${open}>
        <summary>${icon("eye")}<span class="cite">${escapeHtml(item.citation)}</span>
          ${item.score != null ? `<span class="score" title="Dense similarity">${Number(item.score).toFixed(2)}</span>` : ""}</summary>
        <div class="passage">${escapeHtml(item.text)}</div>
      </details>`).join("");
    const streaming = turn.status === "streaming";
    const status = streaming
      ? `<span class="badge badge-neutral"><span class="spinner" style="width:10px;height:10px;border-width:1.5px"></span>Writing answer…</span>`
      : trustBadge(data.trust);
    bot = `${rewritten}
      <div class="md${streaming ? " streaming" : ""}">${renderMarkdown(stripSourcesLine(data.answer)) || "<p></p>"}</div>
      <div class="msg-meta">${status}
        ${turn.papers?.length ? `<span class="badge badge-neutral">Scope: ${turn.papers.map(paperTitle).map(escapeHtml).join(", ")}</span>` : ""}</div>
      ${sources ? `<div class="sources">${sources}</div>` : ""}
      <div class="msg-actions${streaming ? " hidden" : ""}">
        <button type="button" class="icon-btn" data-act="copy" title="Copy answer" aria-label="Copy answer">${icon("copy")}</button>
        <button type="button" class="icon-btn" data-act="regenerate" title="Regenerate" aria-label="Regenerate answer">${icon("refresh")}</button>
        <button type="button" class="icon-btn ${turn.feedback === "up" ? "on" : ""}" data-act="up" title="Helpful" aria-label="Helpful" aria-pressed="${turn.feedback === "up"}">${icon("up")}</button>
        <button type="button" class="icon-btn ${turn.feedback === "down" ? "on" : ""}" data-act="down" title="Not helpful" aria-label="Not helpful" aria-pressed="${turn.feedback === "down"}">${icon("down")}</button>
      </div>`;
  }
  wrapper.innerHTML = `<div class="stack">${user}<div class="msg-bot"><div class="avatar">AI</div><div class="msg-body">${bot}</div></div></div>`;
}

$("#chat-messages").addEventListener("click", async event => {
  const wrapper = event.target.closest("[data-turn]");
  if (!wrapper) return;
  const turn = state.turns.find(item => item.id === wrapper.dataset.turn);
  const cite = event.target.closest(".cite");
  if (cite && cite.dataset.cite) {
    const source = $$(".source", wrapper).find(item => item.dataset.citation === cite.dataset.cite);
    if (source) {
      source.open = true;
      source.classList.remove("flash");
      void source.offsetWidth;
      source.classList.add("flash");
      source.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      toast("That page was cited but is not among the retrieved passages.", "warn");
    }
    return;
  }
  const act = event.target.closest("[data-act]")?.dataset.act;
  if (!act || !turn) return;
  if (act === "retry" || act === "regenerate") return sendChat(turn.query, turn);
  if (act === "copy") {
    try { await navigator.clipboard.writeText(turn.data.answer); toast("Answer copied", "success"); }
    catch (_) { toast("Clipboard is not available in this browser.", "warn"); }
    return;
  }
  if (act === "up" || act === "down") {
    if (turn.feedback === act) return;
    try {
      await api("/api/chat/feedback", { method: "POST", json: { query: turn.query, message: turn.data.answer, feedback: act } });
      turn.feedback = act;
      renderTurn(turn);
      toast("Thanks for the feedback", "success");
    } catch (error) { toast(error.message, "error"); }
  }
});

const chatInput = $("#chat-input");
function autosize() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 200)}px`;
}
function updateSendButton() {
  $("#chat-send").disabled = state.busy || !chatInput.value.trim();
}
chatInput.addEventListener("input", () => { autosize(); updateSendButton(); });
chatInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#chat-form").addEventListener("submit", event => {
  event.preventDefault();
  const query = chatInput.value;
  if (!query.trim() || state.busy) return;
  chatInput.value = "";
  autosize();
  sendChat(query);
});

$("#clear-chat").addEventListener("click", () => {
  if (state.busy) return;
  state.turns = [];
  persistChat();
  renderChatEmpty();
  chatInput.focus();
});

$("#export-chat").addEventListener("click", () => {
  const body = state.turns.filter(turn => turn.status === "done").map(turn => {
    const sources = (turn.data.context || []).map(item => `- ${item.citation}`).join("\n");
    return `## Q: ${turn.query}\n\n${turn.data.answer}\n\n**Confidence:** ${turn.data.trust?.label || "n/a"}\n\n**Retrieved sources**\n${sources || "- none"}`;
  }).join("\n\n---\n\n");
  downloadText(`chat-${new Date().toISOString().slice(0, 10)}.md`, `# Chat export\n\n${body}\n`);
});

function restoreChat() {
  const saved = store.session("chatTurns");
  if (!Array.isArray(saved) || !saved.length) return;
  state.turns = saved.filter(turn => turn && turn.status === "done" && turn.data);
  const container = $("#chat-messages");
  container.innerHTML = state.turns.map(turn => `<div class="turn" data-turn="${turn.id}"></div>`).join("");
  state.turns.forEach(renderTurn);
  persistChat();
}

/* ==========================================================================
   Quiz
   ========================================================================== */
function quizSettings() {
  return {
    pdf: $("#quiz-pdf").value,
    topic: $("#quiz-topic").value.trim(),
    difficulty: $("#quiz-difficulty").value,
    question_count: Number($("#quiz-count").value),
    question_type: $("#quiz-type").value,
    page_range: $("#quiz-pages").value.trim() || null,
  };
}

function updateGenerateLabel() {
  $("#generate-quiz span").textContent = `Generate ${$("#quiz-count").value} questions`;
}
$("#quiz-count").addEventListener("change", updateGenerateLabel);

async function generateQuiz(settings = quizSettings()) {
  if (!settings.pdf) return toast("Add a paper first.", "warn");
  if (settings.page_range && !/^\s*\d+\s*[-–]\s*\d+\s*$/.test(settings.page_range)) {
    return toast("Page range must look like 1-12.", "warn");
  }
  const button = $("#generate-quiz");
  setBusy(button, true, "Generating…");
  const area = $("#quiz-area");
  area.innerHTML = `<div class="stack">${Array.from({ length: 3 }, () => '<div class="card card-pad"><div class="skeleton" style="height:16px;width:70%"></div><div class="skeleton" style="height:38px;margin-top:14px"></div></div>').join("")}</div>`;
  try {
    const quiz = await api("/api/quiz/generate", { method: "POST", json: settings });
    if (!quiz.covered || !quiz.questions?.length) {
      area.innerHTML = `<div class="notice notice-warn">The retrieved passages do not cover “${escapeHtml(settings.topic || "this topic")}” well enough. Try a different topic or page range.</div>`;
      return;
    }
    state.quiz = {
      settings, questions: quiz.questions, context: quiz.context || [],
      answers: quiz.questions.map(() => ""), result: null, retry: false,
    };
    renderQuiz();
    area.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    area.innerHTML = `<div class="notice notice-error"><span>${escapeHtml(error.message)}</span>
      <button type="button" class="btn btn-sm" id="quiz-retry-generate" style="margin-left:auto">Retry</button></div>`;
    $("#quiz-retry-generate").addEventListener("click", () => generateQuiz(settings));
  } finally {
    setBusy(button, false);
  }
}
$("#generate-quiz").addEventListener("click", () => generateQuiz());

function questionCard(question, index) {
  const answer = state.quiz.answers[index];
  const tags = `<div class="q-tags">
      <span class="badge badge-neutral">${escapeHtml(question.type || "Question")}</span>
      <span class="badge badge-accent">${escapeHtml(question.difficulty || state.quiz.settings.difficulty)}</span>
      <span class="cite">${escapeHtml(question.source)}</span></div>`;
  const input = Array.isArray(question.options) && question.options.length
    ? `<div class="options" role="radiogroup" aria-label="Answer for question ${index + 1}">${question.options.map(option => `
        <label class="option"><input type="radio" name="q-${index}" data-answer="${index}" value="${escapeHtml(option)}" ${answer === option ? "checked" : ""}>
        <span>${escapeHtml(option)}</span></label>`).join("")}</div>`
    : `<textarea data-answer="${index}" rows="3" class="field answer-box" placeholder="Type your answer…" aria-label="Answer for question ${index + 1}">${escapeHtml(answer)}</textarea>`;
  return `<div class="card question" data-question="${index}">
      <div class="question-head"><div class="q-num">${index + 1}</div><div><div class="q-text">${escapeHtml(question.question)}</div>${tags}</div></div>
      ${input}
    </div>`;
}

function renderQuiz() {
  const quiz = state.quiz;
  const total = quiz.questions.length;
  $("#quiz-area").innerHTML = `
    <form id="quiz-form" class="stack" novalidate>
      <div class="card quiz-progress no-print">
        <strong>${quiz.retry ? "Retry missed questions" : escapeHtml(paperTitle(quiz.settings.pdf))}</strong>
        <div class="progress-track"><span id="quiz-bar" style="width:0%"></span></div>
        <span id="quiz-count-label" class="hint">0 / ${total}</span>
      </div>
      ${quiz.questions.map(questionCard).join("")}
      <div class="row no-print">
        <button type="submit" id="grade-quiz" class="btn btn-primary">${icon("quiz")}<span>Grade answers</span></button>
        <span class="spacer"></span>
        <button type="button" class="btn btn-ghost btn-sm" data-quiz="export">${icon("download")}Export</button>
        <button type="button" class="btn btn-ghost btn-sm" data-quiz="print">${icon("print")}Print / PDF</button>
      </div>
    </form>`;
  updateQuizProgress();
}

function updateQuizProgress() {
  const answered = state.quiz.answers.filter(answer => answer.trim()).length;
  const total = state.quiz.answers.length;
  $("#quiz-bar").style.width = `${(answered / total) * 100}%`;
  $("#quiz-count-label").textContent = `${answered} / ${total} answered`;
}

$("#quiz-area").addEventListener("input", event => {
  const target = event.target.closest("[data-answer]");
  if (!target || !state.quiz) return;
  const index = Number(target.dataset.answer);
  state.quiz.answers[index] = target.value;
  target.closest(".question")?.classList.remove("needs-answer");
  updateQuizProgress();
});

$("#quiz-area").addEventListener("submit", async event => {
  event.preventDefault();
  const quiz = state.quiz;
  const missing = quiz.answers.map((answer, index) => answer.trim() ? null : index).filter(index => index !== null);
  if (missing.length) {
    missing.forEach(index => $(`[data-question="${index}"]`)?.classList.add("needs-answer"));
    $(`[data-question="${missing[0]}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    return toast(`Please answer all questions (${missing.length} left).`, "warn");
  }
  const button = $("#grade-quiz");
  setBusy(button, true, "Grading…");
  try {
    const result = await api("/api/quiz/grade", {
      method: "POST",
      json: {
        pdf: quiz.settings.pdf, topic: quiz.settings.topic, questions: quiz.questions,
        answers: quiz.answers.map(answer => answer.trim()), page_range: quiz.settings.page_range,
        difficulty: quiz.settings.difficulty, question_count: quiz.settings.question_count,
      },
    });
    quiz.result = result;
    renderQuizResults();
    if (!quiz.retry) {
      const percentage = Math.round((result.score / result.maximum) * 100);
      api("/api/progress/quiz", {
        method: "POST",
        json: { pdf: quiz.settings.pdf, difficulty: quiz.settings.difficulty, score: percentage, topics: [quiz.settings.topic || "General"] },
      }).then(loadQuizHistory).catch(() => {});
    }
  } catch (error) {
    toast(error.message, "error");
    setBusy(button, false);
  }
});

function renderQuizResults() {
  const quiz = state.quiz;
  const { result } = quiz;
  const percentage = Math.round((result.score / result.maximum) * 100);
  const color = percentage >= 80 ? "var(--success)" : percentage >= 50 ? "var(--warn)" : "var(--danger)";
  const headline = percentage >= 80 ? "Excellent work!" : percentage >= 50 ? "Good progress" : "Keep practising";
  const suggestion = percentage >= 80
    ? (quiz.settings.difficulty === "Hard" ? "You have mastered this material." : "Ready for a harder level.")
    : percentage < 40 ? "Review the source passages below, then try an easier level." : "Review the missed questions and retry them.";
  const missed = result.results.map((item, index) => item.grade !== "Correct" ? index : null).filter(index => index !== null);
  const passageFor = source => (quiz.context.find(item => item.citation === source) || {}).text;
  $("#quiz-area").innerHTML = `
    <div class="stack">
      <div class="card score-card">
        <div class="ring" style="--p:${percentage}; --c:${color}"><b>${percentage}%</b></div>
        <div><h3>${headline}</h3><p>${result.score} of ${result.maximum} points · ${escapeHtml(suggestion)}</p></div>
      </div>
      ${result.results.map((item, index) => {
        const grade = item.grade.toLowerCase();
        const badge = { correct: "badge-high", partial: "badge-medium", incorrect: "badge-low" }[grade];
        const passage = passageFor(item.source);
        return `<div class="card result ${grade}">
          <div class="question-head"><div class="q-num">${index + 1}</div>
            <div style="flex:1"><div class="q-text">${escapeHtml(quiz.questions[index].question)}</div>
              <div class="q-tags"><span class="badge ${badge}">${escapeHtml(item.grade)}</span><span class="cite">${escapeHtml(item.source)}</span></div>
              <dl>
                <dt>Your answer</dt><dd>${escapeHtml(quiz.answers[index])}</dd>
                <dt>Correct answer</dt><dd>${escapeHtml(item.correct_answer)}</dd>
                <dt>Feedback</dt><dd>${escapeHtml(item.feedback)}</dd>
              </dl>
              ${passage ? `<details class="source" style="margin-top:10px"><summary>${icon("eye")}Show source passage</summary><div class="passage">${escapeHtml(passage)}</div></details>` : ""}
            </div></div>
        </div>`;
      }).join("")}
      <div class="row no-print">
        ${missed.length ? `<button type="button" class="btn btn-primary" data-quiz="retry-missed">${icon("refresh")}Retry ${missed.length} missed</button>` : ""}
        <button type="button" class="btn" data-quiz="new">${icon("sparkle")}New quiz, same settings</button>
        <span class="spacer"></span>
        <button type="button" class="btn btn-ghost btn-sm" data-quiz="export">${icon("download")}Export</button>
        <button type="button" class="btn btn-ghost btn-sm" data-quiz="print">${icon("print")}Print / PDF</button>
      </div>
    </div>`;
  $("#quiz-area").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#quiz-area").addEventListener("click", event => {
  const action = event.target.closest("[data-quiz]")?.dataset.quiz;
  const quiz = state.quiz;
  if (!action || !quiz) return;
  if (action === "retry-missed") {
    const missed = quiz.result.results.map((item, index) => item.grade !== "Correct" ? index : null).filter(index => index !== null);
    state.quiz = {
      ...quiz, retry: true, result: null,
      questions: missed.map(index => quiz.questions[index]),
      answers: missed.map(() => ""),
    };
    renderQuiz();
  } else if (action === "new") {
    generateQuiz(quiz.settings);
  } else if (action === "export") {
    exportQuiz();
  } else if (action === "print") {
    window.print();
  }
});

function exportQuiz() {
  const quiz = state.quiz;
  const lines = [`# Quiz: ${paperTitle(quiz.settings.pdf)}`, "",
    `- Topic: ${quiz.settings.topic || "main concepts"}`, `- Difficulty: ${quiz.settings.difficulty}`, ""];
  if (quiz.result) lines.push(`**Score:** ${quiz.result.score} / ${quiz.result.maximum}`, "");
  quiz.questions.forEach((question, index) => {
    lines.push(`## ${index + 1}. ${question.question}`, "");
    (question.options || []).forEach(option => lines.push(`- ${option}`));
    if (question.options) lines.push("");
    const result = quiz.result?.results[index];
    if (quiz.answers[index]) lines.push(`**Your answer:** ${quiz.answers[index]}`);
    if (result) lines.push(`**Grade:** ${result.grade}`, `**Correct answer:** ${result.correct_answer}`, `**Feedback:** ${result.feedback}`);
    lines.push(`*Source: ${question.source}*`, "");
  });
  downloadText(`quiz-${paperTitle(quiz.settings.pdf).replace(/\s+/g, "-")}.md`, lines.join("\n"));
}

async function loadQuizHistory() {
  try {
    const progress = await api("/api/progress");
    $("#quiz-history-summary").textContent = progress.quizzes_taken
      ? `${progress.quizzes_taken} quizzes · ${progress.overall_average}% average` : "";
    $("#quiz-history").innerHTML = progress.recent.length
      ? progress.recent.slice(0, 5).map(item => {
        const score = Math.round(item.score);
        const badge = score >= 80 ? "badge-high" : score >= 50 ? "badge-medium" : "badge-low";
        return `<div class="history-item"><span class="name" title="${escapeHtml(item.pdf)}">${escapeHtml(paperTitle(item.pdf))}</span>
          <span class="badge badge-neutral">${escapeHtml(item.difficulty)}</span><span class="badge ${badge}">${score}%</span>
          <span class="hint" style="width:76px;text-align:right">${escapeHtml(relativeDate(item.date))}</span></div>`;
      }).join("")
      : "No attempts yet.";
  } catch (_) { /* history is optional */ }
}

/* ==========================================================================
   Compare
   ========================================================================== */
function updateCompareControls() {
  const enough = state.papers.length >= 2;
  $("#compare-library-warning").classList.toggle("hidden", enough);
  $("#compare-pdf-one").disabled = !enough;
  $("#compare-pdf-two").disabled = !enough;
  $("#compare-swap").disabled = !enough;
  const first = $("#compare-pdf-one").value;
  $$("option", $("#compare-pdf-two")).forEach(option => { option.disabled = option.value === first; });
  $("#compare-button").disabled = !enough || first === $("#compare-pdf-two").value;
}
$("#compare-pdf-one").addEventListener("change", () => {
  const second = $("#compare-pdf-two");
  if (second.value === $("#compare-pdf-one").value) {
    second.value = state.papers.map(paper => paper.filename).find(name => name !== $("#compare-pdf-one").value) || "";
  }
  updateCompareControls();
});
$("#compare-pdf-two").addEventListener("change", updateCompareControls);
$("#compare-swap").addEventListener("click", () => {
  const one = $("#compare-pdf-one"), two = $("#compare-pdf-two");
  [one.value, two.value] = [two.value, one.value];
  updateCompareControls();
});
$$(".compare-chip").forEach(chip => chip.addEventListener("click", () => {
  $("#compare-question").value = chip.textContent.trim();
  $("#compare-button").click();
}));
$("#compare-question").addEventListener("keydown", event => {
  if (event.key === "Enter") { event.preventDefault(); $("#compare-button").click(); }
});

$("#compare-result").addEventListener("click", event => {
  const cite = event.target.closest(".cite[data-cite]");
  if (!cite) return;
  const file = cite.dataset.cite.split(",")[0];
  const column = $$(".paper-col", $("#compare-result")).find(col => col.dataset.paper === file);
  const details = column?.querySelector("details");
  if (!details) return;
  details.open = true;
  details.classList.remove("flash");
  void details.offsetWidth;
  details.classList.add("flash");
  details.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

function summarySection(summary, heading) {
  const escapedHeading = heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const normalized = summary.replace(/\*\*##/g, "##").replace(/^(##[^\n]*?)\*\*\s*$/gm, "$1");
  const match = normalized.match(new RegExp(`##\\s*${escapedHeading}\\s*\\n([\\s\\S]*?)(?=\\n##\\s|$)`, "i"));
  return match ? match[1].trim().replace(/^Answer:\s*/i, "") : "";
}

$("#compare-button").addEventListener("click", async () => {
  const pdfs = [$("#compare-pdf-one").value, $("#compare-pdf-two").value];
  const question = $("#compare-question").value.trim();
  if (!question) { $("#compare-question").focus(); return toast("Enter what you want to compare.", "warn"); }
  if (!pdfs[0] || pdfs[0] === pdfs[1]) return toast("Select two different papers.", "warn");
  const button = $("#compare-button");
  setBusy(button, true, "Comparing…");
  const result = $("#compare-result");
  result.innerHTML = `<div class="grid-2">${'<div class="card card-pad"><div class="skeleton" style="height:16px;width:50%"></div><div class="skeleton" style="height:120px;margin-top:14px"></div></div>'.repeat(2)}</div>`;
  try {
    const data = await api("/api/compare", { method: "POST", json: { pdfs, question } });
    const names = Object.keys(data.papers);
    const column = (name, index) => {
      const section = summarySection(data.summary, name) || summarySection(data.summary, paperTitle(name));
      const passages = data.papers[name] || [];
      return `<section class="card card-pad paper-col" data-paper="${escapeHtml(name)}">
        <h3><span class="badge badge-accent">${index === 0 ? "A" : "B"}</span><span class="name" title="${escapeHtml(name)}">${escapeHtml(paperTitle(name))}</span></h3>
        <div class="md">${renderMarkdown(section || "Not found in this paper for this question.")}</div>
        <details class="source" style="margin-top:14px"><summary>${icon("eye")}${passages.length} retrieved passage${passages.length === 1 ? "" : "s"}</summary>
          <div class="passage">${passages.map(item => `<strong>${escapeHtml(item.citation)}</strong>\n${escapeHtml(item.text)}`).join("\n\n") || "No passages retrieved."}</div>
        </details>
      </section>`;
    };
    const difference = summarySection(data.summary, "Difference");
    const similarities = summarySection(data.summary, "Similarities");
    result.innerHTML = `
      <div class="stack">
        <div class="grid-2">${names.map(column).join("")}</div>
        <div class="card card-pad">
          <div class="summary-section"><h4>Key differences</h4><div class="md">${renderMarkdown(difference || "Not stated.")}</div></div>
          <div class="summary-section"><h4>Similarities</h4><div class="md">${renderMarkdown(similarities || "Not stated.")}</div></div>
        </div>
      </div>`;
  } catch (error) {
    result.innerHTML = `<div class="notice notice-error"><span>${escapeHtml(error.message)}</span>
      <button id="compare-retry" type="button" class="btn btn-sm" style="margin-left:auto">Retry</button></div>`;
    $("#compare-retry").addEventListener("click", () => $("#compare-button").click());
  } finally {
    setBusy(button, false);
    updateCompareControls();
  }
});

/* ==========================================================================
   Settings, theme and progress
   ========================================================================== */
function applyTheme(theme) {
  state.settings.theme = theme;
  store.set("theme", theme);
  const dark = theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $$("[data-theme-value]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.themeValue === theme)));
}
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
  if (state.settings.theme === "system") applyTheme("system");
});
$("#theme-toggle").addEventListener("click", event => {
  const value = event.target.closest("[data-theme-value]")?.dataset.themeValue;
  if (value) applyTheme(value);
});

$("#settings-button").addEventListener("click", async () => {
  openDialog("settings-dialog");
  try {
    const settings = await api("/api/settings");
    state.settings.maxTopK = settings.max_top_k || 10;
    const topK = $("#setting-top-k");
    topK.max = state.settings.maxTopK;
    topK.value = state.settings.topK || settings.top_k;
    topK.dataset.default = settings.top_k;
    $("#setting-model").textContent = `${settings.model} · embeddings ${settings.embedding_model} · chunks ${settings.chunk_size}/${settings.chunk_overlap}`;
  } catch (error) { $("#setting-model").textContent = error.message; }
});
$("#setting-top-k").addEventListener("change", event => {
  const value = Math.round(Number(event.target.value));
  if (!value || value < 1 || value > state.settings.maxTopK) {
    event.target.value = state.settings.topK || event.target.dataset.default || 3;
    return toast(`Choose a value between 1 and ${state.settings.maxTopK}.`, "warn");
  }
  state.settings.topK = value;
  store.set("topK", String(value));
});
$("#setting-context").addEventListener("change", event => {
  state.settings.showContext = event.target.checked;
  store.set("showContext", String(event.target.checked));
  state.turns.forEach(renderTurn);
});

$("#progress-button").addEventListener("click", async () => {
  const content = $("#progress-content");
  content.innerHTML = '<div class="skeleton" style="height:70px"></div>';
  openDialog("progress-dialog");
  try {
    const progress = await api("/api/progress");
    if (!progress.quizzes_taken) {
      content.innerHTML = `<div class="empty-state" style="padding:20px 0"><div class="orb">${icon("chart")}</div>
        <h2 style="font-size:17px">No quizzes yet</h2><p>Take a quiz to start tracking your progress.</p>
        <button type="button" class="btn btn-primary" id="progress-start" style="margin-top:16px">Start a quiz</button></div>`;
      $("#progress-start").addEventListener("click", () => { $("#progress-dialog").close(); setMode("quiz"); });
      return;
    }
    const best = Object.entries(progress.averages).sort((a, b) => b[1] - a[1])[0];
    content.innerHTML = `
      <div class="stats">
        <div class="stat"><div class="v">${progress.quizzes_taken}</div><div class="k">Quizzes taken</div></div>
        <div class="stat"><div class="v">${progress.overall_average ?? "–"}%</div><div class="k">Average score</div></div>
        <div class="stat"><div class="v">${best ? escapeHtml(best[0]) : "–"}</div><div class="k">Strongest level</div></div>
      </div>
      <h3 style="font-size:13px;margin:20px 0 4px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em">By difficulty</h3>
      ${Object.entries(progress.averages).map(([level, score]) => `
        <div class="bar-row"><span>${escapeHtml(level)}</span><div class="progress-track"><span style="width:${score}%"></span></div><span>${score}%</span></div>`).join("")}
      <h3 style="font-size:13px;margin:20px 0 8px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em">Topics to review</h3>
      <div class="row">${progress.most_missed_topics.map(topic => `<span class="chip">${escapeHtml(topic)}</span>`).join("") || '<span class="hint">Nothing below 70%. Nice work.</span>'}</div>`;
  } catch (error) {
    content.innerHTML = `<div class="notice notice-error">${escapeHtml(error.message)}</div>`;
  }
});

/* ==========================================================================
   Keyboard shortcuts
   ========================================================================== */
document.addEventListener("keydown", event => {
  const typing = event.target.closest("input, textarea, select, [contenteditable]");
  if (event.key === "Escape") { closeMenus(); closeSidebar(); return; }
  if (typing || event.metaKey || event.ctrlKey || event.altKey || document.querySelector("dialog[open]")) return;
  if (["1", "2", "3"].includes(event.key)) setMode(["chat", "quiz", "compare"][Number(event.key) - 1]);
  if (event.key === "/") { event.preventDefault(); setMode("chat"); chatInput.focus(); }
});

/* ==========================================================================
   Init
   ========================================================================== */
(function init() {
  state.settings.topK = Number(store.get("topK")) || null;
  state.settings.showContext = store.get("showContext") === "true";
  $("#setting-context").checked = state.settings.showContext;
  applyTheme(store.get("theme", "system"));
  updateGenerateLabel();
  restoreChat();
  const mode = store.get("mode", "chat");
  setMode(MODE_COPY[mode] ? mode : "chat");
  refreshHealth();
  loadLibrary();
})();
