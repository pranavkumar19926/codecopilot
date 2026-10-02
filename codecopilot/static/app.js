"use strict";
const $ = (s) => document.querySelector(s);
const state = { repos: [], current: null, busy: false, fileCache: new Map(), pollTimer: null };

// ---------------------------------------------------------------------------------------------- api
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}

// ------------------------------------------------------------------------------------- text helpers
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CITE = /[\[`]([^\[\]`\s]+?\.\w+):(\d+)(?:-(\d+))?[\]`]/g;

/** Small markdown subset: code fences, inline code, bold, bullets, paragraphs, and citations. */
function render(md, invalid = new Set()) {
  const blocks = md.split(/```/);
  let html = "";
  blocks.forEach((block, i) => {
    if (i % 2 === 1) {
      html += `<pre><code>${esc(block.replace(/^\w*\n/, ""))}</code></pre>`;
      return;
    }
    const paras = block.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
    for (const p of paras) {
      const lines = p.split("\n");
      if (lines.every((l) => /^\s*[-*] /.test(l))) {
        html += "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[-*] /, ""), invalid)}</li>`).join("") + "</ul>";
      } else {
        html += `<p>${lines.map((l) => inline(l, invalid)).join("<br>")}</p>`;
      }
    }
  });
  return html;
}

function inline(text, invalid) {
  // pull citations out first so backtick citations aren't treated as inline code
  const cites = [];
  text = text.replace(CITE, (m, path, a, b) => {
    cites.push({ path, a: +a, b: +(b || a) });
    return `\u0000${cites.length - 1}\u0000`;
  });
  let out = esc(text)
    .replace(/`([^`]+)`/g, (m, code) => {
      const name = code.replace(/\(.*$/, "");
      return /^[A-Za-z_][\w.]*$/.test(name) && name.length > 2
        ? `<code class="sym" data-sym="${esc(name)}" title="Show callers and callees">${code}</code>`
        : `<code>${code}</code>`;
    })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return out.replace(/\u0000(\d+)\u0000/g, (m, i) => {
    const c = cites[+i];
    const key = `${c.path}:${c.a}-${c.b}`;
    const bad = invalid.has(key);
    return `<button class="cite${bad ? " bad" : ""}" data-path="${esc(c.path)}" data-a="${c.a}" data-b="${c.b}"
      title="${bad ? "This range was not in the code shown to the model" : "Open these lines"}">${esc(key)}</button>`;
  });
}

// --------------------------------------------------------------------------------------- repo list
async function loadRepos() {
  try {
    state.repos = await api("/api/repos");
  } catch (e) {
    $("#repo-list").innerHTML = `<li class="empty error">Can't reach the server: ${esc(e.message)}</li>`;
    return;
  }
  renderRepos();
  const pending = state.repos.some((r) => ["queued", "cloning", "indexing"].includes(r.status));
  clearTimeout(state.pollTimer);
  if (pending) state.pollTimer = setTimeout(loadRepos, 1500);
  if (!state.current) {
    const first = state.repos.find((r) => r.status === "ready");
    if (first) selectRepo(first.id);
  } else {
    const cur = state.repos.find((r) => r.id === state.current);
    if (cur) updateHeader(cur);
  }
}

function repoMeta(r) {
  if (r.status === "ready") {
    const parts = [`${r.files} files`, `${r.chunks} chunks`];
    if (r.graph_edges != null) parts.push(`${r.graph_edges} call links`);
    return parts.join(", ");
  }
  if (r.status === "error") return r.message || "Indexing failed";
  return r.message || r.status;
}

function renderRepos() {
  const ul = $("#repo-list");
  if (!state.repos.length) {
    ul.innerHTML = `<li class="empty">No repositories yet. Add one below to start asking questions.</li>`;
    return;
  }
  ul.innerHTML = state.repos.map((r) => {
    const busy = ["queued", "cloning", "indexing"].includes(r.status);
    return `<li><button data-id="${esc(r.id)}" ${r.id === state.current ? 'aria-current="true"' : ""}>
      <span class="repo-name">${esc(r.name)}</span>
      <span class="repo-meta ${r.status === "error" ? "err" : ""} ${busy ? "busy" : ""}">${esc(repoMeta(r))}</span>
    </button></li>`;
  }).join("");
}

$("#repo-list").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-id]");
  if (!b) return;
  const r = state.repos.find((x) => x.id === b.dataset.id);
  if (r && r.status === "ready") selectRepo(r.id);
  $(".repos").classList.remove("open");
});

function updateHeader(r) {
  $("#repo-title").textContent = r.name;
  $("#repo-sub").textContent = r.status === "ready" ? repoMeta(r) : "";
}

function selectRepo(id) {
  state.current = id;
  state.fileCache.clear();
  const r = state.repos.find((x) => x.id === id);
  updateHeader(r);
  renderRepos();
  $("#question").disabled = false;
  $("#ask-btn").disabled = false;
  resetViewer();
  showIntro(r);
}

async function showIntro(r) {
  $("#thread").innerHTML = `<div class="intro">
    <h2>Ask about ${esc(r.name)}</h2>
    <p>Answers cite exact lines. Click a highlighted citation to see the code it points to, or a function name to see who calls it.</p>
    <div class="suggestions"></div>
  </div>`;
  let qs = ["Where is the main entry point?", "How are errors handled?"];
  try { qs = (await api(`/api/repos/${encodeURIComponent(r.id)}/suggestions`)).questions; } catch (_) {}
  const box = $(".intro .suggestions");
  if (box) box.innerHTML = qs.map((q) => `<button>${esc(q)}</button>`).join("");
}

$("#thread").addEventListener("click", (e) => {
  const sug = e.target.closest(".suggestions button");
  if (sug) { $("#question").value = sug.textContent; $("#question").focus(); return; }
  const cite = e.target.closest(".cite, .loc");
  if (cite) { openCode(cite.dataset.path, +cite.dataset.a, +cite.dataset.b); return; }
  const sym = e.target.closest(".sym");
  if (sym) { lookupGraph(sym.dataset.sym); }
});

// ----------------------------------------------------------------------------------------- add repo
$("#add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const v = $("#add-input").value.trim();
  $("#add-error").textContent = "";
  if (!v) { $("#add-error").textContent = "Paste a GitHub URL or a folder path."; return; }
  const body = v.startsWith("http") ? { url: v } : { path: v };
  body.exclude_tests = $("#add-tests").checked;
  try {
    await api("/api/repos", { method: "POST", body: JSON.stringify(body) });
    $("#add-input").value = "";
    loadRepos();
  } catch (err) {
    $("#add-error").textContent = err.message;
  }
});

// -------------------------------------------------------------------------------------------- ask
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});

$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (!q || state.busy || !state.current) return;
  if ($(".intro")) $("#thread").innerHTML = "";
  $("#question").value = "";
  setBusy(true);
  try {
    if ($("#search-only").checked) await runSearch(q); else await runAsk(q);
  } finally {
    setBusy(false);
  }
});

function setBusy(b) {
  state.busy = b;
  $("#ask-btn").disabled = b;
  $("#ask-btn").textContent = b ? "Working…" : "Ask";
}

function newTurn(q) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<p class="q">${esc(q)}</p><div class="a"><div class="status busy">Searching the code…</div><div class="body"></div></div>`;
  $("#thread").appendChild(wrap);
  wrap.scrollIntoView({ block: "end", behavior: "smooth" });
  return wrap.querySelector(".a");
}

function sourcesHtml(hits, related, rewrites) {
  const row = (h) => `<li class="src"><button class="loc" data-path="${esc(h.path)}" data-a="${h.start}" data-b="${h.end}">${esc(h.path)}:${h.start}-${h.end}</button>
    ${h.symbol ? `<code class="sym" data-sym="${esc(h.symbol.split(", ")[0])}">${esc(h.symbol)}</code>` : ""}
    ${h.why ? `<span class="why">${esc(h.why)}</span>` : ""}</li>`;
  let html = `<details class="sources"><summary>Read ${hits.length + related.length} code excerpts</summary><ul>${hits.map(row).join("")}</ul>`;
  if (related.length) html += `<p class="related-label">Added from the call graph</p><ul>${related.map(row).join("")}</ul>`;
  html += "</details>";
  if (rewrites && rewrites.length) {
    html += `<details class="rewrites"><summary>Also searched for</summary><ul>${rewrites.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>`;
  }
  return html;
}

async function runAsk(q) {
  const box = newTurn(q);
  const status = box.querySelector(".status");
  let body = box.querySelector(".body");
  let text = "";
  let sources = "";
  let r;
  try {
    r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify({ repo: state.current, question: q }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  } catch (err) {
    status.classList.remove("busy");
    status.innerHTML = `<span class="error">${esc(err.message)}</span>`;
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const ev = JSON.parse(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
      if (ev.type === "sources") {
        sources = sourcesHtml(ev.hits, ev.related, ev.rewrites);
        status.textContent = `Writing an answer from ${ev.hits.length + ev.related.length} excerpts…`;
      } else if (ev.type === "token") {
        text += ev.text;
        body.innerHTML = render(text);
      } else if (ev.type === "repair") {
        const draft = document.createElement("details");
        draft.className = "draft";
        draft.innerHTML = `<summary>First draft failed the citation check, rewriting it</summary>
          <ul class="problems">${ev.problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
          <div class="body">${render(text)}</div>`;
        body.replaceWith(draft);
        body = document.createElement("div");
        body.className = "body";
        draft.after(body);
        text = "";
        status.textContent = "Revising citations…";
      } else if (ev.type === "done") {
        const invalid = new Set(ev.check.invalid || []);
        body.innerHTML = render(ev.text, invalid);
        status.remove();
        const c = ev.check;
        const good = c.n_citations && !c.invalid.length && !(c.unsupported || []).length && !(c.uncited || []).length;
        const foot = document.createElement("div");
        foot.className = "foot " + (c.refusal ? "" : good ? "ok" : "warn");
        foot.textContent = c.refusal
          ? `Answered in ${ev.seconds}s. The retrieved code doesn't cover this, so the answer says so instead of guessing. Open the excerpts below to see what was searched.`
          : good
          ? `Answered in ${ev.seconds}s. All ${c.n_citations} citations point to code the model was shown.`
          : `Answered in ${ev.seconds}s. Check the citations: ${c.invalid.length} invalid, ${(c.unsupported || []).length} unsupported, ${(c.uncited || []).length} uncited claims.`;
        box.appendChild(foot);
        box.insertAdjacentHTML("beforeend", sources);
      } else if (ev.type === "error") {
        status.classList.remove("busy");
        status.innerHTML = `<span class="error">${esc(ev.message)}</span>`;
      }
    }
  }
}

async function runSearch(q) {
  const box = newTurn(q);
  const status = box.querySelector(".status");
  try {
    const res = await api(`/api/search?repo=${encodeURIComponent(state.current)}&q=${encodeURIComponent(q)}&k=8`);
    status.remove();
    box.querySelector(".body").innerHTML = `<ul class="hits">${res.hits.map((h) =>
      `<li class="src"><button class="loc" data-path="${esc(h.path)}" data-a="${h.start}" data-b="${h.end}">${esc(h.path)}:${h.start}-${h.end}</button>
       ${h.symbol ? `<code class="sym" data-sym="${esc(h.symbol.split(", ")[0])}">${esc(h.symbol)}</code>` : ""}</li>`).join("")}</ul>`
      + (res.rewrites.length ? `<details class="rewrites"><summary>Also searched for</summary><ul>${res.rewrites.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>` : "");
    if (res.hits[0]) openCode(res.hits[0].path, res.hits[0].start, res.hits[0].end);
  } catch (err) {
    status.classList.remove("busy");
    status.innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
}

// ------------------------------------------------------------------------------------ code viewer
function showTab(which) {
  const code = which === "code";
  $("#tab-code").setAttribute("aria-selected", code);
  $("#tab-graph").setAttribute("aria-selected", !code);
  $("#pane-code").hidden = !code;
  $("#pane-graph").hidden = code;
  $("#viewer").classList.add("open");
}
$("#tab-code").onclick = () => showTab("code");
$("#tab-graph").onclick = () => showTab("graph");
$("#close-viewer").onclick = () => $("#viewer").classList.remove("open");
$("#show-repos").onclick = () => $(".repos").classList.toggle("open");

/** A new repo must not show the previous repo's file or call-graph lookup. */
function resetViewer() {
  $("#file-head").textContent = "Click a citation in an answer to see the lines it points to.";
  $("#code").innerHTML = "";
  $("#graph-input").value = "";
  $("#graph-out").innerHTML = `<p class="hint">Look up any function to see who calls it and what it calls. Symbols in answers are clickable too.</p>`;
  showTab("code");
  $("#viewer").classList.remove("open");
}

async function openCode(path, a, b) {
  showTab("code");
  const key = `${state.current}|${path}`;
  let lines = state.fileCache.get(key);
  if (!lines) {
    $("#file-head").innerHTML = `Loading <strong>${esc(path)}</strong>…`;
    try {
      lines = (await api(`/api/file?repo=${encodeURIComponent(state.current)}&path=${encodeURIComponent(path)}`)).lines;
      state.fileCache.set(key, lines);
    } catch (err) {
      $("#file-head").innerHTML = `<span class="error">${esc(err.message)}</span>`;
      $("#code").innerHTML = "";
      return;
    }
  }
  $("#file-head").innerHTML = `<strong>${esc(path)}</strong>, lines ${a}–${b} of ${lines.length}`;
  const frag = lines.map((t, i) => {
    const n = i + 1;
    const hl = n >= a && n <= b;
    return `<div class="ln${hl ? " hl flash" : ""}" ${n === a ? 'id="hl-start"' : ""}><span class="n">${n}</span><span class="t">${esc(t) || " "}</span></div>`;
  }).join("");
  $("#code").innerHTML = frag;
  const start = $("#hl-start");
  if (start) start.scrollIntoView({ block: "center" });
}

// ------------------------------------------------------------------------------------- call graph
$("#graph-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const s = $("#graph-input").value.trim();
  if (s) lookupGraph(s);
});

async function lookupGraph(symbol) {
  showTab("graph");
  $("#graph-input").value = symbol;
  const out = $("#graph-out");
  out.innerHTML = `<p class="hint">Looking up ${esc(symbol)}…</p>`;
  let res;
  try {
    res = await api(`/api/graph?repo=${encodeURIComponent(state.current)}&symbol=${encodeURIComponent(symbol)}`);
  } catch (err) {
    out.innerHTML = `<p class="error">${esc(err.message)}</p>`;
    return;
  }
  if (!res.matches.length) {
    const sim = res.similar || [];
    out.innerHTML = `<p class="hint">No function or class named <code>${esc(symbol)}</code> in this repository.</p>`
      + (sim.length ? `<p class="hint">Did you mean:</p><p class="did-you-mean">${sim.map((n) =>
          `<button class="sym-btn" data-sym="${esc(n)}">${esc(n)}</button>`).join(" ")}</p>` : "");
    return;
  }
  const link = (qual, path, line, extra) =>
    `<li><button data-path="${esc(path)}" data-line="${line}" data-sym="${esc(qual)}">${esc(qual)}</button> <span class="how">${esc(extra)}</span></li>`;
  out.innerHTML = res.matches.map((m) => `
    <h3>${esc(m.qual)}</h3>
    <p class="defined">${esc(m.kind)} in <button class="loc" data-path="${esc(m.path)}" data-a="${m.line}" data-b="${m.end_line}">${esc(m.path)}:${m.line}</button></p>
    <h4>Called by (${m.callers.length})</h4>
    ${m.callers.length ? `<ul>${m.callers.map((c) => link(c.qual, c.path, c.line, `${c.how} it at line ${c.line}`)).join("")}</ul>` : `<p class="hint">Nothing in this repository calls it, or the calls can't be resolved statically.</p>`}
    <h4>Calls (${m.callees.length})</h4>
    ${m.callees.length ? `<ul>${m.callees.map((c) => link(c.qual, c.path, c.line, c.how === "call" ? "" : c.how)).join("")}</ul>` : `<p class="hint">No calls to other definitions in this repository.</p>`}
    <p class="impact">Changing it could affect ${m.impact} definition${m.impact === 1 ? "" : "s"} (callers up to 3 levels up).</p>
  `).join("");
}

$("#graph-out").addEventListener("click", (e) => {
  const sug = e.target.closest(".sym-btn");
  if (sug) { lookupGraph(sug.dataset.sym); return; }
  const loc = e.target.closest(".loc");
  if (loc) { openCode(loc.dataset.path, +loc.dataset.a, +loc.dataset.b); return; }
  const b = e.target.closest("li button");
  if (!b) return;
  if (e.shiftKey) { lookupGraph(b.dataset.sym); return; }
  openCode(b.dataset.path, +b.dataset.line, +b.dataset.line);
});

// ------------------------------------------------------------------------------------------ boot
(async function boot() {
  try {
    const c = await api("/api/config");
    $("#model-line").textContent = `Answers by ${c.llm} on ${c.provider}.`;
    if (!c.allow_local) $("#add-input").placeholder = "https://github.com/owner/repo";
    else $("#add-input").placeholder = "GitHub URL or local folder path";
  } catch (_) {}
  loadRepos();
})();
