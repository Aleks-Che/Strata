// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "monitor", "statistics", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "monitor", "statistics", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "statistics") loadStatistics();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Kept in this browser only."); };

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    applyReasoningCapabilities();
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ persistent token statistics
let vramData = null, vramSaving = false, vramLoaded = false, vramError = "";
function vramModeFields() {
  const mode = $("vram-mode").value;
  $("vram-matrices-field").hidden = mode !== "count";
  $("vram-matrices").disabled = mode !== "count";
  $("vram-target-field").hidden = mode !== "vram";
  $("vram-target").disabled = mode !== "vram";
}
$("vram-mode").onchange = vramModeFields;
$("vram-close").onclick = () => $("vram-dialog").close();
function renderVram(data) {
  if (!data) return;
  vramData = data;
  if (!$("vram-dialog").open) return;
  const live = data.live || {}, available = data.alive && live.telemetry_ok;
  $("vram-used").textContent = available ? `${gb(live.total_bytes - live.free_bytes, 2)} / ${gb(live.total_bytes)} GiB` : "–";
  $("vram-count").textContent = data.alive && live.cached_matrices != null ? fmt(live.cached_matrices) : "–";
  $("vram-cache").textContent = available ? `${gb(live.cache_bytes, 2)} / ${gb(live.limit_bytes, 2)} GiB` : "–";
  $("vram-free").textContent = available ? `${gb(live.free_bytes, 2)} GiB` : "–";
  $("vram-fields").disabled = !vramLoaded || !data.supported || vramSaving;
  let note = data.persistent ? "Settings are saved for this launch profile." : "Settings last until the server is closed.";
  let error = vramError || data.storage_error || "";
  if (!data.supported) error = "This engine does not advertise support for live VRAM settings.";
  else if (!data.alive) note = "Model unloaded. Saved settings will apply when it loads.";
  else if (data.pending) note = "Saved. Waiting for the current GPU operation to finish…";
  else if (!live.enabled) note = "Startup cache budgets are active. Apply settings to enable live memory control.";
  else if (!live.telemetry_ok) error = "GPU memory readings are unavailable. Cache growth is paused.";
  else if (live.target_unreachable) error = "The cache has been released. Fixed model memory and other apps still exceed the target or free-memory reserve.";
  if (live.allocation_failures && !error) note += ` GPU cache allocations failed ${fmt(live.allocation_failures)} time(s); affected matrices were streamed.`;
  $("vram-status").textContent = error || (vramSaving ? "Saving…" : note);
  $("vram-status").dataset.error = String(Boolean(error));
}
$("vram-settings-btn").onclick = async () => {
  vramError = ""; vramLoaded = false;
  $("vram-fields").disabled = true;
  $("vram-status").textContent = "Loading settings…";
  $("vram-dialog").showModal();
  try {
    const response = await fetch("vram/settings", {headers: headers(), cache: "no-store", signal: AbortSignal.timeout(10000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
    $("vram-mode").value = data.settings.mode;
    $("vram-matrices").value = data.settings.matrices;
    $("vram-target").value = data.settings.target_mib ? data.settings.target_mib / 1024 :
      Math.max(0.125, Math.floor(((data.live?.total_bytes || 26 * 1073741824) / 1073741824 - 2) * 8) / 8);
    $("vram-reserve").value = data.settings.reserve_mib;
    vramModeFields();
    vramLoaded = true;
    renderVram(data);
  } catch (error) {
    vramError = error.message;
    $("vram-status").textContent = vramError;
    $("vram-status").dataset.error = "true";
  }
};
$("vram-form").onsubmit = async (event) => {
  event.preventDefault();
  if (vramSaving || !vramData?.supported || !$("vram-form").reportValidity()) return;
  const settings = {mode: $("vram-mode").value, matrices: Number($("vram-matrices").value),
                    target_mib: Math.round(Number($("vram-target").value) * 1024), reserve_mib: Number($("vram-reserve").value)};
  vramSaving = true; vramError = ""; renderVram(vramData);
  try {
    const response = await fetch("vram/settings", {method: "POST", headers: headers(true), body: JSON.stringify(settings),
                                                  signal: AbortSignal.timeout(10000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
    vramData = data;
  } catch (error) { vramError = error.message; }
  finally { vramSaving = false; renderVram(vramData); }
};

// ------------------------------------------------------------------ persistent token statistics
let statisticsLoading = false, statisticsData = null;
const STATS_SERIES = [
  {key: "input_tokens", label: "Processed input", cls: "stats-input"},
  {key: "cached_tokens", label: "Cached input", cls: "stats-cached"},
  {key: "output_tokens", label: "Output", cls: "stats-output"},
];
const savedStatsRange = store.get("statsRange", "24h");
$("stats-range").value = ["24h", "7d", "30d", "all"].includes(savedStatsRange) ? savedStatsRange : "24h";
$("stats-range").onchange = () => {
  store.set("statsRange", $("stats-range").value);
  loadStatistics();
};
async function loadStatistics() {
  if (statisticsLoading) return;
  statisticsLoading = true;
  const period = $("stats-range").value;
  try {
    const response = await fetch(`/statistics?range=${encodeURIComponent(period)}`, {
      headers: headers(), cache: "no-store", signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) throw new Error(response.status === 401 ? "Add the API key under About > Settings." : `HTTP ${response.status}`);
    const data = await response.json();
    if (period !== $("stats-range").value) return;
    statisticsData = data;
    renderStatistics(data);
  } catch (error) {
    $("stats-error").hidden = false;
    $("stats-error").textContent = `Statistics could not be refreshed. ${error.message} Last displayed values may be outdated.`;
    $("stats-saved").textContent = "Unavailable";
  } finally {
    statisticsLoading = false;
    if (period !== $("stats-range").value && tab === "statistics") loadStatistics();
  }
}
setInterval(() => { if (tab === "statistics" && !document.hidden) loadStatistics(); }, 5000);

function renderStatistics(data) {
  const totals = data.totals, selected = data.period_totals;
  $("stats-input").textContent = fmt(totals.input_tokens);
  $("stats-cached").textContent = fmt(totals.cached_tokens);
  $("stats-output").textContent = fmt(totals.output_tokens);
  $("stats-requests").textContent = fmt(totals.requests);
  const input = totals.input_tokens + totals.cached_tokens;
  $("stats-cache-rate").textContent = input ? `${fmt(100 * totals.cached_tokens / input, 1)}% of input reused · RAM + active session` : "Reused from RAM or the active session";
  $("stats-since").textContent = `All time · since ${new Date(data.started_at * 1000).toLocaleString()} · Chat and connected apps`;
  $("stats-saved").textContent = data.storage_error ? "Not saved" : data.persistent ? "Saved on this PC" : "Memory only";
  $("stats-error").hidden = !data.storage_error;
  $("stats-error").textContent = data.storage_error ? `Could not save ${fmt(data.pending_requests)} requests. Retrying automatically: ${data.storage_error}` : "";
  $("stats-file").textContent = data.persistent ? `File: ${data.storage_file}` : "This test server uses memory only.";
  const interval = data.interval_s < 86400 ? `${data.interval_s / 3600} hour` : `${data.interval_s / 86400} day`;
  $("stats-period").textContent = `Tokens per ${interval} interval · local time · updated after each request`;
  $("stats-period-total").textContent = `This period: ${fmt(selected.requests)} requests · ${fmt(selected.input_tokens)} processed · ${fmt(selected.cached_tokens)} cached · ${fmt(selected.output_tokens)} output`;
  $("stats-empty").hidden = selected.requests > 0;
  // Keep a stable chart during polling, including the pointed-to interval and table scroll.
  const signature = JSON.stringify([data.range, data.points]);
  if ($("stats-chart").dataset.signature === signature) return;
  $("stats-chart").dataset.signature = signature;
  $("stats-hover").textContent = "Point to an interval to see its totals. Exact values are also in Activity data below.";
  const points = data.points, left = 66, width = 954, dx = width / Math.max(1, points.length);
  const parts = ['<title>Token activity</title><desc>Three series with independent vertical scales. Exact values are in the Activity data table below.</desc>'];
  STATS_SERIES.forEach((series, index) => {
    const top = 16 + index * 108, bottom = top + 84;
    const max = Math.max(1, ...points.map((p) => p[series.key]));
    parts.push(`<g class="${series.cls}"><text class="stats-series-name" x="${left}" y="${top}">${series.label}</text>`);
    parts.push(`<text x="${left - 10}" y="${top + 26}" text-anchor="end">${kfmt(max)}</text><text x="${left - 10}" y="${bottom}" text-anchor="end">0</text>`);
    parts.push(`<path class="stats-grid" d="M${left},${top + 20}H${left + width}M${left},${bottom}H${left + width}"/>`);
    points.forEach((point, i) => {
      const height = point[series.key] ? Math.max(1, point[series.key] / max * 64) : 0;
      parts.push(`<rect class="stats-bar" x="${(left + i * dx + dx * .12).toFixed(2)}" y="${(bottom - height).toFixed(2)}" width="${(dx * .76).toFixed(2)}" height="${height.toFixed(2)}" rx="2"/>`);
    });
    parts.push("</g>");
  });
  points.forEach((point, i) => {
    parts.push(`<rect class="stats-hit" data-index="${i}" x="${left + i * dx}" y="8" width="${dx}" height="310"><title>${esc(statisticsIntervalText(point, data.interval_s))}</title></rect>`);
  });
  const tickCount = Math.min(5, points.length);
  for (let i = 0; i < tickCount; i++) {
    const index = tickCount === 1 ? 0 : Math.round(i * (points.length - 1) / (tickCount - 1));
    const date = new Date(points[index].time * 1000);
    const label = data.range === "24h" ? date.toLocaleString(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"})
      : date.toLocaleDateString(undefined, {month: "short", day: "numeric", ...(data.range === "all" ? {year: "numeric"} : {})});
    const anchor = i === 0 ? "start" : i === tickCount - 1 ? "end" : "middle";
    parts.push(`<text x="${left + index * dx + dx / 2}" y="353" text-anchor="${anchor}">${esc(label)}</text>`);
  }
  $("stats-chart").innerHTML = parts.join("");
  $("stats-rows").innerHTML = points.filter((p) => p.requests).reverse().map((p) => `<tr><td>${esc(new Date(p.time * 1000).toLocaleString())}</td>${[...STATS_SERIES.map((s) => s.key), "requests"].map((key) => `<td class="num">${fmt(p[key])}</td>`).join("")}</tr>`).join("") || '<tr><td colspan="5" class="muted">No requests in this period</td></tr>';
}
function statisticsIntervalText(point, interval) {
  const begin = new Date(point.time * 1000).toLocaleString();
  const end = new Date((point.time + interval) * 1000).toLocaleString();
  return `${begin} – ${end}: ${fmt(point.input_tokens)} processed · ${fmt(point.cached_tokens)} cached · ${fmt(point.output_tokens)} output · ${fmt(point.requests)} requests`;
}
$("stats-chart").addEventListener("pointerover", (event) => {
  const index = event.target.dataset.index;
  if (index != null && statisticsData) $("stats-hover").textContent = statisticsIntervalText(statisticsData.points[+index], statisticsData.interval_s);
});

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  renderVram(m.vram);
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, m.conversation_cache || {});
  if (tab === "about") renderAbout(eng, hw, st);
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, sessionCache = {}) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = live.phase === "saving session" ? "Saving session" : live.phase === "restoring session" ? "Restoring session" : "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : expertCacheText(eng) || "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = eng.expert_cache_live_bytes ?? (eng.expert_cache_mib || 0) * 1048576;
  $("slots-label").textContent = eng.expert_cached_matrices != null ? "Expert matrices in VRAM" : "Experts in VRAM";
  $("slots-text").textContent = expertCacheText(eng) || "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  $("commit-section").hidden = hw.commit_limit == null;
  if (hw.commit_limit) {
    const used = 100 * hw.commit_used / hw.commit_limit;
    $("commit-text").textContent = `${gb(hw.commit_used)} / ${gb(hw.commit_limit)} GiB`;
    $("commit-bar").style.width = `${Math.min(100, used)}%`;
    $("commit-progress").dataset.tone = used > 95 ? "danger" : "";
    $("commit-note").textContent = `${gb(hw.commit_available)} GiB available for allocations. This is separate from free physical RAM.`;
  }
  const sessionBudget = (eng.conversation_cache_mib || 0) * 1048576;
  const sessionBytes = sessionCache.bytes || 0;
  $("session-cache-text").textContent = sessionBudget ?
    `${fmt(sessionBytes / 1073741824, 2)} / ${fmt(sessionBudget / 1073741824, 0)} GiB · ${fmt(sessionCache.sessions || 0)} saved + ${fmt(sessionCache.active_sessions || 0)} active` : "Off";
  $("session-cache-bar").style.width = sessionBudget ? `${Math.min(100, (sessionBytes + (sessionCache.retained_bytes || 0)) / sessionBudget * 100)}%` : "0%";
  $("session-cache-text").title = `RAM restores: ${sessionCache.hits || 0}; evictions: ${sessionCache.evictions || 0} (${sessionCache.pressure_evictions || 0} under memory pressure); expired: ${sessionCache.expired || 0}; duplicates replaced: ${sessionCache.deduplicated || 0}. Active session memory is separate.`;
  const cacheNote = cacheSaveNote(sessionCache);
  $("session-cache-note").textContent = sessionBudget ? cacheNote.text + (sessionCache.retained_bytes ?
    ` Another ${archiveBytes(sessionCache.retained_bytes)} retains the active session's KV for its next save, within the same budget.` : "") : "";
  $("session-cache-note").dataset.warning = String(cacheNote.warning);
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="8" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%`;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num" title="${esc(r.cache_source || '')}; save ${fmt(r.cache_save_ms, 1)} ms; restore ${fmt(r.cache_restore_ms, 1)} ms">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
// ------------------------------------------------------------------ Archive inspector
let archiveData = null, archiveReceived = 0, archiveTimer = null, archiveReleasing = false, archiveSerial = 0, archiveRows = "";
let archivePolicySaving = false;
function renderArchivePolicy() {
  const policy = archiveData?.auto_release;
  const disabled = !policy || !archiveData.supported || archiveData.stale || archivePolicySaving || archiveReleasing;
  $("archive-auto-release").disabled = disabled;
  $("archive-release-hours").disabled = disabled;
  if (policy && !archivePolicySaving) {
    $("archive-auto-release").checked = policy.enabled;
    if (document.activeElement !== $("archive-release-hours")) $("archive-release-hours").value = policy.hours;
  }
  const error = policy?.storage_error || policy?.last_error;
  const status = $("archive-policy-status");
  status.dataset.error = String(!!error);
  status.textContent = archivePolicySaving ? "Saving…" : error || (!policy ? "Auto-release settings unavailable." :
    policy.enabled ? `On · release after ${fmt(policy.hours, 2)} hours without use. Cleanup runs when idle or between requests.` :
    "Off · sessions are kept until released manually or evicted for memory limits.");
}
async function saveArchivePolicy() {
  if (archivePolicySaving || archiveReleasing || !archiveData?.auto_release || archiveData.stale) return;
  if (!$("archive-policy-form").reportValidity()) return;
  const request = {enabled: $("archive-auto-release").checked, hours: Number($("archive-release-hours").value)};
  archivePolicySaving = true;
  ++archiveSerial; clearTimeout(archiveTimer);
  renderArchive();
  try {
    const r = await fetch("/cache/settings", {method: "POST", headers: headers(true), body: JSON.stringify(request), signal: AbortSignal.timeout(20000)});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error?.message || `HTTP ${r.status}`);
    archiveData = data;
    archiveReceived = performance.now();
    toast("success", "Auto-release saved", request.enabled ? `Saved sessions expire after ${request.hours} hours without use.` : "Automatic session release is off.");
  } catch (e) {
    toast("error", "Could not save auto-release", e.message, 6000);
    archiveData.stale = true;
  } finally {
    archivePolicySaving = false;
    renderArchive();
    refreshArchive();
  }
}
$("archive-auto-release").addEventListener("change", saveArchivePolicy);
$("archive-release-hours").addEventListener("change", saveArchivePolicy);
$("archive-policy-form").addEventListener("submit", event => { event.preventDefault(); saveArchivePolicy(); });
function archiveAge(seconds) {
  if (seconds <= 1800) return "fresh";
  if (seconds <= 3600) return "warm";
  if (seconds <= 7200) return "cool";
  return "old";
}
function archiveBytes(bytes) {
  if (bytes == null) return "Reserved";
  return bytes >= 1073741824 ? `${fmt(bytes / 1073741824, 2)} GiB` : `${fmt(bytes / 1048576, 1)} MiB`;
}
function archiveStatus(text, error = false) {
  $("archive-status").textContent = text;
  $("archive-status").dataset.error = String(error);
}
function cacheSaveNote(cache) {
  const reason = cache.last_save_reason;
  const skipped = cache.skipped_saves || 0;
  const suffix = skipped ? ` Skipped saves since start: ${fmt(skipped)}.` : "";
  if (!reason || reason === "never") return {warning: false,
    text: "Archive RAM counts saved sessions only. A continuing session stays active; a snapshot is saved when switching or rewinding."};
  const when = cache.last_save_at_ms ? new Date(cache.last_save_at_ms).toLocaleString() : "";
  if (reason === "saved") return {warning: false,
    text: `Last save: ${when} · ${archiveBytes(cache.last_save_bytes)}.${suffix}`};
  const causes = {commit_limit: "Windows commit limit", physical_ram: "free physical RAM reserve",
    memory_unknown: "memory information unavailable", free_ram: "memory headroom", budget: "archive size limit",
    idle_expired: "idle timeout", invalid_state: "snapshot state could not be validated", allocation: "memory allocation failed"};
  const memory = reason === "commit_limit" || reason === "physical_ram";
  let detail = ` Snapshot: ${archiveBytes(cache.last_save_bytes)}.`;
  if (memory && cache.last_save_commit >= 0) detail += ` Available commit: ${archiveBytes(cache.last_save_commit)}.`;
  if (memory && cache.last_save_physical >= 0) detail += ` Free physical RAM: ${archiveBytes(cache.last_save_physical)}.`;
  return {warning: true, text: `Last save skipped (${when}): ${causes[reason] || reason}.${detail} A safety reserve is also required.${suffix}`};
}
function renderArchive() {
  const d = archiveData;
  if (!d) return;
  renderArchivePolicy();
  const entries = (d.entries || []).filter(e => e.kind === "session" || e.kind === "active");
  const saved = entries.filter(e => e.kind === "session").length;
  const active = entries.filter(e => e.kind === "active").length;
  $("archive-summary").textContent = `${saved} saved + ${active} active · ${archiveBytes(d.archive_bytes)} / ${archiveBytes(d.budget_bytes)} archive RAM` +
    (d.retained_bytes ? ` · ${archiveBytes(d.retained_bytes)} retained active KV within this budget` : "");
  const note = cacheSaveNote(d.diagnostics || {});
  $("archive-save-note").textContent = note.text;
  $("archive-save-note").dataset.warning = String(note.warning);
  if (!d.supported) archiveStatus("This engine does not support the archive inspector. Start the updated engine.", true);
  else if (!d.alive) archiveStatus("The engine stopped. These entries are no longer available.", true);
  else if (d.stale) archiveStatus("Unable to refresh. Showing the last snapshot; release is disabled.", true);
  else if (archiveReleasing) archiveStatus("Releasing the selected session…");
  else if (d.busy) archiveStatus("A request is running or queued. The list shows the last completed state; release is available when idle.");
  else archiveStatus("Updates every 3 seconds. Times show when each session was last used.");
  const now = d.time_ms + performance.now() - archiveReceived;
  const focused = document.activeElement?.dataset?.cacheRelease;
  const rows = entries.map(e => {
    const seconds = Math.max(0, (now - e.last_used_ms) / 1000);
    const age = archiveAge(seconds);
    const ago = seconds < 60 ? "Just now" : seconds < 3600 ? `${fmt(Math.floor(seconds / 60))} min ago` : `${fmt(seconds / 3600, 1)} hours ago`;
    const title = e.kind === "active" ? "Active session" : "Saved session";
    const identity = e.session_id ? `Session ${e.session_id.slice(0, 12)}…` : "Matched by conversation history";
    const removable = e.kind === "session" && e.deletable;
    const disabled = !removable || d.busy || !d.alive || d.stale || archiveReleasing || archivePolicySaving;
    return `<tr data-age="${age}">
      <td><strong>${title}</strong><span class="small muted" title="${esc(e.session_id || '')}">${esc(identity)} · ${esc(e.id)}</span></td>
      <td class="num">${fmt(e.tokens)}</td><td class="num">${archiveBytes(e.bytes)}</td>
      <td><span class="archive-age">${ago}</span><span class="small muted">${esc(new Date(e.last_used_ms).toLocaleString())}</span></td>
      <td>${removable ? `<button class="st-btn st-btn--danger" type="button" data-cache-release="${esc(e.id)}" ${disabled ? "disabled" : ""}>Release session</button>` : '<span class="small muted">Active session</span>'}</td></tr>`;
  }).join("") || `<tr><td colspan="5" class="muted">${d.supported ? "No cached sessions yet. They appear after requests are processed." : "Archive details unavailable."}</td></tr>`;
  if (archiveRows !== rows) {
    archiveRows = rows;
    $("archive-body").innerHTML = rows;
    if (focused) {
      const next = [...$("archive-body").querySelectorAll("[data-cache-release]")].find(button => button.dataset.cacheRelease === focused && !button.disabled);
      (next || $("archive-close")).focus({preventScroll: true});
    }
  }
}
async function refreshArchive() {
  clearTimeout(archiveTimer);
  if (!$("archive-dialog").open) return;
  const serial = ++archiveSerial;
  try {
    if (archiveReleasing || archivePolicySaving) return;
    const r = await fetch("/cache/entries", {headers: headers(), cache: "no-store", signal: AbortSignal.timeout(8000)});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (serial !== archiveSerial) return;
    archiveData = data;
    archiveReceived = performance.now();
    renderArchive();
  } catch (e) {
    if (serial !== archiveSerial) return;
    if (archiveData) { archiveData.stale = true; renderArchive(); }
    else { archiveStatus(`Could not load the archive: ${e.message}`, true); archiveRows = ""; $("archive-body").innerHTML = '<tr><td colspan="5">Archive unavailable</td></tr>'; }
  } finally {
    if (serial === archiveSerial && $("archive-dialog").open) archiveTimer = setTimeout(refreshArchive, 3000);
  }
}
$("session-cache-open").onclick = () => {
  $("archive-dialog").showModal();
  if (archiveData) { archiveData.stale = true; renderArchive(); }
  refreshArchive();
};
$("archive-close").onclick = () => $("archive-dialog").close();
$("archive-dialog").addEventListener("close", () => { clearTimeout(archiveTimer); ++archiveSerial; });
$("archive-body").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-cache-release]");
  if (!button || button.disabled || archiveReleasing || !archiveData) return;
  const request = {id: button.dataset.cacheRelease, generation: archiveData.generation};
  archiveReleasing = true;
  ++archiveSerial; clearTimeout(archiveTimer);
  renderArchive();
  try {
    const r = await fetch("/cache/release", {method: "POST", headers: headers(true), body: JSON.stringify(request), signal: AbortSignal.timeout(20000)});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error?.message || `HTTP ${r.status}`);
    archiveData = data;
    archiveReceived = performance.now();
    toast("success", "Session released", "Its RAM is now available for reuse.");
  } catch (e) {
    toast("error", "Could not release cache", e.message, 6000);
  } finally {
    archiveReleasing = false;
    renderArchive();
    refreshArchive();
  }
});

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function expertCacheText(eng) {
  const count = eng.expert_cached_matrices ?? eng.expert_slots;
  if (count == null) return null;
  const unit = eng.expert_cached_matrices != null ? "matrices" : "experts";
  const size = eng.expert_cache_live_bytes != null ? `${gb(eng.expert_cache_live_bytes)} GiB used` :
    eng.expert_cache_mib != null ? `${gb(eng.expert_cache_mib * 1048576)} GiB budget` : null;
  return `${fmt(count)} ${unit}${size ? ` · ${size}` : ""}`;
}
function expertPlacementText(eng) {
  const parts = [];
  if (eng.expert_compute) parts.push(`Compute: ${eng.expert_compute}`);
  if (eng.gpu_expert_layers != null) parts.push(`${fmt(eng.gpu_expert_layers)} resident expert layers`);
  if (eng.expert_storage) parts.push(`source: ${eng.expert_storage}`);
  return parts.length ? parts.join("; ") : null;
}
function speculationText(eng) {
  if (eng.speculative === "none" || eng.spec === 0) return "Off";
  if (eng.speculative === "dspark") return eng.spec != null ?
    `DSpark (experimental): up to ${eng.spec} draft tokens, verified by the main model` : "DSpark (experimental)";
  if (eng.speculative === "mtp") return "Native MTP" +
    (eng.draft_tokens != null ? `: up to ${eng.draft_tokens} draft tokens` : "");
  // The legacy engine's mtp_max is a verify-window size, including the main token.
  if (eng.mtp_max != null && eng.spec > 0) return `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}`;
  if (eng.speculative) return String(eng.speculative);
  return eng.spec > 0 ? "Enabled; draft strategy not reported" : null;
}
function expertPipelineText(eng) {
  if (eng.expert_pipeline == null) return null;
  if (!eng.expert_pipeline) return "Off";
  const parts = ["Enabled"];
  if (eng.expert_pipeline_slots != null) parts.push(`${eng.expert_pipeline_slots} staging slots per model`);
  if (eng.expert_readers != null) parts.push(`up to ${eng.expert_readers} readers`);
  if (eng.expert_read_mode) parts.push(`read mode: ${eng.expert_read_mode}`);
  return parts.join("; ");
}
function expertPolicyText(eng) {
  return eng.expert_cache_policy === "frequency" ? "Prefer frequently used matrices; recent use decays" :
    eng.expert_cache_policy === "lru" ? "Least recently used" : eng.expert_cache_policy || null;
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Architecture", eng.architecture],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    [eng.expert_cached_matrices != null ? "Expert matrices in VRAM" : "Experts in VRAM", expertCacheText(eng)],
    ["Expert placement", expertPlacementText(eng)],
    ["Speculation", speculationText(eng)],
    ["DSpark experts", eng.speculative === "dspark" ? `${eng.draft_gpu_expert_layers || 0} resident layers; ${gb((eng.draft_expert_cache_mib || 0) * 1048576)} GB GPU cache; other experts streamed from GGUF` : null],
    ["DSpark confidence filter", eng.speculative === "dspark" ? Number(eng.draft_min_confidence) > 0 ? `Keep draft prefix with confidence ≥ ${Number(eng.draft_min_confidence).toFixed(2)}` : "Off" : null],
    ["DSpark VRAM reserved at startup", eng.draft_vram_weights_bytes ? `${gb(eng.draft_vram_weights_bytes + (eng.draft_vram_context_bytes || 0) + (eng.draft_vram_compute_bytes || 0) + (eng.draft_vram_pipeline_bytes || 0) + (eng.draft_expert_cache_mib || 0) * 1048576, 2)} GB, including expert cache and pipeline budgets` : null],
    ["Expert transfer pipeline", expertPipelineText(eng)],
    ["DSpark shared compute buffer", eng.draft_shared_scratch ? `${gb(eng.draft_shared_scratch_saved_bytes, 2)} GB saved at startup` : null],
    ["Expert cache policy", expertPolicyText(eng)],
    ["Fixed expert cache replacement", eng.architecture === "deepseek4" ?
      eng.expert_cache_match_size === 1 ? "Prefer matching matrix sizes" : "Least recently used" : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : "not readable (NVML)"],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true};
let settings = {...DEFAULTS, clear_thinking: false, ...store.get("sampling", {})};
let messages = store.get("chat", []);
let chatSessionId = store.get("chatSessionId", null) || crypto.randomUUID();
store.set("chatSessionId", chatSessionId);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name}))})));
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!m.error) {
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  const reasoning = m.reasoning || "";
  const replay = reasoningCapabilities().replay_reasoning === true;
  if (!ran.length) return m.text || (replay && reasoning) ? [{role: "assistant", content: m.text || "",
    ...(replay && reasoning ? {reasoning_content: reasoning} : {})}] : [];
  const out = [];
  // Older saved tool records may lack reasoning offsets. Do not guess which
  // round their reasoning belongs to; retain the previous text/call history.
  const replayRounds = replay && ran.every((t, i) => Number.isInteger(t.rat) && t.rat >= 0 &&
    t.rat <= reasoning.length && (!i || t.rat >= ran[i - 1].rat));
  let pos = 0, rpos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    const rat = replayRounds ? calls[calls.length - 1].rat : 0;
    const thought = replayRounds ? reasoning.slice(rpos, rat).trim() : "";
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              ...(thought ? {reasoning_content: thought} : {}),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
    rpos = rat;
  }
  const rest = m.text.slice(pos).trim();
  const thought = replayRounds ? reasoning.slice(rpos).trim() : "";
  if (rest || thought) out.push({role: "assistant", content: rest, ...(thought ? {reasoning_content: thought} : {})});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: health.model, messages: apiMessages(), stream: true,
                ...reasoningRequest(settings)};
  if (settings.temperature > 0) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  } else {
    body.temperature = 0;
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: {...headers(true), "X-Strata-Session-Id": chatSessionId}, body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages, backupSessionId = chatSessionId;
  chatSessionId = crypto.randomUUID();
  store.set("chatSessionId", chatSessionId);
  messages = [];
  saveChat();
  renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; chatSessionId = backupSessionId; store.set("chatSessionId", chatSessionId); saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function reasoningCapabilities() {
  return health.reasoning || {efforts: ["none", "low", "medium", "high"], default: "high", clear_thinking: false};
}
function reasoningSettings(s) {
  const caps = reasoningCapabilities();
  return {...s, thinking: caps.efforts.includes(s.thinking) ? s.thinking : caps.default,
          clear_thinking: caps.clear_thinking && s.clear_thinking === true};
}
function reasoningRequest(s) {
  const normalized = reasoningSettings(s);
  const request = {reasoning_effort: normalized.thinking};
  if (reasoningCapabilities().clear_thinking) request.clear_thinking = normalized.clear_thinking;
  return request;
}
function applyReasoningCapabilities() {
  const caps = reasoningCapabilities();
  if (!store.get("sampling", null)) settings.thinking = caps.default;
  settings = reasoningSettings(settings);
  for (const button of $("s-thinking").children) button.hidden = !caps.efforts.includes(button.dataset.v);
  $("clear-thinking-row").hidden = !caps.clear_thinking;
  loadDrawer();
}
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
function loadDrawer(s = settings) {
  s = reasoningSettings(s);
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-clear-thinking").setAttribute("aria-checked", String(s.clear_thinking));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {...reasoningRequest(s), temperature: +s.temperature};
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  const caps = reasoningCapabilities();
  $("o-thinking").textContent = sel ? (caps.efforts.includes("none") ?
    {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] :
    `reasoning always on${sel.dataset.v === caps.default ? " · default" : ""}`) : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-clear-thinking").onclick = () => $("s-clear-thinking").setAttribute("aria-checked", String($("s-clear-thinking").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer({...DEFAULTS, thinking: reasoningCapabilities().default, clear_thinking: false});
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              clear_thinking: reasoningCapabilities().clear_thinking && $("s-clear-thinking").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();
