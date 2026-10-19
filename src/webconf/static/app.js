// rpi-auto-gym web interface. No build step, no dependencies.
const $ = (sel) => document.querySelector(sel);
let schema = [];
let saved = {};
let savedPath = "";

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const type = res.headers.get("Content-Type") || "";
  const body = type.includes("json") ? await res.json() : await res.text();
  return { ok: res.ok, status: res.status, body };
}

function post(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });
}

function fmtTime(sec) {
  const s = Math.floor(sec), m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}
const title = (name) => name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// ---- tabs -------------------------------------------------------------------
let activeTab = "dashboard";
document.querySelectorAll("nav button").forEach((btn) =>
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${activeTab}`));
    if (activeTab === "logs") loadLogs();
  })
);

// ---- dashboard --------------------------------------------------------------
function setStatus(running) {
  const el = $("#status");
  el.textContent = running ? "app running" : "app not running";
  el.className = `pill ${running ? "ok" : "down"}`;
}

async function pollState() {
  const { ok, body } = await api("/api/state").catch(() => ({ ok: false, body: {} }));
  setStatus(ok);
  if (!ok) return;
  $("#exercise").textContent = body.active_exercise ? title(body.active_exercise) : "—";
  $("#reps").textContent = body.current_rep_count;
  $("#sets").textContent = body.active_sets;
  $("#elapsed").textContent = fmtTime(body.elapsed_seconds);
  $("#fps").textContent = body.fps;
  $("#pose").textContent = `Pose: ${body.pose_detected ? "detected" : "none"}`;
  $("#history").innerHTML = Object.entries(body.exercises)
    .map(([n, e]) => `<tr><td>${title(n)}</td><td>${e.sets} sets / ${e.total_reps} reps</td></tr>`)
    .join("");
}

function pollFrame() {
  if (activeTab !== "dashboard" || !$("#live").checked) return setTimeout(pollFrame, 500);
  const img = $("#frame");
  const next = new Image();
  next.onload = () => { img.src = next.src; $("#frame-msg").textContent = ""; setTimeout(pollFrame, 200); };
  next.onerror = () => { $("#frame-msg").textContent = "No camera frame (is the app running?)"; setTimeout(pollFrame, 2000); };
  next.src = `/api/frame.jpg?t=${Date.now()}`;
}

document.querySelectorAll("[data-action]").forEach((btn) =>
  btn.addEventListener("click", async () => {
    const action = btn.dataset.action;
    if ((action === "stop" || action === "restart") && !confirm(`${title(action)} the app?`)) return;
    const { ok, body } = await post(`/api/action/${action}`);
    if (!ok) alert(body.error || "Action failed");
  })
);

async function loadInfo() {
  const { body } = await api("/api/info");
  const links = body.web_urls.map((u) => `<a href="${u}">${u}</a>`).join("<br>");
  const rows = { "Web address": links, Host: `${body.hostname} (${body.ip})`, Version: body.version, Python: body.python,
    Platform: body.platform, Config: body.config_path, Log: body.log_path };
  $("#info").innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

async function loadHardware() {
  const { body } = await api("/api/hardware");
  const d = body.detected;
  const rows = {
    "Pi model": d.pi_model || "not a Pi",
    "Pi camera": d.picameras.join(", ") || "none",
    Webcams: d.webcams.map(([i, n]) => `${i}: ${n}`).join(", ") || "none",
    Hailo: d.hailo || "none",
    Coral: d.coral || "none",
  };
  $("#hw-detected").innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("#hw-problems").innerHTML = body.problems.map((p) => `<li>${p}</li>`).join("");
  $("#hw-recommended").textContent = Object.entries(body.recommended)
    .map(([k, v]) => `${k.replace("AUTOGYM_", "")}=${v}`).join(", ");
  $("#hw-presets").innerHTML = body.presets.map((p) =>
    `<div class="preset"><div>${p.name}<p class="muted">${p.description}</p></div>
     <button data-preset="${p.id}">Apply</button></div>`).join("");
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => applyPreset(b.dataset.preset)));
}

async function applyPreset(preset) {
  if (!confirm("Apply these settings and restart the app?")) return;
  const { ok, body } = await post("/api/hardware/apply", { preset, restart: true });
  if (!ok) return alert(body.error || Object.values(body.errors || {}).join("\n") || "Failed");
  await Promise.all([loadSettings(), loadHardware()]);
}
$("#hw-apply").addEventListener("click", () => applyPreset("recommended"));

// ---- settings ---------------------------------------------------------------
function inputFor(s, value) {
  let el;
  if (s.type === "choice") {
    el = document.createElement("select");
    s.choices.forEach((c) => el.add(new Option(c, c)));
  } else if (s.type === "bool") {
    el = document.createElement("input");
    el.type = "checkbox";
  } else {
    el = document.createElement("input");
    el.type = { int: "number", float: "number", color: "color", password: "password" }[s.type] || "text";
    if (s.type === "float") el.step = "any";
  }
  el.name = s.key;
  el.id = `f-${s.key}`;
  setValue(el, s, value);
  el.addEventListener("input", markChanged);
  el.addEventListener("change", markChanged);
  return el;
}

function setValue(el, s, value) {
  if (s.type === "bool") el.checked = ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
  else el.value = value;
}

function getValue(el, s) {
  return s.type === "bool" ? (el.checked ? "1" : "0") : el.value;
}

function markChanged() {
  schema.forEach((s) => {
    const el = document.getElementById(`f-${s.key}`);
    el.closest(".field").classList.toggle("changed", getValue(el, s) !== String(saved[s.key] ?? ""));
  });
}

async function loadSettings() {
  [{ body: schema }, { body: { values: saved, path: savedPath } }] = await Promise.all([api("/api/schema"), api("/api/config")]);
  const form = $("#settings-form");
  form.innerHTML = "";
  const sections = {};
  schema.forEach((s) => {
    if (!sections[s.section]) {
      const fs = document.createElement("fieldset");
      fs.innerHTML = `<legend>${s.section}</legend>`;
      form.appendChild(fs);
      sections[s.section] = fs;
    }
    const row = document.createElement("div");
    row.className = "field";
    row.innerHTML = `<label for="f-${s.key}">${s.label}<br><span class="key">${s.key}</span></label>`;
    row.appendChild(inputFor(s, saved[s.key] ?? s.default));
    if (s.help) row.insertAdjacentHTML("beforeend", `<div class="help">${s.help}</div>`);
    row.insertAdjacentHTML("beforeend", `<div class="err"></div>`);
    sections[s.section].appendChild(row);
  });
  $("#save-msg").textContent = `Saved to ${savedPath}`;
}

async function save(restart) {
  const values = {};
  schema.forEach((s) => (values[s.key] = getValue(document.getElementById(`f-${s.key}`), s)));
  document.querySelectorAll(".field .err").forEach((e) => (e.textContent = ""));
  const { ok, body } = await post("/api/config", { values, restart });
  if (!ok) {
    Object.entries(body.errors || {}).forEach(([k, msg]) => {
      const el = document.getElementById(`f-${k}`);
      if (el) el.closest(".field").querySelector(".err").textContent = msg;
    });
    $("#save-msg").textContent = "Not saved: fix the highlighted fields.";
    return;
  }
  await loadSettings();
  $("#save-msg").textContent = restart
    ? (body.restarted ? "Saved. App is restarting…" : "Saved, but the app isn't running to restart.")
    : "Saved. Restart the app to apply.";
}
$("#save").addEventListener("click", () => save(false));
$("#save-restart").addEventListener("click", () => save(true));

// ---- logs -------------------------------------------------------------------
async function loadLogs() {
  const { body } = await api("/api/logs?lines=500");
  const pre = $("#logs");
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
  pre.textContent = body;
  if (atBottom || $("#logs-follow").checked) pre.scrollTop = pre.scrollHeight;
}
$("#logs-refresh").addEventListener("click", loadLogs);
setInterval(() => activeTab === "logs" && $("#logs-follow").checked && loadLogs(), 2000);

// ---- start ------------------------------------------------------------------
loadInfo();
loadHardware();
loadSettings();
pollState();
setInterval(pollState, 1000);
pollFrame();
