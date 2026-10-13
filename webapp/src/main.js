import { Viewer } from "./viewer.js";
import { estimate, recommendGrid, LARGE_FACES } from "./estimate.js";
import { PyodideEngine } from "./engine-pyodide.js";
import { NativeEngine } from "./engine-native.js";

const TOPOLOGIES = {
  sc: "Simple cubic", bcc: "Body-centred cubic (bcc)", fcc: "Face-centred cubic (fcc)",
  bccz: "bcc with vertical posts", fccz: "fcc with vertical posts", fbcc: "fcc and bcc combined",
  octet: "Octet truss", diamond: "Diamond", kelvin: "Kelvin foam cell",
};
const SURFACES = {
  gyroid: "Gyroid", primitive: "Schwarz P", diamond: "Schwarz D", iwp: "Schoen I-WP",
  neovius: "Neovius", lidinoid: "Lidinoid", fischer_koch_s: "Fischer–Koch S", split_p: "Split P",
};
const METHOD_HINTS = {
  struts: "Straight beams of exact thickness. Fastest, smallest files, best for big blocks.",
  sdf: "Beams with smoothly filleted joints, traced from a 3D grid.",
  fourier: "Soft, organic beams that swell gently at the joints.",
  tpms: "Smooth continuous surfaces with no joints at all. Open pores drain well.",
};
const PRESETS = {
  octet20: { method: "struts", topology: "octet", a: 1, density: "uniform", rho: 20,
             nx: 20, ny: 20, nz: 20, sides: 6, node: "union" },
  gyroid: { method: "tpms", topology: "gyroid", sheet: true, a: 2, density: "uniform", rho: 25,
            nx: 2, ny: 2, nz: 2 },
  graded: { method: "sdf", topology: "bcc", a: 2, density: "graded", rho0: 15, rho1: 40,
            nx: 2, ny: 2, nz: 4, smooth: 2 },
  smooth: { method: "sdf", topology: "octet", a: 2, density: "uniform", rho: 30,
            nx: 2, ny: 2, nz: 2, smooth: 2 },
};
const STORE_KEY = "latticegen-form-v2";   // v2: 12 sides, no joint slimming by default
const CHECKBOXES = ["sheet", "check", "islands", "instanced", "repair"];

const $ = (id) => document.getElementById(id);
const form = $("form");
const el = (name) => form.elements.namedItem(name);
const viewer = new Viewer($("plate"));

let engine = null;
let ready = false;
let running = false;
let available = null;          // names the Python side reports
let result = null;             // {save(), sizeMB}
let timer = null;
let logLines = [];

// ------------------------------------------------------------------ theme
function applyGridTheme() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  viewer.setGridColors(dark ? 0x5d6672 : 0x8b95a1, dark ? 0x353c45 : 0xc3c9d0);
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyGridTheme);
applyGridTheme();
viewer.reset();

// ------------------------------------------------------------------ engine
// The desktop app opens this page as ?engine=native&token=...; the web
// version runs Python in the browser.
const query = new URLSearchParams(location.search);
const handlers = {
  status: (text) => setStatus(text, true),
  log: (text) => appendLog(text),
  ready: (options) => {
    ready = true;
    available = { struts: options.topologies.map((t) => t.name), tpms: options.tpms };
    const full = !!options.capabilities?.pymeshlab;
    form.dataset.engine = full ? "native" : "web";
    if (engine.native) $("tagline").textContent = "Periodic lattices for resin printing.";
    fillTopologies();
    setStatus(`Ready. latticegen ${options.version}` + (full ? " with mesh simplification" : ""));
    updateForm();
  },
  result: (res) => {
    const { meta, verts, faces } = res;
    finishRun();
    viewer.show(verts, faces);
    $("empty").hidden = true;
    result = res;
    const [x, y, z] = meta.extents;
    $("dims").textContent = `${fmt(x)} × ${fmt(y)} × ${fmt(z)} mm`;
    $("download").hidden = false;
    $("download").textContent = `${engine.native ? "Save" : "Download"} STL (${res.sizeMB.toFixed(1)} MB)`;
    const t = meta.timings.total ?? 0;
    setStatus(`Done in ${t.toFixed(1)} s: ${meta.n_faces.toLocaleString()} triangles` +
              (meta.preview_partial ? ". Showing one corner; the file has the whole block." : ""));
    $("tab-mesh").textContent = meta.report;
    $("tab-dlp").textContent = meta.dlp ||
      "The printability check was off for this run. Turn it on under More settings.";
    showTab(meta.dlp ? "dlp" : "mesh");
  },
  error: ({ text, detail, fatal }) => {
    if (detail) appendLog(detail);
    fail(friendlyError(text), fatal);
  },
};

function friendlyError(text) {
  if (/MemoryError|Aborted/.test(text))
    return engine.native
      ? "The computer ran out of memory. Use fewer cells or less detail."
      : "The browser ran out of memory. Use fewer cells, less detail, or the Beams shape.";
  if (/stopped|did not answer|Lost contact/.test(text)) return text;
  return `Could not build the lattice: ${text.replace(/^\w+Error:\s*/, "")}`;
}

function fail(message, fatal = false) {
  finishRun();
  setStatus(message, false, true);
  showTab("log");
  // a crashed in-browser interpreter cannot be trusted: start a fresh one
  if (fatal && !engine.native) { ready = false; engine.cancel(); updateButtons(); }
}

// ------------------------------------------------------------------ form
function method() { return el("method").value; }

function fillTopologies() {
  const sel = $("topology");
  const isTpms = method() === "tpms";
  const names = isTpms ? SURFACES : TOPOLOGIES;
  const allowed = available ? available[isTpms ? "tpms" : "struts"] : Object.keys(names);
  const keep = sel.value;
  sel.replaceChildren(...allowed.map((n) => new Option(names[n] || n, n)));
  sel.value = allowed.includes(keep) ? keep : (isTpms ? "gyroid" : "octet");
  $("topology-label").textContent = isTpms ? "Surface" : "Lattice";
}

function num(name) { return parseFloat(el(name).value); }

function readParams() {
  const m = method();
  const general = $("general-cell").open;
  const p = {
    method: m, topology: el("topology").value,
    a: num("a"),
    cell: general ? ["ca", "cb", "cc", "calpha", "cbeta", "cgamma"].map(num) : null,
    cells: ["nx", "ny", "nz"].map((k) => parseInt(el(k).value, 10)),
    rho: num("rho") / 100,
    graded: el("density").value === "graded" && m !== "struts" ? [num("rho0") / 100, num("rho1") / 100] : null,
    sides: parseInt(el("sides").value, 10), node: el("node").value,
    smooth: num("smooth") / 100, node_comp: num("node_comp"),
    sheet: el("sheet").checked, N: parseInt(el("N").value, 10) || 0,
    pixel: num("pixel"), layer: num("layer"), check: el("check").checked,
    islands: el("islands").checked,
  };
  if (form.dataset.engine === "native") {
    const field = m !== "struts";
    p.repair = el("repair").checked;
    p.instanced = field && el("instanced").checked && !p.graded;
    p.edge_length = field ? num("edge_length") || 0 : 0;
    if (p.instanced) {
      const aMin = Math.min(...(p.cell ? p.cell.slice(0, 3) : [p.a]));
      p.seam_spacing = aMin / (parseInt(el("seam_points").value, 10) || 50);
      p.remesh = el("remesh").value;
    }
  }
  return p;
}

function validate(p) {
  const bad = [];
  if (!(p.a > 0) && !p.cell) bad.push("cell size");
  if (p.cell && p.cell.some((x) => !(x > 0))) bad.push("non-cubic cell");
  if (p.cells.some((n) => !(n >= 1))) bad.push("number of cells");
  const rhos = p.graded || [p.rho];
  if (rhos.some((r) => !(r > 0 && r < 0.95))) bad.push("solid fraction (1 to 90 %)");
  if (!(p.pixel > 0) || !(p.layer > 0)) bad.push("printer pixel or layer");
  if (p.instanced) {
    // seam points must be at least one grid voxel apart
    const n = p.N || recommendGrid(Math.min(...(p.cell ? p.cell.slice(0, 3) : [p.a])), p.pixel, true);
    const pts = parseInt(el("seam_points").value, 10);
    if (!(pts >= 4 && pts <= n)) bad.push(`seam points (4 to ${n}, the grid detail)`);
  }
  return bad;
}

/** native Python handles ~4x more than a browser tab before memory gets tight */
function isLarge(e) { return e.faces > (engine?.native ? 4 * LARGE_FACES : LARGE_FACES); }

function fmt(x) { return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2); }
function compact(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)} million` : n >= 1e3 ? `${Math.round(n / 1e3)} thousand`
    : `${Math.round(n)}`;
}

function updateForm() {
  const m = method();
  form.dataset.method = m;
  form.dataset.density = el("density").value;
  const native = form.dataset.engine === "native";
  const instanced = native && m !== "struts" && el("instanced").checked;
  form.dataset.instanced = String(instanced);
  const gradedRadio = form.querySelector('input[name=density][value=graded]');
  gradedRadio.disabled = m === "struts" || instanced;
  if (gradedRadio.disabled && el("density").value === "graded") {
    form.querySelector('input[name=density][value=uniform]').checked = true;
    form.dataset.density = "uniform";
  }
  $("graded-option").title = m === "struts" ? "Graded density needs one of the other shapes"
    : instanced ? "Repeating one cell needs a uniform density" : "";
  $("method-hint").textContent = METHOD_HINTS[m];
  $("a-field").classList.toggle("muted", $("general-cell").open);

  const p = readParams();
  const bad = validate(p);
  const est = $("estimate");
  if (bad.length) {
    est.textContent = `Check the ${bad.join(", ")}.`;
    est.className = "estimate warn";
  } else {
    const e = estimate(p);
    const aMin = Math.min(...(p.cell ? p.cell.slice(0, 3) : [p.a]));
    $("n-hint").textContent = `Blank picks ${recommendGrid(aMin, p.pixel, p.instanced)} for a ${p.pixel} mm pixel` +
      (p.instanced ? " (finer when repeating one cell: only that cell is traced)." : ". More detail means bigger files and longer runs.");
    const lighter = p.instanced || p.edge_length > 0;
    est.textContent = `About ${e.size.map(fmt).join(" × ")} mm, ~${compact(e.faces)} triangles, ` +
      `~${e.stlMB < 10 ? e.stlMB.toFixed(1) : Math.round(e.stlMB)} MB STL` +
      (lighter ? " before simplification." : ".") +
      (isLarge(e) ? (native ? " This is large: expect a long run and a big file."
                            : " This is large for a browser: expect a long run and possible memory errors.") : "");
    est.className = "estimate" + (isLarge(e) ? " warn" : "");
  }
  const seamMm = p.instanced ? p.seam_spacing : 0;
  $("edge-hint").textContent = "About one printer pixel is enough for printing. " + (seamMm
    ? `Blank uses the seam spacing (${+seamMm.toFixed(4)} mm).`
    : "Blank keeps every grid triangle.");
  updateButtons(bad.length > 0);
  save();
}

function updateButtons(invalid = validate(readParams()).length > 0) {
  $("generate").disabled = !ready || running || invalid;
  $("cancel").hidden = !running;
  $("generate").textContent = running ? "Generating…" : "Generate";
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  el("method").value = p.method;
  fillTopologies();
  $("general-cell").open = false;
  el("sheet").checked = !!p.sheet;
  for (const [k, v] of Object.entries(p)) {
    if (k === "method" || k === "sheet") continue;
    if (k === "density") el("density").value = v;
    else if (el(k)) el(k).value = v;
  }
  updateForm();
}

// ------------------------------------------------------------------ persistence
function save() {
  try {
    const data = Object.fromEntries(new FormData(form));
    for (const k of CHECKBOXES) data[k] = el(k).checked;
    data.general = $("general-cell").open;
    localStorage.setItem(STORE_KEY, JSON.stringify(data));
  } catch { /* storage unavailable: nothing to remember */ }
}

function restore() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(STORE_KEY)); } catch { /* ignore */ }
  if (!data) return;
  if (data.method) el("method").value = data.method;
  fillTopologies();
  for (const [k, v] of Object.entries(data)) {
    if (CHECKBOXES.includes(k)) el(k).checked = !!v;
    else if (k === "general") $("general-cell").open = !!v;
    else if (k === "method" || k === "preset") continue;
    else if (k === "density") el("density").value = v;
    else if (el(k) && typeof el(k).value === "string") el(k).value = v;
  }
}

// ------------------------------------------------------------------ run
function setStatus(text, busy = false, error = false) {
  $("status").textContent = text;
  $("status").className = "status" + (error ? " error" : "");
  $("progress").hidden = !busy;
}

function appendLog(text) {
  logLines.push(text);
  if (logLines.length > 2000) logLines = logLines.slice(-1500);
  const el = $("tab-log");
  el.textContent = logLines.join("\n");
  el.scrollTop = el.scrollHeight;
}

function finishRun() {
  running = false;
  clearInterval(timer);
  $("progress").hidden = true;
  updateButtons();
}

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (!ready || running) return;
  const p = readParams();
  if (validate(p).length) return;
  const e = estimate(p);
  if (isLarge(e) && !confirm(`This block has roughly ${compact(e.faces)} triangles ` +
      `(${Math.round(e.stlMB)} MB STL) before any simplification. It may take several minutes ` +
      "and can run out of memory. Generate anyway?")) return;
  running = true;
  logLines = [];
  appendLog(`run: ${JSON.stringify(p)}`);
  $("download").hidden = true;
  result = null;
  updateButtons();
  const t0 = performance.now();
  const tick = () => setStatus(`Generating… ${Math.round((performance.now() - t0) / 1000)} s`, true);
  tick();
  timer = setInterval(tick, 1000);
  engine.run(p);
});

$("cancel").addEventListener("click", () => {
  finishRun();
  appendLog("cancelled");
  ready = engine.native;                // the web engine restarts its interpreter
  engine.cancel();
  updateButtons();
});

$("download").addEventListener("click", async () => {
  if (!result) return;
  const r = await result.save();
  if (r?.saved && engine.native) setStatus(`Saved ${r.saved}`);
  else if (r?.error) setStatus(r.error, false, true);
});

// ------------------------------------------------------------------ tabs
function showTab(name) {
  for (const b of document.querySelectorAll("[role=tab]"))
    b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const t of ["dlp", "mesh", "log"]) $(`tab-${t}`).hidden = t !== name;
}
for (const b of document.querySelectorAll("[role=tab]"))
  b.addEventListener("click", () => showTab(b.dataset.tab));

// ------------------------------------------------------------------ view options
const COLORS = [
  ["Amber resin", 0xe3a008], ["Grey", 0x9aa3ad], ["White", 0xf2f2ee], ["Sky blue", 0x4a90d9],
  ["Green", 0x3fa66b], ["Red", 0xd0443e], ["Violet", 0x8a63d2], ["Charcoal", 0x3a3f46],
];
const VIEW_KEY = "latticegen-view-v1";
const view = { smooth: false, edges: false, color: COLORS[0][1] };
try { Object.assign(view, JSON.parse(localStorage.getItem(VIEW_KEY))); } catch { /* defaults */ }
if (!COLORS.some(([, c]) => c === view.color)) view.color = COLORS[0][1];
const hexCss = (c) => "#" + c.toString(16).padStart(6, "0");

function applyView() {
  viewer.setSmooth(view.smooth);
  viewer.setEdges(view.edges);
  viewer.setColor(view.color);
  $("view-smooth").checked = view.smooth;
  $("view-edges").checked = view.edges;
  $("view-color-dot").style.background = hexCss(view.color);
  for (const b of $("view-colors").children)
    b.setAttribute("aria-checked", String(+b.dataset.color === view.color));
  try { localStorage.setItem(VIEW_KEY, JSON.stringify(view)); } catch { /* not remembered */ }
}

function showColors(open) {
  $("view-colors").hidden = !open;
  $("view-color").setAttribute("aria-expanded", String(open));
}

$("view-colors").append(...COLORS.map(([name, c]) => {
  const b = document.createElement("button");
  b.type = "button";
  b.role = "radio";
  b.title = name;
  b.setAttribute("aria-label", name);
  b.dataset.color = c;
  b.style.background = hexCss(c);
  b.addEventListener("click", () => { view.color = c; applyView(); showColors(false); $("view-color").focus(); });
  return b;
}));
$("view-smooth").addEventListener("change", (ev) => { view.smooth = ev.target.checked; applyView(); });
$("view-edges").addEventListener("change", (ev) => { view.edges = ev.target.checked; applyView(); });
$("view-color").addEventListener("click", () => showColors($("view-colors").hidden));
document.addEventListener("click", (ev) => {
  if (!ev.target.closest(".swatch-menu")) showColors(false);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !$("view-colors").hidden) { showColors(false); $("view-color").focus(); }
});
applyView();

// ------------------------------------------------------------------ wiring
form.addEventListener("input", (ev) => {
  if (ev.target.name === "method") fillTopologies();
  if (ev.target.id !== "preset") $("preset").value = "";
  updateForm();
});
$("preset").addEventListener("change", (ev) => applyPreset(ev.target.value));
$("general-cell").addEventListener("toggle", updateForm);

fillTopologies();
restore();
engine = query.get("engine") === "native"
  ? new NativeEngine(handlers, query.get("token"))
  : new PyodideEngine(handlers);
updateForm();
engine.start();
