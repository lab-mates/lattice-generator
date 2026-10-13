// Runs Python (Pyodide + latticegen) off the main thread.
// in : {type: "init", base}  |  {type: "run", params}
// out: {type: "status"|"log"|"ready"|"result"|"error", ...}
import { registerManifold } from "./manifold-backend.js";

let py = null;
let bridge = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const log = (text) => post({ type: "log", text });

async function init(base) {
  post({ type: "status", text: "Starting Python…" });
  const pyBase = new URL("pyodide/", base).href;
  const { loadPyodide } = await import(/* @vite-ignore */ pyBase + "pyodide.mjs");
  // packageBaseUrl pins wheels to the local copy (no CDN fallback, works offline)
  py = await loadPyodide({ indexURL: pyBase, packageBaseUrl: pyBase, stdout: log, stderr: log });

  post({ type: "status", text: "Loading numpy, scipy and scikit-image…" });
  const pkgs = await (await fetch(pyBase + "packages.json")).json();
  await py.loadPackage(pkgs, { messageCallback: () => {} });

  post({ type: "status", text: "Loading latticegen…" });
  await registerManifold(py, new URL("manifold/manifold.js", base).href);
  const zip = await fetch(new URL("latticegen.zip", base), { cache: "no-cache" });
  py.unpackArchive(await zip.arrayBuffer(), "zip");
  bridge = py.pyimport("bridge");
  post({ type: "ready", options: JSON.parse(bridge.options()) });
}

function run(params) {
  const r = bridge.run(JSON.stringify(params), log);
  const verts = r.pop("verts");
  const faces = r.pop("faces");
  const meta = r.toJs({ dict_converter: Object.fromEntries });
  r.destroy();
  const vb = verts.getBuffer("f32");
  const fb = faces.getBuffer("u32");
  const v = vb.data.slice();
  const f = fb.data.slice();
  vb.release(); fb.release(); verts.destroy(); faces.destroy();
  const stl = py.FS.readFile(meta.stl_path);
  py.FS.unlink(meta.stl_path);
  post({ type: "result", meta, verts: v, faces: f, stl }, [v.buffer, f.buffer, stl.buffer]);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init(data.base);
    else if (data.type === "run") run(data.params);
  } catch (e) {
    const text = String(e && e.message ? e.message : e).trim();
    post({ type: "error", text: text.split("\n").filter(Boolean).pop(), detail: text });
  }
};
