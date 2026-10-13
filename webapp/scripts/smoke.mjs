// Headless check that latticegen runs under the bundled Pyodide:
//   npm run smoke            small builds of every method
//   npm run smoke -- big     also time a 20x20x20 octet struts block
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PY = join(ROOT, "public", "pyodide");
const { loadPyodide } = await import(pathToFileURL(join(PY, "pyodide.mjs")).href);

let t0 = performance.now();
const py = await loadPyodide({ indexURL: PY + "/", stdout: () => {}, stderr: (s) => console.error(s) });
await py.loadPackage(JSON.parse(readFileSync(join(PY, "packages.json"), "utf8")), { messageCallback: () => {} });
const { registerManifold } = await import(pathToFileURL(join(ROOT, "src", "manifold-backend.js")).href);
await registerManifold(py, pathToFileURL(join(ROOT, "public", "manifold", "manifold.js")).href);
py.unpackArchive(new Uint8Array(readFileSync(join(ROOT, "public", "latticegen.zip"))), "zip");
const bridge = py.pyimport("bridge");
console.log(`startup ${((performance.now() - t0) / 1000).toFixed(1)} s`);
const opts = JSON.parse(bridge.options());
console.log(`latticegen ${opts.version}: ${opts.topologies.length} topologies, ${opts.tpms.length} TPMS`);

const cases = process.argv.includes("only-big") ? [] : [
  { topology: "octet", method: "struts", a: 1, rho: 0.2, cells: [2, 2, 2], check: true },
  { topology: "kelvin", method: "struts", a: 1, rho: 0.15, cells: [2, 2, 2], node: "round", sides: 8 },
  { topology: "bcc", method: "struts", a: 1, rho: 0.2, cells: [2, 2, 2], node: "hull" },
  { topology: "diamond", method: "struts", a: 1, rho: 0.2, cells: [2, 2, 2], sides: 8 },
  { topology: "octet", method: "sdf", a: 2, rho: 0.3, cells: [2, 2, 2], N: 32, smooth: 0.02, check: true },
  { topology: "bcc", method: "fourier", a: 2, rho: 0.25, cells: [2, 2, 2], N: 32, node_comp: 0.5 },
  { topology: "gyroid", method: "tpms", a: 3, rho: 0.25, cells: [2, 2, 2], N: 32, sheet: true },
  { topology: "bcc", method: "sdf", a: 2, cells: [2, 2, 3], N: 32, graded: [0.15, 0.4] },
  { topology: "octet", method: "sdf", cell: [2, 2, 2.5, 90, 90, 90], rho: 0.3, cells: [1, 1, 1], N: 32 },
];
if (process.argv.some((a) => a.endsWith("big")))
  cases.push({ topology: "octet", method: "struts", a: 1, rho: 0.2, cells: [20, 20, 20], check: !process.argv.includes("nocheck") });

let failed = 0;
for (const p of cases) {
  t0 = performance.now();
  const logs = [];
  try {
    const r = bridge.run(JSON.stringify(p), (s) => logs.push(s));
    const verts = r.pop("verts"), faces = r.pop("faces");
    const meta = r.toJs({ dict_converter: Object.fromEntries });
    const vb = verts.getBuffer("f32"), fb = faces.getBuffer("u32");
    const stl = py.FS.readFile(meta.stl_path);
    const nStl = new DataView(stl.buffer).getUint32(80, true);
    const ok = nStl === meta.n_faces && stl.length === 84 + 50 * nStl
      && fb.data.length % 3 === 0 && vb.data.length % 3 === 0 && /watertight True/.test(meta.report);
    if (!ok) failed++;
    console.log(`${ok ? "ok  " : "FAIL"} ${p.method.padEnd(7)} ${p.topology.padEnd(7)} `
      + `${meta.n_faces.toLocaleString().padStart(10)} faces  ${(stl.length / 1e6).toFixed(1).padStart(6)} MB  `
      + `${((performance.now() - t0) / 1000).toFixed(1)} s  ${JSON.stringify(meta.timings)}`
      + (meta.preview_partial ? `  (preview ${fb.data.length / 3} faces)` : ""));
    for (const l of logs.filter((l) => /warning|WARN|^\s+- /.test(l))) console.log("       " + l.trim());
    vb.release(); fb.release(); verts.destroy(); faces.destroy(); r.destroy();
    py.FS.unlink(meta.stl_path);
  } catch (e) {
    failed++;
    console.log(`FAIL ${p.method} ${p.topology}: ${e.message.trim().split("\n").pop()}`);
  }
}
console.log(`peak RSS ${(process.resourceUsage().maxRSS / 1024).toFixed(0)} MB, `
  + `wasm heap ${(py._module.HEAP8.length / 1e6).toFixed(0)} MB`);
process.exit(failed ? 1 : 0);
