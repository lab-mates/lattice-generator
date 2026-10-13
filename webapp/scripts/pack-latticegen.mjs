// Zip the latticegen package and the browser bridge into public/latticegen.zip
// (unpacked by the worker into Pyodide's home directory, which is on sys.path).
import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { zipSync } from "fflate";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PKG = join(ROOT, "..", "latticegen");

const files = { "bridge.py": readFileSync(join(ROOT, "py", "bridge.py")) };
for (const f of readdirSync(PKG).filter((f) => f.endsWith(".py")).sort())
  files[`latticegen/${f}`] = readFileSync(join(PKG, f));

writeFileSync(join(ROOT, "public", "latticegen.zip"), zipSync(files, { level: 9 }));
console.log(`public/latticegen.zip: ${Object.keys(files).length} files`);

// Manifold's WebAssembly build (strut-sized joints in the browser)
const MANIFOLD = join(ROOT, "node_modules", "manifold-3d");
mkdirSync(join(ROOT, "public", "manifold"), { recursive: true });
for (const f of ["manifold.js", "manifold.wasm"]) copyFileSync(join(MANIFOLD, f), join(ROOT, "public", "manifold", f));
console.log("public/manifold/: manifold.js, manifold.wasm");
