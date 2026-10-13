// Copy the Pyodide runtime into public/pyodide/ and download only the wheels
// latticegen needs (numpy, scipy, scikit-image and their dependencies), so the
// built app is self-contained and runs offline.  Re-running skips files whose
// sha256 already matches.
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(ROOT, "node_modules", "pyodide");
const DEST = join(ROOT, "public", "pyodide");
export const REQUIRED = ["numpy", "scipy", "scikit-image"];

const version = JSON.parse(readFileSync(join(SRC, "package.json"), "utf8")).version;
const CDN = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
const CORE = ["pyodide.mjs", "pyodide.js", "pyodide.asm.mjs", "pyodide.asm.wasm",
              "python_stdlib.zip", "pyodide-lock.json"];

const norm = (name) => name.toLowerCase().replace(/[-_.]+/g, "-");
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

mkdirSync(DEST, { recursive: true });
for (const f of CORE) copyFileSync(join(SRC, f), join(DEST, f));

const lock = JSON.parse(readFileSync(join(SRC, "pyodide-lock.json"), "utf8"));
const pkgs = {};
for (const [key, p] of Object.entries(lock.packages)) pkgs[norm(key)] = p;

const needed = new Set();
const stack = REQUIRED.map(norm);
while (stack.length) {
  const n = stack.pop();
  if (needed.has(n)) continue;
  if (!pkgs[n]) throw new Error(`package ${n} not in pyodide-lock.json`);
  needed.add(n);
  stack.push(...pkgs[n].depends.map(norm));
}

let total = 0;
for (const n of [...needed].sort()) {
  const p = pkgs[n];
  const path = join(DEST, p.file_name);
  let buf = existsSync(path) ? readFileSync(path) : null;
  if (!buf || sha256(buf) !== p.sha256) {
    const res = await fetch(CDN + p.file_name);
    if (!res.ok) throw new Error(`${p.file_name}: HTTP ${res.status}`);
    buf = Buffer.from(await res.arrayBuffer());
    if (sha256(buf) !== p.sha256) throw new Error(`${p.file_name}: sha256 mismatch`);
    writeFileSync(path, buf);
    console.log(`downloaded ${p.file_name} (${(buf.length / 1e6).toFixed(1)} MB)`);
  }
  total += buf.length;
}
writeFileSync(join(DEST, "packages.json"), JSON.stringify([...needed].sort()));
console.log(`pyodide ${version}: ${needed.size} packages, ${(total / 1e6).toFixed(1)} MB in public/pyodide/`);
