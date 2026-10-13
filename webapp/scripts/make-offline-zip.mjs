// Package dist/ with the Windows launcher into latticegen-app-<version>.zip:
//   LatticeGenerator/Start Lattice Generator.bat
//   LatticeGenerator/README.txt
//   LatticeGenerator/app/...            (the built site + serve.ps1)
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { zipSync } from "fflate";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(ROOT, "dist");
const LAUNCHER = join(ROOT, "launcher");
const version = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")).version;
const STORED = /\.(whl|wasm|zip|png)$/;          // already compressed

function* walk(dir) {
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else yield p;
  }
}

const top = "LatticeGenerator/";
const files = {};
const add = (name, path) => {
  files[name] = [readFileSync(path), { level: STORED.test(name) ? 0 : 6 }];
};
for (const p of walk(DIST)) add(top + "app/" + relative(DIST, p).split("\\").join("/"), p);
add(top + "app/serve.ps1", join(LAUNCHER, "serve.ps1"));
add(top + "Start Lattice Generator.bat", join(LAUNCHER, "Start Lattice Generator.bat"));
add(top + "README.txt", join(LAUNCHER, "README.txt"));

const out = join(ROOT, `latticegen-app-${version}.zip`);
const zip = zipSync(files);
writeFileSync(out, zip);
console.log(`${relative(ROOT, out)}: ${Object.keys(files).length} files, ${(zip.length / 1e6).toFixed(1)} MB`);
