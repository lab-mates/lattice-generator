# latticegen web app

Joaquin Ramirez-Rico, University of Seville — [mates-lab.es](https://mates-lab.es/)

A browser GUI for `latticegen`. The unchanged Python package runs in
WebAssembly ([Pyodide](https://pyodide.org) 314.0.7, Python 3.14) inside a
Web Worker. A Three.js view shows the result and the STL (or a 3MF, converted
on request) is downloaded from the page. Nothing is uploaded, and the built folder has no runtime internet
dependencies.

```
index.html, src/main.js   form, presets, estimates, tabs, download
src/viewer.js             Three.js build-plate view (mm grid, z up; flat or Gouraud shading,
                          triangle edges, lattice colour, set from the view bar)
src/engine-pyodide.js     web engine: Pyodide + latticegen in a Web Worker (src/worker.js)
src/engine-native.js      desktop engine: the local server of desktop/ (?engine=native&token=...)
src/estimate.js           face-count / STL-size estimate before a run
src/manifold-backend.js   Manifold's WebAssembly build as `manifold_js`, for strut-sized joints and plates
py/bridge.py              JSON params -> latticegen.app.build -> arrays + STL file
launcher/                 Windows offline launcher (PowerShell static server)
scripts/                  fetch-pyodide, pack-latticegen, smoke test, offline zip
```

The same page is the front end of the Windows desktop app (`../desktop/`).
There, the native engine adds the "Lighter mesh" options.

What it can build: struts (explicit beams), SDF, Fourier and TPMS, with uniform
or graded (z) density, general cells, solid plates on the top and/or bottom,
removal of floating slivers, and the DLP report; STL or 3MF files. Simplify, remesh, repair and instancing need pymeshlab, which does
not exist for WebAssembly, so they are in the CLI and the desktop app only.

## Build (WSL / Linux, Node >= 20)

```bash
cd webapp
npm install               # vite, three, pyodide, fflate (project-local)
npm run fetch-pyodide     # once: copies the runtime, downloads ~40 MB of wheels
npm run dev               # http://localhost:5173 (also opens from Windows)
npm run smoke             # headless check of every method under Pyodide
npm run smoke -- big      # ...plus a timed 20 x 20 x 20 octet block
npm run build             # -> dist/  (static site, ~52 MB)
npm run zip               # -> latticegen-app-1.0.0.zip (Windows offline bundle)
```

`npm run dev`, `build` and `smoke` re-zip `../latticegen/*.py` first, so
package changes are picked up automatically.
Native tests of the bridge: `python -m unittest tests.test_bridge` from the
repository root.

## Distribute

* **Web:** copy `dist/` to any static host: GitHub Pages, a university web
  server, or an S3 bucket. All URLs are relative, so a sub-path works. The
  server must send `.wasm` as `application/wasm`; most hosts do.
* **Offline zip:** colleagues unzip it and double-click
  `Start Lattice Generator.bat`. It runs `app/serve.ps1` (a PowerShell
  `HttpListener` on `localhost:8765`-`8799`, this computer only) and opens
  the default browser. There is no .exe, and nothing to install. The README.txt
  inside is the user guide.

## Measured (Chrome 155 on this machine)

| job | time |
|---|---|
| first start (Python + numpy/scipy/scikit-image) | ~10 s |
| 3 x 3 x 3 octet beams (20 k faces) | 1.7 s |
| gyroid sheet 2 mm, 2 x 2 x 2, N = 80 (1.0 M faces) | 10 s |
| 20 x 20 x 20 octet beams, 1 mm, 20 % (5.4 M faces, 269 MB STL) | 23 s |

The 20 x 20 x 20 block peaks at about 0.85 GB of WebAssembly memory, with the
DLP check included. Meshes above 3 M faces are previewed as a 3 x 3 x 3
corner, and the download always holds the whole block. Field-based shapes at
the pixel-derived grid grow quickly (about 15 N^2 faces per cell). The form
estimates the size and asks before runs over 6 M faces.
