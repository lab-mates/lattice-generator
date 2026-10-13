# Lattice Generator desktop app (Windows)

Joaquin Ramirez-Rico, University of Seville — [mates-lab.es](https://mates-lab.es/)

The web front end from `webapp/` in its own window, backed by native Python.
On top of the browser version, it can:

- simplify meshes to a target triangle size (`edge_length`);
- build one cell and repeat it with matching seams (`mesh_instanced`, isotropic or
  decimated interiors);
- repair meshes;
- handle bigger blocks, at native speed and with native memory.

```
LatticeGenerator.exe ── pywebview window (Microsoft WebView2) ── same HTML/Three.js page
        └─ desktop/server.py: HTTP on 127.0.0.1, per-session token,
           one child process per build (Cancel = terminate), STL or 3MF saved via a Save dialog
latticegen.exe       ── console: `latticegen selftest`, or the full latticegen CLI
```

Colleagues unzip `LatticeGenerator-win64-<version>.zip` anywhere and run
`LatticeGenerator\LatticeGenerator.exe`. The user manual,
`LatticeGenerator-User-Manual.pdf` (from `docs/manual/`), is in the same folder.
There's nothing to install: WebView2 and .NET ship with Windows 10/11. The exe is unsigned, so the first start shows
"Windows protected your PC": click **More info → Run anyway**.
If anything misbehaves on a PC, run `latticegen.exe selftest` in that folder from
a command prompt.

Windows marks every file extracted from a downloaded zip as "from the internet",
and .NET Framework (used by the window) refuses to load marked DLLs. So on each
start the app removes that mark from its own folder (`desktop/paths.py:unblock_bundle`).
If the window still cannot open, for example on a PC without WebView2, the app
runs in the default browser instead and shows a "click OK to quit" message.

When testing a build, run it from a copy marked like a download, not from the
build folder. To mark a copy, write `ZoneId=3` into each file's `Zone.Identifier`
stream.

## Develop (WSL)

```bash
python -m desktop --browser        # engine + UI in your browser (needs webapp/dist: npm run build)
python -m desktop --selftest       # every method + simplify / instanced / repair
python -m unittest tests.test_desktop_server
```

`--browser` mode needs no pywebview, so nothing extra is installed in WSL.

## Build the Windows app

One-time setup, in the Windows Miniconda:

```bat
conda create -n latticegen-build -c conda-forge --override-channels python=3.14 pip
C:\Users\jrr\miniconda3\envs\latticegen-build\python.exe -m pip install numpy scipy scikit-image pymeshlab pywebview pyinstaller
```

Use pip wheels, not conda's numpy/scipy: conda's builds bring Intel MKL, which is
several hundred MB.

Then, from WSL:

```bash
desktop/build/build.sh     # ~3 min -> desktop/dist/LatticeGenerator-win64-<version>.zip
```

`build.sh` does four things:

1. Builds the web front end.
2. Stages only the needed sources in `%LOCALAPPDATA%\latticegen-build`.
   Building from a `\\wsl$` path is slow and unreliable.
3. Runs `build_windows.ps1`, which runs PyInstaller with `latticegen.spec`,
   then the bundle's own self-test. It fails if any feature is broken.
4. Zips the result and prints the sizes.

### Size

| Part | Wheel | Bundled |
|---|---|---|
| pymeshlab | 182 MB | 25 MB |
| whole app | | 182 MB unzipped, 82 MB zip (incl. the 2.8 MB manual) |

manifold3d (beam-sized joints) adds 2 MB.

`prune_pymeshlab.py` keeps only what the four MeshLab plugins latticegen calls
(`filter_meshing`, `filter_select`, `filter_clean`, `filter_func`) and the Python
binding actually load. It finds them by following their DLL import tables.
If latticegen starts using a filter from another plugin, add that plugin to
`PLUGINS`. The self-test fails until you do. To find a filter's plugin, search
the plugin DLLs for its MeshLab name (e.g. "Conditional Face Selection" is in
`filter_func.dll`).

The rest of the bundle is scipy (~67 MB with its libraries), numpy (~26 MB)
and scikit-image (10 MB).

### Testing the window automatically

Start the exe with `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9223`
to drive the page over the Chrome DevTools protocol. Set
`LATTICEGEN_TEST_SAVE=<folder>` to make Save write there without the dialog.
Both are for tests only.
