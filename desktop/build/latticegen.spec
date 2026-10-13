# -*- mode: python -*-
# PyInstaller spec: one folder with two programs sharing the same libraries,
#   LatticeGenerator.exe  windowed app (pywebview + local engine)
#   latticegen.exe        console: `latticegen selftest`, or the latticegen CLI
# pymeshlab is pruned to the plugins latticegen uses (prune_pymeshlab.py).
import os
import sys

SPEC_DIR = os.path.abspath(SPECPATH)
REPO = os.path.dirname(os.path.dirname(SPEC_DIR))
sys.path.insert(0, SPEC_DIR)
from prune_pymeshlab import filter_toc, keep_set   # noqa: E402

WEB = os.path.join(REPO, "webapp", "dist")
ICON = os.path.join(SPEC_DIR, "icon.ico")
keep = keep_set()

common = dict(
    pathex=[REPO, os.path.join(REPO, "webapp", "py")],
    hiddenimports=["bridge", "latticegen", "latticegen.app", "desktop.server", "desktop.window",
                   "desktop.selftest", "skimage.measure"],
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "polyscope", "PyQt5", "PySide6",
              "skimage.io", "skimage.data", "imageio", "tifffile"],
    noarchive=False,
)

gui = Analysis([os.path.join(SPEC_DIR, "launch.py")],
               datas=[(WEB, "web"), (ICON, ".")], **common)
cli = Analysis([os.path.join(SPEC_DIR, "cli_entry.py")], datas=[], **common)

for a in (gui, cli):
    a.binaries = filter_toc(a.binaries, keep)
    a.datas = [d for d in filter_toc(a.datas, keep)
               if not d[0].replace("\\", "/").startswith(("web/pyodide/", "web/latticegen.zip"))]

gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="LatticeGenerator",
              console=False, icon=ICON, upx=False)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="latticegen",
              console=True, icon=ICON, upx=False)

COLLECT(gui_exe, gui.binaries, gui.datas, cli_exe, cli.binaries, cli.datas,
        name="LatticeGenerator", upx=False)
