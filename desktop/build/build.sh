#!/usr/bin/env bash
# Build the Windows desktop app from WSL:
#   web front end (npm) -> stage sources on the Windows side -> build_windows.ps1
# Needs the Windows conda env `latticegen-build` (see desktop/README.md).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WIN_WORK_W="$(cmd.exe /c 'echo %LOCALAPPDATA%' 2>/dev/null | tr -d '\r')\\latticegen-build"
WIN_WORK="$(wslpath "$WIN_WORK_W")"
VERSION="$(grep -oP '__version__ = "\K[^"]+' "$REPO/latticegen/__init__.py")"

echo "== web front end"
(cd "$REPO/webapp" && npm run build --silent)

echo "== staging sources in $WIN_WORK_W"
SRC="$WIN_WORK/src"
rm -rf "$SRC"
mkdir -p "$SRC/latticegen" "$SRC/desktop/build" "$SRC/webapp/py" "$SRC/webapp/dist"
cp "$REPO"/latticegen/*.py "$SRC/latticegen/"
cp "$REPO"/desktop/*.py "$SRC/desktop/"
cp "$REPO"/desktop/build/{latticegen.spec,prune_pymeshlab.py,launch.py,cli_entry.py,icon.ico} "$SRC/desktop/build/"
cp "$REPO/webapp/py/bridge.py" "$SRC/webapp/py/"
(cd "$REPO/webapp/dist" && tar cf - --exclude=./pyodide --exclude=./latticegen.zip .) | (cd "$SRC/webapp/dist" && tar xf -)

echo "== PyInstaller (Windows)"
status=0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$REPO/desktop/build/build_windows.ps1")" \
  -Work "$WIN_WORK_W" -Version "$VERSION" || status=$?

mkdir -p "$REPO/desktop/dist"
cp "$WIN_WORK/LatticeGenerator-win64-$VERSION.zip" "$REPO/desktop/dist/" 2>/dev/null || true
echo "== zip: desktop/dist/LatticeGenerator-win64-$VERSION.zip (self-test exit $status)"
exit $status
