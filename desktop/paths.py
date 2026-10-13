"""Where things live, in a source checkout and in the PyInstaller bundle."""

import os
import sys

FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    BASE = sys._MEIPASS                              # one-folder bundle: _internal/
    WEB_ROOT = os.path.join(BASE, "web")
    ICON = os.path.join(BASE, "icon.ico")
else:
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEB_ROOT = os.path.join(REPO, "webapp", "dist")
    ICON = os.path.join(REPO, "desktop", "build", "icon.ico")
    for p in (REPO, os.path.join(REPO, "webapp", "py")):
        if p not in sys.path:
            sys.path.insert(0, p)


def fix_streams():
    """A windowed (console=False) exe has no stdout/stderr; print() would fail."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def unblock_bundle():
    """Remove Windows' "downloaded from the internet" mark from the app's files.

    Explorer copies the mark (a Zone.Identifier stream) from a downloaded zip to
    every extracted file, and .NET Framework, which pythonnet uses to open the
    window, refuses to load marked DLLs ("Failed to resolve
    Python.Runtime.Loader.Initialize").  Same as right-click > Properties >
    Unblock, applied to this app's own folder.  Returns the number unblocked.
    """
    if not (FROZEN and sys.platform == "win32"):
        return 0
    n = 0
    for root, _, files in os.walk(os.path.dirname(sys.executable)):
        for f in files:
            try:
                os.remove(os.path.join(root, f) + ":Zone.Identifier")
                n += 1
            except OSError:                 # not marked, or a read-only location
                pass
    return n
