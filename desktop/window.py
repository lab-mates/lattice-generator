"""The app window: pywebview on Microsoft WebView2 (built into Windows 10/11)."""

import os
import shutil

import webview

from .paths import ICON


class Api:
    """Functions the page can call as ``window.pywebview.api.<name>``."""

    def __init__(self, engine):
        self._engine = engine

    def save_stl(self, job_id, filename):
        job = self._engine.job(job_id)
        if job is None or job.state != "done":
            return {"error": "Nothing to save yet: generate a lattice first."}
        test_dir = os.environ.get("LATTICEGEN_TEST_SAVE")     # automated tests only
        if test_dir:
            path = os.path.join(test_dir, filename)
        else:
            res = webview.windows[0].create_file_dialog(
                webview.FileDialog.SAVE, save_filename=filename,
                file_types=("STL files (*.stl)", "All files (*.*)"))
            if not res:
                return {"cancelled": True}
            path = res if isinstance(res, str) else res[0]
            if not path.lower().endswith(".stl"):
                path += ".stl"
        try:
            shutil.copyfile(job.path("lattice.stl"), path)
        except OSError as e:
            return {"error": f"Could not save {path}: {e.strerror}"}
        return {"saved": path}


def open_window(engine, url):
    webview.create_window("Lattice Generator", url, js_api=Api(engine),
                          width=1400, height=900, min_size=(900, 600))
    webview.start(icon=ICON if os.path.exists(ICON) else None)
