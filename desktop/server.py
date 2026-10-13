"""
Local engine for the desktop app: a small HTTP server on 127.0.0.1 that serves
the web front end and runs builds with native latticegen (pymeshlab included).

    GET  /api/info                    versions, topologies, capabilities
    POST /api/run          {params}   -> {"job": id}   (cancels a running job)
    GET  /api/job/<id>?since=n        -> {"state", "log": [...], "next", "meta"}
    GET  /api/job/<id>/preview        -> <u32 nv><u32 nf><f32 verts><u32 faces>
    GET  /api/job/<id>/file?fmt=stl   -> the mesh file; fmt=3mf converts on first request
    GET  /api/job/<id>/stl            -> same as file?fmt=stl
    POST /api/job/<id>/cancel

Every /api request needs the per-session token (header X-Token or ?token=).
Each build runs in its own process, so Cancel can stop it at once.
"""

from __future__ import annotations

import functools
import http.server
import io
import json
import multiprocessing as mp
import os
import secrets
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
from urllib.parse import parse_qs, urlparse

_MP = mp.get_context("spawn")           # same behaviour on Linux and Windows


# ---------------------------------------------------------------- job process
class _QueueWriter(io.TextIOBase):
    """stdout replacement in the job process: complete lines go to the queue."""

    def __init__(self, q):
        self.q, self.buf = q, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(("log", line))
        return len(s)

    def flush(self):
        if self.buf:
            self.q.put(("log", self.buf))
            self.buf = ""


def _job_main(params, job_dir, q):
    sys.stdout = sys.stderr = _QueueWriter(q)
    try:
        import bridge
        r = bridge.run(params, log=lambda s: q.put(("log", str(s))),
                       stl_path=os.path.join(job_dir, "lattice.stl"))
        verts, faces = r.pop("verts"), r.pop("faces")
        with open(os.path.join(job_dir, "preview.bin"), "wb") as fh:
            fh.write(struct.pack("<II", len(verts), len(faces)))
            fh.write(verts.tobytes())
            fh.write(faces.tobytes())
        sys.stdout.flush()
        q.put(("done", r))
    except BaseException as e:                       # report everything, incl. MemoryError
        sys.stdout.flush()
        q.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))


def _convert_main(src, dst, title, q):
    try:
        import bridge
        bridge.convert(src, dst, title)
        q.put(None)
    except BaseException as e:
        q.put(f"{type(e).__name__}: {e}")


FORMATS = {"stl": "model/stl", "3mf": "model/3mf"}


class Job:
    def __init__(self, params, root):
        self.id = uuid.uuid4().hex[:12]
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir)
        self.log, self.state, self.meta, self.error = [], "running", None, None
        self._convert_lock = threading.Lock()
        self.queue = _MP.Queue()
        self.proc = _MP.Process(target=_job_main, args=(params, self.dir, self.queue), daemon=True)
        self.proc.start()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        while True:
            try:
                msg = self.queue.get(timeout=0.5)
            except Exception:                         # queue.Empty, or closed after cancel
                if self.state != "running":
                    return
                if not self.proc.is_alive():
                    self._fail("The engine stopped unexpectedly. Large blocks can use up "
                               "the memory: try fewer cells or less detail. Details are in the Log.")
                    return
                continue
            kind = msg[0]
            if kind == "log":
                self.log.append(msg[1])
            elif kind == "done":
                self.meta = msg[1]
                self.state = "done"
                return
            elif kind == "error":
                self.log.append(msg[2])
                self._fail(msg[1])
                return

    def _fail(self, text):
        if self.state == "running":
            self.error, self.state = text, "error"

    def cancel(self):
        if self.state == "running":
            self.state = "cancelled"
            self.proc.terminate()
        self.proc.join(5)

    def path(self, name):
        return os.path.join(self.dir, name)

    def file(self, fmt="stl"):
        """Path of the finished mesh in format ``fmt`` ('stl' or '3mf').  The
        3MF is converted from the STL on first use, in a child process so
        that its memory goes back to the system."""
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}")
        path = self.path(f"lattice.{fmt}")
        part = self.path(f"partial.{fmt}")                   # the extension picks the writer
        with self._convert_lock:
            if not os.path.exists(path):
                q = _MP.Queue()
                title = os.path.splitext(self.meta["filename"])[0]
                p = _MP.Process(target=_convert_main, args=(self.path("lattice.stl"), part, title, q),
                                daemon=True)
                p.start()
                p.join()
                err = q.get() if not q.empty() else f"the converter stopped (exit {p.exitcode})"
                if err:
                    raise RuntimeError(err)
                os.replace(part, path)
        return path

    def filename(self, fmt="stl"):
        return os.path.splitext(self.meta["filename"])[0] + "." + fmt


# ---------------------------------------------------------------- server
class Engine:
    def __init__(self, web_root, port=0, host="127.0.0.1"):
        self.web_root = os.path.abspath(web_root)
        self.token = secrets.token_urlsafe(24)
        _remove_stale_tmp()
        self.tmp = tempfile.mkdtemp(prefix="latticegen-")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        handler = functools.partial(_Handler, self, directory=self.web_root)
        self.httpd = http.server.ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/?engine=native&token={self.token}"
        self._info = None

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.url

    def info(self):
        if self._info is None:
            import bridge
            self._info = json.loads(bridge.options())
        return self._info

    def run(self, params) -> Job:
        with self.lock:
            for old in list(self.jobs.values()):        # one job at a time
                old.cancel()
                shutil.rmtree(old.dir, ignore_errors=True)
            self.jobs.clear()
            job = Job(params, self.tmp)
            self.jobs[job.id] = job
            return job

    def job(self, job_id) -> Job | None:
        return self.jobs.get(job_id)

    def shutdown(self):
        for job in list(self.jobs.values()):
            job.cancel()
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def _remove_stale_tmp(max_age_h=24):
    """Delete job folders left by sessions that were killed (no cleanup ran)."""
    root = tempfile.gettempdir()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith("latticegen-") and os.path.isdir(path):
            try:
                if time.time() - os.path.getmtime(path) > max_age_h * 3600:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".json": "application/json"}

    def __init__(self, engine, *a, **kw):
        self.engine = engine
        super().__init__(*a, **kw)

    def log_message(self, *a):                          # no console in the windowed app
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---------------------------------------------------------- helpers
    def _authorised(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False                                # DNS-rebinding guard
        q = parse_qs(urlparse(self.path).query)
        token = self.headers.get("X-Token") or q.get("token", [""])[0]
        return secrets.compare_digest(token, self.engine.token)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype, download_name=None):
        if not os.path.exists(path):
            return self._json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def _route(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        return parts[1:] if parts and parts[0] == "api" else None

    # ---------------------------------------------------------- verbs
    def do_GET(self):
        route = self._route()
        if route is None:
            return super().do_GET()
        if not self._authorised():
            return self._json({"error": "forbidden"}, 403)
        if route == ["info"]:
            return self._json(self.engine.info())
        if len(route) >= 2 and route[0] == "job":
            job = self.engine.job(route[1])
            if job is None:
                return self._json({"error": "unknown job"}, 404)
            if len(route) == 2:
                since = int(parse_qs(urlparse(self.path).query).get("since", ["0"])[0])
                lines = job.log[since:]
                return self._json({"state": job.state, "log": lines, "next": since + len(lines),
                                   "meta": job.meta, "error": job.error})
            if route[2] == "preview" and job.state == "done":
                return self._file(job.path("preview.bin"), "application/octet-stream")
            if route[2] in ("stl", "file") and job.state == "done":
                fmt = parse_qs(urlparse(self.path).query).get("fmt", ["stl"])[0]
                if route[2] == "stl":
                    fmt = "stl"
                if fmt not in FORMATS:
                    return self._json({"error": "unknown format"}, 400)
                try:
                    path = job.file(fmt)
                except RuntimeError as e:
                    return self._json({"error": str(e)}, 500)
                return self._file(path, FORMATS[fmt], job.filename(fmt))
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        route = self._route()
        if route is None or not self._authorised():
            return self._json({"error": "forbidden"}, 403)
        if route == ["run"]:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                params = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad JSON"}, 400)
            return self._json({"job": self.engine.run(params).id})
        if len(route) == 3 and route[0] == "job" and route[2] == "cancel":
            job = self.engine.job(route[1])
            if job is None:
                return self._json({"error": "unknown job"}, 404)
            job.cancel()
            return self._json({"state": job.state})
        return self._json({"error": "not found"}, 404)
