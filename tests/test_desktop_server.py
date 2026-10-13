"""The desktop app's local engine (desktop/server.py), driven over HTTP."""

import json
import os
import struct
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "webapp", "py"))

from desktop.server import Engine           # noqa: E402


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = tempfile.mkdtemp()
        with open(os.path.join(cls.web, "index.html"), "w") as fh:
            fh.write("<!doctype html><title>t</title>")
        cls.engine = Engine(cls.web)
        cls.engine.start()
        cls.base = f"http://127.0.0.1:{cls.engine.port}/"

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def req(self, path, data=None, token=True):
        r = urllib.request.Request(self.base + path, method="POST" if data is not None else "GET",
                                   data=json.dumps(data).encode() if data is not None else None)
        if token:
            r.add_header("X-Token", self.engine.token)
        with urllib.request.urlopen(r, timeout=60) as res:
            body = res.read()
            return json.loads(body) if res.headers["Content-Type"] == "application/json" else body

    def wait(self, job, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.req(f"api/job/{job}")
            if s["state"] != "running":
                return s
            time.sleep(0.2)
        self.fail("job did not finish")

    def test_static_and_token(self):
        self.assertIn(b"<title>t</title>", self.req("index.html", token=False))
        for path, data in (("api/info", None), ("api/run", {})):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.req(path, data, token=False)
            self.assertEqual(cm.exception.code, 403)
            cm.exception.close()

    def test_info(self):
        info = self.req("api/info")
        self.assertIn("octet", [t["name"] for t in info["topologies"]])
        self.assertIn("pymeshlab", info["capabilities"])

    def test_struts_job(self):
        job = self.req("api/run", {"topology": "octet", "method": "struts", "a": 1, "rho": 0.2,
                                   "cells": [2, 2, 2], "check": True})["job"]
        s = self.wait(job)
        self.assertEqual(s["state"], "done", s.get("error"))
        self.assertTrue(any("mesh:" in l for l in self.req(f"api/job/{job}?since=0")["log"]))
        prev = self.req(f"api/job/{job}/preview")
        nv, nf = struct.unpack("<II", prev[:8])
        self.assertEqual(len(prev), 8 + 12 * nv + 12 * nf)
        self.assertEqual(nf, s["meta"]["n_faces"])
        stl = self.req(f"api/job/{job}/stl?token={self.engine.token}", token=False)
        self.assertEqual(struct.unpack("<I", stl[80:84])[0], nf)

    def test_plates_and_3mf(self):
        job = self.req("api/run", {"topology": "bcc", "method": "struts", "a": 1, "rho": 0.2,
                                   "cells": [2, 2, 2], "plates": "both", "plate_thickness": 0.5,
                                   "plate_size": 1.2})["job"]
        s = self.wait(job)
        self.assertEqual(s["state"], "done", s.get("error"))
        self.assertIn("plates: both", s["meta"]["report"])
        self.assertAlmostEqual(s["meta"]["extents"][2], 3.0, places=5)
        z = self.req(f"api/job/{job}/file?fmt=3mf")
        self.assertEqual(z[:2], b"PK")
        import io
        import zipfile
        model = zipfile.ZipFile(io.BytesIO(z)).read("3D/3dmodel.model").decode()
        self.assertIn('<metadata name="Title">bcc_2x2x2</metadata>', model)
        path = os.path.join(tempfile.mkdtemp(), "t.3mf")
        with open(path, "wb") as fh:
            fh.write(z)
        import latticegen as lg
        m = lg.LatticeMesh.load(path)
        self.assertEqual(m.n_faces, s["meta"]["n_faces"])
        self.assertTrue(m.is_watertight)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.req(f"api/job/{job}/file?fmt=obj")
        self.assertEqual(cm.exception.code, 400)
        cm.exception.close()

    @unittest.skipUnless(__import__("bridge").has_pymeshlab(), "needs pymeshlab")
    def test_instanced_simplified(self):
        job = self.req("api/run", {"topology": "octet", "method": "sdf", "a": 1, "rho": 0.25,
                                   "cells": [2, 2, 2], "N": 40, "instanced": True,
                                   "seam_spacing": 0.1, "edge_length": 0.1})["job"]
        s = self.wait(job, timeout=300)
        self.assertEqual(s["state"], "done", s.get("error"))
        self.assertIn("watertight True", s["meta"]["report"])

    def test_error_and_cancel(self):
        job = self.req("api/run", {"topology": "nope", "method": "sdf", "cells": [1, 1, 1]})["job"]
        s = self.wait(job)
        self.assertEqual(s["state"], "error")
        self.assertTrue(s["error"])

        job = self.req("api/run", {"topology": "octet", "method": "sdf", "a": 2, "rho": 0.3,
                                   "cells": [6, 6, 6], "N": 96})["job"]
        time.sleep(1.0)
        proc = self.engine.job(job).proc
        self.assertEqual(self.req(f"api/job/{job}/cancel", {})["state"], "cancelled")
        self.assertFalse(proc.is_alive())

    def test_stale_tmp_removed(self):
        from desktop.server import _remove_stale_tmp
        old = tempfile.mkdtemp(prefix="latticegen-")
        fresh = tempfile.mkdtemp(prefix="latticegen-")
        os.utime(old, (time.time() - 3 * 86400,) * 2)
        _remove_stale_tmp()
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(fresh))
        self.assertTrue(os.path.exists(self.engine.tmp))
        os.rmdir(fresh)

    def test_new_run_replaces_old(self):
        first = self.req("api/run", {"topology": "octet", "method": "sdf", "a": 2, "rho": 0.3,
                                     "cells": [6, 6, 6], "N": 96})["job"]
        time.sleep(0.5)
        old = self.engine.job(first)
        second = self.req("api/run", {"topology": "bcc", "method": "struts", "a": 1, "rho": 0.2,
                                      "cells": [1, 1, 1]})["job"]
        self.assertFalse(old.proc.is_alive())
        self.assertFalse(os.path.exists(old.dir))
        self.assertEqual(self.wait(second)["state"], "done")


if __name__ == "__main__":
    unittest.main()
