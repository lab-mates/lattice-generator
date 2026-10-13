"""latticegen.app.build and the browser bridge (webapp/py/bridge.py), run natively."""

import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "webapp", "py"))

import latticegen as lg                     # noqa: E402
from latticegen import meshops               # noqa: E402
from latticegen.app import build             # noqa: E402
import bridge                                # noqa: E402

SMALL = dict(cells=(2, 2, 2), N=32)


class TestBuild(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_methods(self):
        cases = [dict(topology="octet", method="struts", a=1.0, rho=0.2),
                 dict(topology="octet", method="sdf", a=2.0, rho=0.3, smooth=0.02),
                 dict(topology="bcc", method="fourier", a=2.0, rho=0.25),
                 dict(topology="gyroid", method="tpms", a=3.0, rho=0.25),
                 dict(topology="bcc", method="sdf", a=2.0, graded=(0.15, 0.4))]
        for kw in cases:
            with self.subTest(**kw):
                res = build(**kw, **SMALL, check=True, log=lambda *_: None)
                self.assertTrue(res.mesh.is_watertight)
                self.assertIn("DLP check", res.dlp)
                self.assertEqual(res.filename, f"{kw['topology']}_2x2x2.stl")
                self.assertEqual(res.N, 0 if kw["method"] == "struts" else 32)

    def test_auto_method_and_errors(self):
        self.assertEqual(build("gyroid", N=24, cells=(1, 1, 1), log=lambda *_: None).method, "tpms")
        with self.assertRaises(ValueError):
            build("octet", method="struts", graded=(0.1, 0.3), log=lambda *_: None)
        with self.assertRaises(ValueError):
            build("octet", method="nope", log=lambda *_: None)

    def test_islands(self):
        kw = dict(topology="gyroid", method="tpms", a=2.0, rho=0.25, sheet=True, log=lambda *_: None)
        with redirect_stdout(io.StringIO()):
            raw = build(**kw, **SMALL)
            clean = build(**kw, **SMALL, islands=True)
        _, n_raw = raw.mesh.components()
        _, n_clean = clean.mesh.components()
        self.assertGreater(n_raw, 1)                  # corner slivers at the block boundary
        self.assertEqual(n_clean, 1)
        self.assertTrue(clean.mesh.is_watertight)

    def test_check_matches_parts(self):
        m = lg.mesh_struts(lg.UnitCell.cubic(1.0), "bcc", rho=0.2, cells=(3, 3, 3))
        c = m.check(chunk=1000)                       # several chunks
        self.assertEqual(c["watertight"], m.is_watertight)
        self.assertEqual(c["euler"], m.euler_number)
        self.assertEqual(c["components"], m.components()[1])
        self.assertAlmostEqual(c["volume"], m.volume, places=9)
        self.assertAlmostEqual(c["area"], m.area, places=9)
        el = m.edge_lengths()
        self.assertAlmostEqual(c["edge_median"], float(np.median(el)), places=6)

    def test_cli_remove_islands(self):
        from latticegen import cli
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g.stl")
            with redirect_stdout(io.StringIO()):
                cli.main(["build", "gyroid", "--sheet", "--a", "2", "--rho", "0.25", "--N", "32",
                          "--cells", "2", "2", "2", "--remove-islands", "-o", out])
            _, n = lg.LatticeMesh.load(out).components()
            self.assertEqual(n, 1)


class TestBridge(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_options(self):
        opts = json.loads(bridge.options())
        self.assertEqual([t["name"] for t in opts["topologies"]], lg.available_topologies())
        self.assertEqual(opts["tpms"], lg.available_tpms())

    def test_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.stl")
            params = {"topology": "octet", "method": "struts", "a": 1, "rho": 0.2,
                      "cells": [2, 2, 2], "graded": None, "cell": None, "N": 0,
                      "check": True, "unknown_key": 1}
            lines = []
            r = bridge.run(json.dumps(params), log=lines.append, stl_path=path)
            v, f = meshops.read_stl(path)
            self.assertEqual(len(f), r["n_faces"])
            self.assertEqual(r["verts"].dtype, np.float32)
            self.assertEqual(r["faces"].dtype, np.uint32)
            self.assertEqual(r["faces"].shape, (r["n_faces"], 3))
            self.assertFalse(r["preview_partial"])
            self.assertTrue(any(l.startswith("STL ready") for l in lines))

    def test_partial_preview(self):
        old = bridge.PREVIEW_MAX_FACES
        bridge.PREVIEW_MAX_FACES = 10_000
        try:
            with tempfile.TemporaryDirectory() as d:
                r = bridge.run({"topology": "octet", "method": "struts", "a": 1, "rho": 0.2,
                                "cells": [5, 5, 5]}, log=lambda *_: None,
                               stl_path=os.path.join(d, "x.stl"))
        finally:
            bridge.PREVIEW_MAX_FACES = old
        self.assertTrue(r["preview_partial"])
        self.assertLess(len(r["faces"]), r["n_faces"])

    def test_general_cell(self):
        with tempfile.TemporaryDirectory() as d:
            r = bridge.run({"topology": "octet", "method": "sdf", "cell": [2, 2, 2.5, 90, 90, 90],
                            "rho": 0.3, "cells": [1, 1, 1], "N": 24}, log=lambda *_: None,
                           stl_path=os.path.join(d, "x.stl"))
        self.assertAlmostEqual(r["extents"][2] / r["extents"][0], 1.25, delta=0.05)


if __name__ == "__main__":
    unittest.main()
