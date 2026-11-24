"""Plates welded to lattice blocks (latticegen.plates) and 3MF export."""

import os
import sys
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import latticegen as lg                                  # noqa: E402
from latticegen.app import build                         # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)


class TestPlates(unittest.TestCase):
    def test_struts_both(self):
        cell = lg.UnitCell.cubic(1.0)
        m = lg.mesh_struts(cell, "octet", rho=0.2, cells=(2, 2, 2))
        self.assertEqual(m.info["node_z"], (0.0, 2.0))
        p = lg.add_plates(m, cell, (2, 2, 2), "both", thickness=0.5, size=1.2)
        c = p.check()
        self.assertTrue(c["watertight"])
        self.assertEqual(c["components"], 1)
        np.testing.assert_allclose(p.bounds, [[-0.2, -0.2, -0.5], [2.2, 2.2, 2.5]], atol=1e-6)
        # the plates' faces go through the outermost node centres
        self.assertAlmostEqual(p.info["plate_planes"][0], 0.0)
        # the lattice density is reported without the plates
        self.assertAlmostEqual(p.relative_density, m.relative_density, places=4)
        self.assertLess(p.info["plate_volume"], 2 * 0.5 * 2.4 ** 2)
        self.assertIn("plates: both", p.report(c))

    def test_field_block_exact_volume(self):
        cell = lg.UnitCell.cubic(2.0)
        fld = lg.sdf_field(cell, "bcc", N=24)
        m = lg.mesh_stack(fld, rho=0.3, cells=(2, 1, 1))
        p = lg.add_plates(m, cell, (2, 1, 1), "bottom", thickness=0.4, size=1.0)
        # plate against the flat caps: nothing overlaps, volumes add up exactly
        self.assertAlmostEqual(p.volume, m.volume + 4 * 2 * 0.4, places=4)
        self.assertTrue(p.is_watertight)
        self.assertEqual(p.check()["components"], 1)

    def test_oblique_cell(self):
        cell = lg.UnitCell.from_parameters(1, 1, 1.2, 80, 75, 60)
        m = lg.mesh_struts(cell, "sc", rho=0.2, cells=(2, 2, 2))
        p = lg.add_plates(m, cell, (2, 2, 2), "both", thickness=0.3, size=1.1)
        c = p.check()
        self.assertTrue(c["watertight"])
        self.assertEqual(c["components"], 1)

    def test_chained_and_invalid(self):
        cell = lg.UnitCell.cubic(1.0)
        m = lg.mesh_struts(cell, "bcc", rho=0.2, cells=(1, 1, 1))
        p = lg.add_plates(lg.add_plates(m, cell, (1, 1, 1), "bottom", 0.2, 1.0),
                          cell, (1, 1, 1), "top", 0.2, 1.0)
        self.assertEqual(p.info["plates"], "both")
        self.assertAlmostEqual(p.extents[2], 1.4, places=5)
        self.assertIs(lg.add_plates(m, cell, (1, 1, 1), "none"), m)
        for kw in ({"size": 0.9}, {"thickness": 0}, {"where": "left"}):
            with self.assertRaises(ValueError):
                lg.add_plates(m, cell, (1, 1, 1), **{"where": "both", **kw})

    def test_build_option(self):
        res = build("gyroid", method="tpms", a=2.0, N=24, rho=0.3, cells=(1, 1, 1),
                    plates="top", plate_thickness=0.5, plate_size=1.5, log=lambda *_: None)
        np.testing.assert_allclose(res.mesh.extents, [3.0, 3.0, 2.5], atol=1e-5)
        self.assertIn("plates: top", res.report)


class Test3MF(unittest.TestCase):
    def test_round_trip(self):
        cell = lg.UnitCell.cubic(1.0)
        m = lg.add_plates(lg.mesh_struts(cell, "octet", rho=0.2, cells=(2, 2, 2)),
                          cell, (2, 2, 2), "both", 0.3)
        path = os.path.join(tempfile.mkdtemp(), "block.3mf")
        m.export(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            model = zf.read("3D/3dmodel.model").decode()
        self.assertTrue({"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= names)
        self.assertIn('unit="millimeter"', model)
        self.assertIn("MATES Lab", model)
        q = lg.LatticeMesh.load(path)
        self.assertEqual((q.n_vertices, q.n_faces), (m.n_vertices, m.n_faces))
        np.testing.assert_array_equal(q.verts, m.verts)          # %.9g: exact float32
        self.assertTrue(q.is_watertight)


if __name__ == "__main__":
    unittest.main()
