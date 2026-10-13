"""Test suite (run with ``python -m unittest discover tests`` or pytest).

Grid sizes are kept small (N = 24..32) so the whole suite runs in well under
a minute; the properties tested (watertightness, density, welding,
legacy equivalence) do not depend on N.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import latticegen as lg  # noqa: E402

try:
    import pymeshlab  # noqa: F401
    HAVE_PML = True
except ImportError:
    HAVE_PML = False

N = 32


class TestCellsAndTopologies(unittest.TestCase):
    def test_cell_constructors(self):
        c = lg.UnitCell.cubic(2.0)
        self.assertAlmostEqual(c.volume, 8.0)
        self.assertTrue(c.is_cubic)
        h = lg.UnitCell.hexagonal(1.0, 1.5)
        self.assertAlmostEqual(h.volume, np.sqrt(3) / 2 * 1.5)
        B = c.reciprocal()
        np.testing.assert_allclose(B @ c.A.T, 2 * np.pi * np.eye(3))

    def test_strut_counts(self):
        expected = {"sc": 3, "bcc": 8, "fcc": 12, "bccz": 9, "fccz": 13, "fbcc": 20,
                    "octet": 24, "diamond": 16, "kelvin": 24}
        for name, n in expected.items():
            self.assertEqual(lg.skeleton(name).n_struts, n, name)

    def test_connectivity(self):
        self.assertEqual(lg.skeleton("octet").connectivity(), 12)
        self.assertEqual(lg.skeleton("diamond").connectivity(), 4)
        self.assertEqual(lg.skeleton("kelvin").connectivity(), 4)

    def test_custom_dedup(self):
        # the same strut given twice, once translated -> one strut
        s = lg.Skeleton.custom([([0, 0, 0], [1, 0, 0]), ([1, 0, 0], [2, 0, 0]), ([1, 0, 0], [0, 0, 0])])
        self.assertEqual(s.n_struts, 1)

    def test_unknown_topology(self):
        with self.assertRaises(KeyError):
            lg.skeleton("nope")


class TestFields(unittest.TestCase):
    def test_fourier_periodic_and_real(self):
        f = lg.fourier_field(1.0, "bcc", N=N)
        self.assertEqual(f.f.shape, (N, N, N))
        # cubic symmetry of bcc: f(x,y,z) == f(y,x,z)
        np.testing.assert_allclose(f.f, np.transpose(f.f, (1, 0, 2)), atol=1e-4 * abs(f.f).max())

    def test_fourier_aliasing_guard(self):
        with self.assertRaises(ValueError):
            lg.fourier_field(1.0, "octet", N=8)

    def test_sdf_exact_radius(self):
        # simple cubic struts: distance at the cell centre is sqrt(2)/2 * a
        f = lg.sdf_field(2.0, "sc", N=N)
        d = -f.f
        centre = d[N // 2, N // 2, N // 2]
        self.assertAlmostEqual(centre, np.sqrt(2) / 2 * 2.0, places=5)
        self.assertAlmostEqual(d[0, 0, 0], 0.0, places=6)

    def test_sdf_culling_matches_full(self):
        from latticegen import sdf
        segs = lg.skeleton("octet").segments
        d1 = sdf.distance_field(segs, 24, smooth=0.02, cull=True)
        d2 = sdf.distance_field(segs, 24, smooth=0.02, cull=False)
        np.testing.assert_allclose(d1, d2, atol=1e-9)

    def test_smooth_min_bounded(self):
        segs = lg.skeleton("octet").segments
        from latticegen import sdf
        d_hard = sdf.distance_field(segs, 24)
        d_soft = sdf.distance_field(segs, 24, smooth=0.02)
        self.assertTrue(np.all(d_soft <= d_hard + 1e-12))
        self.assertTrue(np.all(d_hard - d_soft <= 0.02 * np.log(24 * 27) + 1e-9))

    def test_tpms(self):
        g = lg.tpms_field(1.0, "gyroid", N=N)
        self.assertAlmostEqual(g.density(0.0), 0.5, places=2)
        s = lg.tpms_field(1.0, "gyroid", N=N, sheet=True)
        self.assertTrue(np.all(s.f <= 0))
        for name in lg.available_tpms():
            lg.tpms_field(1.0, name, N=16)

    def test_threshold_density_roundtrip(self):
        f = lg.sdf_field(1.0, "bcc", N=N)
        for rho in (0.1, 0.3, 0.6):
            self.assertAlmostEqual(f.density(f.threshold(rho)), rho, delta=0.02)

    def test_feature_sizes_sdf(self):
        f = lg.sdf_field(1.0, "sc", N=N)
        r = f.strut_radius(0.2)
        fs = f.feature_sizes(0.2)
        self.assertAlmostEqual(fs["strut_radius"], r, delta=1.5 / N)
        self.assertGreater(fs["pore_radius"], r)

    def test_connectivity_check(self):
        N = 32
        u = (np.arange(N) + 0.5) / N
        X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
        r = np.sqrt((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2)
        hollow = lg.custom_field(1.0, np.where(r < 0.2, -1.0, 1.0))
        self.assertEqual(hollow.connectivity_check(0.9)["enclosed_voids"], 1)
        island = lg.custom_field(1.0, np.where(r < 0.2, 1.0, -1.0))
        self.assertEqual(island.connectivity_check(0.03)["floating_solids"], 1)
        sheet = lg.tpms_field(1.0, "gyroid", N=N, sheet=True).connectivity_check(0.3)
        self.assertEqual((sheet["solid_components"], sheet["void_components"], sheet["enclosed_voids"]), (1, 2, 0))
        octet = lg.sdf_field(1.0, "octet", N=N).connectivity_check(0.3)
        self.assertEqual((octet["floating_solids"], octet["enclosed_voids"]), (0, 0))

    def test_save_load(self):
        f = lg.sdf_field(1.0, "sc", N=16)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.npz")
            f.save(p)
            g = lg.UnitField.load(p)
        np.testing.assert_array_equal(f.f, g.f)
        np.testing.assert_array_equal(f.A, g.A)


class TestMeshStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oct = lg.sdf_field(1.0, "octet", N=N, smooth=0.02)
        cls.gyr = lg.tpms_field(1.0, "gyroid", N=N)

    def test_uniform(self):
        m = lg.mesh_stack(self.oct, rho=0.3, cells=(2, 2, 2))
        self.assertTrue(m.is_watertight)
        self.assertEqual(m.components()[1], 1)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=2e-3)
        np.testing.assert_allclose(m.bounds, [[0, 0, 0], [2, 2, 2]], atol=1e-6)

    def test_slab_streaming_identical(self):
        m1 = lg.mesh_stack(self.oct, rho=0.3, cells=(1, 1, 3), slab_cells=1)
        m3 = lg.mesh_stack(self.oct, rho=0.3, cells=(1, 1, 3), slab_cells=3)
        self.assertEqual(m1.n_vertices, m3.n_vertices)
        self.assertAlmostEqual(m1.volume, m3.volume, places=6)
        self.assertTrue(m1.is_watertight)

    def test_per_cell_and_mixed(self):
        rho = np.array([[[0.2, 0.3, 0.4]] * 2] * 2)
        cmap = np.array([[[0, 1, 0]] * 2] * 2)
        m = lg.mesh_stack([self.oct, self.gyr], rho=rho, cells=(2, 2, 3), cell_map=cmap)
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=5e-3)

    def test_graded_z(self):
        m = lg.mesh_stack(self.oct, rho=lambda z: 0.2 + 0.2 * z, cells=(1, 1, 4))
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=0.015)

    def test_graded_xyz(self):
        rho = lambda x, y, z: 0.2 + 0.2 * x / 2.0
        m = lg.mesh_stack(self.oct, rho=rho, cells=(2, 1, 2))
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=0.015)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            lg.mesh_stack(self.oct, rho=1.2)
        with self.assertRaises(ValueError):
            lg.mesh_stack(self.oct, rho=0.3, cells=(2, 2, 2), cell_map=np.zeros((1, 1, 1), int))
        with self.assertRaises(ValueError):
            lg.mesh_stack([self.oct, lg.tpms_field(1.0, "gyroid", N=16)], rho=0.3)

    def test_export_roundtrip(self):
        m = lg.mesh_stack(self.gyr, rho=0.4, cells=(1, 1, 1))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.stl")
            m.export(p)
            r = lg.LatticeMesh.load(p)
        self.assertEqual(r.n_faces, m.n_faces)
        self.assertAlmostEqual(r.volume, m.volume, places=6)
        self.assertTrue(r.is_watertight)

    def test_field_mesh_shortcut(self):
        m = self.gyr.mesh(rho=0.5, cells=(1, 1, 1))
        self.assertTrue(m.is_watertight)


class TestLegacyEquivalence(unittest.TestCase):
    """The refactored code must reproduce the original scripts."""

    def test_fourier_and_stack(self):
        legacy_dir = os.path.join(os.path.dirname(__file__), "..", "legacy")
        if not os.path.isdir(legacy_dir):
            self.skipTest("legacy scripts not present")
        sys.path.insert(0, legacy_dir)
        import fourier_lattice as fl
        A, hkl, c = fl.octet_struts(node_comp=0.5)
        f_old = fl.field_on_grid(hkl, c, N)
        fld = lg.fourier_field(1.0, "octet", N=N, node_comp=0.5)
        np.testing.assert_allclose(fld.f, f_old, rtol=0, atol=1e-4 * abs(f_old).max())
        v, fc, _ = fl.stacked_lattice_mesh(A, f_old, rho=0.3, cells=(2, 1, 1))
        m = lg.mesh_stack(fld, rho=0.3, cells=(2, 1, 1), exact_density=False)
        self.assertEqual(m.n_vertices, len(v))
        self.assertAlmostEqual(m.volume, fl.mesh_volume(v, fc), places=5)


@unittest.skipUnless(HAVE_PML, "pymeshlab not installed")
class TestRemeshAndInstanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fld = lg.sdf_field(1.0, "octet", N=N, smooth=0.02)
        cls.mesh = lg.mesh_stack(cls.fld, rho=0.3, cells=(2, 2, 2))

    def test_decimate(self):
        d = self.mesh.decimate(fraction=0.2)
        self.assertLess(d.n_faces, 0.25 * self.mesh.n_faces)
        self.assertTrue(d.is_watertight)
        self.assertAlmostEqual(d.volume, self.mesh.volume, delta=0.01 * self.mesh.volume)

    def test_simplify_edge_length(self):
        s = self.mesh.simplify(edge_length=0.1)
        self.assertTrue(s.is_watertight)
        self.assertAlmostEqual(np.median(s.edge_lengths()), 0.1, delta=0.04)

    def test_remesh_isotropic(self):
        r = self.mesh.remesh(0.05, iterations=4)
        self.assertTrue(r.is_watertight)
        self.assertAlmostEqual(np.median(r.edge_lengths()), 0.05, delta=0.02)
        self.assertAlmostEqual(r.volume, self.mesh.volume, delta=0.05 * self.mesh.volume)

    def test_remove_islands_and_repair(self):
        v = np.vstack([self.mesh.verts, [[5, 5, 5], [5.1, 5, 5], [5, 5.1, 5], [5, 5, 5.1]]])
        n = self.mesh.n_vertices
        f = np.vstack([self.mesh.faces, [[n, n + 2, n + 1], [n, n + 1, n + 3],
                                         [n, n + 3, n + 2], [n + 1, n + 2, n + 3]]])
        m = lg.LatticeMesh(v, f)
        self.assertEqual(m.components()[1], 2)
        self.assertEqual(m.remove_islands().components()[1], 1)
        holed = lg.LatticeMesh(self.mesh.verts, self.mesh.faces[6:])
        self.assertFalse(holed.is_watertight)
        rep = holed.repair()
        self.assertTrue(rep.is_watertight)
        self.assertGreater(rep.volume, 0)

    def test_instanced_seam_simplified(self):
        m = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), seam_spacing=0.08)  # mm, ~N/12 voxels
        self.assertTrue(m.is_watertight)
        self.assertEqual(m.components()[1], 1)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=3e-3)
        self.assertLess(m.n_faces, 0.5 * self.mesh.n_faces)
        # coarser seams than interiors
        m2 = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), seam_spacing=0.12, edge_length=0.06)
        self.assertTrue(m2.is_watertight)
        self.assertGreater(m2.n_faces, m.n_faces * 0.8)
        # sub-voxel lengths are a unit mix-up
        with self.assertRaises(ValueError):
            lg.mesh_instanced(self.fld, rho=0.3, cells=(1, 1, 1), seam_spacing=0.001)

    def test_instanced_decimate_interior(self):
        iso = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), seam_spacing=0.08, edge_length=0.06)
        dec = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), seam_spacing=0.08, edge_length=0.06,
                                remesh="decimate")
        self.assertTrue(dec.is_watertight)
        self.assertEqual(dec.components()[1], 1)
        self.assertAlmostEqual(dec.relative_density, 0.3, delta=3e-3)
        self.assertLess(dec.n_faces, 0.6 * iso.n_faces)
        self.assertEqual(dec.info["remesh"], "decimate")

    def test_instanced_edge_length_alone(self):
        # frozen seams, interior remeshed at the requested size
        m = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), edge_length=0.1)
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=3e-3)
        default = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2))
        self.assertLess(m.n_faces, default.n_faces)

    def test_instanced_frozen_and_raw(self):
        m = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2))
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.relative_density, 0.3, delta=3e-3)
        raw = lg.mesh_instanced(self.fld, rho=0.3, cells=(2, 2, 2), remesh=None)
        self.assertTrue(raw.is_watertight)
        self.assertAlmostEqual(raw.volume, self.mesh.volume, delta=0.01 * self.mesh.volume)


class TestStruts(unittest.TestCase):
    """Explicit prism + hull-node mesher (no field, no pymeshlab)."""

    def test_octet_exact(self):
        m = lg.mesh_struts(1.0, "octet", rho=0.2, cells=(2, 2, 2), node="hull")
        self.assertTrue(m.is_watertight)
        self.assertEqual(m.components()[1], 1)
        self.assertGreater(m.volume, 0)
        # mesh volume equals the closed-form volume of prisms + hulls
        self.assertAlmostEqual(m.volume, m.info["analytic_volume"], delta=1e-5 * m.volume)
        # whole struts at the boundary add material: block density above the bulk value
        self.assertGreater(m.relative_density, 0.2)
        self.assertLess(m.relative_density, 0.3)
        self.assertLess(m.n_faces, 900 * 8)                    # ~660 per bulk cell
        self.assertEqual(m.info["mode"], "struts")
        self.assertAlmostEqual(m.info["min_thickness"], 2 * m.info["strut_radius"] * np.cos(np.pi / 6))

    def test_bulk_density_calibration(self):
        # V(r) = alpha r^2 - beta r^3 evaluated at the calibrated radius gives rho
        for topo, rho in (("octet", 0.2), ("bcc", 0.1), ("kelvin", 0.2)):
            r = lg.radius_for_density(1.0, topo, rho)
            a, b, r_max = lg.struts.bulk_volume_coefficients(1.0, topo)
            self.assertAlmostEqual(a * r * r - b * r ** 3, rho, places=9)
            self.assertLess(r, r_max)
        # the SDF strut radius of the sharp union is in the same range (polygon vs circle, hull nodes)
        r_sdf = lg.sdf_field(1.0, "octet", N=N).strut_radius(0.2)
        self.assertAlmostEqual(lg.radius_for_density(1.0, "octet", 0.2), r_sdf, delta=0.25 * r_sdf)

    def test_all_topologies_and_options(self):
        for topo in lg.available_topologies():
            m = lg.mesh_struts(1.0, topo, rho=0.15, cells=(2, 2, 2))
            self.assertTrue(m.is_watertight, topo)
            self.assertEqual(m.components()[1], 1, topo)
            self.assertAlmostEqual(m.volume, m.info["analytic_volume"], delta=1e-5 * m.volume)
        hull = lg.mesh_struts(2.0, "octet", rho=0.2, cells=(1, 1, 1), sides=4, node="hull")
        rnd = lg.mesh_struts(2.0, "octet", rho=0.2, cells=(1, 1, 1), sides=4, node="round")
        oct8 = lg.mesh_struts(2.0, "octet", radius=0.12, cells=(1, 1, 1), sides=8)
        for m in (hull, rnd, oct8):
            self.assertTrue(m.is_watertight)
            self.assertAlmostEqual(m.volume, m.info["analytic_volume"], delta=1e-5 * m.volume)
        self.assertGreater(rnd.n_faces, hull.n_faces)
        self.assertAlmostEqual(oct8.info["strut_radius"], 0.12)
        # custom skeleton and a non-cubic cell
        sk = lg.Skeleton.custom([([0, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 1, 0]), ([0, 0, 0], [0, 0, 1])])
        m = lg.mesh_struts(lg.UnitCell.tetragonal(1.0, 1.5), sk, rho=0.1, cells=(2, 2, 1))
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.extents[2], 1.5 + 2 * m.info["strut_radius"], delta=1e-4)

    def test_union_nodes_are_strut_sized(self):
        from latticegen.struts import _block_struts, _point_segment_distance
        for topo in ("octet", "bcc", "kelvin", "bccz"):
            m = lg.mesh_struts(1.0, topo, rho=0.2, cells=(2, 2, 2))          # node="union" default
            self.assertEqual(m.info["node"], "union")
            self.assertTrue(m.is_watertight, topo)
            self.assertAlmostEqual(m.volume, m.info["analytic_volume"], delta=1e-5 * m.volume)
            r = m.info["strut_radius"]
            self.assertEqual(m.info["node_radius"], r)
            # every vertex lies within one strut radius of the skeleton (hull nodes reach ~2 r)
            P, Q = _block_struts(lg.skeleton(topo), (2, 2, 2))
            V = m.verts.astype(float)
            dmin = np.full(len(V), np.inf)
            for a, b in zip(P, Q):
                dmin = np.minimum(dmin, _point_segment_distance(V, np.broadcast_to(a, V.shape),
                                                                np.broadcast_to(b, V.shape)))
            self.assertLess(dmin.max(), r * (1 + 1e-5), topo)
        # same density, smaller nodes -> thicker struts than with hull nodes
        self.assertGreater(lg.radius_for_density(1.0, "octet", 0.2),
                           lg.radius_for_density(1.0, "octet", 0.2, node="hull"))

    def test_union_backend_override(self):
        from latticegen import struts
        calls = []

        def backend(point_sets):
            calls.append(len(point_sets))
            struts.set_hull_union_backend(None)
            try:
                return struts._hull_union(point_sets)
            finally:
                struts.set_hull_union_backend(backend)
        struts.set_hull_union_backend(backend)
        try:
            m = lg.mesh_struts(1.0, "bcc", rho=0.2, cells=(1, 1, 1))
        finally:
            struts.set_hull_union_backend(None)
        self.assertTrue(calls)
        self.assertTrue(m.is_watertight)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            lg.mesh_struts(1.0, "octet", rho=0.9)                 # struts vanish into the nodes
        with self.assertRaises(ValueError):
            lg.mesh_struts(1.0, "octet", radius=0.5)
        with self.assertRaises(ValueError):
            lg.mesh_struts(1.0, "octet")                          # neither rho nor radius
        with self.assertRaises(ValueError):
            lg.mesh_struts(1.0, "octet", rho=0.2, sides=2)

    def test_report_and_cli(self):
        m = lg.mesh_struts(1.0, "octet", rho=0.2, cells=(2, 2, 2))
        rep = lg.dlp_report(m, pixel_mm=0.05)
        self.assertIn("strut_radius", rep["features_mm"])
        self.assertIn("min_solid_px", rep)
        self.assertIn("strut_radius", lg.format_report(rep))
        from latticegen import cli
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.stl")
            cli.main(["build", "octet", "--method", "struts", "--a", "1", "--rho", "0.2",
                      "--cells", "2", "2", "2", "--sides", "4", "--check", "-o", out])
            back = lg.LatticeMesh.load(out)
            self.assertGreater(back.n_faces, 0)
            self.assertTrue(back.is_watertight)


class TestPrinting(unittest.TestCase):
    def test_report(self):
        fld = lg.sdf_field(2.0, "bcc", N=N)
        m = lg.mesh_stack(fld, rho=0.2, cells=(1, 1, 1))
        rep = lg.dlp_report(m, fld, 0.2, pixel_mm=0.05, layer_mm=0.05)
        self.assertIn("features_mm", rep)
        self.assertGreater(rep["n_layers"], 0)
        txt = lg.format_report(rep)
        self.assertIn("DLP check", txt)
        # a coarse pixel makes the struts too thin -> warning
        rep2 = lg.dlp_report(m, fld, 0.2, pixel_mm=0.3)
        self.assertTrue(any("thinnest" in w for w in rep2["warnings"]))

    def test_recommend_grid(self):
        self.assertEqual(lg.recommend_grid(lg.UnitCell.cubic(2.0), 0.05), 80)


if __name__ == "__main__":
    unittest.main()
