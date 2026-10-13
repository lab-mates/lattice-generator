"""Build one small lattice per feature and check each result.

Run as ``LatticeGenerator.exe --selftest`` (or ``python -m desktop --selftest``)
after packaging: it exercises every pymeshlab filter latticegen uses, so a
pruned bundle that lost a needed plugin fails here, not on a colleague's PC.
"""

from __future__ import annotations

import io
import os
import time
import traceback
import warnings
from contextlib import redirect_stdout

CASES = [
    ("beams, beam-sized joints", dict(topology="octet", method="struts", a=1, rho=0.2,
                                      cells=(3, 3, 3))),                     # manifold3d
    ("beams, hull joints", dict(topology="kelvin", method="struts", a=1, rho=0.2, cells=(2, 2, 2),
                                node="hull")),
    ("rounded beams (sdf)", dict(topology="octet", method="sdf", a=2, rho=0.3, cells=(2, 2, 2),
                                 N=32, smooth=0.02)),
    ("organic (fourier)", dict(topology="bcc", method="fourier", a=2, rho=0.25, cells=(2, 2, 2),
                               N=32)),
    ("surface + islands", dict(topology="gyroid", method="tpms", a=2, rho=0.25, sheet=True,
                               cells=(2, 2, 2), N=32, islands=True)),
    ("graded", dict(topology="bcc", method="sdf", a=2, graded=(0.15, 0.4), cells=(2, 2, 3), N=32)),
    ("plates + 3MF", dict(topology="octet", method="struts", a=1, rho=0.2, cells=(2, 2, 2),
                          plates="both", plate_thickness=0.5, plate_size=1.2)),
    # pymeshlab from here on
    ("simplify", dict(topology="octet", method="sdf", a=2, rho=0.3, cells=(2, 2, 2), N=40,
                      edge_length=0.1)),
    ("instanced, isotropic", dict(topology="octet", method="sdf", a=1, rho=0.25, cells=(3, 3, 3),
                                  N=40, instanced=True, seam_spacing=0.1, edge_length=0.1)),
    ("instanced, decimate", dict(topology="octet", method="sdf", a=1, rho=0.25, cells=(3, 3, 3),
                                 N=40, instanced=True, seam_spacing=0.1, edge_length=0.1,
                                 remesh="decimate")),
    ("repair", dict(topology="bcc", method="struts", a=1, rho=0.2, cells=(2, 2, 2), repair=True)),
]


def _3mf_round_trip(mesh) -> bool:
    import tempfile
    from latticegen import LatticeMesh
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "selftest.3mf")
        mesh.export(path)
        back = LatticeMesh.load(path)
    return back.n_faces == mesh.n_faces and back.is_watertight


def run(report_path=None) -> int:
    from latticegen.app import build
    lines = []
    out = lambda s="": (lines.append(s), print(s))
    try:
        import pymeshlab
        out(f"pymeshlab loaded from {os.path.dirname(pymeshlab.__file__)}")
    except Exception as e:
        out(f"pymeshlab NOT available: {e}")
    failed = 0
    for name, kw in CASES:
        t0 = time.time()
        try:
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                res = build(**kw, log=lambda *_: None)
            m = res.mesh
            _, n_comp = m.components()
            ok = m.is_watertight and m.volume > 0 and (n_comp == 1 or not kw.get("islands"))
            if kw.get("plates"):                    # one solid, and it survives a 3MF round trip
                ok = ok and n_comp == 1 and _3mf_round_trip(m)
            status = "ok  " if ok else "FAIL"
            detail = f"{m.n_faces:>9,} faces  watertight {m.is_watertight}  components {n_comp}"
        except Exception as e:
            ok, status = False, "FAIL"
            detail = f"{type(e).__name__}: {e}"
            lines.append(traceback.format_exc())
        failed += not ok
        out(f"{status} {name:26s} {time.time() - t0:5.1f} s  {detail}")
    out(f"{len(CASES) - failed}/{len(CASES)} passed")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 1 if failed else 0
