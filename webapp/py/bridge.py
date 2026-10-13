"""
Glue between the browser worker and latticegen (runs inside Pyodide, but is
plain Python and is tested natively by tests/test_bridge.py).

    options()          -> JSON with topologies, TPMS names and defaults
    run(params, log)   -> dict(mesh, preview, stl_path, report, dlp, ...)

The STL is written to a file (Pyodide keeps files in JS memory, outside the
WebAssembly heap) and the worker reads it back from there.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

import latticegen as lg
from latticegen.app import build

PREVIEW_MAX_FACES = 3_000_000       # larger meshes are previewed as a corner block
PREVIEW_CELLS = 3

# GUI parameter -> build() keyword (anything else in the JSON is ignored)
_KEYS = ("topology", "method", "a", "cell", "N", "rho", "graded", "cells", "smooth",
         "node_comp", "sheet", "sides", "node", "islands", "check", "pixel", "layer",
         # need pymeshlab (desktop app only)
         "instanced", "seam_spacing", "edge_length", "remesh", "repair")


def _use_manifold_js():
    """Under Pyodide there is no manifold3d wheel: route latticegen's hull
    unions (strut-sized joints) to Manifold's WebAssembly build, which the
    worker registers as the JS module ``manifold_js``."""
    try:
        import manifold3d  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import manifold_js
        from pyodide.ffi import to_js
    except ImportError:
        return
    from latticegen.struts import set_hull_union_backend

    def backend(point_sets):
        res = manifold_js.hull_union(to_js([np.asarray(p, float).tolist() for p in point_sets]))
        v = np.frombuffer(res.verts.to_py(), np.float32).reshape(-1, 3).astype(np.float64)
        t = np.frombuffer(res.tris.to_py(), np.uint32).reshape(-1, 3).astype(np.int64)
        return v, t, float(res.volume)

    set_hull_union_backend(backend)


_use_manifold_js()


def has_pymeshlab() -> bool:
    try:
        import pymeshlab  # noqa: F401
        return True
    except ImportError:
        return False


def options() -> str:
    tops = []
    for n in lg.available_topologies():
        sk = lg.skeleton(n)
        tops.append({"name": n, "struts": sk.n_struts, "connectivity": sk.connectivity()})
    return json.dumps({"version": lg.__version__, "topologies": tops,
                       "tpms": lg.available_tpms(),
                       "capabilities": {"pymeshlab": has_pymeshlab()}})


def _format_warning(message, category, filename, lineno, line=None):
    return f"warning: {message}\n"


def _kwargs(params: dict) -> dict:
    kw = {k: params[k] for k in _KEYS if params.get(k) is not None}
    if kw.get("cell") is not None:
        kw["cell"] = tuple(float(x) for x in kw["cell"])
    if kw.get("graded") is not None:
        kw["graded"] = tuple(float(x) for x in kw["graded"])
    kw["cells"] = tuple(int(c) for c in kw.get("cells", (3, 3, 3)))
    kw["N"] = int(kw.get("N") or 0)
    return kw


def _arrays(mesh):
    return (np.ascontiguousarray(mesh.verts, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.uint32))


def run(params, log=print, stl_path="/tmp/lattice.stl") -> dict:
    """Build the lattice described by ``params`` (a dict or its JSON) and
    write it to ``stl_path``."""
    if isinstance(params, str):
        params = json.loads(params)
    kw = _kwargs(params)
    warnings.formatwarning = _format_warning
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = build(**kw, verbose=bool(params.get("verbose")), log=log)
        mesh = res.mesh
        mesh.export(stl_path)
        log(f"STL ready: {res.filename} ({(84 + 50 * mesh.n_faces) / 1e6:.1f} MB)")

        preview, partial = mesh, False
        if mesh.n_faces > PREVIEW_MAX_FACES:
            small = tuple(min(c, PREVIEW_CELLS) for c in kw["cells"])
            log(f"mesh too large to display whole: previewing a {'x'.join(map(str, small))} "
                f"corner block")
            pkw = dict(kw, cells=small, check=False)
            preview = build(**pkw, log=lambda *_: None).mesh
            partial = True
    verts, faces = _arrays(preview)
    return {
        "verts": verts, "faces": faces, "preview_partial": partial,
        "stl_path": stl_path, "filename": res.filename,
        "n_faces": int(mesh.n_faces), "n_vertices": int(mesh.n_vertices),
        "extents": [float(x) for x in mesh.extents],
        "report": res.report, "dlp": res.dlp, "method": res.method, "N": int(res.N),
        "timings": {k: round(v, 2) for k, v in res.timings.items()},
    }
