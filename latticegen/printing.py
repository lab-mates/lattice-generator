"""
DLP-printability checks.

Digital light processing prints a voxelised image: the lateral resolution is
the projector pixel pitch, the vertical one the layer height.  Features
thinner than a few pixels / layers do not form reliably, closed pockets trap
uncured resin, disconnected pieces float away, and every triangle costs
slicer time.  :func:`dlp_report` gathers the relevant numbers for a mesh /
field pair and flags problems.
"""

from __future__ import annotations

import numpy as np

from .cells import as_cell
from .field import UnitField
from .mesh import LatticeMesh


def recommend_grid(cell, pixel_mm: float, oversample: float = 2.0, minimum: int = 32,
                   multiple_of: int = 8) -> int:
    """Grid size N so that the field voxel is at most ``pixel_mm/oversample``
    along the shortest cell vector (rounded up to a multiple of 8)."""
    cell = as_cell(cell)
    N = int(np.ceil(cell.a_min / (pixel_mm / oversample)))
    N = max(minimum, int(np.ceil(N / multiple_of)) * multiple_of)
    return N


def dlp_report(mesh: LatticeMesh | None = None, field: UnitField | None = None, rho=None,
               pixel_mm: float = 0.05, layer_mm: float = 0.05, min_feature_px: float = 4.0,
               min_pore_mm: float | None = None, resin_density_g_cm3: float = 1.15,
               shrinkage_pct: float = 0.0) -> dict:
    """Printability summary for a DLP job (all lengths in mm).

    mesh            : the block to print (optional; gives size, volume, faces).
    field, rho      : the unit field and density (optional; gives strut /
                      wall / pore sizes from the periodic distance transform).
    pixel_mm        : projector pixel pitch in the build plane.
    layer_mm        : layer thickness.
    min_feature_px  : struts / walls thinner than this many pixels are flagged.
    min_pore_mm     : pores narrower than this are flagged (default 4 px).
    shrinkage_pct   : expected linear shrinkage (post-cure / pyrolysis); the
                      as-designed sizes are scaled to show the final part.

    Returns a dict; ``print(format_report(rep))`` for a readable summary.
    """
    rep = {"pixel_mm": pixel_mm, "layer_mm": layer_mm, "warnings": []}
    warn = rep["warnings"].append
    s = 1.0 - shrinkage_pct / 100.0
    min_pore_mm = 4 * pixel_mm if min_pore_mm is None else min_pore_mm

    if field is not None:
        vox = float(field.voxel_size.max())
        rep["voxel_mm"] = vox
        rep["grid_N"] = field.N
        if vox > pixel_mm:
            warn(f"field voxel ({vox:.4f} mm) is coarser than the pixel ({pixel_mm} mm): "
                 f"raise N to >= {recommend_grid(field.cell, pixel_mm)} to avoid staircase struts")
        if rho is not None:
            fs = field.feature_sizes(rho)
            no_scale = ("node_to_strut_ratio", "solid_components", "void_components",
                        "floating_solids", "enclosed_voids")
            rep["features_mm"] = {k: (v if k in no_scale else v * s) for k, v in fs.items()}
            thick = fs.get("strut_radius", None)
            thick = 2 * thick if thick is not None else fs.get("min_thickness")
            if thick is not None and np.isfinite(thick):
                rep["min_solid_px"] = thick * s / pixel_mm
                rep["min_solid_layers"] = thick * s / layer_mm
                if thick * s < min_feature_px * pixel_mm:
                    warn(f"thinnest solid feature {thick * s:.3f} mm = "
                         f"{thick * s / pixel_mm:.1f} px (< {min_feature_px} px): "
                         f"increase rho or the cell size")
            if fs.get("floating_solids", 0):
                warn(f"the unit cell contains {fs['floating_solids']} isolated solid piece(s): "
                     f"they would float in the vat (check the topology / density)")
            elif fs.get("solid_components", 1) != 1:
                warn(f"the solid splits into {fs['solid_components']} interpenetrating networks "
                     f"that never touch: each prints as a separate part")
            if fs.get("enclosed_voids", 0):
                warn(f"the unit cell contains {fs['enclosed_voids']} enclosed void pocket(s): "
                     f"uncured resin will be trapped")
            pore = fs.get("pore_throat", None)
            if pore is not None and np.isfinite(pore):
                rep["min_pore_px"] = pore * s / pixel_mm
                if pore * s < min_pore_mm:
                    warn(f"narrowest pore throat {pore * s:.3f} mm < {min_pore_mm} mm: "
                         f"resin drainage / cleaning will be difficult")

    if field is None and mesh is not None and mesh.info.get("mode") == "struts":
        # explicit strut mesh: sizes are known exactly from the construction
        inf = mesh.info
        feats = {"strut_radius": inf["strut_radius"], "node_radius": inf["node_radius"],
                 "min_thickness": inf["min_thickness"], "max_thickness": 2 * inf["node_radius"]}
        rep["features_mm"] = {k: v * s for k, v in feats.items()}
        rep["features_mm"]["node_to_strut_ratio"] = inf["node_radius"] / inf["strut_radius"]
        rep["features_mm"]["sides"] = inf["sides"]
        thick = inf["min_thickness"]
        rep["min_solid_px"] = thick * s / pixel_mm
        rep["min_solid_layers"] = thick * s / layer_mm
        if thick * s < min_feature_px * pixel_mm:
            warn(f"thinnest solid feature {thick * s:.3f} mm = {thick * s / pixel_mm:.1f} px "
                 f"(< {min_feature_px} px): increase rho or the cell size")

    if mesh is not None:
        c = mesh.check()
        rep["mesh"] = c
        ext = np.asarray(c["extents"]) * s
        rep["extents_mm"] = tuple(float(x) for x in ext)
        rep["n_layers"] = int(np.ceil(round(ext[2] / layer_mm, 6)))
        rep["volume_mm3"] = c["volume"] * s ** 3
        rep["mass_g"] = rep["volume_mm3"] / 1000.0 * resin_density_g_cm3
        if not c["watertight"]:
            warn("mesh is not watertight: run mesh.repair() before slicing")
        if not c["outward"]:
            warn("negative volume: face orientation is inverted (mesh.flipped())")
        if c["components"] > 1:
            warn(f"{c['components']} disconnected components: floating pieces will "
                 f"detach in the vat (mesh.remove_islands())")
        if c["edge_median"] < 0.25 * pixel_mm:
            warn(f"median edge {c['edge_median']:.4f} mm is far below the pixel: "
                 f"mesh.simplify(edge_length={pixel_mm:g}) will shrink the file "
                 f"({c['stl_size_MB']:.0f} MB) without visible change")
        if c["n_faces"] > 5_000_000:
            warn("more than 5 M faces: many slicers become slow or crash")
    return rep


def format_report(rep: dict) -> str:
    L = [f"DLP check  (pixel {rep['pixel_mm']} mm, layer {rep['layer_mm']} mm)"]
    if "grid_N" in rep:
        L.append(f"  field grid N={rep['grid_N']}, voxel {rep['voxel_mm']:.4f} mm")
    if "features_mm" in rep:
        f = rep["features_mm"]
        items = [(k, f[k]) for k in ("strut_radius", "node_radius", "min_thickness",
                                     "max_thickness", "pore_radius", "pore_throat") if k in f]
        L.append("  feature sizes (mm): " + ", ".join(f"{k} {v:.3f}" for k, v in items))
        if "min_solid_px" in rep:
            L.append(f"  thinnest solid = {rep['min_solid_px']:.1f} px / "
                     f"{rep['min_solid_layers']:.1f} layers; narrowest pore = "
                     f"{rep.get('min_pore_px', float('nan')):.1f} px")
    if "mesh" in rep:
        c = rep["mesh"]
        e = rep["extents_mm"]
        L.append(f"  part {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} mm, {rep['n_layers']} layers, "
                 f"volume {rep['volume_mm3']:.1f} mm^3, ~{rep['mass_g']:.2f} g resin")
        L.append(f"  mesh: {c['n_faces']:,} faces ({c['stl_size_MB']:.1f} MB STL), watertight "
                 f"{c['watertight']}, {c['components']} component(s)")
    if rep["warnings"]:
        L.append("  WARNINGS:")
        L += [f"   - {w}" for w in rep["warnings"]]
    else:
        L.append("  no problems found")
    return "\n".join(L)
