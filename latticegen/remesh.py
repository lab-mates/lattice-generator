"""
Simplification, remeshing and repair of lattice meshes.

Marching cubes produces far more triangles than a printer can use: a
5x5x5 octet block at N = 96 has several million faces.  These tools bring
the count down while keeping the geometry within a controlled tolerance:

decimate          quadric edge-collapse to a target face count / fraction
simplify          decimate to a target edge length (computed from the area)
remesh_isotropic  uniform, FEA-friendly triangles of a chosen edge length
repair            duplicate / degenerate / non-manifold clean-up, hole filling
remove_islands    drop small disconnected pieces (they would float in the vat)

All functions take and return :class:`LatticeMesh`.  They need
``pymeshlab`` (``conda install -c conda-forge pymeshlab``).
"""

from __future__ import annotations

import warnings

import numpy as np

from .mesh import LatticeMesh
from . import meshops


def _pymeshlab():
    try:
        import pymeshlab
    except ImportError as e:                       # pragma: no cover
        raise ImportError("this operation needs pymeshlab: "
                          "conda install -c conda-forge pymeshlab") from e
    return pymeshlab


def _abs_len(x):
    pml = _pymeshlab()
    for name in ("PureValue", "AbsoluteValue"):
        if hasattr(pml, name):
            return getattr(pml, name)(float(x))
    raise RuntimeError("no absolute-length value class in pymeshlab")


def _to_meshset(mesh: LatticeMesh, face_scalars=None):
    pml = _pymeshlab()
    ms = pml.MeshSet()
    kw = {}
    if face_scalars is not None:
        kw["f_scalar_array"] = np.asarray(face_scalars, np.float64)
    ms.add_mesh(pml.Mesh(vertex_matrix=mesh.verts.astype(np.float64),
                         face_matrix=mesh.faces.astype(np.int32), **kw))
    return ms


def _from_meshset(ms, info) -> LatticeMesh:
    m = ms.current_mesh()
    out = LatticeMesh(m.vertex_matrix(), m.face_matrix(), info)
    return out


def _snap_to_bounds(mesh: LatticeMesh, bounds, rel_tol=1e-4) -> LatticeMesh:
    """Vertices that pymeshlab moved off the original bounding planes by a
    tiny amount are put back, so block caps stay exactly planar."""
    v = mesh.verts.copy()
    ext = bounds[1] - bounds[0]
    for ax in range(3):
        tol = rel_tol * ext[ax]
        for plane in bounds[:, ax]:
            near = np.abs(v[:, ax] - plane) <= tol
            v[near, ax] = plane
    return LatticeMesh(v, mesh.faces, mesh.info)


def _check_after(name, before: LatticeMesh, after: LatticeMesh, verbose):
    if verbose:
        dv = (after.volume - before.volume) / max(abs(before.volume), 1e-30)
        print(f"{name}: faces {before.n_faces:,} -> {after.n_faces:,}  "
              f"({after.n_faces / max(before.n_faces, 1):.1%}), volume change {dv:+.3%}, "
              f"watertight {after.is_watertight}")
    if before.is_watertight and not after.is_watertight:
        warnings.warn(f"{name} produced open edges; try a milder setting or repair()",
                      RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# simplification
# --------------------------------------------------------------------------- #

def decimate(mesh: LatticeMesh, target_faces=None, fraction=None, quality_thr=0.3,
             preserve_topology=True, planar=True, planar_weight=0.001,
             preserve_normal=True, verbose=False) -> LatticeMesh:
    """Quadric edge-collapse decimation (Garland-Heckbert).

    target_faces / fraction : the face budget (one of the two).
    quality_thr  : minimum triangle quality accepted (0..1; higher = more
                   isotropic triangles, less reduction).
    preserve_topology : forbid collapses that change the genus or create
                   non-manifold configurations (keeps struts connected).
    planar       : add a planar-quadric term so flat regions (the block caps)
                   collapse into a few large triangles.
    """
    if target_faces is None:
        if fraction is None:
            raise ValueError("give target_faces or fraction")
        target_faces = int(round(mesh.n_faces * float(fraction)))
    target_faces = max(int(target_faces), 4)
    ms = _to_meshset(mesh)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces, qualitythr=float(quality_thr),
        preservenormal=preserve_normal, preservetopology=preserve_topology,
        optimalplacement=True, planarquadric=planar, planarweight=float(planar_weight),
        autoclean=True)
    out = _snap_to_bounds(_from_meshset(ms, mesh.info), mesh.bounds).weld()
    _check_after("decimate", mesh, out, verbose)
    return out


def target_faces_for_edge_length(mesh: LatticeMesh, edge_length: float) -> int:
    """Face count of an isotropic mesh of the same area with the given edge
    length (equilateral triangles: area = sqrt(3)/4 h^2)."""
    return int(np.ceil(mesh.area / (np.sqrt(3) / 4 * edge_length ** 2)))


def simplify(mesh: LatticeMesh, edge_length=None, fraction=None, target_faces=None,
             quality_thr=0.3, verbose=False, **kw) -> LatticeMesh:
    """Reduce the triangle count to what a print needs.

    Give ONE of:
      edge_length  : desired mean triangle edge (cell units, e.g. the DLP pixel
                     size) -- the face budget is derived from the surface area;
      fraction     : keep this fraction of the faces;
      target_faces : absolute budget.
    Uses quadric decimation with topology preservation and planar caps.
    """
    if edge_length is not None:
        target_faces = target_faces_for_edge_length(mesh, edge_length)
    return decimate(mesh, target_faces=target_faces, fraction=fraction,
                    quality_thr=quality_thr, verbose=verbose, **kw)


def remesh_isotropic(mesh: LatticeMesh, edge_length: float, iterations=10,
                     feature_deg=30.0, adaptive=False, reproject=True,
                     max_surface_dist=None, verbose=False) -> LatticeMesh:
    """Isotropic explicit remeshing (split / collapse / flip / smooth) to a
    uniform target edge length.  Sharp creases (dihedral angle above
    ``feature_deg``, e.g. cap rims) are preserved.  Slower than decimation
    but yields near-equilateral triangles suited to FEA.

    max_surface_dist : reprojection tolerance (default 0.5 * edge_length).
    """
    ms = _to_meshset(mesh)
    kw = dict(iterations=int(iterations), adaptive=adaptive, selectedonly=False,
              targetlen=_abs_len(edge_length), featuredeg=float(feature_deg),
              reprojectflag=reproject)
    if max_surface_dist is None:
        max_surface_dist = 0.5 * edge_length
    kw["checksurfdist"] = True
    kw["maxsurfdist"] = _abs_len(max_surface_dist)
    ms.meshing_isotropic_explicit_remeshing(**kw)
    out = _snap_to_bounds(_from_meshset(ms, mesh.info), mesh.bounds).weld()
    _check_after("remesh_isotropic", mesh, out, verbose)
    return out


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #

def remove_islands(mesh: LatticeMesh, min_faces=None, min_volume_fraction=None,
                   keep="all_large", verbose=False) -> LatticeMesh:
    """Drop small disconnected components.

    A lattice block should be ONE connected solid; slivers cut off at the
    boundary or pinched-off struts would float in the resin vat.

    min_faces            : components with fewer faces are removed.
    min_volume_fraction  : components whose |volume| is below this fraction
                           of the largest one are removed (default 1e-3
                           when neither criterion is given).
    keep='largest'       : keep only the largest component.
    """
    labels, n = mesh.components()
    if n <= 1:
        return mesh
    vols = np.zeros(n)
    counts = np.bincount(labels, minlength=n)
    for c in range(n):
        sel = labels == c
        vols[c] = abs(meshops.mesh_volume(mesh.verts, mesh.faces[sel]))
    big = int(np.argmax(vols))
    keep_mask = np.ones(n, bool)
    if keep == "largest":
        keep_mask[:] = False
        keep_mask[big] = True
    else:
        if min_faces is None and min_volume_fraction is None:
            min_volume_fraction = 1e-3
        if min_faces is not None:
            keep_mask &= counts >= int(min_faces)
        if min_volume_fraction is not None:
            keep_mask &= vols >= float(min_volume_fraction) * vols[big]
    sel = keep_mask[labels]
    out = LatticeMesh(mesh.verts, mesh.faces[sel], mesh.info)
    # drop unreferenced vertices
    used = np.unique(out.faces)
    remap = -np.ones(mesh.n_vertices, np.int64)
    remap[used] = np.arange(len(used))
    out = LatticeMesh(mesh.verts[used], remap[out.faces], mesh.info)
    if verbose:
        print(f"remove_islands: {n} components -> {int(keep_mask.sum())} kept "
              f"(removed {int((~keep_mask).sum())}, {int((~sel).sum()):,} faces)")
    return out


def repair(mesh: LatticeMesh, fill_holes=True, max_hole_edges=100, orient=True,
           verbose=False) -> LatticeMesh:
    """Clean-up pass: weld duplicate vertices, drop duplicate / degenerate
    faces, repair non-manifold edges, fill small holes, orient faces
    coherently with outward normals."""
    ms = _to_meshset(mesh)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_remove_unreferenced_vertices()
    if fill_holes:
        try:
            ms.meshing_close_holes(maxholesize=int(max_hole_edges), refinehole=False)
        except Exception as e:                                # pragma: no cover
            warnings.warn(f"hole filling failed: {e}", RuntimeWarning, stacklevel=2)
    if orient:
        ms.meshing_re_orient_faces_coherently()
    out = _from_meshset(ms, mesh.info).weld()
    if orient and out.volume < 0:
        out = out.flipped()
    if verbose:
        print(f"repair: watertight {mesh.is_watertight} -> {out.is_watertight}, "
              f"faces {mesh.n_faces:,} -> {out.n_faces:,}")
    return out
