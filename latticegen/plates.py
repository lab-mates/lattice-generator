"""
Solid plates on the top and/or bottom of a lattice block, e.g. as loading
platens for mechanical tests.

Each plate is the block's footprint (a parallelogram spanned by ``nx a1`` and
``ny a2``; a rectangle for 90 degree cells), scaled about its centre and
extruded by the plate thickness.  It is welded to the lattice by a mesh
boolean union (``manifold3d``), so the result is one watertight solid.

Where the plates sit:

* field meshes (``mesh_stack``, ``mesh_instanced``) end in flat caps on the
  block faces; the plates are placed against those faces;
* strut meshes (``mesh_struts``) end in whole beams and joints that reach
  one beam radius beyond the outermost nodes.  A plate placed below them
  would only touch the joints at a point, so instead its face goes through
  the centres of the lowest (highest) nodes: the joints and the beams lying
  in that plane are half embedded in the plate.
"""

from __future__ import annotations

import numpy as np

from . import meshops
from .cells import as_cell
from .mesh import LatticeMesh

WHERE = ("none", "bottom", "top", "both")

_UNION = None       # optional backend, e.g. Manifold's WebAssembly build under Pyodide
_LID = 1_000_000   # face id of the temporary lids (plates and lattice faces get small ids)


def set_union_backend(fn):
    """Replace the mesh boolean used by :func:`add_plates`.

    The default uses ``manifold3d``. The browser app registers Manifold's
    WebAssembly build through this hook.

    Parameters
    ----------
    fn : callable or None
        ``fn(verts, faces, face_id, plates) -> (verts, faces, face_id)``: the
        union of the closed mesh ``(verts, faces)`` with each convex plate,
        given as an (8, 3) array of corner points. ``face_id`` (uint32, one per
        face) must come back on the output faces that derive from them. None
        restores the default.
    """
    global _UNION
    _UNION = fn


def _union(verts, faces, face_id, plates):
    if _UNION is not None:
        return _UNION(verts, faces, face_id, plates)
    try:
        import manifold3d
    except ImportError:
        raise ImportError("plates need manifold3d (pip install manifold3d)") from None
    m = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(verts, np.float32),
        tri_verts=np.ascontiguousarray(faces, np.uint32),
        face_id=np.ascontiguousarray(face_id, np.uint32)))
    if m.status() != manifold3d.Error.NoError:
        raise ValueError(f"the mesh is not a closed solid ({m.status().name}); "
                         f"repair it before adding plates")
    parts = [manifold3d.Manifold.hull_points(np.asarray(p, np.float64)) for p in plates]
    m = manifold3d.Manifold.batch_boolean([m] + parts, manifold3d.OpType.Add)
    out = m.to_mesh()
    del m
    return (np.array(out.vert_properties[:, :3]), np.array(out.tri_verts),   # own, writable copies
            np.array(out.face_id))


def _band_union(verts, faces, plates, band):
    """Union of the plates with only the faces near them.

    ``band`` marks the faces that reach the plates' side of the planes. Their
    open boundary (shared with the untouched rest) is closed by temporary fan
    lids, which lie beyond the planes and so never meet the plates; after the
    boolean the lids are dropped and the result is stitched back to the rest
    by its original boundary vertices. Returns ``(verts, faces, added_volume)``,
    or None when this does not apply cleanly, and the caller then unites the
    whole mesh.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    B = faces[band]
    n = len(verts)
    e = B[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    key = e[:, 0] * n + e[:, 1]
    be = e[~np.isin(e[:, 1] * n + e[:, 0], key)]              # directed boundary edges of B
    if len(be) == 0 or len(np.unique(be[:, 0])) != len(be):    # nothing to split, or pinched loops
        return None
    used = np.unique(np.concatenate([B.ravel(), be.ravel()]))
    loc = np.full(n, -1, np.int64)
    loc[used] = np.arange(len(used))
    lb = loc[be]
    k = len(used)
    g = coo_matrix((np.ones(len(lb)), (lb[:, 0], lb[:, 1])), shape=(k, k))
    _, lab = connected_components(g, directed=False)
    on_loop = np.unique(lb.ravel())
    loops, lab_c = np.unique(lab[on_loop], return_inverse=True)   # only the boundary loops
    lab[on_loop] = lab_c
    n_loop = len(loops)
    apex = np.zeros((n_loop, 3))
    np.add.at(apex, lab[on_loop], verts[used[on_loop]].astype(np.float64))
    apex /= np.bincount(lab[on_loop], minlength=n_loop)[:, None]
    lid = np.stack([lb[:, 1], lb[:, 0], k + lab[lb[:, 0]]], axis=1)
    V = np.vstack([verts[used].astype(np.float64), apex]).astype(np.float32)
    F = np.vstack([loc[B], lid])
    fid = np.r_[np.zeros(len(B), np.uint32), np.full(len(lid), _LID, np.uint32)]
    v2, f2, id2 = _union(V, F, fid, plates)
    f2 = np.asarray(f2, np.int64)[np.asarray(id2) != _LID]

    # stitch: output vertices equal to a boundary vertex become that vertex
    bv = np.unique(be.ravel())
    v2 = np.asarray(v2, np.float32)
    both = np.vstack([verts[bv], v2])
    both[both == 0.0] = 0.0
    _, first, inv = np.unique(both.view(np.int32), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    hit = first[inv[len(bv):]]                 # first occurrence: a boundary vertex if < len(bv)
    remap = np.where(hit < len(bv), bv[np.minimum(hit, len(bv) - 1)], n + np.arange(len(v2)))
    new = remap[f2]
    # the new part must close exactly the hole that B leaves in the rest
    e2 = new[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    k2 = e2[:, 0] * (n + len(v2)) + e2[:, 1]
    open2 = e2[~np.isin(e2[:, 1] * (n + len(v2)) + e2[:, 0], k2)]
    if len(open2) != len(be) or not np.array_equal(
            np.sort(open2[:, 0] * (n + len(v2)) + open2[:, 1]),
            np.sort(be[:, 0] * (n + len(v2)) + be[:, 1])):
        return None
    allv = np.vstack([verts, v2])
    added = meshops.mesh_volume(allv, new) - meshops.mesh_volume(verts, B)   # the rest is unchanged
    n_rest = len(faces) - len(B)
    allf = np.empty((n_rest + len(new), 3), np.int64)
    np.compress(~band, faces, axis=0, out=allf[:n_rest])
    allf[n_rest:] = new
    keep = np.zeros(len(allv), bool)
    keep[allf.ravel()] = True
    idx = np.cumsum(keep) - 1
    for s in range(0, len(allf), 1_000_000):          # renumber in place: no second face array
        allf[s:s + 1_000_000] = idx[allf[s:s + 1_000_000]]
    return allv[keep], allf, added


def plate_corners(cell, cells, where="both", thickness=1.0, size=1.2, planes=None):
    """Corner points of the plates, without touching any mesh.

    Parameters
    ----------
    cell : UnitCell or float
    cells : tuple of 3 int
        Block size in cells.
    where : {'bottom', 'top', 'both', 'none'}, optional
    thickness : float, optional
        Plate thickness (mm).
    size : float, optional
        Length of each plate side as a multiple of the footprint side, >= 1
        (1.2 = 20 % wider and longer than the block, centred).
    planes : tuple of 2 float, optional
        Heights of the bottom plate's top face and the top plate's bottom
        face. Default: the block faces, ``0`` and ``nz`` times the cell
        height.

    Returns
    -------
    list of ndarray, shape (8, 3)
        One array per plate, bottom first.
    """
    cell = as_cell(cell)
    if where not in WHERE:
        raise ValueError(f"where must be one of {WHERE}, not {where!r}")
    if thickness <= 0:
        raise ValueError("plate thickness must be positive")
    if size < 1.0:
        raise ValueError("plate size must be at least 1 (the block footprint)")
    A = np.asarray(cell.A, float)
    if abs(A[0, 2]) > 1e-9 * cell.a_min or abs(A[1, 2]) > 1e-9 * cell.a_min:
        raise ValueError("plates need a cell with a1 and a2 in the xy plane "
                         "(as UnitCell.from_parameters builds it)")
    nx, ny, nz = (int(c) for c in cells)
    if planes is None:
        planes = (0.0, nz * A[2, 2])
    u, v = nx * A[0, :2], ny * A[1, :2]
    shift = A[2, :2] / A[2, 2]                      # xy drift of the block per unit height
    out = []
    for side in ("bottom", "top"):
        if where not in (side, "both"):
            continue
        z0 = float(planes[0] if side == "bottom" else planes[1])
        z = (z0 - thickness, z0) if side == "bottom" else (z0, z0 + thickness)
        centre = shift * z0 + 0.5 * (u + v)
        pts = [np.r_[centre + 0.5 * size * (i * u + j * v), zz]
               for i in (-1, 1) for j in (-1, 1) for zz in z]
        out.append(np.array(pts))
    return out


def add_plates(mesh, cell, cells, where="both", thickness=1.0, size=1.2, planes=None):
    """Weld solid plates to the top and/or bottom of a lattice block.

    Parameters
    ----------
    mesh : LatticeMesh
        A watertight block, e.g. from :func:`mesh_stack`,
        :func:`mesh_instanced` or :func:`mesh_struts`. Remove floating
        slivers first (:meth:`LatticeMesh.remove_islands`): any that touch
        a block face would be welded onto the plate.
    cell : UnitCell or float
        The cell the block was built from.
    cells : tuple of 3 int
        Block size in cells.
    where : {'bottom', 'top', 'both', 'none'}, optional
    thickness : float, optional
        Plate thickness (mm).
    size : float, optional
        Plate side over footprint side, >= 1 (default 1.2: centred, 20 %
        wider and longer than the block).
    planes : tuple of 2 float, optional
        Heights of the faces the lattice attaches to. Default: for strut
        meshes the lowest and highest node centres (``info['node_z']``), so
        the joints there are half embedded; otherwise the lowest and highest
        points of the mesh, i.e. its flat caps on the block faces.

    Returns
    -------
    LatticeMesh
        One closed solid. ``info`` gains ``plates``, ``plate_thickness``,
        ``plate_size`` and ``plate_volume`` (the volume the plates added), and
        :attr:`LatticeMesh.relative_density` leaves the plates out.

    Raises
    ------
    ValueError
        For a size below 1, a non-positive thickness, a cell whose a1/a2 are
        not horizontal, or a mesh that is not a closed solid.

    Examples
    --------
    >>> import latticegen as lg
    >>> cell = lg.UnitCell.cubic(1.0)
    >>> m = lg.mesh_struts(cell, "bcc", rho=0.2, cells=(2, 2, 2), sides=6)
    >>> p = lg.add_plates(m, cell, (2, 2, 2), "both", thickness=0.5, size=1.2)
    >>> p.is_watertight, [round(float(x), 3) for x in p.extents]
    (True, [2.4, 2.4, 3.0])
    """
    if where == "none":
        return mesh
    if planes is None:
        planes = mesh.info.get("plate_planes") or mesh.info.get("node_z")
    if planes is None:
        # the caps' own float32 heights: the union is then exactly coplanar
        planes = (float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2]))
    plates = plate_corners(cell, cells, where, thickness, size, planes)
    # only faces reaching within ~half a beam of the planes can meet the plates
    h = 0.05 * float(np.asarray(as_cell(cell).A)[2, 2])
    z = mesh.verts[:, 2]
    zf = [z[mesh.faces[:, i]] for i in range(3)]
    band = np.zeros(mesh.n_faces, bool)
    if where in ("bottom", "both"):
        band |= np.minimum(np.minimum(zf[0], zf[1]), zf[2]) < planes[0] + h
    if where in ("top", "both"):
        band |= np.maximum(np.maximum(zf[0], zf[1]), zf[2]) > planes[1] - h
    del zf
    res = _band_union(mesh.verts, mesh.faces, plates, band) if not band.all() else None
    if res is None:
        v, f, _ = _union(mesh.verts, mesh.faces, np.zeros(mesh.n_faces, np.uint32), plates)
        added = meshops.mesh_volume(v, f) - mesh.volume
    else:
        v, f, added = res
    out = LatticeMesh(v, f, mesh.info)
    done = mesh.info.get("plates", "none")
    out.info.update(plates="both" if done not in ("none", where) else where,
                    plate_planes=tuple(planes), plate_thickness=float(thickness),
                    plate_size=float(size), plate_volume=mesh.info.get("plate_volume", 0.0) + added)
    out.info.pop("analytic_volume", None)
    return out
