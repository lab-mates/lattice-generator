"""
Instanced meshing of HOMOGENEOUS, uniform-density lattice blocks.

For a homogeneous block the mesh is periodic, so instead of marching the
whole block: mesh ONE unit cell (open at the cell faces), simplify it while
keeping the welding interfaces consistent, then replicate with integer
offsets and weld exactly.  Memory and time are independent of the block
size, and the result can be 10-50x leaner than a marching-cubes block.

Two simplification levels:

* ``seam_spacing=None`` (frozen seams): interior decimation / remeshing
  with the boundary loops frozen at marching-cubes density -- robust but
  the seams stay dense.
* ``seam_spacing=h`` (full): the seam curves themselves (periodic face
  loops, cap rims, cube-edge chains) are subsampled at arc spacing ~h
  (cell length units), with keep/drop decisions made ONCE on the quotient seam
  graph so every piece (body, caps, periodic partners) inherits identical
  decisions; interiors AND caps are then remeshed against the coarse
  frozen seams: isotropically (near-uniform triangles everywhere) or, with
  ``remesh="decimate"``, by quadric edge collapse that keeps long triangles
  along straight struts (typically 3-5x fewer faces).

Not applicable to graded / heterogeneous blocks -- use :func:`mesh_stack`.
Needs ``pymeshlab`` for the remeshing steps (raw instancing works without).
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from skimage.measure import marching_cubes

from .field import UnitField
from .mesh import LatticeMesh
from .meshops import weld, mesh_volume, mesh_area, open_edges as _open_edges


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _boundary_vertex_ids(faces):
    return np.unique(_open_edges(faces))


def _canon(verts, N):
    """Quotient coordinates (float32): boundary coordinate N maps to 0,
    -0.0 normalised.  Periodic partners get bitwise-identical rows."""
    q = np.ascontiguousarray(verts, np.float32).copy()
    q[q == np.float32(N)] = 0.0
    q[q == 0.0] = 0.0
    return q


def _keys(q):
    """Hashable per-row keys of a float32 (n,3) array."""
    q = np.ascontiguousarray(q, np.float32).copy()
    q.flags.writeable = False
    return q.view([("", np.float32)] * 3).ravel()


def _cls(q):
    """Per-vertex bitmask of quotient planes the vertex lies on."""
    c = np.zeros(len(q), np.int32)
    for ax in range(3):
        c |= (q[:, ax] == 0.0).astype(np.int32) << ax
    return c


def _chains(edges, pos, cls):
    """Decompose a boundary graph into chains.  Breakpoints: degree != 2 or
    plane-class differs from a neighbour.  Deterministic ordering via pos."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    brk = set()
    for v, ns in adj.items():
        if len(ns) != 2 or any(cls[v] != cls[u] for u in ns):
            brk.add(v)
    seen = set()
    ek = lambda a, b: (a, b) if a < b else (b, a)
    key = lambda v: tuple(pos[v])
    chains = []
    for s in sorted(brk, key=key):
        for nb in sorted(adj[s], key=key):
            if ek(s, nb) in seen:
                continue
            ch, prev, cur = [s, nb], s, nb
            seen.add(ek(s, nb))
            while cur not in brk:
                nxt = [u for u in adj[cur] if u != prev]
                if not nxt:
                    break
                nxt = nxt[0]
                seen.add(ek(cur, nxt))
                ch.append(nxt)
                prev, cur = cur, nxt
            chains.append(ch)
    for v in sorted(adj, key=key):                 # pure loops
        if v in brk:
            continue
        for nb in adj[v]:
            if ek(v, nb) in seen:
                continue
            ch, prev, cur = [v, nb], v, nb
            seen.add(ek(v, nb))
            while cur != v:
                nxt = [u for u in adj[cur] if u != prev][0]
                seen.add(ek(cur, nxt))
                ch.append(nxt)
                prev, cur = cur, nxt
            loop = ch[:-1]
            i0 = min(range(len(loop)), key=lambda i: key(loop[i]))
            loop = loop[i0:] + loop[:i0]
            if len(loop) > 2 and key(loop[1]) > key(loop[-1]):
                loop = [loop[0]] + loop[1:][::-1]
            chains.append(loop + [loop[0]])
            break
    return chains


def _keep_along(chain, pos, spacing, closed):
    """Greedy arc-length subsampling; endpoints always kept."""
    keep = [chain[0]]
    acc = 0.0
    last = chain[0]
    for v in chain[1:-1]:
        acc += float(np.linalg.norm(pos[v] - pos[last]))
        last = v
        if acc >= spacing:
            keep.append(v)
            acc = 0.0
    if not closed:
        keep.append(chain[-1])
    return keep


# --------------------------------------------------------------------------- #
# seam decisions on the quotient graph
# --------------------------------------------------------------------------- #

def seam_decisions(pieces, N, spacing):
    """Decide once, on the quotient seam graph, which seam vertices survive
    so that all pieces and periodic partners agree.  Returns {key: bool}."""
    node_pos = {}
    edge_set = set()
    for v, fc in pieces:
        q = _canon(v, N)
        k = _keys(q)
        for a, b in _open_edges(fc):
            ka, kb = k[a], k[b]
            node_pos[ka] = q[a]
            node_pos[kb] = q[b]
            edge_set.add((ka, kb) if ka.tobytes() < kb.tobytes() else (kb, ka))
    nodes = list(node_pos.keys())
    idx = {kk: i for i, kk in enumerate(nodes)}
    pos = np.array([node_pos[kk] for kk in nodes], np.float32)
    cls = _cls(pos)
    edges = np.array([(idx[a], idx[b]) for a, b in edge_set], int)
    keep = {kk: False for kk in nodes}
    for ch in _chains(edges, pos, cls):
        closed = ch[0] == ch[-1] and len(ch) > 2
        for v in _keep_along(ch, pos, spacing, closed):
            keep[nodes[v]] = True
    return keep


def _orient_sign(face, sorted_face):
    a, b, c = sorted_face
    t = tuple(face)
    return 1 if t in ((a, b, c), (b, c, a), (c, a, b)) else -1


def _dedup_fins(f2):
    """Remove duplicate faces: opposite-orientation pairs (zero-thickness
    fins) are both removed; same-orientation duplicates keep one."""
    key = np.sort(f2, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    keep = np.ones(len(f2), bool)
    if np.any(cnt > 1):
        byg = defaultdict(list)
        for i, g in enumerate(inv):
            if cnt[g] > 1:
                byg[g].append(i)
        for g, ids in byg.items():
            signs = [_orient_sign(f2[i], key[ids[0]]) for i in ids]
            if len(ids) == 2 and signs[0] != signs[1]:
                keep[ids] = False
            else:
                keep[ids[1:]] = False
    return f2[keep]


def simplify_piece(verts, faces, keep, N, snap_frac=0.4, spacing=None):
    """Collapse non-kept boundary vertices into the nearest kept vertex
    along their own chain; snap interior vertices closer than
    snap_frac*spacing to the seam onto the nearest kept seam vertex.
    Drops degenerate faces."""
    from scipy.spatial import cKDTree
    q = _canon(verts, N)
    k = _keys(q)
    edges = _open_edges(faces)
    if len(edges) == 0:
        return faces
    cls = _cls(q)
    vmap = np.arange(len(verts))
    bset = np.unique(edges)
    kept_ids = []
    for ch in _chains(edges, q, cls):
        closed = ch[0] == ch[-1] and len(ch) > 2
        seq = ch[:-1] if closed else ch
        kept_flags = [bool(keep.get(k[v], True)) for v in seq]
        if closed and not any(kept_flags):
            kept_flags[0] = True
        arc = np.zeros(len(seq))
        for i in range(1, len(seq)):
            arc[i] = arc[i - 1] + np.linalg.norm(q[seq[i]] - q[seq[i - 1]])
        total = arc[-1] + (np.linalg.norm(q[seq[0]] - q[seq[-1]]) if closed else 0.0)
        kept_i = [i for i, kf in enumerate(kept_flags) if kf]
        kept_ids += [seq[i] for i in kept_i]
        for i, v in enumerate(seq):
            if kept_flags[i]:
                continue
            if closed:
                d = [min(abs(arc[i] - arc[j]), total - abs(arc[i] - arc[j])) for j in kept_i]
            else:
                d = [abs(arc[i] - arc[j]) for j in kept_i]
            vmap[v] = seq[kept_i[int(np.argmin(d))]]
    snapped = np.array([], int)
    if spacing is not None and snap_frac > 0:
        kept_ids = np.unique(kept_ids)
        tree = cKDTree(verts[kept_ids])
        interior = np.setdiff1d(np.arange(len(verts)), bset)
        d, j = tree.query(verts[interior])
        near = d < snap_frac * spacing
        snapped = interior[near]
        vmap[snapped] = kept_ids[j[near]]

    expected_b = np.sort(vmap[edges], axis=1)
    expected_b = {tuple(e) for e in expected_b[expected_b[:, 0] != expected_b[:, 1]]}
    for _ in range(4):
        f2 = vmap[faces]
        ok = ((f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 2] != f2[:, 0]))
        f2 = _dedup_fins(f2[ok])
        e = np.sort(f2[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        eu, cnt = np.unique(e, axis=0, return_counts=True)
        bad = eu[(cnt > 2) | ((cnt == 1) & ~np.array([tuple(x) in expected_b for x in eu]))]
        if len(bad) == 0 or len(snapped) == 0:
            break
        bad_v = set(bad.ravel().tolist())
        revert = snapped[np.isin(vmap[snapped], list(bad_v))]
        if len(revert) == 0:
            break
        vmap[revert] = revert
        snapped = np.setdiff1d(snapped, revert)
    return f2


# --------------------------------------------------------------------------- #
# interior remeshing against frozen borders (pymeshlab)
# --------------------------------------------------------------------------- #

def remesh_interior(verts, faces, targetlen, N, iterations=10):
    """Isotropic explicit remeshing of the interior with the border frozen.

    Faces containing a boundary edge are frozen first (leaves the sliver
    band next to the seams remeshable).  If the remesher nevertheless moves
    a border vertex -- which happens occasionally at targets of only a few
    voxels -- the pass is repeated with every face touching a boundary
    vertex frozen as well.  The boundary canonical-key set must be
    unchanged afterwards.
    """
    from .remesh import _pymeshlab, _abs_len
    pymeshlab = _pymeshlab()
    faces = np.asarray(faces)
    bverts = _boundary_vertex_ids(faces)
    b_before = set(_keys(_canon(verts[bverts], N)).tolist())
    be = _open_edges(faces)
    bekeys = set(map(tuple, be))
    fe = np.sort(np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=2)
    frozen_edge = np.zeros(len(faces), bool)
    for i in range(3):
        frozen_edge |= np.array([tuple(p) in bekeys for p in fe[:, i, :]])
    frozen_vertex = np.isin(faces, bverts).any(axis=1)

    for frozen in (frozen_edge, frozen_vertex):
        fs = np.where(frozen, 0.0, 1.0)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, np.float64),
                                   face_matrix=faces, f_scalar_array=fs))
        ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
        ms.meshing_isotropic_explicit_remeshing(
            selectedonly=True, iterations=iterations, reprojectflag=True,
            adaptive=False, targetlen=_abs_len(targetlen))
        m = ms.current_mesh()
        v2 = m.vertex_matrix().astype(np.float32)
        f2 = m.face_matrix()
        b_after = set(_keys(_canon(v2[_boundary_vertex_ids(f2)], N)).tolist())
        if b_before == b_after:
            return v2, f2
    raise RuntimeError("border changed during interior remeshing; try a larger "
                       "seam_spacing / edge_length")


def decimate_interior(verts, faces, target_faces, N, quality_thr=0.1):
    """Quadric edge-collapse decimation of the interior with the border frozen.

    The border is protected by ``preserveboundary`` with a very large
    boundary weight (pymeshlab's selected-only mode loses its selection as
    it collapses and stalls far above the target); ``target_faces`` is the
    budget for the whole piece.  A low ``quality_thr`` lets the triangles
    stretch along straight struts, which is where the savings over
    isotropic remeshing come from.  The boundary canonical-key set must be
    unchanged afterwards (checked; RuntimeError otherwise).
    """
    from .remesh import _pymeshlab
    pymeshlab = _pymeshlab()
    faces = np.asarray(faces)
    bverts = _boundary_vertex_ids(faces)
    b_before = set(_keys(_canon(verts[bverts], N)).tolist())
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, np.float64), face_matrix=faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=max(int(target_faces), 4), preserveboundary=True, boundaryweight=1e6,
        preservetopology=True, preservenormal=True, optimalplacement=True,
        planarquadric=True, qualitythr=float(quality_thr), autoclean=True)
    m = ms.current_mesh()
    v2 = m.vertex_matrix().astype(np.float32)
    f2 = m.face_matrix()
    b_after = set(_keys(_canon(v2[_boundary_vertex_ids(f2)], N)).tolist())
    if b_before != b_after:
        raise RuntimeError("border changed during interior decimation; try a larger "
                           "seam_spacing / edge_length")
    return v2, f2


def remesh_frozen(verts, faces, method="isotropic", reduction=10, quality_thr=0.3,
                  targetlen=None, targetlen_pct=1.0, iterations=5):
    """Remesh / decimate the unit body with the FULL marching-density
    boundary frozen (legacy 'frozen seams' path).

    targetlen : isotropic target edge in index units (voxels); if None,
                ``targetlen_pct`` percent of the bounding box is used.
    """
    from .remesh import _pymeshlab, _abs_len
    pymeshlab = _pymeshlab()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, np.float64),
                               face_matrix=np.asarray(faces)))
    if method == "decimate":
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(len(faces) / reduction), preserveboundary=True,
            boundaryweight=1e6, preservenormal=True, planarquadric=True,
            qualitythr=quality_thr)
    elif method == "isotropic":
        ms.compute_selection_from_mesh_border()
        for _ in range(2):
            ms.apply_selection_dilatation()
        ms.apply_selection_inverse(invfaces=True, invverts=True)
        tl = (pymeshlab.PercentageValue(targetlen_pct) if targetlen is None
              else _abs_len(targetlen))
        ms.meshing_isotropic_explicit_remeshing(
            selectedonly=True, iterations=iterations, reprojectflag=True, targetlen=tl)
    else:
        raise ValueError(f"unknown method {method!r}")
    m = ms.current_mesh()
    v2, f2 = m.vertex_matrix(), m.face_matrix()
    k1 = {tuple(x) for x in verts[_boundary_vertex_ids(faces)]}
    k2 = {tuple(x) for x in v2[_boundary_vertex_ids(f2)]}
    if k1 != k2:
        raise RuntimeError("boundary loops changed during remeshing")
    return v2.astype(np.float32), f2


# --------------------------------------------------------------------------- #
# unit pieces in global (index) coordinates
# --------------------------------------------------------------------------- #

def periodic_unit_mesh(f, t):
    """Open mesh of one unit cell body (index units, spans [0, N]^3)."""
    level = 1e-6 * (f.max() - f.min())
    F = np.pad(f, ((0, 1), (0, 1), (0, 1)), mode="wrap") - t
    v, fc, _, _ = marching_cubes(F, level=level)
    return v.astype(np.float32), fc


def _unit_caps_global(f, t):
    """Six cap meshes for ONE cell in global index-unit coordinates:
    list of (axis, side, verts, faces) with consistent winding."""
    N = f.shape[0]
    level = 1e-6 * (f.max() - f.min())
    pad_val = np.float32(f.min() - t - (f.max() - f.min()))
    slices = [np.pad(f[0, :, :] - t, ((0, 1), (0, 1)), mode="wrap"),
              np.pad(f[:, 0, :] - t, ((0, 1), (0, 1)), mode="wrap"),
              np.pad(f[:, :, 0] - t, ((0, 1), (0, 1)), mode="wrap")]
    perm = {0: ([2, 0, 1], False), 1: ([0, 2, 1], True), 2: ([0, 1, 2], False)}
    out = []
    for ax in range(3):
        cols, odd = perm[ax]
        for side in (-1, +1):
            sl = slices[ax]
            M = sl.shape[0]
            F = np.empty((M, M, 2), np.float32)
            if side < 0:
                F[:, :, 0], F[:, :, 1] = pad_val, sl
            else:
                F[:, :, 0], F[:, :, 1] = sl, pad_val
            if not (F.min() < level < F.max()):
                continue
            v, fc, _, _ = marching_cubes(F, level=level)
            v[:, 2] = 0.0 if side < 0 else np.float32(N)
            v = v[:, cols].astype(np.float32)
            out.append((ax, side, v, fc[:, ::-1] if odd else fc))
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def mesh_instanced(field: UnitField, rho=0.3, cells=(1, 1, 1), seam_spacing=None,
                   edge_length=None, remesh="isotropic", density_correct=True,
                   exact_density=True, anisotropy=4.0, quality_thr=0.1, verbose=False,
                   **remesh_kw) -> LatticeMesh:
    """Lean mesh of a homogeneous (nx, ny, nz) block by unit-cell instancing.

    seam_spacing : arc spacing of the simplified seam curves, in the CELL
                   LENGTH UNIT (mm), like every other length in the API.
                   If set, the seam curves (periodic face loops, cap rims,
                   cube-edge chains) are subsampled consistently on the
                   quotient graph and the interiors of body and caps are
                   isotropically remeshed at ``edge_length``.  Useful range:
                   1/16 .. 1/8 of the cell edge.  This is the strongest,
                   most uniform simplification.
    edge_length  : target triangle edge of the interior remesh (mm).
                   Default: ``seam_spacing`` when that is given, otherwise
                   pymeshlab's 1 % of the bounding box.  With
                   ``seam_spacing=None`` it sets the interior target of the
                   frozen-seams isotropic remesh (the seams themselves stay
                   at marching-cubes density).
    remesh       : how the interiors are simplified.  With ``seam_spacing``
                   set: 'isotropic' (uniform triangles of ``edge_length``)
                   or 'decimate' (quadric edge collapse to the face budget
                   of an isotropic mesh divided by ``anisotropy``; long
                   triangles along straight struts, 3-5x fewer faces).
                   With ``seam_spacing=None``: 'isotropic', 'decimate'
                   (seams frozen at marching density) or None / False for
                   raw instancing without pymeshlab.
                   Both lengths must be at least one voxel
                   (``field.voxel_size``); smaller values almost always mean
                   a unit mix-up and raise a ValueError.
    anisotropy   : 'decimate' with seams: allowed mean triangle elongation
                   (face budget divisor).
    quality_thr  : 'decimate' with seams: minimum triangle quality accepted
                   by the collapse (0..1; lower = longer triangles).
    density_correct : simplification shifts the volume slightly (coarse
                   seams inscribe the strut sections, smoothing shrinks);
                   measure the unit-cell density and iterate the isovalue
                   (secant, up to 4 rebuilds) until it matches ``rho``.
    exact_density : calibrate the starting isovalue on the meshed unit-cell
                   volume rather than the voxel quantile.

    Returns a welded :class:`LatticeMesh` with outward normals.
    """
    f = field.f
    N = field.N
    A = field.A
    nx, ny, nz = (int(c) for c in cells)
    n_ax = (nx, ny, nz)
    if not (0.0 < float(rho) < 1.0):
        raise ValueError("rho must lie strictly inside (0, 1)")

    # lengths come in cell units (mm); the unit pieces live in voxel units
    vox = float(np.mean(field.voxel_size))

    def to_voxels(value, name):
        if value is None:
            return None
        v = float(value) / vox
        if v < 1.0:
            raise ValueError(f"{name}={value} is smaller than one voxel ({vox:.4g} cell units); "
                             f"lengths are in the cell length unit (mm), not voxels")
        return v

    seam_vox = to_voxels(seam_spacing, "seam_spacing")
    edge_vox = to_voxels(edge_length, "edge_length")
    if edge_length is not None and seam_spacing is None and remesh != "isotropic":
        warnings.warn("edge_length is only used by the isotropic remesh; with "
                      f"remesh={remesh!r} it has no effect", RuntimeWarning, stacklevel=2)

    def build(t):
        v0, f0 = periodic_unit_mesh(f, t)
        caps = _unit_caps_global(f, t)
        if seam_vox is not None:
            tl = seam_vox if edge_vox is None else edge_vox
            pieces = [(v0, f0)] + [(v, fc) for _, _, v, fc in caps]
            keep = seam_decisions(pieces, N, seam_vox)

            def interior(v, fc):
                if remesh == "decimate":
                    budget = mesh_area(v, fc) / (np.sqrt(3) / 4 * tl * tl) / float(anisotropy)
                    return decimate_interior(v, fc, int(np.ceil(budget)), N, quality_thr)
                return remesh_interior(v, fc, tl, N)

            f0 = simplify_piece(v0, f0, keep, N, spacing=seam_vox)
            v0, f0 = interior(v0, f0)
            caps2 = []
            for ax, side, v, fc in caps:
                fc = simplify_piece(v, fc, keep, N, spacing=seam_vox)
                v, fc = interior(v, fc)
                caps2.append((ax, side, v, fc))
            caps = caps2
        elif remesh:
            kw = dict(remesh_kw)
            if remesh == "isotropic" and edge_vox is not None:
                kw.setdefault("targetlen", edge_vox)
            v0, f0 = remesh_frozen(v0, f0, method=remesh, **kw)
        return v0, f0, caps

    def unit_density(v0, f0, caps):
        vv, ff, nvv = [v0.copy()], [f0.copy()], len(v0)
        for _, _, v, fc in caps:
            vv.append(v)
            ff.append(fc + nvv)
            nvv += len(v)
        v1, f1 = weld(np.concatenate(vv), np.concatenate(ff).astype(np.int64))
        return mesh_volume(v1, f1[:, ::-1]) / N ** 3

    t = field.calibrated_threshold(rho) if exact_density else float(field.threshold(rho))
    v0, f0, caps = build(t)
    if (seam_spacing is not None or remesh) and density_correct:
        # simplification / remeshing shifts the volume slightly (coarse seams
        # inscribe the strut sections, smoothing shrinks); secant on the
        # isovalue against the measured unit-cell density.
        fmin, fmax = float(f.min()), float(f.max())
        t0, r0 = t, unit_density(v0, f0, caps)
        if verbose:
            print(f"  unit-cell density {r0:.4f} (target {rho:.4f})")
        if abs(r0 - rho) > 2e-4:
            d = 0.02
            slope = float(field.threshold(max(rho - d, 1e-3)) - field.threshold(min(rho + d, 1 - 1e-3))) / (2 * d)
            if not slope > 0:
                slope = 0.1 * (fmax - fmin)
            t1 = t0 - (rho - r0) * slope
            best = (v0, f0, caps, t0, abs(r0 - rho))
            for _ in range(4):
                t1 = min(max(t1, fmin + 1e-6 * (fmax - fmin)), fmax - 1e-6 * (fmax - fmin))
                v1, f1, caps1 = build(t1)
                r1 = unit_density(v1, f1, caps1)
                if verbose:
                    print(f"  -> isovalue {t1:.6g}: density {r1:.4f}")
                if abs(r1 - rho) < best[4]:
                    best = (v1, f1, caps1, t1, abs(r1 - rho))
                if abs(r1 - rho) <= 2e-4 or r1 == r0:
                    break
                t2 = t1 - (r1 - rho) * (t1 - t0) / (r1 - r0)
                t0, r0, t1 = t1, r1, t2
            v0, f0, caps, t, _ = best

    all_v, all_f, nv = [], [], 0

    def add(v, fc, off):
        nonlocal nv
        if len(fc) == 0:
            return
        all_v.append((v + np.asarray(off, np.float32)).astype(np.float32))
        all_f.append(fc + nv)
        nv += len(v)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                add(v0, f0, (i * N, j * N, k * N))
    for ax, side, v, fc in caps:
        p_ax, q_ax = [a for a in range(3) if a != ax]
        w_off = 0 if side < 0 else (n_ax[ax] - 1) * N
        for p in range(n_ax[p_ax]):
            for q in range(n_ax[q_ax]):
                off = [0, 0, 0]
                off[ax] = w_off
                off[p_ax] = p * N
                off[q_ax] = q * N
                add(v, fc, off)

    verts = np.concatenate(all_v)
    faces = np.concatenate(all_f).astype(np.int64)
    verts, faces = weld(verts, faces)
    verts = (verts / N) @ np.asarray(A, np.float32)
    info = dict(mode="instanced", threshold=t, cells=(nx, ny, nz), N=N, A=A.copy(),
                box_volume=field.cell.volume * nx * ny * nz, seam_spacing=seam_spacing,
                edge_length=edge_length, remesh=remesh)
    mesh = LatticeMesh(verts, faces[:, ::-1], info=info)
    from .meshing import _warn_if_disconnected
    _warn_if_disconnected(mesh)
    return mesh
