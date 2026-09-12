"""
Instanced meshing of homogeneous periodic lattice stacks.

For a HOMOGENEOUS, uniform-density stack the mesh is periodic, so instead
of marching the whole block: mesh ONE unit cell (open at the cell faces),
simplify it while keeping the welding interfaces consistent, then
replicate with integer offsets and weld exactly.

Two simplification levels:

  * remesh='frozen'   — interior decimation/remeshing with the boundary
    loops bitwise frozen at marching-cubes density (robust; dense seams).
  * seam_spacing=h    — FULL simplification: the seam curves themselves
    (periodic face loops, cap rims, cube-edge chains) are subsampled at
    arc spacing ~h, with keep/drop decisions made ONCE on the quotient
    seam graph so every piece (body, caps, periodic partners) inherits
    identical decisions; interiors AND caps are then isotropically
    remeshed against the coarse frozen seams. Yields near-uniform
    FEA-friendly triangles everywhere at a fraction of the count.

Not applicable to graded/heterogeneous stacks (cells not congruent) —
use fourier_lattice.stacked_lattice_mesh for those.

    import fourier_lattice as fl, instanced_mesh as im
    A, hkl, c = fl.octet_struts(node_comp=0.5)
    f = fl.field_on_grid(hkl, c, 96)
    v, fc = im.instanced_stack_mesh(fl.cubic_cell(), f, rho=0.30,
                                    cells=(5, 5, 5), seam_spacing=5.0)
    fl.write_stl("octet_5x5x5.stl", v, fc)
"""

from collections import defaultdict
import numpy as np
from skimage.measure import marching_cubes
import fourier_lattice as fl


# ----------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------

def _open_edges(faces):
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    return eu[cnt == 1]


def _boundary_vertex_ids(faces):
    return np.unique(_open_edges(faces))


def _canon(verts, N):
    """Quotient coordinates (float32): boundary coordinate N maps to 0,
    -0.0 normalized. Periodic partners get bitwise-identical rows."""
    q = np.ascontiguousarray(verts, np.float32).copy()
    q[q == np.float32(N)] = 0.0
    q[q == 0.0] = 0.0
    return q


def _keys(q):
    """Hashable per-row keys of a float32 (n,3) array."""
    q = np.ascontiguousarray(q, np.float32).copy()
    q.flags.writeable = False                     # hashable void scalars
    return q.view([('', np.float32)] * 3).ravel()


def _cls(q):
    """Per-vertex bitmask of quotient planes the vertex lies on."""
    c = np.zeros(len(q), np.int32)
    for ax in range(3):
        c |= (q[:, ax] == 0.0).astype(np.int32) << ax
    return c


# ----------------------------------------------------------------------
# chain extraction (deterministic, orientation-canonical)
# ----------------------------------------------------------------------

def _chains(edges, pos, cls):
    """Decompose a boundary graph into chains. Breakpoints: degree != 2 or
    plane-class differs from a neighbour. Deterministic ordering via pos."""
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
    inner = chain[1:-1] if closed else chain[1:-1]
    for v in inner:
        acc += float(np.linalg.norm(pos[v] - pos[last]))
        last = v
        if acc >= spacing:
            keep.append(v)
            acc = 0.0
    if not closed:
        keep.append(chain[-1])
    return keep


# ----------------------------------------------------------------------
# seam decisions on the quotient graph
# ----------------------------------------------------------------------

def seam_decisions(pieces, N, spacing):
    """
    pieces : list of (verts, faces) in GLOBAL index units of one cell
             (body + 6 oriented unit caps).
    Returns a dict canonical-key -> bool (keep), decided once on the
    quotient seam graph so all pieces and periodic partners agree.
    """
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
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                            return_counts=True)
    keep = np.ones(len(f2), bool)
    dup_groups = np.where(cnt > 1)[0]
    if len(dup_groups):
        byg = defaultdict(list)
        for i, g in enumerate(inv):
            if cnt[g] > 1:
                byg[g].append(i)
        for g, ids in byg.items():
            signs = [_orient_sign(f2[i], key[ids[0]]) for i in ids]
            if len(ids) == 2 and signs[0] != signs[1]:
                keep[ids] = False                     # cancelling fin
            else:
                keep[ids[1:]] = False                 # keep one
    return f2[keep]


def simplify_piece(verts, faces, keep, N, snap_frac=0.4, spacing=None):
    """Collapse non-kept boundary vertices into the nearest kept vertex
    along their own chain; additionally snap interior vertices closer than
    snap_frac*spacing to the seam onto the nearest kept seam vertex (this
    removes the sliver strip the pure chain collapse leaves behind).
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
            kept_flags[0] = True                       # safety
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
                d = [min(abs(arc[i] - arc[j]), total - abs(arc[i] - arc[j]))
                     for j in kept_i]
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
        ok = ((f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2])
              & (f2[:, 2] != f2[:, 0]))
        f2 = _dedup_fins(f2[ok])
        e = np.sort(f2[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        eu, cnt = np.unique(e, axis=0, return_counts=True)
        bad = eu[(cnt > 2) | ((cnt == 1) &
                 ~np.array([tuple(x) in expected_b for x in eu]))]
        if len(bad) == 0 or len(snapped) == 0:
            break
        bad_v = set(bad.ravel().tolist())
        revert = snapped[np.isin(vmap[snapped], list(bad_v))]
        if len(revert) == 0:
            break
        vmap[revert] = revert                        # undo those snaps
        snapped = np.setdiff1d(snapped, revert)
    return f2


# ----------------------------------------------------------------------
# interior remeshing against frozen borders (pymeshlab)
# ----------------------------------------------------------------------

def _pml_len(x):
    import pymeshlab
    for name in ("PureValue", "AbsoluteValue"):
        if hasattr(pymeshlab, name):
            return getattr(pymeshlab, name)(float(x))
    raise RuntimeError("no absolute-length value class in pymeshlab")


def remesh_interior(verts, faces, targetlen, N, iterations=10):
    """Isotropic explicit remeshing of the interior with the border frozen;
    asserts the boundary canonical-key set is unchanged."""
    import pymeshlab
    b_before = set(_keys(_canon(verts[_boundary_vertex_ids(faces)], N)).tolist())
    # freeze ONLY faces that contain a boundary edge; everything else,
    # including the former sliver band, is remeshable
    be = _open_edges(faces)
    bekeys = set(map(tuple, be))
    fs = np.ones(len(faces), np.float64)
    fe = np.sort(np.stack([faces[:, [0, 1]], faces[:, [1, 2]],
                           faces[:, [2, 0]]], axis=1), axis=2)
    for i in range(3):
        pairs = fe[:, i, :]
        mask = np.array([tuple(p) in bekeys for p in pairs])
        fs[mask] = 0.0
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, np.float64),
                               face_matrix=np.asarray(faces),
                               f_scalar_array=fs))
    ms.compute_selection_by_condition_per_face(condselect='fq > 0.5')
    ms.meshing_isotropic_explicit_remeshing(
        selectedonly=True, iterations=iterations, reprojectflag=True,
        adaptive=False, targetlen=_pml_len(targetlen))
    m = ms.current_mesh()
    v2 = m.vertex_matrix().astype(np.float32)
    f2 = m.face_matrix()
    b_after = set(_keys(_canon(v2[_boundary_vertex_ids(f2)], N)).tolist())
    assert b_before == b_after, "border changed during interior remeshing"
    return v2, f2


def remesh_periodic(verts, faces, reduction=10, quality_thr=0.3,
                    method='decimate', targetlen_pct=1.0, iterations=5):
    """Legacy path: remesh with the FULL marching-density boundary frozen."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(verts, np.float64),
                               face_matrix=np.asarray(faces)))
    if method == 'decimate':
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(len(faces) / reduction),
            preserveboundary=True, boundaryweight=1e6,
            preservenormal=True, planarquadric=True, qualitythr=quality_thr)
    elif method == 'isotropic':
        import pymeshlab as pml
        ms.compute_selection_from_mesh_border()
        for _ in range(2):
            ms.apply_selection_dilatation()
        ms.apply_selection_inverse(invfaces=True, invverts=True)
        ms.meshing_isotropic_explicit_remeshing(
            selectedonly=True, iterations=iterations, reprojectflag=True,
            targetlen=pml.PercentageValue(targetlen_pct))
    else:
        raise ValueError(method)
    m = ms.current_mesh()
    v2, f2 = m.vertex_matrix(), m.face_matrix()
    k1 = {tuple(x) for x in verts[_boundary_vertex_ids(faces)]}
    k2 = {tuple(x) for x in v2[_boundary_vertex_ids(f2)]}
    assert k1 == k2, "boundary loops changed during remeshing"
    return v2, f2


# ----------------------------------------------------------------------
# unit pieces in global coordinates
# ----------------------------------------------------------------------

def periodic_unit_mesh(f, rho):
    """Open mesh of one unit cell (index units, spans [0, N]^3)."""
    N = f.shape[0]
    t = np.quantile(f, 1.0 - rho)
    level = 1e-6 * (f.max() - f.min())
    F = np.pad(f, ((0, 1), (0, 1), (0, 1)), mode='wrap') - t
    v, fc, _, _ = marching_cubes(F, level=level)
    return v.astype(np.float32), fc, t


def _unit_caps_global(f, t):
    """Six cap meshes for ONE cell, in global index-unit coordinates.
    Returns list of (axis, side, verts, faces) with consistent winding
    (same convention as the body: final global flip applied later)."""
    N = f.shape[0]
    level = 1e-6 * (f.max() - f.min())
    pad_val = np.float32(f.min() - t - (f.max() - f.min()))
    slices = [np.pad(f[0, :, :] - t, ((0, 1), (0, 1)), mode='wrap'),
              np.pad(f[:, 0, :] - t, ((0, 1), (0, 1)), mode='wrap'),
              np.pad(f[:, :, 0] - t, ((0, 1), (0, 1)), mode='wrap')]
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
            v[:, 2] = 0.0 if side < 0 else np.float32(N)   # onto the face
            if side < 0:
                pass
            v = v[:, cols].astype(np.float32)
            out.append((ax, side, v, fc[:, ::-1] if odd else fc))
    return out


# ----------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------

def instanced_stack_mesh(A, f, rho=0.3, cells=(1, 1, 1), remesh=True,
                         seam_spacing=None, targetlen=None,
                         density_correct=True, **remesh_kw):
    """
    Lean mesh of a homogeneous (nx, ny, nz) stack by unit-cell instancing.

    seam_spacing : if set (index units, e.g. N/16..N/8), the seam curves
                   are subsampled at this arc spacing (consistently on the
                   quotient graph) and interiors AND caps are isotropically
                   remeshed at targetlen (default: = seam_spacing).
                   If None, legacy path: seams stay at marching density
                   (remesh_periodic on the body only, raw caps).
    Returns (verts, faces): welded, outward normals, cartesian via A.
    """
    N = f.shape[0]
    nx, ny, nz = cells
    n_ax = (nx, ny, nz)

    def build(rho_x):
        v0, f0, t = periodic_unit_mesh(f, rho_x)
        caps = _unit_caps_global(f, t)
        if seam_spacing is not None:
            tl = seam_spacing if targetlen is None else targetlen
            pieces = [(v0, f0)] + [(v, fc) for _, _, v, fc in caps]
            keep = seam_decisions(pieces, N, seam_spacing)
            f0 = simplify_piece(v0, f0, keep, N, spacing=seam_spacing)
            v0, f0 = remesh_interior(v0, f0, tl, N)
            caps2 = []
            for ax, side, v, fc in caps:
                fc = simplify_piece(v, fc, keep, N, spacing=seam_spacing)
                v, fc = remesh_interior(v, fc, tl, N)
                caps2.append((ax, side, v, fc))
            caps = caps2
        elif remesh:
            v0, f0 = remesh_periodic(v0, f0, **remesh_kw)
        return v0, f0, caps

    v0, f0, caps = build(rho)
    if seam_spacing is not None and density_correct:
        # coarse seams inscribe the strut cross-sections at every cell
        # interface -> O(h^2) density deficit; measure it on one cell and
        # rebuild once with a proportionally corrected target
        vv, ff, nvv = [v0.copy()], [f0.copy()], len(v0)
        for _, _, v, fc in caps:
            vv.append(v); ff.append(fc + nvv); nvv += len(v)
        v1, f1 = fl.weld(np.concatenate(vv), np.concatenate(ff).astype(np.int64))
        rho_meas = fl.mesh_volume(v1, f1[:, ::-1]) / N ** 3
        if 0.5 * rho < rho_meas < 1.5 * rho:
            v0, f0, caps = build(rho * rho / rho_meas)

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
    verts, faces = fl.weld(verts, faces)
    return (verts / N) @ np.asarray(A, np.float32), faces[:, ::-1]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    N, rho, cells = 96, 0.30, (5, 5, 5)
    A, hkl, c = fl.octet_struts(node_comp=0.5)
    f = fl.field_on_grid(hkl, c, N)
    Vbox = np.linalg.det(A) * np.prod(cells)

    for label, kw in [("frozen-seam isotropic", dict(method='isotropic')),
                      ("seam-simplified h=5", dict(seam_spacing=5.0))]:
        t0 = time.time()
        v, fc = instanced_stack_mesh(A, f, rho=rho, cells=cells, **kw)
        print(f"{label:22s}: {time.time()-t0:5.1f}s  verts={len(v):8d}  "
              f"faces={len(fc):8d}  watertight={fl.is_watertight(fc)}  "
              f"V/V_box={fl.mesh_volume(v, fc)/Vbox:.4f}")
    fl.write_stl("octet_5x5x5_isotropic.stl", v, fc)