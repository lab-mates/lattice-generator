"""
Explicit strut meshes: n-gon prisms joined by convex-hull nodes.

The field-based meshers spend triangles proportionally to surface area over
voxel size squared, and even after remeshing a straight strut keeps a ring
of vertices every triangle edge along its length.  A cylinder only needs
two rings, so for strut topologies the leanest mesh comes from the
:class:`Skeleton` directly:

* every strut is an n-gon **prism** (``sides`` polygon, 2 rings, 2n
  triangles, whatever its length);
* the rings are pulled back from the node centre by
  ``d = r / tan(theta_min / 2)`` (theta_min = smallest angle between struts
  meeting there), which is exactly where two cylinders stop intersecting;
* every node joins its rings.  The default ``node="union"`` makes the joint
  the size of the struts: each strut's prism continues to the node centre
  and is closed behind it by a strut-sized dome; the node is the union of
  these convex pieces (mesh boolean, ``manifold3d``), so nothing reaches
  farther than one strut radius from the node centre except the struts
  themselves (~1.9x the triangles of hull nodes for the octet).  ``node="hull"`` (no extra dependency) takes the convex hull of
  the rings instead, which bulges to ~1.4-2 strut radii.  Either way each
  ring is one facet of the node; that facet is removed and the prism
  attaches to the opening.  Vertices are shared by index, so the result is
  watertight without welding.

No scalar field, no marching cubes, no pymeshlab; memory is proportional to
the number of struts.  An octet cell with hexagonal struts is ~660
triangles with hull nodes (a marching-cubes cell at N = 80 is ~120 000).
Node shapes scale with r and are computed once per node type.

The unit-cell solid volume is a closed-form ``V(r) = alpha r^2 - beta r^3``
(prisms minus pull-back plus hull volumes), so the density -> radius
calibration is an exact one-dimensional root find, no meshing involved.

Nodes are convex: ``node="round"`` adds a sampled sphere to the hull input
for a rounded bulge, but concave fillets in the valley between two struts
cannot be produced by a hull (use ``sdf_field(..., smooth=k)`` and the
field meshers when those matter, i.e. for struts of many pixels).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

from .cells import as_cell
from .mesh import LatticeMesh
from .topologies import Skeleton, skeleton as _lookup_skeleton

_MARGIN = 1.02          # pull-back safety factor over the exact intersection distance
_SPHERE_CLEAR = 1.1     # x r: union nodes keep the rings beyond the node sphere
_MIN_PULLBACK = 0.05    # x r, for (nearly) collinear strut pairs
_KEY_DIGITS = 6         # rounding for node identity / signatures


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #

def polygon_area(r, sides) -> float:
    """Area of a regular polygon of circumradius r."""
    return 0.5 * sides * r * r * np.sin(2.0 * np.pi / sides)


def _frames(U):
    """Deterministic orthonormal frames (e1, e2) with e1 x e2 = u for each
    row of U (n, 3); the same strut direction always gets the same frame."""
    U = np.atleast_2d(U)
    a = np.where(np.abs(U[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(U, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(U, e1)
    return e1, e2


def _pullback(dirs, r):
    """Ring pull-back distance for a node with outward unit directions ``dirs``."""
    k = len(dirs)
    if k == 1:
        return 0.0
    c = dirs @ dirs.T
    np.fill_diagonal(c, -2.0)
    cos_min = float(np.clip(c.max(), -1.0, 1.0))         # closest pair
    theta = np.arccos(cos_min)
    d = _MARGIN * r / np.tan(0.5 * theta) if theta > 1e-9 else np.inf
    return float(max(d, _MIN_PULLBACK * r))


_HULL_UNION = None      # optional backend, e.g. Manifold's WebAssembly build under Pyodide


def set_hull_union_backend(fn):
    """Use ``fn(point_sets) -> (verts, tris, volume)`` (union of the convex
    hulls of each (k, 3) point set, outward-oriented) instead of manifold3d."""
    global _HULL_UNION
    _HULL_UNION = fn


def _hull_union(point_sets):
    if _HULL_UNION is not None:
        v, t, vol = _HULL_UNION(point_sets)
        return np.asarray(v, float).reshape(-1, 3), np.asarray(t, np.int64).reshape(-1, 3), float(vol)
    try:
        import manifold3d
    except ImportError as e:
        raise ImportError("node='union' needs manifold3d (pip install manifold3d); "
                          "node='hull' works without it") from e
    hulls = [manifold3d.Manifold.hull_points(np.asarray(p, np.float64)) for p in point_sets]
    u = manifold3d.Manifold.batch_boolean(hulls, manifold3d.OpType.Add)
    m = u.to_mesh64()
    return (np.asarray(m.vert_properties, np.float64)[:, :3], np.asarray(m.tri_verts, np.int64),
            float(u.volume()))


def _point_segment_distance(P, A, B):
    AB = B - A
    t = np.clip(np.einsum("ij,ij->i", P - A, AB) / np.einsum("ij,ij->i", AB, AB), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


def _segment_segment_distance(P0, P1, Q0, Q1, eps=1e-12):
    """Minimum distance between segments P0P1 and Q0Q1 (vectorised, Ericson)."""
    d1, d2, r = P1 - P0, Q1 - Q0, P0 - Q0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm((P0 + s[:, None] * d1) - (Q0 + t[:, None] * d2), axis=1)


def _fibonacci_sphere(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)
    th = np.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _sphere_points(sides):
    """Points of the node sphere for strut polygons of ``sides`` sides."""
    return _fibonacci_sphere(max(24, (2 * sides * sides) // 3))


def _node_union(pts, ring_of, sides, r):
    """Strut-sized node: the union of one convex piece per strut, each the
    prism of the strut continued from its ring to the node centre and closed
    behind the centre by a dome of the polygon's inradius.  Together they
    contain the sphere of that radius; nothing reaches farther than r from the
    node centre except the struts themselves.  (Prism stubs intersect along
    few edges: about half the triangles of hull(ring, sphere) pieces.)

    Same contract as :func:`_node_hull`: (triangles as local point indices,
    closed volume, extra points); indices >= len(pts) are extra points.
    """
    from scipy.spatial import cKDTree
    sph = r * np.cos(np.pi / sides) * _sphere_points(sides)
    k = int(ring_of.max()) + 1
    pieces = []
    for j in range(k):
        ring = pts[ring_of == j]
        centre = ring.mean(axis=0)
        u = centre / np.linalg.norm(centre)
        pieces.append(np.vstack([ring, ring - centre, sph[sph @ u < -1e-9 * r]]))
    V, T, vol = _hull_union(pieces)
    # output vertex -> ring point (exact up to rounding) or new extra point
    dist, idx = cKDTree(pts).query(V)
    is_ring = dist < 1e-5 * r                      # float32 output from some backends
    if np.unique(idx[is_ring]).size != len(pts) or is_ring.sum() != len(pts):
        raise RuntimeError("union node lost or duplicated a ring vertex")
    local = np.empty(len(V), np.int64)
    local[is_ring] = idx[is_ring]
    local[~is_ring] = len(pts) + np.arange((~is_ring).sum())
    tri = local[T]
    rid = np.full(len(pts) + (~is_ring).sum(), -1)
    rid[:len(pts)] = ring_of
    tr = rid[tri]
    opening = (tr[:, 0] >= 0) & (tr[:, 0] == tr[:, 1]) & (tr[:, 1] == tr[:, 2])
    if np.any(np.bincount(tr[opening][:, 0], minlength=k) != sides - 2):
        raise RuntimeError("a strut ring is not a single facet of its union node")
    return tri[~opening], vol, V[~is_ring]


def _node_hull(pts, ring_of, sides, dirs, d, node, round_points):
    """Convex-hull node from the local ring points (k*sides, 3).

    Returns (triangles as local point indices, hull volume, extra points).
    Local indices >= len(pts) refer to the extra (sphere) points.  Ring
    facets are removed (openings for the prisms); the rest is oriented
    outward.
    """
    extra = np.zeros((0, 3))
    if node == "round":
        R = np.linalg.norm(pts, axis=1)
        if R.max() - R.min() > 1e-6 * R.max():
            raise ValueError("node='round' needs equal strut radii at every node")
        sph = R.max() * _fibonacci_sphere(int(round_points))
        keep = np.all(sph @ dirs.T <= d * (1.0 - 1e-6), axis=1)
        extra = sph[keep]
    P = np.vstack([pts, extra])
    hull = ConvexHull(P)
    if len(hull.vertices) != len(P):
        raise RuntimeError("node hull dropped a vertex (rings too close or too "
                           "different); this is a bug in the pull-back distance")
    tri = hull.simplices
    rid = np.full(len(P), -1)
    rid[:len(pts)] = ring_of
    tr = rid[tri]
    opening = (tr[:, 0] >= 0) & (tr[:, 0] == tr[:, 1]) & (tr[:, 1] == tr[:, 2])
    per_ring = np.bincount(tr[opening][:, 0], minlength=len(dirs))
    if np.any(per_ring != sides - 2):
        raise RuntimeError("a strut ring is not a single hull facet at a node; "
                           "increase the pull-back margin")
    tri = tri[~opening]
    nrm = hull.equations[~opening, :3]
    v = P[tri]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    flip = np.einsum("ij,ij->i", n, nrm) < 0
    tri[flip] = tri[flip][:, ::-1]
    return tri, float(hull.volume), extra


# --------------------------------------------------------------------------- #
# skeleton -> block of nodes and struts
# --------------------------------------------------------------------------- #

def _skel(topology) -> Skeleton:
    return topology if isinstance(topology, Skeleton) else _lookup_skeleton(str(topology))


def _block_struts(sk: Skeleton, n_ax, tol=1e-6):
    """All lattice translates of the skeleton struts whose two endpoints lie
    in the closed block [0, n]^3 (fractional).  Returns (P, Q) (m, 3)."""
    segs = np.array([(p, q) for p, q in sk.segments], float)          # (m, 2, 3)
    offs = np.array(list(itertools.product(*(range(-1, int(n) + 1) for n in n_ax))), float)
    P = (segs[None, :, 0, :] + offs[:, None, :]).reshape(-1, 3)
    Q = (segs[None, :, 1, :] + offs[:, None, :]).reshape(-1, 3)
    hi = np.asarray(n_ax, float) + tol
    ok = np.all((P >= -tol) & (P <= hi) & (Q >= -tol) & (Q <= hi), axis=1)
    return P[ok], Q[ok]


def _periodic_struts(sk: Skeleton):
    """One period: the skeleton struts plus, for each node in [0,1)^3, every
    translate that touches it -- so that node geometry is computed with the
    full periodic connectivity.  Returns (P, Q, is_own) where ``is_own``
    marks the skeleton's own struts (counted once for the volume)."""
    segs = np.array([(p, q) for p, q in sk.segments], float)
    offs = np.array(list(itertools.product(range(-2, 3), repeat=3)), float)
    P = (segs[None, :, 0, :] + offs[:, None, :]).reshape(-1, 3)
    Q = (segs[None, :, 1, :] + offs[:, None, :]).reshape(-1, 3)
    own = np.all(offs == 0, axis=1)[:, None].repeat(len(segs), 1).reshape(-1)
    return P, Q, own


def _nodes_of(P, Q):
    """Node ids for the endpoints (rounded fractional coordinates)."""
    key = np.round(np.vstack([P, Q]) * 10 ** _KEY_DIGITS).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    m = len(P)
    return uniq / 10.0 ** _KEY_DIGITS, inv[:m], inv[m:]


class _Geometry:
    """Struts and nodes of a block in cartesian coordinates, with per-node
    pull-back distances and cached hull triangulations."""

    def __init__(self, cell, P, Q, sides, node, round_points, own=None):
        self.cell = cell
        self.sides = int(sides)
        self.node_kind = node
        self.round_points = round_points
        nodes_f, a, b = _nodes_of(P, Q)
        self.nodes = nodes_f @ cell.A                                 # (n_nodes, 3)
        self.a, self.b = a, b
        self.n_struts = len(P)
        self.own = np.ones(self.n_struts, bool) if own is None else own
        A_, B_ = self.nodes[a], self.nodes[b]
        self.L = np.linalg.norm(B_ - A_, axis=1)
        if np.any(self.L < 1e-12):
            raise ValueError("zero-length strut")
        self.u = (B_ - A_) / self.L[:, None]
        # rings: ring index = s (end at node a) or n_struts + s (end at node b);
        # ring_dir points from the node INTO its strut
        self.ring_node = np.concatenate([a, b])
        self.ring_dir = np.vstack([self.u, -self.u])
        self.ring_end = np.concatenate([np.zeros(self.n_struts, int), np.ones(self.n_struts, int)])
        order = np.argsort(self.ring_node, kind="stable")
        starts = np.searchsorted(self.ring_node[order], np.arange(len(self.nodes) + 1))
        self.incident = [order[starts[i]:starts[i + 1]] for i in range(len(self.nodes))]

    # -- radius-dependent quantities ------------------------------------------ #
    def pullbacks(self, r):
        """Pull-back distance of every ring (n_rings,) for strut radius r."""
        d = np.zeros(2 * self.n_struts)
        cache = {}
        for inc in self.incident:
            dirs = self.ring_dir[inc]
            key = tuple(map(tuple, np.round(dirs, _KEY_DIGITS)))
            if key not in cache:
                cache[key] = _pullback(dirs, r)
                if self.node_kind == "union" and len(dirs) > 1:
                    cache[key] = max(cache[key], _SPHERE_CLEAR * r)
            d[inc] = cache[key]
        return d

    def max_radius(self):
        """Radius at which struts vanish into their nodes (d_a + d_b = L)."""
        d1 = self.pullbacks(1.0)
        return float((self.L / (d1[:self.n_struts] + d1[self.n_struts:] + 1e-12)).min())

    def overlap_radius(self, samples=12, k=48):
        """Largest radius before primitives that do not share a node touch:
        prisms are bounded by cylinders of radius r, hulls by spheres of
        radius R(r) = r sqrt(d1^2 + 1) (union nodes by r); the closest node pair, node /
        non-incident strut pair and non-adjacent strut pair set the limit."""
        from scipy.spatial import cKDTree
        d1 = self.pullbacks(1.0)
        # union nodes stay within r of the skeleton; hulls reach sqrt(d^2 + r^2)
        R1 = 1.0 if self.node_kind == "union" else float(np.sqrt(d1.max() ** 2 + 1.0))
        A_, B_ = self.nodes[self.a], self.nodes[self.b]
        limits = []
        if len(self.nodes) > 1:
            dd, _ = cKDTree(self.nodes).query(self.nodes, k=2)
            limits.append(dd[:, 1].min() / (2.0 * R1))
        # sample points along the struts for candidate search
        ts = np.linspace(0.0, 1.0, samples)
        S = (A_[:, None, :] + ts[None, :, None] * (B_ - A_)[:, None, :]).reshape(-1, 3)
        seg_of = np.repeat(np.arange(self.n_struts), samples)
        tree = cKDTree(S)
        kk = min(k, len(S))
        # node vs non-incident strut
        _, j = tree.query(self.nodes, k=kk)
        cand_seg = seg_of[j]                                          # (n_nodes, kk)
        ni = np.repeat(np.arange(len(self.nodes)), kk)
        cs = cand_seg.reshape(-1)
        ok = (self.a[cs] != ni) & (self.b[cs] != ni)
        if ok.any():
            dist = _point_segment_distance(self.nodes[ni[ok]], A_[cs[ok]], B_[cs[ok]])
            limits.append(dist.min() / (R1 + 1.0))
        # strut vs non-adjacent strut
        _, j = tree.query(S, k=kk)
        si = np.repeat(seg_of, kk)
        sj = seg_of[j].reshape(-1)
        ok = (si != sj) & (self.a[si] != self.a[sj]) & (self.a[si] != self.b[sj]) \
             & (self.b[si] != self.a[sj]) & (self.b[si] != self.b[sj])
        if ok.any():
            pairs = np.unique(np.sort(np.stack([si[ok], sj[ok]], 1), axis=1), axis=0)
            dist = _segment_segment_distance(A_[pairs[:, 0]], B_[pairs[:, 0]],
                                             A_[pairs[:, 1]], B_[pairs[:, 1]])
            limits.append(dist.min() / 2.0)
        return float(min(limits)) if limits else np.inf

    def strut_graph_components(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n = len(self.nodes)
        m = coo_matrix((np.ones(self.n_struts), (self.a, self.b)), shape=(n, n))
        return int(connected_components(m, directed=False)[0])

    def node_hulls(self, r, d):
        """{node: (triangles as ring-local indices, ring order, volume, extra
        local points)} for nodes with >= 2 struts, cached by signature."""
        out = {}
        cache = {}
        for i, inc in enumerate(self.incident):
            if len(inc) < 2:
                continue
            dirs = self.ring_dir[inc]
            key = [tuple(np.round(dirs[j], _KEY_DIGITS)) + (int(self.ring_end[inc[j]]),) for j in range(len(inc))]
            srt = sorted(range(len(inc)), key=lambda j: key[j])
            inc = inc[srt]
            sig = tuple(key[j] for j in srt) + (round(float(d[inc[0]]), 9),)
            if sig not in cache:
                pts, ring_of = self._local_rings(inc, r, d)
                if self.node_kind == "union":
                    cache[sig] = _node_union(pts, ring_of, self.sides, r)
                else:
                    cache[sig] = _node_hull(pts, ring_of, self.sides, self.ring_dir[inc],
                                            d[inc[0]], self.node_kind, self.round_points)
            tri, vol, extra = cache[sig]
            out[i] = (tri, inc, vol, extra)
        return out

    def ring_points(self, rings, r, d):
        """Ring vertices (len(rings)*sides, 3) in cartesian coordinates."""
        rings = np.asarray(rings)
        u = self.u[rings % self.n_struts]
        centre = self.nodes[self.ring_node[rings]] + d[rings][:, None] * self.ring_dir[rings]
        e1, e2 = _frames(u)
        th = np.arange(self.sides) * 2.0 * np.pi / self.sides
        out = (centre[:, None, :] + r * (np.cos(th)[None, :, None] * e1[:, None, :]
                                         + np.sin(th)[None, :, None] * e2[:, None, :]))
        return out.reshape(-1, 3)

    def _local_rings(self, inc, r, d):
        pts = self.ring_points(inc, r, d) - self.nodes[self.ring_node[inc[0]]]
        ring_of = np.repeat(np.arange(len(inc)), self.sides)
        return pts, ring_of

    # -- volume ------------------------------------------------------------------ #
    def volume(self, r):
        """Solid volume (own struts' prisms + every node hull)."""
        d = self.pullbacks(r)
        area = polygon_area(r, self.sides)
        length = self.L - d[:self.n_struts] - d[self.n_struts:]
        if np.any(length <= 0):
            raise ValueError("struts too short for this radius")
        v = float((area * length[self.own]).sum())
        for _, (_, _, hv, _) in self.node_hulls(r, d).items():
            v += hv
        return v


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def bulk_volume_coefficients(cell, topology, sides=6, node="union", round_points=64):
    """(alpha, beta, r_max) with unit-cell solid volume V(r) = alpha r^2 - beta r^3
    for the periodic lattice (every node counted once per cell).  The formula
    (and the mesh) is valid up to ``r_max``, where primitives that do not
    share a node start to touch, or struts vanish into their nodes."""
    cell = as_cell(cell)
    sk = _skel(topology)
    P, Q, own = _periodic_struts(sk)
    g = _Geometry(cell, P, Q, sides, node, round_points, own=own)
    # keep only the nodes that lie in [0,1)^3 for the hull sum
    frac = g.nodes @ np.linalg.inv(cell.A)
    in_cell = np.all((frac >= -1e-6) & (frac < 1.0 - 1e-6), axis=1)
    r_max = min(g.max_radius(), g.overlap_radius())
    r1 = 0.1 * r_max
    d = g.pullbacks(r1)
    area = polygon_area(r1, sides)
    prism = float((area * g.L[own]).sum())                 # alpha r^2 part
    pull = float((area * (d[:g.n_struts] + d[g.n_struts:])[own]).sum())
    hulls = sum(hv for i, (_, _, hv, _) in g.node_hulls(r1, d).items() if in_cell[i])
    alpha = prism / r1 ** 2
    beta = (pull - hulls) / r1 ** 3
    return alpha, beta, r_max


def radius_for_density(cell, topology, rho, sides=6, node="union", round_points=64) -> float:
    """Strut circumradius giving bulk relative density ``rho`` (cell units)."""
    cell = as_cell(cell)
    alpha, beta, r_max = bulk_volume_coefficients(cell, topology, sides, node, round_points)
    V = cell.volume
    f = lambda r: (alpha * r * r - beta * r ** 3) / V - rho
    r_hi = 0.999 * r_max
    if f(r_hi) < 0:
        raise ValueError(f"rho={rho} is not reachable: above a bulk density of {f(r_hi) + rho:.3f} "
                         f"(r = {r_hi:.4g}) struts or nodes that share no node would overlap")
    return float(brentq(f, 1e-9 * r_max, r_hi, xtol=1e-12 * r_max))


def mesh_struts(cell, topology, rho=None, radius=None, cells=(1, 1, 1), sides=6,
                node="union", round_points=64, verbose=False) -> LatticeMesh:
    """Explicit prism-and-node mesh of an (nx, ny, nz) block of a strut topology.

    cell     : UnitCell, 3x3 matrix or cubic edge length (mm).
    topology : built-in name ('octet', 'bcc', ...) or a Skeleton.
    rho      : bulk relative density of the periodic lattice; converted to
               ``radius`` analytically.  Give ``radius`` (mm, polygon
               circumradius) instead to set the strut size directly.
    cells    : block size in cells.  Every strut whose two nodes lie in the
               closed block is kept whole, so the outer faces carry complete
               nodes and the block is ~2 r larger than nx*a (its actual
               relative density, ``mesh.relative_density``, is a little
               above ``rho`` because of the extra boundary struts).
    sides    : polygon of the strut section (3..64).  Minimum thickness is
               2 r cos(pi/sides); with 1-2 pixel struts 4-6 sides suffice.
    node     : 'union' (default: joints the size of the struts -- the union
               of the struts and a sphere of the strut radius; needs
               manifold3d), 'hull' (convex hull of the strut rings: a bulge
               of ~1.4-2 r, no extra dependency) or 'round' (hull of the
               rings plus a sampled sphere: rounded convex bulge).
    round_points : sphere samples for node='round'.

    Returns a watertight :class:`LatticeMesh` (outward normals);
    ``mesh.info`` holds strut_radius, node_radius, min_thickness, sides.
    """
    cell = as_cell(cell)
    sk = _skel(topology)
    sides = int(sides)
    if not 3 <= sides <= 64:
        raise ValueError("sides must be between 3 and 64")
    if node not in ("union", "hull", "round"):
        raise ValueError("node must be 'union', 'hull' or 'round'")
    if sk.points:
        warnings.warn("isolated skeleton points (spheres) are ignored by mesh_struts",
                      RuntimeWarning, stacklevel=2)
    n_ax = tuple(int(c) for c in cells)
    if min(n_ax) < 1:
        raise ValueError("cells must be >= 1 along every axis")
    if (rho is None) == (radius is None):
        raise ValueError("give exactly one of rho or radius")
    if rho is not None:
        if not 0.0 < float(rho) < 1.0:
            raise ValueError("rho must lie strictly inside (0, 1)")
        radius = radius_for_density(cell, sk, float(rho), sides, node, round_points)
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be positive")

    _, _, r_max = bulk_volume_coefficients(cell, sk, sides, node, round_points)
    if r >= r_max:
        raise ValueError(f"radius {r:.4g} too large: above {r_max:.4g} (cell units) struts vanish "
                         f"into their nodes or primitives that share no node overlap")
    P, Q = _block_struts(sk, n_ax)
    if len(P) == 0:
        raise RuntimeError("no strut fits inside the block")
    g = _Geometry(cell, P, Q, sides, node, round_points)
    d = g.pullbacks(r)
    if verbose:
        print(f"  {g.n_struts} struts, {len(g.nodes)} nodes, r = {r:.5g}, "
              f"pull-back {d.min():.4g}..{d.max():.4g}")

    # vertices: ring k occupies ids k*sides .. (k+1)*sides-1
    rings = np.arange(2 * g.n_struts)
    verts = g.ring_points(rings, r, d)

    faces = []
    # prisms
    s = np.arange(g.n_struts)
    i = np.arange(sides)
    j = (i + 1) % sides
    r0 = (s[:, None] * sides)                                    # ring at node a
    r1 = ((g.n_struts + s)[:, None] * sides)                     # ring at node b
    faces.append(np.stack([r0 + i, r0 + j, r1 + j], axis=-1).reshape(-1, 3))
    faces.append(np.stack([r0 + i, r1 + j, r1 + i], axis=-1).reshape(-1, 3))
    # node hulls
    hulls = g.node_hulls(r, d)
    extra_verts = [verts]
    n_extra = len(verts)
    n_cap = 0
    for ni, inc in enumerate(g.incident):
        if len(inc) == 0:
            continue
        if len(inc) == 1:                                           # dangling strut: flat cap
            k = int(inc[0])
            base = k * sides
            fan = np.stack([np.full(sides - 2, base), base + i[1:-1], base + i[2:]], axis=-1)
            if g.ring_end[k] == 0:
                fan = fan[:, ::-1]
            faces.append(fan)
            n_cap += 1
            continue
        tri, inc_sorted, _, extra = hulls[ni]
        local = (inc_sorted[:, None] * sides + i[None, :]).reshape(-1)     # local -> global ids
        if len(extra):                                                  # rounded node: own vertices
            local = np.concatenate([local, n_extra + np.arange(len(extra))])
            extra_verts.append(extra + g.nodes[ni])
            n_extra += len(extra)
        faces.append(local[tri])
    faces = np.vstack(faces).astype(np.int64)
    verts = np.vstack(extra_verts)

    # volume bookkeeping
    area = polygon_area(r, sides)
    length = g.L - d[:g.n_struts] - d[g.n_struts:]
    vol = float((area * length).sum()) + sum(h[2] for h in hulls.values())
    R = r if node == "union" else float(np.sqrt(d.max() ** 2 + r * r))
    info = dict(mode="struts", cells=n_ax, A=cell.A.copy(), box_volume=cell.volume * int(np.prod(n_ax)),
                strut_radius=r, node_radius=R, min_thickness=2.0 * r * np.cos(np.pi / sides),
                sides=sides, node=node, n_struts=int(g.n_struts), n_nodes=int(len(g.nodes)),
                bulk_density=(float(rho) if rho is not None else None), analytic_volume=vol,
                topology=sk.name, n_caps=n_cap)
    mesh = LatticeMesh(verts.astype(np.float32), faces, info=info)
    if verbose:
        print(f"  {mesh.n_faces:,} faces, volume {vol:.6g}")
    n_comp = g.strut_graph_components()          # on the strut graph: cheap for millions of faces
    if n_comp > 1:
        warnings.warn(f"the strut network has {n_comp} disconnected components (floating pieces); "
                      f"see mesh.remove_islands()", RuntimeWarning, stacklevel=2)
    return mesh
