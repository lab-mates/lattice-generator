"""
Signed-distance (SDF) fields of strut skeletons.

The unit-cell field is the exact shortest distance ``d(r)`` from each voxel
to the strut skeleton (point-to-segment distance minimised over all
periodic images).  The level set ``d <= r`` gives struts of EXACTLY circular
cross-section away from junctions, and junctions that are plain unions of
cylinders -- or smoothly filleted ones (``smooth > 0``, exponential
smooth-min with fillet scale ~smooth).

The field is handed downstream as ``f = -d`` so that "solid = f >= t" with
``t = -r``.  Unlike the Fourier sum, the node pile-up is BOUNDED: where n
struts meet the effective distance drops by at most ``smooth*ln(n)``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .cells import as_cell
from .topologies import Skeleton


def periodic_segment_images(segments, margin=0.9):
    """Periodic images of segments near the unit cell.

    Expands the segments over the 27 neighbouring cells and keeps those
    whose bounding box comes within ``margin`` of the cell.

    Parameters
    ----------
    segments : iterable of (p, q)
        Fractional endpoints.
    margin : float, optional
        Fractional distance; must exceed the largest voxel-to-skeleton
        distance (below ``sqrt(3) / 2`` for any connected cubic lattice).

    Returns
    -------
    list of (P, Q)
    """
    out = []
    for p, q in segments:
        p, q = np.asarray(p, float), np.asarray(q, float)
        for off in itertools.product((-1, 0, 1), repeat=3):
            P, Q = p + off, q + off
            lo, hi = np.minimum(P, Q), np.maximum(P, Q)
            if np.all(hi >= -margin) and np.all(lo <= 1.0 + margin):
                out.append((P, Q))
    return out


def _segment_arrays(segments, A):
    """Cartesian origin P, axis D, |D|^2 and bounding boxes of segment images."""
    P = np.array([np.asarray(p) @ A for p, _ in segments])
    Q = np.array([np.asarray(q) @ A for _, q in segments])
    D = Q - P
    L2 = np.einsum("ij,ij->i", D, D)
    lo, hi = np.minimum(P, Q), np.maximum(P, Q)
    return P, D, L2, lo, hi


def _min_dist_block(R, P, D, L2, smooth, max_elems=12_000_000):
    """(Smooth-)min distance from points R (n,3) to segments (P, D).

    Distances to all segments are gathered in a (k, n) block (processed in
    segment batches to bound memory) and reduced with a hard min or a
    log-sum-exp smooth-min:  m_s = m - ln( sum_i exp(-(d_i - m)/k) ) * k.
    """
    n = len(R)
    k = len(P)
    if k == 0:
        return np.full(n, np.inf)
    batch = max(1, min(k, max_elems // max(n, 1)))
    m = np.full(n, np.inf)
    s = np.zeros(n) if smooth > 0 else None
    beta = 1.0 / smooth if smooth > 0 else 0.0
    for b0 in range(0, k, batch):
        b1 = min(b0 + batch, k)
        dist = np.empty((b1 - b0, n))
        for row, (Pi, Di, l2) in enumerate(zip(P[b0:b1], D[b0:b1], L2[b0:b1])):
            RP = R - Pi
            if l2 > 0:
                t = np.clip((RP @ Di) / l2, 0.0, 1.0)
                RP -= t[:, None] * Di
            dist[row] = np.sqrt(np.einsum("ij,ij->i", RP, RP))
        m_b = dist.min(axis=0)
        if smooth == 0:
            np.minimum(m, m_b, out=m)
        else:
            m_new = np.minimum(m, m_b)
            s = s * np.exp(-beta * np.minimum(m - m_new, 700 / beta))
            s += np.exp(-beta * (dist - m_new)).sum(axis=0)
            m = m_new
    if smooth > 0:
        m = m - np.log(np.maximum(s, 1.0)) / beta
    return m


def distance_field(segments, N: int, A=None, smooth: float = 0.0,
                   block: int = 24, cull: bool = True) -> np.ndarray:
    """Exact shortest distance from every voxel of the cell to a strut skeleton.

    Minimised over periodic images. The grid is processed in cubic blocks,
    and segment images that cannot be nearest for a block are skipped (the
    distance is 1-Lipschitz, so a coarse exact pass bounds it).

    Parameters
    ----------
    segments : iterable of (p, q)
        Fractional endpoint pairs, e.g. ``Skeleton.segments``. A segment
        with ``p == q`` is a point (sphere).
    N : int
        Samples per cell edge.
    A : array_like, shape (3, 3), optional
        Lattice vectors as rows; identity by default.
    smooth : float, optional
        0 for the hard minimum. ``k > 0`` uses the exponential smooth
        minimum ``-k log(sum exp(-d_i / k))``: fillets of size about ``k``
        (same length unit as ``A``).
    block : int, optional
        Voxels per side of the blocks processed at a time.
    cull : bool, optional
        Skip segment images that cannot influence a block (exact to about
        ``1e-6 smooth``; a large speed-up for big skeletons).

    Returns
    -------
    ndarray, shape (N, N, N)
        Distances, in the length unit of ``A``.
    """
    A = np.eye(3) if A is None else np.asarray(A, float)
    imgs = periodic_segment_images(segments)
    P, D, L2, lo, hi = _segment_arrays(imgs, A)
    u = np.arange(N) / N
    d = np.empty((N, N, N))
    block = max(4, min(N, int(block)))

    # upper bound on d over the cell from a coarse exact pass (d is 1-Lipschitz):
    #   d_true <= d_coarse_max + half coarse voxel diagonal.  For the smooth
    #   min, segments farther than m + 14*smooth contribute < e^-14 to the sum.
    d_bound = np.inf
    if cull:
        Nc = max(4, N // 8)
        uc = (np.arange(Nc) + 0.5) / Nc
        Xc, Yc, Zc = np.meshgrid(uc, uc, uc, indexing="ij")
        Rc = np.stack([Xc, Yc, Zc], -1).reshape(-1, 3) @ A
        dc = _min_dist_block(Rc, P, D, L2, 0.0)
        half_diag = 0.5 * np.linalg.norm(A.sum(axis=0)) / Nc
        d_bound = dc.max() + half_diag + (14.0 * smooth if smooth > 0 else 0.0)

    for x0 in range(0, N, block):
        for y0 in range(0, N, block):
            for z0 in range(0, N, block):
                sx, sy, sz = (slice(x0, min(x0 + block, N)), slice(y0, min(y0 + block, N)),
                              slice(z0, min(z0 + block, N)))
                X, Y, Z = np.meshgrid(u[sx], u[sy], u[sz], indexing="ij")
                R = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) @ A
                if cull:
                    rlo, rhi = R.min(axis=0), R.max(axis=0)
                    gap = np.maximum(0.0, np.maximum(lo - rhi, rlo - hi))
                    keep = np.linalg.norm(gap, axis=1) <= d_bound
                    m = _min_dist_block(R, P[keep], D[keep], L2[keep], smooth)
                else:
                    m = _min_dist_block(R, P, D, L2, smooth)
                d[sx, sy, sz] = m.reshape(X.shape)
    return d


def sdf_on_grid(cell, skeleton: Skeleton, N: int, smooth: float = 0.0, **kw) -> np.ndarray:
    """Minus the distance to a skeleton, ready for meshing (solid is ``f >= -r``).

    Parameters
    ----------
    cell : UnitCell, float or array_like
    skeleton : Skeleton
    N : int
    smooth : float, optional
    **kw
        Passed to :func:`distance_field`.

    Returns
    -------
    ndarray, shape (N, N, N)
    """
    cell = as_cell(cell)
    segs = list(skeleton.segments) + [(p, p) for p in skeleton.points]
    return -distance_field(segs, N, cell.A, smooth=smooth, **kw)


def radius_for_density(d, rho) -> float:
    """Radius ``r`` whose solid ``{d <= r}`` has volume fraction ``rho``.

    Parameters
    ----------
    d : ndarray
        Distance samples from :func:`distance_field`.
    rho : float

    Returns
    -------
    float
    """
    return float(np.quantile(d, rho))
