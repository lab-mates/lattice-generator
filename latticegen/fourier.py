"""
Fourier synthesis of periodic scalar fields.

A field with the periodicity of a Bravais lattice is a sum of plane waves
over reciprocal-lattice vectors G::

    f(r) = sum_G  c_G exp(i G . r),      c_{-G} = conj(c_G)   (real field)

The coefficients are built like a crystallographic structure factor from a
motif of Gaussian-smeared points and line segments (struts).  The solid is
the level set ``f(r) >= t``, with ``t`` fixed from the target volume
fraction (see :mod:`latticegen.field`).
"""

from __future__ import annotations

import numpy as np

from .cells import as_cell
from .topologies import Skeleton


# --------------------------------------------------------------------------- #
# reciprocal lattice
# --------------------------------------------------------------------------- #

def reciprocal_basis(A) -> np.ndarray:
    """Rows b1,b2,b3 with b_i . a_j = 2*pi*delta_ij."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(A, float)).T


def enumerate_G(A, gmax):
    """All reciprocal vectors with 0 < |G| <= gmax.

    Returns ``(hkl, G)``: integer indices (n,3) and cartesian vectors (n,3).
    """
    A = np.asarray(A, float)
    B = reciprocal_basis(A)
    hmax = np.ceil(np.linalg.norm(A, axis=1) * gmax / (2 * np.pi)).astype(int)
    ranges = [np.arange(-h, h + 1) for h in hmax]
    H = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    G = H @ B
    keep = np.linalg.norm(G, axis=1) <= gmax
    return H[keep], G[keep]


# --------------------------------------------------------------------------- #
# structure-factor coefficients
# --------------------------------------------------------------------------- #

def motif_coefficients(hkl, G, points=(), segments=(), sigma=0.1):
    """Coefficients of a Gaussian-smeared motif::

        c_G = [ sum_j w_j e^{-i G.r_j}
              + sum_s w_s e^{-i G.r_mid} sinc(G.d / 2) ] exp(-sigma^2 |G|^2 / 2)

    points   : iterable of (x_frac, weight)
    segments : iterable of (x1_frac, x2_frac, weight)   -- struts
    sigma    : Gaussian smearing radius (cartesian length units)
    """
    hkl = np.asarray(hkl, float)
    G = np.asarray(G, float)
    c = np.zeros(len(G), dtype=complex)
    for x, w in points:
        c += w * np.exp(-2j * np.pi * (hkl @ np.asarray(x, float)))
    for x1, x2, w in segments:
        x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
        xm, d = 0.5 * (x1 + x2), (x2 - x1)
        phase = np.exp(-2j * np.pi * (hkl @ xm))
        GdotD = 2.0 * np.pi * (hkl @ d)
        c += w * phase * np.sinc(GdotD / (2 * np.pi))     # np.sinc(x) = sin(pi x)/(pi x)
    G2 = np.einsum("ij,ij->i", G, G)
    return c * np.exp(-0.5 * sigma ** 2 * G2)


def field_on_grid(hkl, c, N: int) -> np.ndarray:
    """Evaluate the Fourier sum on an N^3 grid of fractional coordinates
    u = n/N via an inverse FFT.  Requires N > 2*max|h| (no aliasing)."""
    hkl = np.asarray(hkl, int)
    hmax = int(np.abs(hkl).max()) if len(hkl) else 0
    if N <= 2 * hmax:
        raise ValueError(f"N={N} too small for hmax={hmax}; need N > {2 * hmax} "
                         f"(reduce gmax or increase N)")
    C = np.zeros((N, N, N), dtype=complex)
    np.add.at(C, (hkl[:, 0] % N, hkl[:, 1] % N, hkl[:, 2] % N), c)
    return np.real(np.fft.ifftn(C)) * N ** 3


def evaluate(hkl, c, x_frac) -> np.ndarray:
    """Evaluate the Fourier sum at arbitrary fractional points (n,3)."""
    hkl = np.asarray(hkl, float)
    x = np.atleast_2d(np.asarray(x_frac, float))
    return np.real(np.exp(2j * np.pi * (x @ hkl.T)) @ c)


# --------------------------------------------------------------------------- #
# junction (node) compensation
# --------------------------------------------------------------------------- #

def compensate_nodes(hkl, G, c, nodes, ref_points, sigma, alpha=0.5,
                     sigma_node=None, n_iter=5):
    """Reduce the field pile-up where struts meet.

    Summing strut fields makes ``f`` much larger at junctions than at
    mid-strut, so the level set bulges there (octet: node/strut inscribed
    radius ratio ~2.4 uncompensated vs ~1.44 for a sharp cylinder union).
    Negative-weight Gaussian "anti-atoms" are added at the node positions,
    calibrated by fixed-point iteration so that
    ``f(node) -> alpha*f(mid-strut) + (1-alpha)*f(node)``.

    sigma_node : width of the anti-atoms; default 1.7*sigma.  Must be wider
                 than the strut sigma to carve the whole bulge.
    alpha      : strength.  0.5 reaches the sharp-union ratio while keeping
                 a fillet; ~0.6 gives a slightly waisted junction; beyond
                 that the junction may pinch off.
    Returns ``(c_new, node_weights)``.
    """
    hkl = np.asarray(hkl, float)
    nodes = np.atleast_2d(np.asarray(nodes, float))
    refs = np.atleast_2d(np.asarray(ref_points, float))
    if sigma_node is None:
        sigma_node = 1.7 * sigma
    ffn = np.exp(-0.5 * sigma_node ** 2 * np.einsum("ij,ij->i", G, G))
    K = ffn.sum()                                    # self-peak of a unit anti-atom
    phases = np.exp(-2j * np.pi * (nodes @ hkl.T))   # (n_nodes, nG)

    def fvals(X, cc):
        return np.real(np.exp(2j * np.pi * (X @ hkl.T)) @ cc)

    c0 = np.asarray(c, complex).copy()
    target = alpha * fvals(refs, c0).mean() + (1.0 - alpha) * fvals(nodes, c0)
    w = np.zeros(len(nodes))
    for _ in range(n_iter):
        cc = c0 + (w @ (phases * ffn))
        w += (target - fvals(nodes, cc)) / K
    return c0 + (w @ (phases * ffn)), w


# --------------------------------------------------------------------------- #
# high-level builder
# --------------------------------------------------------------------------- #

def fourier_coefficients(cell, skeleton: Skeleton, sigma=None, gmax=None,
                         node_comp=None, node_sigma=None, weights=None):
    """Fourier coefficients of a strut skeleton in a cell.

    sigma     : Gaussian smearing radius (cell length units).  Default: the
                topology's tuned fraction of the shortest cell vector.
    gmax      : reciprocal-space cutoff in units of 2*pi/a_min.  Default:
                topology-tuned (4.5 .. 7).
    node_comp : alpha of :func:`compensate_nodes` (None = off).
    weights   : optional per-strut weights (default 1).
    Returns ``(hkl, G, c, params)``.
    """
    cell = as_cell(cell)
    a = cell.a_min
    fd = skeleton.fourier_defaults
    sigma_frac = fd.get("sigma", 0.08)
    sigma = sigma_frac * a if sigma is None else float(sigma)
    gmax_units = fd.get("gmax", 5.5) if gmax is None else float(gmax)
    hkl, G = enumerate_G(cell.A, gmax_units * 2 * np.pi / a)
    n = skeleton.n_struts
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    segs = [(p, q, wi) for (p, q), wi in zip(skeleton.segments, w)]
    pts = [(p, 1.0) for p in skeleton.points]
    c = motif_coefficients(hkl, G, points=pts, segments=segs, sigma=sigma)
    if node_comp is not None and n:
        c, _ = compensate_nodes(hkl, G, c, nodes=skeleton.nodes(),
                                ref_points=skeleton.midpoints(), sigma=sigma,
                                alpha=node_comp, sigma_node=node_sigma)
    params = dict(sigma=sigma, gmax_units=gmax_units, n_G=len(hkl),
                  node_comp=node_comp, node_sigma=node_sigma)
    return hkl, G, c, params


def min_grid_for(hkl) -> int:
    """Smallest N that resolves the given reciprocal vectors without aliasing."""
    return 2 * int(np.abs(np.asarray(hkl)).max()) + 1
