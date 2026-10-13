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
    """Reciprocal basis of a lattice.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Lattice vectors as rows.

    Returns
    -------
    ndarray, shape (3, 3)
        Rows b1, b2, b3 with ``b_i . a_j = 2 pi delta_ij``.
    """
    return 2.0 * np.pi * np.linalg.inv(np.asarray(A, float)).T


def enumerate_G(A, gmax):
    """All reciprocal-lattice vectors with ``0 < |G| <= gmax``.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Lattice vectors as rows.
    gmax : float
        Cutoff, in inverse cartesian length units.

    Returns
    -------
    hkl : ndarray of int, shape (n, 3)
        Miller indices.
    G : ndarray, shape (n, 3)
        Cartesian wave vectors.
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
    """Fourier coefficients of a Gaussian-smeared motif of points and struts.

    ``c_G = [sum_j w_j exp(-i G.r_j) + sum_s w_s exp(-i G.r_mid) sinc(G.d / 2)] exp(-sigma^2 |G|^2 / 2)``

    Parameters
    ----------
    hkl : array_like, shape (n, 3)
    G : array_like, shape (n, 3)
        From :func:`enumerate_G`.
    points : iterable of (x_frac, weight), optional
        Point "atoms" (spherical nodes).
    segments : iterable of (x1_frac, x2_frac, weight), optional
        Struts, as uniform lines of atoms.
    sigma : float, optional
        Gaussian smearing radius (cartesian length units).

    Returns
    -------
    ndarray of complex, shape (n,)
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
    """Evaluate a Fourier sum on the N^3 fractional grid with one inverse FFT.

    Parameters
    ----------
    hkl : array_like of int, shape (n, 3)
    c : array_like of complex, shape (n,)
    N : int
        Samples per cell edge.

    Returns
    -------
    ndarray, shape (N, N, N)

    Raises
    ------
    ValueError
        If ``N <= 2 max|h|``, which would alias high frequencies.
    """
    hkl = np.asarray(hkl, int)
    hmax = int(np.abs(hkl).max()) if len(hkl) else 0
    if N <= 2 * hmax:
        raise ValueError(f"N={N} too small for hmax={hmax}; need N > {2 * hmax} "
                         f"(reduce gmax or increase N)")
    C = np.zeros((N, N, N), dtype=complex)
    np.add.at(C, (hkl[:, 0] % N, hkl[:, 1] % N, hkl[:, 2] % N), c)
    return np.real(np.fft.ifftn(C)) * N ** 3


def evaluate(hkl, c, x_frac) -> np.ndarray:
    """Evaluate a Fourier sum at arbitrary fractional points.

    Parameters
    ----------
    hkl : array_like, shape (n, 3)
    c : array_like of complex, shape (n,)
    x_frac : array_like, shape (m, 3)

    Returns
    -------
    ndarray, shape (m,)
    """
    hkl = np.asarray(hkl, float)
    x = np.atleast_2d(np.asarray(x_frac, float))
    return np.real(np.exp(2j * np.pi * (x @ hkl.T)) @ c)


# --------------------------------------------------------------------------- #
# junction (node) compensation
# --------------------------------------------------------------------------- #

def compensate_nodes(hkl, G, c, nodes, ref_points, sigma, alpha=0.5,
                     sigma_node=None, n_iter=5):
    """Reduce the field pile-up where struts meet ("junction slimming").

    Strut fields add up, so ``f`` is much larger at junctions than at
    mid-strut and the level set bulges there (octet: node to strut inscribed
    radius 2.4 uncompensated, 1.44 for a sharp cylinder union).
    Negative-weight Gaussians are added at the nodes, with weights found by
    fixed-point iteration so that
    ``f(node) -> alpha f(mid-strut) + (1 - alpha) f(node)``.

    Parameters
    ----------
    hkl, G : array_like, shape (n, 3)
    c : array_like of complex, shape (n,)
        Coefficients to correct.
    nodes : array_like, shape (k, 3)
        Fractional node positions.
    ref_points : array_like, shape (m, 3)
        Fractional mid-strut positions.
    sigma : float
        Strut smearing radius.
    alpha : float, optional
        Strength. 0.5 reaches the sharp-union ratio while keeping a fillet;
        about 0.6 waists the junction; beyond that it may pinch off.
    sigma_node : float, optional
        Width of the compensating Gaussians; default ``1.7 sigma``. It must be
        wider than ``sigma`` to carve the whole bulge.
    n_iter : int, optional
        Fixed-point iterations.

    Returns
    -------
    c_new : ndarray of complex
    node_weights : ndarray
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

    Parameters
    ----------
    cell : UnitCell, float or array_like
    skeleton : Skeleton
    sigma : float, optional
        Gaussian smearing radius (cell length units). Default: the
        topology's tuned fraction of the shortest cell edge.
    gmax : float, optional
        Cutoff in units of ``2 pi / a_min``. Default: topology-tuned (4.5 to 7).
    node_comp : float, optional
        ``alpha`` of :func:`compensate_nodes`; None is off.
    node_sigma : float, optional
        ``sigma_node`` of :func:`compensate_nodes`.
    weights : array_like, optional
        Per-strut weights (default 1).

    Returns
    -------
    hkl : ndarray of int, shape (n, 3)
    G : ndarray, shape (n, 3)
    c : ndarray of complex, shape (n,)
    params : dict
        The parameters actually used.
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
    """Smallest grid size that resolves the given Miller indices without aliasing.

    Parameters
    ----------
    hkl : array_like of int, shape (n, 3)

    Returns
    -------
    int
        ``2 max|h| + 1``.
    """
    return 2 * int(np.abs(np.asarray(hkl)).max()) + 1
