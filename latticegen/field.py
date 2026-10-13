"""
Unit-cell scalar fields and the density <-> threshold relation.

A :class:`UnitField` holds a periodic scalar field ``f`` sampled on an
N x N x N grid of fractional coordinates (u = n/N) of one unit cell, plus
the cell geometry.  The solid is ``f >= t``; the threshold ``t`` for a
target volume fraction ``rho`` is the (1 - rho) quantile of ``f`` over the
cell, so the same object serves any density.
"""

from __future__ import annotations

import numpy as np

from .cells import UnitCell, as_cell
from .topologies import Skeleton, skeleton as _lookup_skeleton


class UnitField:
    """Periodic scalar field of one unit cell; the solid is ``f >= threshold``.

    The field is sampled at fractional coordinates ``u = n / N`` of one cell.
    The threshold for a target volume fraction is the ``1 - rho`` quantile
    of the samples, so the same object serves any density, uniform or graded.
    Build one with :func:`sdf_field`, :func:`fourier_field`,
    :func:`tpms_field` or :func:`custom_field`.

    Parameters
    ----------
    cell : UnitCell, float or array_like
        The cell (see :func:`~latticegen.cells.as_cell`).
    f : array_like, shape (N, N, N)
        Field samples; stored as float32.
    name : str, optional
    method : str, optional
        ``'sdf'``, ``'fourier'``, ``'tpms'`` or ``'custom'``.
    skeleton : Skeleton, optional
        Skeleton the field was built from (enables strut and node sizes in
        :meth:`feature_sizes`).
    meta : dict, optional
        Builder parameters, kept for reference.

    Attributes
    ----------
    cell : UnitCell
    f : ndarray, float32, shape (N, N, N)
    name, method : str
    skeleton : Skeleton or None
    meta : dict
    """

    def __init__(self, cell, f, name="field", method="custom", skeleton=None, meta=None):
        self.cell: UnitCell = as_cell(cell)
        f = np.asarray(f)
        if f.ndim != 3 or len(set(f.shape)) != 1:
            raise ValueError("f must be a cubic N x N x N array")
        self.f = f.astype(np.float32) if f.dtype != np.float32 else f
        self.name = name
        self.method = method
        self.skeleton: Skeleton | None = skeleton
        self.meta = dict(meta or {})
        self._sorted = None
        self._calibrated = {}

    # ------------------------------------------------------------------ #
    @property
    def N(self) -> int:
        """Samples per cell edge.
        """
        return self.f.shape[0]

    @property
    def A(self) -> np.ndarray:
        """Lattice vectors of the cell (rows).
        """
        return self.cell.A

    @property
    def voxel_size(self) -> np.ndarray:
        """Grid spacing along each cell vector.

        Returns
        -------
        ndarray, shape (3,)
            Cell lengths divided by ``N`` (cell units, e.g. mm).
        """
        return self.cell.lengths / self.N

    @property
    def sorted_values(self) -> np.ndarray:
        """Sorted field samples (float64, cached): the quantile function.
        """
        if self._sorted is None:
            self._sorted = np.sort(self.f.astype(np.float64), axis=None)
        return self._sorted

    # -- density <-> threshold --------------------------------------------- #
    def threshold(self, rho):
        """Isovalue whose solid ``{f >= t}`` has volume fraction ``rho``.

        Uses the quantile of the voxel samples. Fast and vectorised, but
        quantised on highly symmetric fields; see :meth:`calibrated_threshold`.

        Parameters
        ----------
        rho : float or array_like
            Target volume fraction(s), strictly inside (0, 1).

        Returns
        -------
        float or ndarray
        """
        rho = np.asarray(rho, float)
        if np.any(rho <= 0.0) or np.any(rho >= 1.0):
            raise ValueError("rho must lie strictly inside (0, 1)")
        S = self.sorted_values
        p = np.arange(S.size) / (S.size - 1.0)
        return np.interp(1.0 - rho, p, S)

    def calibrated_threshold(self, rho, tol=2e-4, max_iter=6) -> float:
        """Isovalue that gives volume fraction ``rho`` on the marching-cubes mesh.

        Symmetric fields have many voxels with equal values, so the plain
        quantile can map nearby densities to one isovalue (errors up to ~1 %
        absolute at N = 48). This meshes one closed unit cell, measures its
        volume, and iterates the isovalue with a secant method. Results are
        cached per ``rho``.

        Parameters
        ----------
        rho : float
            Target volume fraction.
        tol : float, optional
            Accepted absolute density error.
        max_iter : int, optional
            Maximum secant steps.

        Returns
        -------
        float
        """
        from .meshing import mesh_stack
        rho = float(rho)
        key = round(rho, 9)
        if key in self._calibrated:
            return self._calibrated[key]
        fmin, fmax = float(self.f.min()), float(self.f.max())
        vol = self.cell.volume

        def measure(t):
            t = min(max(t, fmin + 1e-6 * (fmax - fmin)), fmax - 1e-6 * (fmax - fmin))
            return t, mesh_stack(self, threshold=t, cells=(1, 1, 1)).volume / vol

        t0, r0 = measure(float(self.threshold(rho)))
        if abs(r0 - rho) <= tol:
            self._calibrated[key] = t0
            return t0
        d = 0.02
        lo, hi = max(rho - d, 1e-3), min(rho + d, 1 - 1e-3)
        slope = float(self.threshold(lo) - self.threshold(hi)) / (hi - lo)   # |dt/drho|
        if not slope > 0:
            slope = 0.1 * (fmax - fmin)
        t1, r1 = measure(t0 - (rho - r0) * slope)
        for _ in range(max_iter):
            if abs(r1 - rho) <= tol or r1 == r0:
                break
            t2 = t1 - (r1 - rho) * (t1 - t0) / (r1 - r0)
            t0, r0 = t1, r1
            t1, r1 = measure(t2)
        best = t1 if abs(r1 - rho) <= abs(r0 - rho) else t0
        self._calibrated[key] = best
        return best

    def density(self, t) -> float:
        """Volume fraction of ``{f >= t}`` by voxel count.

        Voxels exactly on the isovalue count one half.

        Parameters
        ----------
        t : float
            Isovalue.

        Returns
        -------
        float
        """
        return float((self.f > t).mean() + 0.5 * (self.f == t).mean())

    def density_curve(self, ts) -> np.ndarray:
        """:meth:`density` for several isovalues.

        Parameters
        ----------
        ts : array_like

        Returns
        -------
        ndarray
        """
        return np.array([self.density(t) for t in np.atleast_1d(ts)])

    def solid_mask(self, rho) -> np.ndarray:
        """Boolean voxel mask of the solid at density ``rho``.

        Parameters
        ----------
        rho : float

        Returns
        -------
        ndarray of bool, shape (N, N, N)
        """
        return self.f >= self.threshold(rho)

    # -- feature sizes ------------------------------------------------------ #
    def strut_radius(self, rho) -> float:
        """Strut radius at density ``rho`` (cell units).

        Exact for distance fields (``r = -t``); for other fields, the inscribed
        radius at the strut midpoints from a periodic distance transform.

        Parameters
        ----------
        rho : float

        Returns
        -------
        float
        """
        if self.method == "sdf":
            return float(-self.threshold(rho))
        return self.feature_sizes(rho)["strut_radius"]

    def feature_sizes(self, rho, sample_ratio=None) -> dict:
        """Characteristic sizes of the solid and the void at density ``rho``.

        Computed with periodic Euclidean distance transforms of the voxel mask,
        in cell units.

        Parameters
        ----------
        rho : float
        sample_ratio : optional
            Unused; kept for compatibility.

        Returns
        -------
        dict
            ``min_thickness``
                Twice a low percentile of the solid's medial ridge: the thinnest
                place of any solid feature.
            ``max_thickness``
                Diameter of the largest sphere inside the solid.
            ``pore_radius``
                Radius of the largest sphere inside the void.
            ``pore_throat``
                Twice a low percentile of the void's medial ridge: the narrowest
                passage for resin drainage.
            ``strut_radius``, ``strut_radius_min``, ``node_radius``, ``node_to_strut_ratio``
                Only for fields built from a skeleton: inscribed radii at the
                strut midpoints and nodes.

            Plus the keys of :meth:`connectivity_check`.
        """
        from scipy import ndimage

        N = self.N
        spacing = self.voxel_size
        if not np.allclose(spacing, spacing[0], rtol=1e-3) or not self.cell.is_cubic:
            # EDT assumes an orthogonal grid; use the mean spacing as approximation
            spacing = np.full(3, spacing.mean())
        mask = self.solid_mask(rho)
        pad = N // 2

        def periodic_edt(m):
            mp = np.pad(m, pad, mode="wrap")
            e = ndimage.distance_transform_edt(mp, sampling=spacing)
            return e[pad:pad + N, pad:pad + N, pad:pad + N]

        e_solid = periodic_edt(mask)
        e_void = periodic_edt(~mask)
        out = {"max_thickness": 2 * float(e_solid.max()),
               "pore_radius": float(e_void.max())}

        def ridge_min(e, m, pct=2.0):
            # medial ridge = local maxima of the EDT inside the phase; a low
            # percentile of the ridge values is a robust "narrowest place"
            mx = ndimage.maximum_filter(np.pad(e, 2, mode="wrap"), size=5)[2:-2, 2:-2, 2:-2]
            ridge = m & (e >= mx - 1e-9) & (e > 1.5 * spacing.min())
            return float(np.percentile(e[ridge], pct)) if ridge.any() else float("nan")

        out["min_thickness"] = 2 * ridge_min(e_solid, mask)
        out["pore_throat"] = 2 * ridge_min(e_void, ~mask)
        out.update(self.connectivity_check(rho, mask))

        if self.skeleton is not None and self.skeleton.n_struts:
            def sample(points):
                idx = np.round(np.mod(points, 1.0) * N).astype(int) % N
                return e_solid[idx[:, 0], idx[:, 1], idx[:, 2]]
            r_mid = sample(self.skeleton.midpoints())
            r_node = sample(self.skeleton.nodes())
            out["strut_radius"] = float(np.median(r_mid))
            out["strut_radius_min"] = float(r_mid.min())
            out["node_radius"] = float(np.median(r_node))
            out["node_to_strut_ratio"] = out["node_radius"] / max(out["strut_radius"], 1e-12)
        return out

    def phase_components(self, rho, mask=None) -> tuple:
        """Periodically connected components of the solid and the void.

        Parameters
        ----------
        rho : float
        mask : ndarray of bool, optional
            Precomputed :meth:`solid_mask`.

        Returns
        -------
        tuple of int
            ``(solid_components, void_components)``. A printable lattice has one
            solid component; a sheet TPMS legitimately has two void labyrinths.
        """
        c = self.connectivity_check(rho, mask)
        return c["solid_components"], c["void_components"]

    def connectivity_check(self, rho, mask=None) -> dict:
        """Connectivity of the two phases at density ``rho``.

        Components are labelled and merged across the periodic faces
        (union-find). A piece that never reaches the boundary of a 2x2x2 tiling
        is a closed pocket or a floating island.

        Parameters
        ----------
        rho : float
        mask : ndarray of bool, optional
            Precomputed :meth:`solid_mask`.

        Returns
        -------
        dict
            ``solid_components``, ``void_components``
                Periodically connected pieces of each phase.
            ``floating_solids``
                Solid pieces that do not percolate; they would float in the vat.
            ``enclosed_voids``
                Void pockets that do not percolate; they trap uncured resin.
        """
        from scipy import ndimage
        mask = self.solid_mask(rho) if mask is None else mask
        N = self.N

        def periodic_count(m):
            lab, n = ndimage.label(m)
            if n <= 1:
                return int(n)
            parent = np.arange(n + 1)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for ax in range(3):
                lo = np.take(lab, 0, axis=ax).ravel()
                hi = np.take(lab, N - 1, axis=ax).ravel()
                for a, b in set(zip(lo.tolist(), hi.tolist())):
                    if a and b:
                        ra, rb = find(a), find(b)
                        if ra != rb:
                            parent[ra] = rb
            return len({find(x) for x in range(1, n + 1)})

        def pockets(m):
            t = np.tile(m, (2, 2, 2))
            lab, n = ndimage.label(t)
            if n == 0:
                return 0
            border = np.zeros_like(t, bool)
            border[[0, -1], :, :] = border[:, [0, -1], :] = border[:, :, [0, -1]] = True
            touching = np.unique(lab[border & t])
            interior = n - len(touching[touching > 0])
            return int(np.ceil(interior / 8))            # copies -> distinct pockets

        return dict(solid_components=periodic_count(mask),
                    void_components=periodic_count(~mask),
                    floating_solids=pockets(mask),
                    enclosed_voids=pockets(~mask))

    # -- meshing shortcuts ---------------------------------------------------- #
    def mesh(self, rho=0.3, cells=(1, 1, 1), **kw):
        """Mesh a block of this field; shortcut for :func:`~latticegen.mesh_stack`.

        Parameters
        ----------
        rho : float, array_like or callable
        cells : tuple of 3 int
        **kw
            Passed to :func:`~latticegen.mesh_stack`.

        Returns
        -------
        LatticeMesh
        """
        from .meshing import mesh_stack
        return mesh_stack(self, rho=rho, cells=cells, **kw)

    def mesh_instanced(self, rho=0.3, cells=(1, 1, 1), **kw):
        """Lean homogeneous block; shortcut for :func:`~latticegen.mesh_instanced`.

        Parameters
        ----------
        rho : float
        cells : tuple of 3 int
        **kw
            Passed to :func:`~latticegen.mesh_instanced`.

        Returns
        -------
        LatticeMesh
        """
        from .instanced import mesh_instanced
        return mesh_instanced(self, rho=rho, cells=cells, **kw)

    # -- persistence ------------------------------------------------------------ #
    def save(self, path):
        """Save the field and cell to a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or path-like
        """
        np.savez_compressed(path, f=self.f, A=self.A, name=self.name, method=self.method)

    @classmethod
    def load(cls, path) -> "UnitField":
        """Load a field written by :meth:`save`.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        UnitField
            Without the skeleton and builder metadata.
        """
        z = np.load(path, allow_pickle=False)
        return cls(z["A"], z["f"], name=str(z["name"]), method=str(z["method"]))

    def __repr__(self) -> str:
        return f"UnitField({self.name}, method={self.method}, N={self.N}, cell={self.cell.name})"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

def _skel(topology) -> Skeleton:
    return topology if isinstance(topology, Skeleton) else _lookup_skeleton(str(topology))


def fourier_field(cell, topology, N=96, sigma=None, gmax=None, node_comp=None,
                  node_sigma=None, weights=None) -> UnitField:
    """Fourier-synthesis field of a strut topology: soft, organic struts.

    Every strut contributes its structure factor
    ``exp(-i G.r_mid) sinc(G.d / 2)``, damped by a Gaussian form factor
    ``exp(-sigma^2 |G|^2 / 2)``; the band-limited sum is evaluated on the grid
    with one inverse FFT.

    Parameters
    ----------
    cell : UnitCell, float or array_like
        The cell (a float is a cubic edge in mm).
    topology : str or Skeleton
        Built-in name (``'octet'``, ``'bcc'``, ...) or a skeleton.
    N : int, optional
        Samples per cell edge. Must exceed twice the largest Miller index;
        a ``ValueError`` says so otherwise.
    sigma : float, optional
        Gaussian smearing radius (cell length units, e.g. mm). Default: a
        topology-tuned fraction of the shortest cell edge (octet 0.06).
    gmax : float, optional
        Reciprocal cutoff in units of ``2 pi / a_min``. Default: tuned per
        topology (e.g. octet 6.5).
    node_comp : float, optional
        Junction slimming. Struts' fields add up at the nodes, so the solid
        bulges there; a value in (0, 1] subtracts a wider Gaussian at every
        node so that the node field becomes
        ``node_comp * f(mid-strut) + (1 - node_comp) * f(node)``. ``None``
        (default) is off; 0.5 makes an octet junction about as thick as a
        sharp cylinder union.
    node_sigma : float, optional
        Width of the compensating Gaussian; default ``1.7 * sigma``.
    weights : array_like, optional
        Per-strut weights of the motif.

    Returns
    -------
    UnitField
        ``method='fourier'``; ``meta`` holds the coefficients.
    """
    from . import fourier
    cell = as_cell(cell)
    sk = _skel(topology)
    hkl, G, c, params = fourier.fourier_coefficients(cell, sk, sigma=sigma, gmax=gmax,
                                                     node_comp=node_comp, node_sigma=node_sigma,
                                                     weights=weights)
    f = fourier.field_on_grid(hkl, c, N)
    return UnitField(cell, f, name=f"{sk.name}-fourier", method="fourier", skeleton=sk,
                     meta=dict(params, hkl=hkl, c=c))


def sdf_field(cell, topology, N=96, smooth=0.0, **kw) -> UnitField:
    """Distance field of a strut topology: exact cylinders, optional fillets.

    The field is minus the distance to the nearest strut (over all periodic
    images), so the solid at isovalue ``t`` is every point within ``-t`` of
    the skeleton.

    Parameters
    ----------
    cell : UnitCell, float or array_like
        The cell (a float is a cubic edge in mm).
    topology : str or Skeleton
        Built-in name or a skeleton.
    N : int, optional
        Samples per cell edge.
    smooth : float, optional
        0 for sharp cylinder unions. ``k > 0`` blends the strut distances with
        a log-sum-exp smooth minimum, which rounds every junction with a
        fillet of about ``k`` (cell units, e.g. mm). Keep it near 10 to 30 %
        of the strut radius if uniform sections matter.
    **kw
        Passed to :func:`latticegen.sdf.distance_field`.

    Returns
    -------
    UnitField
        ``method='sdf'``; :meth:`UnitField.strut_radius` is exact.
    """
    from . import sdf
    cell = as_cell(cell)
    sk = _skel(topology)
    f = sdf.sdf_on_grid(cell, sk, N, smooth=smooth, **kw)
    return UnitField(cell, f, name=f"{sk.name}-sdf", method="sdf", skeleton=sk,
                     meta=dict(smooth=smooth))


def tpms_field(cell, name="gyroid", N=64, sheet=False, periods=(1, 1, 1), phase=(0, 0, 0)) -> UnitField:
    """Field of a triply periodic minimal surface.

    Parameters
    ----------
    cell : UnitCell, float or array_like
        The cell; the surface is stretched to fit a non-cubic cell.
    name : str, optional
        ``'gyroid'``, ``'primitive'``, ``'diamond'``, ``'iwp'``,
        ``'neovius'``, ``'lidinoid'``, ``'fischer_koch_s'`` or ``'split_p'``.
    N : int, optional
        Samples per cell edge.
    sheet : bool, optional
        False (default): network solid, one labyrinth filled. True: a wall
        of finite thickness around the surface; both labyrinths stay open,
        which drains resin well.
    periods : tuple of 3 int, optional
        Repeats of the surface per cell along each cell vector.
    phase : tuple of 3 float, optional
        Shift of the surface, in fractions of the period.

    Returns
    -------
    UnitField
        ``method='tpms'``.
    """
    from . import tpms
    cell = as_cell(cell)
    f = tpms.tpms_on_grid(name, N, periods=periods, sheet=sheet, phase=phase)
    kind = "sheet" if sheet else "network"
    return UnitField(cell, f, name=f"{name}-{kind}", method="tpms",
                     meta=dict(tpms=name, sheet=sheet, periods=tuple(periods)))


def custom_field(cell, f, name="custom") -> UnitField:
    """Wrap your own periodic field.

    Parameters
    ----------
    cell : UnitCell, float or array_like
    f : array_like, shape (N, N, N)
        Samples at fractional coordinates ``n / N``; must be periodic.
        The solid is ``f >= threshold``.
    name : str, optional

    Returns
    -------
    UnitField
    """
    return UnitField(cell, f, name=name, method="custom")
