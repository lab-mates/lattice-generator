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
    """Periodic scalar field of one unit cell (solid = f >= threshold)."""

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
        return self.f.shape[0]

    @property
    def A(self) -> np.ndarray:
        return self.cell.A

    @property
    def voxel_size(self) -> np.ndarray:
        """Grid spacing along each cell vector (cell units)."""
        return self.cell.lengths / self.N

    @property
    def sorted_values(self) -> np.ndarray:
        """Sorted field samples (float64), cached; the quantile function."""
        if self._sorted is None:
            self._sorted = np.sort(self.f.astype(np.float64), axis=None)
        return self._sorted

    # -- density <-> threshold --------------------------------------------- #
    def threshold(self, rho):
        """Isovalue t such that the volume fraction of {f >= t} is rho.
        Accepts scalars or arrays (vectorised via the quantile function)."""
        rho = np.asarray(rho, float)
        if np.any(rho <= 0.0) or np.any(rho >= 1.0):
            raise ValueError("rho must lie strictly inside (0, 1)")
        S = self.sorted_values
        p = np.arange(S.size) / (S.size - 1.0)
        return np.interp(1.0 - rho, p, S)

    def calibrated_threshold(self, rho, tol=2e-4, max_iter=6) -> float:
        """Isovalue giving volume fraction ``rho`` of the actual marching-cubes
        unit-cell mesh (secant iteration on the closed single-cell volume).

        The plain quantile mapping can be quantised on highly symmetric
        fields (many voxels share the same value), which makes nearby
        densities collapse onto one isovalue; this removes that error at the
        cost of a few single-cell meshings.  Results are cached per rho.
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
        """Volume fraction of the solid {f >= t} estimated by voxel count
        (voxels exactly on the isovalue count one half)."""
        return float((self.f > t).mean() + 0.5 * (self.f == t).mean())

    def density_curve(self, ts) -> np.ndarray:
        return np.array([self.density(t) for t in np.atleast_1d(ts)])

    def solid_mask(self, rho) -> np.ndarray:
        return self.f >= self.threshold(rho)

    # -- feature sizes ------------------------------------------------------ #
    def strut_radius(self, rho) -> float:
        """Strut radius (cell length units).  Exact for SDF fields (r = -t);
        for other fields the inscribed radius at strut midpoints measured
        with a periodic distance transform."""
        if self.method == "sdf":
            return float(-self.threshold(rho))
        return self.feature_sizes(rho)["strut_radius"]

    def feature_sizes(self, rho, sample_ratio=None) -> dict:
        """Estimate characteristic sizes of the solid and the void at density
        ``rho`` with periodic Euclidean distance transforms (cell units):

        strut_radius  inscribed radius at strut midpoints (skeleton fields)
        node_radius   inscribed radius at the nodes (skeleton fields)
        min_thickness 2 x smallest ridge value of the solid EDT (any field)
        max_thickness 2 x largest inscribed sphere of the solid
        pore_radius   largest inscribed sphere of the void (drainage)
        pore_throat   2 x smallest ridge value of the void EDT
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
        """Periodically connected components of (solid, void) at density
        ``rho``.  A printable lattice needs exactly one solid component; a
        sheet TPMS legitimately has two void labyrinths."""
        c = self.connectivity_check(rho, mask)
        return c["solid_components"], c["void_components"]

    def connectivity_check(self, rho, mask=None) -> dict:
        """Connectivity of the two phases at density ``rho``:

        solid_components / void_components : periodically connected pieces
        floating_solids : solid pieces that do not percolate (would float)
        enclosed_voids  : void pockets that do not percolate (trap resin)

        Percolation is judged on a 2x2x2 tiling: a piece that never reaches
        the tiling boundary is a closed pocket / island.
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
        """Mesh an (nx, ny, nz) block of this field (see :func:`mesh_stack`)."""
        from .meshing import mesh_stack
        return mesh_stack(self, rho=rho, cells=cells, **kw)

    def mesh_instanced(self, rho=0.3, cells=(1, 1, 1), **kw):
        """Lean homogeneous block via unit-cell instancing (see :func:`mesh_instanced`)."""
        from .instanced import mesh_instanced
        return mesh_instanced(self, rho=rho, cells=cells, **kw)

    # -- persistence ------------------------------------------------------------ #
    def save(self, path):
        np.savez_compressed(path, f=self.f, A=self.A, name=self.name, method=self.method)

    @classmethod
    def load(cls, path) -> "UnitField":
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
    """Fourier-synthesis field of a strut topology.

    cell      : UnitCell, 3x3 matrix or cubic edge length.
    topology  : Skeleton or built-in name ('octet', 'bcc', ...).
    N         : grid samples per cell edge (must exceed 2*max|hkl|).
    sigma     : Gaussian smearing radius (cell units); default topology-tuned.
    gmax      : reciprocal cutoff in units of 2*pi/a_min; default tuned.
    node_comp : junction de-bulking strength (e.g. 0.5); None = off.
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
    """Distance-field (SDF) of a strut topology; ``f = -distance``.

    smooth : 0 for sharp cylinder unions, k > 0 for junction fillets of size
             ~k (cell units).  Extra keywords go to :func:`sdf.distance_field`.
    """
    from . import sdf
    cell = as_cell(cell)
    sk = _skel(topology)
    f = sdf.sdf_on_grid(cell, sk, N, smooth=smooth, **kw)
    return UnitField(cell, f, name=f"{sk.name}-sdf", method="sdf", skeleton=sk,
                     meta=dict(smooth=smooth))


def tpms_field(cell, name="gyroid", N=64, sheet=False, periods=(1, 1, 1), phase=(0, 0, 0)) -> UnitField:
    """TPMS level-set field ('gyroid', 'primitive', 'diamond', 'iwp',
    'neovius', 'lidinoid', 'fischer_koch_s', 'split_p').  ``sheet=True`` for
    a wall-type solid around the surface (both labyrinths open)."""
    from . import tpms
    cell = as_cell(cell)
    f = tpms.tpms_on_grid(name, N, periods=periods, sheet=sheet, phase=phase)
    kind = "sheet" if sheet else "network"
    return UnitField(cell, f, name=f"{name}-{kind}", method="tpms",
                     meta=dict(tpms=name, sheet=sheet, periods=tuple(periods)))


def custom_field(cell, f, name="custom") -> UnitField:
    """Wrap your own periodic N^3 array (solid where f >= threshold)."""
    return UnitField(cell, f, name=name, method="custom")
