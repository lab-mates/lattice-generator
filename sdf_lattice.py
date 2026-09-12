"""
Distance-field (SDF) periodic lattices.

Complement to fourier_lattice.py: instead of a truncated Fourier sum, the
unit-cell field is the exact shortest distance d(r) from each voxel to the
strut skeleton (point-to-segment, minimised over all periodic images).
The level set d <= r gives struts with EXACTLY circular cross-section away
from junctions, and junctions that are plain unions of cylinders (or
smoothly filleted ones, see `smooth`).

The field handed downstream is f = -d, so the whole fourier_lattice
meshing pipeline applies unchanged (solid = f >= t means d <= -t, the
quantile -> density relation, continuous/graded rho, heterogeneous
stacking, flat caps, welding, STL export):

    import fourier_lattice as fl
    import sdf_lattice as sdfl

    f = sdfl.lattice_field(fl.octet_segments(), N=96, smooth=0.02)
    verts, faces, _ = fl.stacked_lattice_mesh(fl.cubic_cell(), f,
                                              rho=0.30, cells=(3, 3, 3))
    fl.write_stl("octet_sdf.stl", verts, faces)

Fillets: `smooth` = 0 gives the hard min (sharp re-entrant corners at the
nodes). smooth = k > 0 uses an exponential smooth-min (log-sum-exp),
rounding the junctions with a fillet of size ~k. Unlike the Fourier sum,
the node pile-up is BOUNDED: where n struts meet, the effective distance
drops by at most k*ln(n) (~2.5k for the 12-connected octet node), so the
junction bulge is small and directly controllable — convenient when a
uniform section matters (e.g. pyrolysis).
"""

import itertools
import numpy as np


def periodic_segment_images(segments, margin=0.9):
    """
    Expand fractional-coordinate segments (p, q) over the 27 neighbouring
    cells and keep those whose bounding box comes within `margin` of the
    unit cell. margin must exceed the largest voxel-to-skeleton distance
    (< sqrt(3)/2 for any connected cubic lattice); the default is safe.
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


def distance_field(segments, N, A=None, smooth=0.0, chunk_layers=8):
    """
    Exact shortest distance from each voxel of the unit cell to the strut
    skeleton, minimised over periodic images. Cartesian metric via the
    lattice matrix A (rows a_i; default identity, i.e. a = 1).

    segments     : list of fractional endpoint pairs (p, q), e.g. from
                   fourier_lattice.octet_segments().
    smooth       : 0 -> hard min (sharp unions). k > 0 -> exponential
                   smooth-min with fillet scale ~k (same length units as A).
    chunk_layers : z-layers processed at a time (memory control).

    Returns d with shape (N, N, N). Cost ~ N^3 * n_segment_images
    exact point-segment distances (a few seconds at N=96 for the octet).
    """
    A = np.eye(3) if A is None else np.asarray(A, float)
    imgs = periodic_segment_images(segments)
    PQ = [(np.asarray(p) @ A, (np.asarray(q) - np.asarray(p)) @ A)
          for p, q in imgs]                         # cartesian origin + axis
    u = np.arange(N) / N
    d = np.empty((N, N, N))
    if smooth > 0.0:
        beta = 1.0 / smooth

    for z0 in range(0, N, chunk_layers):
        z1 = min(z0 + chunk_layers, N)
        X, Y, Z = np.meshgrid(u, u, u[z0:z1], indexing='ij')
        R = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) @ A
        m = np.full(len(R), np.inf)                  # running (smooth-)min
        if smooth > 0.0:
            s = np.zeros(len(R))                     # sum exp(-beta*(d_i - m))
        for P, D in PQ:
            L2 = D @ D
            t = np.clip(((R - P) @ D) / L2, 0.0, 1.0)
            di = np.linalg.norm(R - P - t[:, None] * D, axis=1)
            if smooth == 0.0:
                np.minimum(m, di, out=m)
            else:                                    # streaming log-sum-exp
                lower = di < m
                s = np.where(lower,
                             s * np.exp(-beta * np.minimum(m - di, 700 / beta)) + 1.0,
                             s + np.exp(-beta * (di - m)))
                m = np.where(lower, di, m)
        if smooth > 0.0:
            m = m - np.log(np.maximum(s, 1.0)) / beta
        d[:, :, z0:z1] = m.reshape(N, N, z1 - z0)
    return d


def lattice_field(segments, N, A=None, smooth=0.0, **kw):
    """f = -distance: ready for fourier_lattice.stacked_lattice_mesh
    (solid = f >= t, threshold from the density quantile as usual)."""
    return -distance_field(segments, N, A=A, smooth=smooth, **kw)


def radius_for_density(d, rho):
    """Strut radius r such that the solid fraction of {d <= r} is rho.
    (Equivalent to the threshold the meshing pipeline picks: t = -r.)"""
    return np.quantile(d, rho)


if __name__ == "__main__":
    import fourier_lattice as fl

    N, rho = 96, 0.30
    A = fl.cubic_cell()
    for name, segs in [("sc", fl.sc_segments()),
                       ("bcc", fl.bcc_segments()),
                       ("diamond", fl.diamond_segments()),
                       ("octet", fl.octet_segments())]:
        for k in (0.0, 0.02):
            f = lattice_field(segs, N, A, smooth=k)
            r = radius_for_density(-f, rho)
            verts, faces, _ = fl.stacked_lattice_mesh(A, f, rho=rho,
                                                      cells=(2, 2, 2))
            vol = fl.mesh_volume(verts, faces) / (8 * np.linalg.det(A))
            print(f"{name:8s} smooth={k:4.2f}  r={r:.4f}a  "
                  f"watertight={fl.is_watertight(faces)}  V/V_box={vol:.3f}")
    f = lattice_field(fl.octet_segments(), N, A, smooth=0.02)
    verts, faces, _ = fl.stacked_lattice_mesh(A, f, rho=rho, cells=(2, 2, 2))
    fl.write_stl("/home/claude/octet_sdf.stl", verts, faces)
