"""
Fourier synthesis of periodic lattices for 3D printing.

A field with the periodicity of a Bravais lattice is a sum of plane waves
over reciprocal-lattice vectors G:

    f(r) = sum_G  c_G * exp(i G . r),   c_{-G} = conj(c_G)  (real field)

Coefficients are built from a structure-factor-like motif (points and/or
line segments = struts, Gaussian-smeared), or specified manually (TPMS).
The solid is the level set  f(r) >= t;  t is fixed from a target volume
fraction via the quantile of f over one unit cell.
"""

import numpy as np
from skimage.measure import marching_cubes
import trimesh


# ----------------------------------------------------------------------
# Reciprocal lattice
# ----------------------------------------------------------------------

def reciprocal_basis(A):
    """A: 3x3, rows = a1,a2,a3. Returns B with rows b1,b2,b3, b_i.a_j = 2*pi*delta_ij."""
    return 2.0 * np.pi * np.linalg.inv(A).T


def enumerate_G(A, gmax):
    """All reciprocal vectors with 0 < |G| <= gmax, as (integer hkl, cartesian G)."""
    B = reciprocal_basis(A)
    # |h_i| = |a_i . G| / 2pi <= |a_i| gmax / 2pi
    hmax = np.ceil(np.linalg.norm(A, axis=1) * gmax / (2 * np.pi)).astype(int)
    hkl, G = [], []
    for h in range(-hmax[0], hmax[0] + 1):
        for k in range(-hmax[1], hmax[1] + 1):
            for l in range(-hmax[2], hmax[2] + 1):
                if h == k == l == 0:
                    continue
                g = h * B[0] + k * B[1] + l * B[2]
                if np.linalg.norm(g) <= gmax:
                    hkl.append((h, k, l))
                    G.append(g)
    return hkl, np.array(G)


# ----------------------------------------------------------------------
# Structure-factor coefficients from a motif
# ----------------------------------------------------------------------

def motif_coefficients(A, hkl, G, points=(), segments=(), sigma=0.1):
    """
    c_G = [ sum_j w_j e^{-i G.r_j}
          + sum_s w_s e^{-i G.r_mid} sinc(G.d/2) ] * exp(-sigma^2 |G|^2 / 2)

    points   : list of (x_frac, weight), x_frac = fractional coords (3,)
    segments : list of (x1_frac, x2_frac, weight)  -- struts
    sigma    : Gaussian smearing radius (same units as A) -> strut/ball radius
    """
    c = np.zeros(len(G), dtype=complex)
    hkl = np.array(hkl, dtype=float)
    for x, w in points:
        c += w * np.exp(-2j * np.pi * (hkl @ np.asarray(x, float)))
    for x1, x2, w in segments:
        x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
        xm, d = 0.5 * (x1 + x2), (x2 - x1)
        phase = np.exp(-2j * np.pi * (hkl @ xm))
        GdotD = 2.0 * np.pi * (hkl @ d)           # G . (r2 - r1)
        c += w * phase * np.sinc(GdotD / (2 * np.pi))  # np.sinc(x)=sin(pi x)/(pi x)
    G2 = np.einsum('ij,ij->i', G, G)
    return c * np.exp(-0.5 * sigma ** 2 * G2)


# ----------------------------------------------------------------------
# Field evaluation on one unit cell (fractional grid) via inverse FFT
# ----------------------------------------------------------------------

def field_on_grid(hkl, c, N):
    """f on an N^3 grid of fractional coordinates u = n/N.  Needs N > 2*max|h|."""
    hmax = max(abs(i) for t in hkl for i in t)
    if N <= 2 * hmax:
        raise ValueError(f"N={N} too small for hmax={hmax}; need N > {2*hmax}")
    C = np.zeros((N, N, N), dtype=complex)
    for (h, k, l), ci in zip(hkl, c):
        C[h % N, k % N, l % N] += ci
    return np.real(np.fft.ifftn(C)) * N ** 3


# ----------------------------------------------------------------------
# Level set -> mesh  (slab-streamed, flat caps, per-cell lattice/density)
# ----------------------------------------------------------------------



def stacked_lattice_mesh(A, unit_fields, rho=0.3, cells=(1, 1, 1),
                         cell_map=None, slab_cells=1):
    """
    Assemble an (nx, ny, nz) stack of unit cells and mesh it.

    unit_fields : one N^3 array, or a list of N^3 arrays (same N) — the
                  available lattice types (from field_on_grid).
    rho         : target volume fraction. Scalar, (nx,ny,nz) array
                  (per-cell), or a CALLABLE rho(zeta) of the fractional
                  stack height zeta = z / (nz*a_z) in [0,1] — a continuous
                  density gradation along z. The callable must accept a
                  numpy array. The rho->isovalue map uses the unit-cell
                  quantile function, so the lattice geometry is preserved
                  and the density is met in the cell-averaged sense
                  (exact in the limit of slow variation over one cell).
    cell_map    : (nx,ny,nz) int array selecting the lattice type of each
                  cell (indexes into unit_fields). Default: all type 0.
    slab_cells  : cells per z-slab processed at a time (memory control).
                  Peak memory ~ (nx*N)(ny*N)(slab_cells*N) floats.

    Marching cubes runs on the *uncapped* periodic field; caps are flat:
    vertices generated in the sacrificial pad layer are clamped onto the
    exact bounding planes (equivalent to isosurface + isocaps).

    Returns (verts, faces, threshold_info) as numpy arrays (welded,
    outward-facing). Use write_stl / is_watertight / mesh_volume, or wrap
    in trimesh for small meshes.
    """
    fields = [unit_fields] if isinstance(unit_fields, np.ndarray) else list(unit_fields)
    Gf = np.stack(fields)                      # (T, N, N, N)
    N = Gf.shape[1]
    nx, ny, nz = cells
    cell_map = np.zeros(cells, int) if cell_map is None else np.asarray(cell_map, int)
    continuous = callable(rho)

    if continuous:
        # threshold per lattice type at every global z sample, via each
        # type's quantile function: t(rho) = quantile(f, 1 - rho)
        zeta = np.arange(nz * N + 1) / (nz * N)
        rho_z = np.asarray(rho(zeta), float)
        if rho_z.min() <= 0.0 or rho_z.max() >= 1.0:
            raise ValueError("rho(zeta) must stay strictly inside (0, 1)")
        Tz = np.empty((len(fields), nz * N + 1))
        for ti, f in enumerate(fields):
            S = np.sort(f, axis=None)
            p = np.arange(S.size) / (S.size - 1.0)
            Tz[ti] = np.interp(1.0 - rho_z, p, S)
        info, t_max = Tz, Tz.max()
    else:
        rho_map = np.broadcast_to(np.asarray(rho, float), cells)
        # per-cell isovalue offsets: threshold = quantile of that cell's field
        cache, O = {}, np.empty(cells)
        for idx in np.ndindex(*cells):
            key = (cell_map[idx], round(float(rho_map[idx]), 6))
            if key not in cache:
                cache[key] = np.quantile(fields[key[0]], 1.0 - key[1])
            O[idx] = cache[key]
        info, t_max = cache, O.max()

    scale = float(Gf.max() - Gf.min())
    level = 1e-6 * scale                       # nudge off symmetric grid values
    pad_val = np.float32(Gf.min() - t_max - scale)     # safely below level

    all_v, all_f, nv = [], [], 0
    for k0 in range(0, nz, slab_cells):
        k1 = min(k0 + slab_cells, nz)
        zpad = (1 if k0 == 0 else 0, 1 if k1 == nz else 0)
        # slab array with cap padding preallocated; interior filled block-wise
        Mx, My, Mz = nx * N + 1, ny * N + 1, (k1 - k0) * N + 1
        F = np.full((Mx + 2, My + 2, Mz + sum(zpad)), pad_val, np.float32)
        ox, oy, oz = 1, 1, zpad[0]
        Fi = F[ox:ox + Mx, oy:oy + My, oz:oz + Mz]     # interior view
        for i in range(nx):
            for j in range(ny):
                for k in range(k0, k1):
                    ti = cell_map[i, j, k]
                    sub = Tz[ti, k*N:(k+1)*N] if continuous else O[i, j, k]
                    Fi[i*N:(i+1)*N, j*N:(j+1)*N, (k-k0)*N:(k-k0+1)*N] = \
                        Gf[ti] - sub
        # top sample layer: at internal joints it belongs to the NEXT cell
        # layer (matches the next slab's bottom layer -> vertices weld);
        # at the global far face it's the periodic closure of the last cell.
        # With a z-dependent threshold the closure can't be a plain copy —
        # it must use the threshold AT the closure height, so fill explicitly.
        kk = min(k1, nz - 1)
        for i in range(nx):
            for j in range(ny):
                ti = cell_map[i, j, kk]
                sub = Tz[ti, k1 * N] if continuous else O[i, j, kk]
                Fi[i*N:(i+1)*N, j*N:(j+1)*N, Mz-1] = Gf[ti][:, :, 0] - sub
        Fi[:Mx-1, My-1, :] = Fi[:Mx-1, My-1-N, :]
        Fi[Mx-1, :, :] = Fi[Mx-1-N, :, :]
        if not (F.min() < level < F.max()):
            continue                            # slab entirely solid/void
        v, f, _, _ = marching_cubes(F, level=level)
        v[:, 0] -= ox
        v[:, 1] -= oy
        v[:, 2] += k0 * N - oz
        all_v.append(v.astype(np.float32))
        all_f.append((f + nv).astype(np.int64))
        nv += len(v)

    verts = np.concatenate(all_v)
    faces = np.concatenate(all_f)
    # flatten caps: clamp pad-layer vertices onto the exact bounding planes
    for i, n in enumerate((nx, ny, nz)):
        verts[:, i] = np.clip(verts[:, i], 0.0, n * N)
    verts = (verts / N) @ A.astype(np.float32)
    verts, faces = weld(verts, faces)
    return verts, faces[:, ::-1], info          # outward normals


def weld(verts, faces):
    """Merge coincident vertices and drop degenerate faces. Duplicates at
    slab joints are bitwise identical by construction, so exact matching
    on the float32 bit patterns is safe (no false merges). Pure numpy."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    verts[verts == 0.0] = 0.0                    # normalize -0.0 -> +0.0
    key = verts.view(np.int32)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    verts, faces = verts[first], inv[faces]
    ok = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
          & (faces[:, 2] != faces[:, 0]))
    return verts, faces[ok]


def is_watertight(faces):
    """True if every edge is shared by exactly two faces."""
    e = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def mesh_volume(verts, faces):
    """Signed volume via divergence theorem (positive for outward normals)."""
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    return float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0)


def write_stl(path, verts, faces, chunk=1_000_000):
    """Chunk-streamed binary STL writer (avoids building the buffer in RAM)."""
    dt = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    with open(path, 'wb') as fh:
        fh.write(b'\0' * 80)
        fh.write(np.uint32(len(faces)).tobytes())
        for s in range(0, len(faces), chunk):
            fc = faces[s:s + chunk]
            tri = verts[fc]                       # (m, 3, 3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            rec = np.empty(len(fc), dt)
            rec['n'], rec['v'], rec['attr'] = n, tri, 0
            rec.tofile(fh)


def lattice_mesh(A, f, rho=0.3, tiles=(1, 1, 1), **kw):
    """Single-lattice convenience wrapper; returns a trimesh (small meshes)."""
    verts, faces, cache = stacked_lattice_mesh(A, f, rho=rho, cells=tiles, **kw)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    return mesh, next(iter(cache.values()))


def density_curve(f, ts):
    """Volume fraction vs isovalue (per unit cell)."""
    return np.array([(f >= t).mean() for t in ts])


# ----------------------------------------------------------------------
# Ready-made cubic examples
# ----------------------------------------------------------------------

def cubic_cell(a=1.0):
    return a * np.eye(3)


# --- strut geometry (fractional coordinates), shared with sdf_lattice.py ---

def sc_segments():
    """Simple cubic: struts along the three cell edges."""
    z = [0, 0, 0]
    return [(z, [1, 0, 0]), (z, [0, 1, 0]), (z, [0, 0, 1])]


def bcc_segments():
    """BCC: 8 half-diagonals corner -> body centre."""
    import itertools
    ctr = [0.5, 0.5, 0.5]
    return [(list(p), ctr) for p in itertools.product((0, 1), repeat=3)]


def diamond_segments():
    """Diamond: tetrahedral bonds between the two FCC sublattices."""
    fcc = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    bonds = np.array([[.25, .25, .25], [.25, -.25, -.25],
                      [-.25, .25, -.25], [-.25, -.25, .25]])
    return [(p, p + b) for p in fcc for b in bonds]


def octet_segments():
    """Octet: all <110> nearest-neighbour FCC bonds, deduplicated modulo
    lattice translations. 24 unique struts per cell."""
    import itertools
    basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    nodes = np.array([b + np.array(o) for b in basis
                      for o in itertools.product((-1, 0, 1), repeat=3)])
    segs, d_nn = {}, np.sqrt(0.5)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            p, q = nodes[i], nodes[j]
            if abs(np.linalg.norm(p - q) - d_nn) > 1e-9:
                continue
            d = q - p
            for comp in d:                     # canonical direction sign
                if abs(comp) > 1e-9:
                    if comp < 0:
                        p, q, d = q, p, -d
                    break
            m = 0.5 * (p + q)
            shift = np.floor(m)                # midpoint into [0,1)^3
            key = (tuple(np.round((m - shift) * 4).astype(int)),
                   tuple(np.round(d * 4).astype(int)))
            segs[key] = (p - shift, q - shift)
    assert len(segs) == 24
    return list(segs.values())


def simple_cubic(a=1.0, sigma=0.12, gmax_units=4.5):
    """SC strut lattice: struts along the three cell edges."""
    A = cubic_cell(a)
    segs = [(p, q, 1.0) for p, q in sc_segments()]
    hkl, G = enumerate_G(A, gmax_units * 2 * np.pi / a)
    c = motif_coefficients(A, hkl, G, segments=segs, sigma=sigma * a)
    return A, hkl, c


def bcc_struts(a=1.0, sigma=0.10, gmax_units=4.5):
    """BCC strut lattice: 8 half-diagonals corner -> body centre."""
    A = cubic_cell(a)
    segs = [(p, q, 1.0) for p, q in bcc_segments()]
    hkl, G = enumerate_G(A, gmax_units * 2 * np.pi / a)
    c = motif_coefficients(A, hkl, G, segments=segs, sigma=sigma * a)
    return A, hkl, c


def diamond_struts(a=1.0, sigma=0.08, gmax_units=5.5):
    """Diamond strut lattice: tetrahedral bonds between the two FCC sublattices."""
    A = cubic_cell(a)
    segs = [(p, q, 1.0) for p, q in diamond_segments()]
    hkl, G = enumerate_G(A, gmax_units * 2 * np.pi / a)
    c = motif_coefficients(A, hkl, G, segments=segs, sigma=sigma * a)
    return A, hkl, c


def compensate_nodes(hkl, G, c, nodes, ref_points, sigma, alpha=0.5,
                     sigma_node=None, n_iter=5):
    """
    Reduce the field pile-up where struts meet. Summing strut fields makes
    f much larger at junctions than mid-strut, so the level set inflates
    there (octet: node/strut inscribed-radius ratio ~2.4 uncompensated,
    vs ~1.44 for an ideal sharp cylinder union). This adds negative-weight
    Gaussian 'anti-atoms' at the node positions, calibrated by fixed-point
    iteration so f(node) -> alpha*f(mid-strut) + (1-alpha)*f(node).

    sigma_node : width of the anti-atoms. Must be wider than the strut
                 sigma to carve the whole bulge rather than a small dimple;
                 default 1.7*sigma (calibrated on the octet).
    alpha      : strength. With the default width: 0.5 reaches the sharp-
                 union ratio (~1.5) while keeping the fillet; ~0.6 gives a
                 slightly waisted junction; beyond that the junction pinches
                 off (check connectivity if you push it).
    Returns (c_new, node_weights).
    """
    hkl = np.asarray(hkl, float)
    nodes = np.asarray(nodes, float)
    refs = np.asarray(ref_points, float)
    if sigma_node is None:
        sigma_node = 1.7 * sigma
    ffn = np.exp(-0.5 * sigma_node ** 2 * np.einsum('ij,ij->i', G, G))
    K = ffn.sum()                                 # self-peak of a unit anti-atom
    phases = np.exp(-2j * np.pi * (nodes @ hkl.T))       # (n_nodes, nG)

    def fvals(X, cc):
        return np.real(np.exp(2j * np.pi * (X @ hkl.T)) @ cc)

    c0 = c.copy()
    target = (alpha * fvals(refs, c0).mean()
              + (1.0 - alpha) * fvals(nodes, c0))
    w = np.zeros(len(nodes))
    for _ in range(n_iter):                       # fixed point (node cross-talk)
        cc = c0 + (w @ (phases * ffn))
        w += (target - fvals(nodes, cc)) / K
    return c0 + (w @ (phases * ffn)), w


def octet_struts(a=1.0, sigma=0.06, gmax_units=6.5, node_comp=None,
                 node_sigma=None):
    """
    Octet truss: FCC nodes, struts along all <110> nearest-neighbour bonds
    (octahedral core + corner tetrahedra), 24 unique struts per cell.
    node_comp : optional alpha for compensate_nodes (junction de-bulking).
    """
    A = cubic_cell(a)
    seg_pq = octet_segments()
    segs = [(p, q, 1.0) for p, q in seg_pq]
    hkl, G = enumerate_G(A, gmax_units * 2 * np.pi / a)
    c = motif_coefficients(A, hkl, G, segments=segs, sigma=sigma * a)
    if node_comp is not None:
        basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        mids = np.array([0.5 * (np.asarray(p) + np.asarray(q))
                         for p, q in seg_pq])
        c, _ = compensate_nodes(hkl, G, c, nodes=basis, ref_points=mids,
                                sigma=sigma * a, alpha=node_comp,
                                sigma_node=node_sigma)
    return A, hkl, c


def gyroid(a=1.0):
    """Manual coefficients: sinX cosY + sinY cosZ + sinZ cosX, X = 2*pi*x/a."""
    A = cubic_cell(a)
    stars = [(1, 1, 0), (1, -1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (-1, 0, 1)]
    hkl, c = [], []
    for g in stars:                       # sin(theta) = -i/2 e^{i th} + i/2 e^{-i th}
        hkl += [g, tuple(-i for i in g)]
        c += [-0.25j, +0.25j]
    B = reciprocal_basis(A)
    return A, hkl, np.array(c)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    N, rho = 96, 0.30
    for name, builder in [("sc", simple_cubic), ("bcc", bcc_struts),
                          ("diamond", diamond_struts), ("octet", octet_struts),
                          ("gyroid", gyroid)]:
        A, hkl, c = builder()
        f = field_on_grid(hkl, c, N)
        mesh, t = lattice_mesh(A, f, rho=rho, tiles=(2, 2, 2))
        vf = (f >= t).mean()
        print(f"{name:8s}  nG={len(hkl):4d}  t={t:+.4f}  vol.frac={vf:.3f}  "
              f"watertight={mesh.is_watertight}  "
              f"V_mesh/V_box={mesh.volume/(np.linalg.det(A)*8):.3f}")
        mesh.export(f"/home/claude/{name}.stl")

    # heterogeneous stack: octet shell around a gyroid core, density graded
    # along z. 6x6x6 for the scalability check, 3x3x3 exported for viewing.
    A, hk1, c1 = octet_struts()
    _, hk2, c2 = gyroid()
    Ns = 64
    f_oct, f_gyr = field_on_grid(hk1, c1, Ns), field_on_grid(hk2, c2, Ns)
    for cells, export in [((6, 6, 6), None), ((3, 3, 3), "stack.stl")]:
        n = cells[2]
        cmap = np.zeros(cells, int)
        cmap[1:-1, 1:-1, 1:-1] = 1                   # gyroid core
        rho_map = (0.20 + 0.15 * (np.arange(n) / (n - 1))[None, None, :]
                   * np.ones(cells))
        verts, faces, _ = stacked_lattice_mesh(A, [f_oct, f_gyr], rho=rho_map,
                                               cells=cells, cell_map=cmap)
        vol = mesh_volume(verts, faces) / (np.linalg.det(A) * np.prod(cells))
        print(f"stack {cells}  verts={len(verts)}  "
              f"watertight={is_watertight(faces)}  "
              f"V/V_box={vol:.3f}  target={rho_map.mean():.3f}")
        if export:
            write_stl(f"/home/claude/{export}", verts, faces)

    # continuous density gradation: octet, rho linear 0.15 -> 0.40 along z
    rho_fun = lambda z: 0.15 + (0.40 - 0.15) * z
    cells = (3, 3, 6)
    verts, faces, Tz = stacked_lattice_mesh(A, f_oct, rho=rho_fun, cells=cells)
    vol = mesh_volume(verts, faces) / (np.linalg.det(A) * np.prod(cells))
    print(f"graded {cells}  verts={len(verts)}  "
          f"watertight={is_watertight(faces)}  "
          f"V/V_box={vol:.4f}  target={0.5*(0.15+0.40):.4f}")
    write_stl("/home/claude/octet_graded.stl", verts, faces)
