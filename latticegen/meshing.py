"""
Level set -> triangle mesh for stacks of unit cells.

:func:`mesh_stack` assembles an (nx, ny, nz) block of unit cells -- possibly
of different lattice types and densities -- and extracts the level set with
marching cubes.  The block is processed in z-slabs to bound memory, the
outer faces are closed with flat caps, and the pieces are welded exactly.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
from skimage.measure import marching_cubes

from .field import UnitField
from .mesh import LatticeMesh
from .meshops import weld


def _as_fields(fields):
    if isinstance(fields, UnitField):
        return [fields]
    fields = list(fields)
    if not fields or not all(isinstance(f, UnitField) for f in fields):
        raise TypeError("fields must be a UnitField or a list of UnitField")
    N = fields[0].N
    if any(f.N != N for f in fields):
        raise ValueError("all fields in a stack must have the same grid size N")
    return fields


def _rho_mode(rho):
    """Classify the density specification."""
    if callable(rho):
        try:
            n_par = len([p for p in inspect.signature(rho).parameters.values()
                         if p.default is inspect.Parameter.empty
                         and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        except (TypeError, ValueError):
            n_par = 1
        if n_par == 3:
            return "xyz"
        if n_par == 1:
            return "z"
        raise ValueError("rho callable must take 1 argument (zeta) or 3 (x, y, z)")
    return "cells"


def mesh_stack(fields, rho=0.3, cells=(1, 1, 1), cell_map=None, slab_cells=1,
               exact_density=True, threshold=None, verbose=False) -> LatticeMesh:
    """Mesh an (nx, ny, nz) block of unit cells.

    fields   : a UnitField, or a list of UnitFields (same N) -- the lattice
               types available to ``cell_map``.
    rho      : target volume fraction.  One of
               * a scalar,
               * an (nx, ny, nz) array (per-cell density),
               * ``rho(zeta)``: callable of the fractional stack height
                 zeta = z / (nz*|a3|) in [0, 1] -- continuous z gradation,
               * ``rho(x, y, z)``: callable of CARTESIAN coordinates (cell
                 length units, block origin at 0) -- arbitrary 3-D gradation.
               Callables must be numpy-vectorised and elementwise (the same
               input value must always give the same output, otherwise slab
               joints will not weld).  Values must stay inside (0, 1).
    cell_map : (nx, ny, nz) int array selecting the field index of each cell.
    slab_cells : cells per z-slab processed at a time (memory control);
               peak memory ~ (nx*N)(ny*N)(slab_cells*N) x 4 bytes.
    exact_density : for scalar / per-cell rho, calibrate each isovalue on
               the meshed unit-cell volume (see UnitField.calibrated_threshold)
               instead of the voxel quantile.  Callable rho always uses the
               quantile mapping.
    threshold : advanced -- give the isovalue(s) directly (scalar or
               (nx, ny, nz) array) instead of rho.

    Returns a closed, welded :class:`LatticeMesh` with outward normals, in
    the cartesian units of the cell.  ``mesh.info`` holds the thresholds.
    """
    fields = _as_fields(fields)
    N = fields[0].N
    A = fields[0].A.astype(np.float32)
    nx, ny, nz = (int(c) for c in cells)
    if min(nx, ny, nz) < 1:
        raise ValueError("cells must be >= 1 along every axis")
    cell_map = np.zeros((nx, ny, nz), int) if cell_map is None else np.asarray(cell_map, int)
    if cell_map.shape != (nx, ny, nz):
        raise ValueError(f"cell_map shape {cell_map.shape} != cells {(nx, ny, nz)}")
    if cell_map.min() < 0 or cell_map.max() >= len(fields):
        raise ValueError("cell_map indexes outside the field list")

    Gf = [f.f for f in fields]                                  # float32 (N,N,N)
    Gp = [np.pad(g, ((0, 1), (0, 1), (0, 1)), mode="wrap") for g in Gf]   # (N+1)^3
    lo = min(float(g.min()) for g in Gf)
    hi = max(float(g.max()) for g in Gf)
    scale = hi - lo
    level = 1e-6 * scale                                        # nudge off grid values
    pad_val = np.float32(-2.0 * scale)                          # always below level

    mode = "cells" if threshold is not None else _rho_mode(rho)
    if threshold is not None:
        O = np.array(np.broadcast_to(np.asarray(threshold, float), (nx, ny, nz)))
        info = {"mode": "cells", "thresholds": O, "cache": {}}
    elif mode == "cells":
        rho_map = np.broadcast_to(np.asarray(rho, float), (nx, ny, nz))
        if rho_map.min() <= 0.0 or rho_map.max() >= 1.0:
            raise ValueError("rho must lie strictly inside (0, 1)")
        keys = {(int(cell_map[idx]), round(float(rho_map[idx]), 9)) for idx in np.ndindex(nx, ny, nz)}
        calibrate = exact_density and len(keys) <= 64
        cache, O = {}, np.empty((nx, ny, nz))
        for idx in np.ndindex(nx, ny, nz):
            key = (int(cell_map[idx]), round(float(rho_map[idx]), 9))
            if key not in cache:
                fld = fields[key[0]]
                cache[key] = (fld.calibrated_threshold(key[1]) if calibrate
                              else float(fld.threshold(key[1])))
            O[idx] = cache[key]
        info = {"mode": "cells", "thresholds": O, "cache": cache, "calibrated": calibrate}
    else:
        info = {"mode": mode}

    def threshold_block(ti, ix, iy, iz):
        """Threshold values at global index coordinates (broadcast grids)."""
        if mode == "cells":
            i = min(int(ix.flat[0]) // N, nx - 1)
            j = min(int(iy.flat[0]) // N, ny - 1)
            k = min(int(iz.flat[0]) // N, nz - 1)
            return O[i, j, k]
        if mode == "z":
            r = np.asarray(rho(iz / (nz * N)), float)
        else:
            X = (ix / N) * A[0, 0] + (iy / N) * A[1, 0] + (iz / N) * A[2, 0]
            Y = (ix / N) * A[0, 1] + (iy / N) * A[1, 1] + (iz / N) * A[2, 1]
            Z = (ix / N) * A[0, 2] + (iy / N) * A[1, 2] + (iz / N) * A[2, 2]
            r = np.asarray(rho(X, Y, Z), float)
        r = np.broadcast_to(r, np.broadcast_shapes(ix.shape, iy.shape, iz.shape))
        if r.min() <= 0.0 or r.max() >= 1.0:
            raise ValueError("rho(...) must stay strictly inside (0, 1)")
        return fields[ti].threshold(r)

    def grids(i0, i1, j0, j1, k0, k1):
        ix = np.arange(i0, i1, dtype=np.float64)[:, None, None]
        iy = np.arange(j0, j1, dtype=np.float64)[None, :, None]
        iz = np.arange(k0, k1, dtype=np.float64)[None, None, :]
        return ix, iy, iz

    all_v, all_f, nv = [], [], 0
    for k0 in range(0, nz, slab_cells):
        k1 = min(k0 + slab_cells, nz)
        zpad = (1 if k0 == 0 else 0, 1 if k1 == nz else 0)
        Mx, My, Mz = nx * N + 1, ny * N + 1, (k1 - k0) * N + 1
        F = np.full((Mx + 2, My + 2, Mz + sum(zpad)), pad_val, np.float32)
        ox, oy, oz = 1, 1, zpad[0]
        Fi = F[ox:ox + Mx, oy:oy + My, oz:oz + Mz]              # interior view
        for i in range(nx):
            ex = 1 if i == nx - 1 else 0
            for j in range(ny):
                ey = 1 if j == ny - 1 else 0
                for k in range(k0, k1):
                    ti = int(cell_map[i, j, k])
                    ix, iy, iz = grids(i * N, (i + 1) * N + ex, j * N, (j + 1) * N + ey,
                                       k * N, (k + 1) * N)
                    blk = Gp[ti][:N + ex, :N + ey, :N] - threshold_block(ti, ix, iy, iz)
                    Fi[i * N:(i + 1) * N + ex, j * N:(j + 1) * N + ey,
                       (k - k0) * N:(k - k0 + 1) * N] = blk
                # top sample layer of the slab: at an internal joint it belongs
                # to the NEXT cell layer (bitwise identical to that slab's
                # bottom layer -> welds); at the far face it is the periodic
                # closure of the last cell.
                kk = min(k1, nz - 1)
                ti = int(cell_map[i, j, kk])
                ix, iy, iz = grids(i * N, (i + 1) * N + ex, j * N, (j + 1) * N + ey,
                                   k1 * N, k1 * N + 1)
                blk = Gp[ti][:N + ex, :N + ey, :1] - threshold_block(ti, ix, iy, iz)
                Fi[i * N:(i + 1) * N + ex, j * N:(j + 1) * N + ey, Mz - 1:Mz] = blk
        if not (F.min() < level < F.max()):
            continue                                            # slab all solid / all void
        v, f, _, _ = marching_cubes(F, level=level)
        v[:, 0] -= ox
        v[:, 1] -= oy
        v[:, 2] += k0 * N - oz
        all_v.append(v.astype(np.float32))
        all_f.append((f + nv).astype(np.int64))
        nv += len(v)
        if verbose:
            print(f"  slab z-cells {k0}..{k1 - 1}: {len(v)} vertices")
        del F, Fi

    if not all_v:
        raise RuntimeError("empty level set: the stack is entirely solid or entirely void")
    verts = np.concatenate(all_v)
    faces = np.concatenate(all_f)
    for ax, n in enumerate((nx, ny, nz)):                       # flatten caps
        verts[:, ax] = np.clip(verts[:, ax], 0.0, n * N)
    verts = (verts / N) @ A
    verts, faces = weld(verts, faces)
    mesh = LatticeMesh(verts, faces[:, ::-1], info=info)        # outward normals
    mesh.info.update(cells=(nx, ny, nz), N=N, A=fields[0].A.copy(),
                     box_volume=fields[0].cell.volume * nx * ny * nz)
    _warn_if_disconnected(mesh)
    return mesh


def _warn_if_disconnected(mesh):
    n = mesh.components()[1]
    if n > 1:
        warnings.warn(f"the mesh has {n} disconnected components (floating pieces or "
                      f"incompatible cell interfaces); see mesh.remove_islands() and "
                      f"UnitField.connectivity_check()", RuntimeWarning, stacklevel=3)
