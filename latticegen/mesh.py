"""
:class:`LatticeMesh` -- an indexed triangle mesh with lattice-oriented
helpers: diagnostics, simplification / remeshing, repair and export.
"""

from __future__ import annotations

import os

import numpy as np

from . import meshops


class LatticeMesh:
    """Indexed triangle mesh (float32 vertices, int64 faces, outward normals).

    All coordinates are in the cartesian units of the unit cell that
    produced the mesh (use mm for DLP printing).
    """

    def __init__(self, verts, faces, info=None):
        self.verts = np.ascontiguousarray(verts, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        if self.verts.ndim != 2 or self.verts.shape[1] != 3:
            raise ValueError("verts must be (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be (m, 3)")
        self.info = dict(info or {})

    # ------------------------------------------------------------------ #
    # basic properties
    # ------------------------------------------------------------------ #
    @property
    def n_vertices(self) -> int:
        return len(self.verts)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) array: [min_xyz, max_xyz]."""
        return np.stack([self.verts.min(axis=0), self.verts.max(axis=0)]).astype(float)

    @property
    def extents(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    @property
    def volume(self) -> float:
        return meshops.mesh_volume(self.verts, self.faces)

    @property
    def area(self) -> float:
        return meshops.mesh_area(self.verts, self.faces)

    @property
    def is_watertight(self) -> bool:
        return meshops.is_watertight(self.faces)

    @property
    def euler_number(self) -> int:
        e = meshops.n_unique_edges(self.faces)
        return int(self.n_vertices - e + self.n_faces)

    @property
    def relative_density(self):
        """Mesh volume / bounding-block volume (needs ``info['box_volume']``)."""
        vb = self.info.get("box_volume")
        return None if vb is None else self.volume / vb

    def edge_lengths(self) -> np.ndarray:
        e = meshops.edges_of(self.faces)
        return np.linalg.norm(self.verts[e[:, 0]] - self.verts[e[:, 1]], axis=1)

    def components(self):
        """(labels per face, number of components)."""
        return meshops.connected_components(self.faces, self.n_vertices)

    def copy(self) -> "LatticeMesh":
        return LatticeMesh(self.verts.copy(), self.faces.copy(), dict(self.info))

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def check(self, chunk=1_000_000) -> dict:
        """Watertightness, orientation, components, volume, edge statistics.

        Streams over face chunks and sorts the edge list once (watertightness,
        Euler number and components all come from it), so a 5 M-face mesh
        needs a few hundred MB of scratch memory instead of ~1 GB."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        v, f = self.verts, self.faces
        nv, nf = self.n_vertices, self.n_faces
        keys = np.empty(3 * nf, np.int64)
        el = np.empty(3 * nf, np.float32)
        volume = area = 0.0
        for s in range(0, nf, chunk):
            fc = f[s:s + chunk]
            e = meshops.edges_of(fc)
            keys[3 * s:3 * s + 3 * len(fc)] = e[:, 0] * nv + e[:, 1]
            el[3 * s:3 * s + 3 * len(fc)] = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
            del e
            v0, v1, v2 = (v[fc[:, i]].astype(np.float64) for i in range(3))
            volume += float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
            area += float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
            del v0, v1, v2
        uk, cnt = np.unique(keys, return_counts=True)
        del keys
        m = coo_matrix((np.ones(len(uk), np.float32),
                        ((uk // nv).astype(np.int32), (uk % nv).astype(np.int32))), shape=(nv, nv))
        _, vlab = connected_components(m, directed=False)
        del m
        n_comp = len(np.unique(vlab[f[:, 0]])) if nf else 0
        out = dict(n_vertices=nv, n_faces=nf,
                   watertight=bool(np.all(cnt == 2)), volume=volume,
                   area=area, components=n_comp,
                   extents=tuple(float(x) for x in self.extents),
                   edge_min=float(el.min()) if len(el) else 0.0,
                   edge_median=float(np.median(el)) if len(el) else 0.0,
                   edge_max=float(el.max()) if len(el) else 0.0,
                   euler=int(nv - len(uk) + nf))
        vb = self.info.get("box_volume")
        if vb is not None:
            out["relative_density"] = volume / vb
        out["outward"] = volume > 0
        out["stl_size_MB"] = (84 + 50 * self.n_faces) / 1e6
        return out

    def report(self, check=None) -> str:
        """Readable summary; pass ``check`` (a :meth:`check` result) to reuse it."""
        c = self.check() if check is None else check
        lines = [f"vertices {c['n_vertices']:,}   faces {c['n_faces']:,}   "
                 f"(binary STL ~{c['stl_size_MB']:.1f} MB)",
                 f"watertight {c['watertight']}   outward normals {c['outward']}   "
                 f"components {c['components']}   Euler {c['euler']}",
                 f"extents {c['extents'][0]:.4g} x {c['extents'][1]:.4g} x {c['extents'][2]:.4g}",
                 f"volume {c['volume']:.6g}   area {c['area']:.6g}"]
        if "relative_density" in c:
            lines[-1] += f"   relative density {c['relative_density']:.4f}"
        lines.append(f"edge length min/median/max {c['edge_min']:.4g} / "
                     f"{c['edge_median']:.4g} / {c['edge_max']:.4g}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # geometry operations (return new meshes)
    # ------------------------------------------------------------------ #
    def scaled(self, factor) -> "LatticeMesh":
        m = LatticeMesh(self.verts * np.float32(factor), self.faces, self.info)
        if "box_volume" in m.info:
            m.info["box_volume"] *= float(np.prod(np.broadcast_to(factor, 3)))
        return m

    def translated(self, offset) -> "LatticeMesh":
        return LatticeMesh(self.verts + np.asarray(offset, np.float32), self.faces, self.info)

    def flipped(self) -> "LatticeMesh":
        """Reverse face orientation."""
        return LatticeMesh(self.verts, self.faces[:, ::-1], self.info)

    def weld(self) -> "LatticeMesh":
        v, f = meshops.weld(self.verts, self.faces)
        return LatticeMesh(v, f, self.info)

    # simplification / remeshing / repair -- implemented in remesh.py
    def decimate(self, *a, **kw) -> "LatticeMesh":
        from .remesh import decimate
        return decimate(self, *a, **kw)

    def simplify(self, *a, **kw) -> "LatticeMesh":
        from .remesh import simplify
        return simplify(self, *a, **kw)

    def remesh(self, *a, **kw) -> "LatticeMesh":
        from .remesh import remesh_isotropic
        return remesh_isotropic(self, *a, **kw)

    def repair(self, *a, **kw) -> "LatticeMesh":
        from .remesh import repair
        return repair(self, *a, **kw)

    def remove_islands(self, *a, **kw) -> "LatticeMesh":
        from .remesh import remove_islands
        return remove_islands(self, *a, **kw)

    # ------------------------------------------------------------------ #
    # conversion / IO
    # ------------------------------------------------------------------ #
    def to_trimesh(self):
        import trimesh
        return trimesh.Trimesh(self.verts.astype(np.float64), self.faces, process=False)

    @classmethod
    def from_trimesh(cls, tm, info=None) -> "LatticeMesh":
        return cls(np.asarray(tm.vertices), np.asarray(tm.faces), info)

    @classmethod
    def load(cls, path) -> "LatticeMesh":
        if str(path).lower().endswith(".stl"):
            try:
                v, f = meshops.read_stl(path)
                return cls(v, f)
            except Exception:
                pass                                  # ASCII STL etc. -> trimesh
        import trimesh
        return cls.from_trimesh(trimesh.load(path, force="mesh"))

    def export(self, path, **kw) -> str:
        """Write the mesh.  ``.stl`` uses the streamed binary writer (any
        size); other formats (.obj, .ply, .3mf, .off, .glb) go through trimesh."""
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".stl":
            meshops.write_stl(path, self.verts, self.faces, **kw)
        else:
            self.to_trimesh().export(path, **kw)
        return path

    def __repr__(self) -> str:
        return (f"LatticeMesh({self.n_vertices:,} vertices, {self.n_faces:,} faces, "
                f"extents {np.round(self.extents, 4).tolist()})")
