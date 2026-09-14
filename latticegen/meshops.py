"""Low-level, pure-numpy mesh utilities shared by the meshers."""

from __future__ import annotations

import numpy as np


def weld(verts, faces):
    """Merge bitwise-identical vertices and drop degenerate faces.

    Duplicates produced at slab joints / instance boundaries are bitwise
    identical by construction, so exact matching on the float32 bit
    patterns is safe (no false merges).
    """
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    verts[verts == 0.0] = 0.0                    # normalise -0.0 -> +0.0
    key = verts.view(np.int32)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    verts, faces = verts[first], inv[np.asarray(faces)]
    ok = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
          & (faces[:, 2] != faces[:, 0]))
    return verts, faces[ok].astype(np.int64)


def edges_of(faces):
    return np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)


def _edge_keys(faces):
    e = edges_of(faces).astype(np.int64)
    n = int(faces.max()) + 1 if len(faces) else 1
    return e[:, 0] * n + e[:, 1], n


def edge_counts(faces):
    """Unique undirected edges (k,2) and the number of faces sharing each."""
    keys, n = _edge_keys(faces)
    uk, cnt = np.unique(keys, return_counts=True)
    return np.stack([uk // n, uk % n], axis=1), cnt


def n_unique_edges(faces) -> int:
    keys, _ = _edge_keys(faces)
    return len(np.unique(keys))


def is_watertight(faces) -> bool:
    """True if every edge is shared by exactly two faces."""
    _, counts = edge_counts(faces)
    return bool(np.all(counts == 2))


def open_edges(faces):
    eu, cnt = edge_counts(faces)
    return eu[cnt == 1]


def mesh_volume(verts, faces) -> float:
    """Signed volume via the divergence theorem (positive for outward normals)."""
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def mesh_area(verts, faces) -> float:
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())


def face_normals(verts, faces):
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def connected_components(faces, n_verts=None):
    """Label faces by vertex-connected component.  Returns (labels, n)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc
    n = int(faces.max()) + 1 if n_verts is None else n_verts
    e = edges_of(faces)
    m = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    k, vlab = cc(m, directed=False)
    flab = vlab[faces[:, 0]]
    # relabel to consecutive ids of face components only
    _, flab = np.unique(flab, return_inverse=True)
    return flab.reshape(-1), int(flab.max()) + 1 if len(flab) else 0


def write_stl(path, verts, faces, chunk=1_000_000, header=b""):
    """Chunk-streamed binary STL writer (does not build the buffer in RAM)."""
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    faces = np.asarray(faces)
    with open(path, "wb") as fh:
        fh.write(header[:80].ljust(80, b"\0"))
        fh.write(np.uint32(len(faces)).tobytes())
        for s in range(0, len(faces), chunk):
            fc = faces[s:s + chunk]
            tri = np.asarray(verts, np.float32)[fc]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float64)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            rec = np.empty(len(fc), dt)
            rec["n"], rec["v"], rec["attr"] = n, tri, 0
            rec.tofile(fh)


def read_stl(path):
    """Read a binary STL into welded (verts, faces)."""
    with open(path, "rb") as fh:
        fh.seek(80)
        n = int(np.frombuffer(fh.read(4), "<u4")[0])
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        rec = np.fromfile(fh, dt, count=n)
    verts = rec["v"].reshape(-1, 3)
    faces = np.arange(3 * n).reshape(-1, 3)
    return weld(verts, faces)
