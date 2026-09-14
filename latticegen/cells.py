"""
Bravais unit cells.

A unit cell is described by a 3x3 matrix ``A`` whose ROWS are the lattice
vectors a1, a2, a3 (cartesian, in the length unit you print in -- use mm
for DLP).  Fractional coordinates ``u`` map to cartesian ``r = u @ A``.
"""

from __future__ import annotations

import numpy as np


class UnitCell:
    """A Bravais unit cell (lattice vectors as rows of ``A``).

    Construct with one of the classmethods, e.g. ``UnitCell.cubic(2.0)`` for
    a 2 mm cubic cell, or pass any 3x3 matrix directly.
    """

    def __init__(self, A, name: str | None = None):
        A = np.array(A, dtype=float)
        if A.shape != (3, 3):
            raise ValueError("A must be a 3x3 matrix with lattice vectors as rows")
        if abs(np.linalg.det(A)) < 1e-12:
            raise ValueError("lattice vectors are linearly dependent")
        self.A = A
        self.name = name or "custom"

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def cubic(cls, a: float = 1.0) -> "UnitCell":
        return cls(a * np.eye(3), name=f"cubic(a={a:g})")

    @classmethod
    def tetragonal(cls, a: float, c: float) -> "UnitCell":
        return cls(np.diag([a, a, c]), name=f"tetragonal(a={a:g}, c={c:g})")

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float) -> "UnitCell":
        return cls(np.diag([a, b, c]), name=f"orthorhombic({a:g}, {b:g}, {c:g})")

    @classmethod
    def hexagonal(cls, a: float, c: float) -> "UnitCell":
        return cls.from_parameters(a, a, c, 90.0, 90.0, 120.0)

    @classmethod
    def from_parameters(cls, a, b, c, alpha=90.0, beta=90.0, gamma=90.0) -> "UnitCell":
        """Crystallographic cell parameters (angles in degrees).
        a1 along x, a2 in the xy plane."""
        al, be, ga = np.radians([alpha, beta, gamma])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        A = np.array([[a, 0.0, 0.0],
                      [b * np.cos(ga), b * np.sin(ga), 0.0],
                      [cx, cy, cz]])
        return cls(A, name=f"cell({a:g},{b:g},{c:g},{alpha:g},{beta:g},{gamma:g})")

    @classmethod
    def from_matrix(cls, A, name=None) -> "UnitCell":
        return cls(A, name=name)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.A)))

    @property
    def lengths(self) -> np.ndarray:
        """|a1|, |a2|, |a3|."""
        return np.linalg.norm(self.A, axis=1)

    @property
    def a_min(self) -> float:
        return float(self.lengths.min())

    @property
    def is_cubic(self) -> bool:
        L = self.lengths
        return bool(np.allclose(self.A, L[0] * np.eye(3)))

    def reciprocal(self) -> np.ndarray:
        """Reciprocal basis B (rows b1,b2,b3) with b_i . a_j = 2*pi*delta_ij."""
        return 2.0 * np.pi * np.linalg.inv(self.A).T

    def frac_to_cart(self, u) -> np.ndarray:
        return np.asarray(u, float) @ self.A

    def cart_to_frac(self, r) -> np.ndarray:
        return np.asarray(r, float) @ np.linalg.inv(self.A)

    def scaled(self, factor: float) -> "UnitCell":
        """Same shape, all lengths multiplied by ``factor``."""
        return UnitCell(self.A * factor, name=f"{self.name}*{factor:g}")

    def __repr__(self) -> str:
        L = self.lengths
        return f"UnitCell({self.name}; |a|=({L[0]:.4g}, {L[1]:.4g}, {L[2]:.4g}), V={self.volume:.4g})"


def as_cell(cell) -> UnitCell:
    """Accept a UnitCell, a 3x3 matrix or a scalar cubic edge length."""
    if isinstance(cell, UnitCell):
        return cell
    if np.isscalar(cell):
        return UnitCell.cubic(float(cell))
    return UnitCell(cell)
