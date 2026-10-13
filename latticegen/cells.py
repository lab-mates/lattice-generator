"""
Bravais unit cells.

A unit cell is described by a 3x3 matrix ``A`` whose ROWS are the lattice
vectors a1, a2, a3 (cartesian, in the length unit you print in -- use mm
for DLP).  Fractional coordinates ``u`` map to cartesian ``r = u @ A``.
"""

from __future__ import annotations

import numpy as np


class UnitCell:
    """A Bravais unit cell, stored as the matrix ``A`` whose rows are the lattice vectors.

    Fractional coordinates ``u`` map to cartesian ``r = u @ A``. All lengths
    are in the unit you print in; use millimetres for DLP.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Lattice vectors a1, a2, a3 as rows. They must be linearly independent.
    name : str, optional
        Label used in ``repr`` and file names.

    Attributes
    ----------
    A : ndarray, shape (3, 3)
        The lattice vectors as rows.
    name : str
        Label of the cell.

    Examples
    --------
    >>> import latticegen as lg
    >>> lg.UnitCell.cubic(2.0)
    UnitCell(cubic(a=2); |a|=(2, 2, 2), V=8)
    >>> lg.UnitCell.from_parameters(1, 1, 1.5, 90, 90, 120).volume   # doctest: +ELLIPSIS
    1.299...
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
        """Cubic cell of edge ``a``.

        Parameters
        ----------
        a : float
            Edge length (mm).

        Returns
        -------
        UnitCell
        """
        return cls(a * np.eye(3), name=f"cubic(a={a:g})")

    @classmethod
    def tetragonal(cls, a: float, c: float) -> "UnitCell":
        """Tetragonal cell: square base of side ``a``, height ``c`` along z.

        Parameters
        ----------
        a, c : float
            Base edge and height (mm).

        Returns
        -------
        UnitCell
        """
        return cls(np.diag([a, a, c]), name=f"tetragonal(a={a:g}, c={c:g})")

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float) -> "UnitCell":
        """Rectangular-box cell with edges ``a``, ``b``, ``c`` along x, y, z.

        Parameters
        ----------
        a, b, c : float
            Edge lengths (mm).

        Returns
        -------
        UnitCell
        """
        return cls(np.diag([a, b, c]), name=f"orthorhombic({a:g}, {b:g}, {c:g})")

    @classmethod
    def hexagonal(cls, a: float, c: float) -> "UnitCell":
        """Hexagonal cell: ``a = b``, ``gamma = 120`` degrees, height ``c``.

        Parameters
        ----------
        a, c : float
            Base edge and height (mm).

        Returns
        -------
        UnitCell
        """
        return cls.from_parameters(a, a, c, 90.0, 90.0, 120.0)

    @classmethod
    def from_parameters(cls, a, b, c, alpha=90.0, beta=90.0, gamma=90.0) -> "UnitCell":
        """Cell from the six crystallographic parameters.

        a1 lies along x and a2 in the xy plane.

        Parameters
        ----------
        a, b, c : float
            Lengths of a1, a2, a3 (mm).
        alpha, beta, gamma : float, optional
            Angles in degrees: alpha between a2 and a3, beta between a1 and a3,
            gamma between a1 and a2. Default 90.

        Returns
        -------
        UnitCell
        """
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
        """Same as ``UnitCell(A, name)``.

        Parameters
        ----------
        A : array_like, shape (3, 3)
            Lattice vectors as rows.
        name : str, optional
            Label of the cell.

        Returns
        -------
        UnitCell
        """
        return cls(A, name=name)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    @property
    def volume(self) -> float:
        """Cell volume ``|det A|``.
        """
        return float(abs(np.linalg.det(self.A)))

    @property
    def lengths(self) -> np.ndarray:
        """Edge lengths ``|a1|, |a2|, |a3|`` as an array of 3.
        """
        return np.linalg.norm(self.A, axis=1)

    @property
    def a_min(self) -> float:
        """Shortest edge length.
        """
        return float(self.lengths.min())

    @property
    def is_cubic(self) -> bool:
        """True if ``A`` is a multiple of the identity.
        """
        L = self.lengths
        return bool(np.allclose(self.A, L[0] * np.eye(3)))

    def reciprocal(self) -> np.ndarray:
        """Reciprocal basis.

        Returns
        -------
        ndarray, shape (3, 3)
            ``B`` with rows b1, b2, b3 such that ``b_i . a_j = 2 pi delta_ij``,
            i.e. ``B = 2 pi (A^-1)^T``.
        """
        return 2.0 * np.pi * np.linalg.inv(self.A).T

    def frac_to_cart(self, u) -> np.ndarray:
        """Fractional to cartesian coordinates.

        Parameters
        ----------
        u : array_like, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3)
            ``u @ A``.
        """
        return np.asarray(u, float) @ self.A

    def cart_to_frac(self, r) -> np.ndarray:
        """Cartesian to fractional coordinates.

        Parameters
        ----------
        r : array_like, shape (..., 3)

        Returns
        -------
        ndarray, shape (..., 3)
            ``r @ A^-1``.
        """
        return np.asarray(r, float) @ np.linalg.inv(self.A)

    def scaled(self, factor: float) -> "UnitCell":
        """Same shape, every length multiplied by ``factor``.

        Parameters
        ----------
        factor : float

        Returns
        -------
        UnitCell
        """
        return UnitCell(self.A * factor, name=f"{self.name}*{factor:g}")

    def __repr__(self) -> str:
        L = self.lengths
        return f"UnitCell({self.name}; |a|=({L[0]:.4g}, {L[1]:.4g}, {L[2]:.4g}), V={self.volume:.4g})"


def as_cell(cell) -> UnitCell:
    """Convert the accepted cell inputs to a :class:`UnitCell`.

    Parameters
    ----------
    cell : UnitCell, float or array_like
        A cell (returned as is), a cubic edge length, or a 3x3 matrix.

    Returns
    -------
    UnitCell
    """
    if isinstance(cell, UnitCell):
        return cell
    if np.isscalar(cell):
        return UnitCell.cubic(float(cell))
    return UnitCell(cell)
