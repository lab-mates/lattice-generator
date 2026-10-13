"""
Triply periodic minimal surface (TPMS) approximants.

Each surface is given by the usual low-order trigonometric level-set
approximation ``F(X, Y, Z)`` with ``X = 2*pi*x/a`` etc.  Two solid types:

* **network** (default): solid where ``F >= t``.  At rho = 0.5 the surface
  divides space into two congruent labyrinths.
* **sheet**: solid where ``|F| <= t``, i.e. a thickened wall around the
  F = 0 surface.  Both labyrinths stay open (good drainage for DLP).

Sheet fields are handed downstream as ``f = -|F|`` so the common
"solid = f >= t" convention holds.
"""

from __future__ import annotations

import numpy as np

_pi2 = 2.0 * np.pi


def _gyroid(X, Y, Z):
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def _primitive(X, Y, Z):                       # Schwarz P
    return np.cos(X) + np.cos(Y) + np.cos(Z)


def _diamond(X, Y, Z):                         # Schwarz D
    return (np.sin(X) * np.sin(Y) * np.sin(Z) + np.sin(X) * np.cos(Y) * np.cos(Z)
            + np.cos(X) * np.sin(Y) * np.cos(Z) + np.cos(X) * np.cos(Y) * np.sin(Z))


def _iwp(X, Y, Z):                             # Schoen I-WP
    return (2 * (np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X))
            - (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z)))


def _neovius(X, Y, Z):
    return 3 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4 * np.cos(X) * np.cos(Y) * np.cos(Z)


def _lidinoid(X, Y, Z):
    return (0.5 * (np.sin(2 * X) * np.cos(Y) * np.sin(Z) + np.sin(2 * Y) * np.cos(Z) * np.sin(X)
                   + np.sin(2 * Z) * np.cos(X) * np.sin(Y))
            - 0.5 * (np.cos(2 * X) * np.cos(2 * Y) + np.cos(2 * Y) * np.cos(2 * Z)
                     + np.cos(2 * Z) * np.cos(2 * X)) + 0.15)


def _fischer_koch_s(X, Y, Z):
    return (np.cos(2 * X) * np.sin(Y) * np.cos(Z) + np.cos(X) * np.cos(2 * Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.cos(2 * Z))


def _split_p(X, Y, Z):
    return (1.1 * (np.sin(2 * X) * np.sin(Z) * np.cos(Y) + np.sin(2 * Y) * np.sin(X) * np.cos(Z)
                   + np.sin(2 * Z) * np.sin(Y) * np.cos(X))
            - 0.2 * (np.cos(2 * X) * np.cos(2 * Y) + np.cos(2 * Y) * np.cos(2 * Z)
                     + np.cos(2 * Z) * np.cos(2 * X))
            - 0.4 * (np.cos(2 * X) + np.cos(2 * Y) + np.cos(2 * Z)))


SURFACES = {
    "gyroid": _gyroid,
    "primitive": _primitive, "schwarz_p": _primitive, "p": _primitive,
    "diamond": _diamond, "schwarz_d": _diamond, "d": _diamond,
    "iwp": _iwp,
    "neovius": _neovius,
    "lidinoid": _lidinoid,
    "fischer_koch_s": _fischer_koch_s, "fks": _fischer_koch_s,
    "split_p": _split_p,
}


def available() -> list:
    """Names of the built-in triply periodic minimal surfaces.

    Returns
    -------
    list of str
        ``['gyroid', 'primitive', 'diamond', 'iwp', 'neovius', 'lidinoid',
        'fischer_koch_s', 'split_p']``. Aliases such as ``schwarz_p`` are also
        accepted by :func:`tpms_function`.
    """
    return ["gyroid", "primitive", "diamond", "iwp", "neovius", "lidinoid",
            "fischer_koch_s", "split_p"]


def tpms_function(name):
    """Level-set function ``F(X, Y, Z)`` of a surface, with ``X = 2 pi x / a``.

    Parameters
    ----------
    name : str
        Surface name or alias, case-insensitive.

    Returns
    -------
    callable
        Vectorised ``F(X, Y, Z)``; the surface is ``F = 0``.

    Raises
    ------
    KeyError
        If the name is unknown.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in SURFACES:
        raise KeyError(f"unknown TPMS {name!r}; available: {', '.join(available())}")
    return SURFACES[key]


def tpms_on_grid(name: str, N: int, periods=(1, 1, 1), sheet: bool = False,
                 phase=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Sample a TPMS level-set function on the N^3 fractional grid of one cell.

    Parameters
    ----------
    name : str
        Surface name (see :func:`available`).
    N : int
        Samples per cell edge.
    periods : tuple of 3 int, optional
        Repeats of the surface along each cell vector.
    sheet : bool, optional
        Return ``-|F|`` (sheet solid around the surface) instead of ``F``
        (network solid).
    phase : tuple of 3 float, optional
        Shift of the surface, in fractions of the period.

    Returns
    -------
    ndarray, shape (N, N, N)
    """
    fn = tpms_function(name)
    u = np.arange(N) / N
    X, Y, Z = np.meshgrid(*[_pi2 * (u * p + ph) for p, ph in zip(periods, phase)], indexing="ij")
    F = fn(X, Y, Z)
    return -np.abs(F) if sheet else F
