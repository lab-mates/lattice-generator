"""
One-call lattice build shared by the command line and the browser GUI::

    from latticegen.app import build
    res = build("octet", method="struts", a=1.0, rho=0.2, cells=(20, 20, 20), check=True)
    res.mesh.export(res.filename)

``build`` takes the same options as ``python -m latticegen build`` and sends
its progress lines to ``log`` (``print`` by default).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as _dc_field

METHODS = ("auto", "sdf", "fourier", "tpms", "struts")


@dataclass
class BuildResult:
    """Result of :func:`build`.

    Attributes
    ----------
    mesh : LatticeMesh
    field : UnitField or None
        None for ``method='struts'``.
    method : str
        The method actually used (``'auto'`` resolved).
    N : int
        Field grid per cell; 0 for ``'struts'``.
    filename : str
        Suggested output name, e.g. ``octet_3x3x3.stl``.
    report : str
        :meth:`LatticeMesh.report` text.
    dlp : str
        :func:`format_report` text; empty unless ``check=True``.
    timings : dict
        Seconds after start: ``field``, ``mesh``, ``total``.
    """
    mesh: object                    # LatticeMesh
    field: object = None            # UnitField (None for method 'struts')
    method: str = ""
    N: int = 0                      # field grid per cell (0 for 'struts')
    filename: str = ""              # default output name, e.g. octet_3x3x3.stl
    report: str = ""                # mesh.report()
    dlp: str = ""                   # formatted DLP report ('' unless check=True)
    timings: dict = _dc_field(default_factory=dict)


def default_grid(cell, pixel, instanced=False) -> int:
    """Automatic grid size used by the CLI and both GUIs.

    Half a printer pixel per voxel with ``N >= 80`` for blocks. With
    instancing only one cell is meshed and a fine cell remeshes better, so a
    quarter pixel.

    Parameters
    ----------
    cell : UnitCell
    pixel : float
        Printer pixel (mm).
    instanced : bool, optional

    Returns
    -------
    int
    """
    from .printing import recommend_grid
    return recommend_grid(cell, pixel, oversample=4.0 if instanced else 2.0, minimum=80)


def build(topology, method="auto", a=1.0, cell=None, N=0, rho=0.3, graded=None,
          cells=(3, 3, 3), smooth=0.02, node_comp=0.5, sheet=False, instanced=False,
          seam_spacing=0.0, edge_length=0.0, remesh="isotropic", sides=6, node="union",
          repair=False, islands=False, check=False, pixel=0.05, layer=0.05, verbose=False,
          log=print) -> BuildResult:
    """Build a lattice block in one call, with the options of the CLI and GUIs.

    ``python -m latticegen build``, the browser app and the desktop app all
    call this function, so a block made in a GUI can be reproduced exactly
    from Python.

    Parameters
    ----------
    topology : str
        Strut topology (``'octet'``, ...) or TPMS name (``'gyroid'``, ...).
    method : {'auto', 'struts', 'sdf', 'fourier', 'tpms'}, optional
        ``'struts'``: explicit beams (:func:`~latticegen.mesh_struts`).
        ``'sdf'``: distance field, rounded beams. ``'fourier'``: organic
        beams. ``'tpms'``: surfaces. ``'auto'`` picks ``'tpms'`` for surface
        names and ``'sdf'`` otherwise.
    a : float, optional
        Cubic cell edge (mm).
    cell : tuple of 6 float, optional
        ``(a, b, c, alpha, beta, gamma)`` for a general cell; replaces ``a``.
    N : int, optional
        Field grid per cell; 0 uses :func:`default_grid`.
    rho : float, optional
        Uniform density.
    graded : tuple of 2 float, optional
        ``(rho_bottom, rho_top)``: linear gradient along z instead of ``rho``.
    cells : tuple of 3 int, optional
    smooth : float, optional
        ``'sdf'`` fillet, as a fraction of the shortest cell edge.
    node_comp : float, optional
        ``'fourier'`` junction slimming; 0 or None is off.
    sheet : bool, optional
        ``'tpms'`` sheet instead of network solid.
    instanced : bool, optional
        Build one cell and repeat it (:func:`~latticegen.mesh_instanced`).
    seam_spacing : float, optional
        Instanced seam spacing (mm); 0 freezes the seams.
    edge_length : float, optional
        Target triangle edge (mm). Simplifies field meshes, or sets the
        instanced interior remesh; 0 is off.
    remesh : {'isotropic', 'decimate'}, optional
        Instanced interior treatment.
    sides : int, optional
        ``'struts'`` polygon sides.
    node : {'union', 'hull', 'round'}, optional
        ``'struts'`` joint shape.
    repair : bool, optional
        Run :func:`~latticegen.repair` at the end.
    islands : bool, optional
        Remove floating pieces if the block has more than one component.
    check : bool, optional
        Run the DLP report.
    pixel, layer : float, optional
        Printer pixel and layer (mm), for ``N`` and the report.
    verbose : bool, optional
    log : callable, optional
        Receives progress lines; ``print`` by default.

    Returns
    -------
    BuildResult

    Notes
    -----
    ``edge_length``, ``instanced`` and ``repair`` need ``pymeshlab``, which
    the browser app does not have.

    Examples
    --------
    >>> from latticegen.app import build
    >>> res = build("octet", method="struts", a=1.0, rho=0.2, cells=(3, 3, 3),
    ...             sides=12, log=lambda *_: None)
    >>> res.filename
    'octet_3x3x3.stl'
    """
    import latticegen as lg
    t_start = time.time()
    timings = {}
    cell = lg.UnitCell.cubic(a) if cell is None else lg.UnitCell.from_parameters(*cell)
    N = N if N else default_grid(cell, pixel, instanced)
    cells = tuple(int(c) for c in cells)
    filename = f"{topology}_{'x'.join(map(str, cells))}.stl"
    if method == "auto":
        method = "tpms" if topology.lower() in lg.available_tpms() else "sdf"
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")

    fld, rho_ref = None, rho
    if method == "struts":
        # explicit prisms + hull nodes: no field, no marching cubes
        if graded is not None or instanced:
            raise ValueError("method 'struts' takes neither graded density nor instancing")
        mesh = lg.mesh_struts(cell, topology, rho=rho, cells=cells, sides=sides,
                              node=node, verbose=verbose)
        N = 0
    else:
        if method == "sdf":
            fld = lg.sdf_field(cell, topology, N=N, smooth=smooth * cell.a_min)
        elif method == "fourier":
            fld = lg.fourier_field(cell, topology, N=N, node_comp=node_comp or None)
        else:
            fld = lg.tpms_field(cell, topology, N=N, sheet=sheet)
        timings["field"] = time.time() - t_start
        log(f"field: {fld}  ({timings['field']:.1f} s)")

        if graded is not None:
            r0, r1 = graded
            rho = lambda z: r0 + (r1 - r0) * z
            rho_ref = 0.5 * (r0 + r1)
        if instanced:
            if graded is not None:
                raise ValueError("instancing needs a uniform density (no graded density)")
            mesh = lg.mesh_instanced(fld, rho=rho, cells=cells, seam_spacing=seam_spacing or None,
                                     edge_length=edge_length or None, remesh=remesh,
                                     verbose=verbose)
        else:
            mesh = lg.mesh_stack(fld, rho=rho, cells=cells, verbose=verbose)
    timings["mesh"] = time.time() - t_start
    log(f"mesh: {mesh}  ({timings['mesh']:.1f} s)")

    if edge_length and not instanced and fld is not None:
        mesh = mesh.simplify(edge_length=edge_length, verbose=True)
    if repair:
        mesh = mesh.repair(verbose=True)
    c = mesh.check()
    if islands and c["components"] > 1:
        mesh = mesh.remove_islands(verbose=True)
        c = mesh.check()
    report = mesh.report(c)
    log(report)
    dlp = ""
    if check:
        dlp = lg.format_report(lg.dlp_report(mesh, fld, rho_ref if fld is not None else None,
                                             pixel_mm=pixel, layer_mm=layer, check=c))
        log(dlp)
    timings["total"] = time.time() - t_start
    return BuildResult(mesh, fld, method, N, filename, report, dlp, timings)
