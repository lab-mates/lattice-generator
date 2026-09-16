"""
Command-line interface::

    python -m latticegen build octet --method sdf --a 2 --rho 0.3 --cells 5 5 5 -o octet.stl
    python -m latticegen build gyroid --method tpms --sheet --a 3 --rho 0.25 --cells 4 4 4 -o gyr.stl
    python -m latticegen build bcc --graded 0.15 0.40 --cells 4 4 8 -o graded.stl
    python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 20 20 20 -o cube.stl
    python -m latticegen list
    python -m latticegen check part.stl --pixel 0.05 --layer 0.05
    python -m latticegen simplify part.stl --edge-length 0.05 -o part_lean.stl
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _build(args):
    import latticegen as lg
    t_start = time.time()
    cell = lg.UnitCell.cubic(args.a) if args.cell is None else lg.UnitCell.from_parameters(*args.cell)
    N = args.N if args.N else lg.recommend_grid(cell, args.pixel)
    method = args.method
    cells = tuple(args.cells)
    if method == "auto":
        method = "tpms" if args.topology.lower() in lg.available_tpms() else "sdf"
    if method == "struts":
        # explicit prisms + hull nodes: no field, no marching cubes
        if args.graded is not None or args.instanced:
            raise SystemExit("--method struts takes neither --graded nor --instanced")
        mesh = lg.mesh_struts(cell, args.topology, rho=args.rho, cells=cells, sides=args.sides,
                              node=args.node, verbose=args.verbose)
        print(f"mesh: {mesh}  ({time.time() - t_start:.1f} s)")
        if args.repair:
            mesh = mesh.repair(verbose=True)
        print(mesh.report())
        if args.check:
            print(lg.format_report(lg.dlp_report(mesh, pixel_mm=args.pixel, layer_mm=args.layer)))
        out = args.output or f"{args.topology}_{'x'.join(map(str, cells))}.stl"
        mesh.export(out)
        print(f"written {out}  ({time.time() - t_start:.1f} s total)")
        return
    if method == "sdf":
        field = lg.sdf_field(cell, args.topology, N=N, smooth=args.smooth * cell.a_min)
    elif method == "fourier":
        field = lg.fourier_field(cell, args.topology, N=N, node_comp=args.node_comp)
    elif method == "tpms":
        field = lg.tpms_field(cell, args.topology, N=N, sheet=args.sheet)
    else:
        raise SystemExit(f"unknown method {method}")
    print(f"field: {field}  ({time.time() - t_start:.1f} s)")

    if args.graded is not None:
        r0, r1 = args.graded
        rho = lambda z: r0 + (r1 - r0) * z
        rho_ref = 0.5 * (r0 + r1)
    else:
        rho = rho_ref = args.rho

    if args.instanced:
        if args.graded is not None:
            raise SystemExit("--instanced needs a uniform density (no --graded)")
        spacing = args.seam_spacing if args.seam_spacing else None
        edge = args.edge_length if args.edge_length else None
        mesh = lg.mesh_instanced(field, rho=rho, cells=cells, seam_spacing=spacing,
                                 edge_length=edge, remesh=args.remesh, verbose=args.verbose)
    else:
        mesh = lg.mesh_stack(field, rho=rho, cells=cells, verbose=args.verbose)
    print(f"mesh: {mesh}  ({time.time() - t_start:.1f} s)")

    if args.edge_length and not args.instanced:
        mesh = mesh.simplify(edge_length=args.edge_length, verbose=True)
    if args.repair:
        mesh = mesh.repair(verbose=True)
    print(mesh.report())
    if args.check:
        print(lg.format_report(lg.dlp_report(mesh, field, rho_ref, pixel_mm=args.pixel,
                                             layer_mm=args.layer)))
    out = args.output or f"{args.topology}_{'x'.join(map(str, cells))}.stl"
    mesh.export(out)
    print(f"written {out}  ({time.time() - t_start:.1f} s total)")


def _list(args):
    import latticegen as lg
    print("strut topologies (methods sdf / fourier):")
    for name in lg.available_topologies():
        print(f"  {lg.skeleton(name)}")
    print("TPMS (method tpms):")
    for name in lg.available_tpms():
        print(f"  {name}")


def _check(args):
    import latticegen as lg
    mesh = lg.LatticeMesh.load(args.mesh)
    print(mesh.report())
    print(lg.format_report(lg.dlp_report(mesh, pixel_mm=args.pixel, layer_mm=args.layer)))


def _simplify(args):
    import latticegen as lg
    mesh = lg.LatticeMesh.load(args.mesh)
    print(mesh.report())
    if args.remesh:
        out = mesh.remesh(args.edge_length, verbose=True)
    else:
        out = mesh.simplify(edge_length=args.edge_length, fraction=args.fraction, verbose=True)
    if args.repair:
        out = out.repair(verbose=True)
    print(out.report())
    path = args.output or args.mesh.rsplit(".", 1)[0] + "_simplified.stl"
    out.export(path)
    print(f"written {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="latticegen", description="periodic lattices for 3D printing")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate a lattice block and write a mesh")
    b.add_argument("topology", help="strut topology (octet, bcc, ...) or TPMS name (gyroid, ...)")
    b.add_argument("--method", default="auto", choices=["auto", "sdf", "fourier", "tpms", "struts"],
                   help="field type, or 'struts' for explicit prisms + hull nodes (leanest)")
    b.add_argument("--a", type=float, default=1.0, help="cubic cell edge (mm)")
    b.add_argument("--cell", type=float, nargs=6, metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
                   help="general cell parameters instead of --a")
    b.add_argument("--N", type=int, default=0, help="grid samples per cell (default from --pixel)")
    b.add_argument("--rho", type=float, default=0.3, help="relative density (0-1)")
    b.add_argument("--graded", type=float, nargs=2, metavar=("RHO_BOTTOM", "RHO_TOP"),
                   help="linear density gradient along z")
    b.add_argument("--cells", type=int, nargs=3, default=[3, 3, 3], metavar=("NX", "NY", "NZ"))
    b.add_argument("--smooth", type=float, default=0.02, help="SDF junction fillet (fraction of a)")
    b.add_argument("--node-comp", type=float, default=0.5, help="Fourier junction de-bulking (0=off)")
    b.add_argument("--sheet", action="store_true", help="TPMS sheet solid instead of network")
    b.add_argument("--instanced", action="store_true", help="unit-cell instancing (uniform blocks)")
    b.add_argument("--seam-spacing", type=float, default=0.0,
                   help="instanced: seam subsampling arc length in mm (e.g. a/12); 0 = frozen seams")
    b.add_argument("--edge-length", type=float, default=0.0,
                   help="target triangle edge in mm: decimation target, or the interior "
                        "remesh target with --instanced")
    b.add_argument("--remesh", default="isotropic", choices=["isotropic", "decimate"],
                   help="instanced: interior remesh (uniform triangles) or quadric decimation "
                        "(long triangles along struts, fewer faces)")
    b.add_argument("--sides", type=int, default=6, help="struts: polygon sides of the strut section")
    b.add_argument("--node", default="hull", choices=["hull", "round"],
                   help="struts: node shape (convex hull of the strut rings, or rounded)")
    b.add_argument("--repair", action="store_true")
    b.add_argument("--check", action="store_true", help="print the DLP printability report")
    b.add_argument("--pixel", type=float, default=0.05, help="DLP pixel pitch (mm)")
    b.add_argument("--layer", type=float, default=0.05, help="layer thickness (mm)")
    b.add_argument("-o", "--output", help="output mesh path (.stl, .obj, .ply, .3mf)")
    b.add_argument("-v", "--verbose", action="store_true")
    b.set_defaults(func=_build)

    l = sub.add_parser("list", help="list available topologies and TPMS")
    l.set_defaults(func=_list)

    c = sub.add_parser("check", help="report on an existing mesh file")
    c.add_argument("mesh")
    c.add_argument("--pixel", type=float, default=0.05)
    c.add_argument("--layer", type=float, default=0.05)
    c.set_defaults(func=_check)

    s = sub.add_parser("simplify", help="decimate / remesh an existing mesh file")
    s.add_argument("mesh")
    s.add_argument("--edge-length", type=float, help="target edge length (mm)")
    s.add_argument("--fraction", type=float, help="keep this fraction of faces")
    s.add_argument("--remesh", action="store_true", help="isotropic remesh instead of decimation")
    s.add_argument("--repair", action="store_true")
    s.add_argument("-o", "--output")
    s.set_defaults(func=_simplify)

    args = p.parse_args(argv)
    if args.cmd == "build" and args.node_comp == 0:
        args.node_comp = None
    if args.cmd == "simplify" and not (args.edge_length or args.fraction):
        p.error("simplify needs --edge-length or --fraction")
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
