# latticegen

Periodic lattice structures for 3D printing (DLP / resin), as one Python
package.

MATES Lab, Universidad de Sevilla-CSIC — [mates-lab.es](https://mates-lab.es/)

    UnitCell  →  Skeleton or TPMS  →  UnitField  →  LatticeMesh  →  STL

* **Three ways to build the unit-cell field**
  * `sdf_field` – exact distance to the strut skeleton: circular struts,
    unions of cylinders at the nodes, optional fillets.
  * `fourier_field` – Fourier synthesis from a structure-factor motif:
    smooth, organic struts, tunable junction bulge.
  * `tpms_field` – gyroid, Schwarz P/D, I-WP, Neovius, Lidinoid,
    Fischer-Koch S, split-P; network or sheet solids.
* **Nine built-in strut topologies** (sc, bcc, fcc, bccz, fccz, fbcc, octet,
  diamond, kelvin), any Bravais cell, custom skeletons, skeleton algebra.
* **Meshing** with flat caps, exact welding, streamed memory use; density as a
  scalar, a per-cell array, a function of height, or any function of x, y, z;
  different topologies per cell.
* **Lean meshes**: explicit prism meshes straight from the skeleton, with
  joints no bigger than the struts (no field, ~1300 triangles per octet
  cell, a 20 x 20 x 20 block in seconds; ~700 with convex-hull joints); unit-cell instancing with seam-consistent simplification and
  isotropic or anisotropic interiors for field-based blocks; quadric
  decimation / isotropic remeshing / repair / island removal for everything
  else.
* **Plates** for mechanical tests: solid plates welded to the bottom and/or
  top of any block (`add_plates`), one watertight object.
* **Export** to binary STL or 3MF, both streamed (any size).
* **DLP printability report**: strut, wall and pore sizes in pixels and
  layers, watertightness, floating parts, file size, resin mass.
* Command-line interface, and two GUIs for colleagues who do not use Python,
  sharing one front end with a 3D preview:
  * `webapp/`: runs this package in the browser (WebAssembly), from a web page
    or an offline zip; everything except the pymeshlab steps.
  * `desktop/`: a Windows app (one zip, no install) with the native engine,
    including simplification, seam-consistent instancing and repair.

## Install

The package is pure Python. In the `general` conda environment everything
needed is already present (numpy, scipy, scikit-image, trimesh, pymeshlab,
manifold3d). `manifold3d` (`pip install manifold3d`) builds the default
strut-sized joints of `mesh_struts`; `node="hull"` works without it.

```bash
cd lattice-generator
pip install -e .            # optional; or just run from this directory
python -m unittest discover -s tests
```

`pymeshlab` is only needed for the simplification / remeshing / repair tools
and for the instanced mesher's remeshing steps
(`conda install -c conda-forge pymeshlab`).

## Quick start

```python
import latticegen as lg

cell  = lg.UnitCell.cubic(2.0)                          # 2 mm cell
field = lg.sdf_field(cell, "octet", N=80, smooth=0.04)  # distance field, filleted nodes
mesh  = lg.mesh_stack(field, rho=0.30, cells=(3, 3, 3)) # 30 % dense, 6 mm block
mesh  = mesh.simplify(edge_length=0.05)                 # ~1 pixel triangles
mesh.export("octet_3x3x3.stl")

# large uniform blocks: instance one simplified cell instead (seconds, low memory)
lean = lg.mesh_instanced(field, rho=0.30, cells=(8, 8, 8), seam_spacing=0.17)   # mm
lean.export("octet_8x8x8.stl")

# thousands of cells: build the struts explicitly (prisms + strut-sized joints, no field);
# a 2 cm cube of 1 mm cells is 10 M triangles (5 M with node="hull") and takes ~5 s
cube = lg.mesh_struts(lg.UnitCell.cubic(1.0), "octet", rho=0.20, cells=(20, 20, 20))
cube.export("octet_2cm.stl")

# 1 mm plates on both ends, 20 % wider than the block, as one 3MF object
plated = lg.add_plates(cube, lg.UnitCell.cubic(1.0), (20, 20, 20), "both", thickness=1.0, size=1.2)
plated.export("octet_2cm_plates.3mf")

print(mesh.report())
print(lg.format_report(lg.dlp_report(mesh, field, 0.30, pixel_mm=0.05, layer_mm=0.05)))
```

Same thing from the shell:

```bash
python -m latticegen build octet --a 2 --rho 0.3 --cells 3 3 3 --smooth 0.02 \
       --edge-length 0.05 --check -o octet_3x3x3.stl
python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 20 20 20 \
       --check -o octet_2cm.stl
python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 10 10 10 \
       --plates both --plate-thickness 1 -o octet_plates.3mf
```

## Documentation

* [`docs/fundamentals.md`](docs/fundamentals.md) – the maths and the geometry:
  reciprocal lattices and Fourier synthesis, distance fields and smooth
  minima, TPMS, level sets and density, marching cubes, welding, instancing,
  explicit strut meshes, simplification, DLP considerations.
* [`docs/usage.md`](docs/usage.md) – installation, walkthrough, every public
  function and option, CLI reference, recipes, troubleshooting.
* [`docs/api/`](docs/api/) – API reference built from the docstrings with
  Sphinx: `pip install sphinx furo`, then
  `python -m sphinx -b html docs/api docs/api/_build/html`.
* [`docs/manual/LatticeGenerator-User-Manual.pdf`](docs/manual/LatticeGenerator-User-Manual.pdf)
  – user manual of the Windows desktop app, with screenshots.
* [`examples/`](examples/) – runnable scripts.

## Layout

```
latticegen/
  cells.py       UnitCell: Bravais cells, reciprocal basis
  topologies.py  Skeleton + built-in strut topologies
  fourier.py     reciprocal vectors, structure-factor coefficients, node compensation
  sdf.py         periodic point-segment distance fields, smooth-min
  tpms.py        TPMS level-set functions
  field.py       UnitField: density<->isovalue, feature sizes, builders
  meshing.py     mesh_stack: slab-streamed marching cubes with caps, graded density
  instanced.py   mesh_instanced: unit-cell instancing + seam-consistent simplification
  struts.py      mesh_struts: explicit n-gon prisms + strut-sized (manifold3d) or hull nodes, closed-form density
  remesh.py      decimate / simplify / remesh_isotropic / repair / remove_islands
  mesh.py        LatticeMesh: diagnostics, export
  meshops.py     pure-numpy weld / watertight / volume / STL and 3MF IO
  printing.py    DLP report and grid recommendation
  plates.py      add_plates: solid plates welded to the block (manifold3d union)
  app.py         build(): one-call lattice build shared by the CLI and the web app
  cli.py         command line
webapp/          browser GUI (Pyodide + Three.js); see webapp/README.md
desktop/         Windows app: same GUI + native engine; see desktop/README.md
```

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 MATES Lab, Universidad de Sevilla-CSIC.
