# latticegen

Periodic lattice structures for 3D printing (DLP / resin), as one Python
package.

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
* **Lean meshes**: unit-cell instancing with seam-consistent simplification
  for uniform blocks, quadric decimation / isotropic remeshing / repair /
  island removal for everything else.
* **DLP printability report**: strut, wall and pore sizes in pixels and
  layers, watertightness, floating parts, file size, resin mass.
* Command-line interface.

## Install

The package is pure Python. In the `general` conda environment everything
needed is already present (numpy, scipy, scikit-image, trimesh, pymeshlab).

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
lean = lg.mesh_instanced(field, rho=0.30, cells=(8, 8, 8), seam_spacing=field.N / 12)
lean.export("octet_8x8x8.stl")

print(mesh.report())
print(lg.format_report(lg.dlp_report(mesh, field, 0.30, pixel_mm=0.05, layer_mm=0.05)))
```

Same thing from the shell:

```bash
python -m latticegen build octet --a 2 --rho 0.3 --cells 3 3 3 --smooth 0.02 \
       --edge-length 0.05 --check -o octet_3x3x3.stl
```

## Documentation

* [`docs/fundamentals.md`](docs/fundamentals.md) – the maths and the geometry:
  reciprocal lattices and Fourier synthesis, distance fields and smooth
  minima, TPMS, level sets and density, marching cubes, welding, instancing,
  simplification, DLP considerations.
* [`docs/usage.md`](docs/usage.md) – installation, walkthrough, every public
  function and option, CLI reference, recipes, troubleshooting.
* [`examples/`](examples/) – runnable scripts.
* [`legacy/`](legacy/) – the original three scripts this package grew from.

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
  remesh.py      decimate / simplify / remesh_isotropic / repair / remove_islands
  mesh.py        LatticeMesh: diagnostics, export
  meshops.py     pure-numpy weld / watertight / volume / STL IO
  printing.py    DLP report and grid recommendation
  cli.py         command line
```
