# Usage guide

Everything you need to build lattice blocks with `latticegen`, from install
to print-ready STL or 3MF. latticegen is developed by Joaquin Ramirez-Rico,
University of Seville ([mates-lab.es](https://mates-lab.es/)). The [fundamentals](fundamentals.md) explain *why* the
tools work the way they do; this guide explains *how* to use them.

Contents

1. [Installation](#1-installation)
2. [Five-minute walkthrough](#2-five-minute-walkthrough)
3. [Unit cells](#3-unit-cells)
4. [Topologies (skeletons)](#4-topologies-skeletons)
5. [Unit fields](#5-unit-fields)
6. [Meshing a block](#6-meshing-a-block)
7. [Graded and mixed blocks](#7-graded-and-mixed-blocks)
8. [Lean meshes: instancing](#8-lean-meshes-instancing)
9. [Lean meshes: explicit struts](#9-lean-meshes-explicit-struts)
10. [Simplification, remeshing, repair](#10-simplification-remeshing-repair)
11. [Diagnostics and the DLP report](#11-diagnostics-and-the-dlp-report)
12. [Plates for mechanical tests](#12-plates-for-mechanical-tests)
13. [Export and import](#13-export-and-import)
14. [Command-line interface](#14-command-line-interface)
15. [Recipes](#15-recipes)
16. [Performance and memory](#16-performance-and-memory)
17. [Troubleshooting](#17-troubleshooting)
18. [API summary](#18-api-summary)

---

## 1. Installation

Requirements: Python 3.10+, numpy, scipy, scikit-image, trimesh. Optional:
pymeshlab (simplification, remeshing, repair, the remeshing steps of the
instanced mesher). All are present in the `general` conda environment.

```bash
cd lattice-generator
pip install -e .                       # optional: makes `import latticegen` work anywhere
python -m unittest discover -s tests   # ~20 s
```

Without installing, run scripts from the `lattice-generator` directory or
set `PYTHONPATH` to it. If pymeshlab is missing:

```bash
conda install -c conda-forge pymeshlab
```

## 2. Five-minute walkthrough

```python
import latticegen as lg

# 1. the unit cell, in the units you print in (mm)
cell = lg.UnitCell.cubic(2.0)

# 2. a periodic field of one cell.  Distance field of the octet truss with a
#    small fillet at the nodes; N = samples per cell edge, chosen so that a
#    voxel is half a printer pixel (N = 80 -> 0.025 mm)
N = lg.recommend_grid(cell, pixel_mm=0.05)
field = lg.sdf_field(cell, "octet", N=N, smooth=0.04)

# 3. mesh a 3 x 3 x 3 block at 30 % relative density
#    (for large uniform blocks use lg.mesh_instanced, section 8, or the
#    field-free lg.mesh_struts, section 9)
mesh = lg.mesh_stack(field, rho=0.30, cells=(3, 3, 3))
print(mesh.report())

# 4. marching cubes over-resolves: decimate to ~1 printer pixel triangles
mesh = mesh.simplify(edge_length=0.05, verbose=True)

# 5. printability report and export
print(lg.format_report(lg.dlp_report(mesh, field, 0.30, pixel_mm=0.05, layer_mm=0.05)))
mesh.export("octet_3x3x3.stl")
```

Output (steps 3 to 5; the field takes 4 s, the block 8 s, the decimation
of 3 M faces about a minute):

```
vertices 1,516,232   faces 3,034,800   (binary STL ~151.7 MB)
watertight True   outward normals True   components 1   Euler -1168
extents 6 x 6 x 6
volume 64.8   area 661.725   relative density 0.3000
edge length min/median/max 2.905e-05 / 0.02516 / 0.0401
decimate: faces 3,034,800 -> 611,276  (20.1%), volume change -0.069%, watertight True
DLP check  (pixel 0.05 mm, layer 0.05 mm)
  field grid N=80, voxel 0.0250 mm
  feature sizes (mm): strut_radius 0.173, node_radius 0.325, min_thickness 0.346, max_thickness 0.650, pore_radius 0.548, pore_throat 0.700
  thinnest solid = 6.9 px / 6.9 layers; narrowest pore = 14.0 px
  part 6.00 x 6.00 x 6.00 mm, 120 layers, volume 64.8 mm^3, ~0.07 g resin
  mesh: 611,276 faces (30.6 MB STL), watertight True, 1 component(s)
  no problems found
```

Everything else in this guide is a variation on these five steps.

## 3. Unit cells

`UnitCell` holds the lattice vectors as rows of a 3 x 3 matrix `A`
(cartesian, your length unit). Fractional coordinates `u` map to cartesian
`r = u @ A`.

```python
lg.UnitCell.cubic(2.0)                         # a = 2 mm
lg.UnitCell.tetragonal(2.0, 3.0)               # a = b = 2, c = 3
lg.UnitCell.orthorhombic(2.0, 2.5, 3.0)
lg.UnitCell.hexagonal(2.0, 3.0)                # gamma = 120 deg
lg.UnitCell.from_parameters(2, 2, 3, 90, 90, 100)   # a, b, c, alpha, beta, gamma
lg.UnitCell.from_matrix([[2, 0, 0], [0.3, 2, 0], [0, 0, 3]])
```

Useful attributes: `A`, `volume`, `lengths` (|a1|, |a2|, |a3|), `a_min`,
`is_cubic`, `reciprocal()`, `frac_to_cart(u)`, `cart_to_frac(r)`,
`scaled(f)`. Anywhere a `cell` is expected you may also pass a plain number
(cubic edge) or a 3 x 3 array.

## 4. Topologies (skeletons)

A `Skeleton` is a list of straight struts with fractional endpoints, one copy
per unit cell.

```python
lg.available_topologies()      # ['sc', 'bcc', 'fcc', 'bccz', 'fccz', 'fbcc', 'octet', 'diamond', 'kelvin']
sk = lg.skeleton("octet")      # Skeleton(octet: 24 struts, 4 nodes, connectivity 12)
sk.segments                    # list of (p, q) arrays
sk.nodes()                     # unique node positions in [0,1)^3
sk.midpoints()                 # strut midpoints
sk.strut_lengths(cell)         # cartesian lengths
sk.connectivity()              # struts per node (max)
```

Combining and defining your own:

```python
bcc_sc = lg.skeleton("bcc") + lg.skeleton("sc")          # struts are merged, duplicates dropped

cross = lg.Skeleton.custom(
    [([.5, .5, .5], [1, .5, .5]), ([.5, .5, .5], [.5, 1, .5]), ([.5, .5, .5], [.5, .5, 1]),
     ([.5, .5, .5], [0, .5, .5]), ([.5, .5, .5], [.5, 0, .5]), ([.5, .5, .5], [.5, .5, 0])],
    name="cross", sigma=0.12, gmax=5.0)                  # sigma/gmax = Fourier defaults

# from nodes and bond vectors (bonds are applied at every node)
dia = lg.Skeleton.from_nodes_and_bonds(
    nodes=[[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
    bond_vectors=[[.25, .25, .25], [.25, -.25, -.25], [-.25, .25, -.25], [-.25, -.25, .25]])
```

Segments that are lattice translates of one another are merged
automatically, so you can list struts generously (all 27 neighbours, both
endpoint orders) without double counting. Isolated points (spherical
nodes) go in `points=`.

## 5. Unit fields

A `UnitField` is the sampled periodic field of one cell plus the cell
geometry. Solid is where `f >= threshold`. Three builders:

### 5.1 Distance field (recommended for trusses)

```python
f = lg.sdf_field(cell, "octet", N=96, smooth=0.0)   # sharp cylinder unions
f = lg.sdf_field(cell, "octet", N=96, smooth=0.04)  # fillets of ~0.04 mm at the nodes
```

* `smooth` is in cell length units; keep it around 10 to 30 % of the strut
  radius, larger values inflate the nodes (see the fundamentals, section 5).
* `f.strut_radius(rho)` is exact for distance fields.
* Extra keywords: `block` (chunk size, default 24 voxels), `cull=False` to
  disable the exact bounding-box culling (only for debugging).

### 5.2 Fourier synthesis

```python
f = lg.fourier_field(cell, "octet", N=96)                        # tuned defaults
f = lg.fourier_field(cell, "octet", N=96, node_comp=0.5)         # de-bulked junctions
f = lg.fourier_field(cell, cross, N=64, sigma=0.18, gmax=4.5)    # custom skeleton, manual smearing
```

* `sigma`: Gaussian smearing radius (cell units). Default: the topology's
  tuned fraction of the shortest cell vector.
* `gmax`: reciprocal cutoff in units of `2 pi / a_min`. `N` must exceed
  `2 max|hkl|`; a `ValueError` tells you the minimum.
* `node_comp`: junction de-bulking strength, 0.5 is a good value for
  12-connected nodes at densities of 20 % and above. Check
  `connectivity_check` afterwards (section 11): strong compensation or
  small `sigma` can leave floating blobs.
* `weights`: per-strut weights (thicker / thinner struts in one cell).
* `f.meta` holds `hkl`, the coefficients `c`, and the parameters used.

### 5.3 TPMS

```python
lg.available_tpms()   # gyroid, primitive, diamond, iwp, neovius, lidinoid, fischer_koch_s, split_p
g  = lg.tpms_field(cell, "gyroid", N=64)                 # network solid
gs = lg.tpms_field(cell, "gyroid", N=64, sheet=True)     # sheet (wall) solid
g2 = lg.tpms_field(cell, "primitive", N=64, periods=(2, 2, 1), phase=(0.25, 0, 0))
```

### 5.4 Common `UnitField` methods

| method | purpose |
|---|---|
| `threshold(rho)` | isovalue from the voxel quantile (vectorised) |
| `calibrated_threshold(rho)` | isovalue calibrated on the meshed unit-cell volume (used by the meshers) |
| `density(t)` | voxel-count volume fraction of `f >= t` |
| `solid_mask(rho)` | boolean N^3 array |
| `strut_radius(rho)` | exact (SDF) or EDT-measured strut radius |
| `feature_sizes(rho)` | strut / node / thickness / pore sizes, connectivity (dict) |
| `connectivity_check(rho)` | solid & void components, floating islands, enclosed voids |
| `mesh(rho, cells, ...)` | shortcut for `mesh_stack(self, ...)` |
| `mesh_instanced(rho, cells, ...)` | shortcut for `mesh_instanced(self, ...)` |
| `save(path)`, `UnitField.load(path)` | `.npz` persistence of expensive fields |
| `custom_field(cell, array)` | wrap any periodic N^3 array of your own |

`N`, `A`, `cell`, `voxel_size`, `name`, `method`, `skeleton` are attributes.

## 6. Meshing a block

```python
mesh = lg.mesh_stack(field, rho=0.30, cells=(5, 5, 5))
```

`mesh_stack(fields, rho, cells, cell_map=None, slab_cells=1, exact_density=True, threshold=None, verbose=False)`

* `fields`: one `UnitField` or a list (same `N`) for `cell_map`.
* `rho`: scalar, `(nx, ny, nz)` array, `rho(zeta)` or `rho(x, y, z)` (next section).
* `slab_cells`: cell layers per marching-cubes pass; raise it for speed if
  memory allows, lower it for very large footprints.
* `exact_density`: calibrate isovalues on the meshed unit-cell volume
  (default). Set `False` to reproduce the plain quantile mapping.
* `threshold`: give isovalue(s) directly instead of `rho` (advanced).

The result is a `LatticeMesh`: closed, welded, outward normals, cartesian
units, with `mesh.info` holding the thresholds, the cell count and the
block volume. `mesh_stack` warns if the mesh has more than one connected
component.

## 7. Graded and mixed blocks

```python
# per-cell density array
rho = np.full((4, 4, 4), 0.25); rho[1:-1, 1:-1, 1:-1] = 0.40
mesh = lg.mesh_stack(field, rho=rho, cells=(4, 4, 4))

# continuous gradient along the build direction: zeta in [0, 1]
mesh = lg.mesh_stack(field, rho=lambda zeta: 0.15 + 0.25 * zeta, cells=(3, 3, 6))

# any function of cartesian position (mm, block origin at 0)
def radial(x, y, z):
    r = np.hypot(x - 4.0, y - 4.0)
    return np.clip(0.45 - 0.06 * r, 0.12, 0.6)
mesh = lg.mesh_stack(field, rho=radial, cells=(4, 4, 3))

# several topologies in one block
octet = lg.sdf_field(cell, "octet", N=64, smooth=0.04)
fcc   = lg.sdf_field(cell, "fcc",   N=64, smooth=0.04)
cmap  = np.zeros((4, 4, 4), int); cmap[1:-1, 1:-1, 1:-1] = 1     # 0 = octet, 1 = fcc
mesh  = lg.mesh_stack([octet, fcc], rho=rho, cells=(4, 4, 4), cell_map=cmap)
```

Rules:

* Callables must be numpy-vectorised, elementwise and deterministic, and
  return values strictly inside (0, 1). `np.clip` is your friend.
* A 1-argument callable is interpreted as `rho(zeta)`, a 3-argument one as
  `rho(x, y, z)`. Use `lambda x, y, z: ...` explicitly for the 3-D form.
* Mixed topologies connect only where both are solid on the shared face.
  Safe pairs: any topology with itself at another density; fcc inside
  octet; bcc inside fbcc; sc inside bccz/fccz. Incompatible pairs (e.g.
  octet and gyroid) produce separate solids, and `mesh_stack` warns.

## 8. Lean meshes: instancing

For a homogeneous, uniform-density block, `mesh_instanced` meshes one cell,
simplifies it consistently, replicates and welds. Time and memory do not
grow with the block.

```python
lean = lg.mesh_instanced(field, rho=0.30, cells=(8, 8, 8), seam_spacing=0.17)   # mm
```

`mesh_instanced(field, rho, cells, seam_spacing=None, edge_length=None, remesh="isotropic", density_correct=True, exact_density=True, anisotropy=4.0, quality_thr=0.1, verbose=False)`

All lengths are in the cell unit (mm), like everywhere else in the API;
values below one voxel raise an error because they almost always mean a
unit mix-up.

* `seam_spacing` (mm): subsample the seam curves (periodic face loops, cap
  rims, cube-edge chains) at this arc spacing and remesh the interiors of
  body and caps at `edge_length`, which defaults to the same value. The
  useful range is 1/16 to 1/8 of the cell edge: 0.17 mm on a 2 mm cell.
  The strongest, most uniform simplification.
* `edge_length` (mm): triangle size of the interior remesh. Give it alone
  (with `seam_spacing=None`) to keep the seams at marching-cubes density
  and remesh only the interiors to that size; give both to make the seams
  coarser or finer than the interiors.
* `seam_spacing=None, edge_length=None` remeshes the interior at pymeshlab's
  default (1 % of the bounding box) with `remesh="isotropic"`, or decimates
  it with `remesh="decimate"`; `remesh=None` gives raw instancing (no
  pymeshlab needed, same triangle count as `mesh_stack` but far less
  memory).

| call | seams | interiors |
|---|---|---|
| `mesh_instanced(f, ...)` | marching density | isotropic, 1 % of bbox |
| `mesh_instanced(f, ..., edge_length=0.1)` | marching density | isotropic, 0.1 mm |
| `mesh_instanced(f, ..., seam_spacing=0.17)` | 0.17 mm | 0.17 mm |
| `mesh_instanced(f, ..., seam_spacing=0.17, edge_length=0.1)` | 0.17 mm | 0.1 mm |
| `mesh_instanced(f, ..., seam_spacing=0.17, remesh="decimate")` | 0.17 mm | quadric decimation, ~1/4 of the isotropic face count |
| `mesh_instanced(f, ..., remesh=None)` | marching density | marching density |

* `remesh="decimate"` (with `seam_spacing`): instead of uniform triangles,
  quadric edge collapse with a low quality bound (`quality_thr`, default
  0.1) so triangles stretch along straight struts; the face budget is the
  isotropic estimate divided by `anisotropy` (default 4). Typically 3-5x
  fewer faces than isotropic at the same seams. Values of `anisotropy`
  much above 4 change the volume enough that the density correction may
  not fully converge; check `mesh.relative_density`.
* `density_correct`: after simplification the unit-cell volume is measured
  and the isovalue is iterated until the density matches `rho` (default on).

Comparison on a 6 x 6 x 6 octet block, 2 mm cell, `N = 96`, 30 %:

| method | faces | time |
|---|---|---|
| `mesh_stack` (marching cubes) | ~20 M | not attempted (RAM) |
| `mesh_instanced(seam_spacing=0.17)` | 0.70 M | 18 s |
| `mesh_stack` 3 x 3 x 3 then `simplify(edge_length=0.05)` | 4.5 M -> 0.61 M | 3 s + 95 s |

On a 1 mm octet at 20 %, `N = 80`, seams and interiors at 0.05 mm, 3 x 3 x 3:
`remesh="isotropic"` 176 k faces (6 500 per cell), `remesh="decimate"`
36 k faces (1 350 per cell); the explicit strut mesh of the next section
is 660 per cell.

## 9. Lean meshes: explicit struts

For strut topologies the leanest mesh needs no field at all: every strut
is an n-gon prism (two rings, `2n` triangles whatever its length), and by
default every node is the union of the strut ends and a strut-sized sphere.
The joints are no bigger than the struts.
The maths is in [fundamentals, section 11](fundamentals.md#11-explicit-strut-meshes-prisms-and-hull-nodes).

```python
cube = lg.mesh_struts(lg.UnitCell.cubic(1.0), "octet", rho=0.20, cells=(20, 20, 20))
cube.export("octet_2cm.stl")               # 9.9 M faces, 500 MB, ~5 s (5.4 M with node="hull")
print(lg.format_report(lg.dlp_report(cube, pixel_mm=0.05)))   # sizes come from the mesh itself
```

`mesh_struts(cell, topology, rho=None, radius=None, cells=(1, 1, 1), sides=6, node="union", round_points=64, verbose=False)`

* `topology`: a built-in name or a `Skeleton` (isolated `points` are
  ignored). Any `UnitCell`.
* `rho`: **bulk** relative density of the periodic lattice; the strut
  radius is solved from the closed-form volume `V(r) = alpha r^2 - beta r^3`
  (`lg.radius_for_density(cell, topology, rho)` gives it alone). Or give
  `radius` (mm) directly. Unreachable densities (struts vanishing into the
  nodes) raise a `ValueError` naming the maximum.
* `sides`: polygon of the strut section. `r` is the circumradius; the
  thinnest dimension is `2 r cos(pi/sides)`. With struts of one or two
  pixels 4 to 6 sides are indistinguishable in print; 8 to 12 for thick
  struts.
* `node="union"` (default): joints the size of the struts. Nothing lies
  farther than `r` from the skeleton. Needs `manifold3d` (a pip dependency);
  ~1.9x the triangles of hull nodes for the octet.
  `node="hull"`: convex hull of the rings, no extra dependency, but node
  vertices at about `2 r` for the octet (1.4-2 r depending on the angles).
  `node="round"`: a sampled sphere added to the hull (rounded convex
  bulge). None produces concave fillets; for those use
  `sdf_field(..., smooth=k)` and `mesh_instanced`. See
  [`img/joints-union-vs-hull.png`](img/joints-union-vs-hull.png).
* Boundary: every strut whose two nodes lie inside the closed block is
  kept whole, so the outer faces carry complete nodes, the part is about
  `2 r` larger than `n a`, and `mesh.relative_density` (volume over the
  nominal box) sits a little above `rho`: +3 % on 20 cells, +30 % on 2.
  Struts that end at the block boundary with no partner (bcc and diamond
  corners) are closed with a flat cap.
* The result is watertight by construction (shared vertex indices, no
  welding) and `mesh.info` holds `strut_radius`, `node_radius`,
  `min_thickness`, `sides`, `n_struts`, `n_nodes`, `bulk_density`,
  `analytic_volume`; `dlp_report(mesh)` reads them without a field.

Triangles per bulk cell with hexagons, union / hull nodes: octet 1287 / ~700,
kelvin 1345 / 662, bcc 336 / 214, sc 181 / 101. A 20 x 20 x 20 octet block is
9.9 M faces (500 MB STL) with union nodes and 5.4 M (270 MB) with hull nodes,
against ~50 M for `mesh_instanced` at 0.05 mm triangles and ~1000 M for raw
marching cubes at `N = 80`.

What you give up: fillets and organic blends (convex nodes only),
gradation (uniform radius; per-strut radii are a possible extension), and
TPMS / Fourier geometries, which have no straight parts and stay on the
field route.

## 10. Simplification, remeshing, repair

All are methods of `LatticeMesh` (and functions in `latticegen.remesh`) and
return a **new** mesh. They need pymeshlab.

```python
m2 = mesh.simplify(edge_length=0.05)        # face budget from area, ~1 pixel triangles
m2 = mesh.simplify(fraction=0.2)            # keep 20 % of the faces
m2 = mesh.decimate(target_faces=500_000, quality_thr=0.3, preserve_topology=True, planar=True)
m3 = mesh.remesh(edge_length=0.1, iterations=10, feature_deg=30)   # isotropic, FEA-style
m4 = mesh.repair(fill_holes=True, max_hole_edges=100)             # clean-up + orientation
m5 = mesh.remove_islands()                   # drop components < 0.1 % of the largest
m5 = mesh.remove_islands(keep="largest")
```

Guidance:

* For print files use `simplify` (quadric decimation): watertight, volume
  within ~0.3 %, flat caps become a handful of triangles. An edge length of
  one printer pixel is invisible in the print.
* `remesh` gives uniform triangles but shrinks thin struts slightly
  (smoothing); pick an edge length below the strut radius and check the
  volume change printed with `verbose=True`.
* `repair` fixes meshes imported from elsewhere or damaged by aggressive
  decimation; it also fills small holes and re-orients faces outward.
* All functions warn when a watertight input comes out with open edges.

## 11. Diagnostics and the DLP report

```python
mesh.check()          # dict: n_vertices, n_faces, watertight, volume, area, components,
                      #       extents, edge_min/median/max, euler, relative_density, outward, stl_size_MB
print(mesh.report())  # the same, formatted
mesh.components()     # (labels per face, count)
mesh.relative_density # volume / block volume
mesh.edge_lengths()

field.feature_sizes(0.30)
# {'strut_radius': 0.172, 'node_radius': 0.333, 'node_to_strut_ratio': 1.94,
#  'min_thickness': 0.344, 'max_thickness': 0.667, 'pore_radius': 0.561, 'pore_throat': 0.667,
#  'solid_components': 1, 'void_components': 1, 'floating_solids': 0, 'enclosed_voids': 0}

rep = lg.dlp_report(mesh, field, rho=0.30, pixel_mm=0.05, layer_mm=0.05,
                    min_feature_px=4, resin_density_g_cm3=1.15, shrinkage_pct=0.0)
print(lg.format_report(rep))
lg.recommend_grid(cell, pixel_mm=0.05)     # N such that the voxel <= half a pixel (80 for a 2 mm cell)
```

The report flags: voxel coarser than the pixel, thinnest solid below
`min_feature_px`, pore throat below `min_pore_mm` (default 4 px), floating
solid pieces and enclosed voids in the unit cell, non-watertight or
inverted meshes, disconnected components, over-fine triangulation, and
more than 5 M faces. Either argument (`mesh`, `field`) may be omitted.

## 12. Plates for mechanical tests

`add_plates` welds solid plates to the bottom and/or top of a block, e.g. as
loading platens for compression tests. The result is one closed solid, so
the STL or 3MF holds a single object.

```python
cell = lg.UnitCell.cubic(1.0)
cube = lg.mesh_struts(cell, "octet", rho=0.20, cells=(10, 10, 10))
cube = lg.add_plates(cube, cell, (10, 10, 10), where="both",   # "bottom", "top", "both"
                     thickness=1.0,                            # mm
                     size=1.2)                                 # side / block side, >= 1, centred
cube.export("octet_plates.3mf")
cube.relative_density            # the lattice's density: the plates are left out
cube.info["plate_volume"]        # volume the plates added
```

* **Where the plates attach.** Field meshes (`mesh_stack`, `mesh_instanced`)
  end in flat caps on the block faces, and the plates are placed against
  them. Strut meshes end in whole beams and joints that reach one beam
  radius past the outermost nodes; a plate below them would only touch the
  joints at a point. So for `mesh_struts` the plate faces go through the
  centres of the lowest and highest nodes (`mesh.info["node_z"]`), and the
  joints and in-plane beams there are half embedded. `planes=(z_bottom,
  z_top)` overrides both.
* **Shape.** Each plate is the block's footprint (spanned by `nx a1` and
  `ny a2`) scaled about its centre: a rectangle for 90 degree cells, a
  parallelogram otherwise. Oblique `a3` is fine; the top plate follows the
  sheared block.
* **How.** A mesh boolean union (`manifold3d`, a dependency already) of the
  faces near the plates only; the rest of the block is stitched back
  unchanged. A 20 x 20 x 20 octet block (9.9 M faces) gets both plates in
  ~3 s.
* Add the plates last: after `simplify` (which would coarsen them) and after
  `remove_islands`, or slivers cut off at the block faces get welded onto the
  plate. `build()` does it in that order.

`latticegen.app.build` and the command line take `plates`, `plate_thickness`
and `plate_size` (`--plates {none,bottom,top,both}`, `--plate-thickness`,
`--plate-size`).

## 13. Export and import

```python
mesh.export("part.stl")              # streamed binary STL, any size
mesh.export("part.3mf")              # streamed 3MF: one object, millimetres, any size
mesh.export("part.obj")              # via trimesh: .obj, .ply, .off, .glb
m = lg.LatticeMesh.load("part.stl")  # welded on load (binary STL and 3MF read directly)
tm = mesh.to_trimesh()               # for anything trimesh can do (viewing, boolean ops, ...)
field.save("octet_N96.npz"); field = lg.UnitField.load("octet_N96.npz")
```

Geometry helpers: `mesh.scaled(f)`, `mesh.translated(v)`, `mesh.flipped()`,
`mesh.weld()`, `mesh.copy()`.

## 14. Command-line interface

```bash
python -m latticegen list
python -m latticegen build octet --a 2 --rho 0.3 --cells 5 5 5 --smooth 0.02 --check -o octet.stl
python -m latticegen build gyroid --sheet --a 3 --rho 0.25 --cells 4 4 4 -o gyroid_sheet.stl
python -m latticegen build bcc --graded 0.15 0.40 --a 2 --cells 4 4 8 -o bcc_graded.stl
python -m latticegen build octet --a 2 --rho 0.3 --cells 8 8 8 --instanced --seam-spacing 0.17 -o lean.stl
python -m latticegen build octet --method fourier --node-comp 0.5 --a 2 --cells 3 3 3 --edge-length 0.05 -o f.stl
python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 20 20 20 --check -o cube.stl
python -m latticegen build octet --a 2 --rho 0.3 --cells 8 8 8 --instanced --seam-spacing 0.17 --remesh decimate -o leaner.stl
python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 10 10 10 --plates both --plate-thickness 1 -o test.3mf
python -m latticegen simplify part.stl --edge-length 0.05 -o part_lean.stl
python -m latticegen check part_lean.stl --pixel 0.05 --layer 0.05
```

`build` options: `--method {auto,sdf,fourier,tpms,struts}` (auto picks tpms
for TPMS names, sdf otherwise; `struts` builds explicit prisms and hull
nodes, section 9), `--a` or `--cell A B C ALPHA BETA GAMMA`, `--N`
(default from `--pixel` via `recommend_grid`), `--rho`, `--graded LO HI`,
`--cells NX NY NZ`, `--smooth` (fraction of `a`), `--node-comp`, `--sheet`,
`--instanced`, `--seam-spacing` (mm), `--edge-length` (mm: decimation target, or the
interior remesh target with `--instanced`), `--remesh {isotropic,decimate}`
(interior treatment with `--instanced`), `--sides` and `--node {hull,round}`
(with `--method struts`; `--node` is `union` by default, or `hull` / `round`),
`--plates {none,bottom,top,both}`, `--plate-thickness` (mm), `--plate-size` (plate side over
block side, >= 1, default 1.2), `--repair`, `--remove-islands` (drop floating slivers,
`mesh.remove_islands()` defaults), `--check`, `--pixel`, `--layer`, `-o`, `-v`. If the package is
`pip install`ed, `latticegen` is also available as a console command.

The same build is available from Python as `latticegen.app.build(topology, **options)`
(keyword names as the CLI flags, `log=` for progress lines); the CLI and the browser
app in `webapp/` both call it.

## 15. Recipes

**Choose N for a printer.** `N = lg.recommend_grid(cell, pixel_mm)`; for a
2 mm cell and 50 um pixels that is 80. Above ~128 the field arrays (N^3
float32) and the meshes grow quickly. The CLI and the GUIs default to
`latticegen.app.default_grid`: half a pixel but at least 80 for blocks, and
a quarter pixel for `--instanced`. Only the unit cell is meshed there, and a
finely traced cell remeshes better (1 mm cell, 0.05 mm pixel: N = 80).
For the seams, 0.02 mm (50 points along a 1 mm cell edge) works well; the GUI
asks for seam points per cell edge.

**A strut radius you specify.** With a distance field `r = -threshold`, so
invert numerically:

```python
from scipy.optimize import brentq
target_r = 0.20                               # mm
rho = brentq(lambda p: f.strut_radius(p) - target_r, 0.02, 0.9)
```

**Density from a stiffness target.** Gibson-Ashby scaling for a
stretching-dominated lattice (octet): `E/E_s ~ 0.3 rho`; bending-dominated
(bcc, kelvin): `E/E_s ~ rho^2`. Use these to get a starting `rho`, then
check strut sizes with the report.

**Uniform strut section for pyrolysis.** Distance field with `smooth=0`
(node/strut radius 1.43) or Fourier with `node_comp=0.5` (1.44); avoid
`smooth` above ~0.2 r.

**Skin on the outside.** Mesh the block, then union a thin shell in your
CAD/slicer, or make the outer cells denser with a per-cell `rho` array.

**Sandwich panel.** `cells=(nx, ny, 1)` with the caps as face sheets is not
a sandwich (caps are only where the lattice is solid); instead grade with
`rho(zeta)` going to ~0.9 at both ends.

**Reproducible large jobs.** Save fields with `field.save`, build with
`mesh_instanced` where the block is uniform, decimate the rest, and keep
`verbose=True` to log densities and face counts.

**A 2 cm cube of 1 mm octet cells for DLP.** 8000 cells with 0.13 mm struts:
build it as explicit struts, never as a field.

```python
cell = lg.UnitCell.cubic(1.0)
cube = lg.mesh_struts(cell, "octet", rho=0.20, cells=(20, 20, 20), sides=6)
print(lg.format_report(lg.dlp_report(cube, pixel_mm=0.05, layer_mm=0.05)))
cube.export("octet_2cm.stl")            # 5.4 M faces; sides=4 gives 3.8 M
```

If the slicer still struggles, `sides=4`, or split the cube into two
`(20, 20, 10)` halves and print them as separate parts.

## 16. Performance and memory

Measured on 8 cores, `N = 96`, 2 mm octet cell:

| step | time |
|---|---|
| Fourier field | < 0.1 s |
| distance field, sharp | 3.4 s |
| distance field, `smooth=0.02` | 4.6 s |
| `mesh_stack` 3 x 3 x 3 (4.5 M faces) | 3.4 s |
| `mesh_instanced` 6 x 6 x 6, `seam_spacing=0.17` (0.7 M faces) | 18 s |
| `simplify(edge_length=0.05)` on 4.5 M faces | 95 s |
| `mesh_instanced` 3 x 3 x 3, 1 mm octet, `N = 80`, seams 0.05 mm, `remesh="decimate"` (36 k faces) | 6 s |
| `mesh_struts` 20 x 20 x 20, 1 mm octet (5.4 M faces, 270 MB STL) | 5 s |

Memory rules of thumb:

* Marching cubes output: ~40 k vertices per cell at `N = 64` on an octet at
  30 %, ~90 k at `N = 96`; times 12 bytes for vertices and 48 for faces.
  A 6 x 6 x 6 block at `N = 96` (20 M faces) needs several GB; use
  `mesh_instanced` for uniform blocks and keep `cells` modest for graded
  ones on machines with 8 GB or less.
* `slab_cells=1` keeps the working field array to `(nx N)(ny N)(N) x 4`
  bytes; the assembled mesh dominates anyway.
* pymeshlab decimation needs roughly 1 GB per 5 M input faces.
* `mesh_struts` holds only the rings and hull triangles (~660 faces per
  octet cell): building and exporting a 20 x 20 x 20 block peaks at about
  0.6 GB. `mesh.check()` (behind `mesh.report()` and `dlp_report()`)
  streams over the faces and sorts the edges once: about 4 s and 0.5 GB
  extra on 5 M faces. Pass the result on (`mesh.report(c)`,
  `dlp_report(..., check=c)`) instead of computing it twice.

## 17. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ValueError: N=.. too small for hmax=..` | Fourier cutoff needs a finer grid: raise `N` or lower `gmax`. |
| `rho must lie strictly inside (0, 1)` | a callable returned 0 or 1 somewhere: clip it. |
| `RuntimeWarning: ... disconnected components` | floating blobs (Fourier ringing, low density) or incompatible cell interfaces. Run `field.connectivity_check(rho)`, raise `sigma`, lower `node_comp`, increase `rho`, or change the topology pair. `mesh.remove_islands()` deletes existing debris. |
| density of a graded block differs from the mean of `rho` | expected: the quantile mapping is cell-averaged and the caps cut struts; differences of ~1 % are normal. |
| `simplify` warns about open edges | the target was too aggressive for `preserve_topology`; use a larger face budget or `mesh.repair()`. |
| the instanced mesh has a slightly different density from `mesh_stack` | both are calibrated to `rho` within 2e-4; if `density_correct=False` the simplified seams lose `O(h^2)`. |
| `ImportError: this operation needs pymeshlab` | `conda install -c conda-forge pymeshlab`. Raw instancing (`remesh=None`) and everything else work without it. |
| slicer says the STL is not manifold | check `mesh.is_watertight`; if `True`, the slicer objects to the mesh size or to coincident triangles at flat caps: run `simplify` and re-export. |
| struts look faceted in the slicer preview | voxel coarser than the pixel; use `recommend_grid`. |
| Kelvin / small-strut Fourier fields look blobby | short struts need larger `sigma` relative to the cutoff; the defaults (`0.07 a`, `gmax 5.5`) are blob-free from 10 % to 40 %, tune from there. |

## 18. API summary

```
latticegen
├── UnitCell(A) .cubic .tetragonal .orthorhombic .hexagonal .from_parameters .from_matrix
│     .A .volume .lengths .a_min .is_cubic .reciprocal() .frac_to_cart() .cart_to_frac() .scaled()
├── Skeleton(name, segments, points) .custom .from_nodes_and_bonds  +  (merge)
│     .segments .points .n_struts .nodes() .midpoints() .connectivity() .strut_lengths() .total_length()
├── skeleton(name), available_topologies(), topologies.{sc,bcc,fcc,bccz,fccz,fbcc,octet,diamond,kelvin}
├── available_tpms(), tpms.tpms_on_grid(name, N, periods, sheet, phase)
├── fourier_field(cell, topology, N, sigma, gmax, node_comp, node_sigma, weights)
├── sdf_field(cell, topology, N, smooth, block, cull)
├── tpms_field(cell, name, N, sheet, periods, phase)
├── custom_field(cell, array, name)
├── UnitField .N .A .cell .voxel_size .f .name .method .skeleton .meta
│     .threshold() .calibrated_threshold() .density() .solid_mask() .strut_radius()
│     .feature_sizes() .connectivity_check() .phase_components() .mesh() .mesh_instanced() .save() .load()
├── mesh_stack(fields, rho, cells, cell_map, slab_cells, exact_density, threshold, verbose)
├── mesh_instanced(field, rho, cells, seam_spacing, edge_length, remesh, density_correct, exact_density, anisotropy, quality_thr, verbose)
├── mesh_struts(cell, topology, rho, radius, cells, sides, node, round_points, verbose), radius_for_density(cell, topology, rho, sides, node)
├── struts.{bulk_volume_coefficients, polygon_area}
├── LatticeMesh(verts, faces, info)
│     .n_vertices .n_faces .bounds .extents .volume .area .is_watertight .euler_number .relative_density
│     .edge_lengths() .components() .check() .report()
│     .scaled() .translated() .flipped() .weld() .copy()
│     .decimate() .simplify() .remesh() .repair() .remove_islands()
│     .to_trimesh() .from_trimesh() .load() .export()
├── decimate, simplify, remesh_isotropic, repair, remove_islands   (functions in latticegen.remesh)
├── add_plates(mesh, cell, cells, where, thickness, size, planes), set_union_backend(fn)
├── dlp_report(mesh, field, rho, pixel_mm, layer_mm, min_feature_px, min_pore_mm, resin_density_g_cm3, shrinkage_pct)
├── format_report(rep), recommend_grid(cell, pixel_mm, oversample)
├── fourier.{reciprocal_basis, enumerate_G, motif_coefficients, field_on_grid, evaluate, compensate_nodes, fourier_coefficients}
├── sdf.{periodic_segment_images, distance_field, sdf_on_grid, radius_for_density}
└── meshops.{weld, is_watertight, open_edges, mesh_volume, mesh_area, connected_components, write_stl, read_stl, write_3mf, read_3mf}
```
