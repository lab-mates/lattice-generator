"""Quickstart: one octet block, three ways to build the field, STL out."""
import latticegen as lg

cell = lg.UnitCell.cubic(2.0)                 # 2 mm unit cell (all lengths in mm)

# 1) distance field: exactly circular struts, filleted junctions
field = lg.sdf_field(cell, "octet", N=64, smooth=0.04)

# 2) Fourier synthesis: smooth organic struts (junction de-bulking on)
# field = lg.fourier_field(cell, "octet", N=64, node_comp=0.5)

# 3) TPMS gyroid (network solid); sheet=True for a wall-type gyroid
# field = lg.tpms_field(cell, "gyroid", N=64)

mesh = lg.mesh_stack(field, rho=0.30, cells=(3, 3, 3))
print(mesh.report())

# marching cubes over-resolves: bring the triangles down to ~1 pixel (needs pymeshlab)
mesh = mesh.simplify(edge_length=0.05, verbose=True)
mesh.export("octet_3x3x3.stl")

# feature sizes and printability for a 50 um pixel / 50 um layer printer
print(lg.format_report(lg.dlp_report(mesh, field, rho=0.30, pixel_mm=0.05, layer_mm=0.05)))
