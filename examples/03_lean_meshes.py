"""Getting the triangle count down: instancing and simplification."""
import time
import latticegen as lg

cell = lg.UnitCell.cubic(2.0)
field = lg.sdf_field(cell, "octet", N=64, smooth=0.04)
rho, cells = 0.30, (6, 6, 6)

# A) uniform block by unit-cell instancing with seam-consistent simplification.
#    Memory and time do not depend on the block size.  seam_spacing is in
#    voxels: N/12 = 8 voxels ~ 0.17 mm here.
t0 = time.time()
lean = lg.mesh_instanced(field, rho=rho, cells=cells, seam_spacing=field.N / 12)
print(f"instanced: {lean}  {time.time() - t0:.1f} s")
print(lean.report())
lean.export("octet_6x6x6_instanced.stl")

# B) marching cubes on the full block (works for graded blocks too), then
#    quadric decimation to a target edge length ~ 1 printer pixel.
#    (3x3x3 here: a full 6x6x6 block at N=64 is ~11 M faces and needs
#    several GB of RAM to decimate -- that is what instancing is for.)
t0 = time.time()
full = lg.mesh_stack(field, rho=rho, cells=(3, 3, 3))
print(f"full block: {full}  {time.time() - t0:.1f} s")
simple = full.simplify(edge_length=0.05, verbose=True)
simple.export("octet_3x3x3_simplified.stl")

# C) isotropic remesh (uniform triangles, e.g. for FEA)
iso = lean.remesh(edge_length=0.2, iterations=6, verbose=True)
iso.export("octet_6x6x6_isotropic.stl")
