"""Density gradients and mixed topologies in one block."""
import numpy as np
import latticegen as lg

cell = lg.UnitCell.cubic(2.0)
octet = lg.sdf_field(cell, "octet", N=64, smooth=0.04)
gyroid = lg.tpms_field(cell, "gyroid", N=64)

# a) linear density gradient along z: 15 % at the bottom, 40 % at the top
graded = lg.mesh_stack(octet, rho=lambda zeta: 0.15 + 0.25 * zeta, cells=(3, 3, 6))
graded.export("octet_graded_z.stl")

# b) radial gradient in cartesian coordinates (mm): denser near the block axis
cells = (4, 4, 3)
cx = cy = 0.5 * 2.0 * 4
rho_r = lambda x, y, z: np.clip(0.45 - 0.25 * np.hypot(x - cx, y - cy) / cx, 0.12, 0.6)
radial = lg.mesh_stack(octet, rho=rho_r, cells=cells)
radial.export("octet_radial.stl")

# c) per-cell density array + per-cell topology map: dense octet shell around
#    a lighter fcc core.  Topologies that share a cell can only connect where
#    both are solid on the common face -- fcc struts are a subset of the
#    octet's, so the interface is seamless.  (An octet shell around a gyroid
#    core, for instance, comes out as TWO disconnected solids.)
cells = (4, 4, 4)
fcc = lg.sdf_field(cell, "fcc", N=64, smooth=0.04)
cmap = np.zeros(cells, int)
cmap[1:-1, 1:-1, 1:-1] = 1                    # index into [octet, fcc]
rho = np.full(cells, 0.30)
rho[1:-1, 1:-1, 1:-1] = 0.20
mixed = lg.mesh_stack([octet, fcc], rho=rho, cells=cells, cell_map=cmap)
print(mixed.report())                          # check: components 1
mixed.export("octet_shell_fcc_core.stl")

# gyroid core inside a gyroid shell is fine too (same topology, different rho)
same = lg.mesh_stack(gyroid, rho=rho, cells=cells)
print("gyroid graded shell/core components:", same.components()[1])
