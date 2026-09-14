"""Your own strut topology, non-cubic cells, combined skeletons."""
import numpy as np
import latticegen as lg

# a) a re-entrant "cross" cell: struts from the body centre to the face centres
segments = [([.5, .5, .5], [1, .5, .5]), ([.5, .5, .5], [.5, 1, .5]), ([.5, .5, .5], [.5, .5, 1]),
            ([.5, .5, .5], [0, .5, .5]), ([.5, .5, .5], [.5, 0, .5]), ([.5, .5, .5], [.5, .5, 0])]
cross = lg.Skeleton.custom(segments, name="cross")
print(cross)                                   # duplicates modulo translation are merged

# b) combine built-ins with '+' : bcc struts plus sc edges
bcc_sc = lg.skeleton("bcc") + lg.skeleton("sc")
print(bcc_sc)

# c) tetragonal cell: 2 mm x 2 mm x 3 mm
cell = lg.UnitCell.tetragonal(2.0, 3.0)
field = lg.sdf_field(cell, bcc_sc, N=64, smooth=0.05)
mesh = lg.mesh_stack(field, rho=0.25, cells=(4, 4, 2))
print(mesh.report())
mesh.export("bcc_sc_tetragonal.stl")

# d) Fourier field of the custom skeleton with hand-picked smearing.  Fourier
#    fields are truncated series: too small a sigma for the cutoff, or strong
#    node compensation, rings and can leave tiny floating blobs.  Always check.
f2 = lg.fourier_field(1.5, cross, N=64, sigma=0.18, gmax=4.5, node_comp=0.3)
print(f2.connectivity_check(0.3))              # floating_solids must be 0
print({k: round(v, 3) for k, v in f2.feature_sizes(0.3).items()})
