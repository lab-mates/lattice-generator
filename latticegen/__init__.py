"""
latticegen -- periodic lattice structures for 3D printing.

Pipeline:  UnitCell  ->  Skeleton / TPMS  ->  UnitField  ->  LatticeMesh  ->  STL

    import latticegen as lg

    cell  = lg.UnitCell.cubic(2.0)                       # 2 mm cell
    field = lg.sdf_field(cell, "octet", N=96, smooth=0.04)
    mesh  = lg.mesh_stack(field, rho=0.30, cells=(5, 5, 5))
    mesh  = mesh.simplify(edge_length=0.05)
    mesh.export("octet_5x5x5.stl")

    lean  = lg.mesh_struts(cell, "octet", rho=0.30, cells=(20, 20, 20))   # explicit beams
    print(lg.format_report(lg.dlp_report(mesh, field, 0.30, pixel_mm=0.05)))
"""

from .cells import UnitCell
from .topologies import Skeleton, skeleton, available as available_topologies
from .tpms import available as available_tpms
from .field import UnitField, fourier_field, sdf_field, tpms_field, custom_field
from .mesh import LatticeMesh
from .meshing import mesh_stack
from .instanced import mesh_instanced
from .struts import mesh_struts, radius_for_density
from .remesh import decimate, simplify, remesh_isotropic, repair, remove_islands
from .printing import dlp_report, format_report, recommend_grid
from . import topologies, tpms, fourier, sdf, meshops

__version__ = "1.0.0"

__all__ = [
    "UnitCell", "Skeleton", "skeleton", "available_topologies", "available_tpms",
    "UnitField", "fourier_field", "sdf_field", "tpms_field", "custom_field",
    "LatticeMesh", "mesh_stack", "mesh_instanced", "mesh_struts", "radius_for_density",
    "decimate", "simplify", "remesh_isotropic", "repair", "remove_islands",
    "dlp_report", "format_report", "recommend_grid",
    "topologies", "tpms", "fourier", "sdf", "meshops",
]
