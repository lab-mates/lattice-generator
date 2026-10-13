"""A 2 cm cube of 1 mm octet cells for DLP printing, built as explicit
prisms and hull nodes: no field, no marching cubes, ~660 triangles per
cell, seconds and a few hundred MB of RAM for 8000 cells."""
import time

import latticegen as lg

cell = lg.UnitCell.cubic(1.0)                      # 1 mm cell
rho, cells = 0.20, (20, 20, 20)                    # bulk density, 2 cm edge

t0 = time.time()
mesh = lg.mesh_struts(cell, "octet", rho=rho, cells=cells, sides=6, verbose=True)
print(f"built in {time.time() - t0:.1f} s")
print(mesh.report())
# strut / node sizes are known exactly from the construction; no field needed
print(lg.format_report(lg.dlp_report(mesh, pixel_mm=0.05, layer_mm=0.05)))
mesh.export("octet_20x20x20_struts.stl")           # ~270 MB binary STL

# variants: square struts (fewer triangles), rounded nodes, explicit radius
square = lg.mesh_struts(cell, "octet", rho=rho, cells=(3, 3, 3), sides=4)
rounded = lg.mesh_struts(cell, "octet", rho=rho, cells=(3, 3, 3), node="round")
fixed_r = lg.mesh_struts(cell, "bcc", radius=0.1, cells=(3, 3, 3))
for name, m in (("square", square), ("rounded", rounded), ("bcc r=0.1", fixed_r)):
    print(f"{name:10s} {m.n_faces:8,d} faces  r={m.info['strut_radius']:.4f} mm  "
          f"node R={m.info['node_radius']:.4f} mm  watertight {m.is_watertight}")
