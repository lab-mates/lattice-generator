Plates
======

.. currentmodule:: latticegen

Solid plates welded to the bottom and/or top of a block, e.g. as loading
platens for compression tests. The plated block is one closed solid, so
an STL or 3MF export holds a single object.

.. code-block:: python

   import latticegen as lg

   cell = lg.UnitCell.cubic(1.0)
   cube = lg.mesh_struts(cell, "octet", rho=0.20, cells=(10, 10, 10))
   cube = lg.add_plates(cube, cell, (10, 10, 10), "both", thickness=1.0, size=1.2)
   cube.export("octet_plates.3mf")

.. automodule:: latticegen.plates
   :no-members:

.. autofunction:: add_plates

.. autofunction:: latticegen.plates.plate_corners

.. autofunction:: set_union_backend
