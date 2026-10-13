latticegen
==========

Periodic lattice structures for 3D printing (DLP / resin), as one Python
package. This is the API reference, built from the docstrings. The ideas
behind the methods are in ``docs/fundamentals.md`` and a task-oriented
guide with recipes is in ``docs/usage.md``.

.. code-block:: text

   UnitCell  →  Skeleton or TPMS  →  UnitField  →  LatticeMesh  →  STL

A lattice is either built **explicitly** from its beams
(:func:`~latticegen.mesh_struts`), or from a periodic scalar **field**
sampled over one unit cell, whose level set is meshed
(:func:`~latticegen.mesh_stack`, :func:`~latticegen.mesh_instanced`).
All lengths are in the unit of the cell; use millimetres.

Install
-------

.. code-block:: bash

   cd lattice-generator
   pip install -e .

Dependencies: numpy, scipy, scikit-image, trimesh, manifold3d (beam-sized
joints). ``pymeshlab`` is needed only for simplification, remeshing,
repair and the remeshing steps of instancing.

Quick start
-----------

.. code-block:: python

   import latticegen as lg

   cell = lg.UnitCell.cubic(1.0)                                  # 1 mm cells

   # explicit beams: thousands of cells in seconds
   cube = lg.mesh_struts(cell, "octet", rho=0.20, cells=(20, 20, 20), sides=12)
   cube.export("octet_2cm.stl")

   # field-based: rounded joints, graded density, TPMS
   field = lg.sdf_field(lg.UnitCell.cubic(2.0), "bcc", N=80, smooth=0.04)
   graded = lg.mesh_stack(field, rho=lambda z: 0.15 + 0.25 * z, cells=(2, 2, 4))
   graded = graded.simplify(edge_length=0.05)                     # needs pymeshlab

   print(lg.format_report(lg.dlp_report(cube, pixel_mm=0.05, layer_mm=0.05)))

Or in one call with the options of the GUIs and the command line,
:func:`latticegen.app.build`.

Which mesher?
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Use it for
     - Cost
   * - :func:`~latticegen.mesh_struts`
     - uniform beam lattices, large blocks
     - ~1300 triangles per octet cell (6 sides, beam-sized joints)
   * - :func:`~latticegen.mesh_stack`
     - graded or mixed blocks, any field
     - ~15 N² triangles per cell
   * - :func:`~latticegen.mesh_instanced`
     - uniform field lattices with lean meshes
     - independent of the block size

.. toctree::
   :maxdepth: 2
   :caption: Reference

   geometry
   fields
   meshing
   mesh
   processing
   printing
   app
   cli
   lowlevel

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
