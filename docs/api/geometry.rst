Cells and topologies
====================

.. currentmodule:: latticegen

Unit cells
----------

.. autoclass:: UnitCell
   :members:
   :exclude-members: name, A

.. autofunction:: latticegen.cells.as_cell

Strut topologies
----------------

A topology is a :class:`Skeleton`: straight struts in fractional
coordinates, listed once per cell.

==========  ===============  =============  ================================================
Name        Struts per cell  Connectivity   Description
==========  ===============  =============  ================================================
sc          3                6              cell edges
bcc         8                8              corners to the body centre
fcc         12               12 / 4         corners to the face centres
bccz        9                10             bcc plus vertical pillars
fccz        13               14             fcc plus vertical pillars
fbcc        20               20             fcc and bcc combined
octet       24               12             fcc nodes, all nearest-neighbour bonds
diamond     16               4              tetrahedral bonds of the diamond structure
kelvin      24               4              edges of the packing of truncated octahedra
==========  ===============  =============  ================================================

.. autoclass:: Skeleton
   :members:
   :exclude-members: name, segments, points, fourier_defaults

.. autofunction:: latticegen.topologies.skeleton

.. autofunction:: latticegen.topologies.available

.. automodule:: latticegen.topologies
   :members: sc, bcc, fcc, bccz, fccz, fbcc, octet, diamond, kelvin
   :no-index:

Triply periodic minimal surfaces
--------------------------------

TPMS lattices have no skeleton; :func:`tpms_field` builds their field
directly.

.. autofunction:: latticegen.tpms.available
