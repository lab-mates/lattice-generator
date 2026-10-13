Simplification, remeshing and repair
====================================

.. currentmodule:: latticegen

These functions take and return a :class:`LatticeMesh` and are also
available as methods of it (``mesh.simplify(...)``). They need
``pymeshlab``.

.. autofunction:: simplify

.. autofunction:: decimate

.. autofunction:: remesh_isotropic

.. autofunction:: repair

.. autofunction:: remove_islands

.. autofunction:: latticegen.remesh.target_faces_for_edge_length
