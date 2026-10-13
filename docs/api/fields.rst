Fields
======

.. currentmodule:: latticegen

A field-based lattice is the level set ``f >= t`` of a periodic scalar
field sampled on an ``N x N x N`` grid over one cell. The threshold ``t``
sets the density, so one field serves every density, uniform or graded.

Builders
--------

.. autofunction:: sdf_field

.. autofunction:: fourier_field

.. autofunction:: tpms_field

.. autofunction:: custom_field

UnitField
---------

.. autoclass:: UnitField
   :members:
   :exclude-members: cell, f, name, method, skeleton, meta
