Command line
============

Everything :func:`latticegen.app.build` does is also available from the
shell as ``python -m latticegen`` (or ``latticegen`` when installed).

.. code-block:: bash

   python -m latticegen list
   python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 20 20 20 \
          --sides 12 --check -o octet_2cm.stl
   python -m latticegen build gyroid --a 2 --rho 0.25 --sheet --cells 3 3 3 \
          --edge-length 0.05 -o gyroid.stl
   python -m latticegen build octet --method struts --a 1 --rho 0.2 --cells 10 10 10 \
          --plates both --plate-thickness 1 -o octet_plates.3mf
   python -m latticegen check octet_2cm.stl --pixel 0.05

``python -m latticegen --help``
-------------------------------

.. literalinclude:: _generated/cli_main.txt
   :language: text

``build``
---------

.. literalinclude:: _generated/cli_build.txt
   :language: text

``check``
---------

.. literalinclude:: _generated/cli_check.txt
   :language: text

``simplify``
------------

.. literalinclude:: _generated/cli_simplify.txt
   :language: text
