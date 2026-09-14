#!/usr/bin/env bash
# The same things from the shell.
set -e
python -m latticegen list
python -m latticegen build octet --a 2 --rho 0.3 --cells 4 4 4 --smooth 0.02 --check -o octet_cli.stl
python -m latticegen build gyroid --sheet --a 3 --rho 0.25 --cells 3 3 3 -o gyroid_sheet.stl
python -m latticegen build bcc --graded 0.15 0.40 --a 2 --cells 3 3 6 -o bcc_graded.stl
python -m latticegen build octet --a 2 --rho 0.3 --cells 6 6 6 --instanced --seam-spacing 8 -o octet_lean.stl
python -m latticegen simplify octet_cli.stl --edge-length 0.05 -o octet_cli_lean.stl
python -m latticegen check octet_cli_lean.stl --pixel 0.05 --layer 0.05
