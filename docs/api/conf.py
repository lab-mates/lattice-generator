# Sphinx configuration for the latticegen API reference.
#
#   python -m sphinx -b html docs/api docs/api/_build/html
#
# Needs sphinx and furo (pip install sphinx furo). The reference is built
# from the docstrings (NumPy style), so the package itself must import.

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import latticegen  # noqa: E402

project = "latticegen"
author = "MATES Lab, Universidad de Sevilla-CSIC"
copyright = "2026, MATES Lab, Universidad de Sevilla-CSIC"
release = version = latticegen.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"            # types are in the docstrings
autodoc_default_options = {"members": True, "show-inheritance": False}
add_module_names = False

html_theme = "furo"
html_title = f"latticegen {release}"
html_static_path = []
exclude_patterns = ["_build"]

# The CLI reference shows the real --help text of this version.
_gen = os.path.join(os.path.dirname(__file__), "_generated")
os.makedirs(_gen, exist_ok=True)
for name, args in (("cli_main", []), ("cli_build", ["build"]), ("cli_check", ["check"]),
                   ("cli_simplify", ["simplify"])):
    out = subprocess.run([sys.executable, "-m", "latticegen", *args, "--help"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "COLUMNS": "88"})
    with open(os.path.join(_gen, name + ".txt"), "w") as fh:
        fh.write(out.stdout)
