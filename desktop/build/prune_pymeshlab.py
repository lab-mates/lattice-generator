"""Which pymeshlab files the desktop app needs (Windows wheel layout).

latticegen only calls filters from four MeshLab plugins (filter_meshing,
filter_select, filter_clean, filter_func).  Starting from those and the Python
binding, follow the DLL import tables (pefile, a PyInstaller dependency) and
keep that closure; everything else in the wheel (Embree, 3MF/E57/U3D, Qt image
and platform plugins, software OpenGL, ~55 other MeshLab plugins, .lib files,
tests) is left out.

    python prune_pymeshlab.py      # prints the kept files and the saving
"""

import importlib.util
import os

PLUGINS = ("filter_meshing", "filter_select", "filter_clean", "filter_func")
SKIP_DIRS = ("tests", "translations", "styles", "bearer", "iconengines", "imageformats",
             "platforms")


def pymeshlab_dir():
    return importlib.util.find_spec("pymeshlab").submodule_search_locations[0]


def _imports(path):
    import pefile
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                                           pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    names = [e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
    names += [e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])]
    pe.close()
    return names


def keep_set(pkg=None):
    """Paths relative to the pymeshlab package (forward slashes) to bundle."""
    pkg = pkg or pymeshlab_dir()
    local = {f.lower(): f for f in os.listdir(pkg) if f.lower().endswith(".dll")}
    roots = [f for f in os.listdir(pkg) if f.startswith("pmeshlab") and f.endswith(".pyd")]
    roots += [f"lib/plugins/{p}.dll" for p in PLUGINS]
    keep, todo = set(), list(roots)
    while todo:
        rel = todo.pop()
        if rel in keep:
            continue
        keep.add(rel)
        for dll in _imports(os.path.join(pkg, rel)):
            if dll.lower() in local:
                todo.append(local[dll.lower()])
    # small non-binary files the package reads (keys.txt, setup.cfg, ...)
    for f in os.listdir(pkg):
        full = os.path.join(pkg, f)
        if os.path.isfile(full) and not f.lower().endswith((".dll", ".pyd", ".lib", ".exe")):
            keep.add(f)
    return keep


def filter_toc(toc, keep):
    """Drop pymeshlab entries not in ``keep`` from a PyInstaller TOC list."""
    out = []
    for entry in toc:
        dest = entry[0].replace("\\", "/")
        if dest.startswith("pymeshlab/"):
            rel = dest[len("pymeshlab/"):]
            if rel.split("/")[0] in SKIP_DIRS or rel not in keep:
                continue
        out.append(entry)
    return out


if __name__ == "__main__":
    pkg = pymeshlab_dir()
    keep = keep_set(pkg)
    total = kept = 0
    for root, _, files in os.walk(pkg):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, pkg).replace("\\", "/")
            size = os.path.getsize(full)
            total += size
            if rel in keep:
                kept += size
    for rel in sorted(keep):
        print("  keep", rel)
    print(f"pymeshlab: {total / 1e6:.0f} MB in the wheel, {kept / 1e6:.0f} MB kept")
