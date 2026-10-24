"""Runtime-hook собранного приложения: библиотеки MKL и Gmsh лежат в папке программы."""

import glob
import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
if not os.environ.get("PYPARDISO_MKL_RT"):
    _pat = "mkl_rt*.dll" if os.name == "nt" else "libmkl_rt.so*"
    _cands = sorted(glob.glob(os.path.join(_base, _pat)))
    if _cands:
        os.environ["PYPARDISO_MKL_RT"] = _cands[0]
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_base)  # зависимые DLL MKL (mkl_core, libiomp5md) — рядом
if sys.platform.startswith("linux"):
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
