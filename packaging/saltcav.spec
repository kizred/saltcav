# -*- mode: python ; coding: utf-8 -*-
# Сборка отдельного приложения saltcav (PyInstaller, одна папка):
#   pyinstaller packaging/saltcav.spec --noconfirm
# Результат — dist/saltcav/: saltcav-gui(.exe) — окно, saltcav(.exe) — командная строка,
# примеры (configs, data, examples) и все библиотеки (Qt, VTK, Gmsh, MKL) внутри.
# Установщик Windows — packaging/saltcav.iss (Inno Setup) поверх этой папки.

import glob
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 — задаёт PyInstaller
sys.path.insert(0, str(ROOT / "src"))
from saltcav import __version__  # noqa: E402

WIN = sys.platform == "win32"
ICON = str(ROOT / "src" / "saltcav" / "gui" / "icons" / ("saltcav.ico" if WIN else "saltcav.png"))

# --- нативные библиотеки вне site-packages: Gmsh и Intel MKL (PARDISO) ------------------
prefix = Path(sys.prefix)
lib_dirs = [prefix / "Library" / "bin", prefix / "Lib", prefix / "lib", prefix / "bin"]
binaries = []


def add(pattern: str, dest: str = ".") -> int:
    n = 0
    for d in lib_dirs:
        for f in glob.glob(str(d / pattern)):
            if os.path.isfile(f):
                binaries.append((f, dest))
                n += 1
    return n


if add("gmsh-*.dll" if WIN else "libgmsh.so*") == 0:
    raise SystemExit("не найдена библиотека Gmsh (pip install gmsh)")
# MKL: только то, что нужно PARDISO (без SYCL, ScaLAPACK, кластерных и т. п.)
MKL = ["mkl_rt", "mkl_core", "mkl_intel_thread", "mkl_gnu_thread", "mkl_sequential",
       "mkl_def", "mkl_avx2", "mkl_avx512", "mkl_mc3", "mkl_intel_lp64"]
for name in MKL:
    add(f"{name}*.dll" if WIN else f"lib{name}.so*")
add("libiomp5md.dll" if WIN else "libiomp5.so")

# --- данные --------------------------------------------------------------------------------
datas = collect_data_files("saltcav")  # значки, руководство
for rel in ("configs", "data/demo", "data/open", "examples", "schema"):
    datas.append((str(ROOT / rel), rel))
datas.append((str(ROOT / "benchmarks" / "sansmic" / "reference_direct.npz"), "benchmarks/sansmic"))

hidden = (
    collect_submodules("saltcav")
    + collect_submodules("pypardiso")
    + ["sklearn.gaussian_process", "emcee", "matplotlib.backends.backend_qtagg",
       "matplotlib.backends.backend_agg", "pyvistaqt"]
)
EXCLUDES = ["mypy", "mypyc", "tkinter", "IPython", "jupyter", "notebook", "pytest", "numba", "sphinx",
            "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "PyQt5", "PyQt6"]


# VTK целиком — 550 МБ; не нужны: ускорители VTK-m/Viskores, тесты VTK, GTK (Linux)
DROP = ("viskores", "vtkAcceleratorsVTKm", "vtkTesting", "libvtkTesting", "libgtk", "libgdk",
        "libpango", "libcairo", "libatk", "libgio-", "libharfbuzz")


def trim(entries):
    return [e for e in entries if not any(d in os.path.basename(e[0]) for d in DROP)]


def analysis(script: str):
    a = Analysis(  # noqa: F821
        [str(ROOT / "packaging" / script)],
        pathex=[str(ROOT / "src")],
        binaries=binaries,
        datas=datas,
        hiddenimports=hidden,
        runtime_hooks=[str(ROOT / "packaging" / "rthook_saltcav.py")],
        excludes=EXCLUDES,
        noarchive=False,
    )
    a.binaries = trim(a.binaries)
    a.datas = trim(a.datas)
    return a


# версия и автор в свойствах файла (Windows)
version_file = None
if WIN:
    nums = tuple(int(x) for x in __version__.split(".")[:3]) + (0,)
    vf = ROOT / "build" / "version_info.txt"
    vf.parent.mkdir(exist_ok=True)
    vf.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}),
  kids=[StringFileInfo([StringTable('041904B0', [
    StringStruct('CompanyName', 'Sergei Kizred'),
    StringStruct('FileDescription', 'saltcav — моделирование соляных камер'),
    StringStruct('FileVersion', '{__version__}'),
    StringStruct('LegalCopyright', '© Sergei Kizred'),
    StringStruct('ProductName', 'saltcav'),
    StringStruct('ProductVersion', '{__version__}')])]),
    VarFileInfo([VarStruct('Translation', [1049, 1200])])]
)""", encoding="utf-8")
    version_file = str(vf)

a_gui = analysis("saltcav_gui.py")
a_cli = analysis("saltcav_cli.py")
exe_gui = EXE(  # noqa: F821
    PYZ(a_gui.pure),  # noqa: F821
    a_gui.scripts, [], exclude_binaries=True, name="saltcav-gui", icon=ICON,
    console=False, version=version_file, upx=False,
)
exe_cli = EXE(  # noqa: F821
    PYZ(a_cli.pure),  # noqa: F821
    a_cli.scripts, [], exclude_binaries=True, name="saltcav", icon=ICON,
    console=True, version=version_file, upx=False,
)
coll = COLLECT(  # noqa: F821
    exe_gui, a_gui.binaries, a_gui.datas,
    exe_cli, a_cli.binaries, a_cli.datas,
    name="saltcav", upx=False,
)
