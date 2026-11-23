"""Проверка поставки библиотеки в установленном wheel."""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_and_loads_materials_library(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
        cwd=ROOT,
        check=True,
    )
    wheel = next(dist.glob("*.whl"))
    library_root = ROOT / "data" / "materials_library"
    expected = {
        f"saltcav/_materials_library/{path.relative_to(library_root).as_posix()}"
        for path in library_root.rglob("*.yaml")
    }
    with zipfile.ZipFile(wheel) as archive:
        assert expected
        assert expected <= set(archive.namelist())

    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        cwd=tmp_path,
        check=True,
    )
    code = """
import sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import saltcav
from saltcav.materials import load_library
from saltcav.workspace import bundled_materials_library
assert Path(saltcav.__file__).resolve().is_relative_to(target)
root = bundled_materials_library().resolve()
assert root == target / 'saltcav' / '_materials_library'
library = load_library(root)
assert library.materials and library.fluids
"""
    subprocess.run([sys.executable, "-c", code, str(target)], cwd=tmp_path, check=True)
