"""Где лежат примеры и куда пишет программа.

Из исходников (pip install -e .) примеры — в папке проекта. В собранном приложении
(PyInstaller, установщик) они лежат рядом с программой только для чтения; при первом
запуске копируются в «Документы\\saltcav» — там их можно править, туда же по умолчанию
сохраняются расчёты.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

EXAMPLES = ("configs", "data/demo", "data/open", "examples", "schema")


def frozen() -> bool:
    """Запущено собранное приложение (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def bundled_root() -> Path:
    """Корень с примерами: папка приложения или корень проекта."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def documents_dir() -> Path:
    """Папка «Документы» пользователя (Windows — с учётом переноса в OneDrive)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5: «Документы», как их видит пользователь
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # type: ignore[attr-defined]
                return Path(buf.value)
        except (AttributeError, OSError):  # pragma: no cover
            pass
    return Path.home() / "Documents"


def workspace_dir() -> Path:
    """Рабочая папка пользователя для собранного приложения."""
    env = os.environ.get("SALTCAV_HOME")
    return Path(env) if env else documents_dir() / "saltcav"


def ensure_workspace(target: Path | None = None) -> Path:
    """Скопировать примеры в рабочую папку (недостающие файлы; свои правки не
    затираются) и вернуть её путь."""
    dst = target or workspace_dir()
    src = bundled_root()
    dst.mkdir(parents=True, exist_ok=True)
    for rel in EXAMPLES:
        s = src / rel
        if not s.exists():
            continue
        for f in s.rglob("*") if s.is_dir() else [s]:
            if f.is_dir() or "__pycache__" in f.parts:
                continue
            d = dst / f.relative_to(src)
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d)
    return dst


def sansmic_reference() -> Path:
    return bundled_root() / "benchmarks" / "sansmic" / "reference_direct.npz"
