"""Значок программы: saltcav.png (окно), saltcav.ico (ярлык Windows)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any


def icon_path(kind: str = "png") -> Path:
    """Путь к файлу значка (``png`` — 512 px, ``ico`` — набор размеров для Windows)."""
    name = {"png": "saltcav.png", "ico": "saltcav.ico", "png256": "saltcav-256.png"}[kind]
    return Path(str(resources.files("saltcav.gui.icons") / name))


def app_icon() -> Any:
    """QIcon программы (нужен PySide6)."""
    from PySide6 import QtGui

    return QtGui.QIcon(str(icon_path("png")))


def set_windows_app_id() -> None:
    """Windows: свой идентификатор процесса — тогда на панели задач значок saltcav,
    а не Python."""
    import sys

    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Kizre.saltcav")  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover
            pass
