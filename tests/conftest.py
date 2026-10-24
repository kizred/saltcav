"""Общие настройки тестов."""

import os
import sys

import pytest

_status = {"code": 0}


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _status["code"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Qt (offscreen), VTK и MKL иногда падают в деструкторах при выходе интерпретатора
    (код 139 после успешных тестов). Отчёты уже записаны — выходим без деструкторов."""
    if any(m in sys.modules for m in ("PySide6.QtWidgets", "pypardiso", "pyvista")):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_status["code"])
