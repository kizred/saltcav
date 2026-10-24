"""Точка входа окна в собранном приложении (PyInstaller)."""

import faulthandler
import multiprocessing
import os
import sys

if __name__ == "__main__":
    # у оконного .exe нет консоли (sys.stdout = None) — ни у окна, ни у процессов
    # расчёта; вывод библиотек и сбои — в «Документы\saltcav\saltcav-gui.log»
    if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
        from saltcav.workspace import workspace_dir

        child = any(a.startswith("--multiprocessing") for a in sys.argv)
        try:
            workspace_dir().mkdir(parents=True, exist_ok=True)
            log = open(workspace_dir() / "saltcav-gui.log", "a" if child else "w",
                       encoding="utf-8", buffering=1)  # fmt: skip
        except OSError:
            log = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
        faulthandler.enable(log)
    multiprocessing.freeze_support()  # дочерние процессы расчёта — тем же .exe
    from saltcav.workspace import ensure_workspace, frozen

    if frozen():
        os.chdir(ensure_workspace())  # «Документы\saltcav»: примеры, расчёты
    from saltcav.gui.app import main

    sys.exit(main(sys.argv[1:]))
