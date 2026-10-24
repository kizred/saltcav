"""Точка входа окна в собранном приложении (PyInstaller)."""

import faulthandler
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # дочерние процессы ансамбля — тем же .exe
    from saltcav.workspace import ensure_workspace, frozen

    if frozen():
        ws = ensure_workspace()  # «Документы\saltcav»: примеры, расчёты
        os.chdir(ws)
        # у оконного .exe нет консоли (sys.stdout = None): вывод библиотек и сбои — в журнал
        if sys.stdout is None or sys.stderr is None:
            log = open(ws / "saltcav-gui.log", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
            sys.stdout = sys.stdout or log
            sys.stderr = sys.stderr or log
            faulthandler.enable(log)
    from saltcav.gui.app import main

    sys.exit(main(sys.argv[1:]))
