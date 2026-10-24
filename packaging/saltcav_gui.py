"""Точка входа окна в собранном приложении (PyInstaller)."""

import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # дочерние процессы ансамбля — тем же .exe
    from saltcav.workspace import ensure_workspace, frozen

    if frozen():
        os.chdir(ensure_workspace())  # «Документы\saltcav»: примеры, расчёты
    from saltcav.gui.app import main

    sys.exit(main(sys.argv[1:]))
