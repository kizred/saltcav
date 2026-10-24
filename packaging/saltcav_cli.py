"""Точка входа командной строки в собранном приложении (PyInstaller)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from saltcav.cli import main

    sys.exit(main())
