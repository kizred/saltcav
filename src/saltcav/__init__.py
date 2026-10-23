"""saltcav — стохастическая модель скважинного выщелачивания соляных камер.

Внутри пакета все величины в СИ; перевод единиц только на вводе и выводе
(см. :mod:`saltcav.core.units`).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("saltcav")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
