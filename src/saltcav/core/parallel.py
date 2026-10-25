"""Параллельные расчёты в процессах: сколько процессов и как их запускать.

Каждый процесс загружает NumPy, SciPy и BLAS заново, а BLAS (OpenBLAS, MKL) по
умолчанию заводит столько потоков, сколько ядер, и под каждый — буферы. На машине
с многими ядрами N процессов × N потоков исчерпывают память (на Windows — «файл
подкачки слишком мал», процесс падает, «BrokenProcessPool»). Поэтому:

- в процессах BLAS однопоточный (параллельность — уже процессами);
- процессов не больше, чем позволяет свободная память (≈ 400 МБ на процесс);
- если процесс всё же упал, расчёт продолжается в одном процессе (медленнее, но
  до конца) — см. :func:`run_with_fallback`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

WORKER_MB = 400  # память одного процесса-исполнителя с запасом, МБ
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")  # fmt: skip


def available_memory_mb() -> float | None:
    """Свободная память, МБ: меньшее из свободной физической и доступной для выделения
    (на Windows — с учётом файла подкачки). None — не удалось узнать."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]  # fmt: skip

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return None
            return min(st.ullAvailPhys, st.ullAvailPageFile) / 2**20
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, AttributeError):
        return None
    return None


def default_jobs(jobs: int = 0, n_tasks: int | None = None, worker_mb: float = WORKER_MB) -> int:
    """Число процессов: ``jobs`` > 0 — как задано; 0 — по ядрам, но не больше, чем
    позволяет свободная память, и не больше числа задач."""
    if jobs > 0:
        n = jobs
    else:
        n = os.cpu_count() or 1
        mem = available_memory_mb()
        if mem is not None:
            n = min(n, max(1, int(0.8 * mem / worker_mb)))
        if sys.platform == "win32":
            n = min(n, 61)  # предел ProcessPoolExecutor на Windows
    if n_tasks is not None:
        n = min(n, n_tasks)
    return max(1, n)


@contextmanager
def single_thread_blas() -> Iterator[None]:
    """Дочерние процессы, запущенные внутри, получают однопоточный BLAS (переменные
    окружения наследуются при запуске процесса)."""
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def blas_threads(n: int = 1) -> Iterator[None]:
    """Ограничить потоки BLAS в текущем процессе: задача ансамбля считается одинаково
    (бит в бит) и в основном процессе, и в процессе-исполнителе."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=n):
        yield


@contextmanager
def process_pool(jobs: int) -> Iterator[ProcessPoolExecutor]:
    """Пул процессов: forkserver (Linux, macOS) или spawn (Windows, собранное
    приложение), однопоточный BLAS в процессах."""
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    with (
        single_thread_blas(),
        ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context(method)) as pool,
    ):
        yield pool


def run_with_fallback(
    parallel: Callable[[], T],
    serial: Callable[[], T],
    note: Callable[[str], None] | None = None,
) -> T:
    """Выполнить ``parallel``; если процесс-исполнитель упал (обычно — не хватило
    памяти), повторить ``serial`` в одном процессе."""
    try:
        return parallel()
    except BrokenProcessPool:
        if note is not None:
            note("процесс расчёта завершился аварийно (мало памяти?) — продолжаю в одном "
                 "процессе")  # fmt: skip
        return serial()
