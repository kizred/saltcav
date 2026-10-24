"""Процессы расчёта: число по памяти, однопоточный BLAS, продолжение после сбоя."""

import os
from pathlib import Path
from typing import Any

import pytest

from saltcav.core import parallel
from saltcav.stochastic import ensemble as ens_mod

ROOT = Path(__file__).resolve().parents[2]
PARENT = os.getpid()


def _crashy(task: Any) -> Any:
    """В процессе-исполнителе падает, как при нехватке памяти; в основном — считает."""
    if os.getpid() != PARENT:
        os._exit(3)
    return ens_mod._run_one(task)


def _blas_threads(_: int) -> str:
    return os.environ.get("OPENBLAS_NUM_THREADS", "")


def test_default_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parallel.default_jobs(3) == 3
    assert parallel.default_jobs(8, n_tasks=2) == 2
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(parallel, "available_memory_mb", lambda: 2000.0)
    assert parallel.default_jobs() == 4  # 0,8 · 2000 / 400 — память, а не ядра
    monkeypatch.setattr(parallel, "available_memory_mb", lambda: None)
    assert parallel.default_jobs() == 64


def test_single_thread_blas_in_workers() -> None:
    before = os.environ.get("OPENBLAS_NUM_THREADS")
    with parallel.process_pool(2) as pool:
        assert list(pool.map(_blas_threads, range(2))) == ["1", "1"]
    assert os.environ.get("OPENBLAS_NUM_THREADS") == before


def test_ensemble_survives_broken_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import saltcav.api as sc

    cfg = sc.set(sc.load(ROOT / "configs" / "example.yaml"), {"grid.nz": 40})
    ref = sc.ensemble(cfg, n=3, jobs=1)
    monkeypatch.setattr(ens_mod, "_run_one", _crashy)
    got = sc.ensemble(cfg, n=3, jobs=2)  # процессы падают — досчитывается здесь
    assert got.n == 3
    assert (got.volume_t[:, -1] == ref.volume_t[:, -1]).all()


def test_run_with_fallback() -> None:
    from concurrent.futures.process import BrokenProcessPool

    notes: list[str] = []

    def broken() -> int:
        raise BrokenProcessPool("упал")

    assert parallel.run_with_fallback(broken, lambda: 7, notes.append) == 7
    assert notes and "одном процессе" in notes[0]
