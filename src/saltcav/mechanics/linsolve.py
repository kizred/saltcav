"""Решение K u = f для шагов ползучести: одно разложение — много шагов.

Касательная матрица ползучести меняется от шага к шагу медленно, поэтому старое
разложение служит предобусловливателем сопряжённых градиентов (PCG), а заново матрица
раскладывается, только если итераций стало много. Решение доводится до малой
относительной невязки (по умолчанию 1e-10 — на уровне округления прямого решателя),
так что результат тот же, что при разложении на каждом шаге, а разложений — в разы
меньше. Матрица симметрична и положительно определена (закреплённые степени свободы
исключены): разложение — PARDISO (``pypardiso``), иначе SuperLU с упорядочением для
симметричных матриц.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def factorize_spd(
    k: sp.csr_matrix, solver: str = "auto", holder: list[Any] | None = None
) -> Callable[[FloatArray], FloatArray]:
    """Разложение симметричной положительно определённой матрицы → функция решения.

    ``holder`` хранит один экземпляр PARDISO на задачу: при новом разложении память
    прежнего освобождается, а не копится."""
    holder = [] if holder is None else holder
    if solver in ("auto", "pardiso"):
        if sys.platform.startswith("linux"):
            # Gmsh на Linux собран с GNU OpenMP (libgomp), MKL по умолчанию — с Intel
            # OpenMP (libiomp5); две среды OpenMP в одном процессе падают. Переводим MKL
            # на libgomp — до первой загрузки MKL.
            os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        try:
            import pypardiso
        except ImportError:
            if solver == "pardiso":
                raise
        else:
            if holder:
                ps = holder[0]
                ps.free_memory(everything=False)
            else:
                ps = pypardiso.PyPardisoSolver()
                ps.set_matrix_type(2)  # вещественная симметричная положительно определённая
                holder.append(ps)
            upper = sp.triu(k, format="csr")
            ps.factorize(upper)
            return lambda b: np.asarray(ps.solve(upper, b))
    lu = spla.splu(sp.csc_matrix(k), permc_spec="MMD_AT_PLUS_A")
    return lambda b: np.asarray(lu.solve(b))


@dataclass
class ReusedFactorSolver:
    """Решатель с повторным использованием разложения (PCG со старым разложением)."""

    solver: str = "auto"  # auto | pardiso | superlu
    rtol: float = 1e-10
    max_it: int = 12
    factorizations: int = 0
    iterations: list[int] = field(default_factory=list)
    _factor: Callable[[FloatArray], FloatArray] | None = field(default=None, repr=False)
    _n: int = 0
    _pardiso: list[Any] = field(default_factory=list, repr=False)

    def solve(self, a: sp.csr_matrix, b: FloatArray) -> FloatArray:
        if self._factor is None or self._n != a.shape[0]:
            self.refactor(a)
        x, it = self._pcg(a, b)
        if x is None:
            self.refactor(a)
            x, it = self._pcg(a, b)
            if x is None:  # pragma: no cover - разложение точное, сюда не попадаем
                raise RuntimeError("линейная система не решилась")
        self.iterations.append(it)
        return x

    def refactor(self, a: sp.csr_matrix) -> None:
        self._factor = None  # старое разложение освобождается до нового (память)
        self._factor = factorize_spd(a, self.solver, self._pardiso)
        self._n = a.shape[0]
        self.factorizations += 1

    def _pcg(self, a: sp.csr_matrix, b: FloatArray) -> tuple[FloatArray | None, int]:
        solve = self._factor
        assert solve is not None
        nb = float(np.linalg.norm(b))
        if nb == 0:
            return np.zeros_like(b), 0
        x = solve(b)
        r = b - a @ x
        z = solve(r)
        p = z.copy()
        rz = float(r @ z)
        for it in range(1, self.max_it + 1):
            if np.linalg.norm(r) <= self.rtol * nb:
                return x, it - 1
            ap = a @ p
            alpha = rz / float(p @ ap)
            x += alpha * p
            r -= alpha * ap
            z = solve(r)
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        ok = np.linalg.norm(r) <= self.rtol * nb
        return (x, self.max_it) if ok else (None, self.max_it)


def csr_pattern(
    dofs: NDArray[np.int64], ndof: int
) -> tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.int32]]:
    """Шаблон глобальной матрицы по номерам степеней свободы элементов (M, k):
    индекс каждого элементного коэффициента в массиве данных CSR, indices, indptr.
    Сборка тогда — ``bincount`` по готовым индексам без сортировки."""
    k = dofs.shape[1]
    rows = np.repeat(dofs, k, axis=1).ravel()
    cols = np.tile(dofs, (1, k)).ravel()
    key = rows * ndof + cols
    uniq, inv = np.unique(key, return_inverse=True)
    r_u = uniq // ndof
    indices = (uniq % ndof).astype(np.int32)
    indptr = np.zeros(ndof + 1, dtype=np.int64)
    np.add.at(indptr, r_u + 1, 1)
    ptr = np.cumsum(indptr).astype(np.int32)
    return inv.astype(np.int64), indices, ptr
