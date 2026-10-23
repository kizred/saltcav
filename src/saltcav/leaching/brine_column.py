"""Баланс соли в рассоле по слоям глубины (одномерная колонна).

Состояние — концентрация C_k (кг/м³) в каждом слое k = 0 … nz−1 (0 — верхний).
Объём рассола в слое V_k равен объёму камеры в слое (сумма по секторам).

Шаг по времени dt складывается из двух частей:

1. **Перенос и источники** — неявная схема против потока (безусловно
   устойчива, консервативна). Пресная вода (C = 0) входит в слой башмака закачки
   k_in и проходит по слоям до слоя отбора k_out; рассол уходит из k_out.
   Объём камеры за шаг растёт на ΔV, поэтому отбор меньше закачки:
   Q_out = Q_in − ΔV/dt. Растворённая со стенок соль Δm_k добавляется в слой k.

       V_k' C_k' = V_k C_k + dt·(Q_in·C_up' − Q_k·C_k') + Δm_k,

   где Q_k = Q_in на пути и Q_out в слое отбора, ' — новый момент.

2. **Плотностная устойчивость** — рассол тяжелее при большей C, поэтому
   устойчив только профиль, где C не убывает с глубиной. Неустойчивые соседние
   слои мгновенно перемешиваются (взвешенно по объёму) — изотоническая регрессия
   (алгоритм PAV). Так всплывает пресная вода из нижнего башмака при прямотоке.

Соль сохраняется точно: Σ V'C' = Σ VC + ΣΔm − dt·Q_out·C_out'.
Допущения: рассол несжимаем, изменение объёма смеси при растворении не
учитывается, турбулентная диффузия между слоями пока не включена.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TransportResult:
    c: FloatArray  # новая концентрация, кг/м³
    produced_salt: float  # соль, ушедшая с рассолом за шаг, кг
    q_out: float  # расход отбора, м³/с


def transport_step(
    c: FloatArray,
    v_old: FloatArray,
    v_new: FloatArray,
    dm: FloatArray,
    q_in: float,
    k_in: int,
    k_out: int,
    dt: float,
    c_in: float = 0.0,
) -> TransportResult:
    """Неявный шаг переноса от k_in к k_out с ростом объёмов и источниками соли."""
    c = np.asarray(c, dtype=float)
    v_old = np.asarray(v_old, dtype=float)
    v_new = np.asarray(v_new, dtype=float)
    dm = np.asarray(dm, dtype=float)
    n = c.size
    if not (v_old.shape == v_new.shape == dm.shape == (n,)):
        raise ValueError("размеры c, v_old, v_new, dm должны совпадать")
    if not (0 <= k_in < n and 0 <= k_out < n):
        raise ValueError("индексы башмаков вне колонны")
    if dt <= 0 or q_in < 0:
        raise ValueError("нужны dt > 0 и q_in ≥ 0")
    if np.any(v_new <= 0):
        raise ValueError("объёмы слоёв должны быть положительны")

    q_out = q_in - float(v_new.sum() - v_old.sum()) / dt
    if q_out < 0:
        raise ValueError("рост камеры за шаг больше закачки — уменьшите dt")

    mass = v_old * c + dm  # масса соли до переноса, кг
    c_new = mass / v_new  # слои вне пути: только источники и изменение объёма

    step = -1 if k_out < k_in else 1  # прямоток: вверх; противоток: вниз
    path = range(k_in, k_out + step, step)
    c_up = c_in
    for k in path:
        q_leave = q_out if k == k_out else q_in
        c_new[k] = (mass[k] + dt * q_in * c_up) / (v_new[k] + dt * q_leave)
        c_up = c_new[k]

    produced = dt * q_out * float(c_new[k_out])
    return TransportResult(c_new, produced, q_out)


def stabilize(c: FloatArray, v: FloatArray) -> FloatArray:
    """Плотностная устойчивость: перемешивание до C, неубывающей с глубиной.

    Взвешенная по объёму изотоническая регрессия (pool adjacent violators):
    результат — ближайший в смысле Σ V(C'−C)² устойчивый профиль, масса соли
    сохраняется точно.
    """
    c = np.asarray(c, dtype=float)
    v = np.asarray(v, dtype=float)
    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for ck, vk in zip(c, v, strict=True):
        means.append(float(ck))
        weights.append(float(vk))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            cnt = counts[-2] + counts[-1]
            del means[-1], weights[-1], counts[-1]
            means[-1], weights[-1], counts[-1] = m, w, cnt
    return np.repeat(np.array(means), counts)


def column_step(
    c: FloatArray,
    v_old: FloatArray,
    v_new: FloatArray,
    dm: FloatArray,
    q_in: float,
    k_in: int,
    k_out: int,
    dt: float,
) -> TransportResult:
    """Полный шаг колонны: перенос + плотностное перемешивание."""
    res = transport_step(c, v_old, v_new, dm, q_in, k_in, k_out, dt)
    return TransportResult(stabilize(res.c, v_new), res.produced_salt, res.q_out)


def plume_transport_step(
    c: FloatArray,
    v_old: FloatArray,
    v_new: FloatArray,
    dm: FloatArray,
    q_in: float,
    entrained: FloatArray,
    k_deposit: int,
    k_out: int,
    dt: float,
    c_in: float = 0.0,
) -> TransportResult:
    """Неявный консервативный шаг колонны с плюмом.

    Плюм забирает из слоёв расходы ``entrained`` и отдаёт смесь (закачка + всё
    вовлечённое) в слой ``k_deposit``; рассол уходит из ``k_out`` с расходом
    Q_out = Q_in − ΔV/dt. Чтобы объём каждого слоя сходился, между слоями возникает
    обратное течение окружающего рассола F_k; оно переносит соль против потока.

    Концентрация смеси считается по новым концентрациям слоёв (неявно), поэтому
    слой отдачи связан со всеми слоями вовлечения — система решается целиком.
    Матрица — M-матрица со строгим диагональным преобладанием по столбцам:
    концентрации не становятся отрицательными, соль сохраняется точно:
    Σ V'C' = Σ VC + ΣΔm + dt·Q_in·C_in − dt·Q_out·C_out'.
    """
    c = np.asarray(c, dtype=float)
    v_old = np.asarray(v_old, dtype=float)
    v_new = np.asarray(v_new, dtype=float)
    dm = np.asarray(dm, dtype=float)
    e = np.asarray(entrained, dtype=float)
    n = c.size
    if not (v_old.shape == v_new.shape == dm.shape == e.shape == (n,)):
        raise ValueError("размеры массивов должны совпадать")
    if not (0 <= k_deposit < n and 0 <= k_out < n):
        raise ValueError("индексы слоёв вне колонны")
    if dt <= 0 or q_in < 0 or np.any(e < 0) or np.any(v_new <= 0):
        raise ValueError("нужны dt > 0, q_in ≥ 0, вовлечение ≥ 0, объёмы > 0")

    q_out = q_in - float(v_new.sum() - v_old.sum()) / dt
    if q_out < 0:
        raise ValueError("рост камеры за шаг больше закачки — уменьшите dt")

    inflow = np.zeros(n)
    inflow[k_deposit] = q_in + e.sum()
    outflow = e.copy()
    outflow[k_out] += q_out

    # поток через верх слоя k вверх (в слой k−1): f_up[k]; через дно колонны и верх — 0
    f_up = np.zeros(n + 1)
    for k in range(n - 1, 0, -1):
        f_up[k] = f_up[k + 1] + inflow[k] - outflow[k] - (v_new[k] - v_old[k]) / dt

    a = np.zeros((n, n))
    up = np.maximum(f_up[:n], 0.0)  # из k вверх
    down = np.maximum(-f_up[1:], 0.0)  # из k вниз (через его дно)
    idx = np.arange(n)
    a[idx, idx] = v_new + dt * (outflow + up + down)
    a[idx[:-1], idx[1:]] -= dt * np.maximum(f_up[1:n], 0.0)  # приток снизу
    a[idx[1:], idx[:-1]] -= dt * np.maximum(-f_up[1:n], 0.0)  # приток сверху
    a[k_deposit, :] -= dt * e  # смесь несёт вовлечённую соль
    rhs = v_old * c + dm
    rhs[k_deposit] += dt * q_in * c_in

    c_new = np.linalg.solve(a, rhs)
    produced = dt * q_out * float(c_new[k_out])
    return TransportResult(np.asarray(c_new, dtype=np.float64), produced, q_out)
