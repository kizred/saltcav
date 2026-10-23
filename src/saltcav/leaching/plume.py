"""Струя пресной воды из башмака колонны и всплывающий плюм.

Пресная вода выходит из колонны вниз (струя с начальным импульсом). Она легче
рассола, поэтому плавучесть тормозит её; струя вовлекает окружающий рассол и
на глубине разворота останавливается. Дальше смесь всплывает как плюм, продолжая
вовлекать рассол, пока не потеряет плавучесть (уровень нейтральной плавучести)
или не упрётся в верх колонны рассола. Там смесь отдаётся в слой.

Модель — одномерный «фонтан» в приближении Мортона–Тейлора–Тёрнера с профилем
«цилиндр» (top-hat). Переменные вдоль пути s:

    Q = π b² w          объёмный расход плюма, м³/с
    M = π b² w²         поток импульса, м⁴/с²
    F = Q · C_p         поток соли, кг/с

    dQ/ds = 2 α √(π M)                  вовлечение (α — коэффициент вовлечения)
    d(M²)/ds = ± 2 Q² g',  g' = g (ρ_a − ρ_p)/ρ_a
    dF/ds = C_a dQ/ds

(уравнение записано для M², а не для M: в точке разворота M → 0, и форма
dM/ds = ± Q² g'/M там вырождается).

Знак «+» при всплытии (лёгкий плюм ускоряется), «−» для струи вниз (тормозится).
Вовлечение прекращается, когда радиус плюма достигает радиуса камеры в слое:
дальше вовлекать нечего. Коэффициент α = 0,09 — значение по умолчанию в SANSMIC.

Результат — сколько рассола вовлечено из каждого слоя и в какой слой отдана
смесь; перенос с этими источниками и стоками считается в
:func:`saltcav.leaching.brine_column.plume_column_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine

FloatArray = NDArray[np.float64]

G = 9.81  # м/с²
SUBSTEPS = 8  # подшагов интегрирования на слой


@dataclass(frozen=True)
class PlumeResult:
    entrained: FloatArray  # (n,) расход рассола, вовлечённого из слоя, м³/с
    k_turn: int  # слой разворота струи
    k_deposit: int  # слой, куда отдана смесь
    q_deposit: float  # расход смеси, м³/с (= закачка + Σ вовлечённого)
    c_deposit: float  # концентрация смеси при концентрациях начала шага, кг/м³


def plume_fluxes(
    c: FloatArray,
    dz: FloatArray,
    r_layer: FloatArray,
    k_src: int,
    q0: float,
    r_tube: float,
    alpha: float,
    temperature_c: float,
    c_in: float = 0.0,
) -> PlumeResult:
    """Путь струи и плюма в колонне слоёв 0 … n−1 (0 — верхний).

    ``c`` — концентрация в слоях, ``dz`` — толщины слоёв, ``r_layer`` — радиус
    камеры в слое (ограничивает вовлечение), ``k_src`` — слой башмака закачки.
    """
    c = np.asarray(c, dtype=float)
    n = c.size
    if not 0 <= k_src < n:
        raise ValueError("башмак вне колонны")
    if q0 <= 0 or r_tube <= 0 or alpha < 0:
        raise ValueError("нужны q0 > 0, r_tube > 0, alpha ≥ 0")
    # плотность рассола ρ(C) — квадратичная аппроксимация таблицы (ошибка в перепаде
    # плотности ≈ 0,25 %); табличная интерполяция на каждом подшаге слишком медленна
    c_fit = np.linspace(0.0, float(brine.saturation_concentration(temperature_c)), 40)
    a2, a1, a0 = np.polyfit(c_fit, brine.density_from_concentration(c_fit, temperature_c), 2)
    rho_a = a0 + c * (a1 + a2 * c)
    entrained = np.zeros(n)

    w0 = q0 / (np.pi * r_tube**2)
    # вместо M интегрируем P = M²: dP/ds = ±2 Q² g' — без особенности при M → 0
    # в точке разворота струи (уравнение для M там вырождается)
    q, p, f = q0, (q0 * w0) ** 2, q0 * c_in

    def advance(k: int, length: float, sign: float) -> bool:
        """Пройти длину length в слое k; False — плюм остановился внутри слоя."""
        nonlocal q, p, f
        ds = length / SUBSTEPS
        for _ in range(SUBSTEPS):
            c_p = f / q
            rho_p = a0 + c_p * (a1 + a2 * c_p)
            g_red = G * (rho_a[k] - rho_p) / rho_a[k]
            m = np.sqrt(p)
            # радиус плюма b = Q/√(πM); шире камеры вовлекать нечего
            free = q < r_layer[k] * np.sqrt(np.pi * m)
            dq = 2.0 * alpha * np.sqrt(np.pi * m) * ds if free else 0.0
            p += sign * 2.0 * q * q * g_red * ds
            q += dq
            f += c[k] * dq
            entrained[k] += dq
            if p <= 0.0:
                return False
        return True

    # струя вниз от середины слоя башмака
    k = k_src
    length = 0.5 * dz[k]
    while True:
        if not advance(k, length, -1.0) or k == n - 1:
            break
        k += 1
        length = dz[k]
    k_turn = k

    # плюм вверх от слоя разворота (импульс в точке разворота нулевой)
    p = 0.0
    k = k_turn
    length = 0.5 * dz[k]
    while True:
        if not advance(k, length, +1.0) or k == 0:
            break
        k -= 1
        length = dz[k]
    k_dep = k

    return PlumeResult(entrained, k_turn, k_dep, q, f / q)
