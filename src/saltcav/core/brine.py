"""Свойства воды и рассола NaCl.

Соглашения
----------
* ``w`` — массовая доля NaCl в растворе, кг/кг (0 … w_sat).
* ``C`` — массовая концентрация, кг NaCl на м³ раствора: ``C = w · ρ(w, T)``.
  В модели выщелачивания баланс соли ведётся именно в ``C``.
* ``T`` — температура, °C (на границе ввода); внутри функций переводится в К,
  где это нужно.

Источники и точность
--------------------
* Плотность воды и растворимость NaCl — табличные значения справочника CRC
  (Handbook of Chemistry and Physics), линейная интерполяция.
* Плотность рассола — таблица CRC для 20 °C по массовой доле, пересчитанная на
  другую температуру множителем ρ_w(T)/ρ_w(20 °C). Это приближение: погрешность
  растёт с удалением от 20 °C (порядка десятых долей процента в диапазоне
  10–60 °C). Влияние давления не учитывается.
* Вязкость воды — уравнение Фогеля; коэффициент диффузии — пересчёт по
  Стоксу–Эйнштейну от опорного значения.
* Вязкость рассола пока не реализована (открытый вопрос, см. docs/decisions.md).

Таблицы — кандидаты на сверку при получении лабораторных данных компании.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

KELVIN = 273.15

# Плотность чистой воды при 1 атм, кг/м³ (CRC).
_T_WATER = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_RHO_WATER = np.array(
    [999.84, 999.70, 998.21, 995.65, 992.22, 988.04, 983.20, 977.76, 971.79, 965.31, 958.35]
)

# Растворимость NaCl, массовая доля насыщенного раствора (CRC).
_T_SOL = np.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_W_SAT = (
    np.array([26.28, 26.32, 26.41, 26.45, 26.52, 26.67, 26.84, 27.03, 27.25, 27.50, 27.78, 28.05])
    / 100.0
)

# Плотность раствора NaCl при 20 °C в зависимости от массовой доли, кг/м³ (CRC).
_W_TAB = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
                   0.20, 0.22, 0.24, 0.26])  # fmt: skip
_RHO20_TAB = np.array([998.21, 1005.3, 1012.5, 1026.8, 1041.3, 1055.9, 1070.7, 1085.7,
                       1100.9, 1116.2, 1131.9, 1147.8, 1164.0, 1180.4, 1197.2])  # fmt: skip

# Вязкость воды, уравнение Фогеля: μ = A·10^(B/(T−C)), T в К.
_VOGEL_A = 2.414e-5  # Па·с
_VOGEL_B = 247.8  # К
_VOGEL_C = 140.0  # К

# Коэффициент диффузии NaCl в воде при 25 °C (разбавленный раствор), м²/с.
D_NACL_25C = 1.61e-9


def _check_temperature(t: FloatArray) -> None:
    if np.any(t < 0.0) or np.any(t > 100.0):
        raise ValueError("температура вне диапазона таблиц 0–100 °C")


def water_density(t_c: ArrayLike) -> FloatArray:
    """Плотность чистой воды, кг/м³."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    return np.asarray(np.interp(t, _T_WATER, _RHO_WATER), dtype=np.float64)


def saturation_mass_fraction(t_c: ArrayLike) -> FloatArray:
    """Массовая доля NaCl в насыщенном растворе."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    return np.asarray(np.interp(t, _T_SOL, _W_SAT), dtype=np.float64)


def brine_density(w: ArrayLike, t_c: ArrayLike = 20.0) -> FloatArray:
    """Плотность рассола по массовой доле, кг/м³.

    Допускается небольшой выход за таблицу (до 0.30) линейной экстраполяцией —
    растворимость при 100 °C чуть выше последней табличной точки 0.26.
    """
    w_arr = np.asarray(w, dtype=float)
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    if np.any(w_arr < 0.0) or np.any(w_arr > 0.30):
        raise ValueError("массовая доля вне диапазона 0–0.30")
    slope = (_RHO20_TAB[-1] - _RHO20_TAB[-2]) / (_W_TAB[-1] - _W_TAB[-2])
    rho20 = np.where(
        w_arr <= _W_TAB[-1],
        np.interp(w_arr, _W_TAB, _RHO20_TAB),
        _RHO20_TAB[-1] + slope * (w_arr - _W_TAB[-1]),
    )
    return np.asarray(rho20 * water_density(t) / _RHO_WATER[2], dtype=np.float64)


def concentration_from_mass_fraction(w: ArrayLike, t_c: ArrayLike = 20.0) -> FloatArray:
    """Массовая доля → концентрация C, кг/м³."""
    w_arr = np.asarray(w, dtype=float)
    return w_arr * brine_density(w_arr, t_c)


def mass_fraction_from_concentration(c: ArrayLike, t_c: float = 20.0) -> FloatArray:
    """Концентрация C, кг/м³ → массовая доля (обратная функция по таблице)."""
    c_arr = np.asarray(c, dtype=float)
    w_grid = np.linspace(0.0, 0.30, 3001)
    c_grid = concentration_from_mass_fraction(w_grid, t_c)
    if np.any(c_arr < 0.0) or np.any(c_arr > c_grid[-1]):
        raise ValueError("концентрация вне допустимого диапазона")
    return np.asarray(np.interp(c_arr, c_grid, w_grid), dtype=np.float64)


def saturation_concentration(t_c: ArrayLike) -> FloatArray:
    """Концентрация насыщения C_s(T), кг/м³."""
    t = np.asarray(t_c, dtype=float)
    return concentration_from_mass_fraction(saturation_mass_fraction(t), t)


def density_from_concentration(c: ArrayLike, t_c: float = 20.0) -> FloatArray:
    """Плотность рассола по концентрации C, кг/м³ (для стратификации)."""
    return brine_density(mass_fraction_from_concentration(c, t_c), t_c)


def water_viscosity(t_c: ArrayLike) -> FloatArray:
    """Динамическая вязкость чистой воды, Па·с (уравнение Фогеля)."""
    t = np.asarray(t_c, dtype=float)
    _check_temperature(t)
    tk = t + KELVIN
    return _VOGEL_A * 10.0 ** (_VOGEL_B / (tk - _VOGEL_C))


def diffusivity(t_c: ArrayLike) -> FloatArray:
    """Коэффициент диффузии NaCl, м²/с: пересчёт D(25 °C) по Стоксу–Эйнштейну D ∝ T/μ."""
    t = np.asarray(t_c, dtype=float)
    tk = t + KELVIN
    ref_t = 25.0 + KELVIN
    return D_NACL_25C * (tk / ref_t) * (water_viscosity(25.0) / water_viscosity(t))
