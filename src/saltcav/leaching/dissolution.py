"""Растворение стенки: закон потока соли и скорость отступания стенки.

Закон растворения
-----------------
Поток соли с единицы площади стенки, кг/(м²·с):

    j = k0 · f(ψ) · C_s · u^n,    u = max(0, (C_s − C)/C_s),

где u — недонасыщение рассола у стенки, n ≈ 4/3 для турбулентной свободной
конвекции (скорость массопереноса ~ Δρ^(1/3), сам перепад ~ u), k0 — масштаб
скорости массопереноса, м/с. Нормальная скорость отступания стенки, м/с:

    V = j / (ρ_s · (1 − f_ins)).

Множитель ориентации f(ψ) линеен по cos ψ: f = 1 на вертикальной стенке,
``roof_ratio`` на кровле (ψ = 0), ``floor_ratio`` на подошве (ψ = 180°).

Движение стенки
---------------
Стенка — график R(z) в каждом секторе, отступает по нормали со скоростью V.
Для графика это уравнение Гамильтона–Якоби

    ∂R/∂t = V · √(1 + (∂R/∂z)²),

то есть ∂R/∂t = V₀ · H̃(∂R/∂z), H̃(p) = √(1+p²)·f(ψ(p)), где V₀ — скорость
вертикальной стенки. Решается явной схемой с численным гамильтонианом Годунова
(:func:`godunov_hamiltonian`): направление «против потока» выбирается по самому
H̃, а не по знаку наклона. Это важно, потому что при разной скорости кровли и
подошвы H̃ несимметричен и даже невыпуклый (излом при p = 0).

Так кровля сама «поднимается»: узел над камерой растёт, пока не догонит соседа
снизу. Схема монотонна при dt ≤ Δz / max|V₀·H̃'| (условие КФЛ); R не убывает.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DissolutionLaw:
    k0: float  # м/с
    exponent: float = 4.0 / 3.0
    roof_ratio: float = 1.5
    floor_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.k0 <= 0 or self.exponent <= 0:
            raise ValueError("k0 и exponent должны быть положительны")
        if self.roof_ratio < 0 or self.floor_ratio < 0:
            raise ValueError("коэффициенты ориентации неотрицательны")

    def orientation_factor(self, cos_psi: FloatArray) -> FloatArray:
        """f(ψ) по cos ψ: 1 + (roof−1)·cos ψ сверху, 1 + (1−floor)·cos ψ снизу."""
        c = np.clip(np.asarray(cos_psi, dtype=float), -1.0, 1.0)
        up = 1.0 + (self.roof_ratio - 1.0) * c
        down = 1.0 + (1.0 - self.floor_ratio) * c
        return np.asarray(np.where(c >= 0.0, up, down), dtype=np.float64)

    def mass_flux(self, c: FloatArray, c_sat: float, cos_psi: FloatArray) -> FloatArray:
        """Поток соли со стенки, кг/(м²·с)."""
        u = np.clip((c_sat - np.asarray(c, dtype=float)) / c_sat, 0.0, 1.0)
        flux = self.k0 * self.orientation_factor(cos_psi) * c_sat * u**self.exponent
        return np.asarray(flux, dtype=np.float64)


def node_concentration(c_layers: FloatArray) -> FloatArray:
    """Концентрация в узлах стенки (nz+1) по слоям (nz): максимум соседних слоёв.

    Узел — общая граница двух слоёв; его движение меняет объём обоих. Если один
    из слоёв насыщен, соль из прироста его объёма некуда деть, поэтому узел
    растворяется по более насыщенному слою. Насыщенный рассол не растворяет.
    """
    c = np.asarray(c_layers, dtype=float)
    out = np.empty(c.size + 1)
    out[0], out[-1] = c[0], c[-1]
    out[1:-1] = np.maximum(c[:-1], c[1:])
    return out


def wall_growth_rate(
    radius: FloatArray,
    dz: FloatArray,
    c_nodes: FloatArray,
    c_sat: float,
    law: DissolutionLaw,
    solid_density: float | FloatArray,
    active: NDArray[np.bool_],
    k_mult: float | FloatArray = 1.0,
) -> tuple[FloatArray, float]:
    """Скорость роста радиуса ∂R/∂t в узлах (nz+1, nphi) и максимум V для КФЛ.

    ``active`` (nz+1,) — узлы, где идёт растворение (ниже нерастворителя).
    ``solid_density`` — ρ_s·(1 − f_ins), кг растворимой соли на м³ породы: число или
    массив (nz+1, nphi). ``k_mult`` — множитель скорости растворения, число или
    массив (nz+1, nphi) (неоднородность по глубине и азимуту). Секторы по φ движутся
    независимо: рассол в слое перемешан по горизонтали, азимутальная связь — только
    через общую концентрацию слоя.
    """
    r = np.asarray(radius, dtype=float)
    h = np.asarray(dz, dtype=float)[:, None]
    d = np.diff(r, axis=0) / h  # наклон на интервалах (nz, nphi)
    zero = np.zeros((1, r.shape[1]))
    p_minus = np.vstack([zero, d])  # D⁻R = (R_i − R_{i−1})/Δz, у верхнего узла 0
    p_plus = np.vstack([d, zero])  # D⁺R = (R_{i+1} − R_i)/Δz, у нижнего узла 0

    # скорость отступания вертикальной стенки при местной концентрации, м/с
    flux = law.mass_flux(np.asarray(c_nodes, dtype=float), c_sat, np.zeros(len(c_nodes)))
    v_base = flux[:, None] * np.asarray(k_mult) / np.asarray(solid_density)
    v_base = np.where(np.asarray(active)[:, None], np.broadcast_to(v_base, r.shape), 0.0)

    h_god = godunov_hamiltonian(p_minus, p_plus, law)
    # узел без полости (R = 0) не имеет своей стенки: вычитаем вклад вертикальной
    # стенки H̃(0) = 1 и оставляем только рост от наклона к соседу (подъём кровли,
    # опускание подошвы). В прогоне от скважины (R ≥ r_скв > 0) это не включается.
    h_eff = np.where(r > 0.0, h_god, np.maximum(h_god - 1.0, 0.0))
    rate = v_base * h_eff

    a_max = max(abs(law.roof_ratio - 1.0), abs(1.0 - law.floor_ratio))
    v_char = float(np.max(v_base)) * (1.0 + a_max)  # оценка max|H̃'| для КФЛ
    return np.asarray(rate, dtype=np.float64), v_char


def hamiltonian(p: FloatArray, law: DissolutionLaw) -> FloatArray:
    """H̃(p) = √(1+p²)·f(ψ(p)): скорость роста R при единичной базовой скорости."""
    s = np.sqrt(1.0 + p * p)
    return np.asarray(s * law.orientation_factor(p / s), dtype=np.float64)


def _critical_slopes(law: DissolutionLaw) -> list[float]:
    """Точки, где H̃ может иметь экстремум: излом в 0 и стационарные точки ветвей.

    На ветви с f = 1 + a·cos ψ: H̃ = √(1+p²) + a·p, H̃' = 0 при p = −a/√(1−a²), |a| < 1.
    """
    pts = [0.0]
    a_up = law.roof_ratio - 1.0  # ветвь p ≥ 0
    if -1.0 < a_up < 0.0:
        pts.append(-a_up / np.sqrt(1.0 - a_up * a_up))
    a_down = 1.0 - law.floor_ratio  # ветвь p < 0
    if 0.0 < a_down < 1.0:
        pts.append(-a_down / np.sqrt(1.0 - a_down * a_down))
    return pts


def godunov_hamiltonian(p_minus: FloatArray, p_plus: FloatArray, law: DissolutionLaw) -> FloatArray:
    """Численный гамильтониан Годунова для ∂R/∂t = H̃(∂R/∂z).

    max H̃ на [p⁻, p⁺] при p⁻ ≤ p⁺ и min H̃ на [p⁺, p⁻] при p⁺ < p⁻. Экстремумы
    ищутся среди концов интервала и критических точек H̃, попавших внутрь.
    Корректно и при невыпуклом H̃ (кровля и подошва растворяются по-разному).
    """
    lo = np.minimum(p_minus, p_plus)
    hi = np.maximum(p_minus, p_plus)
    use_max = p_minus <= p_plus
    h_lo, h_hi = hamiltonian(lo, law), hamiltonian(hi, law)
    best = np.where(use_max, np.maximum(h_lo, h_hi), np.minimum(h_lo, h_hi))
    for pc in _critical_slopes(law):
        inside = (lo < pc) & (pc < hi)
        hc = float(hamiltonian(np.array(pc), law))
        best = np.where(inside & use_max, np.maximum(best, hc), best)
        best = np.where(inside & ~use_max, np.minimum(best, hc), best)
    return np.asarray(best, dtype=np.float64)
