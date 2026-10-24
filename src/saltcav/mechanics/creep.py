"""Законы ползучести каменной соли (векторизовано по точкам интегрирования).

Установившаяся ползучесть — сумма степенных ветвей с аррениусовской зависимостью
от температуры:

    ε̇_ss(σ, T) = Σ A_i · exp(−Q_i / (R·T)) · (σ / 1 МПа)^n_i,

где σ — эквивалентное напряжение Мизеса. По умолчанию одна ветвь BGRa (Hunsche,
Hampel, 1999): A = 0,18 сут⁻¹, n = 5, Q = 54 кДж/моль.

Неустановившаяся ползучесть — функция Мансона–Доусона (Munson, Dawson): полная
скорость F·ε̇_ss, внутренняя переменная ζ (накопленная неустановившаяся деформация):

    ε_t* = K0 · exp(c·T) · (σ/μ)^m,
    F = exp( Δ·(1 − ζ/ε_t*)²),  Δ = α_w + β_w·lg(σ/μ)   при ζ < ε_t* (упрочнение),
    F = exp(−δ·(1 − ζ/ε_t*)²)                             при ζ > ε_t* (возврат),
    ζ̇ = (F − 1)·ε̇_ss.

Параметры по умолчанию — чистая соль WIPP (Munson, 1997); как и BGRa, это
иллюстрация до испытаний керна.

Направление течения — закон Прандтля–Рейсса: ε̇ = (3/2)·ε̇_eq·s/σ_eq.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saltcav.mechanics.fem import P_DEV

FloatArray = NDArray[np.float64]
R_GAS = 8.314462618  # Дж/(моль·К)
MPA = 1.0e6
SIGMA_FLOOR = 1.0  # Па: ниже — напряжение считается нулевым (нет течения)


@dataclass(frozen=True)
class PowerBranch:
    """ε̇ = A·exp(−Q/RT)·(σ/1 МПа)^n."""

    a_per_s: float  # 1/с
    n: float
    q_j_per_mol: float

    def rate(self, sig_eq: FloatArray, t_k: FloatArray) -> FloatArray:
        s = np.maximum(sig_eq, 0.0) / MPA
        return np.asarray(self.a_per_s * np.exp(-self.q_j_per_mol / (R_GAS * t_k)) * s**self.n)


BGRA = PowerBranch(a_per_s=0.18 / 86_400.0, n=5.0, q_j_per_mol=54_000.0)


@dataclass(frozen=True)
class MunsonDawson:
    """Неустановившаяся ползучесть Мансона–Доусона (параметры WIPP clean salt)."""

    mu_pa: float = 12.4e9
    k0: float = 6.275e5
    c_per_k: float = 9.198e-3
    m: float = 3.0
    alpha_w: float = -17.37
    beta_w: float = -7.738
    delta: float = 0.58

    def limit(self, sig_eq: FloatArray, t_k: FloatArray) -> FloatArray:
        """ε_t* — предельная неустановившаяся деформация."""
        x = np.maximum(sig_eq, SIGMA_FLOOR) / self.mu_pa
        return np.asarray(self.k0 * np.exp(self.c_per_k * t_k) * x**self.m)

    def factor(self, sig_eq: FloatArray, t_k: FloatArray, zeta: FloatArray) -> FloatArray:
        lim = self.limit(sig_eq, t_k)
        x = np.maximum(sig_eq, SIGMA_FLOOR) / self.mu_pa
        big = self.alpha_w + self.beta_w * np.log10(x)
        d = 1.0 - zeta / lim
        f = np.where(zeta < lim, np.exp(np.minimum(big * d * d, 50.0)), np.exp(-self.delta * d * d))
        return np.asarray(f)


@dataclass(frozen=True)
class CreepLaw:
    branches: tuple[PowerBranch, ...] = (BGRA,)
    transient: MunsonDawson | None = field(default_factory=MunsonDawson)

    def steady(self, sig_eq: FloatArray, t_k: FloatArray) -> tuple[FloatArray, FloatArray]:
        """ε̇_ss и dε̇_ss/dσ_eq."""
        rate = np.zeros_like(sig_eq)
        drate = np.zeros_like(sig_eq)
        s = np.maximum(sig_eq, SIGMA_FLOOR)
        for br in self.branches:
            r = br.rate(s, t_k)
            rate += r
            drate += br.n * r / s
        return rate, drate

    def evaluate(
        self, sig: FloatArray, t_k: FloatArray, zeta: FloatArray, dt: float = 0.0
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """По напряжениям (…, 4) в записи Манделя: скорость ползучести (…, 4),
        касательная dε̇/dσ (…, 4, 4) при замороженном F, ε̇_ss и F.

        При dt > 0 множитель F усредняется по шагу: ζ не может перейти ε_t* за шаг,
        поэтому неустановившаяся деформация за шаг не больше ε_t* − ζ. Иначе при
        малых напряжениях, где F огромен, а ε_t* мал, шаг был бы привязан к секундам.
        """
        s = np.einsum("ij,...j->...i", P_DEV, sig)
        seq = np.sqrt(1.5 * np.einsum("...i,...i->...", s, s))
        ss, dss = self.steady(seq, t_k)
        if self.transient is None:
            f = np.ones_like(seq)
        else:
            f = self.transient.factor(seq, t_k, zeta)
            if dt > 0:
                lim = self.transient.limit(seq, t_k)
                base = np.maximum(ss * dt, 1e-300)
                extra = np.where(
                    f >= 1.0,
                    np.minimum((f - 1.0) * base, np.maximum(lim - zeta, 0.0)),
                    -np.minimum((1.0 - f) * base, np.maximum(zeta - lim, 0.0)),
                )
                f = 1.0 + extra / base
        safe = np.maximum(seq, SIGMA_FLOOR)
        active = seq > SIGMA_FLOOR
        eq_rate = f * ss
        coef = np.where(active, 1.5 * eq_rate / safe, 0.0)
        rate = coef[..., None] * s
        # dε̇/dσ = (3/2)F[(f/σ)P + (3/2)(f'/σ² − f/σ³) s sᵀ]
        a1 = coef
        a2 = np.where(active, 2.25 * f * (dss / safe**2 - ss / safe**3), 0.0)
        tang = a1[..., None, None] * P_DEV + a2[..., None, None] * np.einsum(
            "...i,...j->...ij", s, s
        )
        return rate, tang, ss, f


def von_mises(sig: FloatArray) -> FloatArray:
    s = np.einsum("ij,...j->...i", P_DEV, sig)
    return np.asarray(np.sqrt(1.5 * np.einsum("...i,...i->...", s, s)))
