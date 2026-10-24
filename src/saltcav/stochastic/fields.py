"""Коррелированные гауссовы поля на сетке узлов стенки (z, φ).

Поле g(z, φ) — стандартное гауссово (среднее 0, дисперсия 1) с разделимой
экспоненциальной ковариацией:

    cov(g(z₁, φ₁), g(z₂, φ₂)) = exp(−|z₁ − z₂| / ℓ_z) · exp(−d(φ₁, φ₂) / d_φ),

где d(φ₁, φ₂) = 2·sin(|φ₁ − φ₂|/2) — хордовое расстояние на единичной окружности,
d_φ = 2·sin(ℓ_φ/2). Экспонента от хордового расстояния положительно определена на
окружности, поэтому поле периодично по азимуту без шва; на малых углах d ≈ |Δφ|.

Генерация: g = L_z · W · L_φᵀ, где W — матрица независимых N(0, 1), а L_z, L_φ —
множители Холецкого корреляционных матриц по z и по φ. Множители считаются один раз
на ансамбль.

Из g строятся физические поля:

* множитель скорости растворения k = exp(σ·g − σ²/2) — логнормальный со средним 1;
* доля нерастворимых f = clip(f̄ + σ_f·g, 0, f_max).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
JITTER = 1e-10


def exponential_correlation(distance: FloatArray, length: float) -> FloatArray:
    """exp(−d/ℓ); при ℓ = 0 — независимые значения (единичная матрица)."""
    d = np.asarray(distance, dtype=float)
    if length <= 0.0:
        return np.asarray((d == 0.0).astype(float))
    return np.asarray(np.exp(-d / length), dtype=np.float64)


def depth_correlation(z: FloatArray, length_m: float) -> FloatArray:
    z = np.asarray(z, dtype=float)
    return exponential_correlation(np.abs(z[:, None] - z[None, :]), length_m)


def azimuth_correlation(phi: FloatArray, length_rad: float) -> FloatArray:
    """Корреляция по азимуту через хордовое расстояние (периодична, без шва)."""
    phi = np.asarray(phi, dtype=float)
    chord = 2.0 * np.abs(np.sin(0.5 * (phi[:, None] - phi[None, :])))
    chord_len = 2.0 * np.sin(0.5 * min(length_rad, np.pi)) if length_rad > 0 else 0.0
    return exponential_correlation(chord, chord_len)


def _cholesky(corr: FloatArray) -> FloatArray:
    n = corr.shape[0]
    return np.asarray(np.linalg.cholesky(corr + JITTER * np.eye(n)), dtype=np.float64)


@dataclass(frozen=True)
class GaussianField:
    """Генератор стандартного гауссова поля на узлах (z, φ)."""

    z: FloatArray
    phi: FloatArray
    length_z_m: float
    length_phi_rad: float
    _lz: FloatArray = field(init=False, repr=False)
    _lphi: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lz", _cholesky(depth_correlation(self.z, self.length_z_m)))
        corr_phi = azimuth_correlation(self.phi, self.length_phi_rad)
        object.__setattr__(self, "_lphi", _cholesky(corr_phi))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.z), len(self.phi))

    def sample(self, rng: np.random.Generator) -> FloatArray:
        w = rng.standard_normal(self.shape)
        return np.asarray(self._lz @ w @ self._lphi.T, dtype=np.float64)


def lognormal_multiplier(g: FloatArray, sigma: float) -> FloatArray:
    """exp(σ·g − σ²/2): логнормальный множитель со средним 1 и медианой exp(−σ²/2)."""
    return np.asarray(np.exp(sigma * np.asarray(g) - 0.5 * sigma * sigma), dtype=np.float64)


def bounded_fraction(g: FloatArray, mean: float, sd: float, upper: float = 0.6) -> FloatArray:
    """Доля в [0, upper]: f̄ + σ·g с обрезкой."""
    return np.asarray(np.clip(mean + sd * np.asarray(g), 0.0, upper), dtype=np.float64)
