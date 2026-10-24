"""Свойства стенки в узлах (z, φ): множитель скорости растворения и доля нерастворимых.

Массивы задаются в узлах стенки — там же, где радиус R, форма (nz+1, nphi).
Значения на слое берутся как среднее двух его узлов. Сейчас поля строятся из
конфигурации (однородные или с косинусной неоднородностью по азимуту); на этапе 5
их будут генерировать случайные поля.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import GeologyConfig
from saltcav.geometry import CavernGrid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WallProperties:
    k_mult: FloatArray  # (nz+1, nphi), множитель скорости растворения, > 0
    insoluble: FloatArray  # (nz+1, nphi), доля нерастворимых в породе, 0…1
    salt_density: float  # кг/м³, плотность каменной соли

    def __post_init__(self) -> None:
        if self.k_mult.shape != self.insoluble.shape:
            raise ValueError("формы k_mult и insoluble должны совпадать")
        if np.any(self.k_mult <= 0) or np.any((self.insoluble < 0) | (self.insoluble >= 1)):
            raise ValueError("k_mult > 0, 0 ≤ insoluble < 1")

    @classmethod
    def from_config(cls, geo: GeologyConfig, grid: CavernGrid) -> WallProperties:
        shape = (grid.nz + 1, grid.nphi)
        phi0 = np.deg2rad(geo.k_variation_azimuth_deg)
        k_phi = 1.0 + geo.k_variation * np.cos(grid.phi_centers - phi0)
        return cls(
            k_mult=np.broadcast_to(k_phi, shape).astype(float),
            insoluble=np.full(shape, geo.insoluble_fraction),
            salt_density=geo.salt_density_kg_m3,
        )

    @property
    def solid_nodes(self) -> FloatArray:
        """ρ_s·(1 − f_ins) в узлах: кг растворимой соли на м³ породы."""
        return np.asarray(self.salt_density * (1.0 - self.insoluble), dtype=np.float64)

    @property
    def insoluble_layers(self) -> FloatArray:
        """Доля нерастворимых на слоях (nz, nphi) — среднее узлов."""
        return np.asarray(0.5 * (self.insoluble[:-1] + self.insoluble[1:]), dtype=np.float64)

    def dissolved_salt(self, dv_sectors: FloatArray) -> FloatArray:
        """Растворённая соль по слоям (nz,) для приращений объёма по секторам (nz, nphi)."""
        solid = self.salt_density * (1.0 - self.insoluble_layers)
        return np.asarray((solid * dv_sectors).sum(axis=1), dtype=np.float64)

    def insoluble_volume(self, dv_sectors: FloatArray) -> float:
        """Твёрдый объём нерастворимых, высвобожденный при растворении, м³."""
        return float((self.insoluble_layers * dv_sectors).sum())
