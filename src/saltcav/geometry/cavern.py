"""Геометрия камеры: стенка R(z, φ) на сетке «узлы по глубине × секторы по азимуту».

Представление
-------------
* Глубина ``z`` отсчитывается вниз. Узлы ``z_edges`` (nz+1) делят интервал на
  nz слоёв; в слоях позже живёт концентрация рассола C(z).
* Азимут разбит на ``nphi`` равных секторов шириной Δφ = 2π/nphi; внутри
  сектора радиус не зависит от φ. ``nphi = 1`` — осесимметричная камера.
* ``R[i, j]`` — радиус стенки в узле глубины i, секторе j, м. Между соседними
  узлами стенка — усечённый конус (фрустум). Для цилиндра и конуса это точно,
  для гладких форм даёт второй порядок сходимости по шагу Δz.
* Радиальные грани между секторами (ступенька по φ) в площадь стенки не входят:
  в секторной модели растворяется только «фасад» каждого сектора.

Угол ориентации стенки
----------------------
``psi`` — угол между нормалью, направленной из рассола в соль (направление
отступания стенки), и вертикалью вверх:

* ψ = 0°   — кровля (стенка уходит вверх, растворяется быстрее всего);
* ψ = 90°  — вертикальная стенка;
* ψ = 180° — подошва.

Для фрустума между узлами (R₁, z₁) и (R₂, z₂), z₂ > z₁: cos ψ = (R₂ − R₁)/s,
где s = √(Δz² + (R₂ − R₁)²). Расширение камеры книзу (R₂ > R₁) — нависающая
кровля, ψ < 90°.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import GridConfig

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CavernGrid:
    """Неизменяемая сетка: узлы по глубине и число азимутальных секторов."""

    z_edges: FloatArray
    nphi: int

    def __post_init__(self) -> None:
        z = np.asarray(self.z_edges, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("нужно минимум два узла по глубине")
        if np.any(np.diff(z) <= 0):
            raise ValueError("узлы z должны строго возрастать (глубина вниз)")
        if self.nphi < 1:
            raise ValueError("nphi ≥ 1")
        z.setflags(write=False)
        object.__setattr__(self, "z_edges", z)

    @classmethod
    def uniform(cls, z_top: float, z_bottom: float, nz: int, nphi: int = 1) -> CavernGrid:
        return cls(np.linspace(z_top, z_bottom, nz + 1), nphi)

    @classmethod
    def from_config(cls, cfg: GridConfig) -> CavernGrid:
        return cls.uniform(cfg.z_top_m, cfg.z_bottom_m, cfg.nz, cfg.nphi)

    @property
    def nz(self) -> int:
        """Число слоёв."""
        return self.z_edges.size - 1

    @property
    def dz(self) -> FloatArray:
        return np.diff(self.z_edges)

    @property
    def z_centers(self) -> FloatArray:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def dphi(self) -> float:
        return 2.0 * np.pi / self.nphi

    @property
    def phi_centers(self) -> FloatArray:
        return (np.arange(self.nphi, dtype=np.float64) + 0.5) * self.dphi


@dataclass(frozen=True)
class Cavern:
    """Форма камеры в один момент времени. Изменение формы — новый объект."""

    grid: CavernGrid
    radius: FloatArray  # (nz+1, nphi), м

    def __post_init__(self) -> None:
        r = np.array(self.radius, dtype=float)  # копия: объект не зависит от входного массива
        if r.ndim == 1:
            r = r[:, None]
        expected = (self.grid.nz + 1, self.grid.nphi)
        if r.shape != expected:
            raise ValueError(f"форма R {r.shape}, ожидалось {expected}")
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("R должен быть конечным и неотрицательным")
        r.setflags(write=False)
        object.__setattr__(self, "radius", r)

    # --- конструкторы ---------------------------------------------------------

    @classmethod
    def from_profile(
        cls, grid: CavernGrid, profile: Callable[[FloatArray, FloatArray], FloatArray]
    ) -> Cavern:
        """Камера по функции R(z, φ), вычисленной в узлах z и центрах секторов φ."""
        zz, pp = np.meshgrid(grid.z_edges, grid.phi_centers, indexing="ij")
        return cls(grid, np.asarray(profile(zz, pp), dtype=float))

    @classmethod
    def cylinder(cls, grid: CavernGrid, radius: float) -> Cavern:
        """Начальное состояние: скважина или цилиндрическая затравка."""
        return cls(grid, np.full((grid.nz + 1, grid.nphi), float(radius)))

    def with_radius(self, radius: FloatArray) -> Cavern:
        return Cavern(self.grid, radius)

    # --- геометрия фрустумов (nz, nphi) ---------------------------------------

    @property
    def _r1(self) -> FloatArray:
        return self.radius[:-1]

    @property
    def _r2(self) -> FloatArray:
        return self.radius[1:]

    @property
    def _h(self) -> FloatArray:
        return self.grid.dz[:, None]

    def layer_volumes(self) -> FloatArray:
        """Объём каждого слоя в каждом секторе, м³: Δφ·h/6·(R₁² + R₁R₂ + R₂²)."""
        r1, r2 = self._r1, self._r2
        return self.grid.dphi * self._h / 6.0 * (r1 * r1 + r1 * r2 + r2 * r2)

    def volume(self) -> float:
        return float(self.layer_volumes().sum())

    def slant_lengths(self) -> FloatArray:
        """Длина образующей фрустума s, м."""
        return np.hypot(self._h, self._r2 - self._r1)

    def wall_areas(self) -> FloatArray:
        """Площадь боковой стенки слоя в секторе, м²: Δφ/2·(R₁ + R₂)·s."""
        return 0.5 * self.grid.dphi * (self._r1 + self._r2) * self.slant_lengths()

    def wall_angles(self) -> FloatArray:
        """Угол ψ, рад: 0 — кровля, π/2 — вертикаль, π — подошва (см. модуль)."""
        cos_psi = (self._r2 - self._r1) / self.slant_lengths()
        return np.arccos(np.clip(cos_psi, -1.0, 1.0))

    def wall_normals(self) -> tuple[FloatArray, FloatArray]:
        """Единичная нормаль «в соль» в плоскости r–z: (n_r, n_z), z вниз."""
        s = self.slant_lengths()
        return self._h / s, -(self._r2 - self._r1) / s

    def overhang_mask(self, threshold_deg: float = 90.0) -> BoolArray:
        """Участки, обращённые вверх сильнее порога (карнизы и кровля)."""
        return np.asarray(self.wall_angles() < np.deg2rad(threshold_deg), dtype=np.bool_)

    # --- горизонтальные торцы -------------------------------------------------

    def roof_area(self) -> float:
        """Площадь верхнего торца (кровля над верхним узлом), м²."""
        return float(0.5 * self.grid.dphi * np.sum(self.radius[0] ** 2))

    def floor_area(self) -> float:
        """Площадь нижнего торца (подошва под нижним узлом), м²."""
        return float(0.5 * self.grid.dphi * np.sum(self.radius[-1] ** 2))

    def surface_area(self) -> float:
        """Полная площадь поверхности: стенка + торцы, м²."""
        return float(self.wall_areas().sum()) + self.roof_area() + self.floor_area()

    # --- сводные характеристики -----------------------------------------------

    def max_radius(self) -> float:
        return float(np.max(self.radius))

    def equivalent_radius(self) -> FloatArray:
        """Радиус круга той же площади сечения в каждом узле, м."""
        return np.asarray(np.sqrt(np.mean(self.radius**2, axis=1)), dtype=np.float64)

    def centroid_offset(self) -> FloatArray:
        """Смещение центра сечения от оси скважины в каждом узле (x, y), м.

        Для секторной формы центр масс сектора радиуса R: 2R·sin(Δφ/2)/(3Δφ/2)
        вдоль биссектрисы; ось — центр скважины. Для nphi = 1 смещение нулевое.
        """
        g = self.grid
        if g.nphi == 1:
            return np.zeros((g.nz + 1, 2))
        half = 0.5 * g.dphi
        r = self.radius
        area = half * r**2  # площадь сектора
        arm = 2.0 * r * np.sin(half) / (3.0 * half)
        total = area.sum(axis=1)
        safe = np.where(total > 0, total, 1.0)
        x = (area * arm * np.cos(g.phi_centers)).sum(axis=1) / safe
        y = (area * arm * np.sin(g.phi_centers)).sum(axis=1) / safe
        return np.stack([x, y], axis=1)
