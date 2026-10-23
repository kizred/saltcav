"""График технологического режима: ступени во времени и их привязка к сетке.

Время внутри модели — секунды от начала выщелачивания. Ступень i действует на
полуинтервале [t_i, t_{i+1}); после окончания последней ступени режим не задан.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saltcav.core.config import LeachMode, ScheduleConfig, StageConfig
from saltcav.geometry import CavernGrid


@dataclass(frozen=True)
class Stage:
    """Ступень режима в СИ с индексами слоёв сетки."""

    t_start: float  # с
    t_end: float  # с
    mode: LeachMode
    q_in: float  # м³/с, расход закачки пресной воды
    k_in: int  # слой башмака закачки
    k_out: int  # слой башмака отбора
    k_blanket: int  # первый слой ниже уровня нерастворителя

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def layer_index(grid: CavernGrid, depth: float) -> int:
    """Индекс слоя, содержащего глубину depth (граница относится к нижнему слою)."""
    z = grid.z_edges
    if not z[0] <= depth <= z[-1]:
        raise ValueError(f"глубина {depth} вне сетки [{z[0]}, {z[-1]}]")
    k = int(np.searchsorted(z, depth, side="right")) - 1
    return min(k, grid.nz - 1)


@dataclass(frozen=True)
class Schedule:
    stages: tuple[Stage, ...]

    @classmethod
    def from_config(cls, cfg: ScheduleConfig, grid: CavernGrid) -> Schedule:
        stages: list[Stage] = []
        t = 0.0
        for st in cfg.stages:
            stages.append(_stage(st, t, grid))
            t += st.duration_s
        return cls(tuple(stages))

    @property
    def total_duration(self) -> float:
        return self.stages[-1].t_end

    @property
    def boundaries(self) -> np.ndarray:
        """Моменты смены ступеней, включая начало и конец, с."""
        return np.array([s.t_start for s in self.stages] + [self.total_duration])

    def stage_at(self, t: float) -> Stage:
        if not 0.0 <= t < self.total_duration:
            raise ValueError(f"момент {t} вне графика [0, {self.total_duration})")
        i = int(np.searchsorted(self.boundaries, t, side="right") - 1)
        return self.stages[i]


def _stage(st: StageConfig, t0: float, grid: CavernGrid) -> Stage:
    return Stage(
        t_start=t0,
        t_end=t0 + st.duration_s,
        mode=st.mode,
        q_in=st.rate_si,
        k_in=layer_index(grid, st.injection_depth_m),
        k_out=layer_index(grid, st.production_depth_m),
        k_blanket=layer_index(grid, st.blanket_depth_m),
    )
