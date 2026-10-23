"""Детерминированный прогон выщелачивания одной камеры.

Шаг по времени связывает две части модели:

1. стенка отступает по закону растворения при концентрациях начала шага
   (:mod:`saltcav.leaching.dissolution`);
2. растворённая соль Δm_k = ρ_s(1 − f_ins)·ΔV_k поступает в слои рассола, затем
   перенос от закачки к отбору и плотностное перемешивание
   (:mod:`saltcav.leaching.brine_column`).

Шаг выбирается адаптивно: условие КФЛ для стенки, верхняя граница dt_max,
граница ступени режима, ограничение роста объёма ΔV ≤ доля·Q·dt (иначе отбор
становится отрицательным) и запрет пересыщения: за шаг в слой не может
перейти больше соли, чем примет его рассол до C_s. Последнее важно в начале,
когда объёмы слоёв у скважины малы, а площадь стенки относительно велика.
Связь явная, первый порядок по времени.

Колонна рассола начинается от уровня нерастворителя текущей ступени: слои выше
заняты нерастворителем и в переносе не участвуют. При подъёме нерастворителя
освободившиеся слои входят в колонну со своей последней концентрацией.

Ограничения этой версии: нерастворимые не накапливаются на подошве (их объём
пока считается рассолом), струя из колонны не моделируется, прослоев нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.config import RunConfig
from saltcav.core.units import days_to_seconds
from saltcav.geometry import Cavern, CavernGrid
from saltcav.leaching.brine_column import column_step
from saltcav.leaching.dissolution import DissolutionLaw, node_concentration, wall_growth_rate
from saltcav.operations import Schedule

FloatArray = NDArray[np.float64]

MIN_DT = 1e-3  # с; меньший шаг означает ошибку в постановке, а не жёсткость задачи


@dataclass
class History:
    """Сохранённые моменты прогона. Массивы по времени — первая ось."""

    grid: CavernGrid
    times: list[float] = field(default_factory=list)  # с
    radius: list[FloatArray] = field(default_factory=list)  # (nz+1, nphi)
    concentration: list[FloatArray] = field(default_factory=list)  # (nz,)
    volume: list[float] = field(default_factory=list)  # м³
    produced_salt: list[float] = field(default_factory=list)  # накоплено, кг
    dissolved_salt: list[float] = field(default_factory=list)  # накоплено, кг
    n_steps: int = 0

    def record(
        self, t: float, cav: Cavern, c: FloatArray, produced: float, dissolved: float
    ) -> None:
        self.times.append(t)
        self.radius.append(np.array(cav.radius))
        self.concentration.append(np.array(c))
        self.volume.append(cav.volume())
        self.produced_salt.append(produced)
        self.dissolved_salt.append(dissolved)

    def cavern(self, i: int = -1) -> Cavern:
        return Cavern(self.grid, self.radius[i])


def _layer_volumes(cav: Cavern) -> FloatArray:
    return np.asarray(cav.layer_volumes().sum(axis=1), dtype=np.float64)


def simulate(cfg: RunConfig) -> History:
    """Прогнать весь график режима для одной камеры с параметрами из конфигурации."""
    grid = CavernGrid.from_config(cfg.grid)
    schedule = Schedule.from_config(cfg.schedule, grid)
    lp = cfg.leaching
    law = DissolutionLaw(lp.k0_m_per_s, lp.exponent, lp.roof_ratio, lp.floor_ratio)
    c_sat = float(brine.saturation_concentration(lp.temperature_c))
    solid = cfg.geology.salt_density_kg_m3 * (1.0 - cfg.geology.insoluble_fraction)

    cav = Cavern.cylinder(grid, cfg.grid.r_borehole_m)
    c = np.full(grid.nz, lp.initial_saturation * c_sat)
    produced = dissolved = 0.0
    hist = History(grid)
    hist.record(0.0, cav, c, produced, dissolved)

    dt_max = days_to_seconds(lp.dt_max_days)
    dt_out = days_to_seconds(lp.output_every_days)
    next_out = dt_out
    dz_min = float(np.min(grid.dz))

    t = 0.0
    for stage in schedule.stages:
        kb = stage.k_blanket
        # растворяются узлы строго ниже уровня нерастворителя: узел kb — граница с ним,
        # иначе прирост объёма слоя kb−1 (над нерастворителем) получил бы соль
        active = grid.z_edges > grid.z_edges[kb]
        while t < stage.t_end - 1e-9:
            rate, v_max = wall_growth_rate(
                cav.radius, grid.dz, node_concentration(c), c_sat, law, solid, active
            )
            dt = min(dt_max, stage.t_end - t, next_out - t)
            if v_max > 0:
                dt = min(dt, lp.cfl * dz_min / v_max)

            v_old = _layer_volumes(cav)
            c_cap = np.maximum(c, c_sat) * (1.0 + 1e-9)
            while True:
                new_cav = cav.with_radius(cav.radius + dt * rate)
                v_new = _layer_volumes(new_cav)
                dm = solid * (v_new - v_old)
                growth = float(v_new.sum() - v_old.sum())
                # 1) отбор не становится отрицательным; 2) явный шаг не растворяет больше,
                # чем может принять рассол слоя (иначе C > C_s в тонких слоях у скважины)
                ok_growth = growth <= lp.max_growth_fraction * stage.q_in * dt
                ok_sat = bool(np.all((v_old * c + dm) / v_new <= c_cap))
                if ok_growth and ok_sat:
                    break
                if dt < MIN_DT:
                    raise RuntimeError(f"шаг по времени упал ниже {MIN_DT} с при t = {t:.0f} с")
                dt *= 0.5

            # колонна рассола — только слои ниже нерастворителя; выше стоит сам
            # нерастворитель, там концентрация не меняется
            res = column_step(
                c[kb:], v_old[kb:], v_new[kb:], dm[kb:], stage.q_in,
                stage.k_in - kb, stage.k_out - kb, dt,
            )  # fmt: skip
            c = np.concatenate([c[:kb], res.c])
            cav = new_cav
            produced += res.produced_salt
            dissolved += float(dm.sum())
            t += dt
            hist.n_steps += 1
            if t >= next_out - 1e-9 or t >= schedule.total_duration - 1e-9:
                hist.record(t, cav, c, produced, dissolved)
                next_out = t + dt_out
    return hist
