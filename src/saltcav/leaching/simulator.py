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
перейти больше соли, чем примет до C_s рассол, бывший в слое до шага (прирост
объёма заполняется рассолом соседей). Последнее важно в начале,
когда объёмы слоёв у скважины малы, а площадь стенки относительно велика.
Связь явная, первый порядок по времени.

Колонна рассола начинается от уровня нерастворителя текущей ступени: слои выше
заняты нерастворителем и в переносе не участвуют. При подъёме нерастворителя
освободившиеся слои входят в колонну со своей последней концентрацией.

Нерастворимые (:mod:`saltcav.leaching.insolubles`): каждый шаг твёрдая доля
f_ins растворённой породы оседает на дно насыпью с коэффициентом разрыхления B.
Слои, погребённые насыпью, выходят из колонны: твёрдое вытесняет к отбору долю
1/B их рассола, остальное остаётся в порах. Стенка ниже насыпи не растворяется.
Башмак, оказавшийся в насыпи, переносится на первый слой над ней (как подъём
колонны на промысле); число таких шагов пишется в историю.

Струя и плюм (:mod:`saltcav.leaching.plume`, включено по умолчанию): пресная вода
выходит из башмака вниз, разворачивается, всплывает с вовлечением рассола и
отдаётся на уровне нейтральной плавучести; перенос — неявная консервативная
схема с обратным течением между слоями. При ``plume: false`` — прежняя модель
переноса от закачки к отбору с мгновенным перемешиванием.

Прослои (:mod:`saltcav.leaching.ledges`): ангидрит и глина не растворяются и
остаются карнизами; карниз с вылетом больше критического обрушается в начале шага,
обломки уходят в насыпь. Место карниза заполняет рассол с концентрацией слоя
(накопленная соль — ``collapse_fill_salt``; баланс: растворено + заполнение =
прирост в камере + вынесено).

Ограничения этой версии: турбулентная диффузия между слоями не
учитывается (её роль частично играет перемешивание неустойчивых слоёв).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.config import RunConfig
from saltcav.core.units import days_to_seconds
from saltcav.geometry import Cavern, CavernGrid
from saltcav.leaching.brine_column import column_step, plume_transport_step, stabilize
from saltcav.leaching.dissolution import DissolutionLaw, node_concentration, wall_growth_rate
from saltcav.leaching.insolubles import first_buried_layer, floor_depth
from saltcav.leaching.ledges import CollapseEvent, collapse_ledges, interlayers_from_config
from saltcav.leaching.plume import plume_fluxes
from saltcav.leaching.properties import WallProperties
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
    salt_in_brine: list[float] = field(default_factory=list)  # в камере, вкл. поры насыпи, кг
    floor_depth: list[float] = field(default_factory=list)  # верх насыпи, м
    insoluble_bulk: list[float] = field(default_factory=list)  # насыпной объём, м³
    n_steps: int = 0
    n_shoe_moves: int = 0  # шагов, где башмак пришлось поднять над насыпью
    plume_deposit_depth: list[float] = field(default_factory=list)  # по шагам, м
    collapse_fill_salt: list[float] = field(default_factory=list)  # накоплено, кг
    collapses: list[tuple[float, CollapseEvent]] = field(default_factory=list)  # (t, событие)
    # для просмотра по шагам (в моменты записи)
    wall_rate: list[FloatArray] = field(default_factory=list)  # (nz+1, nphi), м/с
    outlet_concentration: list[float] = field(default_factory=list)  # кг/м³
    shoe_in_depth: list[float] = field(default_factory=list)  # м, с учётом подъёма над насыпью
    shoe_out_depth: list[float] = field(default_factory=list)  # м
    blanket_depth: list[float] = field(default_factory=list)  # м
    stage_index: list[int] = field(default_factory=list)
    shoe_move_times: list[float] = field(default_factory=list)  # с, начало подъёма башмака
    props: WallProperties | None = None

    def record(
        self,
        t: float,
        cav: Cavern,
        c: FloatArray,
        state: _State,
        ops: _Ops,
    ) -> None:
        self.times.append(t)
        self.radius.append(np.array(cav.radius))
        self.concentration.append(np.array(c))
        self.volume.append(cav.volume())
        self.produced_salt.append(state.produced)
        self.dissolved_salt.append(state.dissolved)
        self.salt_in_brine.append(state.salt_in_brine(cav, c))
        self.floor_depth.append(state.z_floor)
        self.insoluble_bulk.append(state.bulk)
        self.collapse_fill_salt.append(state.fill)
        self.wall_rate.append(np.array(ops.rate))
        self.outlet_concentration.append(float(c[ops.k_out]))
        z = self.grid.z_centers
        self.shoe_in_depth.append(float(z[ops.k_in]))
        self.shoe_out_depth.append(float(z[ops.k_out]))
        self.blanket_depth.append(float(self.grid.z_edges[ops.k_blanket]))
        self.stage_index.append(ops.stage)

    def cavern(self, i: int = -1) -> Cavern:
        return Cavern(self.grid, self.radius[i])


def _layer_volumes(cav: Cavern) -> FloatArray:
    return np.asarray(cav.layer_volumes().sum(axis=1), dtype=np.float64)


@dataclass
class _Ops:
    """Режим и скорость стенки на последнем шаге — для записи истории."""

    rate: FloatArray  # (nz+1, nphi), м/с
    k_in: int
    k_out: int
    k_blanket: int
    stage: int


@dataclass
class _State:
    """Накопленные величины прогона помимо формы и концентрации."""

    produced: float  # кг соли, вынесено
    dissolved: float  # кг соли, растворено
    bulk: float  # м³, насыпной объём нерастворимых
    z_floor: float  # м, верх насыпи
    k_floor: int  # первый погребённый слой (nz — нет погребённых)
    trapped: float = 0.0  # кг соли в порах насыпи
    fill: float = 0.0  # кг соли в рассоле, заполнившем место обрушенных карнизов

    def salt_in_brine(self, cav: Cavern, c: FloatArray) -> float:
        v = _layer_volumes(cav)
        k = self.k_floor
        return float(np.dot(v[:k], c[:k])) + self.trapped


def _bury(st: _State, grid: CavernGrid, v: FloatArray, c: FloatArray, porosity: float) -> None:
    """Пересчитать уровень насыпи; погребённые слои выходят из колонны рассола."""
    st.z_floor = floor_depth(grid.z_edges, v, st.bulk)
    kf_new = min(first_buried_layer(grid.z_centers, st.z_floor), st.k_floor)
    for k in range(kf_new, st.k_floor):
        salt_k = float(v[k] * c[k])
        st.trapped += porosity * salt_k  # остаётся в порах
        st.produced += (1.0 - porosity) * salt_k  # вытеснено твёрдым к отбору
    st.k_floor = kf_new


def simulate(cfg: RunConfig, props: WallProperties | None = None) -> History:
    """Прогнать весь график режима для одной камеры.

    ``props`` — свойства стенки по узлам (z, φ); по умолчанию строятся из конфигурации.
    Ансамбль передаёт сюда поля одной реализации.
    """
    grid = CavernGrid.from_config(cfg.grid)
    schedule = Schedule.from_config(cfg.schedule, grid)
    lp = cfg.leaching
    law = DissolutionLaw(lp.k0_m_per_s, lp.exponent, lp.roof_ratio, lp.floor_ratio)
    c_sat = float(brine.saturation_concentration(lp.temperature_c))
    if props is None:
        props = WallProperties.from_config(cfg.geology, grid)
    solid = props.solid_nodes
    interlayers = interlayers_from_config(cfg.geology, grid)

    cav = Cavern.cylinder(grid, cfg.grid.r_borehole_m)
    c = np.full(grid.nz, lp.initial_saturation * c_sat)
    geo = cfg.geology
    porosity = 1.0 - 1.0 / geo.bulking_factor
    st = _State(0.0, 0.0, 0.0, float(grid.z_edges[-1]), grid.nz)
    hist = History(grid, props=props)
    s0 = schedule.stages[0]
    ops = _Ops(np.zeros_like(cav.radius), s0.k_in, s0.k_out, s0.k_blanket, 0)
    hist.record(0.0, cav, c, st, ops)

    dt_max = days_to_seconds(lp.dt_max_days)
    dt_out = days_to_seconds(lp.output_every_days)
    next_out = dt_out
    dz_min = float(np.min(grid.dz))
    node_idx = np.arange(grid.nz + 1)

    t = 0.0
    last_move = -np.inf
    for i_stage, stage in enumerate(schedule.stages):
        kb = stage.k_blanket
        while t < stage.t_end - 1e-9:
            # карнизы с вылетом больше критического обрушаются (прослои над насыпью)
            live = [il for il in interlayers if il.i1 < st.k_floor]
            if live:
                r_new, events = collapse_ledges(cav.radius, live, grid.z_edges)
                if events:
                    v_before = _layer_volumes(cav)
                    cav = cav.with_radius(r_new)
                    v_after = _layer_volumes(cav)
                    dv = v_after - v_before
                    # место карниза занимает рассол слоя (с той же концентрацией)
                    st.fill += float(np.dot(dv, c))
                    st.bulk += geo.bulking_factor * float(dv.sum())  # обломки в насыпь
                    _bury(st, grid, v_after, c, porosity)
                    hist.collapses.extend((t, ev) for ev in events)

            kf = st.k_floor
            if kf <= kb + 1:
                raise RuntimeError(
                    f"камера заполнена нерастворимыми до нерастворителя, t = {t:.0f} с"
                )
            # растворяются узлы строго между нерастворителем (узел kb) и насыпью (узел kf)
            active = (node_idx > kb) & (node_idx < kf)
            rate, v_max = wall_growth_rate(
                cav.radius, grid.dz, node_concentration(c), c_sat, law, solid, active,
                props.k_mult,
            )  # fmt: skip
            dt = min(dt_max, stage.t_end - t, next_out - t)
            if v_max > 0:
                dt = min(dt, lp.cfl * dz_min / v_max)

            v_old = _layer_volumes(cav)
            c_cap = np.maximum(c, c_sat) * (1.0 + 1e-9)
            while True:
                new_cav = cav.with_radius(cav.radius + dt * rate)
                v_new = _layer_volumes(new_cav)
                dv_sec = new_cav.layer_volumes() - cav.layer_volumes()
                dm = props.dissolved_salt(dv_sec)
                growth = float(v_new.sum() - v_old.sum())
                # 1) отбор не становится отрицательным; 2) явный шаг не растворяет больше,
                # чем может принять рассол, бывший в слое до шага: прирост объёма
                # заполняется рассолом соседей, а не пустотой (иначе C > C_s)
                ok_growth = growth <= lp.max_growth_fraction * stage.q_in * dt
                ok_sat = bool(np.all(v_old * c + dm <= c_cap * v_old))
                if ok_growth and ok_sat:
                    break
                if dt < MIN_DT:
                    raise RuntimeError(f"шаг по времени упал ниже {MIN_DT} с при t = {t:.0f} с")
                dt *= 0.5

            # башмак в насыпи поднимается на первый слой над ней
            k_in, k_out = min(stage.k_in, kf - 1), min(stage.k_out, kf - 1)
            if (k_in, k_out) != (stage.k_in, stage.k_out):
                if t - last_move > dt_out:
                    hist.shoe_move_times.append(t)  # начало эпизода подъёма
                last_move = t
                hist.n_shoe_moves += 1

            # колонна рассола — слои между нерастворителем и насыпью
            if lp.plume:
                col = slice(kb, kf)
                dz_col = grid.dz[col]
                r_col = np.sqrt(v_old[col] / (np.pi * dz_col))  # эквивалентный радиус слоя
                pl = plume_fluxes(
                    c[col], dz_col, r_col, k_in - kb, stage.q_in,
                    lp.tubing_radius_m, lp.entrainment_coeff, lp.temperature_c,
                )  # fmt: skip
                hist.plume_deposit_depth.append(float(grid.z_centers[kb + pl.k_deposit]))
                raw = plume_transport_step(
                    c[col], v_old[col], v_new[col], dm[col], stage.q_in,
                    pl.entrained, pl.k_deposit, k_out - kb, dt,
                )  # fmt: skip
                res = type(raw)(stabilize(raw.c, v_new[col]), raw.produced_salt, raw.q_out)
            else:
                res = column_step(
                    c[kb:kf], v_old[kb:kf], v_new[kb:kf], dm[kb:kf], stage.q_in,
                    k_in - kb, k_out - kb, dt,
                )  # fmt: skip
            c = np.concatenate([c[:kb], res.c, c[kf:]])
            cav = new_cav
            st.produced += res.produced_salt
            st.dissolved += float(dm.sum())

            # нерастворимые оседают; погребённые слои выходят из колонны
            st.bulk += geo.bulking_factor * props.insoluble_volume(dv_sec)
            _bury(st, grid, v_new, c, porosity)

            t += dt
            hist.n_steps += 1
            if t >= next_out - 1e-9 or t >= schedule.total_duration - 1e-9:
                hist.record(t, cav, c, st, _Ops(rate, k_in, k_out, kb, i_stage))
                next_out = t + dt_out
    return hist
