"""Схема конфигурации расчёта (YAML → проверенные pydantic-модели).

Единицы указаны в имени поля (``_m``, ``_m3_per_h``, ``_days``); перевод в СИ
делают свойства ``*_si``. Конфигурация каждого прогона сохраняется рядом с
результатом, поэтому схема строгая: лишние поля запрещены.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from saltcav.core import units


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GridConfig(_Strict):
    """Расчётная сетка стенки R(z, φ). Глубина z отсчитывается вниз от поверхности."""

    z_top_m: Annotated[float, Field(gt=0, description="Глубина верха расчётной сетки, м")]
    z_bottom_m: Annotated[float, Field(gt=0, description="Глубина низа сетки, м (глубже z_top_m)")]
    nz: Annotated[int, Field(ge=2, description="Число слоёв по глубине (шаг = высота / nz)")]
    nphi: Annotated[
        int, Field(ge=1, description="Число секторов по азимуту; 1 — осесимметрия, 12–36 — 3D")
    ] = 1
    r_borehole_m: Annotated[
        float, Field(gt=0, description="Начальный радиус ствола скважины, м")
    ] = 0.15

    @model_validator(mode="after")
    def _check_depths(self) -> GridConfig:
        if self.z_bottom_m <= self.z_top_m:
            raise ValueError("z_bottom_m должна быть глубже z_top_m")
        return self


class LayerKind(StrEnum):
    halite = "halite"
    anhydrite = "anhydrite"
    clay = "clay"


class LayerConfig(_Strict):
    """Прослой; для стохастики задаётся неопределённость положения и мощности.

    Ангидрит и глина не растворяются: соль над и под ними уходит, прослой остаётся
    карнизом. Карниз обрушивается, когда его вылет превышает критический:
    ``critical_span_m`` или, если не задан, ``ledge_span_ratio × мощность``.
    """

    kind: Annotated[
        LayerKind, Field(description="Порода: anhydrite, clay — не растворяются; halite — соль")
    ]
    top_m: Annotated[float, Field(gt=0, description="Глубина кровли прослоя, м")]
    thickness_m: Annotated[float, Field(gt=0, description="Мощность прослоя, м")]
    top_sd_m: Annotated[
        float, Field(ge=0, description="Ансамбль: ст. отклонение отметки кровли, м")
    ] = 0.0
    thickness_sd_m: Annotated[
        float, Field(ge=0, description="Ансамбль: ст. отклонение мощности, м")
    ] = 0.0
    critical_span_m: Annotated[
        float | None,
        Field(
            gt=0, description="Критический вылет карниза, м; пусто — ledge_span_ratio × мощность"
        ),
    ] = None


class GeologyConfig(_Strict):
    """Свойства массива.

    Детерминированная неоднородность по азимуту: множитель скорости растворения
    m(φ) = 1 + a·cos(φ − φ₀). Случайные поля по z и φ — в разделе uncertainty.
    """

    insoluble_fraction: Annotated[
        float, Field(ge=0, le=1, description="Доля нерастворимых в породе (объёмная), 0…1")
    ] = 0.05
    insoluble_fraction_sd: Annotated[
        float, Field(ge=0, description="Ансамбль: ст. отклонение доли нерастворимых (поле)")
    ] = 0.0
    bulking_factor: Annotated[
        float, Field(ge=1, description="Коэффициент разрыхления нерастворимых в насыпи, ≥ 1")
    ] = 1.5
    salt_density_kg_m3: Annotated[
        float, Field(gt=0, description="Плотность каменной соли, кг/м³")
    ] = 2160.0
    k_variation: Annotated[
        float,
        Field(ge=0, lt=1, description="Асимметрия: амплитуда a в m(φ) = 1 + a·cos(φ − φ₀), 0…1"),
    ] = 0.0
    k_variation_azimuth_deg: Annotated[
        float, Field(description="Асимметрия: азимут φ₀ быстрого растворения, °")
    ] = 0.0
    ledge_span_ratio: Annotated[
        float, Field(gt=0, description="Критический вылет карниза / мощность прослоя")
    ] = 3.0
    layers: list[LayerConfig] = Field(
        default_factory=list, description="Нерастворимые прослои (ангидрит, глина)"
    )


class LeachMode(StrEnum):
    direct = "direct"  # прямоток: вода через центральную колонну, рассол по затрубу
    reverse = "reverse"  # противоток


class StageConfig(_Strict):
    """Ступень режима выщелачивания."""

    duration_days: Annotated[float, Field(gt=0, description="Длительность ступени, сут")]
    mode: Annotated[LeachMode, Field(description="direct — прямоток, reverse — противоток")]
    rate_m3_per_h: Annotated[float, Field(gt=0, description="Расход закачки пресной воды, м³/ч")]
    injection_depth_m: Annotated[
        float, Field(gt=0, description="Глубина башмака колонны закачки, м")
    ]
    production_depth_m: Annotated[
        float, Field(gt=0, description="Глубина башмака колонны отбора рассола, м")
    ]
    blanket_depth_m: Annotated[
        float,
        Field(gt=0, description="Уровень нерастворителя (кровля зоны растворения), м"),
    ]

    @property
    def duration_s(self) -> float:
        return units.days_to_seconds(self.duration_days)

    @property
    def rate_si(self) -> float:
        return units.m3_per_hour_to_si(self.rate_m3_per_h)


class ScheduleConfig(_Strict):
    """График режима выщелачивания."""

    stages: Annotated[
        list[StageConfig], Field(min_length=1, description="Ступени режима по порядку")
    ]


class EnsembleConfig(_Strict):
    """Ансамбль Монте-Карло (saltcav ensemble)."""

    n_realizations: Annotated[int, Field(ge=1, description="Число реализаций по умолчанию")] = 100
    seed: Annotated[
        int, Field(ge=0, description="Seed: тот же seed — тот же ансамбль бит в бит")
    ] = 12345
    sampler: str = Field(
        default="mc",
        pattern="^(mc|lhs|sobol)$",
        description="План: mc — Монте-Карло, lhs — латинский гиперкуб, sobol — Соболь",
    )


class LeachingConfig(_Strict):
    """Параметры закона растворения и численной схемы.

    Поток соли со стенки: j = k0 · f(ψ) · C_s · u^n, u = (C_s − C)/C_s.
    k0 по умолчанию подобран по бенчмарку SANSMIC (benchmarks/sansmic): объём камеры
    совпадает в пределах 1 %. Оценка по корреляции свободной конвекции (Sh ~ Ra^(1/3))
    даёт 1,5e-5 — тот же порядок. Это априорное значение: калибровать по физической
    модели и данным компании.
    """

    temperature_c: Annotated[float, Field(ge=0, le=100, description="Температура рассола, °C")] = (
        30.0
    )
    k0_m_per_s: Annotated[
        float, Field(gt=0, description="Коэффициент растворения k0, м/с (SANSMIC: 3e-5)")
    ] = 3.0e-5
    exponent: Annotated[float, Field(gt=0, description="Показатель n в u^n, u — недонасыщение")] = (
        4.0 / 3.0
    )
    roof_ratio: Annotated[
        float, Field(ge=0, description="Скорость растворения кровли / вертикальной стенки")
    ] = 1.5
    floor_ratio: Annotated[
        float, Field(ge=0, description="Скорость растворения подошвы / вертикальной стенки")
    ] = 0.0
    initial_saturation: Annotated[
        float, Field(ge=0, le=1, description="Начальное насыщение рассола в скважине C₀/C_s")
    ] = 1.0
    plume: bool = Field(
        default=True,
        description="Струя и плюм от башмака закачки; false — мгновенное перемешивание",
    )
    entrainment_coeff: Annotated[
        float, Field(ge=0, le=0.5, description="Коэффициент вовлечения плюма α (SANSMIC: 0,09)")
    ] = 0.09
    tubing_radius_m: Annotated[
        float, Field(gt=0, description="Внутренний радиус колонны закачки, м")
    ] = 0.1
    cfl: Annotated[float, Field(gt=0, le=1, description="Число Куранта для движения стенки")] = 0.5
    max_growth_fraction: Annotated[
        float, Field(gt=0, lt=1, description="Ограничение шага: прирост объёма ≤ доля·Q·dt")
    ] = 0.5
    dt_max_days: Annotated[float, Field(gt=0, description="Наибольший шаг по времени, сут")] = 1.0
    output_every_days: Annotated[
        float, Field(gt=0, description="Сохранять результаты каждые N сут (кадры просмотра)")
    ] = 10.0


class UncertaintyConfig(_Strict):
    """Неопределённость для ансамбля (этап 5). Нули — детерминированный прогон.

    Поля по (z, φ): ln k с σ = ``k_field_sigma``; доля нерастворимых с σ =
    ``geology.insoluble_fraction_sd``; общие длины корреляции. Скаляры на реализацию:
    k0 (логнормальный множитель), отношение кровля/стенка (равномерно в диапазоне),
    критический вылет карниза (логнормальный множитель), расход на каждой ступени
    (нормальный множитель); положение и мощность прослоев — ``top_sd_m``,
    ``thickness_sd_m`` у прослоя.
    """

    k_field_sigma: Annotated[
        float, Field(ge=0, le=2, description="Поле ln k по (z, φ): ст. отклонение σ")
    ] = 0.0
    corr_length_z_m: Annotated[
        float, Field(ge=0, description="Длина корреляции полей по глубине, м")
    ] = 10.0
    corr_length_phi_deg: Annotated[
        float, Field(ge=0, le=360, description="Длина корреляции полей по азимуту, °")
    ] = 60.0
    k0_log_sd: Annotated[float, Field(ge=0, le=2, description="k0: σ логнормального множителя")] = (
        0.0
    )
    roof_ratio_range: tuple[float, float] | None = Field(
        default=None, description="roof_ratio: равномерно в [нижняя, верхняя]"
    )
    ledge_span_log_sd: Annotated[
        float,
        Field(ge=0, le=2, description="Критический вылет карниза: σ логнормального множителя"),
    ] = 0.0
    rate_rel_sd: Annotated[
        float, Field(ge=0, le=0.5, description="Расход: относительное ст. отклонение по ступеням")
    ] = 0.0

    @model_validator(mode="after")
    def _check_range(self) -> UncertaintyConfig:
        r = self.roof_ratio_range
        if r is not None and not 0 <= r[0] <= r[1]:
            raise ValueError("roof_ratio_range: нужно 0 ≤ нижняя ≤ верхняя")
        return self


class CreepBranchConfig(_Strict):
    """Ветвь установившейся ползучести: ε̇ = A·exp(−Q/RT)·(σ/1 МПа)^n."""

    a_per_day: Annotated[float, Field(gt=0, description="A, 1/сут (BGRa: 0,18)")]
    n: Annotated[float, Field(gt=0, description="Показатель степени n (BGRa: 5)")]
    q_kj_per_mol: Annotated[
        float, Field(ge=0, description="Энергия активации Q, кДж/моль (BGRa: 54)")
    ]


class TransientConfig(_Strict):
    """Неустановившаяся ползучесть Мансона–Доусона (по умолчанию — соль WIPP)."""

    mu_gpa: Annotated[float, Field(gt=0, description="Модуль сдвига в законе, ГПа")] = 12.4
    k0: Annotated[float, Field(gt=0, description="K0 в ε_t* = K0·exp(cT)·(σ/μ)^m")] = 6.275e5
    c_per_k: Annotated[float, Field(description="c, 1/К")] = 9.198e-3
    m: Annotated[float, Field(gt=0, description="Показатель m")] = 3.0
    alpha_w: Annotated[float, Field(description="α_w — упрочнение")] = -17.37
    beta_w: Annotated[float, Field(description="β_w — упрочнение")] = -7.738
    delta: Annotated[float, Field(ge=0, description="δ — возврат")] = 0.58


class PressureStageConfig(_Strict):
    """Ступень эксплуатации камеры после выщелачивания."""

    duration_years: Annotated[float, Field(gt=0, description="Длительность, лет")]
    wellhead_pressure_mpa: Annotated[
        float, Field(ge=0, description="Давление на устье, МПа (рассол, открытая скважина — 0)")
    ] = 0.0
    fluid_density_kg_m3: Annotated[
        float, Field(gt=0, description="Плотность среды в камере, кг/м³ (рассол ≈ 1200)")
    ] = 1200.0
    label: str = Field(default="", description="Подпись ступени")


class MechanicsConfig(_Strict):
    """Геомеханика (этап 6): осесимметричный МКЭ вокруг формы камеры, ползучесть соли.

    Форма камеры — эквивалентный радиус √(среднее R²) по азимуту на выбранный момент.
    """

    day: Annotated[
        float | None, Field(ge=0, description="Момент формы, сут; пусто — конец выщелачивания")
    ] = None
    h_wall_m: Annotated[float, Field(gt=0, description="Размер элемента у стенки, м")] = 1.0
    r_min_m: Annotated[
        float, Field(gt=0, description="Радиус, ниже которого сечение считается массивом, м")
    ] = 0.5
    r_factor: Annotated[
        float, Field(gt=1, description="Внешний радиус области / наибольший радиус камеры")
    ] = 8.0
    height_factor: Annotated[
        float, Field(gt=0, description="Запас области сверху и снизу / размер камеры")
    ] = 3.0
    outer: str = Field(
        default="free",
        pattern="^(free|roller)$",
        description="Внешняя граница: free — одиночная камера, roller — ряд камер (ячейка)",
    )
    cell_radius_m: Annotated[
        float | None,
        Field(gt=0, description="Радиус ячейки (половина расстояния между камерами) для roller"),
    ] = None
    youngs_modulus_gpa: Annotated[float, Field(gt=0, description="Модуль Юнга соли, ГПа")] = 25.0
    poisson: Annotated[float, Field(gt=0, lt=0.5, description="Коэффициент Пуассона")] = 0.25
    overburden_density_kg_m3: Annotated[
        float, Field(gt=0, description="Средняя плотность пород над камерой, кг/м³")
    ] = 2300.0
    surface_temperature_c: Annotated[float, Field(description="Температура на поверхности, °C")] = (
        10.0
    )
    geothermal_gradient_c_per_km: Annotated[
        float, Field(ge=0, description="Геотермический градиент, °C/км")
    ] = 30.0
    creep: list[CreepBranchConfig] = Field(
        default_factory=lambda: [CreepBranchConfig(a_per_day=0.18, n=5.0, q_kj_per_mol=54.0)],
        description="Ветви установившейся ползучести (сумма); по умолчанию BGRa",
    )
    transient: TransientConfig | None = Field(
        default_factory=TransientConfig,
        description="Неустановившаяся ползучесть Мансона–Доусона; null — выключить",
    )
    stages: Annotated[list[PressureStageConfig], Field(min_length=1)] = Field(
        default_factory=lambda: [PressureStageConfig(duration_years=30.0, label="рассол")],
        description="Ступени давления в камере по порядку",
    )
    output_every_years: Annotated[float, Field(gt=0, description="Сохранять каждые N лет")] = 1.0
    dilatancy_coeff: Annotated[
        float, Field(gt=0, description="Критерий дилатансии √J2 = a·I1 (Ratigan: 0,27)")
    ] = 0.27
    step_tolerance: Annotated[
        float, Field(gt=0, le=0.5, description="Изменение девиатора напряжений за шаг, доля")
    ] = 0.1


class RunConfig(_Strict):
    """Конфигурация расчёта saltcav (YAML)."""

    name: str = Field(description="Название расчёта (в выводе и файлах)")
    grid: GridConfig
    geology: GeologyConfig = GeologyConfig()
    schedule: ScheduleConfig
    leaching: LeachingConfig = LeachingConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    mechanics: MechanicsConfig = MechanicsConfig()

    @model_validator(mode="after")
    def _check_depths_inside_grid(self) -> RunConfig:
        lo, hi = self.grid.z_top_m, self.grid.z_bottom_m
        for i, st in enumerate(self.schedule.stages):
            for name in ("injection_depth_m", "production_depth_m", "blanket_depth_m"):
                z = getattr(st, name)
                if not lo <= z <= hi:
                    raise ValueError(f"ступень {i}: {name}={z} вне сетки [{lo}, {hi}]")
            if st.blanket_depth_m >= max(st.injection_depth_m, st.production_depth_m):
                raise ValueError(f"ступень {i}: нерастворитель должен быть выше башмаков колонн")
        return self


SCHEMA_ID = "https://github.com/kizred/saltcav/schema/saltcav-config.schema.json"


def config_schema() -> dict[str, object]:
    """JSON Schema конфигурации — для подсказок и проверки YAML в редакторе."""
    schema = RunConfig.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID, **schema}


def load_config(path: str | Path) -> RunConfig:
    """Прочитать и проверить YAML-конфигурацию."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RunConfig.model_validate(raw)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    """Сохранить конфигурацию рядом с результатами прогона."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, allow_unicode=True, sort_keys=False)
