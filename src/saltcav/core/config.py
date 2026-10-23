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

Positive = Annotated[float, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GridConfig(_Strict):
    """Расчётная сетка стенки R(z, φ). Глубина z отсчитывается вниз от поверхности."""

    z_top_m: Positive
    z_bottom_m: Positive
    nz: Annotated[int, Field(ge=2)]
    nphi: Annotated[int, Field(ge=1)] = 1  # 1 = осесимметрия
    r_borehole_m: Positive = 0.15

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
    """Прослой; для стохастики задаётся неопределённость положения и мощности."""

    kind: LayerKind
    top_m: Positive
    thickness_m: Positive
    top_sd_m: Annotated[float, Field(ge=0)] = 0.0
    thickness_sd_m: Annotated[float, Field(ge=0)] = 0.0


class GeologyConfig(_Strict):
    insoluble_fraction: Fraction = 0.05
    insoluble_fraction_sd: Annotated[float, Field(ge=0)] = 0.0
    bulking_factor: Annotated[float, Field(ge=1)] = 1.5
    salt_density_kg_m3: Positive = 2160.0
    layers: list[LayerConfig] = Field(default_factory=list)


class LeachMode(StrEnum):
    direct = "direct"  # прямоток: вода через центральную колонну, рассол по затрубу
    reverse = "reverse"  # противоток


class StageConfig(_Strict):
    """Ступень режима выщелачивания."""

    duration_days: Positive
    mode: LeachMode
    rate_m3_per_h: Positive
    injection_depth_m: Positive
    production_depth_m: Positive
    blanket_depth_m: Positive  # уровень нерастворителя (кровля зоны растворения)

    @property
    def duration_s(self) -> float:
        return units.days_to_seconds(self.duration_days)

    @property
    def rate_si(self) -> float:
        return units.m3_per_hour_to_si(self.rate_m3_per_h)


class ScheduleConfig(_Strict):
    stages: Annotated[list[StageConfig], Field(min_length=1)]


class EnsembleConfig(_Strict):
    n_realizations: Annotated[int, Field(ge=1)] = 100
    seed: Annotated[int, Field(ge=0)] = 12345
    sampler: str = Field(default="mc", pattern="^(mc|lhs|sobol)$")


class RunConfig(_Strict):
    name: str
    grid: GridConfig
    geology: GeologyConfig = GeologyConfig()
    schedule: ScheduleConfig
    ensemble: EnsembleConfig = EnsembleConfig()

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


def load_config(path: str | Path) -> RunConfig:
    """Прочитать и проверить YAML-конфигурацию."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RunConfig.model_validate(raw)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    """Сохранить конфигурацию рядом с результатами прогона."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, allow_unicode=True, sort_keys=False)
