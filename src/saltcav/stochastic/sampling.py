"""Реализации неопределённости: скалярные параметры и поля свойств стенки.

Скалярные параметры реализации (k0, кровля/стенка, критический вылет карниза,
расход по ступеням, положение и мощность прослоев) берутся из плана эксперимента —
матрицы U (n × d) равномерных чисел в [0, 1): Монте-Карло, латинский гиперкуб или
последовательность Соболя. Поля по (z, φ) многомерны, для них квази-Монте-Карло
бессмысленно — они всегда Монте-Карло от генератора реализации.

Реализация i полностью определяется (конфигурация, seed, i): строка U[i] и
генератор ``realization_rng(seed, i)`` не зависят от числа процессов.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from saltcav.core.config import RunConfig
from saltcav.geometry import CavernGrid
from saltcav.leaching.properties import WallProperties
from saltcav.stochastic.fields import GaussianField, bounded_fraction, lognormal_multiplier

FloatArray = NDArray[np.float64]
U_EPS = 1e-12  # отступ от 0 и 1 перед обратной функцией нормального распределения


def parameter_names(cfg: RunConfig) -> list[str]:
    """Имена скалярных параметров в порядке столбцов плана эксперимента."""
    names = [
        "k0",
        "roof_ratio",
        "ledge_span",
        *(f"rate_{j}" for j in range(len(cfg.schedule.stages))),
    ]
    for k, _ in enumerate(cfg.geology.layers):
        names += [f"layer{k}_top", f"layer{k}_thickness"]
    return names


def design(cfg: RunConfig, n: int, seed: int) -> FloatArray:
    """План эксперимента U (n × d) в [0, 1) по сэмплеру из конфигурации."""
    d = len(parameter_names(cfg))
    kind = cfg.ensemble.sampler
    if kind == "mc":
        u = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2**31,))).random((n, d))
    elif kind == "lhs":
        u = qmc.LatinHypercube(d=d, seed=seed).random(n)
    else:  # sobol
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        u = sampler.random_base2(m)[:n]  # степень двойки, обрезанная до n
    return np.asarray(np.clip(u, U_EPS, 1.0 - U_EPS), dtype=np.float64)


@dataclass(frozen=True)
class Realization:
    cfg: RunConfig
    props: WallProperties
    params: FloatArray  # значения скалярных параметров (в порядке parameter_names)


class RealizationSampler:
    """Строит реализации по базовой конфигурации; множители Холецкого — один раз."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.grid = CavernGrid.from_config(cfg.grid)
        unc = cfg.uncertainty
        self.field = GaussianField(
            self.grid.z_edges,
            self.grid.phi_centers,
            unc.corr_length_z_m,
            np.deg2rad(unc.corr_length_phi_deg),
        )

    def realize(self, u_row: FloatArray, rng: np.random.Generator) -> Realization:
        cfg, unc = self.cfg, self.cfg.uncertainty
        z = ndtri(np.asarray(u_row, dtype=float))  # стандартные нормальные из плана
        raw = cfg.model_dump(mode="json")
        lp, geo = raw["leaching"], raw["geology"]

        k0 = lp["k0_m_per_s"] * float(np.exp(unc.k0_log_sd * z[0] - 0.5 * unc.k0_log_sd**2))
        lp["k0_m_per_s"] = k0
        if unc.roof_ratio_range is not None:
            lo, hi = unc.roof_ratio_range
            lp["roof_ratio"] = lo + (hi - lo) * float(u_row[1])
        ledge_mult = float(np.exp(unc.ledge_span_log_sd * z[2] - 0.5 * unc.ledge_span_log_sd**2))
        geo["ledge_span_ratio"] = geo["ledge_span_ratio"] * ledge_mult
        params = [k0, lp["roof_ratio"], geo["ledge_span_ratio"]]

        col = 3
        for st in raw["schedule"]["stages"]:
            mult = max(0.1, 1.0 + unc.rate_rel_sd * float(z[col]))
            st["rate_m3_per_h"] = st["rate_m3_per_h"] * mult
            params.append(st["rate_m3_per_h"])
            col += 1
        for layer in geo["layers"]:
            layer["top_m"] = max(1e-3, layer["top_m"] + layer["top_sd_m"] * float(z[col]))
            layer["thickness_m"] = max(
                0.05, layer["thickness_m"] + layer["thickness_sd_m"] * float(z[col + 1])
            )
            if layer.get("critical_span_m") is not None:
                layer["critical_span_m"] = layer["critical_span_m"] * ledge_mult
            params += [layer["top_m"], layer["thickness_m"]]
            col += 2

        cfg_i = RunConfig.model_validate(raw)
        base = WallProperties.from_config(cfg_i.geology, self.grid)  # прослои уже обнулены
        k_mult = base.k_mult
        if unc.k_field_sigma > 0:
            k_mult = k_mult * lognormal_multiplier(self.field.sample(rng), unc.k_field_sigma)
        insoluble = base.insoluble
        sd_f = cfg_i.geology.insoluble_fraction_sd
        if sd_f > 0:
            insoluble = bounded_fraction(
                self.field.sample(rng), cfg_i.geology.insoluble_fraction, sd_f
            )
        props = WallProperties(k_mult, insoluble, base.salt_density)
        return Realization(cfg_i, props, np.asarray(params, dtype=np.float64))
