"""Ансамбль Монте-Карло: запуск реализаций и статистика по ним.

Реализация i определяется конфигурацией, seed и i (см. :mod:`.sampling`), поэтому
результат не зависит от числа процессов и порядка их завершения: один seed — тот же
ансамбль бит в бит.

От каждой реализации сохраняется итоговая форма R(z, φ), объём во времени и сводные
величины. По ансамблю считаются квантили формы (огибающие P10/P50/P90), вероятности
превышения порогов, сходимость квантилей с ростом N и чувствительность выходов к
скалярным параметрам (ранговая корреляция Спирмена).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from saltcav.core.config import RunConfig
from saltcav.core.rng import realization_rng
from saltcav.geometry import Cavern, CavernGrid
from saltcav.leaching.simulator import simulate
from saltcav.stochastic.sampling import (
    Realization,
    RealizationSampler,
    design,
    parameter_names,
)

FloatArray = NDArray[np.float64]
WIDE_M = 1.0  # радиус, начиная с которого узел считается частью камеры, а не ствола

OUTPUTS = ("volume", "max_radius", "floor_depth", "roof_depth", "centroid_offset", "collapses")


@dataclass(frozen=True)
class RealizationOutput:
    radius: FloatArray  # (nz+1, nphi) в конце
    volume_t: FloatArray  # объём в моменты записи
    floor_depth: float
    produced_salt: float
    dissolved_salt: float
    n_collapses: int
    params: FloatArray


def _run_one(task: tuple[RunConfig, int, int, FloatArray]) -> RealizationOutput:
    cfg, seed, index, u_row = task
    real = _sampler(cfg).realize(u_row, realization_rng(seed, index))
    hist = simulate(real.cfg, real.props)
    return RealizationOutput(
        radius=np.array(hist.radius[-1]),
        volume_t=np.array(hist.volume),
        floor_depth=hist.floor_depth[-1],
        produced_salt=hist.produced_salt[-1],
        dissolved_salt=hist.dissolved_salt[-1],
        n_collapses=len(hist.collapses),
        params=real.params,
    )


_SAMPLER_CACHE: dict[int, RealizationSampler] = {}


def _sampler(cfg: RunConfig) -> RealizationSampler:
    """Сэмплер на процесс: множители Холецкого считаются один раз на конфигурацию."""
    key = hash(cfg.model_dump_json())
    if key not in _SAMPLER_CACHE:
        _SAMPLER_CACHE.clear()
        _SAMPLER_CACHE[key] = RealizationSampler(cfg)
    return _SAMPLER_CACHE[key]


@dataclass
class EnsembleResult:
    z_edges: FloatArray
    phi: FloatArray
    times: FloatArray  # моменты записи объёма, с
    radius: FloatArray  # (n, nz+1, nphi)
    volume_t: FloatArray  # (n, nt)
    floor_depth: FloatArray  # (n,)
    produced_salt: FloatArray
    dissolved_salt: FloatArray
    collapses: FloatArray
    params: FloatArray  # (n, d)
    param_names: list[str]
    seed: int

    @property
    def n(self) -> int:
        return int(self.radius.shape[0])

    @property
    def grid(self) -> CavernGrid:
        return CavernGrid(self.z_edges, self.phi.size)

    # --- сводные величины по реализациям ----------------------------------------

    def output(self, name: str) -> FloatArray:
        """Скалярный выход по реализациям (n,)."""
        if name == "volume":
            return np.asarray(self.volume_t[:, -1])
        if name == "max_radius":
            return np.asarray(self.radius.max(axis=(1, 2)))
        if name == "floor_depth":
            return self.floor_depth
        if name == "roof_depth":
            return np.array([self._roof(r) for r in self.radius])
        if name == "centroid_offset":
            g = self.grid
            off = [Cavern(g, r).centroid_offset() for r in self.radius]
            return np.array([float(np.hypot(o[:, 0], o[:, 1]).max()) for o in off])
        if name == "collapses":
            return self.collapses
        raise KeyError(f"неизвестный выход {name!r}; есть {OUTPUTS}")

    def _roof(self, r: FloatArray) -> float:
        wide = np.nonzero(r.max(axis=1) > WIDE_M)[0]
        return float(self.z_edges[wide[0]]) if wide.size else float("nan")

    # --- статистика -----------------------------------------------------------

    def radius_quantiles(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> FloatArray:
        """Квантили радиуса в каждом узле (len(qs), nz+1, nphi)."""
        return np.asarray(np.quantile(self.radius, qs, axis=0), dtype=np.float64)

    def equivalent_radius_quantiles(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> FloatArray:
        """Квантили эквивалентного радиуса сечения по глубине (len(qs), nz+1)."""
        r_eq = np.sqrt(np.mean(self.radius**2, axis=2))
        return np.asarray(np.quantile(r_eq, qs, axis=0), dtype=np.float64)

    def prob_radius_exceeds(self, limit_m: float) -> FloatArray:
        """Вероятность R(z, φ) > limit в каждом узле (nz+1, nphi)."""
        return np.asarray((self.radius > limit_m).mean(axis=0), dtype=np.float64)

    def prob(self, name: str, above: float | None = None, below: float | None = None) -> float:
        """P(выход > above) или P(выход < below)."""
        x = self.output(name)
        if above is not None:
            return float(np.mean(x > above))
        if below is not None:
            return float(np.mean(x < below))
        raise ValueError("нужен порог above или below")

    def convergence(
        self, name: str = "volume", qs: tuple[float, ...] = (0.1, 0.5, 0.9)
    ) -> FloatArray:
        """Квантили выхода по первым k реализациям, k = 1…n: (n, len(qs))."""
        x = self.output(name)
        return np.array([np.quantile(x[: k + 1], qs) for k in range(x.size)])

    def sensitivity(self, outputs: tuple[str, ...] = OUTPUTS) -> dict[str, dict[str, float]]:
        """Ранговая корреляция Спирмена параметр → выход (параметры без разброса пропущены)."""
        res: dict[str, dict[str, float]] = {}
        varying = [j for j in range(self.params.shape[1]) if np.ptp(self.params[:, j]) > 0]
        for name in outputs:
            y = self.output(name)
            if np.ptp(y[np.isfinite(y)]) == 0 or self.n < 3:
                continue
            res[name] = {}
            for j in varying:
                rho = spearmanr(self.params[:, j], y, nan_policy="omit").statistic
                res[name][self.param_names[j]] = float(rho)
        return res

    # --- сохранение -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            z_edges_m=self.z_edges, phi_rad=self.phi, times_s=self.times, radius_m=self.radius,
            volume_t_m3=self.volume_t, floor_depth_m=self.floor_depth,
            produced_salt_kg=self.produced_salt, dissolved_salt_kg=self.dissolved_salt,
            collapses=self.collapses, params=self.params,
            param_names=np.array(self.param_names), seed=np.array(self.seed),
        )  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> EnsembleResult:
        d = np.load(path)
        return cls(
            d["z_edges_m"], d["phi_rad"], d["times_s"], d["radius_m"], d["volume_t_m3"],
            d["floor_depth_m"], d["produced_salt_kg"], d["dissolved_salt_kg"], d["collapses"],
            d["params"], [str(x) for x in d["param_names"]], int(d["seed"]),
        )  # fmt: skip


def make_realization(
    cfg: RunConfig, index: int, n: int | None = None, seed: int | None = None
) -> Realization:
    """Реализация ``index`` ансамбля (n, seed) — та же, что в :func:`run_ensemble`.

    План LHS/Соболя зависит от n, поэтому n и seed должны совпадать с ансамблем.
    """
    n = cfg.ensemble.n_realizations if n is None else n
    seed = cfg.ensemble.seed if seed is None else seed
    if not 0 <= index < n:
        raise ValueError(f"номер реализации {index} вне 0…{n - 1}")
    u = design(cfg, n, seed)
    return _sampler(cfg).realize(u[index], realization_rng(seed, index))


def run_ensemble(
    cfg: RunConfig,
    n: int | None = None,
    seed: int | None = None,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> EnsembleResult:
    """Запустить ансамбль: n реализаций (по умолчанию из конфигурации), jobs процессов.

    ``progress(k, n)`` вызывается после каждой реализации; исключение в нём прерывает
    расчёт (так окно отменяет ансамбль).
    """
    n = cfg.ensemble.n_realizations if n is None else n
    seed = cfg.ensemble.seed if seed is None else seed
    if n < 1:
        raise ValueError("n ≥ 1")
    u = design(cfg, n, seed)
    tasks = [(cfg, seed, i, u[i]) for i in range(n)]
    jobs = max(1, min(jobs if jobs > 0 else (os.cpu_count() or 1), n))
    outs: list[RealizationOutput] = []
    if jobs == 1:
        for t in tasks:
            outs.append(_run_one(t))
            if progress:
                progress(len(outs), n)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            chunk = max(1, min(n // (4 * jobs), 16))
            try:
                for o in pool.map(_run_one, tasks, chunksize=chunk):
                    outs.append(o)
                    if progress:
                        progress(len(outs), n)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    grid = CavernGrid.from_config(cfg.grid)
    hist0 = simulate(cfg)  # моменты записи (одинаковы для всех реализаций)
    nt = min(o.volume_t.size for o in outs)
    return EnsembleResult(
        z_edges=grid.z_edges,
        phi=grid.phi_centers,
        times=np.asarray(hist0.times[:nt]),
        radius=np.stack([o.radius for o in outs]),
        volume_t=np.stack([o.volume_t[:nt] for o in outs]),
        floor_depth=np.array([o.floor_depth for o in outs]),
        produced_salt=np.array([o.produced_salt for o in outs]),
        dissolved_salt=np.array([o.dissolved_salt for o in outs]),
        collapses=np.array([o.n_collapses for o in outs], dtype=float),
        params=np.stack([o.params for o in outs]),
        param_names=parameter_names(cfg),
        seed=seed,
    )
