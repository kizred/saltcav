"""Вероятность потери устойчивости по ансамблю: геомеханика для части реализаций,
гауссов процесс для остальных, уточнение у границы устойчивости.

Порядок:

1. Признаки формы всех реализаций ансамбля (:mod:`saltcav.surrogate.features`).
2. Начальная выборка: крайние по каждому признаку реализации и далее — самые далёкие
   от уже выбранных (жадный maximin в стандартизованных признаках).
3. Геомеханика (осесимметрия) для выборки, параллельно; гауссов процесс на каждый выход.
4. Уточнение (AK-MCS, Echard et al., 2011): пока есть реализации, для которых знак
   «запас − порог» не определён (U = |μ − порог|/σ < u_stop), считать механику для
   самых неопределённых и переобучать.
5. Поправка на асимметрию (по желанию): 3D-расчёт для части выборки и гауссов процесс
   на разность «3D − осесимметрия»; итог = осесимметрия + поправка.
6. Вероятности и квантили по всем реализациям; перекрёстная проверка суррогата.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from saltcav.core.config import MechanicsConfig, RunConfig
from saltcav.surrogate.features import FEATURES, feature_matrix
from saltcav.surrogate.model import GPModel, cross_validate

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TARGETS: dict[str, tuple[str, bool]] = {
    "fos_p5_min": ("запас по дилатансии на 95 % стенки", False),
    "fos_min": ("мин. запас в точке стенки (зависит от сетки)", False),
    "conv_pct": ("потеря объёма за срок, %", True),
    "tension_p5_min": ("σ_t − p на 95 % стенки, МПа", False),
}
# По нему — уточнение и вероятность потери устойчивости. Минимум в точке не годится:
# у острых углов (карнизы, ступени сетки выщелачивания) он не сходится при измельчении
# сетки МКЭ (0,99 → 0,94 при h 1 → 0,5 м), и этот «шум» больше разброса по ансамблю.
# Запас, который выдерживает 95 % стенки (по объёму пристенных элементов), устойчив.
KEY = "fos_p5_min"

Progress = Callable[[float, str], None]


def mechanics_targets(
    mc: MechanicsConfig, z: FloatArray, radius: FloatArray, layers: list[Any] | None = None
) -> FloatArray:
    """Выходы механики для одной формы (в порядке TARGETS). Момент t = 0 (мгновенная
    выемка) не учитывается: в реальности камера растёт годами."""
    from saltcav.mechanics.run import run_from_shape

    run = run_from_shape(mc, z, radius, layers=layers)
    s = run.summary
    p5 = s.fos_wall_p5 if s.fos_wall_p5 is not None else s.min_fos_wall
    tm5 = s.tension_margin_p5_mpa if s.tension_margin_p5_mpa is not None else s.tension_margin_mpa
    return np.array(
        [
            float(np.min(p5[1:])),
            float(np.min(s.min_fos_wall[1:])),
            float(s.convergence_pct[-1]),
            float(np.min(tm5[1:])),
        ]
    )


def _task(args: tuple[MechanicsConfig, FloatArray, FloatArray, list[Any] | None]) -> FloatArray:
    return mechanics_targets(*args)


MECH_FEATURES: dict[str, str] = {
    "creep_log_a": "ln множителя скорости ползучести",
    "overburden": "плотность покрывающих пород, кг/м³",
}


def mechanics_parameters(cfg: RunConfig, n: int, seed: int) -> FloatArray:
    """Неопределённые параметры геомеханики по реализациям (n, 2): ln-множитель
    скорости ползучести и плотность покрывающих. Свой поток случайных чисел — форма
    камеры (ансамбль) от них не зависит."""
    unc, mc = cfg.uncertainty, cfg.mechanics
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2**31 - 7,)))
    z = rng.standard_normal((n, 2))
    sd = unc.creep_a_log_sd
    log_a = sd * z[:, 0] - 0.5 * sd**2  # среднее множителя = 1
    rho = mc.overburden_density_kg_m3 + unc.overburden_density_sd * z[:, 1]
    return np.stack([log_a, rho], axis=1)


def with_parameters(mc: MechanicsConfig, params: FloatArray) -> MechanicsConfig:
    """Конфигурация механики реализации: A всех ветвей × exp(ln-множитель), плотность."""
    mult = float(np.exp(params[0]))
    creep = [b.model_copy(update={"a_per_day": b.a_per_day * mult}) for b in mc.creep]
    return mc.model_copy(update={"creep": creep, "overburden_density_kg_m3": float(params[1])})


def select_design(x: FloatArray, n: int, exclude: IntArray | None = None) -> IntArray:
    """Крайние по каждому признаку, затем жадный maximin в стандартизованных признаках."""
    xs = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    chosen: list[int] = [] if exclude is None else [int(i) for i in exclude]
    start = len(chosen)
    for j in range(xs.shape[1]):
        if np.ptp(xs[:, j]) == 0:
            continue
        for i in (int(np.argmin(xs[:, j])), int(np.argmax(xs[:, j]))):
            if i not in chosen and len(chosen) - start < n:
                chosen.append(i)
    if not chosen:
        chosen.append(int(np.argmin(np.linalg.norm(xs, axis=1))))
    dist = np.min(np.linalg.norm(xs[:, None, :] - xs[chosen][None, :, :], axis=2), axis=1)
    while len(chosen) - start < n and len(chosen) < x.shape[0]:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(xs - xs[i], axis=1))
    return np.array(chosen[start:], dtype=np.int64)


@dataclass
class StudyResult:
    feature_names: list[str]
    features: FloatArray  # (n, d) все реализации
    train: IntArray  # номера реализаций с расчётом механики (осесимметрия)
    y_train: FloatArray  # (n_train, n_targets)
    hf: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # с 3D
    y_hf: FloatArray = field(default_factory=lambda: np.zeros((0, len(TARGETS))))
    mu: FloatArray = field(default_factory=lambda: np.zeros((0, len(TARGETS))))  # (n, t)
    sd: FloatArray = field(default_factory=lambda: np.zeros((0, len(TARGETS))))
    cv: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (n_runs, Pf, min U)
    threshold: float = 1.0
    relevance: FloatArray = field(default_factory=lambda: np.zeros(0))
    seed: int = 0

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    def column(self, name: str) -> int:
        return list(TARGETS).index(name)

    def prob_below(self, name: str = KEY, threshold: float | None = None) -> tuple[float, float]:
        """P(выход < порог): по средним суррогата и с учётом его неопределённости."""
        thr = self.threshold if threshold is None else threshold
        j = self.column(name)
        mu, sd = self.mu[:, j], np.maximum(self.sd[:, j], 1e-12)
        return float(np.mean(mu < thr)), float(np.mean(norm.cdf((thr - mu) / sd)))

    def prob_ci(self, p: float) -> float:
        """Половина 90 %-интервала вероятности из-за конечного числа реализаций."""
        return float(1.645 * np.sqrt(max(p * (1 - p), 1.0 / self.n) / self.n))

    def quantiles(self, name: str, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> FloatArray:
        return np.asarray(np.quantile(self.mu[:, self.column(name)], qs))

    def explains(self, name: str) -> bool:
        """Объясняет ли суррогат разброс выхода (R² вне обучения ≥ 0,5). Если нет, разброс
        выхода по ансамблю сопоставим с погрешностью расчёта: выход почти не зависит от
        неопределённых входов, и его можно считать постоянным (среднее ± разброс)."""
        r2 = self.cv.get(name, {}).get("r2", float("nan"))
        return bool(np.isfinite(r2) and r2 >= 0.5)

    def critical(self, k: int = 10) -> IntArray:
        """Реализации с наименьшим запасом (по суррогату)."""
        return np.asarray(np.argsort(self.mu[:, self.column(KEY)])[:k], dtype=np.int64)

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {
            f"cv_{t}_{k}": np.asarray(v) for t, d in self.cv.items() for k, v in d.items()
        }
        arrays.update(
            kind=np.array("saltcav-surrogate"), feature_names=np.array(self.feature_names),
            features=self.features, train=self.train, y_train=self.y_train, hf=self.hf,
            y_hf=self.y_hf, mu=self.mu, sd=self.sd, history=np.array(self.history).reshape(-1, 3),
            threshold=np.array(self.threshold), relevance=self.relevance,
            seed=np.array(self.seed), targets=np.array(list(TARGETS)),
        )  # fmt: skip
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> StudyResult:
        with np.load(path) as d:
            if str(d.get("kind", "")) != "saltcav-surrogate":
                raise ValueError(f"{path}: это не результат saltcav surrogate")
            cv: dict[str, dict[str, Any]] = {}
            for key in d.files:
                if key.startswith("cv_"):
                    t, k = _split_cv(key[3:])
                    v = d[key]
                    cv.setdefault(t, {})[k] = v if v.ndim else float(v)
            hist = [(int(r[0]), float(r[1]), float(r[2])) for r in d["history"]]
            return cls(
                feature_names=[str(x) for x in d["feature_names"]], features=d["features"],
                train=d["train"], y_train=d["y_train"], hf=d["hf"], y_hf=d["y_hf"], mu=d["mu"],
                sd=d["sd"], cv=cv, history=hist,
                threshold=float(d["threshold"]), relevance=d["relevance"], seed=int(d["seed"]),
            )  # fmt: skip


def _split_cv(rest: str) -> tuple[str, str]:
    for t in TARGETS:
        if rest.startswith(t + "_"):
            return t, rest[len(t) + 1 :]
    raise ValueError(rest)


def _pool(jobs: int) -> Any:
    from saltcav.core.parallel import process_pool

    return process_pool(jobs)


class _Runner:
    """Расчёты механики: параллельно для осесимметрии, последовательно для 3D (PARDISO
    сам использует все ядра)."""

    def __init__(
        self, mc: MechanicsConfig, z: FloatArray, radii: FloatArray, params: FloatArray, jobs: int,
        layers: list[list[Any]] | None = None,
    ) -> None:  # fmt: skip
        self.mc, self.z, self.radii, self.params = mc, z, radii, params
        self.layers = layers
        from saltcav.core.parallel import default_jobs

        self.jobs = default_jobs(jobs)
        self.done = 0

    def run(
        self, idx: IntArray, model: str, report: Callable[[int], None] | None = None
    ) -> FloatArray:
        mc = self.mc.model_copy(update={"model": model})
        tasks = [
            (with_parameters(mc, self.params[i]), self.z, self.radii[i],
             None if self.layers is None else self.layers[i])
            for i in idx
        ]  # fmt: skip
        out: list[FloatArray] = []
        if not (self.jobs == 1 or model == "3d" or len(tasks) == 1):
            from concurrent.futures.process import BrokenProcessPool

            with _pool(min(self.jobs, len(tasks))) as pool:
                try:
                    for y in pool.map(_task, tasks):
                        out.append(y)
                        self.done += 1
                        if report:
                            report(self.done)
                except BrokenProcessPool:  # процесс упал (память): остальное — здесь
                    pool.shutdown(wait=False, cancel_futures=True)
                    self.jobs = 1
        for t in tasks[len(out) :]:
            out.append(_task(t))
            self.done += 1
            if report:
                report(self.done)
        return np.array(out).reshape(-1, len(TARGETS))


def _fit_all(x: FloatArray, y: FloatArray, seed: int) -> list[GPModel]:
    return [
        GPModel(log=log, seed=seed).fit(x, y[:, j]) for j, (_, log) in enumerate(TARGETS.values())
    ]


def _predict(models: list[GPModel], x: FloatArray) -> tuple[FloatArray, FloatArray]:
    preds = [m.predict(x) for m in models]
    return np.stack([p[0] for p in preds], 1), np.stack([p[1] for p in preds], 1)


def run_study(
    cfg: RunConfig,
    z_edges: FloatArray,
    radii: FloatArray,
    seed: int = 0,
    jobs: int = 0,
    progress: Progress | None = None,
    layers: list[list[Any]] | None = None,
) -> StudyResult:
    """Суррогат устойчивости по формам ансамбля ``radii`` (n, nz+1, nphi).

    ``layers[i]`` — прослои реализации i (их отметки в ансамбле случайны); по умолчанию —
    прослои конфигурации для всех."""
    sc, mc = cfg.surrogate, cfg.mechanics
    report = progress or (lambda f, t: None)
    n = np.asarray(radii).shape[0]
    params = mechanics_parameters(cfg, n, seed)
    x = np.hstack([feature_matrix(z_edges, radii), params])
    budget = min(n, sc.n_train + sc.adaptive) + min(sc.n_3d, n)
    if layers is None:
        layers = [list(cfg.geology.layers)] * n
    runner = _Runner(mc, np.asarray(z_edges), np.asarray(radii), params, jobs, layers)

    def tick(stage: str) -> Callable[[int], None]:
        return lambda k: report(min(k / max(budget, 1), 1.0), f"{stage}: расчёт {k} из {budget}")

    train = select_design(x, min(sc.n_train, n))
    y = runner.run(train, "axisym", tick("начальная выборка"))
    models = _fit_all(x[train], y, seed)
    mu, sd = _predict(models, x)
    j = list(TARGETS).index(KEY)
    history: list[tuple[int, float, float]] = []

    # уточнение у границы устойчивости (AK-MCS), пачками по числу процессов
    left = sc.adaptive
    while left > 0 and train.size < n:
        u = np.abs(mu[:, j] - sc.fos_threshold) / np.maximum(sd[:, j], 1e-12)
        u[train] = np.inf
        history.append(
            (int(train.size), float(np.mean(mu[:, j] < sc.fos_threshold)), float(np.min(u)))
        )
        if np.min(u) >= sc.u_stop:
            break
        batch = min(left, runner.jobs, int(np.sum(np.isfinite(u))))
        cand = np.argsort(u)[: batch * 5]  # самые неопределённые, из них — непохожие
        new = cand[select_design(x[cand], batch)] if cand.size > batch else cand
        y_new = runner.run(new, "axisym", tick("уточнение"))
        train = np.concatenate([train, new])
        y = np.vstack([y, y_new])
        models = _fit_all(x[train], y, seed)
        mu, sd = _predict(models, x)
        left -= new.size
    history.append((int(train.size), float(np.mean(mu[:, j] < sc.fos_threshold)), float("nan")))

    # поправка на асимметрию по 3D-расчётам
    hf = np.zeros(0, dtype=np.int64)
    y_hf = np.zeros((0, len(TARGETS)))
    if sc.n_3d > 0:
        k = min(sc.n_3d, train.size)
        u = np.abs(mu[train, j] - sc.fos_threshold) / np.maximum(sd[train, j], 1e-12)
        near = train[np.argsort(u)[: k // 2]]  # половина — у границы устойчивости
        rest = np.setdiff1d(train, near)
        far = rest[select_design(x[rest], k - near.size)] if k > near.size else rest[:0]
        hf = np.concatenate([near, far])
        y_hf = runner.run(hf, "3d", tick("3D"))
        order = np.argsort(train)
        y_lo = y[order][np.searchsorted(train[order], hf)]
        delta = [
            GPModel(log=False, seed=seed).fit(x[hf], y_hf[:, t] - y_lo[:, t])
            for t in range(len(TARGETS))
        ]
        d_mu, d_sd = _predict(delta, x)
        mu, sd = mu + d_mu, np.sqrt(sd**2 + d_sd**2)

    report(1.0, "перекрёстная проверка")
    cv = {
        name: cross_validate(x[train], y[:, t], log=log, seed=seed)
        for t, (name, (_, log)) in enumerate(TARGETS.items())
    }
    return StudyResult(
        feature_names=list(FEATURES) + list(MECH_FEATURES),
        features=x,
        train=train,
        y_train=y,
        hf=hf,
        y_hf=y_hf,
        mu=mu,
        sd=sd,
        cv=cv,
        history=history,
        threshold=sc.fos_threshold,
        relevance=models[j].importance(x, seed),
        seed=seed,
    )
