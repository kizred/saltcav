"""Подбор параметров по данным: MAP + лапласово приближение или MCMC (emcee).

MAP — нелинейные наименьшие квадраты (scipy ``least_squares``) по невязкам данных и
априорным; производные — разностные, с шагом ~2 % масштаба параметра (прямая модель
шагает по времени адаптивно и негладка в мелком масштабе), d прогонов на итерацию
считаются параллельно. Ковариация апостериорного в окрестности MAP — (JᵀJ)⁻¹.

MCMC — аффинно-инвариантный ансамбль emcee, старт — облако вокруг MAP по лапласовой
ковариации. Дорого (walkers × steps прогонов), зато не предполагает гауссовости.

Прогноз — прогоны по выборке из апостериорного: полоса 5–95 % концентрации и
профилей, проверка на отложенных замерах (holdout).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from saltcav.calibration.problem import Param, Prediction, Problem, with_values
from saltcav.core.config import RunConfig
from saltcav.io.fielddata import FieldData

FloatArray = NDArray[np.float64]
Progress = Callable[[float, str], None]

FD_STEP = 0.02  # шаг разностной производной в долях масштаба параметра


class _Eval:
    """Вызываемый объект для процессов: x → прогноз / невязки / log апостериорного."""

    def __init__(self, problem: Problem, what: str) -> None:
        self.problem = problem
        self.what = what

    def __call__(self, x: FloatArray) -> Any:
        if self.what == "residuals":
            return self.problem.residuals(np.asarray(x))
        if self.what == "logp":
            return self.problem.log_posterior(np.asarray(x))
        return self.problem.predict(np.asarray(x))


@contextmanager
def _mapper(jobs: int) -> Iterator[Callable[..., list[Any]]]:
    jobs = jobs if jobs > 0 else (os.cpu_count() or 1)
    if jobs == 1:
        yield lambda f, xs: [f(x) for x in xs]
        return
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context(method)) as pool:
        yield lambda f, xs: list(pool.map(f, list(xs)))


@dataclass
class _Pmap:  # для emcee: объект с методом map
    fn: Callable[..., list[Any]]

    def map(self, f: Callable[..., Any], xs: Any) -> list[Any]:
        return self.fn(f, xs)


@dataclass
class CalibResult:
    names: list[str]
    labels: list[str]
    priors: list[str]  # текст априорного
    log_scale: list[bool]  # x = ln θ
    method: str
    x_map: FloatArray
    cov: FloatArray  # ковариация x (лапласова)
    samples: FloatArray  # (k, d) выборка x из апостериорного
    chi2: float  # Σ r² по данным подгонки в MAP
    n_fit: int
    n_runs: int
    seconds: float
    holdout_day: float | None
    # данные и прогноз (для картинок и проверки)
    brine_day: FloatArray
    brine_obs: FloatArray
    brine_sigma: FloatArray
    brine_fit: NDArray[np.bool_]
    brine_map: FloatArray
    curve_days: FloatArray
    curve_map: FloatArray
    curve_lo: FloatArray
    curve_hi: FloatArray
    z_edges: FloatArray
    sonar_days: FloatArray
    sonar_fit: NDArray[np.bool_]
    sonar_obs: list[tuple[FloatArray, FloatArray]]  # (глубина, эквивалентный R) по съёмкам
    profile_map: FloatArray  # (n_съёмок, nz+1)
    profile_lo: FloatArray
    profile_hi: FloatArray
    rms_fit: dict[str, float] = field(default_factory=dict)  # brine (кг/м³), sonar (м)
    rms_holdout: dict[str, float] = field(default_factory=dict)
    coverage_holdout: float = float("nan")  # доля отложенных замеров рассола в 90 %-полосе
    acceptance: float = float("nan")
    truth: dict[str, float] = field(default_factory=dict)  # для синтетики

    # --- параметры --------------------------------------------------------------------

    def theta_samples(self) -> FloatArray:
        s = np.array(self.samples, dtype=float)
        for j, lg in enumerate(self.log_scale):
            if lg:
                s[:, j] = np.exp(s[:, j])
        return s

    def theta_map(self) -> FloatArray:
        return np.array(
            [np.exp(x) if lg else x for x, lg in zip(self.x_map, self.log_scale, strict=True)]
        )

    def quantiles(self, q: tuple[float, ...] = (0.05, 0.5, 0.95)) -> FloatArray:
        """(len(q), d) квантили θ по апостериорной выборке."""
        return np.quantile(self.theta_samples(), q, axis=0)

    def estimate(self) -> dict[str, float]:
        """Оценка для конфигурации: медиана MCMC или MAP (лаплас)."""
        v = self.quantiles((0.5,))[0] if self.method == "mcmc" else self.theta_map()
        return dict(zip(self.names, map(float, v), strict=True))

    @property
    def chi2_reduced(self) -> float:
        return self.chi2 / max(self.n_fit - len(self.names), 1)

    def correlation(self) -> FloatArray:
        c = np.cov(self.samples.T) if self.samples.shape[0] > len(self.names) else self.cov
        c = np.atleast_2d(c)
        sd = np.sqrt(np.maximum(np.diag(c), 1e-300))
        return np.asarray(c / np.outer(sd, sd))

    def table(self) -> list[list[str]]:
        q = self.quantiles()
        tm = self.theta_map()
        rows = []
        for j, n in enumerate(self.names):
            row = [
                self.labels[j],
                self.priors[j],
                f"{tm[j]:.3g}",
                f"{q[1, j]:.3g}",
                f"{q[0, j]:.3g} … {q[2, j]:.3g}",
            ]
            if self.truth:
                row.append(f"{self.truth[n]:.3g}" if n in self.truth else "—")
            rows.append(row)
        return rows

    def table_header(self) -> list[str]:
        h = ["параметр", "априорное", "MAP", "медиана", "90 % интервал"]
        return h + (["истина"] if self.truth else [])

    def notes(self) -> list[str]:
        out = []
        if self.chi2_reduced > 2.0:
            out.append(
                f"χ²/N = {self.chi2_reduced:.1f} > 2: модель не описывает данные в пределах σ — "
                "погрешности занижены или нужна другая модель; интервалы занижены"
            )
        elif self.chi2_reduced < 0.3:
            out.append(
                f"χ²/N = {self.chi2_reduced:.2f} < 0,3: σ, вероятно, завышены — "
                "интервалы шире нужного"
            )
        corr = self.correlation()
        d = len(self.names)
        for i in range(d):
            for j in range(i + 1, d):
                if abs(corr[i, j]) > 0.8:
                    out.append(
                        f"{self.labels[i]} и {self.labels[j]} связаны (r = {corr[i, j]:+.2f}): "
                        "данные различают в основном их комбинацию"
                    )
        q = self.quantiles()
        for j in range(d):
            if self.priors[j].startswith("равном."):
                lo, hi = (float(v) for v in self.priors[j].split("[")[1].rstrip("]").split(","))
                if q[2, j] - q[0, j] > 0.7 * (hi - lo):
                    out.append(
                        f"{self.labels[j]}: данные почти не ограничивают (интервал ≈ априорному)"
                    )
        return out

    # --- в конфигурацию ---------------------------------------------------------------

    def calibrated_config(self, cfg: RunConfig, data: FieldData | None = None) -> RunConfig:
        """Конфигурация с оценками параметров и их неопределённостью для ансамбля.

        k0 → ``uncertainty.k0_log_sd`` = σ ln k0 по апостериорному; roof_ratio →
        ``uncertainty.roof_ratio_range`` = 90 % интервал. Остальные — точечные оценки.
        """
        if data is not None:
            cfg = data.apply(cfg)
        out = with_values(cfg, self.estimate())
        th = self.theta_samples()
        unc: dict[str, Any] = {}
        if "k0" in self.names:
            unc["k0_log_sd"] = float(np.std(np.log(th[:, self.names.index("k0")])))
        if "roof_ratio" in self.names:
            j = self.names.index("roof_ratio")
            lo, hi = np.quantile(th[:, j], (0.05, 0.95))
            unc["roof_ratio_range"] = (round(float(lo), 3), round(float(hi), 3))
        out = out.model_copy(update={"uncertainty": out.uncertainty.model_copy(update=unc)})
        return RunConfig.model_validate(out.model_dump())

    # --- сохранение -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        meta = {
            "kind": "saltcav-calibration",
            "names": self.names,
            "labels": self.labels,
            "priors": self.priors,
            "log_scale": self.log_scale,
            "method": self.method,
            "chi2": self.chi2,
            "n_fit": self.n_fit,
            "n_runs": self.n_runs,
            "seconds": self.seconds,
            "holdout_day": self.holdout_day,
            "rms_fit": self.rms_fit,
            "rms_holdout": self.rms_holdout,
            "coverage_holdout": self.coverage_holdout,
            "acceptance": self.acceptance,
            "truth": self.truth,
        }
        arrays = {k: getattr(self, k) for k in _ARRAYS}
        for i, (d, r) in enumerate(self.sonar_obs):
            arrays[f"sonar_obs_d{i}"] = d
            arrays[f"sonar_obs_r{i}"] = r
        np.savez_compressed(
            path,
            kind=np.array("saltcav-calibration"),
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibResult:
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("kind") != "saltcav-calibration":
                raise ValueError(f"{path}: не результат калибровки")
            arrays = {k: np.asarray(f[k]) for k in _ARRAYS}
            n = int(arrays["sonar_days"].size)
            sonar = [
                (np.asarray(f[f"sonar_obs_d{i}"]), np.asarray(f[f"sonar_obs_r{i}"]))
                for i in range(n)
            ]
        meta.pop("kind")
        return cls(**meta, **arrays, sonar_obs=sonar)  # type: ignore[arg-type]


_ARRAYS = (
    "x_map",
    "cov",
    "samples",
    "brine_day",
    "brine_obs",
    "brine_sigma",
    "brine_fit",
    "brine_map",
    "curve_days",
    "curve_map",
    "curve_lo",
    "curve_hi",
    "z_edges",
    "sonar_days",
    "sonar_fit",
    "profile_map",
    "profile_lo",
    "profile_hi",
)


# --- подбор ---------------------------------------------------------------------------------


def _laplace(jac: FloatArray, width: FloatArray) -> FloatArray:
    """(JᵀJ + W)⁻¹; W = 1/ширина² — слабая добавка, чтобы неограниченный данными
    параметр получил дисперсию порядка своего диапазона, а не бесконечность."""
    h = jac.T @ jac + np.diag(1.0 / np.asarray(width) ** 2)
    try:
        return np.asarray(np.linalg.inv(h))
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.asarray(np.linalg.pinv(h))


def _gauss_samples(
    x0: FloatArray, cov: FloatArray, n: int, lo: FloatArray, hi: FloatArray, seed: int
) -> FloatArray:
    """Гауссово облако, усечённое границами (отбор, не обрезка: обрезка копит массу на
    границе). Параметр, который данные не ограничивают, выходит ≈ равномерным."""
    rng = np.random.default_rng(seed)
    kept: list[FloatArray] = []
    total = 0
    for _ in range(50):
        s = rng.multivariate_normal(x0, cov, size=4 * n, method="eigh")
        s = s[np.all((s > lo) & (s < hi), axis=1)]
        kept.append(s)
        total += s.shape[0]
        if total >= n:
            break
    out = np.concatenate(kept)[:n] if total else np.zeros((0, x0.size))
    if out.shape[0] < n:  # почти вся масса за границами: дополнить обрезкой
        s = rng.multivariate_normal(x0, cov, size=n - out.shape[0], method="eigh")
        eps = 1e-6 * (hi - lo)
        out = np.concatenate([out, np.clip(s, lo + eps, hi - eps)])
    return np.asarray(out)


def calibrate(
    cfg: RunConfig,
    data: FieldData,
    method: str | None = None,
    holdout_day: float | None = None,
    jobs: int = 1,
    seed: int = 0,
    progress: Progress | None = None,
    truth: dict[str, float] | None = None,
) -> CalibResult:
    """Подобрать параметры ``cfg.calibration.parameters`` по данным скважины."""
    t0 = time.perf_counter()
    cc = cfg.calibration
    method = method or cc.method
    prob = Problem.build(cfg, data, holdout_day)
    params = prob.params
    d = len(params)
    if d == 0:
        raise ValueError("calibration.parameters пуст — нечего подбирать")
    lo, hi = prob.bounds
    scale = np.array([p.scale for p in params])
    runs = 0
    say = progress or (lambda f, m: None)

    with _mapper(jobs) as pmap:
        f_res = _Eval(prob, "residuals")
        cache: dict[bytes, FloatArray] = {}

        def fun(x: FloatArray) -> FloatArray:
            nonlocal runs
            key = np.asarray(x).tobytes()
            if key not in cache:
                cache[key] = f_res(x)
                runs += 1
            return cache[key]

        def jac(x: FloatArray) -> FloatArray:
            nonlocal runs
            r0 = fun(x)
            steps = FD_STEP * scale
            pts = []
            for j in range(d):
                xp = np.array(x, dtype=float)
                h = steps[j] if x[j] + steps[j] < hi[j] else -steps[j]
                xp[j] += h
                pts.append(xp)
            rs = pmap(f_res, pts)
            runs += d
            say(min(0.6, runs / (30.0 * (d + 1))), f"поиск MAP: {runs} прогонов")
            return np.stack(
                [(r - r0) / (p[j] - x[j]) for j, (r, p) in enumerate(zip(rs, pts, strict=True))],
                axis=1,
            )

        x0 = np.array([p.start() for p in params])
        say(0.0, "поиск MAP")
        sol = least_squares(
            fun,
            x0,
            jac=jac,
            bounds=(lo, hi),
            x_scale=scale,
            method="trf",
            max_nfev=40,
            xtol=1e-4,
            ftol=1e-6,
        )
        x_map = np.asarray(sol.x)
        nd = prob.n_fit()
        r_map = fun(x_map)
        chi2 = float(np.sum(r_map[:nd] ** 2))
        cov = _laplace(np.asarray(sol.jac), hi - lo)

        acceptance = float("nan")
        if method == "mcmc":
            try:
                import emcee
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError('для MCMC нужен emcee: pip install -e ".[calibration]"') from exc
            nw = max(cc.walkers, 2 * d + 2)
            start = _gauss_samples(x_map, cov * 0.25, nw, lo, hi, seed)
            start[0] = x_map
            sampler = emcee.EnsembleSampler(nw, d, _Eval(prob, "logp"), pool=_Pmap(pmap))
            for k, _ in enumerate(sampler.sample(start, iterations=cc.steps), start=1):
                runs += nw
                say(0.6 + 0.3 * k / cc.steps, f"MCMC: шаг {k} из {cc.steps} ({runs} прогонов)")
            burn = min(cc.burn, cc.steps - 1)
            samples = np.asarray(sampler.get_chain(discard=burn, flat=True))
            acceptance = float(np.mean(sampler.acceptance_fraction))
        else:
            samples = _gauss_samples(x_map, cov, 4000, lo, hi, seed)

        # прогноз по апостериорному
        say(0.9, "прогноз по апостериорному")
        rng = np.random.default_rng(seed + 1)
        n_pred = min(cc.n_predictive, samples.shape[0])
        pick = (
            samples[rng.choice(samples.shape[0], n_pred, replace=False)] if n_pred else samples[:0]
        )
        preds: list[Prediction] = pmap(_Eval(prob, "predict"), [x_map, *pick])
        runs += len(preds)

    best = preds[0]
    ens = [p for p in preds[1:] if p.ok] or [best]
    days = best.days

    def band(
        get: Callable[[Prediction], FloatArray], ref: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        stack = np.stack([np.interp(days, p.days, get(p)) if ref is days else get(p) for p in ens])
        return np.quantile(stack, 0.05, axis=0), np.quantile(stack, 0.95, axis=0)

    c_lo, c_hi = band(lambda p: p.outlet, days)
    ns = len(data.sonar)
    nz1 = cfg.grid.nz + 1
    if ns and best.profiles:
        prof = np.stack(best.profiles)
        stack = np.stack([np.stack(p.profiles) for p in ens])
        p_lo, p_hi = np.quantile(stack, 0.05, axis=0), np.quantile(stack, 0.95, axis=0)
    else:
        prof = p_lo = p_hi = np.zeros((ns, nz1))

    res = CalibResult(
        names=prob.names,
        labels=[p.label for p in params],
        priors=[p.prior_text() for p in params],
        log_scale=[p.prior == "lognormal" for p in params],
        method=method,
        x_map=x_map,
        cov=cov,
        samples=samples,
        chi2=chi2,
        n_fit=nd,
        n_runs=runs,
        seconds=time.perf_counter() - t0,
        holdout_day=holdout_day,
        brine_day=data.brine_day,
        brine_obs=data.brine_conc,
        brine_sigma=prob.sigma_c,
        brine_fit=prob.fit_brine,
        brine_map=best.brine,
        curve_days=days,
        curve_map=best.outlet,
        curve_lo=c_lo,
        curve_hi=c_hi,
        z_edges=np.linspace(cfg.grid.z_top_m, cfg.grid.z_bottom_m, nz1),
        sonar_days=np.array([s.day for s in data.sonar]),
        sonar_fit=np.array(prob.fit_sonar, dtype=bool),
        sonar_obs=[s.equivalent_profile() for s in data.sonar],
        profile_map=prof,
        profile_lo=p_lo,
        profile_hi=p_hi,
        acceptance=acceptance,
        truth=dict(truth or {}),
    )
    res.rms_fit, res.rms_holdout = _rms(prob, best)
    if (~prob.fit_brine).any() and len(ens) > 1:
        bstack = np.stack([p.brine for p in ens])
        lo_b, hi_b = np.quantile(bstack, 0.05, axis=0), np.quantile(bstack, 0.95, axis=0)
        m = ~prob.fit_brine
        obs = data.brine_conc[m]
        sig = prob.sigma_c[m]  # полоса модели + погрешность замера
        inside = (obs >= lo_b[m] - 1.645 * sig) & (obs <= hi_b[m] + 1.645 * sig)
        res.coverage_holdout = float(np.mean(inside))
    say(1.0, "готово")
    return res


def _rms(prob: Problem, pred: Prediction) -> tuple[dict[str, float], dict[str, float]]:
    data = prob.data
    out: list[dict[str, float]] = [{}, {}]
    for k, want in enumerate((True, False)):
        m = prob.fit_brine == want
        if m.any():
            out[k]["brine"] = float(np.sqrt(np.nanmean((pred.brine[m] - data.brine_conc[m]) ** 2)))
        diffs = [
            p - prob._sonar_obs(s)[1]
            for s, p, f in zip(data.sonar, pred.sonar, prob.fit_sonar, strict=True)
            if f == want
        ]
        if diffs:
            out[k]["sonar"] = float(np.sqrt(np.nanmean(np.concatenate(diffs) ** 2)))
    return out[0], out[1]


def describe_params(params: list[Param]) -> list[str]:
    return [f"{p.label}: {p.prior_text()}" for p in params]
