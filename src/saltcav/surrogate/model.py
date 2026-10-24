"""Гауссов процесс для одного выхода геомеханики по признакам формы.

Входы стандартизуются, постоянные признаки отбрасываются; ядро — Матерн 5/2 с
отдельной длиной корреляции по каждому признаку (ARD) плюс шум. Положительные
выходы (потеря объёма) моделируются в логарифме. Предсказание — среднее и
стандартное отклонение (неопределённость суррогата).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sklearn() -> Any:
    try:
        from sklearn import gaussian_process
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'для суррогата нужен scikit-learn: pip install -e ".[surrogate]"'
        ) from exc
    return gaussian_process


@dataclass
class GPModel:
    log: bool = False
    seed: int = 0
    keep: NDArray[np.bool_] | None = None
    x_mean: FloatArray = field(default_factory=lambda: np.zeros(0))
    x_std: FloatArray = field(default_factory=lambda: np.ones(0))
    gp: Any = None

    def _x(self, x: FloatArray) -> FloatArray:
        assert self.keep is not None
        return np.asarray((x[:, self.keep] - self.x_mean) / self.x_std)

    def fit(self, x: FloatArray, y: FloatArray) -> GPModel:
        gpm = _sklearn()
        k = gpm.kernels
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        std = x.std(axis=0)
        self.keep = std > 1e-9 * (np.abs(x).max(axis=0) + 1e-12)
        self.x_mean = x[:, self.keep].mean(axis=0)
        self.x_std = std[self.keep]
        d = int(self.keep.sum())
        kernel = k.ConstantKernel(1.0, (1e-3, 1e3)) * k.Matern(
            length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=2.5
        ) + k.WhiteKernel(1e-3, (1e-8, 1e-1))
        self.gp = gpm.GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=self.seed
        )
        target = np.log(np.maximum(y, 1e-9)) if self.log else y
        with warnings.catch_warnings():  # длины корреляции на границе — обычное дело:
            warnings.simplefilter("ignore")  # признак не влияет на выход
            self.gp.fit(self._x(x), target)
        return self

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        mu, sd = self.gp.predict(self._x(np.asarray(x, dtype=float)), return_std=True)
        if self.log:
            m = np.exp(mu)
            return np.asarray(m), np.asarray(m * sd)  # дельта-метод
        return np.asarray(mu), np.asarray(sd)

    def importance(self, x: FloatArray, seed: int = 0) -> FloatArray:
        """Важность признаков (перестановочная): насколько в среднем (СКО) меняется
        предсказание, если значения признака перемешать между реализациями. В единицах
        выхода; 0 у постоянных признаков."""
        x = np.asarray(x, dtype=float)
        base = self.predict(x)[0]
        rng = np.random.default_rng(seed)
        out = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            out[j] = float(np.sqrt(np.mean((self.predict(xp)[0] - base) ** 2)))
        return out


def cross_validate(
    x: FloatArray, y: FloatArray, log: bool = False, folds: int = 5, seed: int = 0
) -> dict[str, Any]:
    """K-кратная перекрёстная проверка: предсказания вне обучения, R², RMSE, доля
    попаданий в 90 %-интервал."""
    n = y.size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    mu = np.empty(n)
    sd = np.empty(n)
    for f in range(min(folds, n)):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        m = GPModel(log=log, seed=seed).fit(x[train], y[train])
        mu[test], sd[test] = m.predict(x[test])
    resid = y - mu
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "mu": mu,
        "sd": sd,
        "r2": 1.0 - float(np.sum(resid**2)) / ss if ss > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "coverage90": float(np.mean(np.abs(resid) <= 1.645 * sd)),
    }
