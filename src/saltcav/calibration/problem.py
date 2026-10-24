"""Постановка калибровки: параметры с априорными распределениями, прямая модель и невязки.

Параметры живут в «рабочих» координатах x: для логнормального априорного x = ln θ
(априорное по x — нормальное), для равномерного x = θ в границах. Невязки
стандартизованы: (модель − замер) / σ, так что ½·Σr² — минус логарифм правдоподобия.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import CalibParamConfig, RunConfig
from saltcav.io.fielddata import FieldData
from saltcav.leaching import History, simulate

FloatArray = NDArray[np.float64]

BAD_RESIDUAL = 30.0  # стандартизованная невязка, если прогон с этими параметрами упал


@dataclass(frozen=True)
class ParamSpec:
    section: str
    field: str
    label: str
    prior: str  # lognormal | uniform
    log_sd: float
    low: float
    high: float
    fmt: str = "{:.3g}"


SPECS: dict[str, ParamSpec] = {
    "k0": ParamSpec("leaching", "k0_m_per_s", "k0, м/с", "lognormal", 0.7, 1e-7, 1e-3, "{:.3g}"),
    "roof_ratio": ParamSpec("leaching", "roof_ratio", "кровля / стенка", "uniform", 0, 0.3, 4.0),
    "floor_ratio": ParamSpec("leaching", "floor_ratio", "подошва / стенка", "uniform", 0, 0.0, 1.0),
    "exponent": ParamSpec("leaching", "exponent", "показатель n", "uniform", 0, 0.8, 2.5),
    "insoluble_fraction": ParamSpec(
        "geology", "insoluble_fraction", "доля нерастворимых", "uniform", 0, 0.0, 0.3
    ),
    "bulking_factor": ParamSpec(
        "geology", "bulking_factor", "разрыхление насыпи", "uniform", 0, 1.1, 2.2
    ),
    "k_variation": ParamSpec("geology", "k_variation", "асимметрия a", "uniform", 0, 0.0, 0.7),
    "entrainment_coeff": ParamSpec(
        "leaching", "entrainment_coeff", "вовлечение плюма α", "uniform", 0, 0.02, 0.2
    ),
}


@dataclass(frozen=True)
class Param:
    name: str
    spec: ParamSpec
    prior: str
    center: float  # ln медианы (lognormal) или значение из конфигурации
    log_sd: float
    low: float  # границы в рабочих координатах x
    high: float

    @property
    def label(self) -> str:
        return self.spec.label

    def to_theta(self, x: float) -> float:
        return float(np.exp(x)) if self.prior == "lognormal" else float(x)

    def to_x(self, theta: float) -> float:
        return float(np.log(theta)) if self.prior == "lognormal" else float(theta)

    def prior_residual(self, x: float) -> list[float]:
        return [(x - self.center) / self.log_sd] if self.prior == "lognormal" else []

    def prior_text(self) -> str:
        if self.prior == "lognormal":
            m = self.spec.fmt.format(np.exp(self.center))
            return f"логнорм., медиана {m}, σ ln {self.log_sd:g}"
        return f"равном. [{self.spec.fmt.format(self.low)}, {self.spec.fmt.format(self.high)}]"

    def start(self) -> float:
        """Начальная точка поиска: центр априорного, внутри границ."""
        lo, hi = self.low, self.high
        x = self.center
        if not lo < x < hi:
            x = 0.5 * (lo + hi)
        return float(np.clip(x, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo)))

    @property
    def scale(self) -> float:
        """Характерный масштаб x — для шагов разностной производной и стартового облака."""
        return self.log_sd if self.prior == "lognormal" else (self.high - self.low) / 4


def make_params(cfg: RunConfig, items: list[CalibParamConfig] | None = None) -> list[Param]:
    out = []
    for it in items if items is not None else cfg.calibration.parameters:
        s = SPECS[it.name]
        prior = it.prior or s.prior
        value = float(getattr(getattr(cfg, s.section), s.field))
        if prior == "lognormal":
            if value <= 0:
                raise ValueError(
                    f"{it.name}: для логнормального априорного значение должно быть > 0"
                )
            sd = it.log_sd or s.log_sd or 0.5
            lo = np.log(it.low if it.low is not None else max(s.low, value * np.exp(-6 * sd)))
            hi = np.log(it.high if it.high is not None else min(s.high, value * np.exp(6 * sd)))
            out.append(Param(it.name, s, prior, float(np.log(value)), sd, float(lo), float(hi)))
        else:
            lo = it.low if it.low is not None else s.low
            hi = it.high if it.high is not None else s.high
            out.append(Param(it.name, s, prior, value, 0.0, float(lo), float(hi)))
    return out


def with_values(cfg: RunConfig, values: dict[str, float]) -> RunConfig:
    """Конфигурация с подставленными значениями параметров (в естественных единицах)."""
    sections: dict[str, dict[str, Any]] = {}
    for name, v in values.items():
        s = SPECS[name]
        sections.setdefault(s.section, {})[s.field] = float(v)
    upd = {sec: getattr(cfg, sec).model_copy(update=f) for sec, f in sections.items()}
    return cfg.model_copy(update=upd)


# --- прямая модель и сравнение с данными ---------------------------------------------------


@dataclass
class Prediction:
    """Что модель говорит о замерах: рассол в моменты замеров, радиусы сонара, кривые."""

    brine: FloatArray
    sonar: list[FloatArray]
    days: FloatArray  # моменты записи прогона
    outlet: FloatArray  # концентрация на выходе в эти моменты
    profiles: list[FloatArray]  # эквивалентный радиус по z_edges в дни съёмок
    volume: FloatArray
    ok: bool = True


def _frame_radius(hist: History, day: float) -> FloatArray:
    """R(z, φ) в момент day: линейно между соседними записями."""
    t = np.asarray(hist.times) / 86_400.0
    i = int(np.clip(np.searchsorted(t, day), 1, t.size - 1))
    w = float(np.clip((day - t[i - 1]) / max(t[i] - t[i - 1], 1e-12), 0.0, 1.0))
    return np.asarray((1 - w) * hist.radius[i - 1] + w * hist.radius[i])


@dataclass
class Problem:
    """Калибровочная задача: базовая конфигурация (режим из данных), данные, параметры."""

    cfg: RunConfig
    data: FieldData
    params: list[Param]
    sigma_c: FloatArray  # по замерам рассола
    sigma_r: float
    fit_brine: NDArray[np.bool_]  # замеры, участвующие в подгонке (остальные — проверка)
    fit_sonar: list[bool]

    @classmethod
    def build(cls, cfg: RunConfig, data: FieldData, holdout_day: float | None = None) -> Problem:
        base = data.apply(cfg)
        lp = base.leaching
        base = base.model_copy(
            update={
                "leaching": lp.model_copy(
                    update={"output_every_days": min(lp.output_every_days, 1.0)}
                )
            }
        )
        cc = cfg.calibration
        sig = np.full(data.brine_day.size, cc.sigma_concentration_kg_m3)
        if data.brine_sigma is not None:  # σ из файла, где задана
            given = np.asarray(data.brine_sigma, dtype=float)
            sig = np.where(np.isfinite(given) & (given > 0), given, sig)
        h = np.inf if holdout_day is None else float(holdout_day)
        fit_b = np.asarray(data.brine_day <= h)
        fit_s = [s.day <= h for s in data.sonar]
        if not fit_b.any() and not any(fit_s):
            raise ValueError("нет данных для подгонки (все замеры позже holdout)")
        return cls(base, data, make_params(cfg), sig, cc.sigma_radius_m, fit_b, fit_s)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        return np.array([p.low for p in self.params]), np.array([p.high for p in self.params])

    def theta(self, x: FloatArray) -> dict[str, float]:
        return {p.name: p.to_theta(float(v)) for p, v in zip(self.params, x, strict=True)}

    def config(self, x: FloatArray) -> RunConfig:
        return with_values(self.cfg, self.theta(x))

    def n_fit(self) -> int:
        n = int(self.fit_brine.sum())
        for s, f in zip(self.data.sonar, self.fit_sonar, strict=True):
            if f:
                n += self._sonar_obs(s)[1].size
        return n

    def _sonar_obs(self, s: Any) -> tuple[FloatArray, FloatArray]:
        """(глубины, радиусы) съёмки: по точкам, если есть азимут, иначе эквивалентный R."""
        if s.azimuth_deg is not None:
            return s.depth, s.radius
        return s.equivalent_profile()  # type: ignore[no-any-return]

    def predict(self, x: FloatArray) -> Prediction:
        cfg = self.config(x)
        try:
            hist = simulate(cfg)
        except Exception:
            return self._failed()
        days = np.asarray(hist.times) / 86_400.0
        outlet = np.asarray(hist.outlet_concentration, dtype=float)
        brine = np.interp(self.data.brine_day, days, outlet)
        z = hist.grid.z_edges
        nphi = hist.grid.nphi
        dphi = 2 * np.pi / nphi
        sonar, profiles = [], []
        for s in self.data.sonar:
            r = _frame_radius(hist, s.day)
            profiles.append(np.sqrt(np.mean(r**2, axis=1)))
            if s.azimuth_deg is not None:
                sec = (np.floor(np.radians(s.azimuth_deg) / dphi).astype(int)) % nphi
                pred = np.array(
                    [np.interp(d, z, r[:, k]) for d, k in zip(s.depth, sec, strict=True)]
                )
            else:
                d, _ = s.equivalent_profile()
                pred = np.interp(d, z, profiles[-1])
            sonar.append(pred)
        return Prediction(brine, sonar, days, outlet, profiles, np.asarray(hist.volume))

    def _failed(self) -> Prediction:
        return Prediction(
            np.full(self.data.brine_day.size, np.nan),
            [np.full(self._sonar_obs(s)[1].size, np.nan) for s in self.data.sonar],
            np.zeros(0),
            np.zeros(0),
            [],
            np.zeros(0),
            ok=False,
        )

    def data_residuals(self, pred: Prediction, which: str = "fit") -> FloatArray:
        """Стандартизованные невязки по замерам подгонки (``fit``) или проверки (``holdout``)."""
        want = which == "fit"
        parts = []
        m = self.fit_brine if want else ~self.fit_brine
        parts.append(((pred.brine - self.data.brine_conc) / self.sigma_c)[m])
        for s, p, f in zip(self.data.sonar, pred.sonar, self.fit_sonar, strict=True):
            if f == want:
                parts.append((p - self._sonar_obs(s)[1]) / self.sigma_r)
        r = np.concatenate(parts) if parts else np.zeros(0)
        return np.where(np.isfinite(r), r, BAD_RESIDUAL)

    def residuals(self, x: FloatArray) -> FloatArray:
        """Невязки данных + априорные (для логнормальных параметров)."""
        r = self.data_residuals(self.predict(x))
        prior = [
            v for p, xi in zip(self.params, x, strict=True) for v in p.prior_residual(float(xi))
        ]
        return np.concatenate([r, np.asarray(prior, dtype=float)])

    def log_posterior(self, x: FloatArray) -> float:
        lo, hi = self.bounds
        if np.any(x <= lo) or np.any(x >= hi):
            return -np.inf
        return -0.5 * float(np.sum(self.residuals(x) ** 2))
