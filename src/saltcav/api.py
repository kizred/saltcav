"""Python API saltcav: те же расчёты, что в окне и командной строке, — из скрипта,
Jupyter или консоли окна.

Пример::

    import saltcav.api as sc

    cfg = sc.load("configs/example_ensemble.yaml")
    cfg = sc.set(cfg, {"leaching.k0_m_per_s": 2e-5, "schedule.stages.1.rate_m3_per_h": 80})
    res = sc.run(cfg)                       # выщелачивание
    print(sc.table(res))                    # объём, радиус, рассол по времени
    sc.plot(res, "leach.png")

    ens = sc.ensemble(cfg, n=200, jobs=0)   # Монте-Карло
    mech = sc.mechanics(cfg, res)           # геомеханика по форме
    stab = sc.stability(cfg, ens)           # P(потери устойчивости)
    cal = sc.calibrate(cfg, "data/demo/well-1")
    cfg2 = cal.calibrated_config(cfg)

    # серия расчётов по параметру
    for k0, res in sc.sweep(cfg, "leaching.k0_m_per_s", [1e-5, 2e-5, 3e-5]):
        print(k0, res.volume[-1])

Все объекты результатов сохраняются ``sc.save(obj, "file.npz")`` и открываются
``sc.load`` (в том числе в окне программы).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from saltcav.core.config import RunConfig, dump_config, load_config

if TYPE_CHECKING:
    from saltcav.calibration.fit import CalibResult
    from saltcav.io.fielddata import FieldData
    from saltcav.io.results import Results
    from saltcav.mechanics.run import MechanicsRun
    from saltcav.stochastic.ensemble import EnsembleResult
    from saltcav.surrogate.study import StudyResult

__all__ = [
    "Table", "calibrate", "data", "ensemble", "get", "load", "mechanics", "plot", "report",
    "run", "save", "set", "stability", "sweep", "table",
]  # fmt: skip

Progress = Callable[[float, str], None]

# --- конфигурация -------------------------------------------------------------------------


def load(path: str | Path) -> Any:
    """Открыть файл saltcav: YAML → конфигурация; .npz → результат (выщелачивание,
    ансамбль, геомеханика, устойчивость, калибровка); папка → данные скважины."""
    p = Path(path)
    if p.is_dir():
        return data(p)
    if p.suffix in (".yaml", ".yml"):
        return load_config(p)
    if p.suffix != ".npz":
        raise ValueError(f"{p}: ожидается .yaml, .npz или папка данных")
    with np.load(p, allow_pickle=False) as d:
        kind = str(d["kind"]) if "kind" in d else ""
        is_ens = "param_names" in d
    if kind == "saltcav-mechanics":
        from saltcav.mechanics.run import MechanicsRun

        return MechanicsRun.load(p)
    if kind == "saltcav-surrogate":
        from saltcav.surrogate.study import StudyResult

        return StudyResult.load(p)
    if kind == "saltcav-calibration":
        from saltcav.calibration.fit import CalibResult

        return CalibResult.load(p)
    if is_ens:
        from saltcav.stochastic.ensemble import EnsembleResult

        return EnsembleResult.load(p)
    from saltcav.io.results import Results

    return Results.load(p)


def save(obj: Any, path: str | Path) -> Path:
    """Сохранить конфигурацию (.yaml) или результат (.npz)."""
    p = Path(path)
    if isinstance(obj, RunConfig):
        dump_config(obj, p)
    elif hasattr(obj, "save"):
        obj.save(p)
    else:
        raise TypeError(f"не умею сохранять {type(obj).__name__}")
    return p


def _split(path: str) -> list[str | int]:
    return [int(k) if k.isdigit() else k for k in path.replace("__", ".").split(".")]


def get(cfg: RunConfig, path: str) -> Any:
    """Значение параметра по пути: ``get(cfg, "schedule.stages.0.rate_m3_per_h")``."""
    node: Any = cfg.model_dump(mode="json")
    for k in _split(path):
        try:
            node = node[k]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(f"нет параметра {path!r} (на шаге {k!r})") from exc
    return node


def set(cfg: RunConfig, values: dict[str, Any] | None = None, **kwargs: Any) -> RunConfig:
    """Новая конфигурация с изменёнными параметрами (исходная не меняется).

    ``set(cfg, {"leaching.k0_m_per_s": 2e-5})`` или ``set(cfg, leaching__k0_m_per_s=2e-5)``;
    элементы списков — по номеру: ``"geology.layers.0.top_m"``. Проверяется так же, как
    YAML: ошибка — исключение с русским сообщением."""
    raw = cfg.model_dump(mode="json")
    for path, value in {**(values or {}), **kwargs}.items():
        keys = _split(path)
        node: Any = raw
        for k in keys[:-1]:
            try:
                node = node[k]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(f"нет параметра {path!r} (на шаге {k!r})") from exc
        last = keys[-1]
        if isinstance(node, dict) and last not in node:
            raise KeyError(f"нет параметра {path!r}")
        node[last] = value
    return RunConfig.model_validate(raw)


# --- расчёты --------------------------------------------------------------------------


def _cfg(cfg: RunConfig | str | Path) -> RunConfig:
    return cfg if isinstance(cfg, RunConfig) else load_config(cfg)


def run(
    cfg: RunConfig | str | Path,
    realization: int | None = None,
    n: int | None = None,
    seed: int | None = None,
) -> Results:
    """Выщелачивание одной камеры; ``realization`` — реализация ансамбля (n, seed)."""
    from saltcav.io.results import Results
    from saltcav.leaching import simulate

    c = _cfg(cfg)
    props = None
    if realization is not None:
        from saltcav.stochastic.ensemble import make_realization

        real = make_realization(c, realization, n, seed)
        c = real.cfg.model_copy(update={"name": f"{c.name} #{realization}"})
        props = real.props
    return Results.from_history(simulate(c, props), c)


def ensemble(
    cfg: RunConfig | str | Path,
    n: int | None = None,
    seed: int | None = None,
    jobs: int = 0,
    progress: Progress | None = None,
) -> EnsembleResult:
    """Ансамбль Монте-Карло; ``jobs=0`` — все ядра."""
    from saltcav.stochastic.ensemble import run_ensemble

    cb = (lambda k, m: progress(k / m, f"реализация {k} из {m}")) if progress else None
    return run_ensemble(_cfg(cfg), n=n, seed=seed, jobs=jobs, progress=cb)


def mechanics(
    cfg: RunConfig | str | Path,
    shape: Results | None = None,
    day: float | None = None,
    model: str | None = None,
    years: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> MechanicsRun:
    """Геомеханика по форме ``shape`` (по умолчанию — расчёт выщелачивания по ``cfg``)
    на сутки ``day`` (по умолчанию из конфигурации или конец). ``model`` — axisym | 3d;
    ``years`` — длительность одной ступени давления."""
    from saltcav.mechanics.run import run_from_shape

    c = _cfg(cfg)
    mc = c.mechanics
    if model is not None:
        mc = mc.model_copy(update={"model": model})
    if years is not None:
        if len(mc.stages) != 1:
            raise ValueError("years — только для одной ступени давления; иначе задайте stages")
        mc = mc.model_copy(
            update={"stages": [mc.stages[0].model_copy(update={"duration_years": years})]}
        )
    res = shape if shape is not None else run(c)
    day = day if day is not None else mc.day
    i = res.nt - 1 if day is None else res.frame_at(day)
    layers = list(c.geology.layers)
    if res.config_yaml:  # прослои формы (у реализации ансамбля — её отметки)
        layers = list(RunConfig.model_validate(yaml.safe_load(res.config_yaml)).geology.layers)
    return run_from_shape(mc, res.z_edges, res.radius[i], res.name, float(res.days[i]),
                          float(res.volume[i]), progress=progress, layers=layers)  # fmt: skip


def stability(
    cfg: RunConfig | str | Path,
    ens: EnsembleResult | None = None,
    n: int | None = None,
    jobs: int = 0,
    progress: Progress | None = None,
) -> StudyResult:
    """Вероятность потери устойчивости по ансамблю (суррогат геомеханики)."""
    from saltcav.mechanics.run import ensemble_layers
    from saltcav.surrogate.study import run_study

    c = _cfg(cfg)
    e = ens if ens is not None else ensemble(c, n=n, jobs=jobs)
    return run_study(c, e.z_edges, e.radius, seed=e.seed, jobs=jobs, progress=progress,
                     layers=ensemble_layers(c, e))  # fmt: skip


def data(folder: str | Path) -> FieldData:
    """Данные скважины из папки CSV (``docs/data-format.md``)."""
    from saltcav.io.fielddata import load_field_data

    return load_field_data(folder)


def calibrate(
    cfg: RunConfig | str | Path,
    field_data: FieldData | str | Path,
    method: str | None = None,
    holdout_days: float | None = None,
    jobs: int = 0,
    progress: Progress | None = None,
) -> CalibResult:
    """Калибровка параметров ``cfg.calibration`` по данным скважины."""
    from saltcav.calibration import calibrate as _calibrate
    from saltcav.calibration import read_truth

    truth: dict[str, float] = {}
    if isinstance(field_data, str | Path):
        truth = read_truth(field_data)
        field_data = data(field_data)
    return _calibrate(_cfg(cfg), field_data, method=method, holdout_day=holdout_days,
                      jobs=jobs, progress=progress, truth=truth)  # fmt: skip


def sweep(
    cfg: RunConfig | str | Path,
    path: str,
    values: Sequence[Any],
    fn: Callable[[RunConfig], Any] | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Серия расчётов по параметру: для каждого значения — (значение, fn(cfg)),
    по умолчанию fn = :func:`run`. Генератор: результаты идут по мере расчёта."""
    c = _cfg(cfg)
    f = fn or run
    for v in values:
        yield v, f(set(c, {path: v}))


def report(cfg: RunConfig | str | Path, out: str | Path = "report.html", **kwargs: Any) -> Path:
    """Отчёт HTML в один файл (печать в PDF — из браузера). Готовые результаты —
    ``results=``, ``ensemble=``, ``mechanics=``, ``stability=``, ``calibration=``,
    ``data=``; недостающее досчитать — ``compute=("leaching", "ensemble", "mechanics",
    "stability")``, ``n=``, ``jobs=``. См. :func:`saltcav.report.make_report`."""
    from saltcav.report import make_report

    if isinstance(kwargs.get("data"), str | Path):
        kwargs["data"] = data(kwargs["data"])
    return make_report(_cfg(cfg), out, **kwargs)


# --- картинки и таблицы ----------------------------------------------------------------


def plot(obj: Any, out: str | Path | None = None, title: str = "", **kwargs: Any) -> Any:
    """Картинка результата: в файл ``out`` (PNG, PDF, SVG) или фигура matplotlib, если
    ``out`` не задан (для Jupyter и консоли окна)."""
    from matplotlib.figure import Figure

    from saltcav.calibration.fit import CalibResult
    from saltcav.io.results import Results
    from saltcav.mechanics.run import MechanicsRun
    from saltcav.stochastic.ensemble import EnsembleResult
    from saltcav.surrogate.study import StudyResult

    surface = "#fcfcfb"
    if isinstance(obj, Results):
        from saltcav.viz.leaching_plot import draw_leaching

        fig = Figure(figsize=(12, 7.5), facecolor=surface)
        draw_leaching(fig, obj, title)
    elif isinstance(obj, EnsembleResult):
        from saltcav.viz.envelope import draw_envelope

        fig = Figure(figsize=(10.5, 6.2), facecolor=surface)
        draw_envelope(fig, obj, title, kwargs.get("layers"))
    elif isinstance(obj, MechanicsRun):
        from saltcav.viz.mechanics_plot import plot_mechanics

        if out is None:
            raise ValueError("для геомеханики укажите файл: plot(run, 'mech.png')")
        plot_mechanics(obj, out, kwargs.get("index", -1), title)
        return Path(out)
    elif isinstance(obj, StudyResult):
        from saltcav.viz.surrogate_plot import draw_surrogate

        fig = Figure(figsize=(13, 9), facecolor=surface)
        draw_surrogate(fig, obj, title)
    elif isinstance(obj, CalibResult):
        from saltcav.viz.calibration_plot import draw_calibration

        fig = Figure(figsize=(13, 9), facecolor=surface)
        draw_calibration(fig, obj, title)
    else:
        raise TypeError(f"не умею рисовать {type(obj).__name__}")
    if out is None:
        return fig
    fig.savefig(out, dpi=140, facecolor=surface)
    return Path(out)


@dataclass
class Table:
    """Таблица результата: строки — словари; ``print(t)`` — текстом, ``t.to_csv(...)``."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    title: str = ""

    @property
    def columns(self) -> list[str]:
        cols: list[str] = []
        for r in self.rows:
            cols += [k for k in r if k not in cols]
        return cols

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str) -> list[Any]:
        return [r.get(key) for r in self.rows]

    def __str__(self) -> str:
        cols = self.columns
        cells = [[_fmt(r.get(c, "")) for c in cols] for r in self.rows]
        w = [max([len(c), *(len(row[i]) for row in cells)]) for i, c in enumerate(cols)]
        lines = [self.title] if self.title else []
        lines.append("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        lines += ["  ".join(v.rjust(w[i]) for i, v in enumerate(row)) for row in cells]
        return "\n".join(lines)

    __repr__ = __str__

    def to_csv(self, path: str | Path, excel: bool = False) -> Path:
        """CSV; ``excel`` — «;» и десятичная запятая (русский Excel)."""
        p = Path(path)
        cols = self.columns
        with p.open("w", newline="", encoding="utf-8-sig" if excel else "utf-8") as fh:
            w = csv.writer(fh, delimiter=";" if excel else ",")
            w.writerow(cols)
            for r in self.rows:
                vals = [r.get(c, "") for c in cols]
                if excel:
                    vals = [str(v).replace(".", ",") if isinstance(v, float) else v for v in vals]
                w.writerow(vals)
        return p


def _fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        if not np.isfinite(v):
            return "—"
        a = abs(float(v))
        return f"{v:.4g}" if a < 1e4 or a == 0 else f"{v:,.0f}".replace(",", " ")
    return str(v)


def table(obj: Any) -> Table:
    """Сводная таблица результата (или параметров конфигурации)."""
    from saltcav.calibration.fit import CalibResult
    from saltcav.io.fielddata import FieldData
    from saltcav.io.results import Results
    from saltcav.mechanics.run import MechanicsRun
    from saltcav.stochastic.ensemble import OUTPUTS, EnsembleResult
    from saltcav.surrogate.study import TARGETS, StudyResult

    if isinstance(obj, RunConfig):
        rows = [{"параметр": k, "значение": v} for k, v in _flat(obj.model_dump(mode="json"))]
        return Table(rows, f"«{obj.name}»: параметры")
    if isinstance(obj, Results):
        rows = []
        for i in range(obj.nt):
            r = obj.radius[i]
            rows.append({
                "сут": float(obj.days[i]), "объём, м³": float(obj.volume[i]),
                "R макс, м": float(r.max()), "насыпь, м": float(obj.floor_depth[i]),
                "рассол, кг/м³": float(obj.outlet_concentration[i])
                if obj.outlet_concentration is not None else float("nan"),
            })  # fmt: skip
        return Table(rows, f"«{obj.name}»: выщелачивание")
    if isinstance(obj, EnsembleResult):
        labels = {
            "volume": "объём, м³", "max_radius": "R макс, м", "floor_depth": "насыпь, м",
            "roof_depth": "кровля, м", "centroid_offset": "смещение центра, м",
            "collapses": "обрушений карнизов",
        }  # fmt: skip
        rows = []
        for name in OUTPUTS:
            q = np.quantile(obj.output(name), (0.1, 0.5, 0.9))
            rows.append({"выход": labels[name], "P10": float(q[0]), "P50": float(q[1]),
                         "P90": float(q[2])})  # fmt: skip
        return Table(rows, f"ансамбль: {obj.n} реализаций")
    if isinstance(obj, MechanicsRun):
        s = obj.summary
        rows = []
        for k in range(s.years.size):
            row = {
                "годы": float(s.years[k]), "потеря V, %": float(s.convergence_pct[k]),
                "%/год": float(s.convergence_rate_pct_yr[k]),
                "FOS мин": float(s.min_fos_wall[k]),
                "FOS 5 %": float(s.fos_wall_p5[k]) if s.fos_wall_p5 is not None else float("nan"),
                "σt−p мин, МПа": float(s.tension_margin_mpa[k]),
            }  # fmt: skip
            if s.layer_tension_ratio is not None:
                row["прослой: растяж./прочн."] = float(s.layer_tension_ratio[k])
            rows.append(row)
        return Table(rows, f"«{obj.name}»: геомеханика")
    if isinstance(obj, StudyResult):
        rows = []
        for name, (label, _) in TARGETS.items():
            q = obj.quantiles(name)
            rows.append({"показатель": label, "P10": float(q[0]), "P50": float(q[1]),
                         "P90": float(q[2])})  # fmt: skip
        p, _ = obj.prob_below()
        rows.append({"показатель": f"P(запас < {obj.threshold:g})", "P50": p})
        rows.append({"показатель": "P(запас < 1,5)", "P50": obj.prob_below(threshold=1.5)[0]})
        return Table(rows, f"устойчивость: {obj.n} реализаций")
    if isinstance(obj, CalibResult):
        head = obj.table_header()
        return Table([dict(zip(head, r, strict=True)) for r in obj.table()], "калибровка")
    if isinstance(obj, FieldData):
        rows = [{"замер": "рассол", "сут": float(d), "значение": float(c)}
                for d, c in zip(obj.brine_day, obj.brine_conc, strict=True)]  # fmt: skip
        return Table(rows, f"«{obj.name}»: рассол на выходе, кг/м³")
    raise TypeError(f"нет таблицы для {type(obj).__name__}")


def _flat(d: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        for i, v in enumerate(d):
            yield from _flat(v, f"{prefix}.{i}")
    else:
        yield prefix, json.dumps(d, ensure_ascii=False) if isinstance(d, list) else d
