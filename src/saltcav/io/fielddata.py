"""Данные скважины (компании, физической модели или синтетические): чтение, проверка, запись.

Папка набора данных::

    data/<скважина>/
      operations.csv  режим по ступеням: day_start, day_end, mode, rate_m3_per_h,
                      injection_depth_m, production_depth_m, blanket_depth_m
      brine.csv       рассол на выходе: day и одно из concentration_kg_m3 | density_kg_m3,
                      по желанию sigma_kg_m3 (погрешность измерения)
      sonar.csv       сонарные съёмки: day, depth_m, azimuth_deg (пусто — без азимута),
                      radius_m
      layers.csv      прослои (по желанию): kind, top_m, thickness_m
      well.yaml       по желанию: name, temperature_c, notes

CSV из Excel читается как есть: разделитель «,» или «;», десятичная запятая при «;»,
кодировка UTF-8 (с BOM или без) или cp1251. Лишние столбцы игнорируются.

Проверка не исправляет данные, а возвращает список замечаний: пропуски и разрывы
режима, концентрация выше насыщения, немонотонное время, отрицательные радиусы.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from saltcav.core import brine
from saltcav.core.config import LayerConfig, LeachMode, RunConfig, ScheduleConfig, StageConfig

FloatArray = NDArray[np.float64]

OPERATIONS_COLUMNS = (
    "day_start",
    "day_end",
    "mode",
    "rate_m3_per_h",
    "injection_depth_m",
    "production_depth_m",
    "blanket_depth_m",
)


@dataclass
class SonarSurvey:
    day: float
    depth: FloatArray
    azimuth_deg: FloatArray | None  # None — съёмка без азимута (эквивалентный радиус)
    radius: FloatArray

    def equivalent_profile(self) -> tuple[FloatArray, FloatArray]:
        """Эквивалентный радиус √(среднее R²) по глубинам съёмки."""
        depths = np.unique(self.depth)
        req = np.array([np.sqrt(np.mean(self.radius[self.depth == d] ** 2)) for d in depths])
        return depths, req


@dataclass
class FieldData:
    name: str = "скважина"
    temperature_c: float | None = None
    operations: list[dict[str, Any]] = field(default_factory=list)
    brine_day: FloatArray = field(default_factory=lambda: np.zeros(0))
    brine_conc: FloatArray = field(default_factory=lambda: np.zeros(0))  # кг/м³
    brine_sigma: FloatArray | None = None
    sonar: list[SonarSurvey] = field(default_factory=list)
    layers: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    # --- в конфигурацию ----------------------------------------------------------------

    def schedule(self) -> ScheduleConfig:
        stages = [
            StageConfig(
                duration_days=float(r["day_end"]) - float(r["day_start"]),
                mode=LeachMode(str(r["mode"]).strip().lower()),
                rate_m3_per_h=float(r["rate_m3_per_h"]),
                injection_depth_m=float(r["injection_depth_m"]),
                production_depth_m=float(r["production_depth_m"]),
                blanket_depth_m=float(r["blanket_depth_m"]),
            )
            for r in self.operations
        ]
        return ScheduleConfig(stages=stages)

    def apply(self, cfg: RunConfig) -> RunConfig:
        """Конфигурация с режимом, прослоями и температурой из данных (если они есть)."""
        upd: dict[str, Any] = {}
        if self.operations:
            upd["schedule"] = self.schedule()
        if self.layers:
            layers = [
                LayerConfig(**{k: v for k, v in r.items() if v not in ("", None)})
                for r in self.layers
            ]
            upd["geology"] = cfg.geology.model_copy(update={"layers": layers})
        if self.temperature_c is not None:
            upd["leaching"] = cfg.leaching.model_copy(update={"temperature_c": self.temperature_c})
        out = cfg.model_copy(update=upd)
        return RunConfig.model_validate(out.model_dump())  # проверить связи между полями

    @property
    def duration_days(self) -> float:
        return float(self.operations[-1]["day_end"]) if self.operations else 0.0

    # --- проверка ---------------------------------------------------------------------

    def issues(self, cfg: RunConfig | None = None) -> list[str]:
        out: list[str] = []
        prev_end = 0.0
        for i, r in enumerate(self.operations):
            a, b = float(r["day_start"]), float(r["day_end"])
            if b <= a:
                out.append(f"режим, строка {i + 1}: конец ступени не позже начала")
            if abs(a - prev_end) > 1e-6:
                kind = "разрыв" if a > prev_end else "перекрытие"
                gap = f"({prev_end:g} → {a:g} сут)"
                out.append(f"режим, строка {i + 1}: {kind} с предыдущей ступенью {gap}")
            prev_end = b
        if self.brine_day.size:
            if np.any(np.diff(self.brine_day) <= 0):
                out.append("рассол: время не возрастает строго")
            t = cfg.leaching.temperature_c if cfg else (self.temperature_c or 20.0)
            cs = float(brine.saturation_concentration(t))
            high = int(np.sum(self.brine_conc > 1.02 * cs))
            if high:
                out.append(f"рассол: {high} замеров выше насыщения ({cs:.0f} кг/м³) — единицы?")
            if np.any(self.brine_conc < 0):
                out.append("рассол: отрицательная концентрация")
            if self.operations and np.any(self.brine_day > self.duration_days + 1e-6):
                out.append("рассол: замеры после конца режима")
        for s in self.sonar:
            if np.any(s.radius < 0):
                out.append(f"сонар {s.day:g} сут: отрицательный радиус")
            if self.operations and s.day > self.duration_days + 1e-6:
                out.append(f"сонар {s.day:g} сут: съёмка после конца режима")
        if cfg is not None:
            z0, z1 = cfg.grid.z_top_m, cfg.grid.z_bottom_m
            for s in self.sonar:
                if float(np.min(s.depth)) < z0 or float(np.max(s.depth)) > z1:
                    out.append(f"сонар {s.day:g} сут: глубины вне сетки [{z0:g}, {z1:g}] м")
        return out


# --- чтение ---------------------------------------------------------------------------------


def read_table(path: Path) -> list[dict[str, str]]:
    """CSV в список словарей (ключи — имена столбцов в нижнем регистре)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:4096]
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    out = []
    for r in rows:
        clean = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if delim == ";":
            clean = {k: v.replace(",", ".") if _numeric_comma(v) else v for k, v in clean.items()}
        if any(clean.values()):
            out.append(clean)
    return out


def _numeric_comma(v: str) -> bool:
    return v.count(",") == 1 and _is_number(v.replace(",", "."))


def _num(rows: list[dict[str, str]], key: str, path: Path) -> FloatArray:
    try:
        return np.array([float(r[key]) if r.get(key, "") != "" else np.nan for r in rows])
    except KeyError as exc:  # pragma: no cover - проверено выше
        raise ValueError(f"{path.name}: нет столбца {key}") from exc
    except ValueError as exc:
        raise ValueError(f"{path.name}: столбец {key} — не число ({exc})") from exc


def _require(rows: list[dict[str, str]], cols: tuple[str, ...], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path.name}: пустой файл")
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise ValueError(f"{path.name}: нет столбцов {', '.join(missing)}")


def load_field_data(folder: str | Path) -> FieldData:
    """Прочитать папку набора данных (обязателен хотя бы один из файлов)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder}: не папка")
    data = FieldData(name=folder.name)
    meta = folder / "well.yaml"
    if meta.exists():
        m = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        data.name = str(m.get("name", data.name))
        data.temperature_c = m.get("temperature_c")
        data.notes = str(m.get("notes", ""))
    found = False

    p = folder / "operations.csv"
    if p.exists():
        rows = read_table(p)
        _require(rows, OPERATIONS_COLUMNS, p)
        data.operations = [{k: r[k] for k in OPERATIONS_COLUMNS} for r in rows]
        found = True

    p = folder / "brine.csv"
    if p.exists():
        rows = read_table(p)
        _require(rows, ("day",), p)
        data.brine_day = _num(rows, "day", p)
        if "concentration_kg_m3" in rows[0]:
            data.brine_conc = _num(rows, "concentration_kg_m3", p)
        elif "density_kg_m3" in rows[0]:
            t = data.temperature_c if data.temperature_c is not None else 20.0
            data.brine_conc = concentration_from_density(_num(rows, "density_kg_m3", p), t)
        else:
            raise ValueError(f"{p.name}: нужен столбец concentration_kg_m3 или density_kg_m3")
        if "sigma_kg_m3" in rows[0]:
            data.brine_sigma = _num(rows, "sigma_kg_m3", p)
        ok = np.isfinite(data.brine_day) & np.isfinite(data.brine_conc)
        data.brine_day, data.brine_conc = data.brine_day[ok], data.brine_conc[ok]
        if data.brine_sigma is not None:
            data.brine_sigma = data.brine_sigma[ok]
        found = True

    p = folder / "sonar.csv"
    if p.exists():
        rows = read_table(p)
        _require(rows, ("day", "depth_m", "radius_m"), p)
        day = _num(rows, "day", p)
        depth = _num(rows, "depth_m", p)
        rad = _num(rows, "radius_m", p)
        az = _num(rows, "azimuth_deg", p) if "azimuth_deg" in rows[0] else np.full(day.size, np.nan)
        for d in np.unique(day[np.isfinite(day)]):
            m = (day == d) & np.isfinite(depth) & np.isfinite(rad)
            a = az[m]
            data.sonar.append(
                SonarSurvey(float(d), depth[m], None if np.all(np.isnan(a)) else a, rad[m])
            )
        found = True

    p = folder / "layers.csv"
    if p.exists():
        rows = read_table(p)
        _require(rows, ("kind", "top_m", "thickness_m"), p)
        data.layers = [
            {"kind": r["kind"], "top_m": float(r["top_m"]), "thickness_m": float(r["thickness_m"])}
            for r in rows
        ]
        found = True

    if not found:
        raise ValueError(f"{folder}: нет ни одного файла данных (operations, brine, sonar, layers)")
    return data


def concentration_from_density(rho: FloatArray, t_c: float = 20.0) -> FloatArray:
    """Концентрация NaCl по плотности рассола, кг/м³ (обратная таблица)."""
    w = np.linspace(0.0, 0.30, 3001)
    rho_grid = brine.brine_density(w, t_c)
    c_grid = w * rho_grid
    return np.asarray(np.interp(np.asarray(rho, dtype=float), rho_grid, c_grid))


# --- запись (синтетические и открытые наборы) ----------------------------------------------


def write_field_data(folder: str | Path, data: FieldData, excel: bool = False) -> None:
    """Записать набор; ``excel`` — как выгрузка из русского Excel: «;» и десятичная запятая."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"name": data.name}
    if data.temperature_c is not None:
        meta["temperature_c"] = data.temperature_c
    if data.notes:
        meta["notes"] = data.notes
    (folder / "well.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True), encoding="utf-8")
    if data.operations:
        _write(
            folder / "operations.csv",
            list(OPERATIONS_COLUMNS),
            [[r[c] for c in OPERATIONS_COLUMNS] for r in data.operations],
            excel,
        )
    if data.brine_day.size:
        cols = ["day", "concentration_kg_m3"] + (
            ["sigma_kg_m3"] if data.brine_sigma is not None else []
        )
        rows = [
            [f"{d:g}", f"{c:.2f}"] + ([f"{s:.2f}"] if data.brine_sigma is not None else [])
            for d, c, s in zip(
                data.brine_day,
                data.brine_conc,
                data.brine_sigma if data.brine_sigma is not None else data.brine_day,
                strict=True,
            )
        ]
        _write(folder / "brine.csv", cols, rows, excel)
    if data.sonar:
        rows = []
        for s in data.sonar:
            az = s.azimuth_deg if s.azimuth_deg is not None else np.full(s.depth.size, np.nan)
            for d, a, r in zip(s.depth, az, s.radius, strict=True):
                rows.append(
                    [f"{s.day:g}", f"{d:.2f}", "" if np.isnan(a) else f"{a:.1f}", f"{r:.3f}"]
                )
        _write(folder / "sonar.csv", ["day", "depth_m", "azimuth_deg", "radius_m"], rows, excel)
    if data.layers:
        _write(
            folder / "layers.csv",
            ["kind", "top_m", "thickness_m"],
            [[r["kind"], r["top_m"], r["thickness_m"]] for r in data.layers],
            excel,
        )


def _write(path: Path, cols: list[str], rows: list[list[Any]], excel: bool = False) -> None:
    if excel:
        rows = [[str(v).replace(".", ",") if _is_number(v) else v for v in r] for r in rows]
    with path.open("w", newline="", encoding="utf-8-sig" if excel else "utf-8") as fh:
        w = csv.writer(fh, delimiter=";" if excel else ",")
        w.writerow(cols)
        w.writerows(rows)


def _is_number(v: Any) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True
