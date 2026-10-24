"""Наборы данных без компании: синтетический «как от компании» и открытый эталон SANSMIC.

Синтетический набор выдаёт сама модель со скрытыми «истинными» параметрами плюс шум
замеров — на нём проверяется, что калибровка находит истину (приёмка этапа 8).
Открытый набор — расчёт SANSMIC (Sandia, открытый код) для камеры 100 м, 100 сут
прямотока: замеры рассола — ежесуточная плотность на выходе, сонар — профили R(z).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from saltcav.calibration.problem import SPECS, _frame_radius, with_values
from saltcav.core import brine
from saltcav.core.config import RunConfig, load_config
from saltcav.io.fielddata import (
    OPERATIONS_COLUMNS,
    FieldData,
    SonarSurvey,
    concentration_from_density,
)
from saltcav.leaching import simulate

ROOT = Path(__file__).resolve().parents[3]


def operations_from_config(cfg: RunConfig) -> list[dict[str, Any]]:
    rows, t = [], 0.0
    for st in cfg.schedule.stages:
        rows.append(
            {
                "day_start": f"{t:g}",
                "day_end": f"{t + st.duration_days:g}",
                "mode": st.mode.value,
                "rate_m3_per_h": f"{st.rate_m3_per_h:g}",
                "injection_depth_m": f"{st.injection_depth_m:g}",
                "production_depth_m": f"{st.production_depth_m:g}",
                "blanket_depth_m": f"{st.blanket_depth_m:g}",
            }
        )
        t += st.duration_days
    assert list(rows[0]) == list(OPERATIONS_COLUMNS)
    return rows


def synthesize(
    cfg: RunConfig,
    truth: dict[str, float],
    brine_every_days: float = 5.0,
    sonar_days: list[float] | None = None,
    n_azimuth: int = 8,
    dz_m: float = 2.0,
    sigma_c: float = 3.0,
    sigma_r: float = 0.3,
    seed: int = 0,
    name: str = "синтетическая скважина",
) -> FieldData:
    """Данные «как от компании»: прогон с параметрами ``truth`` + шум замеров.

    Рассол — каждые ``brine_every_days`` сут; сонар — в дни ``sonar_days`` (по умолчанию
    середина и конец режима), шаг по глубине ``dz_m``, ``n_azimuth`` направлений при
    nphi > 1, только там, где камера шире 1 м (уже сонар не видит стенку).
    """
    unknown = set(truth) - set(SPECS)
    if unknown:
        raise ValueError(f"неизвестные параметры: {', '.join(sorted(unknown))}")
    true_cfg = with_values(cfg, truth)
    true_cfg = true_cfg.model_copy(
        update={"leaching": true_cfg.leaching.model_copy(update={"output_every_days": 1.0})}
    )
    hist = simulate(true_cfg)
    rng = np.random.default_rng(seed)
    days = np.asarray(hist.times) / 86_400.0
    end = float(days[-1])

    b_day = np.arange(brine_every_days, end + 1e-9, brine_every_days)
    c_true = np.interp(b_day, days, np.asarray(hist.outlet_concentration))
    c_obs = np.maximum(c_true + rng.normal(0.0, sigma_c, b_day.size), 0.0)

    z = hist.grid.z_edges
    nphi = hist.grid.nphi
    sonar = []
    for sd in sonar_days if sonar_days else [round(end / 2), end]:
        r = _frame_radius(hist, float(sd))
        depths = np.arange(z[0], z[-1] + 1e-9, dz_m)
        if nphi > 1:
            az = np.arange(n_azimuth) * 360.0 / n_azimuth
            g_d, g_a = np.meshgrid(depths, az, indexing="ij")
            dd, aa = g_d.ravel(), g_a.ravel()
            sec = (np.floor(np.radians(aa) / (2 * np.pi / nphi)).astype(int)) % nphi
            rt = np.array([np.interp(d, z, r[:, k]) for d, k in zip(dd, sec, strict=True)])
        else:
            dd, aa = depths, None
            rt = np.interp(depths, z, r[:, 0])
        keep = rt > 1.0
        ro = np.maximum(rt[keep] + rng.normal(0.0, sigma_r, int(keep.sum())), 0.0)
        sonar.append(SonarSurvey(float(sd), dd[keep], None if aa is None else aa[keep], ro))

    return FieldData(
        name=name,
        temperature_c=cfg.leaching.temperature_c,
        operations=operations_from_config(cfg),
        brine_day=b_day,
        brine_conc=c_obs,
        brine_sigma=np.full(b_day.size, sigma_c),
        sonar=sonar,
        layers=[
            {"kind": ly.kind.value, "top_m": ly.top_m, "thickness_m": ly.thickness_m}
            for ly in cfg.geology.layers
        ],
        notes=f"синтетика saltcav: истинные параметры в truth.yaml, шум рассола {sigma_c:g} кг/м³, "
        f"сонара {sigma_r:g} м, seed {seed}",
    )


def write_truth(folder: str | Path, truth: dict[str, float]) -> None:
    Path(folder, "truth.yaml").write_text(
        "# скрытые «истинные» параметры синтетического набора — только для проверки\n"
        + yaml.safe_dump({k: float(v) for k, v in truth.items()}, allow_unicode=True),
        encoding="utf-8",
    )


def read_truth(folder: str | Path) -> dict[str, float]:
    p = Path(folder, "truth.yaml")
    if not p.exists():
        return {}
    return {k: float(v) for k, v in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).items()}


SANSMIC_CONFIG = ROOT / "configs" / "benchmark_sansmic_direct.yaml"
SANSMIC_REFERENCE = ROOT / "benchmarks" / "sansmic" / "reference_direct.npz"


def sansmic_dataset(
    reference: str | Path = SANSMIC_REFERENCE,
    config: str | Path = SANSMIC_CONFIG,
    sonar_days: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0),
    brine_every_days: int = 2,
) -> FieldData:
    """Открытый набор из расчёта SANSMIC: плотность на выходе → концентрация, R(z) по дням."""
    cfg = load_config(config)
    ref = np.load(reference)
    t_c = cfg.leaching.temperature_c
    day = np.asarray(ref["time_d"], dtype=float)
    rho = np.asarray(ref["sg_out"], dtype=float) * float(brine.water_density(4.0))
    conc = concentration_from_density(rho, t_c)
    keep = (day > 0) & (np.mod(np.round(day), brine_every_days) == 0)
    z = np.asarray(ref["z_m"], dtype=float)
    rad = np.asarray(ref["radius_m"], dtype=float)  # (z, t)
    sonar = []
    for sd in sonar_days:
        k = int(np.argmin(np.abs(day - sd)))
        r = rad[:, k]
        m = r > float(np.min(rad[:, 0])) + 0.05  # только растворённая часть, не ствол
        sonar.append(SonarSurvey(float(day[k]), z[m], None, r[m]))
    return FieldData(
        name="SANSMIC, прямоток 100 сут",
        temperature_c=t_c,
        operations=operations_from_config(cfg),
        brine_day=day[keep],
        brine_conc=conc[keep],
        sonar=sonar,
        notes="открытый эталон: расчёт SANSMIC (Sandia National Laboratories, "
        "github.com/sandialabs/sansmic), сценарий configs/benchmark_sansmic_direct.yaml; "
        "benchmarks/sansmic/run_reference.py",
    )
