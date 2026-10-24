"""Поля на стенке по моментам записи — данные для просмотрщика, без графики.

Каждое поле задаётся в узлах стенки (nz+1, nphi), как радиус; поверхность и
раскраска строятся из него одинаково для окна, анимации и ParaView.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.io.results import Results
from saltcav.leaching.dissolution import node_concentration
from saltcav.viz.surface import cavity_window, ring_radius, wall_surface

FloatArray = NDArray[np.float64]
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class FieldInfo:
    key: str
    label: str
    unit: str
    cmap: str


FIELDS: dict[str, FieldInfo] = {
    f.key: f
    for f in (
        FieldInfo("radius", "Радиус", "м", "Blues"),
        FieldInfo("rate", "Скорость растворения", "см/сут", "viridis"),
        FieldInfo("growth", "Отступ стенки с начала", "м", "magma"),
        FieldInfo("undersat", "Недонасыщение рассола у стенки", "%", "YlGnBu"),
        FieldInfo("k_mult", "Множитель скорости (порода)", "—", "coolwarm"),
        FieldInfo("insoluble", "Доля нерастворимых", "%", "copper_r"),
    )
}


def available_fields(res: Results) -> list[str]:
    """Поля, которые есть в этом файле (старые файлы — только геометрия)."""
    out = ["radius", "growth", "undersat"]
    if res.wall_rate is not None:
        out.insert(1, "rate")
    if res.k_mult is not None:
        out.append("k_mult")
    if res.insoluble is not None:
        out.append("insoluble")
    return out


def node_field(res: Results, i: int, key: str) -> FloatArray:
    """Поле ``key`` в узлах стенки (nz+1, nphi) в момент записи ``i``."""
    shape = (res.nz + 1, res.nphi)
    if key == "radius":
        return np.asarray(res.radius[i], dtype=float)
    if key == "growth":
        return np.asarray(res.radius[i] - res.radius[0], dtype=float)
    if key == "rate":
        if res.wall_rate is None:
            raise KeyError("в файле нет скорости растворения")
        return np.asarray(res.wall_rate[i] * SECONDS_PER_DAY * 100.0, dtype=float)
    if key == "undersat":
        c = node_concentration(res.concentration[i])
        u = 100.0 * np.clip(1.0 - c / res.c_sat, 0.0, 1.0)
        return np.broadcast_to(u[:, None], shape).astype(float)
    if key == "k_mult":
        if res.k_mult is None:
            raise KeyError("в файле нет свойств породы")
        return np.asarray(res.k_mult, dtype=float)
    if key == "insoluble":
        if res.insoluble is None:
            raise KeyError("в файле нет свойств породы")
        return np.asarray(100.0 * res.insoluble, dtype=float)
    raise KeyError(f"неизвестное поле {key!r}; есть {list(FIELDS)}")


def field_range(
    res: Results, key: str, window: NDArray[np.bool_] | None = None
) -> tuple[float, float]:
    """Общий диапазон шкалы по всем моментам — чтобы цвета не «прыгали» между шагами."""
    sel = slice(None) if window is None else window
    frames = [0] if key in ("k_mult", "insoluble") else range(res.nt)
    v = np.concatenate([node_field(res, i, key)[sel].ravel() for i in frames])
    if key == "rate":
        # в первые сутки у ствола скорость на порядок выше — по ней шкала была бы пустой
        lo, hi = 0.0, float(np.nanpercentile(v, 99.5))
    else:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return lo, hi


def display_window(res: Results) -> NDArray[np.bool_]:
    """Узлы для показа — по последнему моменту (камера только растёт), одинаково для всех шагов."""
    return cavity_window(res.z_edges, res.radius[-1])


def ring_values(row: FloatArray, n_ring: int) -> FloatArray:
    """Значения на кольце точек поверхности по значениям в центрах секторов."""
    return ring_radius(row, n_ring)[1]


def frame_surface(
    res: Results, i: int, key: str, window: NDArray[np.bool_] | None = None, n_ring: int = 72
) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Точки, четырёхугольники и значения поля в точках для момента ``i``."""
    keep = display_window(res) if window is None else window
    z = res.z_edges[keep]
    pts, quads = wall_surface(z, res.radius[i][keep], n_ring)
    f = node_field(res, i, key)[keep]
    vals = np.concatenate([ring_values(f[k], n_ring) for k in range(z.size)])
    return pts, quads, vals


def section_profile(res: Results, i: int, sector: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Разрез через ось в плоскости сектора: глубина, R(сектор), R(противоположный)."""
    opp = (sector + res.nphi // 2) % res.nphi
    r = res.radius[i]
    return res.z_edges, np.asarray(r[:, sector]), np.asarray(r[:, opp])
