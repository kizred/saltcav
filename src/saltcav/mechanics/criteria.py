"""Критерии устойчивости по полю напряжений (сжатие — положительно, МПа в выводе).

- **Дилатансия** (Ratigan, Van Sambeek, DeVries, Nieland, 1991): повреждение соли
  начинается при √J₂ > a·I₁, a = 0,27. Запас FOS_dil = a·I₁/√J₂; < 1 — зона
  дилатансии (рост проницаемости, разуплотнение), проектное требование обычно ≥ 1,5.
- **Отсутствие растяжения / проникновения рассола** (критерий SMRI): нормальные
  напряжения вдоль стенки — кольцевое σ_θ и меридиональное σ_t (по касательной к
  профилю) — должны превышать давление в камере: min(σ_θ, σ_t) − p_кам > 0.
  Радиальное напряжение на стенке равно p_кам по граничному условию и в критерий
  не входит.
- **Конвергенция**: потеря объёма камеры, %/год.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.mechanics.fem import QW, SQRT2, AxiFEM
from saltcav.mechanics.solver import YEAR, MechanicsResult, PressureFn

FloatArray = NDArray[np.float64]
MPA = 1.0e6


def principal_compressive(sig: FloatArray) -> FloatArray:
    """Главные напряжения, сжатие положительно, по убыванию (…, 3)."""
    c = -np.asarray(sig)
    sr, sz, st, srz = c[..., 0], c[..., 1], c[..., 2], c[..., 3] / SQRT2
    mean = 0.5 * (sr + sz)
    rad = np.sqrt((0.5 * (sr - sz)) ** 2 + srz**2)
    p = np.stack([mean + rad, mean - rad, st], axis=-1)
    return np.asarray(-np.sort(-p, axis=-1))


def invariants(sig: FloatArray) -> tuple[FloatArray, FloatArray]:
    """I₁ (сжатие положительно) и √J₂, Па."""
    p = principal_compressive(sig)
    i1 = p.sum(axis=-1)
    s = p - i1[..., None] / 3.0
    return np.asarray(i1), np.asarray(np.sqrt(0.5 * np.sum(s**2, axis=-1)))


def dilatancy_fos(sig: FloatArray, coeff: float = 0.27) -> FloatArray:
    """Запас по дилатансии (Ratigan): a·I₁/√J₂ (ограничен сверху 10)."""
    i1, j2 = invariants(sig)
    return np.asarray(np.minimum(coeff * i1 / np.maximum(j2, 1.0), 10.0))


@dataclass
class ElementGeometry:
    centroid: FloatArray  # (M, 2): r, y
    volume: FloatArray  # (M,), м³ (с множителем 2π)
    wall: NDArray[np.bool_]  # элементы с ребром на стенке полости
    wall_tangent: FloatArray  # (M, 2): касательная к стенке у стеновых элементов, иначе 0


def element_geometry(fem: AxiFEM) -> ElementGeometry:
    m = fem.mesh
    w = QW / QW.sum()
    cen = np.stack([fem.rq @ w, fem.yq @ w], axis=1)
    vol = 2 * np.pi * fem.wq.sum(axis=1)
    owner: dict[tuple[int, int], int] = {}
    for e, t in enumerate(m.tri[:, :3]):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            owner[(min(a, b), max(a, b))] = e
    wall = np.zeros(m.n_elements, dtype=bool)
    tang = np.zeros((m.n_elements, 2))
    for a, b, _ in m.edges.get("cavity", np.zeros((0, 3), dtype=np.int64)):
        el = owner.get((min(a, b), max(a, b)))
        if el is None:
            continue
        d = m.nodes[b] - m.nodes[a]
        wall[el] = True
        tang[el] = d / np.hypot(*d)
    return ElementGeometry(cen, vol, wall, tang)


def wall_tangential(sig: FloatArray, tangent: FloatArray) -> FloatArray:
    """Наименьшее (сжатие положительно) из σ_θ и σ_t вдоль стенки, Па (M,)."""
    c = -np.asarray(sig)
    tr, tz = tangent[:, 0], tangent[:, 1]
    s_t = c[:, 0] * tr**2 + c[:, 1] * tz**2 + 2 * (c[:, 3] / SQRT2) * tr * tz
    return np.asarray(np.minimum(s_t, c[:, 2]))


@dataclass(frozen=True)
class Summary:
    """Сводка устойчивости по моментам записи."""

    years: FloatArray
    convergence_pct: FloatArray  # потеря объёма с начала, %
    convergence_rate_pct_yr: FloatArray  # %/год (за последний интервал)
    min_fos_wall: FloatArray  # мин. FOS_dil у стенки
    min_fos: FloatArray  # мин. FOS_dil во всём массиве
    dilatant_volume_m3: FloatArray  # объём зоны FOS_dil < 1
    tension_margin_mpa: FloatArray  # мин. (σ_min − p_кам) у стенки, МПа
    max_wall_disp_m: FloatArray  # наибольшее смещение стенки, м


def summarize(
    fem: AxiFEM,
    res: MechanicsResult,
    cavity_pressure: list[PressureFn],
    coeff: float = 0.27,
) -> Summary:
    """``cavity_pressure[k]`` — давление в камере на ступени k."""
    geo = element_geometry(fem)
    cav_nodes = fem.mesh.boundary_nodes("cavity")
    years = np.asarray(res.times) / YEAR
    conv = 100.0 * res.convergence()
    rate = np.zeros_like(conv)
    for k in range(1, conv.size):  # скачок давления — нулевой интервал: скорость прежняя
        dy = years[k] - years[k - 1]
        rate[k] = (conv[k] - conv[k - 1]) / dy if dy > 1e-9 else rate[k - 1]
    out: dict[str, list[float]] = {k: [] for k in ("fw", "f", "dv", "tm", "wd")}
    for k, (sig, u) in enumerate(zip(res.stress, res.u, strict=True)):
        fos = dilatancy_fos(sig, coeff)
        out["fw"].append(float(np.min(fos[geo.wall])) if geo.wall.any() else float("nan"))
        out["f"].append(float(np.min(fos)))
        out["dv"].append(float(geo.volume[fos < 1.0].sum()))
        p_cav = cavity_pressure[res.stage[k]](geo.centroid[:, 1])
        tm = (wall_tangential(sig, geo.wall_tangent) - p_cav)[geo.wall]
        out["tm"].append(float(np.min(tm)) / MPA if tm.size else float("nan"))
        out["wd"].append(float(np.max(np.hypot(*u[cav_nodes].T))) if cav_nodes.size else 0.0)
    return Summary(
        years=years,
        convergence_pct=conv,
        convergence_rate_pct_yr=rate,
        min_fos_wall=np.array(out["fw"]),
        min_fos=np.array(out["f"]),
        dilatant_volume_m3=np.array(out["dv"]),
        tension_margin_mpa=np.array(out["tm"]),
        max_wall_disp_m=np.array(out["wd"]),
    )
