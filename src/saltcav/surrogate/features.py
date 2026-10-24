"""Признаки формы камеры для суррогата устойчивости.

Небольшой набор величин с понятным механическим смыслом — суррогат учится на
сотнях расчётов, поэтому признаков должно быть немного (≈ 10), и каждый должен
отвечать за свой механизм: пролёт кровли, вытянутость, карнизы, асимметрия, глубина.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from saltcav.geometry import Cavern, CavernGrid

FloatArray = NDArray[np.float64]

FEATURES: dict[str, str] = {
    "volume": "объём камеры, тыс. м³",
    "height": "высота камеры, м",
    "r_max": "наибольший эквивалентный радиус, м",
    "z_rmax": "положение наибольшего радиуса: 0 — кровля, 1 — подошва",
    "roof_span": "пролёт кровли (диаметр в верхних 15 % высоты), м",
    "roof_shape": "радиус на 10 % высоты от кровли / наибольший",
    "h_over_d": "высота / диаметр",
    "notch": "наибольшая глубина ниши под карнизом / наибольший радиус",
    "asym": "наибольшая разница радиусов по азимуту / эквивалентный радиус",
    "offset": "наибольшее смещение центра сечения от оси, м",
    "depth": "глубина середины камеры, м",
}
SHORT: dict[str, str] = {
    "volume": "объём",
    "height": "высота",
    "r_max": "макс. радиус",
    "z_rmax": "где макс. радиус",
    "roof_span": "пролёт кровли",
    "roof_shape": "форма кровли",
    "h_over_d": "высота / диаметр",
    "notch": "ниши под карнизами",
    "asym": "асимметрия",
    "offset": "смещение центра",
    "depth": "глубина",
    "creep_log_a": "скорость ползучести",
    "overburden": "плотность покрывающих",
}
R_CAVITY = 1.0  # м: радиус, начиная с которого узел считается камерой


def shape_features(z_edges: FloatArray, radius: FloatArray) -> FloatArray:
    """Вектор признаков (len(FEATURES),) по R(z) или R(z, φ)."""
    z = np.asarray(z_edges, dtype=float)
    r = np.asarray(radius, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    req = np.sqrt(np.mean(r**2, axis=1))
    wide = np.flatnonzero(req >= R_CAVITY)
    if wide.size == 0:
        return np.zeros(len(FEATURES))
    k0, k1 = int(wide[0]), int(wide[-1])
    roof, floor = float(z[k0]), float(z[k1])
    height = max(floor - roof, float(np.min(np.diff(z))))
    kmax = int(np.argmax(req))
    r_max = float(req[kmax])
    top = (z >= roof) & (z <= roof + 0.15 * height)
    roof_span = 2.0 * float(np.max(req[top]))
    r10 = float(np.interp(roof + 0.1 * height, z, req))

    # ниша под карнизом: насколько радиус в узле меньше меньшего из максимумов сверху и снизу
    notch = 0.0
    rr = r[k0 : k1 + 1]
    if rr.shape[0] >= 3:
        above = np.maximum.accumulate(rr, axis=0)
        below = np.maximum.accumulate(rr[::-1], axis=0)[::-1]
        depth = np.minimum(above, below) - rr
        notch = float(np.max(depth)) / r_max

    asym = float(np.max((r.max(axis=1) - r.min(axis=1))[k0 : k1 + 1] / (req[k0 : k1 + 1] + 0.5)))
    offset = 0.0
    if r.shape[1] > 1:
        cav = Cavern(CavernGrid(z, r.shape[1]), r)
        off = cav.centroid_offset()[k0 : k1 + 1]
        offset = float(np.max(np.hypot(off[:, 0], off[:, 1])))
    volume = Cavern(CavernGrid(z, r.shape[1]), r).volume() / 1e3
    return np.array(
        [
            volume,
            height,
            r_max,
            (float(z[kmax]) - roof) / height,
            roof_span,
            r10 / r_max,
            height / (2.0 * r_max),
            notch,
            asym,
            offset,
            0.5 * (roof + floor),
        ]
    )


def feature_matrix(z_edges: FloatArray, radii: FloatArray) -> FloatArray:
    """Признаки для набора форм (n, nz+1, nphi) → (n, len(FEATURES))."""
    return np.stack([shape_features(z_edges, r) for r in radii])
