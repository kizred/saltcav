"""Нерастворимые прослои, карнизы и их обрушение.

Прослой (ангидрит, глина) занимает узлы стенки i0…i1 и не растворяется: множитель
скорости растворения там нулевой. Соль над и под прослоем уходит, и прослой
остаётся плитой, выступающей в камеру, — карнизом.

Плита закреплена там, где с обеих сторон ещё есть порода, то есть на радиусе
min(R_над, R_под). Вылет карниза в секторе:

    L = min(R_над, R_под) − R_прослоя.

Если камера есть только с одной стороны (прослой — кровля или подошва), плита лежит
на соли целиком и не обрушивается: L ≤ 0. Камера не может подняться сквозь
прослой-кровлю, но обходит его через ствол скважины и размывает соль над ним.

Когда L ≥ L_крит, карниз в этом секторе обрушивается: радиус узлов прослоя
становится min(R_над, R_под), а обломки (нерастворимая порода) уходят в насыпь.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import GeologyConfig, LayerKind
from saltcav.geometry import CavernGrid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Interlayer:
    i0: int  # верхний узел прослоя
    i1: int  # нижний узел прослоя (включительно)
    critical_span: float  # м
    kind: LayerKind
    top_m: float


@dataclass(frozen=True)
class CollapseEvent:
    layer: int  # индекс прослоя
    sector: int
    span: float  # вылет в момент обрушения, м
    depth_m: float


def interlayers_from_config(geo: GeologyConfig, grid: CavernGrid) -> list[Interlayer]:
    """Нерастворимые прослои в узлах сетки. Прослой тоньше шага — ближайший узел."""
    z = grid.z_edges
    out: list[Interlayer] = []
    for layer in geo.layers:
        if layer.kind is LayerKind.halite:
            continue
        top, bottom = layer.top_m, layer.top_m + layer.thickness_m
        if bottom < z[0] or top > z[-1]:
            continue  # вне сетки
        inside = np.nonzero((z >= top) & (z <= bottom))[0]
        if inside.size == 0:
            inside = np.array([int(np.argmin(np.abs(z - 0.5 * (top + bottom))))])
        span = layer.critical_span_m or geo.ledge_span_ratio * layer.thickness_m
        out.append(Interlayer(int(inside[0]), int(inside[-1]), float(span), layer.kind, top))
    return out


def overhang(radius: FloatArray, il: Interlayer) -> FloatArray:
    """Вылет карниза L по секторам (nphi,); ≤ 0 — карниза нет."""
    r = np.asarray(radius, dtype=float)
    n = r.shape[0]
    r_layer = r[il.i0 : il.i1 + 1].max(axis=0)
    above = r[il.i0 - 1] if il.i0 > 0 else np.zeros(r.shape[1])
    below = r[il.i1 + 1] if il.i1 < n - 1 else np.zeros(r.shape[1])
    return np.asarray(np.minimum(above, below) - r_layer, dtype=np.float64)


def collapse_ledges(
    radius: FloatArray, layers: list[Interlayer], depths: FloatArray
) -> tuple[FloatArray, list[CollapseEvent]]:
    """Обрушить карнизы с вылетом ≥ критического. Возвращает новый радиус и события."""
    r = np.array(radius, dtype=float)
    events: list[CollapseEvent] = []
    for k, il in enumerate(layers):
        span = overhang(r, il)
        for j in np.nonzero(span >= il.critical_span)[0]:
            n = r.shape[0]
            above = r[il.i0 - 1, j] if il.i0 > 0 else 0.0
            below = r[il.i1 + 1, j] if il.i1 < n - 1 else 0.0
            r_new = min(above, below)
            r[il.i0 : il.i1 + 1, j] = np.maximum(r[il.i0 : il.i1 + 1, j], r_new)
            events.append(CollapseEvent(k, int(j), float(span[j]), float(depths[il.i0])))
    return r, events
