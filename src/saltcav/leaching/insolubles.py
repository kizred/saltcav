"""Нерастворимые: осаждение на подошву и подъём её уровня.

При растворении породы объёмом ΔV в камере остаётся твёрдая нерастворимая часть
f_ins·ΔV. Она сразу оседает на дно и укладывается рыхло, с коэффициентом
разрыхления B ≥ 1: насыпной объём B·f_ins·ΔV, пористость насыпи 1 − 1/B,
поры заполнены рассолом.

Насыпь заполняет камеру снизу вверх. Уровень подошвы z_floor — глубина, при
которой объём камеры ниже неё равен накопленному насыпному объёму.

Слой считается погребённым, когда уровень насыпи поднялся выше его середины.
Погребённый слой выходит из колонны рассола: доля 1/B его объёма занята твёрдым
и вытесняет рассол к отбору, остальной рассол остаётся в порах неподвижным.
Стенка ниже насыпи не растворяется.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def floor_depth(z_edges: FloatArray, layer_volumes: FloatArray, bulk_volume: float) -> float:
    """Глубина верха насыпи, м: объём камеры ниже неё равен bulk_volume.

    Внутри слоя объём считается распределённым равномерно по глубине.
    Если насыпь больше всей камеры, возвращается верх сетки.
    """
    z = np.asarray(z_edges, dtype=float)
    v = np.asarray(layer_volumes, dtype=float)
    if bulk_volume <= 0.0:
        return float(z[-1])
    filled = 0.0
    for k in range(v.size - 1, -1, -1):  # снизу вверх
        if filled + v[k] >= bulk_volume:
            frac = (bulk_volume - filled) / v[k] if v[k] > 0 else 1.0
            return float(z[k + 1] - frac * (z[k + 1] - z[k]))
        filled += v[k]
    return float(z[0])


def first_buried_layer(z_centers: FloatArray, z_floor: float) -> int:
    """Индекс верхнего погребённого слоя (середина слоя ниже уровня насыпи).

    Если погребённых слоёв нет, возвращается число слоёв.
    """
    zc = np.asarray(z_centers, dtype=float)
    buried = np.nonzero(zc >= z_floor)[0]
    return int(buried[0]) if buried.size else int(zc.size)
