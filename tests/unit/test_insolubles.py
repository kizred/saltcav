import numpy as np
import pytest

from saltcav.leaching.insolubles import first_buried_layer, floor_depth

Z = np.linspace(0.0, 10.0, 11)  # 10 слоёв по 1 м


def test_empty_floor_at_bottom() -> None:
    assert floor_depth(Z, np.full(10, 3.0), 0.0) == 10.0


def test_cylinder_fill_level() -> None:
    v = np.full(10, 3.0)  # 3 м³ на метр
    assert floor_depth(Z, v, 4.5) == pytest.approx(8.5)  # 1.5 м снизу
    assert floor_depth(Z, v, 3.0) == pytest.approx(9.0)


def test_fill_skips_thin_layers() -> None:
    v = np.array([1.0] * 8 + [0.0, 0.0])  # два нижних слоя без полости
    assert floor_depth(Z, v, 0.5) == pytest.approx(7.5)


def test_overfill_returns_top() -> None:
    assert floor_depth(Z, np.ones(10), 100.0) == 0.0


def test_first_buried_layer() -> None:
    zc = 0.5 * (Z[:-1] + Z[1:])
    assert first_buried_layer(zc, 10.0) == 10  # нет погребённых
    assert first_buried_layer(zc, 9.6) == 10  # насыпь ниже середины нижнего слоя
    assert first_buried_layer(zc, 9.4) == 9
    assert first_buried_layer(zc, 7.0) == 7
