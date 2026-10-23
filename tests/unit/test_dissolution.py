import numpy as np
import pytest

from saltcav.leaching import DissolutionLaw, node_concentration, wall_growth_rate

LAW = DissolutionLaw(k0=1e-5, exponent=4 / 3, roof_ratio=1.5, floor_ratio=0.0)


def test_orientation_factor_endpoints() -> None:
    f = LAW.orientation_factor(np.array([1.0, 0.0, -1.0, 0.5]))
    np.testing.assert_allclose(f, [1.5, 1.0, 0.0, 1.25])


def test_flux_vanishes_at_saturation_and_grows_with_undersaturation() -> None:
    cs = 320.0
    j = LAW.mass_flux(np.array([cs, 0.75 * cs, 0.0, cs + 5]), cs, np.zeros(4))
    assert j[0] == 0.0 and j[3] == 0.0  # насыщение и пересыщение — без растворения
    assert j[2] == pytest.approx(1e-5 * cs)
    assert j[1] == pytest.approx(1e-5 * cs * 0.25 ** (4 / 3))


def test_law_validation() -> None:
    with pytest.raises(ValueError):
        DissolutionLaw(k0=0.0)
    with pytest.raises(ValueError):
        DissolutionLaw(k0=1e-5, roof_ratio=-1.0)


def test_node_concentration() -> None:
    np.testing.assert_allclose(node_concentration(np.array([0.0, 10.0, 30.0])), [0, 10, 30, 30])


def _rate(radius: np.ndarray, law: DissolutionLaw = LAW) -> np.ndarray:
    n = radius.shape[0]
    rate, _ = wall_growth_rate(
        radius, np.ones(n - 1), np.zeros(n), 300.0, law, 2000.0, np.ones(n, dtype=bool)
    )
    return rate


def test_vertical_wall_grows_at_base_speed() -> None:
    rate = _rate(np.full((5, 2), 10.0))
    np.testing.assert_allclose(rate, 1e-5 * 300.0 / 2000.0)


def test_roof_node_climbs_faster_than_wall() -> None:
    r = np.array([[0.1], [0.1], [10.0], [10.0]])  # камера под узлом 1 — кровля
    rate = _rate(r)
    v_wall = 1e-5 * 300.0 / 2000.0
    assert rate[1, 0] > 5 * v_wall  # узел над кровлей растёт быстро — кровля поднимается
    assert rate[0, 0] == pytest.approx(v_wall)  # выше — только своя вертикальная стенка


def test_floor_edge_almost_static_with_zero_floor_ratio() -> None:
    r = np.array([[10.0], [10.0], [0.1], [0.1]])  # подошва между узлами 1 и 2
    rate = _rate(r)
    v_wall = 1e-5 * 300.0 / 2000.0
    assert rate[1, 0] < 0.1 * v_wall  # край камеры над подошвой почти не растёт
    assert rate[2, 0] == pytest.approx(v_wall)  # стенка скважины ниже — обычная вертикаль


def test_empty_nodes_fill_only_from_neighbors() -> None:
    r = np.array([[0.0], [0.0], [0.0], [5.0]])
    rate = _rate(r)
    assert rate[0, 0] == 0.0 and rate[1, 0] == 0.0  # далеко от полости — нет растворения
    assert rate[2, 0] > 0.0  # над полостью — кровля поднимается


def test_inactive_nodes_do_not_grow() -> None:
    r = np.full((4, 1), 5.0)
    active = np.array([False, False, True, True])
    rate, _ = wall_growth_rate(r, np.ones(3), np.zeros(4), 300.0, LAW, 2000.0, active)
    assert np.all(rate[:2] == 0) and np.all(rate[2:] > 0)
