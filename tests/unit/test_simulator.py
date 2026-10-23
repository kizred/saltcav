import numpy as np
import pytest

from saltcav.core import brine
from saltcav.core.config import RunConfig, load_config
from saltcav.leaching import simulate
from tests.unit.test_config import EXAMPLE


def _short_config(**leaching: float) -> RunConfig:
    """Пример, укороченный до 30 + 30 сут для быстрых тестов."""
    raw = load_config(EXAMPLE).model_dump(mode="json")
    raw["schedule"]["stages"][0]["duration_days"] = 30
    raw["schedule"]["stages"][1]["duration_days"] = 30
    raw["leaching"].update(leaching)
    return RunConfig.model_validate(raw)


@pytest.fixture(scope="module")
def hist():  # type: ignore[no-untyped-def]
    return simulate(_short_config(initial_saturation=0.0))


def test_runs_and_records(hist) -> None:  # type: ignore[no-untyped-def]
    assert hist.times[0] == 0.0
    assert hist.times[-1] == pytest.approx(60 * 86_400)
    assert len(hist.times) == len(hist.radius) == len(hist.volume) >= 6


def test_cavern_grows_monotonically(hist) -> None:  # type: ignore[no-untyped-def]
    vols = np.array(hist.volume)
    assert np.all(np.diff(vols) > 0)
    for a, b in zip(hist.radius[:-1], hist.radius[1:], strict=True):
        assert np.all(b >= a)  # стенка не наступает


def test_salt_balance(hist) -> None:  # type: ignore[no-untyped-def]
    # растворённое = прирост соли в камере (рассол + поры насыпи) + вынесенное
    for i in range(1, len(hist.times)):
        lhs = hist.dissolved_salt[i]
        rhs = hist.salt_in_brine[i] - hist.salt_in_brine[0] + hist.produced_salt[i]
        assert lhs == pytest.approx(rhs, rel=1e-9)


def test_salt_balance_without_burial_matches_column_sum(hist) -> None:  # type: ignore[no-untyped-def]
    # пока погребённых слоёв нет, соль в камере = Σ V·C по всем слоям
    cav0 = hist.cavern(0)
    direct = float(np.dot(hist.concentration[0], cav0.layer_volumes().sum(axis=1)))
    assert hist.salt_in_brine[0] == pytest.approx(direct, rel=1e-12)


def test_concentration_bounded_and_stable(hist) -> None:  # type: ignore[no-untyped-def]
    cs = float(brine.saturation_concentration(30.0))
    zc = hist.grid.z_centers
    for c, z_floor in zip(hist.concentration, hist.floor_depth, strict=True):
        assert np.all(c >= 0) and np.all(c <= cs + 1e-9)
        free = c[zc < z_floor]  # свободный рассол над насыпью (в порах — неподвижный)
        assert np.all(np.diff(free) >= -1e-9)  # устойчиво по плотности


def test_no_dissolution_above_blanket(hist) -> None:  # type: ignore[no-untyped-def]
    g = hist.grid
    # нерастворитель первой ступени на 1060 м, второй — на 1020 м: выше 1020 м роста нет
    above = g.z_edges < 1020.0
    np.testing.assert_allclose(hist.radius[-1][above], hist.radius[0][above])


def test_saturated_start_dissolves_less() -> None:
    fresh = simulate(_short_config(initial_saturation=0.0))
    sat = simulate(_short_config(initial_saturation=1.0))
    assert sat.volume[-1] < fresh.volume[-1]
    assert sat.dissolved_salt[-1] > 0  # пресная вода всё равно растворяет


def test_roof_faster_than_floor(hist) -> None:  # type: ignore[no-untyped-def]
    # нижняя часть камеры (подошва) почти не растёт при floor_ratio = 0
    r = hist.radius[-1][:, 0]
    assert r.max() > 1.0
    assert r[-1] < 0.5 * r.max()


def test_brine_above_blanket_untouched() -> None:
    # слои выше самого высокого уровня нерастворителя (1020 м) не входят в колонну рассола
    h = simulate(_short_config(initial_saturation=1.0))
    above = h.grid.z_centers < 1020.0
    np.testing.assert_array_equal(h.concentration[-1][above], h.concentration[0][above])


# --- нерастворимые ----------------------------------------------------------------


def _example(**geology: float) -> RunConfig:
    raw = load_config(EXAMPLE).model_dump(mode="json")
    raw["geology"].update(geology)
    return RunConfig.model_validate(raw)


@pytest.fixture(scope="module")
def full():  # type: ignore[no-untyped-def]
    return simulate(_example())


def test_floor_rises_monotonically(full) -> None:  # type: ignore[no-untyped-def]
    floors = np.array(full.floor_depth)
    assert floors[0] == pytest.approx(1100.0)
    assert np.all(np.diff(floors) <= 1e-9)  # подошва только поднимается
    assert floors[-1] < 1095.0


def test_bulk_volume_matches_dissolved_rock(full) -> None:  # type: ignore[no-untyped-def]
    # насыпной объём = B · f_ins · (объём камеры − начальный)
    geo = load_config(EXAMPLE).geology
    grown = full.volume[-1] - full.volume[0]
    expected = geo.bulking_factor * geo.insoluble_fraction * grown
    assert full.insoluble_bulk[-1] == pytest.approx(expected, rel=1e-9)


def test_salt_balance_with_burial(full) -> None:  # type: ignore[no-untyped-def]
    lhs = full.dissolved_salt[-1]
    rhs = full.salt_in_brine[-1] - full.salt_in_brine[0] + full.produced_salt[-1]
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_wall_below_floor_does_not_grow(full) -> None:  # type: ignore[no-untyped-def]
    # всё, что засыпано к середине прогона, к концу не растёт
    g = full.grid
    i_mid = len(full.times) // 2
    below = g.z_edges > full.floor_depth[i_mid] + 1.0  # с запасом в слой
    np.testing.assert_allclose(full.radius[-1][below], full.radius[i_mid][below])


def test_no_insolubles_no_floor() -> None:
    h = simulate(_example(insoluble_fraction=0.0, insoluble_fraction_sd=0.0))
    assert h.insoluble_bulk[-1] == 0.0
    assert h.floor_depth[-1] == pytest.approx(1100.0)


def test_more_insolubles_higher_floor() -> None:
    lo = simulate(_example(insoluble_fraction=0.02))
    hi = simulate(_example(insoluble_fraction=0.15))
    assert hi.floor_depth[-1] < lo.floor_depth[-1]
