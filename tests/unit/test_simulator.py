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
    # растворённое = прирост соли в рассоле + вынесенное с рассолом
    cav0, cav1 = hist.cavern(0), hist.cavern(-1)
    in_brine0 = float(np.dot(hist.concentration[0], cav0.layer_volumes().sum(axis=1)))
    in_brine1 = float(np.dot(hist.concentration[-1], cav1.layer_volumes().sum(axis=1)))
    lhs = hist.dissolved_salt[-1]
    rhs = in_brine1 - in_brine0 + hist.produced_salt[-1]
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_concentration_bounded_and_stable(hist) -> None:  # type: ignore[no-untyped-def]
    cs = float(brine.saturation_concentration(30.0))
    for c in hist.concentration:
        assert np.all(c >= 0) and np.all(c <= cs + 1e-9)
        assert np.all(np.diff(c) >= -1e-9)  # устойчиво по плотности


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
