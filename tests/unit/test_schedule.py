import pytest

from saltcav.core.config import LeachMode, load_config
from saltcav.geometry import CavernGrid
from saltcav.operations import Schedule, layer_index
from tests.unit.test_config import EXAMPLE


def _schedule() -> tuple[Schedule, CavernGrid]:
    cfg = load_config(EXAMPLE)
    grid = CavernGrid.from_config(cfg.grid)
    return Schedule.from_config(cfg.schedule, grid), grid


def test_layer_index() -> None:
    g = CavernGrid.uniform(900, 1100, 200)  # слои по 1 м
    assert layer_index(g, 900.0) == 0
    assert layer_index(g, 900.5) == 0
    assert layer_index(g, 901.0) == 1  # граница — нижнему слою
    assert layer_index(g, 1100.0) == 199  # нижняя граница — последнему слою
    with pytest.raises(ValueError):
        layer_index(g, 1101.0)


def test_stages_in_si_and_continuous() -> None:
    sch, _ = _schedule()
    s0, s1 = sch.stages
    assert s0.t_start == 0.0
    assert s1.t_start == s0.t_end
    assert sch.total_duration == pytest.approx(240 * 86_400)
    assert s0.q_in == pytest.approx(50 / 3600)
    assert s0.mode is LeachMode.direct and s1.mode is LeachMode.reverse


def test_layer_indices_of_strings() -> None:
    sch, _ = _schedule()
    s0, s1 = sch.stages
    assert (s0.k_in, s0.k_out, s0.k_blanket) == (190, 180, 160)
    assert s0.k_out < s0.k_in  # прямоток: отбор выше закачки
    assert s1.k_out > s1.k_in  # противоток: отбор ниже закачки


def test_stage_at() -> None:
    sch, _ = _schedule()
    t1 = sch.stages[1].t_start
    assert sch.stage_at(0.0) is sch.stages[0]
    assert sch.stage_at(t1 - 1.0) is sch.stages[0]
    assert sch.stage_at(t1) is sch.stages[1]
    with pytest.raises(ValueError):
        sch.stage_at(sch.total_duration)
