"""Этап 4: нерастворимые прослои, карнизы и их обрушение."""

import numpy as np
import pytest

from saltcav.core.config import GeologyConfig, LayerConfig, LayerKind, RunConfig, load_config
from saltcav.geometry import CavernGrid
from saltcav.leaching import simulate
from saltcav.leaching.ledges import (
    Interlayer,
    collapse_ledges,
    interlayers_from_config,
    overhang,
)
from tests.unit.test_config import EXAMPLE

GRID = CavernGrid.uniform(1000.0, 1100.0, 100)  # узлы через 1 м


def _geo(*layers: LayerConfig, ratio: float = 3.0) -> GeologyConfig:
    return GeologyConfig(layers=list(layers), ledge_span_ratio=ratio)


def test_interlayers_from_config() -> None:
    geo = _geo(
        LayerConfig(kind=LayerKind.anhydrite, top_m=1040.0, thickness_m=2.0),
        LayerConfig(kind=LayerKind.clay, top_m=1075.2, thickness_m=0.3),  # тоньше шага
        LayerConfig(kind=LayerKind.halite, top_m=1050.0, thickness_m=5.0),  # растворим
        LayerConfig(kind=LayerKind.clay, top_m=1200.0, thickness_m=1.0),  # вне сетки
        LayerConfig(kind=LayerKind.clay, top_m=1090.0, thickness_m=1.0, critical_span_m=7.0),
    )
    ils = interlayers_from_config(geo, GRID)
    assert [(il.i0, il.i1) for il in ils] == [(40, 42), (75, 75), (90, 91)]
    assert ils[0].critical_span == pytest.approx(6.0)  # 3 × 2 м
    assert ils[2].critical_span == pytest.approx(7.0)  # задан явно


def test_overhang_double_sided_and_roof() -> None:
    il = Interlayer(2, 2, 1.0, LayerKind.anhydrite, 1002.0)
    r = np.array([[5.0], [6.0], [0.5], [8.0], [8.0]])
    assert overhang(r, il)[0] == pytest.approx(5.5)  # min(6, 8) − 0.5
    roof = np.array([[0.2], [0.2], [0.2], [8.0], [8.0]])  # камера только снизу
    assert overhang(roof, il)[0] <= 0.0


def test_collapse_only_beyond_critical_span() -> None:
    il = Interlayer(2, 2, 3.0, LayerKind.anhydrite, 1002.0)
    r = np.array([[5.0, 5.0], [6.0, 3.0], [0.5, 0.5], [8.0, 8.0], [8.0, 8.0]])
    z = np.arange(5.0)
    new, events = collapse_ledges(r, [il], z)
    assert [ev.sector for ev in events] == [0]  # вылет 5,5 ≥ 3; во втором секторе 2,5 < 3
    assert new[2, 0] == pytest.approx(6.0) and new[2, 1] == pytest.approx(0.5)


def _example(ratio: float, days: float = 80) -> RunConfig:
    raw = load_config(EXAMPLE).model_dump(mode="json")
    raw["geology"]["layers"] = [
        dict(kind="anhydrite", top_m=1040.0, thickness_m=1.0),
        dict(kind="clay", top_m=1075.0, thickness_m=0.5),
    ]
    raw["geology"]["ledge_span_ratio"] = ratio
    for st in raw["schedule"]["stages"]:
        st["duration_days"] = days
    return RunConfig.model_validate(raw)


@pytest.fixture(scope="module")
def strong():  # type: ignore[no-untyped-def]
    return simulate(_example(ratio=1000.0))


@pytest.fixture(scope="module")
def weak():  # type: ignore[no-untyped-def]
    return simulate(_example(ratio=3.0))


def test_interlayer_does_not_dissolve(strong) -> None:  # type: ignore[no-untyped-def]
    z = strong.grid.z_edges
    r = strong.radius[-1][:, 0]
    i = int(np.argmin(abs(z - 1075.0)))
    assert r[i] == pytest.approx(0.15)  # прослой остался стволом
    assert r[i - 1] > 2.0 and r[i + 1] > 2.0  # камера с обеих сторон — карниз
    assert not strong.collapses


def test_ledges_collapse_when_weak(weak, strong) -> None:  # type: ignore[no-untyped-def]
    assert weak.collapses
    for _, ev in weak.collapses:
        assert ev.span >= 1.5 - 1e-9  # глина 0,5 м × 3
    assert weak.insoluble_bulk[-1] > strong.insoluble_bulk[-1]  # обломки в насыпи


def test_salt_balance_with_collapses(weak) -> None:  # type: ignore[no-untyped-def]
    for i in range(1, len(weak.times)):
        lhs = weak.dissolved_salt[i] + weak.collapse_fill_salt[i]
        rhs = weak.salt_in_brine[i] - weak.salt_in_brine[0] + weak.produced_salt[i]
        assert lhs == pytest.approx(rhs, rel=1e-9)
