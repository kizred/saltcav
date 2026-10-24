"""Этап 3: растворение по азимуту."""

import time

import numpy as np
import pytest

from saltcav.core.config import RunConfig, load_config
from saltcav.geometry import CavernGrid
from saltcav.leaching import simulate
from saltcav.leaching.properties import WallProperties
from tests.unit.test_config import EXAMPLE


def _cfg(nphi: int, a: float = 0.0, az: float = 0.0, days: float | None = None) -> RunConfig:
    raw = load_config(EXAMPLE).model_dump(mode="json")
    raw["grid"]["nphi"] = nphi
    raw["geology"].update(k_variation=a, k_variation_azimuth_deg=az)
    if days is not None:
        for st in raw["schedule"]["stages"]:
            st["duration_days"] = days
    return RunConfig.model_validate(raw)


@pytest.fixture(scope="module")
def axi():  # type: ignore[no-untyped-def]
    return simulate(_cfg(1, days=40))


@pytest.fixture(scope="module")
def asym():  # type: ignore[no-untyped-def]
    return simulate(_cfg(36, a=0.3, az=45.0, days=40))


def test_properties_from_config() -> None:
    g = CavernGrid.uniform(0, 10, 10, 8)
    p = WallProperties.from_config(_cfg(8, a=0.2, az=90.0).geology, g)
    assert p.k_mult.shape == (11, 8)
    assert p.k_mult.mean() == pytest.approx(1.0)  # косинус среднее не меняет
    j = int(np.argmax(p.k_mult[0]))
    assert np.degrees(g.phi_centers[j]) == pytest.approx(90.0, abs=360 / 8)
    with pytest.raises(ValueError):
        WallProperties(-np.ones((2, 2)), np.zeros((2, 2)), 2160.0)


def test_homogeneous_3d_equals_axisymmetric(axi) -> None:  # type: ignore[no-untyped-def]
    # критерий приёмки этапа 3: при однородных свойствах 3D совпадает с осесимметрией
    h3 = simulate(_cfg(12, days=40))
    for r1, r3 in zip(axi.radius, h3.radius, strict=True):
        np.testing.assert_allclose(r3, np.broadcast_to(r1, r3.shape), rtol=1e-10, atol=1e-12)
    assert h3.volume[-1] == pytest.approx(axi.volume[-1], rel=1e-10)


def test_fast_direction_grows_more(asym) -> None:  # type: ignore[no-untyped-def]
    r = asym.radius[-1]
    g = asym.grid
    row = int(np.argmax(r.max(axis=1)))
    j_fast = int(np.argmax(r[row]))
    assert np.degrees(g.phi_centers[j_fast]) == pytest.approx(45.0, abs=10.0)
    off = asym.cavern().centroid_offset()[row]
    assert np.degrees(np.arctan2(off[1], off[0])) == pytest.approx(45.0, abs=5.0)


def test_asymmetric_salt_balance(asym) -> None:  # type: ignore[no-untyped-def]
    lhs = asym.dissolved_salt[-1]
    rhs = asym.salt_in_brine[-1] - asym.salt_in_brine[0] + asym.produced_salt[-1]
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_symmetric_variation_keeps_volume_close(axi, asym) -> None:  # type: ignore[no-untyped-def]
    # косинусная вариация не меняет средний k — объём близок к осесимметричному
    assert asym.volume[-1] == pytest.approx(axi.volume[-1], rel=0.05)


def test_36_sectors_fast_enough() -> None:
    t0 = time.perf_counter()
    simulate(_cfg(36, a=0.3))
    assert time.perf_counter() - t0 < 10.0  # критерий этапа 3: ≤ 10 с
