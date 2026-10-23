import numpy as np
import pytest

from saltcav.core import brine


def test_water_density_reference_points() -> None:
    assert brine.water_density(4.0) == pytest.approx(999.97, abs=0.2)
    assert brine.water_density(20.0) == pytest.approx(998.21)


def test_saturation_mass_fraction_25c() -> None:
    assert brine.saturation_mass_fraction(25.0) == pytest.approx(0.2645, rel=1e-3)


def test_saturated_brine_density_20c() -> None:
    w = brine.saturation_mass_fraction(20.0)
    assert brine.brine_density(w, 20.0) == pytest.approx(1197.0, rel=5e-3)


def test_saturation_concentration_magnitude() -> None:
    # ≈ 317 кг/м³ при 20 °C; растёт с температурой
    cs20 = float(brine.saturation_concentration(20.0))
    assert cs20 == pytest.approx(317.0, rel=1e-2)
    assert brine.saturation_concentration(60.0) > cs20


def test_density_monotonic_in_concentration() -> None:
    c = np.linspace(0, float(brine.saturation_concentration(20.0)), 50)
    rho = brine.density_from_concentration(c, 20.0)
    assert np.all(np.diff(rho) > 0)
    assert rho[0] == pytest.approx(998.21)


@pytest.mark.parametrize("t", [10.0, 20.0, 45.0])
def test_concentration_roundtrip(t: float) -> None:
    w = np.linspace(0, 0.26, 27)
    c = brine.concentration_from_mass_fraction(w, t)
    np.testing.assert_allclose(brine.mass_fraction_from_concentration(c, t), w, atol=1e-5)


def test_water_viscosity_20c() -> None:
    assert brine.water_viscosity(20.0) == pytest.approx(1.002e-3, rel=2e-2)


def test_diffusivity_reference_and_trend() -> None:
    assert brine.diffusivity(25.0) == pytest.approx(brine.D_NACL_25C)
    assert brine.diffusivity(50.0) > brine.diffusivity(25.0)


@pytest.mark.parametrize("bad", [-1.0, 101.0])
def test_temperature_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        brine.water_density(bad)


def test_mass_fraction_out_of_range() -> None:
    with pytest.raises(ValueError):
        brine.brine_density(0.4)
    with pytest.raises(ValueError):
        brine.mass_fraction_from_concentration(1000.0)
