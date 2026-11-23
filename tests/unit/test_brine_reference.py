from pathlib import Path

import numpy as np
import pytest

from saltcav.brine_reference import BrineReference, LegacyBrineTables
from saltcav.core import brine
from saltcav.materials import MaterialLibrary, load_library

LIBRARY = Path(__file__).resolve().parents[2] / "data" / "materials_library"


def test_nist_reference_points_and_interpolation() -> None:
    reference = BrineReference(load_library(LIBRARY))
    np.testing.assert_allclose(
        reference.saturation_mass_fraction([20, 60, 100]), [0.264, 0.2703, 0.2806]
    )
    np.testing.assert_allclose(reference.saturated_density([20, 60, 100]), [1200.4, 1185.4, 1168.2])
    np.testing.assert_allclose(
        reference.viscosity_6molal([20, 60, 100]), [0.0019508, 0.0009246, 0.0005648]
    )
    assert reference.saturated_density(25) == pytest.approx(1198.95)
    assert reference.viscosity_6molal(25) == pytest.approx(0.00175535)


@pytest.mark.parametrize("temperature", [19.9, 100.1, float("nan"), float("inf")])
def test_temperature_outside_series_is_rejected(temperature: float) -> None:
    reference = BrineReference(load_library(LIBRARY))
    with pytest.raises(ValueError, match="temperature"):
        reference.viscosity_6molal(temperature)


def test_viscosity_temperature_and_molality_interpolation() -> None:
    reference = BrineReference(load_library(LIBRARY))
    np.testing.assert_allclose(
        reference.viscosity_molal([0, 1, 6], 20), [0.001002, 0.0010903, 0.0019508]
    )
    assert reference.viscosity_molal(0.25, 20) == pytest.approx(0.0010226)
    assert reference.viscosity_molal(6, 25) == pytest.approx(0.00175535)
    assert reference.viscosity_molal(4, 30) == pytest.approx(0.0012254)
    assert reference.viscosity_molal(5.5, 80) == pytest.approx(0.0006679)
    assert reference.viscosity_molal([[0], [6]], [20, 100]).shape == (2, 2)
    with pytest.raises(ValueError, match="molality"):
        reference.viscosity_molal(6.1, 20)
    with pytest.raises(ValueError, match="temperature"):
        reference.viscosity_molal(1, 10)


def test_missing_series_fails_at_binding() -> None:
    library = load_library(LIBRARY)
    entry = library.fluids["nacl-reference-tables"]
    properties = dict(entry.properties)
    del properties["viscosity_6molal_t_20"]
    bad_entry = entry.model_copy(update={"properties": properties})
    bad = MaterialLibrary(materials=library.materials, fluids={entry.id: bad_entry})
    with pytest.raises(ValueError, match="viscosity_6molal_t_20"):
        BrineReference(bad)


def test_binding_uses_library_values() -> None:
    library = load_library(LIBRARY)
    entry = library.fluids["nacl-reference-tables"]
    properties = dict(entry.properties)
    prop = properties["viscosity_m_0_t_20"]
    properties["viscosity_m_0_t_20"] = prop.model_copy(update={"value": 0.002})
    modified = entry.model_copy(update={"properties": properties})
    library = MaterialLibrary(materials=library.materials, fluids={entry.id: modified})
    assert BrineReference(library).viscosity_molal(0, 20) == pytest.approx(0.002)


def test_legacy_tables_reproduce_model() -> None:
    legacy = LegacyBrineTables(load_library(LIBRARY))
    temperature = np.linspace(0, 100, 201)
    salt = np.linspace(0, 0.30, 201)
    np.testing.assert_allclose(legacy.water_density(temperature), brine.water_density(temperature))
    np.testing.assert_allclose(
        legacy.saturation_mass_fraction(temperature), brine.saturation_mass_fraction(temperature)
    )
    np.testing.assert_allclose(
        legacy.brine_density(salt, temperature), brine.brine_density(salt, temperature), rtol=1e-14
    )
