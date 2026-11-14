from pathlib import Path
from typing import Any

import pytest
import yaml

from saltcav.materials import LibraryValidationError, load_library

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "materials_library"
FILENAMES = {"materials": "synthetic-rock.yaml", "fluids": "synthetic-fluid.yaml"}


def _raw(category: str = "materials") -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load((FIXTURES / category / FILENAMES[category]).read_text())
    return raw


def _write(
    root: Path, raw: dict[str, Any], category: str = "materials", filename: str = "bad.yaml"
) -> Path:
    path = root / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _assert_error(root: Path, path: Path, field: str) -> str:
    with pytest.raises(LibraryValidationError) as caught:
        load_library(root)
    message = str(caught.value)
    assert str(path) in message
    assert field in message
    return message


def test_loads_synthetic_material_and_fluid_fixtures() -> None:
    library = load_library(FIXTURES)

    assert set(library.materials) == {"test-synthetic-rock"}
    assert set(library.fluids) == {"test-synthetic-fluid"}
    rock = library.materials["test-synthetic-rock"]
    fluid = library.fluids["test-synthetic-fluid"]
    assert rock.kind == "synthetic_rock"
    assert rock.properties["youngs_modulus"].unit == "Pa"
    assert rock.properties["density"].value == 2000.0
    assert fluid.properties["dynamic_viscosity"].range == (0.001, 0.1)
    assert "not a physical source" in fluid.properties["density"].source.reference


@pytest.mark.parametrize(
    ("marker", "replacement", "key"),
    [
        (
            "kind: synthetic_rock\n",
            "kind: synthetic_rock\nkind: duplicate-kind\n",
            "kind",
        ),
        ("  youngs_modulus:\n", "  density: {}\n  youngs_modulus:\n", "density"),
        (
            "    value: 2000.0\n",
            "    value: 2000.0\n    value: 2100.0\n",
            "value",
        ),
        ("    source:\n", "    source: {}\n    source:\n", "source"),
    ],
)
def test_duplicate_yaml_key_is_rejected(
    tmp_path: Path, marker: str, replacement: str, key: str
) -> None:
    text = (FIXTURES / "materials" / FILENAMES["materials"]).read_text(encoding="utf-8")
    assert marker in text
    path = tmp_path / "materials" / "duplicate.yaml"
    path.parent.mkdir()
    path.write_text(text.replace(marker, replacement, 1), encoding="utf-8")

    message = _assert_error(tmp_path, path, key)
    assert "повторяющийся ключ" in message


@pytest.mark.parametrize("field", ["id", "name", "kind", "properties"])
def test_required_entry_fields(tmp_path: Path, field: str) -> None:
    raw = _raw()
    del raw[field]
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, field)


def test_property_source_is_required(tmp_path: Path) -> None:
    raw = _raw()
    del raw["properties"]["density"]["source"]
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.source")


def test_blank_source_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["properties"]["density"]["source"]["reference"] = "  "
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.source.reference")


@pytest.mark.parametrize("unit", ["Pa", "kg/l"])
def test_wrong_or_non_si_unit_is_rejected(tmp_path: Path, unit: str) -> None:
    raw = _raw("fluids")
    raw["properties"]["density"]["unit"] = unit
    path = _write(tmp_path, raw, "fluids")
    message = _assert_error(tmp_path, path, "properties.density.unit")
    assert "density" in message
    assert "kg/m3" in message


def test_unknown_quantity_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["properties"]["density"]["quantity"] = "colour"
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.quantity")


def test_inverted_range_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["properties"]["density"]["range"] = [3000.0, 1000.0]
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.range")


def test_value_outside_range_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["properties"]["density"]["value"] = 3001.0
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.value")


@pytest.mark.parametrize("bad", [True, False])
def test_bool_value_is_rejected(tmp_path: Path, bad: bool) -> None:
    raw = _raw()
    raw["properties"]["density"]["value"] = bad
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.value")


@pytest.mark.parametrize(("index", "bad"), [(0, True), (0, False), (1, True), (1, False)])
def test_bool_range_bound_is_rejected(tmp_path: Path, index: int, bad: bool) -> None:
    raw = _raw()
    raw["properties"]["density"]["range"][index] = bad
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, f"properties.density.range.{index}")


@pytest.mark.parametrize(
    ("value", "bounds"),
    [(2000, [1000, 3000]), (2000.5, [1000.5, 3000.5])],
)
def test_integer_and_float_numbers_are_accepted(
    tmp_path: Path, value: float, bounds: list[float]
) -> None:
    raw = _raw()
    raw["properties"]["density"]["value"] = value
    raw["properties"]["density"]["range"] = bounds
    _write(tmp_path, raw)

    density = load_library(tmp_path).materials["test-synthetic-rock"].properties["density"]
    assert density.value == value
    assert density.range == tuple(bounds)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_value_is_rejected(tmp_path: Path, bad: float) -> None:
    raw = _raw()
    raw["properties"]["density"]["value"] = bad
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, "properties.density.value")


@pytest.mark.parametrize(("index", "bad"), [(0, float("nan")), (1, float("inf"))])
def test_non_finite_range_is_rejected(tmp_path: Path, index: int, bad: float) -> None:
    raw = _raw()
    raw["properties"]["density"]["range"][index] = bad
    path = _write(tmp_path, raw)
    _assert_error(tmp_path, path, f"properties.density.range.{index}")


def test_duplicate_id_across_categories_is_rejected(tmp_path: Path) -> None:
    first = _write(tmp_path, _raw(), filename="first.yaml")
    raw = _raw("fluids")
    raw["id"] = "test-synthetic-rock"
    second = _write(tmp_path, raw, "fluids", "second.yaml")

    message = _assert_error(tmp_path, second, "id")
    assert "test-synthetic-rock" in message
    assert str(first) in message


def test_yaml_is_loaded_safely(tmp_path: Path) -> None:
    path = tmp_path / "materials" / "unsafe.yaml"
    path.parent.mkdir()
    path.write_text("!!python/name:os.system\n", encoding="utf-8")
    _assert_error(tmp_path, path, "некорректный YAML")
