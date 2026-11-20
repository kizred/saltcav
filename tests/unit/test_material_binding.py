from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from saltcav.api import run
from saltcav.cli import main
from saltcav.core.config import dump_config, load_config
from saltcav.geometry import CavernGrid
from saltcav.leaching.ledges import interlayers_from_config
from saltcav.material_binding import LayerMaterialError
from saltcav.materials import load_library

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "example_ledges.yaml"
MATERIAL_ID = "test-sc006-synthetic-anhydrite"
SOURCE = "Synthetic SC-006 binding test fixture; not a physical source"


def _numeric_property(
    value: float, quantity: str = "density", unit: str = "kg/m3"
) -> dict[str, Any]:
    bounds = [0.1, 20.0] if quantity == "length" else [0.0, max(2.0 * value, 1.0)]
    return {
        "quantity": quantity,
        "value": value,
        "unit": unit,
        "range": bounds,
        "source": {
            "reference": SOURCE,
            "location": "Not applicable — arbitrary synthetic test value",
        },
    }


def _write_entry(
    root: Path,
    *,
    category: str = "materials",
    material_id: str = MATERIAL_ID,
    kind: str = "anhydrite",
    critical_span: float | None = 2.0,
    quantity: str = "length",
    unit: str = "m",
) -> None:
    properties = {"density": _numeric_property(2000.0)}
    if critical_span is not None:
        properties["critical_span"] = _numeric_property(critical_span, quantity, unit)
    raw = {
        "id": material_id,
        "name": "Synthetic SC-006 test material; not reference data",
        "kind": kind,
        "properties": properties,
    }
    path = root / category / "synthetic.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _raw(material_id: str = MATERIAL_ID, library: str = "../library") -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["materials_library"] = library
    raw["geology"]["layers"] = [
        {
            "kind": "anhydrite",
            "material_id": material_id,
            "top_m": 1040.0,
            "thickness_m": 1.0,
        }
    ]
    return raw


def _write_case(tmp_path: Path, raw: dict[str, Any], name: str = "case.yaml") -> Path:
    path = tmp_path / "config" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _error(path: Path) -> str:
    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)
    return str(caught.value)


def test_layer_material_is_resolved_with_provenance(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    path = _write_case(tmp_path, _raw())

    cfg = load_config(path)
    layer = cfg.geology.layers[0]

    assert layer.material_id == MATERIAL_ID
    assert layer.critical_span_m == 2.0
    assert layer.material_sources["critical_span_m"].reference == SOURCE
    assert cfg.materials_library == str((tmp_path / "library").resolve())
    assert cfg.geology.salt_density_kg_m3 == 2160.0
    assert cfg.mechanics.layer_materials[0].youngs_modulus_gpa == 60.0


def test_linked_config_roundtrip_preserves_material_and_sources(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    cfg = load_config(_write_case(tmp_path, _raw()))
    path = tmp_path / "config" / "dumped.yaml"

    dump_config(cfg, path)
    restored = load_config(path)

    assert restored == cfg
    assert restored.geology.layers[0].material_id == MATERIAL_ID
    assert restored.geology.layers[0].material_sources["critical_span_m"].reference == SOURCE


def test_unknown_material_id_is_an_error(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    path = _write_case(tmp_path, _raw("test-unknown-material"))

    message = _error(path)

    assert str(path) in message
    assert "geology.layers.0.material_id" in message
    assert "test-unknown-material" in message
    assert "неизвестный id" in message


def test_fluid_id_cannot_be_assigned_to_a_layer(tmp_path: Path) -> None:
    fluid_id = "test-sc006-synthetic-fluid"
    _write_entry(
        tmp_path / "library",
        category="fluids",
        material_id=fluid_id,
        kind="nacl_brine",
        critical_span=None,
    )
    path = _write_case(tmp_path, _raw(fluid_id))

    message = _error(path)

    assert str(path) in message
    assert "geology.layers.0.material_id" in message
    assert "fluids" in message
    assert "materials" in message


def test_layer_and_material_kind_must_match(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library", kind="clay")
    path = _write_case(tmp_path, _raw())

    message = _error(path)

    assert str(path) in message
    assert "geology.layers.0.kind" in message
    assert "anhydrite" in message
    assert "clay" in message


@pytest.mark.parametrize(
    ("quantity", "unit"), [("stress", "Pa"), ("length", "Pa")], ids=["quantity", "unit"]
)
def test_applied_property_quantity_and_unit_are_checked(
    tmp_path: Path, quantity: str, unit: str
) -> None:
    _write_entry(tmp_path / "library", quantity=quantity, unit=unit)
    path = _write_case(tmp_path, _raw())

    message = _error(path)

    assert str(path) in message
    assert "geology.layers.0.critical_span_m" in message
    assert "critical_span" in message
    assert "length" in message
    assert "m" in message


def test_explicit_layer_value_has_priority(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    raw = _raw()
    raw["geology"]["layers"][0]["critical_span_m"] = 7.0
    path = _write_case(tmp_path, raw)

    layer = load_config(path).geology.layers[0]

    assert layer.critical_span_m == 7.0
    assert "critical_span_m" not in layer.material_sources


def test_explicit_null_is_filled_and_missing_property_uses_old_default(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    raw = _raw()
    raw["geology"]["layers"][0]["critical_span_m"] = None
    path = _write_case(tmp_path, raw)
    assert load_config(path).geology.layers[0].critical_span_m == 2.0

    other = tmp_path / "without-span"
    _write_entry(other / "library", critical_span=None)
    path = _write_case(other, _raw())
    cfg = load_config(path)
    layer = cfg.geology.layers[0]
    assert layer.critical_span_m is None
    interlayers = interlayers_from_config(cfg.geology, CavernGrid.from_config(cfg.grid))
    assert interlayers[0].critical_span == cfg.geology.ledge_span_ratio * layer.thickness_m


def test_two_layers_do_not_mutate_shared_library_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    _write_entry(root)
    library = load_library(root)
    before = library.materials[MATERIAL_ID].model_dump_json()
    monkeypatch.setattr("saltcav.material_binding.load_library", lambda _: library)
    raw = _raw()
    first = raw["geology"]["layers"][0]
    second = {**first, "top_m": 1075.0, "thickness_m": 0.5}
    first["critical_span_m"] = 6.0
    raw["geology"]["layers"] = [first, second]

    layers = load_config(_write_case(tmp_path, raw)).geology.layers

    assert [layer.critical_span_m for layer in layers] == [6.0, 2.0]
    assert library.materials[MATERIAL_ID].model_dump_json() == before
    assert "critical_span_m" not in layers[0].material_sources
    assert layers[1].material_sources["critical_span_m"].reference == SOURCE


def test_relative_library_path_does_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_entry(tmp_path / "library")
    path = _write_case(tmp_path, _raw())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cfg = load_config(path)

    assert cfg.geology.layers[0].critical_span_m == 2.0


def test_missing_library_and_missing_root_setting_are_errors(tmp_path: Path) -> None:
    path = _write_case(tmp_path, _raw(library="../missing"))
    message = _error(path)
    assert str(path) in message
    assert "geology.layers.0.material_id" in message
    assert "materials_library" in message

    raw = _raw()
    del raw["materials_library"]
    path = _write_case(tmp_path, raw, "no-root.yaml")
    message = _error(path)
    assert str(path) in message
    assert "geology.layers.0.material_id" in message
    assert "materials_library" in message


def test_legacy_config_does_not_require_a_library(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: object) -> None:
        raise AssertionError("load_library не должен вызываться без material_id")

    monkeypatch.setattr("saltcav.material_binding.load_library", fail)
    cfg = load_config(EXAMPLE)
    assert cfg.materials_library is None
    assert cfg.geology.layers[0].material_id is None


def test_reference_and_explicit_properties_have_equivalent_results(tmp_path: Path) -> None:
    _write_entry(tmp_path / "library")
    linked_raw = _raw()
    linked_raw["grid"]["nz"] = 50
    for stage, days in zip(linked_raw["schedule"]["stages"], (20.0, 60.0), strict=True):
        stage["duration_days"] = days
    linked_path = _write_case(tmp_path, linked_raw, "linked.yaml")

    explicit_raw = _raw()
    explicit_raw["grid"]["nz"] = 50
    for stage, days in zip(explicit_raw["schedule"]["stages"], (20.0, 60.0), strict=True):
        stage["duration_days"] = days
    layer = explicit_raw["geology"]["layers"][0]
    del layer["material_id"]
    layer["critical_span_m"] = 2.0
    del explicit_raw["materials_library"]
    explicit_path = _write_case(tmp_path, explicit_raw, "explicit.yaml")

    linked_cfg = load_config(linked_path)
    explicit_cfg = load_config(explicit_path)
    linked_params = linked_cfg.model_dump(mode="json")
    explicit_params = explicit_cfg.model_dump(mode="json")
    for params in (linked_params, explicit_params):
        params.pop("materials_library")
        for layer_params in params["geology"]["layers"]:
            layer_params.pop("material_id")
            layer_params.pop("material_sources")
    assert linked_params == explicit_params
    linked_interlayer = interlayers_from_config(
        linked_cfg.geology, CavernGrid.from_config(linked_cfg.grid)
    )[0]
    explicit_interlayer = interlayers_from_config(
        explicit_cfg.geology, CavernGrid.from_config(explicit_cfg.grid)
    )[0]
    assert linked_interlayer == explicit_interlayer
    assert main(["check", str(linked_path)]) == 0

    linked = run(linked_path)
    explicit = run(explicit_path)

    assert linked.collapses.size > 0
    # Параметры и сетка совпадают: допуск покрывает только округление, не разницу моделей.
    for name in ("radius", "volume", "floor_depth", "dissolved_salt", "produced_salt"):
        np.testing.assert_allclose(
            getattr(linked, name),
            getattr(explicit, name),
            rtol=1e-12,
            atol=1e-12,
            err_msg=name,
        )

    fallback_raw = deepcopy(explicit_raw)
    del fallback_raw["geology"]["layers"][0]["critical_span_m"]
    fallback = run(_write_case(tmp_path, fallback_raw, "fallback.yaml"))
    assert not np.allclose(linked.radius, fallback.radius, rtol=1e-9, atol=1e-9)
