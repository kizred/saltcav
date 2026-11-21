"""SC-013: выборка свойств слоя из диапазонов синтетической библиотеки."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from saltcav.core.config import LayerConfig, RunConfig, load_config
from saltcav.geometry import CavernGrid
from saltcav.leaching.ledges import interlayers_from_config
from saltcav.leaching.simulator import simulate
from saltcav.material_binding import LayerMaterialError, sampled_layer_property_ranges
from saltcav.stochastic.ensemble import make_realization, run_ensemble
from saltcav.stochastic.sampling import design, parameter_names

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "example_ledges.yaml"
ENSEMBLE_EXAMPLE = ROOT / "configs" / "example_ensemble.yaml"
MATERIAL_ID = "test-sc013-synthetic-anhydrite"
BOUNDS = (1.0, 4.0)
SOURCE = "Synthetic SC-013 sampling test fixture; not a physical source"


def _numeric_property(
    value: float, quantity: str = "density", unit: str = "kg/m3"
) -> dict[str, Any]:
    bounds = BOUNDS if quantity == "length" else [0.0, 2.0 * value]
    return {
        "quantity": quantity,
        "value": value,
        "unit": unit,
        "range": list(bounds),
        "source": {
            "reference": SOURCE,
            "location": "Not applicable — arbitrary synthetic test value",
        },
    }


def _write_library(
    root: Path, *, include_critical_span: bool = True, include_range: bool = True
) -> Path:
    properties = {"density": _numeric_property(2000.0)}
    if include_critical_span:
        critical_span = _numeric_property(2.0, "length", "m")
        if not include_range:
            del critical_span["range"]
        properties["critical_span"] = critical_span
    raw = {
        "id": MATERIAL_ID,
        "name": "Synthetic SC-013 test material; not reference data",
        "kind": "anhydrite",
        "properties": properties,
    }
    path = root / "materials" / "synthetic.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["materials_library"] = "../library"
    raw["grid"]["nz"] = 50
    raw["geology"]["insoluble_fraction_sd"] = 0.0
    raw["geology"]["layers"] = [
        {
            "kind": "anhydrite",
            "material_id": MATERIAL_ID,
            "top_m": 1040.0,
            "thickness_m": 1.0,
        },
        {
            "kind": "anhydrite",
            "material_id": MATERIAL_ID,
            "top_m": 1075.0,
            "thickness_m": 0.5,
        },
    ]
    for stage, days in zip(raw["schedule"]["stages"], (20.0, 60.0), strict=True):
        stage["duration_days"] = days
    raw["ensemble"] = {
        "n_realizations": 8,
        "seed": 17,
        "sampler": "lhs",
        "layer_properties": [{"layer": 0, "property": "critical_span"}],
    }
    return raw


def _write_case(tmp_path: Path, raw: dict[str, Any], name: str = "case.yaml") -> Path:
    path = tmp_path / "config" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_sampling_is_seeded_local_and_matches_explicit_single_run(tmp_path: Path) -> None:
    library_path = _write_library(tmp_path / "library")
    path = _write_case(tmp_path, _raw())
    cfg = load_config(path)
    original_cfg = cfg.model_dump_json()
    original_library = library_path.read_bytes()

    realizations = [make_realization(cfg, i) for i in range(cfg.ensemble.n_realizations)]
    values = np.array([r.cfg.geology.layers[0].critical_span_m for r in realizations])
    expected = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * design(cfg, 8, 17)[:, -1]

    assert len(np.unique(values)) == values.size
    assert np.all((BOUNDS[0] <= values) & (values <= BOUNDS[1]))
    np.testing.assert_array_equal(values, expected)
    assert parameter_names(cfg)[-1] == "layer0_critical_span"
    for real, value in zip(realizations, values, strict=True):
        assert real.params[-1] == value
        assert real.cfg.geology.layers[1] == cfg.geology.layers[1]
        interlayers = interlayers_from_config(
            real.cfg.geology, CavernGrid.from_config(real.cfg.grid)
        )
        assert interlayers[0].critical_span == value
        assert interlayers[1].critical_span == cfg.geology.layers[1].critical_span_m

    repeated = make_realization(cfg, 3)
    assert repeated.cfg == realizations[3].cfg
    np.testing.assert_array_equal(repeated.params, realizations[3].params)
    np.testing.assert_array_equal(repeated.props.k_mult, realizations[3].props.k_mult)
    np.testing.assert_array_equal(repeated.props.insoluble, realizations[3].props.insoluble)

    chosen = max(realizations, key=lambda r: abs(r.cfg.geology.layers[0].critical_span_m - 2.0))
    sampled_value = chosen.cfg.geology.layers[0].critical_span_m
    explicit_raw = cfg.model_dump(mode="json")
    explicit_raw["ensemble"]["layer_properties"] = []
    explicit_raw["geology"]["layers"][0]["critical_span_m"] = sampled_value
    explicit_cfg = RunConfig.model_validate(explicit_raw)
    sampled_history = simulate(chosen.cfg, chosen.props)
    explicit_history = simulate(explicit_cfg)
    for name in ("radius", "volume", "floor_depth", "produced_salt", "dissolved_salt"):
        np.testing.assert_array_equal(
            getattr(sampled_history, name), getattr(explicit_history, name), err_msg=name
        )

    assert cfg.model_dump_json() == original_cfg
    assert library_path.read_bytes() == original_library


def test_ensemble_with_sampling_repeats_bit_for_bit(tmp_path: Path) -> None:
    _write_library(tmp_path / "library")
    raw = _raw()
    raw["grid"]["nz"] = 20
    for stage, days in zip(raw["schedule"]["stages"], (2.0, 3.0), strict=True):
        stage["duration_days"] = days
    cfg = load_config(_write_case(tmp_path, raw))

    first = run_ensemble(cfg, n=4, seed=23, jobs=1)
    second = run_ensemble(cfg, n=4, seed=23, jobs=2)

    assert first.param_names[-1] == "layer0_critical_span"
    assert len(np.unique(first.params[:, -1])) == 4
    assert np.all((BOUNDS[0] <= first.params[:, -1]) & (first.params[:, -1] <= BOUNDS[1]))
    np.testing.assert_array_equal(second.params, first.params)
    np.testing.assert_array_equal(second.radius, first.radius)
    np.testing.assert_array_equal(second.volume_t, first.volume_t)


def test_missing_property_range_is_an_error(tmp_path: Path) -> None:
    _write_library(tmp_path / "library", include_critical_span=False)
    path = _write_case(tmp_path, _raw())

    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)

    message = str(caught.value)
    assert "geology.layers.0.critical_span_m" in message
    assert "critical_span" in message and "range" in message


def test_missing_range_field_identifies_layer_and_property(tmp_path: Path) -> None:
    _write_library(tmp_path / "library", include_range=False)
    raw = _raw()
    raw["ensemble"]["layer_properties"][0]["layer"] = 1
    path = _write_case(tmp_path, raw)

    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)

    message = str(caught.value)
    assert "geology.layers.1.critical_span_m" in message
    assert "properties.critical_span.range" in message


def test_explicit_layer_value_conflicts_with_sampling(tmp_path: Path) -> None:
    _write_library(tmp_path / "library")
    raw = _raw()
    raw["geology"]["layers"][0]["critical_span_m"] = 7.0
    path = _write_case(tmp_path, raw)

    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)

    message = str(caught.value)
    assert "geology.layers.0.critical_span_m" in message
    assert "critical_span" in message and "конфликтует" in message


def test_sc011_local_override_conflicts_with_sampling(tmp_path: Path) -> None:
    _write_library(tmp_path / "library")
    if "material_overrides" in LayerConfig.model_fields:
        raw = _raw()
        raw["geology"]["layers"][0]["material_overrides"] = {"critical_span": 3.0}
        path = _write_case(tmp_path, raw)
        with pytest.raises(LayerMaterialError) as caught:
            load_config(path)
    else:
        path = _write_case(tmp_path, _raw())
        cfg = load_config(path)
        layer = cfg.geology.layers[0].model_copy(
            update={"material_overrides": {"critical_span": 3.0}}
        )
        layers = [layer, *cfg.geology.layers[1:]]
        overridden = cfg.model_copy(
            update={"geology": cfg.geology.model_copy(update={"layers": layers})}
        )
        with pytest.raises(LayerMaterialError) as caught:
            sampled_layer_property_ranges(overridden, path)

    message = str(caught.value)
    assert "geology.layers.0.material_overrides.critical_span" in message
    assert "critical_span" in message and "конфликтует" in message


def test_existing_ledge_sampling_conflicts_with_property_sampling(tmp_path: Path) -> None:
    _write_library(tmp_path / "library")
    raw = _raw()
    raw["uncertainty"] = {"ledge_span_log_sd": 0.2}
    path = _write_case(tmp_path, raw)

    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)

    message = str(caught.value)
    assert "geology.layers.0.critical_span_m" in message
    assert "critical_span" in message and "ledge_span_log_sd" in message


def test_duplicate_property_sampling_rule_is_an_error(tmp_path: Path) -> None:
    _write_library(tmp_path / "library")
    raw = _raw()
    raw["ensemble"]["layer_properties"].append({"layer": 0, "property": "critical_span"})
    path = _write_case(tmp_path, raw)

    with pytest.raises(LayerMaterialError) as caught:
        load_config(path)

    message = str(caught.value)
    assert "geology.layers.0.critical_span_m" in message
    assert "critical_span" in message and "уже задано" in message


def test_legacy_single_and_fixed_seed_ensemble_regression() -> None:
    raw: dict[str, Any] = yaml.safe_load(ENSEMBLE_EXAMPLE.read_text(encoding="utf-8"))
    assert "layer_properties" not in raw["ensemble"]
    raw["grid"]["nz"] = 40
    raw["grid"]["nphi"] = 1
    for stage, days in zip(raw["schedule"]["stages"], (5.0, 10.0), strict=True):
        stage["duration_days"] = days
    raw["uncertainty"]["k_field_sigma"] = 0.0
    cfg = RunConfig.model_validate(raw)
    assert cfg.ensemble.layer_properties == []

    single = simulate(cfg)
    ensemble = run_ensemble(cfg, n=4, seed=11, jobs=1)

    assert _array_digest(single.radius[-1]) == (
        "7491daf7ef8252208c03842b353518c6760c8e627b3e0a678528ed630dc861c9"
    )
    assert _array_digest(single.volume) == (
        "e15739e5be06ddd44ac2335f610b8c1d2b471ea3262fa7624dc1925449e449a7"
    )
    assert _array_digest(ensemble.radius) == (
        "e354db8823c0f5cea8ac500d77931a6891778fe1b49309dad5c6d8aaeb749944"
    )
    assert _array_digest(ensemble.volume_t) == (
        "8c61425edc3fce0e7b183ae978b2436bd5d4cc514a31acdc386499319f72067d"
    )
    assert _array_digest(ensemble.params) == (
        "dea99f4a269fb0bc8bcd8a54f3b68b14d86adf730f0ca6c58812af53e31b514d"
    )
    assert ensemble.param_names == [
        "k0",
        "roof_ratio",
        "ledge_span",
        "rate_0",
        "rate_1",
        "layer0_top",
        "layer0_thickness",
        "layer1_top",
        "layer1_thickness",
    ]
