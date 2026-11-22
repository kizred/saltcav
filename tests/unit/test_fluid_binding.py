from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import saltcav.leaching.simulator as simulator
from saltcav.core.config import RunConfig, dump_config, load_config
from saltcav.leaching import History, simulate
from saltcav.material_binding import InjectionFluidError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "example.yaml"
LIBRARY = ROOT / "data" / "materials_library"
FLUID_ID = "test-synthetic-fluid"
ROCK_ID = "test-synthetic-rock"
CONCENTRATION = 120.0


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["grid"]["nz"] = 50
    raw["grid"]["nphi"] = 1
    raw["geology"]["layers"] = []
    raw["geology"]["insoluble_fraction_sd"] = 0.0
    raw["schedule"]["stages"] = [dict(raw["schedule"]["stages"][0], duration_days=10.0)]
    raw["leaching"] = {"output_every_days": 5.0}
    return raw


def _linked_raw(fluid_id: str = FLUID_ID) -> dict[str, Any]:
    raw = _raw()
    raw["materials_library"] = str(LIBRARY)
    raw["leaching"]["injection_fluid_id"] = fluid_id
    return raw


def _write_case(tmp_path: Path, raw: dict[str, Any], name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _simulate_with_direct_c_in(
    cfg: RunConfig, c_in: float, monkeypatch: pytest.MonkeyPatch
) -> History:
    def bind_c_in(function: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            kwargs["c_in"] = c_in
            return function(*args, **kwargs)

        return call

    with monkeypatch.context() as patch:
        for name in ("plume_fluxes", "plume_transport_step", "column_step"):
            patch.setattr(simulator, name, bind_c_in(getattr(simulator, name)))
        return simulator.simulate(cfg)


def test_unknown_injection_fluid_id_is_rejected(tmp_path: Path) -> None:
    fluid_id = "test-sc012-unknown-fluid"
    path = _write_case(tmp_path, _linked_raw(fluid_id), "unknown.yaml")

    with pytest.raises(InjectionFluidError) as caught:
        load_config(path)

    message = str(caught.value)
    assert str(path) in message
    assert "leaching.injection_fluid_id" in message
    assert fluid_id in message
    assert "неизвестный id" in message


def test_rock_id_cannot_be_assigned_to_injection_fluid(tmp_path: Path) -> None:
    path = _write_case(tmp_path, _linked_raw(ROCK_ID), "rock.yaml")

    with pytest.raises(InjectionFluidError) as caught:
        load_config(path)

    message = str(caught.value)
    assert str(path) in message
    assert "leaching.injection_fluid_id" in message
    assert ROCK_ID in message
    assert "materials" in message
    assert "fluids" in message


@pytest.mark.parametrize("explicit_value", [25.0, 0.0])
def test_explicit_injection_concentration_has_priority(
    tmp_path: Path, explicit_value: float
) -> None:
    raw = _linked_raw()
    raw["leaching"]["injection_concentration_kg_m3"] = explicit_value

    cfg = load_config(_write_case(tmp_path, raw, "priority.yaml"))

    assert cfg.leaching.injection_fluid_id == FLUID_ID
    assert cfg.leaching.injection_concentration_kg_m3 == explicit_value


@pytest.mark.parametrize("plume", [True, False], ids=["plume", "column"])
def test_fluid_reference_and_direct_model_input_have_equivalent_results(
    tmp_path: Path, plume: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked_raw = _linked_raw()
    linked_raw["leaching"]["plume"] = plume
    linked_cfg = load_config(_write_case(tmp_path, linked_raw, "linked.yaml"))
    direct_raw = _raw()
    direct_raw["leaching"]["plume"] = plume
    direct_cfg = load_config(_write_case(tmp_path, direct_raw, "direct.yaml"))
    assert linked_cfg.leaching.injection_concentration_kg_m3 == CONCENTRATION
    assert "injection_concentration_kg_m3" not in direct_cfg.leaching.model_fields_set

    linked = simulate(linked_cfg)
    direct = _simulate_with_direct_c_in(direct_cfg, CONCENTRATION, monkeypatch)

    for name in ("radius", "volume", "concentration", "dissolved_salt", "produced_salt"):
        np.testing.assert_allclose(
            getattr(linked, name), getattr(direct, name), rtol=1e-12, atol=1e-12, err_msg=name
        )

    control = _simulate_with_direct_c_in(direct_cfg, 0.0, monkeypatch)
    assert not np.allclose(linked.volume, control.volume, rtol=1e-6, atol=1e-6)


def test_legacy_config_without_fluid_reference_keeps_numerical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: object) -> None:
        raise AssertionError("load_library не должен вызываться без ссылок")

    monkeypatch.setattr("saltcav.material_binding.load_library", fail)
    cfg = load_config(_write_case(tmp_path, _raw(), "legacy.yaml"))

    hist = simulate(cfg)

    assert cfg.leaching.injection_fluid_id is None
    assert cfg.leaching.injection_concentration_kg_m3 == 0.0
    assert hist.volume[-1] == pytest.approx(423.41846901994495, rel=1e-10)
    assert hist.outlet_concentration[-1] == pytest.approx(101.36342690110702, rel=1e-10)
    assert float(hist.radius[-1].sum()) == pytest.approx(20.831567575399514, rel=1e-10)


def test_linked_config_roundtrip_preserves_fluid_and_value(tmp_path: Path) -> None:
    cfg = load_config(_write_case(tmp_path, _linked_raw(), "linked.yaml"))
    dumped = tmp_path / "dumped.yaml"
    dump_config(cfg, dumped)

    assert load_config(dumped) == cfg
