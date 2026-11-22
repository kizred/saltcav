from pathlib import Path

import numpy as np
import pytest
import yaml

import saltcav.api as sc
from saltcav.core.config import RunConfig
from saltcav.leaching import simulate

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "data" / "materials_library"


def _config() -> RunConfig:
    raw = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["grid"].update(nz=50, nphi=1)
    raw["geology"] = {}
    raw["leaching"] = {"output_every_days": 5.0}
    raw["schedule"]["stages"] = [dict(raw["schedule"]["stages"][0], duration_days=10.0)]
    raw.pop("mechanics", None)
    return RunConfig.model_validate(raw)


@pytest.mark.parametrize("plume", [True, False])
def test_legacy_bulk_defaults_and_results(plume: bool) -> None:
    legacy = sc.set(_config(), {"leaching.plume": plume})
    linked = sc.apply_bulk_salt(legacy, LIBRARY, "salt-default-1x")
    assert linked.geology == legacy.geology
    assert linked.mechanics == legacy.mechanics
    before, after = simulate(legacy), simulate(linked)
    for name in ("radius", "volume", "concentration", "dissolved_salt", "produced_salt"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name), err_msg=name)


def test_explicit_application_changes_only_supported_values() -> None:
    cfg = sc.set(
        _config(),
        {"geology.salt_density_kg_m3": 1900.0, "mechanics.youngs_modulus_gpa": 10.0},
    )
    changed = sc.apply_bulk_salt(cfg, LIBRARY, "salt-default-1x")
    assert changed.geology.salt_density_kg_m3 == 2160.0
    assert changed.mechanics.youngs_modulus_gpa == 25.0
    assert changed.mechanics.poisson == 0.25
    assert cfg.geology.salt_density_kg_m3 == 1900.0
    assert cfg.mechanics.youngs_modulus_gpa == 10.0
    assert changed.leaching == cfg.leaching

    logged = sc.apply_bulk_salt(cfg, LIBRARY, "jintan-fusi-layer-1")
    assert logged.geology.salt_density_kg_m3 == 2060.0
    assert logged.mechanics == cfg.mechanics


def test_rejects_quantity_mismatch_and_keeps_input(tmp_path: Path) -> None:
    entry = yaml.safe_load(
        (LIBRARY / "materials" / "salt-default-1x.yaml").read_text(encoding="utf-8")
    )
    entry["properties"]["density"].update(quantity="stress", unit="Pa")
    directory = tmp_path / "materials"
    directory.mkdir()
    (directory / "bad.yaml").write_text(yaml.safe_dump(entry), encoding="utf-8")
    cfg = _config()
    original = cfg.model_dump()
    with pytest.raises(ValueError, match="density"):
        sc.apply_bulk_salt(cfg, tmp_path, "salt-default-1x")
    assert cfg.model_dump() == original


def test_unknown_or_wrong_kind_is_rejected() -> None:
    cfg = _config()
    with pytest.raises(ValueError, match="неизвестный id"):
        sc.apply_bulk_salt(cfg, LIBRARY, "missing")
    with pytest.raises(ValueError, match="kind"):
        sc.apply_bulk_salt(cfg, LIBRARY, "test-synthetic-rock")
