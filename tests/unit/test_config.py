from pathlib import Path

import pytest
from pydantic import ValidationError

from saltcav.core.config import RunConfig, dump_config, load_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "example.yaml"


def test_example_config_is_valid() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.grid.nz == 200
    assert cfg.schedule.stages[0].rate_si == pytest.approx(50 / 3600)
    assert cfg.schedule.stages[0].duration_s == pytest.approx(60 * 86_400)


def test_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE)
    out = tmp_path / "run.yaml"
    dump_config(cfg, out)
    assert load_config(out) == cfg


def _raw() -> dict:  # type: ignore[type-arg]
    return load_config(EXAMPLE).model_dump(mode="json")


def test_unknown_field_rejected() -> None:
    raw = _raw()
    raw["grid"]["nr"] = 10
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_inverted_grid_rejected() -> None:
    raw = _raw()
    raw["grid"]["z_bottom_m"] = 800
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_depth_outside_grid_rejected() -> None:
    raw = _raw()
    raw["schedule"]["stages"][0]["injection_depth_m"] = 1200
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_blanket_below_strings_rejected() -> None:
    raw = _raw()
    raw["schedule"]["stages"][0]["blanket_depth_m"] = 1095
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_config_warnings_and_single_sector_ignores_azimuth() -> None:
    import numpy as np

    from saltcav.core.config import config_warnings
    from saltcav.geometry import CavernGrid
    from saltcav.leaching.properties import WallProperties

    base = load_config(EXAMPLE)
    assert config_warnings(base) == []
    geo = base.geology.model_copy(update={"k_variation": 0.3, "k_variation_azimuth_deg": 0.0})
    cfg = base.model_copy(update={"geology": geo})
    assert [loc for loc, _ in config_warnings(cfg)] == [("geology", "k_variation")]
    props = WallProperties.from_config(cfg.geology, CavernGrid.from_config(cfg.grid))
    # раньше единственный сектор (φ = 180°) получал множитель 1 + 0,3·cos 180° = 0,7
    assert np.all(props.k_mult[props.k_mult > 0] == 1.0)
    grid3d = base.grid.model_copy(update={"nphi": 12})
    assert [loc for loc, _ in config_warnings(base.model_copy(update={"grid": grid3d}))] == [
        ("grid", "nphi")
    ]
