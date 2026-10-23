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
