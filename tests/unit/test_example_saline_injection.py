"""Пример закачки рассола соответствует явно заданной конфигурации."""

from pathlib import Path

import numpy as np

from saltcav.core.config import LeachingConfig, RunConfig, load_config
from saltcav.leaching import simulate

ROOT = Path(__file__).resolve().parents[2]


def test_example_saline_injection_matches_explicit_configuration() -> None:
    linked = load_config(ROOT / "configs" / "example_saline_injection.yaml")
    explicit = RunConfig.model_validate({
        "name": "example-saline-injection",
        "grid": {"z_top_m": 900, "z_bottom_m": 1100, "nz": 40, "nphi": 1,
                 "r_borehole_m": 0.15},
        "leaching": {"injection_concentration_kg_m3": 120.0, "output_every_days": 5},
        "schedule": {"stages": [{"duration_days": 10, "mode": "direct",
                                 "rate_m3_per_h": 50, "injection_depth_m": 1090,
                                 "production_depth_m": 1080, "blanket_depth_m": 1060}]},
    })  # fmt: skip
    assert linked.leaching.injection_concentration_kg_m3 == 120.0
    assert explicit.leaching.injection_concentration_kg_m3 != (
        LeachingConfig().injection_concentration_kg_m3
    )
    actual = simulate(linked)
    expected = simulate(explicit)
    for name in ("radius", "volume", "concentration", "dissolved_salt", "produced_salt"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), err_msg=name)
