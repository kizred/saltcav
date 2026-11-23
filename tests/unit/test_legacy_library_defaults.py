import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from saltcav.core import brine
from saltcav.materials import load_library

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "data" / "materials_library"


def test_legacy_brine_values_preserve_float_rounding() -> None:
    np.testing.assert_array_equal(
        brine.saturation_mass_fraction(brine._T_SOL),
        np.array(
            [26.28, 26.32, 26.41, 26.45, 26.52, 26.67, 26.84, 27.03, 27.25, 27.50, 27.78, 28.05]
        )
        / 100.0,
    )


def test_saturated_viscosity_is_explicitly_extrapolated() -> None:
    prop = load_library(LIBRARY).fluids["nacl-saturated-20c-nist"].properties["dynamic_viscosity"]
    assert prop.unit == "Pa*s"
    assert prop.value == (1.990 + (1.990 - 1.821) * 0.4 / 2) / 1000
    assert prop.range == (prop.value, prop.value)
    assert "method: extrapolation" in prop.source.location
    assert "24 mass% = 1.821 mPa s" in prop.source.location
    assert "26 mass% = 1.990 mPa s" in prop.source.location
    assert "0.0012 mPa s" in prop.source.location
    assert "errors are unknown" in prop.source.location


def test_calculation_defaults_read_bundled_library_and_preserve_overrides(tmp_path: Path) -> None:
    library = tmp_path / "library"
    shutil.copytree(LIBRARY, library)
    for file, changes in (
        ("materials/salt-default-1x.yaml", {"density": 2200.0, "creep_n": 6.0}),
        ("fluids/nacl-default-1x.yaml", {"water_density_20": 1000.0}),
        ("fluids/water-default-1x.yaml", {"mass_concentration": 2.0}),
    ):
        path = library / file
        entry = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, number in changes.items():
            entry["properties"][name].update(value=number, range=[number, number])
        path.write_text(yaml.safe_dump(entry, allow_unicode=True), encoding="utf-8")
    env = {**os.environ, "SALTCAV_TEST_LIBRARY": str(library)}
    code = """
import os
from pathlib import Path
import saltcav.workspace as workspace
workspace.bundled_materials_library = lambda: Path(os.environ['SALTCAV_TEST_LIBRARY'])
from saltcav.core.config import GeologyConfig, LeachingConfig, MechanicsConfig
from saltcav.core.brine import water_density
assert GeologyConfig().salt_density_kg_m3 == 2200
assert GeologyConfig(salt_density_kg_m3=2100).salt_density_kg_m3 == 2100
assert MechanicsConfig().creep[0].n == 6
assert LeachingConfig().injection_concentration_kg_m3 == 2
assert LeachingConfig(injection_concentration_kg_m3=3).injection_concentration_kg_m3 == 3
assert water_density(20) == 1000
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT)
