from pathlib import Path

import numpy as np
import pytest

from saltcav.cli import main
from saltcav.core import units
from saltcav.core.rng import realization_rng, realization_rngs

EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"


def test_ensemble_is_reproducible() -> None:
    a = [g.random(5) for g in realization_rngs(42, 4)]
    b = [g.random(5) for g in realization_rngs(42, 4)]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_single_realization_matches_ensemble() -> None:
    ens = realization_rngs(7, 10)
    for i in (0, 3, 9):
        np.testing.assert_array_equal(ens[i].random(3), realization_rng(7, i).random(3))


def test_realizations_differ() -> None:
    g0, g1 = realization_rngs(1, 2)
    assert not np.array_equal(g0.random(5), g1.random(5))


@pytest.mark.parametrize(
    ("fwd", "back", "x"),
    [
        (units.m3_per_hour_to_si, units.si_to_m3_per_hour, 123.4),
        (units.days_to_seconds, units.seconds_to_days, 365.0),
        (units.mpa_to_pa, units.pa_to_mpa, 21.5),
        (units.bbl_to_m3, units.m3_to_bbl, 1.0e6),
    ],
)
def test_units_roundtrip(fwd, back, x) -> None:  # type: ignore[no-untyped-def]
    assert back(fwd(x)) == pytest.approx(x)


def test_cli_check_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(EXAMPLE)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_check_missing_file() -> None:
    assert main(["check", "no_such_file.yaml"]) == 1


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_run_saves_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "hist.npz"
    assert main(["run", str(EXAMPLE), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "объём камеры" in text
    data = np.load(out)
    assert data["radius_m"].shape[1:] == (201, 1)
    assert data["volume_m3"][-1] > data["volume_m3"][0]
    assert data["floor_depth_m"][-1] < data["floor_depth_m"][0]  # насыпь подняла подошву


def test_cli_run_bad_config() -> None:
    assert main(["run", "no_such_file.yaml"]) == 1
