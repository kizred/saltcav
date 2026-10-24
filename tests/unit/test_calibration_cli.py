"""Команды saltcav data / synth / calibrate."""

from pathlib import Path

import pytest

from saltcav.cli import main
from saltcav.core.config import dump_config, load_config
from tests.verification.test_calibration import small_config


@pytest.fixture(scope="module")
def cfg_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("cfg") / "small.yaml"
    dump_config(small_config(n_predictive=2), p)
    return p


def test_synth_data_calibrate(
    cfg_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    well = tmp_path / "well"
    assert (
        main(
            [
                "synth",
                str(cfg_path),
                "--out",
                str(well),
                "--truth",
                "k0=2e-5",
                "--sonar-days",
                "20,40",
                "--seed",
                "1",
                "--excel",
            ]
        )
        == 0
    )
    assert (well / "truth.yaml").exists()
    assert main(["data", str(well), "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "сонар 20 сут" in out and "OK" in out
    new = tmp_path / "cal.yaml"
    assert (
        main(
            [
                "calibrate",
                str(cfg_path),
                "--data",
                str(well),
                "--jobs",
                "1",
                "--holdout-days",
                "30",
                "--out",
                str(tmp_path / "c.npz"),
                "--write-config",
                str(new),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "истина" in out and "проверка после 30 сут" in out
    cal = load_config(new)
    assert cal.leaching.k0_m_per_s == pytest.approx(2e-5, rel=0.05)
    assert cal.name.endswith("-calibrated")


def test_errors(cfg_path: Path, tmp_path: Path) -> None:
    assert main(["data", str(tmp_path)]) == 1  # пустая папка
    assert main(["synth", str(cfg_path), "--out", str(tmp_path / "x"), "--truth", "nope=1"]) == 1
    assert main(["synth", "--out", str(tmp_path / "y")]) == 1
    assert main(["synth", "--sansmic", "--out", str(tmp_path / "s")]) == 0
    assert (tmp_path / "s" / "sonar.csv").exists()
