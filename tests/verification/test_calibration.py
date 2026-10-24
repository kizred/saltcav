"""Калибровка (этап 8). Приёмка: на синтетических данных со скрытыми параметрами
калибровка их находит; на открытом эталоне SANSMIC — k0 того же порядка, что и
подобранный ранее по объёму, с невязками в пределах погрешностей."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.calibration import CalibResult, calibrate, read_truth, synthesize, write_truth
from saltcav.core.config import CalibParamConfig, RunConfig, load_config
from saltcav.io.fielddata import load_field_data, write_field_data

ROOT = Path(__file__).resolve().parents[2]
TRUTH = {"k0": 2.0e-5, "roof_ratio": 2.2, "insoluble_fraction": 0.1}


def small_config(
    params: tuple[str, ...] = ("k0", "insoluble_fraction"), **cal: object
) -> RunConfig:
    cfg = load_config(ROOT / "configs" / "benchmark_sansmic_direct.yaml")
    st = cfg.schedule.stages[0].model_copy(update={"duration_days": 40})
    cc = cfg.calibration.model_copy(
        update={"parameters": [CalibParamConfig(name=n) for n in params], "n_predictive": 6, **cal}
    )
    return cfg.model_copy(
        update={
            "grid": cfg.grid.model_copy(update={"nz": 40}),
            "schedule": cfg.schedule.model_copy(update={"stages": [st]}),
            "calibration": cc,
        }
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunConfig, Path]:
    cfg = small_config()
    folder = tmp_path_factory.mktemp("well")
    data = synthesize(
        cfg, TRUTH, brine_every_days=2, sonar_days=[20, 40], sigma_c=3, sigma_r=0.3, seed=3
    )
    write_field_data(folder, data, excel=True)
    write_truth(folder, TRUTH)
    return cfg, folder


@pytest.fixture(scope="module")
def fitted(synthetic: tuple[RunConfig, Path]) -> CalibResult:
    cfg, folder = synthetic
    return calibrate(cfg, load_field_data(folder), holdout_day=30, truth=read_truth(folder))


def test_recovers_hidden_parameters(fitted: CalibResult) -> None:
    est = fitted.estimate()
    assert est["k0"] == pytest.approx(TRUTH["k0"], rel=0.05)
    lo, hi = fitted.quantiles((0.05, 0.95))[:, fitted.names.index("insoluble_fraction")]
    assert lo < TRUTH["insoluble_fraction"] < hi  # 40 сут: подъём насыпи мал, интервал широкий
    q = fitted.quantiles((0.001, 0.999))  # истина внутри широкого интервала
    for j, n in enumerate(fitted.names):
        tol = 0.02 * TRUTH[n]  # погрешность самой модели (шаг по времени, сетка)
        assert q[0, j] - tol <= TRUTH[n] <= q[1, j] + tol, n
    assert 0.3 < fitted.chi2_reduced < 2.0
    assert fitted.rms_fit["brine"] < 5 and fitted.rms_fit["sonar"] < 0.5


def test_holdout_and_band(fitted: CalibResult) -> None:
    assert fitted.brine_fit.sum() < fitted.brine_fit.size  # часть замеров — проверка
    assert fitted.rms_holdout["brine"] < 6
    assert fitted.coverage_holdout >= 0.8
    assert np.all(fitted.curve_lo <= fitted.curve_hi + 1e-9)


def test_calibrated_config_and_save(
    fitted: CalibResult, synthetic: tuple[RunConfig, Path], tmp_path: Path
) -> None:
    cfg, folder = synthetic
    new = fitted.calibrated_config(cfg, load_field_data(folder))
    assert new.leaching.k0_m_per_s == pytest.approx(fitted.estimate()["k0"])
    assert 0 < new.uncertainty.k0_log_sd < 0.2
    fitted.save(tmp_path / "c.npz")
    back = CalibResult.load(tmp_path / "c.npz")
    assert back.names == fitted.names and back.truth == TRUTH
    assert np.allclose(back.samples, fitted.samples)
    assert len(back.sonar_obs) == 2
    from saltcav.viz.calibration_plot import plot_calibration

    plot_calibration(back, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 10_000


def test_unidentified_parameter_is_flagged(synthetic: tuple[RunConfig, Path]) -> None:
    """Прямоток с нерастворителем: кровля не растёт — roof_ratio данные не ограничивают."""
    cfg, folder = synthetic
    cfg = small_config(("k0", "roof_ratio"))
    res = calibrate(cfg, load_field_data(folder))
    assert any("почти не ограничивают" in n for n in res.notes())


def test_mcmc_runs(synthetic: tuple[RunConfig, Path]) -> None:
    cfg, folder = synthetic
    cfg = small_config(("k0",), walkers=4, steps=12, burn=4, n_predictive=3)
    res = calibrate(cfg, load_field_data(folder), method="mcmc", jobs=2)
    assert res.method == "mcmc" and res.samples.shape == (4 * 8, 1)
    assert 0 < res.acceptance <= 1
    assert res.estimate()["k0"] == pytest.approx(TRUTH["k0"], rel=0.05)


def test_sansmic_open_dataset() -> None:
    """Открытый эталон: k0 из данных рассола и сонара SANSMIC при априорном 1,5e-5."""
    cfg = load_config(ROOT / "configs" / "benchmark_sansmic_direct.yaml")
    cc = cfg.calibration.model_copy(
        update={"parameters": [CalibParamConfig(name="k0")], "n_predictive": 0}
    )
    cfg = cfg.model_copy(
        update={
            "calibration": cc,
            "leaching": cfg.leaching.model_copy(update={"k0_m_per_s": 1.5e-5}),
        }
    )
    res = calibrate(cfg, load_field_data(ROOT / "data" / "open" / "sansmic-direct"))
    k0 = res.estimate()["k0"]
    assert 1.5e-5 < k0 < 4e-5  # порядок SANSMIC-подбора по объёму (3e-5)
    assert res.rms_fit["brine"] < 3 and res.rms_fit["sonar"] < 0.6
