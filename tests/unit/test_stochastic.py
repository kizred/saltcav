"""Этап 5: случайные поля, сэмплинг параметров, ансамбль."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.core.config import RunConfig, load_config
from saltcav.core.rng import realization_rng
from saltcav.leaching import simulate
from saltcav.stochastic.ensemble import EnsembleResult, run_ensemble
from saltcav.stochastic.fields import (
    GaussianField,
    azimuth_correlation,
    bounded_fraction,
    lognormal_multiplier,
)
from saltcav.stochastic.sampling import RealizationSampler, design, parameter_names

ROOT = Path(__file__).resolve().parents[2]
ENS = ROOT / "configs" / "example_ensemble.yaml"


# --- поля -----------------------------------------------------------------------


def test_field_statistics() -> None:
    z = np.linspace(0.0, 50.0, 51)
    phi = (np.arange(12) + 0.5) * 2 * np.pi / 12
    f = GaussianField(z, phi, 5.0, np.deg2rad(60.0))
    rng = np.random.default_rng(3)
    s = np.array([f.sample(rng) for _ in range(4000)])
    assert s.mean() == pytest.approx(0.0, abs=0.03)
    assert s.var() == pytest.approx(1.0, abs=0.05)
    assert np.corrcoef(s[:, 20, 0], s[:, 25, 0])[0, 1] == pytest.approx(np.exp(-1), abs=0.05)
    assert np.corrcoef(s[:, 20, 0], s[:, 20, 2])[0, 1] == pytest.approx(np.exp(-1), abs=0.05)


def test_azimuth_correlation_is_periodic() -> None:
    phi = (np.arange(8) + 0.5) * 2 * np.pi / 8
    c = azimuth_correlation(phi, np.deg2rad(45.0))
    assert c[0, 1] == pytest.approx(c[0, 7])  # соседи через шов 0/2π равноправны
    assert np.all(np.linalg.eigvalsh(c) > 0)


def test_zero_length_means_independent() -> None:
    f = GaussianField(np.arange(5.0), np.array([0.0]), 0.0, 0.0)
    s = np.array([f.sample(np.random.default_rng(i)) for i in range(2000)])[:, :, 0]
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.06


def test_physical_transforms() -> None:
    g = np.random.default_rng(0).standard_normal(200_000)
    assert lognormal_multiplier(g, 0.4).mean() == pytest.approx(1.0, abs=0.01)
    f = bounded_fraction(g, 0.05, 0.1)
    assert f.min() == 0.0 and f.max() <= 0.6


# --- сэмплинг параметров ----------------------------------------------------------


def _cfg(sampler: str = "lhs", days: float = 20, **unc: object) -> RunConfig:
    raw = load_config(ENS).model_dump(mode="json")
    raw["ensemble"]["sampler"] = sampler
    for st in raw["schedule"]["stages"]:
        st["duration_days"] = days
    if unc:
        raw["uncertainty"] = dict(raw["uncertainty"], **unc)
    return RunConfig.model_validate(raw)


@pytest.mark.parametrize("sampler", ["mc", "lhs", "sobol"])
def test_design(sampler: str) -> None:
    cfg = _cfg(sampler)
    u = design(cfg, 64, 5)
    assert u.shape == (64, len(parameter_names(cfg)))
    assert u.min() > 0 and u.max() < 1
    np.testing.assert_array_equal(u, design(cfg, 64, 5))  # воспроизводимо
    if sampler == "lhs":  # латинский гиперкуб: по одной точке в каждой из 64 полос
        assert all(len(set((u[:, j] * 64).astype(int))) == 64 for j in range(u.shape[1]))


def _no_uncertainty() -> RunConfig:
    raw = _cfg().model_dump(mode="json")
    raw["uncertainty"] = {}
    raw["geology"]["insoluble_fraction_sd"] = 0.0
    for lay in raw["geology"]["layers"]:
        lay.update(top_sd_m=0.0, thickness_sd_m=0.0)
    return RunConfig.model_validate(raw)


def test_zero_uncertainty_reproduces_base() -> None:
    cfg = _no_uncertainty()
    real = RealizationSampler(cfg).realize(design(cfg, 1, 0)[0], realization_rng(0, 0))
    assert real.cfg == cfg
    base = simulate(cfg)
    np.testing.assert_array_equal(simulate(real.cfg, real.props).radius[-1], base.radius[-1])


def test_interlayers_stay_insoluble_under_fields() -> None:
    cfg = _cfg()
    real = RealizationSampler(cfg).realize(design(cfg, 1, 1)[0], realization_rng(1, 0))
    rows_zero = np.all(real.props.k_mult == 0.0, axis=1)
    assert rows_zero.sum() >= 2  # узлы двух прослоев
    assert np.all(real.props.k_mult[~rows_zero] > 0)


# --- ансамбль --------------------------------------------------------------------


@pytest.fixture(scope="module")
def ens() -> EnsembleResult:
    return run_ensemble(_cfg(), n=8, seed=11, jobs=1)


def test_parallel_equals_serial(ens: EnsembleResult) -> None:
    par = run_ensemble(_cfg(), n=8, seed=11, jobs=2)
    np.testing.assert_array_equal(par.radius, ens.radius)
    np.testing.assert_array_equal(par.params, ens.params)
    np.testing.assert_array_equal(par.volume_t, ens.volume_t)


def test_quantiles_ordered_and_probabilities(ens: EnsembleResult) -> None:
    q = ens.radius_quantiles()
    assert np.all(q[0] <= q[1] + 1e-12) and np.all(q[1] <= q[2] + 1e-12)
    vol = ens.output("volume")
    assert ens.prob("volume", above=float(np.median(vol))) <= 0.5
    p = ens.prob_radius_exceeds(5.0)
    assert p.shape == ens.radius.shape[1:] and p.min() >= 0 and p.max() <= 1
    assert ens.convergence("volume").shape == (8, 3)


def test_save_load_roundtrip(ens: EnsembleResult, tmp_path: Path) -> None:
    path = tmp_path / "e.npz"
    ens.save(path)
    back = EnsembleResult.load(path)
    np.testing.assert_array_equal(back.radius, ens.radius)
    assert back.param_names == ens.param_names and back.seed == ens.seed


def test_sensitivity_finds_driver() -> None:
    # единственная неопределённость — k0: объём монотонен по k0, Спирмен ≈ 1
    raw = _no_uncertainty().model_dump(mode="json")
    raw["uncertainty"]["k0_log_sd"] = 0.3
    e = run_ensemble(RunConfig.model_validate(raw), n=10, seed=2)
    s = e.sensitivity(("volume",))
    assert s["volume"]["k0"] > 0.95
    assert set(s["volume"]) == {"k0"}  # параметры без разброса не участвуют


def test_cli_ensemble(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from saltcav.cli import main

    out = tmp_path / "e.npz"
    assert main(["ensemble", str(ENS), "-n", "4", "--radius-limit", "10", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "P10 / P50 / P90" in text and "P(макс. радиус > 10 м)" in text
    assert EnsembleResult.load(out).n == 4


def test_envelope_plot(ens: EnsembleResult, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from saltcav.viz.envelope import plot_envelope

    out = tmp_path / "env.png"
    plot_envelope(ens, out, layers=[(1040.0, 1.0)])
    assert out.stat().st_size > 1000
