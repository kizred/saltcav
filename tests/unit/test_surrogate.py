"""Этап 7: признаки формы, план выборки, гауссов процесс, вероятность потери устойчивости."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from saltcav.core.config import load_config
from saltcav.surrogate import study
from saltcav.surrogate.features import FEATURES, shape_features
from saltcav.surrogate.model import GPModel, cross_validate

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[2]
Z = np.linspace(1000.0, 1040.0, 41)


def test_features_of_cylinder_and_ledge() -> None:
    r = np.full((41, 1), 0.1)
    r[5:36] = 10.0
    f = dict(zip(FEATURES, shape_features(Z, r), strict=True))
    assert f["height"] == pytest.approx(30.0)
    assert f["r_max"] == pytest.approx(10.0)
    assert f["h_over_d"] == pytest.approx(1.5)
    assert f["notch"] == 0 and f["asym"] == 0 and f["offset"] == 0
    assert f["volume"] == pytest.approx(np.pi * 100 * 30 / 1e3, rel=0.05)
    r2 = r.copy()
    r2[20] = 4.0  # ниша — карниз
    assert dict(zip(FEATURES, shape_features(Z, r2), strict=True))["notch"] == pytest.approx(0.6)
    r3 = np.repeat(r, 4, axis=1)
    r3[5:36, 0] = 14.0  # асимметрия по азимуту
    f3 = dict(zip(FEATURES, shape_features(Z, r3), strict=True))
    assert f3["asym"] > 0.3 and f3["offset"] > 1.0


def test_select_design_extremes_and_spread() -> None:
    x = np.random.default_rng(1).random((200, 3))
    idx = study.select_design(x, 20)
    assert idx.size == 20 and np.unique(idx).size == 20
    for j in range(3):
        assert np.argmin(x[:, j]) in idx and np.argmax(x[:, j]) in idx
    more = study.select_design(x, 5, exclude=idx)
    assert not set(more) & set(idx)


def test_gp_learns_smooth_function() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((60, 3))
    y = 1.0 + x[:, 0] ** 2 + 0.5 * x[:, 1]  # третий признак не влияет
    cv = cross_validate(x, y)
    assert cv["r2"] > 0.97 and cv["coverage90"] > 0.7
    m = GPModel().fit(x, y)
    rel = m.importance(x)
    assert rel[2] < 0.1 * min(rel[0], rel[1])
    pos = GPModel(log=True).fit(x, np.exp(y))
    assert np.all(pos.predict(x)[0] > 0)


def test_mechanics_parameters() -> None:
    cfg = load_config(ROOT / "configs" / "example_ensemble.yaml")
    p = study.mechanics_parameters(cfg, 4000, seed=3)
    assert np.mean(np.exp(p[:, 0])) == pytest.approx(1.0, abs=0.03)
    assert np.std(p[:, 1]) == pytest.approx(50, rel=0.05)
    np.testing.assert_array_equal(p, study.mechanics_parameters(cfg, 4000, seed=3))
    mc = study.with_parameters(cfg.mechanics, np.array([np.log(2.0), 2400.0]))
    assert mc.creep[0].a_per_day == pytest.approx(2 * cfg.mechanics.creep[0].a_per_day)
    assert mc.overburden_density_kg_m3 == 2400.0


def test_study_finds_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пайплайн на искусственной «механике»: запас — гладкая функция признаков;
    уточнение у границы даёт вероятность, близкую к точной."""

    def fake(mc: Any, z: np.ndarray, radius: np.ndarray, layers: Any = None) -> np.ndarray:
        f = dict(zip(FEATURES, shape_features(z, radius), strict=True))
        log_a = np.log(mc.creep[0].a_per_day / 0.18)
        fos = 1.6 - 0.04 * f["r_max"] - 0.4 * log_a
        return np.array([fos, fos - 0.1, 2.0 + log_a, 0.0])

    monkeypatch.setattr(study, "mechanics_targets", fake)
    cfg = load_config(ROOT / "configs" / "example_ensemble.yaml")
    sc = cfg.surrogate.model_copy(update={"n_train": 15, "adaptive": 15, "u_stop": 2.0})
    cfg = cfg.model_copy(update={"surrogate": sc})
    rng = np.random.default_rng(5)
    radii = np.zeros((150, 41, 4))
    for i in range(150):
        radii[i, 5:36] = rng.uniform(8, 20) * rng.uniform(0.9, 1.1, 4)
    res = study.run_study(cfg, Z, radii, seed=7, jobs=1)
    params = study.mechanics_parameters(cfg, 150, 7)
    truth = np.array([fake(study.with_parameters(cfg.mechanics, params[i]), Z, radii[i])[0]
                      for i in range(150)])  # fmt: skip
    p_true = float(np.mean(truth < 1.0))
    assert 0.05 < p_true < 0.95
    assert res.prob_below()[0] == pytest.approx(p_true, abs=0.04)
    assert res.train.size <= 30 and len(res.history) >= 2
    assert res.cv["fos_p5_min"]["r2"] > 0.9


def test_cli_surrogate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("gmsh")
    from saltcav.cli import main
    from saltcav.surrogate.study import StudyResult

    raw = yaml.safe_load((ROOT / "configs" / "example_ensemble.yaml").read_text("utf-8"))
    raw["grid"]["nphi"] = 4
    raw["mechanics"]["h_wall_m"] = 3.0
    for st in raw["mechanics"]["stages"]:
        st["duration_years"] = 1
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    out, png = tmp_path / "s.npz", tmp_path / "s.png"
    args = ["surrogate", str(path), "-n", "10", "--train", "6", "--adaptive", "2", "--jobs", "1"]
    assert main([*args, "--out", str(out), "--plot", str(png)]) == 0
    text = capsys.readouterr().out
    assert "P(запас по дилатансии <" in text
    res = StudyResult.load(out)
    assert res.n == 10 and 6 <= res.train.size <= 8
    assert np.all(np.isfinite(res.mu)) and png.stat().st_size > 10_000
