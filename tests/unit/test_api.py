"""Python API: конфигурация, расчёты, таблицы, картинки, сохранение."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import saltcav.api as sc
from saltcav.surrogate import study

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg() -> Any:
    c = sc.load(ROOT / "configs" / "example_ensemble.yaml")
    return sc.set(c, {"grid.nz": 60, "grid.nphi": 4, "leaching.output_every_days": 20})


def test_get_set(cfg: Any) -> None:
    assert sc.get(cfg, "leaching.k0_m_per_s") == pytest.approx(3e-5)
    c2 = sc.set(cfg, {"schedule.stages.1.rate_m3_per_h": 80}, leaching__roof_ratio=2.0)
    assert sc.get(c2, "schedule.stages.1.rate_m3_per_h") == 80
    assert c2.leaching.roof_ratio == 2.0 and cfg.leaching.roof_ratio == 1.5  # исходная цела
    assert sc.get(c2, "geology.layers.0.kind") == "anhydrite"
    with pytest.raises(KeyError, match="нет параметра"):
        sc.set(cfg, {"leaching.k0": 1})
    with pytest.raises(KeyError):
        sc.get(cfg, "schedule.stages.9.rate_m3_per_h")
    with pytest.raises(ValueError):  # проверка как у YAML
        sc.set(cfg, {"leaching.k0_m_per_s": -1})


def test_run_table_plot_save(cfg: Any, tmp_path: Path) -> None:
    res = sc.run(cfg)
    t = sc.table(res)
    assert len(t) == res.nt and t["объём, м³"][-1] == pytest.approx(res.volume[-1])
    assert "объём" in str(t)
    t.to_csv(tmp_path / "t.csv", excel=True)
    assert ";" in (tmp_path / "t.csv").read_text(encoding="utf-8-sig")
    assert sc.plot(res, tmp_path / "r.png").stat().st_size > 20_000
    assert type(sc.plot(res)).__name__ == "Figure"
    back = sc.load(sc.save(res, tmp_path / "r.npz"))
    assert np.allclose(back.volume, res.volume)
    assert len(sc.table(cfg)) > 50
    vols = [r.volume[-1] for _, r in sc.sweep(cfg, "leaching.k0_m_per_s", [1.5e-5, 3e-5])]
    assert vols[0] < vols[1]
    real = sc.run(cfg, realization=1, n=4)
    assert real.name.endswith("#1")


def test_ensemble_and_stability(cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ens = sc.ensemble(cfg, n=6, jobs=1)
    t = sc.table(ens)
    assert t["выход"][0] == "объём, м³" and t["P10"][0] <= t["P90"][0]
    sc.plot(ens, tmp_path / "e.png")
    assert type(sc.load(sc.save(ens, tmp_path / "e.npz"))).__name__ == "EnsembleResult"

    def fake(mc: Any, z: Any, radius: Any, layers: Any = None) -> Any:
        return np.array([1.2, 1.1, 3.0, 0.5])

    monkeypatch.setattr(study, "mechanics_targets", fake)
    c = sc.set(cfg, {"surrogate.n_train": 5, "surrogate.adaptive": 0})
    st = sc.stability(c, ens, jobs=1)
    assert any("P(запас" in str(r) for r in sc.table(st)["показатель"])
    assert type(sc.load(sc.save(st, tmp_path / "s.npz"))).__name__ == "StudyResult"


def test_mechanics_and_calibration(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    from tests.verification.test_calibration import small_config

    c = small_config(("k0",), n_predictive=2)
    run = sc.mechanics(c, years=1.0)
    assert run.summary.years[-1] == pytest.approx(1.0, abs=0.01)
    assert "FOS мин" in sc.table(run).columns
    assert type(sc.load(sc.save(run, tmp_path / "m.npz"))).__name__ == "MechanicsRun"
    data = sc.data(ROOT / "data" / "open" / "sansmic-direct")
    assert data.brine_day.size > 10
    from saltcav.calibration import synthesize
    from saltcav.io.fielddata import write_field_data

    write_field_data(tmp_path / "w", synthesize(c, {"k0": 2e-5}, sonar_days=[40], seed=1))
    cal = sc.calibrate(c, tmp_path / "w", jobs=1)
    assert cal.estimate()["k0"] == pytest.approx(2e-5, rel=0.05)
    assert sc.table(cal).columns[0] == "параметр"
    assert type(sc.load(sc.save(cal, tmp_path / "c.npz"))).__name__ == "CalibResult"
    with pytest.raises(ValueError):
        sc.load(tmp_path / "x.txt")
