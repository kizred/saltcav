"""Окно saltcav: формы по модели, проверка, файлы, расчёты в фоне.

Без дисплея окно работает на платформе Qt «offscreen»; 3D-просмотр там недоступен
(нет OpenGL), поэтому вкладка выщелачивания подменяется сводкой.
"""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qtw = pytest.importorskip("PySide6.QtWidgets")

from saltcav.core.config import RunConfig, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.fixture(scope="module")
def app() -> Any:
    return qtw.QApplication.instance() or qtw.QApplication([])


@pytest.fixture
def win(app: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from saltcav.gui.app import MainWindow

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        import saltcav.viz.viewer as viewer

        def no_gl(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("нет OpenGL")

        monkeypatch.setattr(viewer, "build_window", no_gl)
    w = MainWindow()
    yield w
    w.dirty = False
    w.close()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_form_roundtrip_matches_config(win: Any, path: Path) -> None:
    """YAML → форма → словарь → модель: та же конфигурация, что и из файла."""
    win.open_path(path)
    assert win.cfg is not None, [win.errors.item(i).text() for i in range(win.errors.count())]
    assert RunConfig.model_validate(win.form_dict()) == load_config(path)
    assert not win.dirty


def test_errors_are_highlighted_in_russian(win: Any) -> None:
    win.open_path(ROOT / "configs" / "example_mechanics.yaml")
    nz = win.forms["grid"].scalars["nz"]
    nz.widget.setText("1,5")
    assert win.validate() is None
    texts = [win.errors.item(i).text() for i in range(win.errors.count())]
    assert texts == ["grid.nz: нужно целое число"]
    assert "fbe3e1" in nz.widget.styleSheet()
    nz.widget.setText("200")
    stages = win.forms["schedule"].tables["stages"]
    stages.table.item(0, 3).setText("2000")  # башмак ниже сетки — проверка модели целиком
    assert win.validate() is None
    assert "ступень 0" in win.errors.item(0).text()
    assert stages.table.item(0, 0).background().color().name() == "#fbe3e1"
    stages.table.item(0, 3).setText("1090")
    k0 = win.forms["leaching"].scalars["k0_m_per_s"]
    k0.widget.setText("3e-5")
    win.forms["mechanics"].scalars["poisson"].widget.setText("0,3")  # запятая — разделитель
    cfg = win.validate()
    assert cfg is not None and cfg.mechanics.poisson == pytest.approx(0.3)
    assert win.dirty


def test_table_rows_and_optional_group(win: Any) -> None:
    win.open_path(ROOT / "configs" / "example_ledges.yaml")
    layers = win.forms["geology"].tables["layers"]
    layers.table.setCurrentCell(0, 0)
    layers.copy_row()
    assert win.validate().geology.layers[1].top_m == 1040.0
    layers.remove_row()
    assert len(win.validate().geology.layers) == 2
    mech = win.forms["mechanics"]
    mech.optional_groups["transient"].setChecked(False)
    assert win.validate().mechanics.transient is None


def test_save_and_reopen(win: Any, tmp_path: Path) -> None:
    win.open_path(ROOT / "configs" / "example_3d.yaml")
    win.forms["grid"].scalars["nphi"].widget.setText("12")
    out = tmp_path / "case.yaml"
    win._write(out)
    text = out.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    assert raw["grid"]["nphi"] == 12
    assert "leaching" in raw and "cfl" not in raw["leaching"]  # умолчания не пишутся
    cfg = load_config(out)
    win.open_path(out)
    assert win.cfg == cfg and not win.dirty
    assert win.windowTitle().endswith("case.yaml")


def test_runs_in_background(win: Any, tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    raw = yaml.safe_load((ROOT / "configs" / "example_mechanics.yaml").read_text("utf-8"))
    raw["mechanics"].update({"h_wall_m": 3.0})
    for st in raw["mechanics"]["stages"]:
        st["duration_years"] = 1
    path = tmp_path / "fast.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    win.open_path(path)

    win.run_leaching()
    win.wait()
    assert win.results is not None and win.results.volume[-1] > 1e4
    win.run_mechanics()  # форма уже посчитана по этим параметрам — без повтора
    win.wait()
    assert win.mech is not None and win.mech.summary.convergence_pct[-1] > 0
    assert win.tabs.currentIndex() == 3
    view = win.tabs.widget(3)
    view.slider.setValue(0)
    view.left.setCurrentIndex(view.left.findData("creep"))
    assert np.all(view.fields_at("creep") == 0)  # в момент выемки ползучести ещё нет

    win.run_ensemble(n=4, jobs=1)
    win.wait()
    assert win.ensemble is not None and win.ensemble.n == 4
    win.open_realization(2, mech=False)
    win.wait()
    assert win.results.name.endswith("#2")


def test_cancel(win: Any) -> None:
    win.open_path(ROOT / "configs" / "example_ensemble.yaml")
    win.run_ensemble(n=50, jobs=1)
    win.cancel()
    win.wait()
    assert win.ensemble is None
    assert "отменено" in win.statusBar().currentMessage()


def test_warnings_shown_but_not_blocking(win: Any) -> None:
    win.open_path(ROOT / "configs" / "example_ledges.yaml")
    win.forms["geology"].scalars["k_variation"].widget.setText("0.3")
    cfg = win.validate()
    assert cfg is not None and win.run_act.isEnabled()  # предупреждение не блокирует
    items = [win.errors.item(i).text() for i in range(win.errors.count())]
    assert len(items) == 1 and items[0].startswith("⚠ geology.k_variation")
    assert "fdf1d6" in win.forms["geology"].scalars["k_variation"].widget.styleSheet()


def test_help_dialog_and_f1(win: Any) -> None:
    from saltcav.gui.help import guide_markdown, parameters_html

    assert "Рабочий цикл" in guide_markdown()
    ref = parameters_html()
    for path in ("grid.nphi", "mechanics.stages", "geology.layers.kind"):
        assert f"name='p-{path.replace('.', '-')}'" in ref, path
    win.section_tabs.setCurrentIndex(0)
    nz = win.forms["grid"].scalars["nz"].widget
    win.help_for_focus(nz)
    dlg = win._help
    assert dlg.isVisible() and dlg.tabs.currentIndex() == 1
    dlg.close()


def test_mechanics_view_3d(app: Any, tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("pyvista")
    from saltcav.core.config import MechanicsConfig, PressureStageConfig
    from saltcav.gui.mech_view import MechanicsView
    from saltcav.mechanics.run import MechanicsRun, run_from_shape

    z = np.linspace(1000.0, 1030.0, 31)
    r = np.c_[8.0, 12.0, 8.0, 12.0] * np.sin(np.pi * (z[:, None] - 1000.0) / 30.0) ** 0.5 + 0.1
    mc = MechanicsConfig(
        model="3d", h_wall_3d_m=3.0, transient=None, r_factor=5.0, height_factor=2.0,
        stages=[PressureStageConfig(duration_years=0.2)], output_every_years=0.1,
    )  # fmt: skip
    run = run_from_shape(mc, z, r, "асимметрия")
    assert run.dim == 3 and run.wall_elements is not None and run.wall_elements.size > 0
    path = tmp_path / "m3.npz"
    run.save(path)
    back = MechanicsRun.load(path)
    view = MechanicsView(back)
    assert view.azimuth.isVisibleTo(view) or view.azimuth.parent() is not None
    view.azimuth.setValue(45)
    view.slider.setValue(0)
    view.left.setCurrentIndex(view.left.findData("vm"))
    png = tmp_path / "v.png"
    view.save_png(str(png))
    assert png.stat().st_size > 10_000


def test_surrogate_tab(win: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sklearn")
    from saltcav.surrogate import study

    def fake(mc: Any, z: np.ndarray, radius: np.ndarray) -> np.ndarray:
        rmax = float(np.max(radius))
        return np.array([1.6 - 0.03 * rmax, 1.5 - 0.03 * rmax, 3.0, 0.1])

    monkeypatch.setattr(study, "mechanics_targets", fake)
    cfg = load_config(ROOT / "configs" / "example_ensemble.yaml")
    rng = np.random.default_rng(2)
    z = np.linspace(1000.0, 1040.0, 41)
    radii = np.zeros((40, 41, 4))
    for i in range(40):
        radii[i, 5:36] = rng.uniform(10, 30)
    res = study.run_study(cfg, z, radii, seed=1, jobs=1)
    win._show_surrogate(res)
    assert win.tabs.tabText(4) == "Устойчивость" and win.surrogate is res
    path = tmp_path / "stab.npz"
    res.save(path)
    win.surrogate = None
    win.open_path(path)
    assert win.surrogate is not None and win.surrogate.n == 40
    assert win.tabs.currentIndex() == 4


def test_calibration_tab(win: Any, tmp_path: Path) -> None:
    from saltcav.calibration import synthesize, write_truth
    from saltcav.core.config import dump_config
    from saltcav.gui.app import CALIB_TAB
    from saltcav.io.fielddata import write_field_data
    from tests.verification.test_calibration import small_config

    cfg = small_config(("k0",), n_predictive=2)
    dump_config(cfg, tmp_path / "c.yaml")
    write_field_data(tmp_path / "well", synthesize(cfg, {"k0": 2e-5}, sonar_days=[40], seed=2))
    write_truth(tmp_path / "well", {"k0": 2e-5})
    win.open_path(tmp_path / "c.yaml")
    assert win.cfg is not None
    win.load_data(str(tmp_path / "well"))
    assert "рассол" in win.data_info.text()
    win.run_calibration(jobs=1)
    win.wait()
    assert win.calib is not None and win.tabs.currentIndex() == CALIB_TAB
    assert win.calib.estimate()["k0"] == pytest.approx(2e-5, rel=0.05)
    win.apply_calibration()
    assert win.cfg.leaching.k0_m_per_s == pytest.approx(win.calib.estimate()["k0"])
    assert win.dirty
    win.calib.save(tmp_path / "cal.npz")
    win.open_path(tmp_path / "cal.npz")
    assert win.tabs.currentIndex() == CALIB_TAB
