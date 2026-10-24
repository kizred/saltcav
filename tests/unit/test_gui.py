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
