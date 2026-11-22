"""Окно saltcav: формы по модели, проверка, файлы, расчёты в фоне.

Без дисплея окно работает на платформе Qt «offscreen»; 3D-просмотр там недоступен
(нет OpenGL), поэтому вкладка выщелачивания подменяется сводкой.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qtw = pytest.importorskip("PySide6.QtWidgets")

from saltcav.api import run  # noqa: E402
from saltcav.core.config import RunConfig, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))
LIBRARY = ROOT / "data" / "materials_library"
FLUID_ID = "test-synthetic-fluid"
ROCK_A = "test-sc014-anhydrite-a"
ROCK_B = "test-sc014-anhydrite-b"
FLUID = "test-sc014-fluid"


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


def _write_library_entry(
    root: Path,
    category: str,
    entry_id: str,
    kind: str,
    property_name: str,
    quantity: str,
    unit: str,
    value: float,
    bounds: list[float],
) -> None:
    raw = {
        "id": entry_id,
        "name": f"Synthetic SC-014 {entry_id}; test data only",
        "kind": kind,
        "properties": {
            property_name: {
                "quantity": quantity,
                "value": value,
                "unit": unit,
                "range": bounds,
                "source": {
                    "reference": "Synthetic SC-014 GUI test fixture; not a physical source",
                    "location": "Not applicable — arbitrary synthetic test value",
                },
            }
        },
    }
    path = root / category / f"{entry_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _linked_case(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    library = tmp_path / "library"
    _write_library_entry(
        library, "materials", ROCK_A, "anhydrite", "critical_span", "length", "m", 2.0,
        [0.1, 20.0],
    )  # fmt: skip
    _write_library_entry(
        library, "materials", ROCK_B, "anhydrite", "critical_span", "length", "m", 4.0,
        [0.1, 20.0],
    )  # fmt: skip
    _write_library_entry(
        library, "fluids", FLUID, "nacl_brine", "mass_concentration", "mass_concentration",
        "kg/m3", 120.0, [0.0, 320.0],
    )  # fmt: skip
    raw: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs" / "example_ledges.yaml").read_text("utf-8")
    )
    raw["materials_library"] = "../library"
    raw["geology"]["layers"] = [
        {"kind": "anhydrite", "top_m": 1030.0, "thickness_m": 1.25, "top_sd_m": 0.3},
        {"kind": "clay", "top_m": 1075.0, "thickness_m": 0.5, "thickness_sd_m": 0.1},
        {
            "kind": "anhydrite",
            "material_id": ROCK_A,
            "top_m": 1085.0,
            "thickness_m": 0.75,
            "critical_span_m": 7.0,
            "material_overrides": {"critical_span": 6.0},
        },
    ]
    path = tmp_path / "case" / "case.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path, raw


def _material_combo(win: Any, row: int = 0) -> Any:
    layers = win.forms["geology"].tables["layers"]
    column = next(j for j, (name, _) in enumerate(layers.fields) if name == "material_id")
    return layers.table.cellWidget(row, column)


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


def test_injection_fluid_choice_save_reopen_and_load(win: Any, tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["materials_library"] = str(LIBRARY)
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    win.open_path(source)

    combo = win.forms["leaching"].scalars["injection_fluid_id"].widget
    assert isinstance(combo, qtw.QComboBox)
    assert {combo.itemData(i) for i in range(combo.count())} == {
        "",
        FLUID_ID,
        "nacl-saturated-20c-nist",
        "water-default-1x",
    }
    assert combo.itemText(0) == "Без привязки к библиотеке"
    combo.setCurrentIndex(combo.findData(FLUID_ID))

    out = tmp_path / "gui-fluid.yaml"
    assert win._write(out)
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert saved["materials_library"] == str(LIBRARY)
    assert saved["leaching"]["injection_fluid_id"] == FLUID_ID
    assert "injection_concentration_kg_m3" not in saved["leaching"]
    assert "properties" not in saved["leaching"]

    manual_raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    manual_raw["leaching"] = {"injection_fluid_id": FLUID_ID}
    manual = tmp_path / "manual-fluid.yaml"
    manual.write_text(
        yaml.safe_dump(manual_raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    assert (
        load_config(out).leaching.injection_fluid_id
        == load_config(manual).leaching.injection_fluid_id
    )

    win.open_path(out)
    assert combo.currentData() == FLUID_ID
    assert combo.currentText() == FLUID_ID
    assert not win.dirty


def test_legacy_injection_parameters_survive_fluid_choice_and_reset(
    win: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    explicit = {
        "k0_m_per_s": 2.0e-5,
        "injection_concentration_kg_m3": 37.0,
        "plume": False,
        "tubing_radius_m": 0.12,
    }
    raw["leaching"] = explicit
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    win.open_path(legacy)
    combo = win.forms["leaching"].scalars["injection_fluid_id"].widget
    assert combo.currentData() in (None, "")

    unchanged = tmp_path / "unchanged.yaml"
    win._write(unchanged)
    saved = yaml.safe_load(unchanged.read_text(encoding="utf-8"))
    assert "materials_library" not in saved
    assert "injection_fluid_id" not in saved["leaching"]
    for name, value in explicit.items():
        assert saved["leaching"][name] == value

    monkeypatch.setattr(
        qtw.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(LIBRARY)
    )
    win.choose_materials_library()
    combo.setCurrentIndex(combo.findData(FLUID_ID))
    selected = tmp_path / "selected.yaml"
    win._write(selected)
    saved = yaml.safe_load(selected.read_text(encoding="utf-8"))
    assert saved["leaching"]["injection_fluid_id"] == FLUID_ID
    for name, value in explicit.items():
        assert saved["leaching"][name] == value

    combo.setCurrentIndex(combo.findData(""))
    reset = tmp_path / "reset.yaml"
    win._write(reset)
    saved = yaml.safe_load(reset.read_text(encoding="utf-8"))
    assert "injection_fluid_id" not in saved["leaching"]
    for name, value in explicit.items():
        assert saved["leaching"][name] == value


def test_material_list_contains_only_library_rocks(win: Any, tmp_path: Path) -> None:
    path, _ = _linked_case(tmp_path)
    win.open_path(path)

    combo = _material_combo(win)
    ids = [combo.itemData(i) for i in range(combo.count())]

    assert ids == [None, ROCK_A, ROCK_B]
    assert FLUID not in ids
    assert combo.currentData() is None
    assert _material_combo(win, 2).currentData() == ROCK_A
    assert win.cfg is not None


def test_material_selection_and_clear_roundtrip(win: Any, tmp_path: Path) -> None:
    path, raw = _linked_case(tmp_path)
    original_layers = deepcopy(raw["geology"]["layers"])
    win.open_path(path)

    combo = _material_combo(win)
    combo.setCurrentIndex(combo.findData(ROCK_B))
    assert win._write(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_layers = deepcopy(original_layers)
    expected_layers[0]["material_id"] = ROCK_B
    assert saved["geology"]["layers"] == expected_layers
    assert "critical_span_m" not in saved["geology"]["layers"][0]
    assert win.validate() is not None
    assert win.cfg.geology.layers[0].critical_span_m == pytest.approx(4.0)

    win.open_path(path)
    combo = _material_combo(win)
    assert combo.currentData() == ROCK_B
    combo.setCurrentIndex(combo.findData(None))
    assert win._write(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["geology"]["layers"] == original_layers

    win.open_path(path)
    assert _material_combo(win).currentData() is None
    assert win.validate() is not None


def test_unknown_material_id_is_shown_and_preserved(win: Any, tmp_path: Path) -> None:
    path, raw = _linked_case(tmp_path)
    unknown = "test-sc014-unknown"
    raw["geology"]["layers"][1]["material_id"] = unknown
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    win.open_path(path)

    assert win.validate() is None
    messages = [win.errors.item(i).text() for i in range(win.errors.count())]
    assert any("geology.layers.1.material_id" in message for message in messages)
    assert any(unknown in message and "неизвестный id" in message for message in messages)
    combo = _material_combo(win, 1)
    assert combo.currentData() == unknown
    assert unknown in combo.currentText()
    assert win._write(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["geology"]["layers"][1]["material_id"] == unknown

    win.open_path(path)
    assert win.validate() is None
    assert _material_combo(win, 1).currentData() == unknown


def test_legacy_config_gui_roundtrip_preserves_results(win: Any, tmp_path: Path) -> None:
    raw: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text("utf-8"))
    raw["grid"]["nz"] = 50
    for stage, days in zip(raw["schedule"]["stages"], (20.0, 60.0), strict=True):
        stage["duration_days"] = days
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    expected = run(load_config(path))

    win.open_path(path)
    assert _material_combo(win).currentData() is None
    assert win._write(path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "materials_library" not in saved
    assert all("material_id" not in layer for layer in saved["geology"]["layers"])
    actual = run(load_config(path))

    for name in ("radius", "volume", "floor_depth", "dissolved_salt", "produced_salt"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=1e-12, err_msg=name
        )
    win.open_path(path)
    assert _material_combo(win).currentData() is None
    assert win.cfg == load_config(path)


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

    def fake(mc: Any, z: np.ndarray, radius: np.ndarray, layers: Any = None) -> np.ndarray:
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


def test_console_and_script(win: Any, tmp_path: Path) -> None:
    """Консоль: выражение печатает значение, ошибки — в консоль, show/apply — в окно."""
    win.open_path(ROOT / "configs" / "example.yaml")
    con = win.console()
    con.inp.setPlainText("sc.get(cfg, 'leaching.k0_m_per_s')")
    con.run_input()
    win.wait()
    app_ = qtw.QApplication.instance()
    app_.processEvents()
    assert "3e-05" in con.out.toPlainText()
    con.inp.setPlainText("1/0")
    con.run_input()
    win.wait()
    app_.processEvents()
    assert "ZeroDivisionError" in con.out.toPlainText()
    con.inp.setPlainText(
        "c = sc.set(cfg, grid__nz=60, name='из консоли')\n"
        "res = sc.run(c)\nprint('объём', round(res.volume[-1]))\nshow(res)\napply(c)"
    )
    con.run_input()
    win.wait()
    app_.processEvents()
    assert "объём" in con.out.toPlainText()
    assert win.results is not None and win.cfg.grid.nz == 60 and win.cfg.name == "из консоли"
    assert con.ns["res"].volume[-1] > 0  # переменные остаются между запусками
    script = tmp_path / "s.py"
    script.write_text("fig = sc.plot(res)\nprint('рисунок', type(fig).__name__)\nshow(fig)\n",
                      encoding="utf-8")  # fmt: skip
    assert win.run_script(script)
    win.wait()
    app_.processEvents()
    assert "рисунок Figure" in con.out.toPlainText()


def test_report_from_window(win: Any, tmp_path: Path) -> None:
    win.open_path(ROOT / "configs" / "example.yaml")
    out = tmp_path / "w.html"
    win.make_report(out, ["leaching"], open_browser=False)
    win.wait()
    assert out.exists() and "2. Выщелачивание" in out.read_text(encoding="utf-8")


def test_window_icon(win: Any) -> None:
    assert not win.windowIcon().isNull()


def test_data_template_button(win: Any, tmp_path: Path) -> None:
    win.open_path(ROOT / "configs" / "demo_calibration.yaml")
    out = win.make_data_template(tmp_path / "скв-7", open_folder=False)
    assert out is not None and (out / "operations.csv").exists()
    assert win.data_ed.text() == str(out)


def test_license_dialog(win: Any) -> None:
    dlg = win.show_license()
    assert "Sergei Kizred" in dlg.findChild(qtw.QPlainTextEdit).toPlainText()
    dlg.close()
