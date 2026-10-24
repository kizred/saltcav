"""Окно saltcav: параметры → расчёт → просмотр (препроцессор, решатель, постпроцессор).

Вкладки:

- **Параметры** — формы по разделам конфигурации, схема скважины, список ошибок.
  Проверка — той же моделью, что и ``saltcav check``; файл — тот же YAML.
- **Выщелачивание** — просмотрщик по шагам (3D, разрез, графики, события).
- **Ансамбль** — огибающие, сводка, таблица реализаций; любую реализацию можно
  пересчитать с полной историей или посчитать для неё геомеханику.
- **Геомеханика** — разрез массива по годам, конвергенция и запас по дилатансии.

Долгие расчёты идут в отдельном потоке с индикатором и кнопкой «Отмена».

Запуск: ``saltcav gui [файл.yaml | результаты.npz]`` или ярлык ``saltcav-gui``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pydantic import ValidationError
from PySide6 import QtCore, QtGui, QtWidgets

from saltcav import __version__
from saltcav.core.config import RunConfig, config_warnings
from saltcav.gui.forms import ModelForm, ScalarEditor, ru_message
from saltcav.gui.well import draw_well
from saltcav.io.results import Results

SECTIONS: tuple[tuple[str, str], ...] = (
    ("grid", "Сетка"),
    ("geology", "Геология и прослои"),
    ("schedule", "Режим"),
    ("leaching", "Выщелачивание"),
    ("uncertainty", "Неопределённость"),
    ("ensemble", "Ансамбль"),
    ("mechanics", "Геомеханика"),
    ("surrogate", "Устойчивость"),
    ("calibration", "Калибровка"),
)
CALIB_TAB = 5
SCHEMA_REL = Path("schema") / "saltcav-config.schema.json"

TEMPLATE: dict[str, Any] = {
    "name": "new-cavern",
    "grid": {"z_top_m": 900, "z_bottom_m": 1100, "nz": 200, "nphi": 1},
    "geology": {"layers": []},
    "schedule": {
        "stages": [
            {
                "duration_days": 60,
                "mode": "direct",
                "rate_m3_per_h": 50,
                "injection_depth_m": 1090,
                "production_depth_m": 1080,
                "blanket_depth_m": 1060,
            },
            {
                "duration_days": 180,
                "mode": "reverse",
                "rate_m3_per_h": 100,
                "injection_depth_m": 1040,
                "production_depth_m": 1090,
                "blanket_depth_m": 1020,
            },
        ]
    },
}


class Cancelled(Exception):
    """Расчёт отменён пользователем."""


class Worker(QtCore.QObject):
    """Функция в отдельном потоке: fn(report) → результат; report(доля, текст)."""

    progress = QtCore.Signal(float, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn: Callable[[Callable[[float, str], None]], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def report(self, frac: float, text: str = "") -> None:
        if self.cancelled:
            raise Cancelled
        self.progress.emit(frac, text)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.done.emit(self.fn(self.report))
        except Cancelled:
            self.failed.emit("отменено")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def find_schema(start: Path) -> Path | None:
    for d in (start, *start.parents):
        cand = d / SCHEMA_REL
        if cand.exists():
            return cand
    return None


def _shape_layers(res: Results, cfg: RunConfig) -> list[Any]:
    """Прослои формы: из её конфигурации (у реализации ансамбля — её отметки)."""
    if res.config_yaml:
        try:
            return list(RunConfig.model_validate(yaml.safe_load(res.config_yaml)).geology.layers)
        except (ValidationError, yaml.YAMLError):
            pass
    return list(cfg.geology.layers)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path: Path | None = None
        self.dirty = False
        self.cfg: RunConfig | None = None  # последняя корректная конфигурация
        self.results: Results | None = None
        self.results_key = ""  # конфигурация, по которой получены результаты
        self.ensemble: Any = None
        self.ens_key = ""
        self.mech: Any = None
        self._thread: QtCore.QThread | None = None
        self._worker: Worker | None = None
        self._viewer: Any = None
        self._help: Any = None
        self.surrogate: Any = None
        self.resize(1500, 950)
        from saltcav.gui.icons import app_icon

        self.setWindowIcon(app_icon())

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_params()
        self.leach_tab = self._placeholder(
            "Результатов выщелачивания пока нет — «Рассчитать» (F5)."
        )
        self.ens_tab = self._placeholder("Ансамбль не считался — «Ансамбль» (F6).")
        self.mech_tab = self._placeholder("Геомеханика не считалась — «Геомеханика» (F7).")
        self.tabs.addTab(self.params_tab, "Параметры")
        self.tabs.addTab(self.leach_tab, "Выщелачивание")
        self.tabs.addTab(self.ens_tab, "Ансамбль")
        self.tabs.addTab(self.mech_tab, "Геомеханика")
        self.sur_tab = self._placeholder(
            "Устойчивость по ансамблю не считалась — вкладка «Ансамбль» → «Устойчивость»."
        )
        self.tabs.addTab(self.sur_tab, "Устойчивость")
        self.calib: Any = None
        self.calib_data: Any = None
        self.tabs.addTab(self._calib_page(None), "Калибровка")
        self._build_actions()

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.cancel_btn = QtWidgets.QPushButton("Отмена")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.validate)

        if path is not None:
            self.open_path(path)
        else:
            self.load_dict(TEMPLATE)
            self.dirty = False
            self._update_title()

    # --- построение -----------------------------------------------------------------

    def _placeholder(self, text: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lab = QtWidgets.QLabel(text)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lab.setStyleSheet("color: #52514e; font-size: 14px;")
        lay.addWidget(lab)
        return w

    def _build_params(self) -> None:
        self.params_tab = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        top = QtWidgets.QFormLayout()
        self.name_ed = ScalarEditor("name", RunConfig.model_fields["name"], self._changed)
        self.name_ed.widget.setProperty("saltcav_path", "name")
        top.addRow("Название расчёта", self.name_ed.widget)
        ll.addLayout(top)
        self.section_tabs = QtWidgets.QTabWidget()
        self.forms: dict[str, ModelForm] = {}
        for key, title in SECTIONS:
            model = RunConfig.model_fields[key].annotation
            form = ModelForm(model, self._changed, prefix=key)  # type: ignore[arg-type]
            self.forms[key] = form
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form)
            self.section_tabs.addTab(scroll, title)
        ll.addWidget(self.section_tabs, 1)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        self.well_fig = Figure(figsize=(5, 7), facecolor="#fcfcfb", layout="constrained")
        self.well_canvas = FigureCanvasQTAgg(self.well_fig)
        self.well_ax = self.well_fig.add_subplot(111)
        rl.addWidget(self.well_canvas, 3)
        self.status_lab = QtWidgets.QLabel()
        self.status_lab.setWordWrap(True)
        rl.addWidget(self.status_lab)
        self.errors = QtWidgets.QListWidget()
        self.errors.setWordWrap(True)
        self.errors.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.errors.itemClicked.connect(self._goto_error)
        rl.addWidget(self.errors, 1)
        self.params_tab.addWidget(left)
        self.params_tab.addWidget(right)
        self.params_tab.setSizes([900, 600])

    def _build_actions(self) -> None:
        tb = self.addToolBar("Главная")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        style = self.style()
        sp = QtWidgets.QStyle.StandardPixmap

        def act(text: str, slot: Callable[[], Any], key: str | None = None,
                icon: Any = None, menu: Any = None) -> QtGui.QAction:  # fmt: skip
            a = QtGui.QAction(text, self)
            if icon is not None:
                a.setIcon(style.standardIcon(icon))
            if key:
                a.setShortcut(QtGui.QKeySequence(key))
            a.triggered.connect(slot)
            if menu is not None:
                menu.addAction(a)
            return a

        mf = self.menuBar().addMenu("Файл")
        tb.addAction(act("Новый", self.new_file, "Ctrl+N", sp.SP_FileIcon, mf))
        tb.addAction(act("Открыть…", self.open_dialog, "Ctrl+O", sp.SP_DialogOpenButton, mf))
        tb.addAction(act("Сохранить", self.save, "Ctrl+S", sp.SP_DialogSaveButton, mf))
        act("Сохранить как…", self.save_as, "Ctrl+Shift+S", menu=mf)
        mf.addSeparator()
        act("Сохранить результаты выщелачивания (.npz)…", self.save_results, menu=mf)
        act("Сохранить ансамбль (.npz)…", self.save_ensemble, menu=mf)
        act("Сохранить геомеханику (.npz)…", self.save_mech, menu=mf)
        act("Картинка геомеханики (.png)…", self.save_mech_png, menu=mf)
        act("Сохранить устойчивость (.npz)…", self.save_surrogate, menu=mf)
        act("Сохранить калибровку (.npz)…", self.save_calibration, menu=mf)
        mf.addSeparator()
        act("Выполнить скрипт Python…", self.run_script_dialog, "Ctrl+R", menu=mf)
        self.report_act = act("Отчёт…", self.report_dialog, "Ctrl+P", sp.SP_FileDialogInfoView, mf)
        mf.addSeparator()
        act("Выход", self.close, "Ctrl+Q", menu=mf)
        tb.addSeparator()
        mr = self.menuBar().addMenu("Расчёт")
        tb.addAction(act("Проверить", self.check, "Ctrl+K", sp.SP_DialogApplyButton, mr))
        self.run_act = act("Рассчитать", self.run_leaching, "F5", sp.SP_MediaPlay, mr)
        self.ens_act = act("Ансамбль", lambda: self.run_ensemble(), "F6",
                           sp.SP_FileDialogDetailedView, mr)  # fmt: skip
        self.mech_act = act("Геомеханика", lambda: self.run_mechanics(), "F7",
                            sp.SP_DriveHDIcon, mr)  # fmt: skip
        self.cal_act = act("Калибровка", lambda: self.tabs.setCurrentIndex(CALIB_TAB), "F8",
                           sp.SP_FileDialogContentsView, mr)  # fmt: skip
        for a in (self.run_act, self.ens_act, self.mech_act, self.cal_act):
            tb.addAction(a)
        mv = self.menuBar().addMenu("Вид")
        self.console_act = act("Консоль Python", self.toggle_console, "Ctrl+`",
                               sp.SP_TitleBarNormalButton, mv)  # fmt: skip
        tb.addSeparator()
        tb.addAction(self.report_act)
        tb.addAction(self.console_act)
        mh = self.menuBar().addMenu("Справка")
        act("Справка по полю", lambda: self.help_for_focus(), "F1", menu=mh)
        act("Руководство", lambda: self.show_help(), menu=mh)
        act("Справочник параметров", lambda: self.show_help("parameters"), menu=mh)
        mh.addSeparator()
        act("О программе", self.about, menu=mh)

    # --- отчёт ------------------------------------------------------------------------

    def report_dialog(self) -> None:
        """Отчёт в один клик: что уже посчитано — входит, недостающее — по галочкам."""
        cfg = self.validate()
        if cfg is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Отчёт")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel("Разделы отчёта (готовые результаты берутся как есть):"))
        have_res = self.results is not None and self.results_key == self._key(cfg)
        boxes: dict[str, QtWidgets.QCheckBox] = {}
        items = (
            ("leaching", "Выщелачивание", have_res, "секунды"),
            ("ensemble", "Ансамбль и карты вероятностей", self.ensemble is not None, "минуты"),
            ("mechanics", "Геомеханика", self.mech is not None, "десятки секунд"),
            ("stability", "Устойчивость по ансамблю", self.surrogate is not None,
             "минуты — десятки минут"),
        )  # fmt: skip
        for key, label, have, cost in items:
            b = QtWidgets.QCheckBox(f"{label} — " + ("готово" if have else f"посчитать ({cost})"))
            b.setChecked(have or key == "leaching")
            if key == "leaching":
                b.setEnabled(False)
            boxes[key] = b
            lay.addWidget(b)
        cal = QtWidgets.QCheckBox("Калибровка — готово")
        cal.setChecked(self.calib is not None)
        cal.setEnabled(self.calib is not None)
        lay.addWidget(cal)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        default = self.path.with_suffix(".html").name if self.path else f"{cfg.name}.html"
        out = self._save_dialog("Отчёт", "HTML (*.html)", default)
        if out:
            self.make_report(Path(out), [k for k, b in boxes.items() if b.isChecked()],
                             with_calibration=cal.isChecked())  # fmt: skip

    def make_report(self, out: Path, parts: list[str], with_calibration: bool = True,
                    open_browser: bool = True) -> None:  # fmt: skip
        cfg = self.cfg
        if cfg is None:
            return
        have_res = self.results is not None and self.results_key == self._key(cfg)
        kw: dict[str, Any] = {
            "results": self.results if have_res else None,
            "ensemble": self.ensemble if "ensemble" in parts else None,
            "mechanics": self.mech if "mechanics" in parts else None,
            "stability": self.surrogate if "stability" in parts else None,
            "calibration": self.calib if with_calibration else None,
            "data": getattr(self, "calib_fielddata", None) if with_calibration else None,
        }
        jobs = os.cpu_count() or 1

        def job(report: Callable[[float, str], None]) -> Any:
            from saltcav.report import make_report

            return make_report(cfg, out, compute=parts, jobs=jobs, progress=report, **kw)

        def done(path: Any) -> None:
            self.statusBar().showMessage(f"отчёт: {path}", 10000)
            if open_browser:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

        self._start("Отчёт", job, done)

    # --- консоль и скрипты ------------------------------------------------------------

    def console(self) -> Any:
        """Консоль Python (создаётся при первом обращении) в нижней панели окна."""
        if getattr(self, "_console", None) is None:
            from saltcav.gui.console import Console

            self._console = Console(self)
            dock = QtWidgets.QDockWidget("Консоль Python", self)
            dock.setObjectName("console")
            dock.setWidget(self._console)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._console_dock = dock
        return self._console

    def toggle_console(self) -> None:
        self.console()
        dock, inp = self._console_dock, self._console.inp
        if dock.isVisible() and inp.hasFocus():
            dock.hide()
            return
        dock.show()
        dock.raise_()
        inp.setFocus()

    def run_script_dialog(self) -> None:
        start = str(self.path.parent) if self.path else ""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Скрипт Python", start,
                                                     "Python (*.py)")  # fmt: skip
        if p:
            self.run_script(Path(p))

    def run_script(self, path: Path) -> bool:
        """Выполнить файл .py в консоли (вывод — в консоль)."""
        con = self.console()
        self._console_dock.show()
        con.write(f"— скрипт {path}\n")
        return bool(con.run_source(path.read_text(encoding="utf-8"), str(path), path=path))

    def apply_config(self, cfg: Any) -> None:
        """Загрузить конфигурацию (RunConfig или словарь) в форму параметров."""
        data = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
        self.load_dict(data)
        self.dirty = True
        self._update_title()
        self.tabs.setCurrentIndex(0)

    def show_object(self, obj: Any) -> None:
        """Показать результат на его вкладке; фигуру matplotlib — в отдельном окне."""
        from matplotlib.figure import Figure as MplFigure

        from saltcav.calibration.fit import CalibResult
        from saltcav.mechanics.run import MechanicsRun
        from saltcav.stochastic.ensemble import EnsembleResult
        from saltcav.surrogate.study import StudyResult

        if isinstance(obj, Results):
            self._show_results(obj)
            self.tabs.setCurrentIndex(1)
        elif isinstance(obj, EnsembleResult):
            self._show_ensemble(obj, self.cfg)
            self.tabs.setCurrentIndex(2)
        elif isinstance(obj, MechanicsRun):
            self._show_mech(obj)
        elif isinstance(obj, StudyResult):
            self._show_surrogate(obj)
            self.tabs.setCurrentIndex(4)
        elif isinstance(obj, CalibResult):
            self._show_calibration(obj)
        elif isinstance(obj, MplFigure):
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Рисунок")
            lay = QtWidgets.QVBoxLayout(dlg)
            canvas = FigureCanvasQTAgg(obj)
            lay.addWidget(canvas)
            w, h = obj.get_size_inches()
            dlg.resize(int(w * 90), int(h * 90))
            dlg.show()
            self._figures = [*getattr(self, "_figures", []), dlg]
        elif hasattr(obj, "model_dump"):
            self.apply_config(obj)
        else:
            raise TypeError(f"не умею показать {type(obj).__name__}")

    # --- данные формы ---------------------------------------------------------------

    def form_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        given, name = self.name_ed.get()
        if given:
            data["name"] = name
        for key, form in self.forms.items():
            data[key] = form.get()
        return data

    def load_dict(self, data: dict[str, Any]) -> None:
        self.name_ed.set(data.get("name"))
        for key, form in self.forms.items():
            form.set(data.get(key) or {})
        self.validate()

    def _changed(self) -> None:
        self.dirty = True
        self._update_title()
        self._timer.start()

    def _update_title(self) -> None:
        name = self.path.name if self.path else "без имени"
        self.setWindowTitle(f"saltcav {__version__} — {name}{' *' if self.dirty else ''}")

    def validate(self) -> RunConfig | None:
        """Проверить форму моделью; подсветить ошибки; обновить схему скважины."""
        self.name_ed.set_error(None)
        for form in self.forms.values():
            form.clear_errors()
        self.errors.clear()
        try:
            cfg = RunConfig.model_validate(self.form_dict())
        except ValidationError as exc:
            for err in exc.errors():
                loc = tuple(err["loc"])
                msg = ru_message(dict(err))
                where = ".".join(str(x) for x in loc) or "конфигурация"
                item = QtWidgets.QListWidgetItem(f"{where}: {msg}")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, loc)
                self.errors.addItem(item)
                self._mark(loc, msg)
            n = len(exc.errors())
            self.status_lab.setText(f"<span style='color:#c0392b'>Ошибок: {n}</span>")
            self._set_actions(False)
            return None
        self.cfg = cfg
        warns = config_warnings(cfg)
        for loc, msg in warns:
            item = QtWidgets.QListWidgetItem(f"⚠ {'.'.join(loc)}: {msg}")
            item.setForeground(QtGui.QColor("#a86b00"))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, loc)
            self.errors.addItem(item)
            if loc[0] in self.forms:
                self.forms[loc[0]].mark_error(loc[1:], msg, warn=True)
        note = f", предупреждений: {len(warns)}" if warns else ""
        self.status_lab.setText(f"<span style='color:#2e7d32'>Конфигурация корректна</span>{note}")
        self._set_actions(True)
        self._draw_well()
        return cfg

    def _mark(self, loc: tuple[Any, ...], msg: str) -> None:
        if not loc:
            hit = re.search(r"ступень (\d+)", msg)
            if hit:
                self.forms["schedule"].mark_error(("stages", int(hit.group(1))), msg)
            return
        if loc[0] == "name":
            self.name_ed.set_error(msg)
        elif loc[0] in self.forms:
            self.forms[str(loc[0])].mark_error(loc[1:], msg)

    def _goto_error(self, item: QtWidgets.QListWidgetItem) -> None:
        loc = item.data(QtCore.Qt.ItemDataRole.UserRole) or ()
        key = loc[0] if loc else "schedule"
        for i, (k, _) in enumerate(SECTIONS):
            if k == key:
                self.section_tabs.setCurrentIndex(i)

    def _set_actions(self, ok: bool) -> None:
        busy = self._thread is not None
        for a in (self.run_act, self.ens_act, self.mech_act):
            a.setEnabled(ok and not busy)

    def _draw_well(self) -> None:
        if self.cfg is None:
            return
        profile = None
        if self.results is not None:
            r = self.results.radius[-1]
            profile = (self.results.z_edges, np.sqrt(np.mean(r**2, axis=1)))
        try:
            draw_well(self.well_ax, self.cfg, profile)
        except Exception as exc:
            self.well_ax.clear()
            self.well_ax.text(
                0.5, 0.5, f"схема: {exc}", ha="center", transform=self.well_ax.transAxes
            )
        self.well_canvas.draw_idle()

    def _key(self, cfg: RunConfig) -> str:
        return cfg.model_dump_json()

    # --- файлы ------------------------------------------------------------------------

    def _ask_discard(self) -> bool:
        if not self.dirty:
            return True
        ans = QtWidgets.QMessageBox.question(
            self, "saltcav", "Изменения не сохранены. Продолжить без сохранения?"
        )
        return ans == QtWidgets.QMessageBox.StandardButton.Yes

    def new_file(self) -> None:
        if self._ask_discard():
            self.path = None
            self.load_dict(TEMPLATE)
            self.dirty = False
            self._update_title()

    def open_dialog(self) -> None:
        if not self._ask_discard():
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Открыть", str(self.path.parent if self.path else Path.cwd()),
            "Конфигурация или результаты (*.yaml *.yml *.npz)",
        )  # fmt: skip
        if p:
            self.open_path(Path(p))

    def open_path(self, path: Path) -> None:
        try:
            if path.suffix == ".npz":
                self._open_npz(path)
                return
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "Открыть", f"{path}:\n{exc}")
            return
        self.path = path
        self.load_dict(data)
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"открыт {path}", 5000)

    def _open_npz(self, path: Path) -> None:
        with np.load(path) as d:
            kind = str(d["kind"]) if "kind" in d else ""
        if kind == "saltcav-surrogate":
            from saltcav.surrogate.study import StudyResult

            self._show_surrogate(StudyResult.load(path))
            self.tabs.setCurrentIndex(4)
            return
        if kind == "saltcav-calibration":
            from saltcav.calibration import CalibResult

            self._show_calibration(CalibResult.load(path))
            self.tabs.setCurrentIndex(CALIB_TAB)
            return
        if kind == "saltcav-mechanics":
            from saltcav.mechanics.run import MechanicsRun

            self._show_mech(MechanicsRun.load(path))
            return
        if "param_names" in np.load(path):
            from saltcav.stochastic.ensemble import EnsembleResult

            self._show_ensemble(EnsembleResult.load(path), None)
            return
        res = Results.load(path)
        if res.config_yaml:
            self.load_dict(yaml.safe_load(res.config_yaml))
            self.dirty = True
            self._update_title()
        self._show_results(res)

    def _schema_header(self, path: Path) -> str:
        schema = find_schema(path.resolve().parent)
        if schema is None:
            return ""
        rel = os.path.relpath(schema, path.resolve().parent).replace(os.sep, "/")
        return f"# $schema: {rel}\n# yaml-language-server: $schema={rel}\n"

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        return self._write(self.path)

    def save_as(self) -> bool:
        start = str(self.path or Path.cwd() / f"{self.form_dict().get('name', 'case')}.yaml")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить", start, "YAML (*.yaml)")
        if not p:
            return False
        return self._write(Path(p))

    def _write(self, path: Path) -> bool:
        data = self.form_dict()
        text = self._schema_header(path) + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        self.path = path
        self.dirty = False
        self._update_title()
        if self.validate() is None:
            self.statusBar().showMessage(f"сохранено с ошибками: {path}", 8000)
        else:
            self.statusBar().showMessage(f"сохранено: {path}", 5000)
        return True

    def _save_dialog(self, title: str, filt: str, default: str) -> str:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default, filt)
        return str(p)

    def save_results(self) -> None:
        if self.results is None:
            return
        p = self._save_dialog("Результаты", "saltcav (*.npz)", f"{self.results.name}.npz")
        if p:
            self.results.save(p)

    def save_ensemble(self) -> None:
        if self.ensemble is None:
            return
        p = self._save_dialog("Ансамбль", "saltcav (*.npz)", "ensemble.npz")
        if p:
            self.ensemble.save(p)

    def save_mech(self) -> None:
        if self.mech is None:
            return
        p = self._save_dialog("Геомеханика", "saltcav (*.npz)", f"{self.mech.name}_mech.npz")
        if p:
            self.mech.save(p)

    def save_mech_png(self) -> None:
        view = self.tabs.widget(3)
        if self.mech is None or not hasattr(view, "save_png"):
            return
        p = self._save_dialog("Картинка", "PNG (*.png)", f"{self.mech.name}_mech.png")
        if p:
            view.save_png(p)

    # --- расчёты --------------------------------------------------------------------

    def check(self) -> None:
        cfg = self.validate()
        if cfg is not None:
            total = sum(s.duration_days for s in cfg.schedule.stages)
            self.statusBar().showMessage(
                f"OK: сетка {cfg.grid.nz}×{cfg.grid.nphi}, {len(cfg.schedule.stages)} ступ., "
                f"{total:g} сут", 8000,
            )  # fmt: skip

    def _start(self, title: str, fn: Callable[..., Any], on_done: Callable[[Any], None]) -> None:
        if self._thread is not None:
            return
        self._thread = QtCore.QThread(self)
        self._worker = Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._job_title, self._job_done = title, on_done
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        # слоты окна: выполняются в главном потоке (лямбда выполнилась бы в рабочем)
        self._worker.progress.connect(self._on_progress, queued)
        self._worker.done.connect(self._on_done, queued)
        self._worker.failed.connect(self._on_failed, queued)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.cancel_btn.setVisible(True)
        self.statusBar().showMessage(f"{title}…")
        self._set_actions(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.BusyCursor)
        self._thread.start()

    @QtCore.Slot(object)
    def _on_done(self, result: Any) -> None:
        self._finish(self._job_title, result, self._job_done)

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        self._finish(self._job_title, None, None, msg)

    @QtCore.Slot(float, str)
    def _on_progress(self, frac: float, text: str) -> None:
        self.progress.setValue(int(100 * frac))
        if text:
            self.statusBar().showMessage(text)

    def _finish(self, title: str, result: Any, on_done: Callable[[Any], None] | None,
                error: str | None = None) -> None:  # fmt: skip
        QtWidgets.QApplication.restoreOverrideCursor()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._set_actions(self.cfg is not None)
        if error is not None:
            self.statusBar().showMessage(f"{title}: {error}", 10000)
            if error != "отменено":
                QtWidgets.QMessageBox.warning(self, title, error)
            return
        if on_done is not None:
            on_done(result)

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancelled = True
            self.statusBar().showMessage("отмена…")

    def wait(self, ms: int = 120_000) -> None:
        """Дождаться конца фонового расчёта (для тестов и сценариев)."""
        loop = QtCore.QEventLoop()
        deadline = QtCore.QDeadlineTimer(ms)
        while self._thread is not None and not deadline.hasExpired():
            loop.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

    def _leaching_job(self, cfg: RunConfig) -> Callable[[Callable[[float, str], None]], Results]:
        def job(report: Callable[[float, str], None]) -> Results:
            from saltcav.leaching import simulate

            report(0.1, "выщелачивание…")
            res = Results.from_history(simulate(cfg), cfg)
            report(1.0, "")
            return res

        return job

    def run_leaching(self) -> None:
        cfg = self.validate()
        if cfg is None:
            return
        key = self._key(cfg)

        def done(res: Results) -> None:
            self.results_key = key
            self._show_results(res)
            self.tabs.setCurrentIndex(1)

        self._start("Выщелачивание", self._leaching_job(cfg), done)

    def run_ensemble(self, n: int | None = None, jobs: int | None = None) -> None:
        cfg = self.validate()
        if cfg is None:
            return
        if n is None:
            n, ok = QtWidgets.QInputDialog.getInt(
                self, "Ансамбль", "Число реализаций:", cfg.ensemble.n_realizations, 2, 100_000
            )
            if not ok:
                return
        jobs = jobs or os.cpu_count() or 1

        def job(report: Callable[[float, str], None]) -> Any:
            from saltcav.stochastic.ensemble import run_ensemble

            return run_ensemble(
                cfg, n=n, jobs=jobs, progress=lambda k, m: report(k / m, f"реализация {k}/{m}")
            )

        def done(ens: Any) -> None:
            self._show_ensemble(ens, cfg)
            self.tabs.setCurrentIndex(2)

        self._start(f"Ансамбль ({n} реализаций, {jobs} проц.)", job, done)

    def run_mechanics(self, shape: Results | None = None) -> None:
        cfg = self.validate()
        if cfg is None:
            return
        res = shape
        if res is None and self.results is not None and self.results_key == self._key(cfg):
            res = self.results
        mc = cfg.mechanics

        def job(report: Callable[[float, str], None]) -> Any:
            from saltcav.leaching import simulate
            from saltcav.mechanics.run import run_from_shape

            r = res
            if r is None:  # формы по текущим параметрам нет — сначала выщелачивание
                report(0.0, "выщелачивание…")
                r = Results.from_history(simulate(cfg), cfg)
            i = r.nt - 1 if mc.day is None else r.frame_at(mc.day)
            run = run_from_shape(
                mc, r.z_edges, r.radius[i], r.name, float(r.days[i]), float(r.volume[i]),
                progress=lambda f: report(f, f"геомеханика: {100 * f:.0f} %"),
                layers=_shape_layers(r, cfg),
            )  # fmt: skip
            return r, run

        def done(out: Any) -> None:
            r, run = out
            if shape is None and self.results is None:
                self.results_key = self._key(cfg)
                self._show_results(r)
            self._show_mech(run)
            self.tabs.setCurrentIndex(3)

        self._start("Геомеханика", job, done)

    # --- показ результатов --------------------------------------------------------------

    def _replace_tab(self, index: int, widget: QtWidgets.QWidget, title: str) -> None:
        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, widget, title)
        if old is not None:
            if hasattr(old, "vtk"):
                old.close()
            old.deleteLater()

    def _show_results(self, res: Results) -> None:
        self.results = res
        try:
            from saltcav.viz.viewer import build_window

            view = build_window(res, None, embedded=True)
            self._viewer = view
        except Exception as exc:
            view = self._placeholder(f"3D-просмотр недоступен: {exc}")
        self._replace_tab(1, view, "Выщелачивание")
        self._draw_well()
        v = f"{res.volume[-1]:,.0f}".replace(",", " ")
        self.statusBar().showMessage(
            f"{res.name}: {res.days[-1]:g} сут, V = {v} м³, R_max = {res.radius[-1].max():.1f} м",
            10000,
        )

    def _show_ensemble(self, ens: Any, cfg: RunConfig | None) -> None:
        self.ensemble = ens
        self.ens_cfg = cfg
        w = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        fig = Figure(figsize=(9, 6), facecolor="#fcfcfb")
        canvas = FigureCanvasQTAgg(fig)
        from saltcav.viz.envelope import draw_envelope

        layers = [(lay.top_m, lay.thickness_m) for lay in cfg.geology.layers] if cfg else []
        draw_envelope(fig, ens, f"{ens.n} реализаций, seed {ens.seed}", layers)
        w.addWidget(canvas)

        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        lines = []
        for key, label, scale in (
            ("volume", "объём, тыс. м³", 1e3), ("max_radius", "макс. радиус, м", 1.0),
            ("floor_depth", "подошва, м", 1.0), ("centroid_offset", "смещение центра, м", 1.0),
        ):  # fmt: skip
            q = np.nanquantile(ens.output(key) / scale, (0.1, 0.5, 0.9))
            lines.append(f"{label}: {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}")
        info = QtWidgets.QLabel("<b>P10 / P50 / P90</b><br>" + "<br>".join(lines))
        sl.addWidget(info)
        table = QtWidgets.QTableWidget(ens.n, 5)
        table.setHorizontalHeaderLabels(["№", "объём, м³", "R_max, м", "подошва, м", "обрушений"])
        vol, rmax = ens.output("volume"), ens.output("max_radius")
        for i in range(ens.n):
            vals = (i, round(float(vol[i])), round(float(rmax[i]), 2),
                    round(float(ens.floor_depth[i]), 1), int(ens.collapses[i]))  # fmt: skip
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem()
                it.setData(QtCore.Qt.ItemDataRole.DisplayRole, v)
                table.setItem(i, j, it)
        table.setSortingEnabled(True)
        table.sortItems(2, QtCore.Qt.SortOrder.DescendingOrder)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        sl.addWidget(QtWidgets.QLabel("Реализации (щелчок по заголовку — сортировка):"))
        sl.addWidget(table, 1)
        b1 = QtWidgets.QPushButton("Открыть реализацию в просмотрщике")
        b2 = QtWidgets.QPushButton("Геомеханика реализации")
        b3 = QtWidgets.QPushButton("Устойчивость по ансамблю (суррогат)…")
        b3.setToolTip(
            "Геомеханика для части реализаций, гауссов процесс — для остальных; "
            "вероятность потери устойчивости. Параметры — раздел «Устойчивость»."
        )
        sl.addWidget(b1)
        sl.addWidget(b2)
        sl.addWidget(b3)
        b3.clicked.connect(lambda: self.run_surrogate())
        self.ens_table = table

        def chosen() -> int | None:
            r = table.currentRow()
            return None if r < 0 else int(table.item(r, 0).data(QtCore.Qt.ItemDataRole.DisplayRole))

        b1.clicked.connect(lambda: self.open_realization(chosen(), mech=False))
        b2.clicked.connect(lambda: self.open_realization(chosen(), mech=True))
        b1.setEnabled(cfg is not None)
        b2.setEnabled(cfg is not None)
        b3.setEnabled(cfg is not None)
        w.addWidget(side)
        w.setSizes([900, 500])
        self._replace_tab(2, w, "Ансамбль")

    def run_surrogate(self) -> None:
        cfg = self.validate()
        if cfg is None or self.ensemble is None:
            return
        ens = self.ensemble
        jobs = os.cpu_count() or 1

        def job(report: Callable[[float, str], None]) -> Any:
            from saltcav.mechanics.run import ensemble_layers
            from saltcav.surrogate.study import run_study

            return run_study(cfg, ens.z_edges, ens.radius, seed=ens.seed, jobs=jobs,
                             progress=report, layers=ensemble_layers(cfg, ens))  # fmt: skip

        def done(res: Any) -> None:
            self._show_surrogate(res)
            self.tabs.setCurrentIndex(4)

        self._start(f"Устойчивость ({ens.n} реализаций)", job, done)

    def _show_surrogate(self, res: Any) -> None:
        from saltcav.surrogate.study import KEY, TARGETS
        from saltcav.viz.surrogate_plot import SURFACE, draw_surrogate

        self.surrogate = res
        w = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        fig = Figure(figsize=(11, 8), facecolor=SURFACE)
        canvas = FigureCanvasQTAgg(fig)
        draw_surrogate(fig, res)
        w.addWidget(canvas)
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        p, p_unc = res.prob_below()
        lines = [
            f"<b>P(запас &lt; {res.threshold:g}) = {p:.3f} ± {res.prob_ci(p):.3f}</b>",
            f"с неопределённостью суррогата: {p_unc:.3f}",
            f"P(запас &lt; 1,5) = {res.prob_below(threshold=1.5)[0]:.3f}",
            f"расчётов МКЭ: {res.train.size}" + (f" + {res.hf.size} в 3D" if res.hf.size else ""),
            "<br><b>P10 / P50 / P90</b>",
        ]
        for name, (label, _) in TARGETS.items():
            q = res.quantiles(name)
            lines.append(f"{label}: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
        info = QtWidgets.QLabel("<br>".join(lines))
        info.setWordWrap(True)
        sl.addWidget(info)
        sl.addWidget(QtWidgets.QLabel("Наименьший запас (по суррогату):"))
        j = res.column(KEY)
        crit = res.critical(20)
        table = QtWidgets.QTableWidget(crit.size, 4)
        table.setHorizontalHeaderLabels(["№", "запас", "± (σ)", "расчёт МКЭ"])
        trained = {int(i): k for k, i in enumerate(res.train)}
        for r, i in enumerate(crit):
            fe = res.y_train[trained[int(i)], j] if int(i) in trained else None
            vals = (int(i), round(float(res.mu[i, j]), 3), round(float(res.sd[i, j]), 3),
                    "" if fe is None else round(float(fe), 3))  # fmt: skip
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem()
                it.setData(QtCore.Qt.ItemDataRole.DisplayRole, v)
                table.setItem(r, c, it)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        sl.addWidget(table, 1)
        b = QtWidgets.QPushButton("Полная геомеханика выбранной реализации")
        b.setEnabled(self.ensemble is not None and getattr(self, "ens_cfg", None) is not None)

        def chosen() -> int | None:
            r = table.currentRow()
            return None if r < 0 else int(table.item(r, 0).data(QtCore.Qt.ItemDataRole.DisplayRole))

        b.clicked.connect(lambda: self.open_realization(chosen(), mech=True))
        sl.addWidget(b)
        w.addWidget(side)
        w.setSizes([1000, 420])
        self._replace_tab(4, w, "Устойчивость")

    def save_surrogate(self) -> None:
        if getattr(self, "surrogate", None) is None:
            return
        p = self._save_dialog("Устойчивость", "saltcav (*.npz)", "stability.npz")
        if p:
            self.surrogate.save(p)

    # --- калибровка -----------------------------------------------------------------

    def _calib_page(self, res: Any) -> QtWidgets.QWidget:
        """Вкладка калибровки: выбор данных и запуск; после расчёта — картинка и таблица."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        bar = QtWidgets.QHBoxLayout()
        self.data_ed = QtWidgets.QLineEdit(str(self.calib_data or ""))
        self.data_ed.setPlaceholderText(
            "папка данных скважины: operations.csv, brine.csv, sonar.csv …"
        )
        self.data_ed.editingFinished.connect(lambda: self.load_data(self.data_ed.text()))
        pick = QtWidgets.QPushButton("Данные…")
        pick.clicked.connect(self.choose_data)
        self.method_cb = QtWidgets.QComboBox()
        self.method_cb.addItem("MAP + лаплас (быстро)", "map")
        self.method_cb.addItem("MCMC (точнее, долго)", "mcmc")
        self.holdout_sb = QtWidgets.QDoubleSpinBox()
        self.holdout_sb.setRange(0, 1e5)
        self.holdout_sb.setDecimals(0)
        self.holdout_sb.setSpecialValueText("все замеры")
        self.holdout_sb.setPrefix("подгонка до ")
        self.holdout_sb.setSuffix(" сут")
        self.holdout_sb.setToolTip("Замеры позже — проверка прогноза; 0 — подгонять по всем")
        go = QtWidgets.QPushButton("Калибровать")
        go.clicked.connect(self.run_calibration)
        for wdg in (self.data_ed, pick, self.method_cb, self.holdout_sb, go):
            bar.addWidget(wdg, 1 if wdg is self.data_ed else 0)
        lay.addLayout(bar)
        self.data_info = QtWidgets.QLabel(
            "Подбор параметров растворения по данным скважины (раздел «Калибровка» — что "
            "подбирать и априорные). Демо: data/demo/well-1 с configs/demo_calibration.yaml; "
            "открытый эталон: data/open/sansmic-direct."
        )
        self.data_info.setWordWrap(True)
        self.data_info.setStyleSheet("color: #52514e;")
        lay.addWidget(self.data_info)
        if res is None:
            lay.addStretch(1)
            return page

        from saltcav.viz.calibration_plot import SURFACE, draw_calibration

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        fig = Figure(figsize=(11, 8), facecolor=SURFACE)
        draw_calibration(fig, res)
        split.addWidget(FigureCanvasQTAgg(fig))
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        head = res.table_header()
        rows = res.table()
        table = QtWidgets.QTableWidget(len(rows), len(head))
        table.setHorizontalHeaderLabels(head)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        table.resizeColumnsToContents()
        sl.addWidget(table)
        lines = [f"χ²/N = {res.chi2_reduced:.2f} по {res.n_fit} замерам, {res.n_runs} прогонов"]
        unit = {"brine": "рассол, кг/м³", "sonar": "сонар, м"}
        lines += [f"СКО ({unit[k]}): {v:.2f}" for k, v in res.rms_fit.items()]
        if res.rms_holdout:
            lines.append(f"<b>Проверка после {res.holdout_day:g} сут</b>")
            lines += [f"СКО ({unit[k]}): {v:.2f}" for k, v in res.rms_holdout.items()]
        lines += [f"<span style='color:#a86b00'>⚠ {n}</span>" for n in res.notes()]
        info = QtWidgets.QLabel("<br>".join(lines))
        info.setWordWrap(True)
        sl.addWidget(info)
        sl.addStretch(1)
        apply = QtWidgets.QPushButton("Применить к параметрам")
        apply.setToolTip(
            "Оценки — в разделы «Выщелачивание» и «Геология», их разброс (k0, кровля/стенка) — "
            "в «Неопределённость», режим и прослои — из данных. Дальше — «Ансамбль»."
        )
        apply.setEnabled(self.cfg is not None)
        apply.clicked.connect(self.apply_calibration)
        sl.addWidget(apply)
        split.addWidget(side)
        split.setSizes([1050, 450])
        lay.addWidget(split, 1)
        return page

    def choose_data(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Данные скважины", str(self.calib_data or "")
        )
        if d:
            self.load_data(d)

    def load_data(self, folder: str) -> Any:
        """Прочитать папку данных и показать сводку и замечания."""
        from saltcav.io.fielddata import load_field_data

        if not folder:
            return None
        self.data_ed.setText(folder)
        try:
            data = load_field_data(folder)
        except (OSError, ValueError) as exc:
            self.data_info.setText(f"<span style='color:#c0392b'>{exc}</span>")
            return None
        self.calib_data = Path(folder)
        parts = [f"<b>{data.name}</b>"]
        if data.operations:
            parts.append(f"режим {len(data.operations)} ступ., {data.duration_days:g} сут")
        if data.brine_day.size:
            parts.append(f"рассол {data.brine_day.size} замеров")
        if data.sonar:
            parts.append("сонар " + ", ".join(f"{s.day:g}" for s in data.sonar) + " сут")
        issues = data.issues(self.cfg)
        text = " · ".join(parts)
        if issues:
            text += "<br>" + "<br>".join(
                f"<span style='color:#a86b00'>⚠ {m}</span>" for m in issues
            )
        self.data_info.setText(text)
        return data

    def run_calibration(self, jobs: int | None = None) -> None:
        cfg = self.validate()
        data = self.load_data(self.data_ed.text())
        if cfg is None or data is None:
            return
        method = self.method_cb.currentData()
        hold = self.holdout_sb.value() or None
        jobs = jobs or os.cpu_count() or 1
        folder = self.calib_data

        def job(report: Callable[[float, str], None]) -> Any:
            from saltcav.calibration import calibrate, read_truth

            return calibrate(cfg, data, method=method, holdout_day=hold, jobs=jobs, progress=report,
                             truth=read_truth(folder))  # fmt: skip

        def done(res: Any) -> None:
            self.calib_fielddata = data
            self._show_calibration(res)

        self._start("Калибровка", job, done)

    def _show_calibration(self, res: Any) -> None:
        self.calib = res
        method = self.method_cb.currentIndex()
        hold = self.holdout_sb.value()
        self._replace_tab(CALIB_TAB, self._calib_page(res), "Калибровка")
        self.method_cb.setCurrentIndex(method)
        self.holdout_sb.setValue(hold)
        self.tabs.setCurrentIndex(CALIB_TAB)

    def apply_calibration(self) -> None:
        cfg = self.validate()
        if cfg is None or self.calib is None:
            return
        new = self.calib.calibrated_config(cfg, getattr(self, "calib_fielddata", None))
        self.load_dict(new.model_dump(mode="json", exclude_defaults=False))
        self.dirty = True
        self._update_title()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            "параметры обновлены по калибровке — дальше «Ансамбль» (F6)", 10000
        )

    def save_calibration(self) -> None:
        if self.calib is None:
            return
        p = self._save_dialog("Калибровка", "saltcav (*.npz)", "calibration.npz")
        if p:
            self.calib.save(p)

    def open_realization(self, index: int | None, mech: bool) -> None:
        if index is None or self.ensemble is None or self.ens_cfg is None:
            return
        cfg, ens = self.ens_cfg, self.ensemble

        def job(report: Callable[[float, str], None]) -> Results:
            from saltcav.leaching import simulate
            from saltcav.stochastic.ensemble import make_realization

            report(0.2, f"реализация №{index}…")
            real = make_realization(cfg, index, ens.n, ens.seed)
            rc = real.cfg.model_copy(update={"name": f"{cfg.name} #{index}"})
            return Results.from_history(simulate(rc, real.props), rc)

        def done(res: Results) -> None:
            if mech:
                self.run_mechanics(shape=res)
            else:
                self._show_results(res)
                self.tabs.setCurrentIndex(1)

        self._start(f"Реализация №{index}", job, done)

    def _show_mech(self, run: Any) -> None:
        from saltcav.gui.mech_view import MechanicsView

        self.mech = run
        view = MechanicsView(run)
        self._replace_tab(3, view, "Геомеханика")
        self.statusBar().showMessage(view.summary_text(), 15000)

    # --- прочее ------------------------------------------------------------------------

    def show_help(self, page: str = "guide", path: str = "") -> Any:
        from saltcav.gui.help import HelpDialog

        if getattr(self, "_help", None) is None:
            self._help = HelpDialog(self)
        dlg = self._help
        if path:
            dlg.show_parameter(path)
        elif page == "parameters":
            dlg.tabs.setCurrentIndex(1)
        else:
            dlg.tabs.setCurrentIndex(0)
        dlg.show()
        dlg.raise_()
        return dlg

    def help_for_focus(self, widget: Any = None) -> None:
        """F1: справка по полю под курсором, иначе — руководство."""
        w = widget if widget is not None else QtWidgets.QApplication.focusWidget()
        while w is not None and not w.property("saltcav_path"):
            w = w.parentWidget()
        path = str(w.property("saltcav_path")).removesuffix("[]") if w is not None else ""
        self.show_help(path=path)

    def about(self) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("О программе")
        box.setIconPixmap(self.windowIcon().pixmap(96, 96))
        box.setText(f"<b>saltcav {__version__}</b>")
        box.setInformativeText(
            "Моделирование скважинного выщелачивания соляных камер: рост камеры, "
            "неопределённость (Монте-Карло), калибровка по данным, геомеханика и "
            "устойчивость, отчёт.<br><br>Автор: <b>Sergei Kizre</b><br>"
            "Работает без интернета: всё считается на этом компьютере."
        )
        box.exec()

    def closeEvent(self, event: Any) -> None:
        if not self._ask_discard():
            event.ignore()
            return
        self.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        if self._viewer is not None:
            self._viewer.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - запуск окна
    args = sys.argv[1:] if argv is None else argv
    from saltcav.gui.icons import app_icon, set_windows_app_id

    set_windows_app_id()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("saltcav")
    app.setOrganizationName("Sergei Kizre")
    app.setWindowIcon(app_icon())
    win = MainWindow(Path(args[0]) if args else None)
    win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
