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
)
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
        self.resize(1500, 950)

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
        for a in (self.run_act, self.ens_act, self.mech_act):
            tb.addAction(a)
        mh = self.menuBar().addMenu("Справка")
        act("Справка по полю", lambda: self.help_for_focus(), "F1", menu=mh)
        act("Руководство", lambda: self.show_help(), menu=mh)
        act("Справочник параметров", lambda: self.show_help("parameters"), menu=mh)
        mh.addSeparator()
        act("О программе", self.about, menu=mh)

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
        sl.addWidget(b1)
        sl.addWidget(b2)
        self.ens_table = table

        def chosen() -> int | None:
            r = table.currentRow()
            return None if r < 0 else int(table.item(r, 0).data(QtCore.Qt.ItemDataRole.DisplayRole))

        b1.clicked.connect(lambda: self.open_realization(chosen(), mech=False))
        b2.clicked.connect(lambda: self.open_realization(chosen(), mech=True))
        b1.setEnabled(cfg is not None)
        b2.setEnabled(cfg is not None)
        w.addWidget(side)
        w.setSizes([900, 500])
        self._replace_tab(2, w, "Ансамбль")

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
        QtWidgets.QMessageBox.information(
            self, "saltcav",
            f"saltcav {__version__}\nМодель скважинного выщелачивания соляных камер:\n"
            "выщелачивание, Монте-Карло, геомеханика.\nПараметры — в YAML; подсказки — "
            "у каждого поля.",
        )  # fmt: skip

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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow(Path(args[0]) if args else None)
    win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
