"""Просмотрщик результатов по шагам (аналог General Postproc / Results Viewer в ANSYS).

Окно: слева 3D-камера с раскраской по выбранному полю, справа разрез через ось,
профиль насыщения рассола и график по времени; внизу ползунок шагов с
воспроизведением. Журнал событий (обрушения карнизов, подъёмы башмака, смены
ступеней) — двойной щелчок переходит к шагу события; щелчок по графику времени —
к ближайшему шагу.

Клавиши: ← → — шаг, Home/End — начало/конец, пробел — пуск/пауза.

Зависимости: ``pip install -e ".[gui]"`` (PySide6, pyvistaqt, imageio).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from saltcav.io.results import Results
from saltcav.viz.frames import FIELDS, available_fields, display_window, section_profile

SERIES: dict[str, tuple[str, str]] = {
    "volume": ("Объём камеры", "тыс. м³"),
    "outlet": ("Концентрация на выходе", "% насыщения"),
    "produced": ("Вынесено соли", "тыс. т"),
    "floor": ("Верх насыпи", "м"),
    "max_radius": ("Макс. радиус", "м"),
}

INK, INK2, GRID, LINE, BAND, ACCENT = (
    "#0b0b0b", "#52514e", "#e4e3df", "#1c5cab", "#b7d3f6", "#c0392b",
)  # fmt: skip


def series_values(res: Results, key: str) -> np.ndarray:
    """Величина по моментам записи для графика времени."""
    if key == "volume":
        return np.asarray(res.volume / 1e3)
    if key == "outlet":
        if res.outlet_concentration is None:
            return np.full(res.nt, np.nan)
        return np.asarray(100.0 * res.outlet_concentration / res.c_sat)
    if key == "produced":
        return np.asarray(res.produced_salt / 1e6)
    if key == "floor":
        return np.asarray(res.floor_depth)
    if key == "max_radius":
        return np.asarray(res.radius.max(axis=(1, 2)))
    raise KeyError(key)


def _qt() -> tuple[Any, Any, Any]:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError('для окна просмотра нужен PySide6: pip install -e ".[gui]"') from exc
    return QtCore, QtGui, QtWidgets


def build_window(
    res: Results,
    source: Path | None = None,
    off_screen: bool = False,
    start: int | None = None,
    field: str = "radius",
    embedded: bool = False,
) -> Any:
    """Создать окно просмотрщика (QApplication должно существовать).

    ``embedded`` — для вкладки главного окна: виджет без собственной рамки.
    """
    QtCore, QtGui, QtWidgets = _qt()
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from pyvistaqt import QtInteractor

    from saltcav.viz.scene import CavernScene, export_animation, export_paraview

    class ViewerWindow(QtWidgets.QMainWindow):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.res = res
            self.source = source
            self.setWindowTitle(f"saltcav — {res.name}")
            if embedded:
                self.setWindowFlags(QtCore.Qt.WindowType.Widget)
            if not embedded:
                self.resize(1500, 920)

            # --- 3D ---------------------------------------------------------------
            self.vtk = QtInteractor(self, off_screen=off_screen)
            self.scene = CavernScene(self.vtk, res, field)

            # --- графики ----------------------------------------------------------
            self.fig = Figure(figsize=(6, 7), facecolor="#fcfcfb", layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            gs = self.fig.add_gridspec(2, 2, height_ratios=(1.7, 1), width_ratios=(1.6, 1))
            self.ax_sec = self.fig.add_subplot(gs[0, 0])
            self.ax_c = self.fig.add_subplot(gs[0, 1], sharey=self.ax_sec)
            self.ax_t = self.fig.add_subplot(gs[1, :])
            self.canvas.mpl_connect("button_press_event", self._on_click)

            self.events = res.events()
            self.ev_list = QtWidgets.QListWidget()
            for ev in self.events:
                mark = {"collapse": "▼", "shoe": "↑", "stage": "■"}[ev.kind]
                depth = f", {ev.depth_m:.1f} м" if np.isfinite(ev.depth_m) else ""
                self.ev_list.addItem(f"{mark} {ev.t / 86_400:7.1f} сут  {ev.text}{depth}")
            if not self.events:
                self.ev_list.addItem("событий нет")
            self.ev_list.itemDoubleClicked.connect(self._on_event)

            right = QtWidgets.QWidget()
            rl = QtWidgets.QVBoxLayout(right)
            rl.setContentsMargins(0, 0, 0, 0)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("разрез по сектору"))
            self.sector = QtWidgets.QComboBox()
            for j, phi in enumerate(np.rad2deg(res.phi)):
                self.sector.addItem(f"{j + 1}: φ = {phi:.0f}°")
            self.sector.setEnabled(res.nphi > 1)
            self.sector.currentIndexChanged.connect(lambda _: self._draw_plots())
            row.addWidget(self.sector)
            row.addSpacing(16)
            row.addWidget(QtWidgets.QLabel("график"))
            self.series = QtWidgets.QComboBox()
            for key, (label, unit) in SERIES.items():
                self.series.addItem(f"{label}, {unit}", key)
            self.series.currentIndexChanged.connect(lambda _: self._draw_plots())
            row.addWidget(self.series)
            row.addStretch(1)
            rl.addLayout(row)
            rl.addWidget(self.canvas, 4)
            rl.addWidget(QtWidgets.QLabel("События (двойной щелчок — перейти):"))
            rl.addWidget(self.ev_list, 1)

            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            split.addWidget(self.vtk.interactor)
            split.addWidget(right)
            split.setSizes([820, 680])

            # --- управление -------------------------------------------------------
            bar = QtWidgets.QHBoxLayout()
            style = self.style()
            sp = QtWidgets.QStyle.StandardPixmap

            def button(icon: Any, tip: str, slot: Any) -> Any:
                b = QtWidgets.QToolButton()
                b.setIcon(style.standardIcon(icon))
                b.setToolTip(tip)
                b.clicked.connect(slot)
                bar.addWidget(b)
                return b

            button(sp.SP_MediaSkipBackward, "в начало (Home)", lambda: self.go(0))
            button(sp.SP_MediaSeekBackward, "шаг назад (←)", lambda: self.go(self.i - 1))
            self.play_btn = button(sp.SP_MediaPlay, "пуск/пауза (пробел)", self.toggle_play)
            button(sp.SP_MediaSeekForward, "шаг вперёд (→)", lambda: self.go(self.i + 1))
            button(sp.SP_MediaSkipForward, "в конец (End)", lambda: self.go(res.nt - 1))

            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setRange(0, res.nt - 1)
            self.slider.valueChanged.connect(self.go)
            bar.addWidget(self.slider, 1)
            self.label = QtWidgets.QLabel()
            self.label.setMinimumWidth(190)
            bar.addWidget(self.label)

            bar.addWidget(QtWidgets.QLabel("кадров/с"))
            self.fps = QtWidgets.QSpinBox()
            self.fps.setRange(1, 30)
            self.fps.setValue(6)
            self.fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
            bar.addWidget(self.fps)
            bar.addSpacing(12)
            bar.addWidget(QtWidgets.QLabel("цвет"))
            self.field = QtWidgets.QComboBox()
            for key in available_fields(res):
                self.field.addItem(f"{FIELDS[key].label}, {FIELDS[key].unit}", key)
            self.field.setCurrentIndex(max(self.field.findData(field), 0))
            self.field.currentIndexChanged.connect(
                lambda _: self.scene.set_field(self.field.currentData())
            )
            bar.addWidget(self.field)
            self.clip = QtWidgets.QCheckBox("разрез пополам")
            self.clip.toggled.connect(self.scene.set_clip)
            bar.addWidget(self.clip)

            central = QtWidgets.QWidget()
            cl = QtWidgets.QVBoxLayout(central)
            cl.addWidget(split, 1)
            cl.addLayout(bar)
            self.setCentralWidget(central)

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(int(1000 / self.fps.value()))
            self.timer.timeout.connect(self._tick)

            self._menus()
            for key, slot in (
                (QtCore.Qt.Key.Key_Left, lambda: self.go(self.i - 1)),
                (QtCore.Qt.Key.Key_Right, lambda: self.go(self.i + 1)),
                (QtCore.Qt.Key.Key_Home, lambda: self.go(0)),
                (QtCore.Qt.Key.Key_End, lambda: self.go(res.nt - 1)),
                (QtCore.Qt.Key.Key_Space, self.toggle_play),
            ):
                sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
                sc.activated.connect(slot)

            self.i = -1
            self.go(res.nt - 1 if start is None else start)
            self.statusBar().showMessage(
                f"{res.nt} кадров, {res.n_steps} шагов расчёта, сетка "
                f"{res.nz}×{res.nphi}" + (f"   {source}" if source else "")
            )

        # --- меню -----------------------------------------------------------------

        def _menus(self) -> None:
            m = self.menuBar().addMenu("Файл")
            m.addAction("Скриншот 3D…", self.save_screenshot)
            m.addAction("Картинка графиков…", self.save_plots)
            m.addAction("Анимация (GIF/MP4)…", self.save_animation)
            m.addAction("Экспорт в ParaView (.pvd)…", self.save_paraview)
            m.addAction("Сохранить результаты (.npz)…", self.save_results)
            m.addSeparator()
            m.addAction("Выход", self.close)
            v = self.menuBar().addMenu("Вид")
            v.addAction("Сбросить камеру", self._reset_camera)
            v.addAction("Вид сбоку", lambda: self._view("xz"))
            v.addAction("Вид сверху", lambda: self._view("xy"))

        def _reset_camera(self) -> None:
            self.vtk.reset_camera()
            self.vtk.render()

        def _view(self, plane: str) -> None:
            self.vtk.camera_position = plane
            self.vtk.reset_camera()
            self.vtk.render()

        def _ask(self, title: str, filt: str, default: str) -> str:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, default, filt)
            return str(path)

        def save_screenshot(self) -> None:
            p = self._ask("Скриншот", "PNG (*.png)", f"{res.name}_{self.i:04d}.png")
            if p:
                self.scene.screenshot(p)
                self.statusBar().showMessage(f"сохранено: {p}")

        def save_plots(self) -> None:
            p = self._ask(
                "Графики", "PNG (*.png);;PDF (*.pdf)", f"{res.name}_{self.i:04d}_plots.png"
            )
            if p:
                self.fig.savefig(p, dpi=150, facecolor="#fcfcfb")
                self.statusBar().showMessage(f"сохранено: {p}")

        def save_animation(self) -> None:
            p = self._ask("Анимация", "GIF (*.gif);;MP4 (*.mp4)", f"{res.name}.gif")
            if not p:
                return
            with _busy(QtWidgets, QtCore):
                try:
                    n = export_animation(res, p, self.field.currentData(), self.fps.value())
                except (ModuleNotFoundError, RuntimeError) as exc:
                    QtWidgets.QMessageBox.warning(self, "Анимация", str(exc))
                    return
            self.statusBar().showMessage(f"анимация: {p}, {n} кадров")

        def save_paraview(self) -> None:
            p = self._ask("ParaView", "ParaView (*.pvd)", f"{res.name}.pvd")
            if p:
                with _busy(QtWidgets, QtCore):
                    n = export_paraview(res, p)
                self.statusBar().showMessage(f"ParaView: {p}, {n} шагов")

        def save_results(self) -> None:
            p = self._ask("Результаты", "saltcav (*.npz)", f"{res.name}.npz")
            if p:
                res.save(p)
                self.statusBar().showMessage(f"сохранено: {p}")

        # --- шаги -----------------------------------------------------------------

        def go(self, i: int) -> None:
            i = int(np.clip(i, 0, res.nt - 1))
            if i == self.i:
                return
            self.i = i
            if self.slider.value() != i:
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
            self.scene.show_frame(i)
            stage = "" if res.stage_index is None else f", ступень {int(res.stage_index[i]) + 1}"
            self.label.setText(f"сутки {res.days[i]:.0f}  ({i + 1}/{res.nt}{stage})")
            self._draw_plots()

        def toggle_play(self) -> None:
            if self.timer.isActive():
                self.timer.stop()
                self.play_btn.setIcon(self.style().standardIcon(sp_play))
            else:
                if self.i >= res.nt - 1:
                    self.go(0)
                self.timer.start()
                self.play_btn.setIcon(self.style().standardIcon(sp_pause))

        def _tick(self) -> None:
            if self.i >= res.nt - 1:
                self.toggle_play()
            else:
                self.go(self.i + 1)

        def _on_event(self, item: Any) -> None:
            row = self.ev_list.row(item)
            if row < len(self.events):
                t = self.events[row].t
                # первый сохранённый момент после события — там оно уже видно
                self.go(int(np.searchsorted(res.times, t - 1e-6)))

        def _on_click(self, event: Any) -> None:
            if event.inaxes is self.ax_t and event.xdata is not None:
                self.go(res.frame_at(float(event.xdata)))

        # --- графики --------------------------------------------------------------

        def _draw_plots(self) -> None:
            i = self.i
            if i < 0:
                return
            keep = display_window(res)
            z_lo, z_hi = float(res.z_edges[keep][0]), float(res.z_edges[keep][-1])

            ax = self.ax_sec
            ax.clear()
            j = max(self.sector.currentIndex(), 0)
            z, r1, r2 = section_profile(res, i, j)
            _, f1, f2 = section_profile(res, res.nt - 1, j)
            for top, th in res.layers:
                ax.axhspan(top, top + th, color="#eda100", alpha=0.3, lw=0)
            ax.fill_betweenx(z, -r2, r1, color=BAND, lw=0)
            ax.plot(r1, z, color=LINE, lw=1.8)
            ax.plot(-r2, z, color=LINE, lw=1.8)
            ax.plot(f1, z, color=INK2, lw=0.8, ls="--")
            ax.plot(-f2, z, color=INK2, lw=0.8, ls="--", label="итоговая форма")
            if np.isfinite(res.floor_depth[i]) and res.floor_depth[i] < z[-1]:
                ax.axhline(res.floor_depth[i], color="#8a8578", lw=1.5, label="верх насыпи")
            if res.blanket_depth is not None:
                ax.axhline(res.blanket_depth[i], color="#eda100", lw=1.5, label="нерастворитель")
            for depth, color, lab in (
                (res.shoe_in_depth, LINE, "башмак закачки"),
                (res.shoe_out_depth, ACCENT, "башмак отбора"),
            ):
                if depth is not None:
                    ax.plot([0], [depth[i]], marker="v", color=color, ms=8, ls="", label=lab)
            r_lim = float(np.max(res.radius[:, keep])) * 1.08 + 0.5
            ax.set_xlim(-r_lim, r_lim)
            ax.set_ylim(z_hi + 1, z_lo - 1)
            phi = np.rad2deg(res.phi[j])
            ax.set_xlabel(f"← φ = {(phi + 180) % 360:.0f}°   R, м   φ = {phi:.0f}° →")
            ax.set_ylabel("глубина, м")
            ax.set_title("Разрез через ось", loc="left", fontsize=10, color=INK)
            handles, labels = ax.get_legend_handles_labels()
            ax.grid(color=GRID, lw=0.5)

            ax = self.ax_c
            ax.clear()
            zc = 0.5 * (res.z_edges[:-1] + res.z_edges[1:])
            sat = 100.0 * res.concentration[i] / res.c_sat
            ax.plot(sat, zc, color=LINE, lw=1.6)
            ax.set_xlim(min(80.0, float(np.nanmin(sat[keep[:-1]])) - 2), 101)
            ax.set_xlabel("насыщение, %")
            ax.set_title("Рассол", loc="left", fontsize=10, color=INK)
            ax.tick_params(labelleft=False)
            # легенда разреза — здесь: у подошвы рассол недонасыщен, правый низ свободен
            ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9,
                      edgecolor=GRID, loc="lower right")  # fmt: skip
            ax.grid(color=GRID, lw=0.5)

            ax = self.ax_t
            ax.clear()
            key = self.series.currentData() or "volume"
            y = series_values(res, key)
            ax.plot(res.days, y, color=LINE, lw=1.6)
            ax.axvline(res.days[i], color=ACCENT, lw=1)
            ax.plot([res.days[i]], [y[i]], "o", color=ACCENT, ms=5)
            for ev in self.events:
                if ev.kind == "stage":
                    ax.axvline(ev.t / 86_400, color=INK2, lw=0.6, ls=":")
            for t in res.collapses[:, 0] / 86_400 if res.collapses.size else []:
                ax.plot([t], [np.interp(t, res.days, y)], "v", color="#eda100", ms=5)
            label, unit = SERIES[key]
            ax.set_title(f"{label}: {y[i]:.3g} {unit}", loc="left", fontsize=10, color=INK)
            ax.set_xlabel("сутки (щелчок — перейти)")
            if key == "floor":
                ax.invert_yaxis()
            ax.grid(color=GRID, lw=0.5)
            self.canvas.draw_idle()

        def closeEvent(self, event: Any) -> None:
            self.timer.stop()
            self.vtk.close()
            super().closeEvent(event)

    sp_play = QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
    sp_pause = QtWidgets.QStyle.StandardPixmap.SP_MediaPause
    return ViewerWindow()


class _busy:
    """Курсор ожидания на время долгой операции."""

    def __init__(self, qtwidgets: Any, qtcore: Any) -> None:
        self.w, self.c = qtwidgets, qtcore

    def __enter__(self) -> None:
        self.w.QApplication.setOverrideCursor(self.c.Qt.CursorShape.WaitCursor)

    def __exit__(self, *exc: object) -> None:
        self.w.QApplication.restoreOverrideCursor()


def run_viewer(  # pragma: no cover - окно
    res: Results, source: Path | None = None, start: int | None = None, field: str = "radius"
) -> int:
    """Открыть окно и ждать его закрытия."""
    _, _, QtWidgets = _qt()
    from saltcav.gui.icons import app_icon, set_windows_app_id

    set_windows_app_id()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    win = build_window(res, source, start=start, field=field)
    win.show()
    return int(app.exec())
