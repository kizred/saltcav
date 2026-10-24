"""Вкладка геомеханики: разрез массива по годам, выбор полей, сводка во времени."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from saltcav.mechanics.run import MechanicsRun
from saltcav.viz.mechanics_plot import MECH_FIELDS, SURFACE, draw_section, draw_series, field_limits


class MechanicsView(QtWidgets.QWidget):
    """Разрез (слева и справа — выбранные поля), ползунок по годам, графики."""

    def __init__(self, run: MechanicsRun, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.run = run
        self.limits = {k: field_limits(run, k) for k in MECH_FIELDS}
        lay = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.left = QtWidgets.QComboBox()
        self.right = QtWidgets.QComboBox()
        for key, (label, _) in MECH_FIELDS.items():
            self.left.addItem(label, key)
            self.right.addItem(label, key)
        self.left.setCurrentIndex(self.left.findData("fos"))
        self.right.setCurrentIndex(self.right.findData("vm"))
        bar.addWidget(QtWidgets.QLabel("слева"))
        bar.addWidget(self.left)
        bar.addWidget(QtWidgets.QLabel("справа"))
        bar.addWidget(self.right)
        bar.addWidget(QtWidgets.QLabel("контур: смещения ×"))
        self.scale = QtWidgets.QSpinBox()
        self.scale.setRange(0, 500)
        self.scale.setValue(0)
        self.scale.setToolTip("0 — исходный контур; 50 — смещения увеличены в 50 раз")
        bar.addWidget(self.scale)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.fig = Figure(figsize=(12, 6.5), facecolor=SURFACE)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, 1)

        row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, run.times.size - 1)
        self.slider.setValue(run.times.size - 1)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(260)
        row.addWidget(self.slider, 1)
        row.addWidget(self.label)
        lay.addLayout(row)

        for w in (self.left, self.right):
            w.currentIndexChanged.connect(lambda _: self.redraw())
        self.scale.valueChanged.connect(lambda _: self.redraw())
        self.slider.valueChanged.connect(lambda _: self.redraw())
        self.redraw()

    @property
    def index(self) -> int:
        return int(self.slider.value())

    def redraw(self) -> None:
        run, i = self.run, self.index
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, width_ratios=(1.2, 1), hspace=0.45, wspace=0.25)
        ax = self.fig.add_subplot(gs[:, 0])
        ax3 = self.fig.add_subplot(gs[0, 1])
        ax4 = self.fig.add_subplot(gs[1, 1])
        draw_section(
            self.fig, ax, run, i, self.left.currentData(), self.right.currentData(),
            self.limits, float(self.scale.value()),
        )  # fmt: skip
        draw_series(ax3, ax4, run, i)
        stage = run.stage[i]
        label = run.stage_labels[stage] if stage < len(run.stage_labels) else ""
        ax.set_title(f"{run.years[i]:.2f} лет, ступень {stage + 1} {label}", loc="left",
                     fontsize=10)  # fmt: skip
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.2)
        self.label.setText(f"{run.years[i]:.2f} лет  (кадр {i + 1}/{run.times.size})")
        self.canvas.draw_idle()

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=150, facecolor=SURFACE)

    def fields_at(self, key: str) -> np.ndarray:
        from saltcav.viz.mechanics_plot import element_field

        return element_field(self.run, self.index, key)

    def summary_text(self) -> str:
        s = self.run.summary
        k = len(s.years) - 1
        return (
            f"{s.years[k]:.0f} лет: потеря объёма {s.convergence_pct[k]:.2f} %, "
            f"{s.convergence_rate_pct_yr[k]:.3f} %/год; мин. запас у стенки "
            f"{np.nanmin(s.min_fos_wall):.2f} (в конце {s.min_fos_wall[k]:.2f})"
        )

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == QtCore.Qt.Key.Key_Left:
            self.slider.setValue(self.index - 1)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self.slider.setValue(self.index + 1)
        else:
            super().keyPressEvent(event)
