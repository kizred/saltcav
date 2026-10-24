"""Консоль Python в окне (как макросы в ANSYS APDL, но на Python).

Код выполняется в фоне, окно не замирает. В пространстве имён:

- ``sc`` — :mod:`saltcav.api` (load, set, run, ensemble, mechanics, stability,
  calibrate, plot, table, sweep, save);
- ``cfg`` — конфигурация из формы параметров на момент запуска;
- ``apply(cfg)`` — загрузить конфигурацию в форму;
- ``show(obj)`` — показать результат на своей вкладке (выщелачивание, ансамбль,
  геомеханика, устойчивость, калибровка) или фигуру matplotlib в отдельном окне;
- ``progress(доля, текст)`` — полоса хода; через неё же работает «Отмена»;
- ``np`` — NumPy, ``window`` — само окно (для опытных).

Ctrl+Enter — выполнить; одно выражение печатает своё значение, как в Python.
Переменные сохраняются между запусками. «Выполнить скрипт…» — файл .py в том же
пространстве имён.
"""

from __future__ import annotations

import contextlib
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

HELP = __doc__.split("\n\n", 1)[1] if __doc__ else ""
MONO = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)


class _Stream(QtCore.QObject):
    """Поток вывода для print в фоновом потоке → текст в консоли (через сигнал)."""

    text = QtCore.Signal(str)

    def write(self, s: str) -> int:
        if s:
            self.text.emit(s)
        return len(s)

    def flush(self) -> None:
        pass


class _Input(QtWidgets.QPlainTextEdit):
    run = QtCore.Signal()
    history_step = QtCore.Signal(int)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        mods = e.modifiers()
        enter = e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
        if enter and mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.run.emit()
            return
        if mods & QtCore.Qt.KeyboardModifier.ControlModifier and e.key() in (
            QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down,
        ):  # fmt: skip
            self.history_step.emit(-1 if e.key() == QtCore.Qt.Key.Key_Up else 1)
            return
        super().keyPressEvent(e)


class Console(QtWidgets.QWidget):
    """Консоль: вывод сверху, ввод снизу. ``window`` — главное окно saltcav."""

    def __init__(self, window: Any) -> None:
        super().__init__()
        self.window = window
        self.history: list[str] = []
        self._hpos = 0
        self._pending: list[Callable[[], None]] = []
        self.busy = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(MONO)
        self.out.setMaximumBlockCount(20_000)
        self.inp = _Input()
        self.inp.setFont(MONO)
        self.inp.setPlaceholderText(
            "Python: res = sc.run(cfg); print(sc.table(res)); show(res)   — Ctrl+Enter"
        )
        self.inp.setTabChangesFocus(False)
        self.inp.setMaximumHeight(140)
        self.inp.run.connect(self.run_input)
        self.inp.history_step.connect(self._history)
        bar = QtWidgets.QHBoxLayout()
        for text, slot in (
            ("Выполнить (Ctrl+Enter)", self.run_input),
            ("Скрипт…", self.window.run_script_dialog),
            ("Очистить", self.out.clear),
            ("Справка", lambda: self.write(HELP + "\n")),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addWidget(self.out, 1)
        lay.addWidget(self.inp)
        lay.addLayout(bar)
        self.stream = _Stream()
        self.stream.text.connect(self.write, QtCore.Qt.ConnectionType.QueuedConnection)
        self.ns: dict[str, Any] = {"__name__": "__console__", "np": np}
        self.write("Консоль saltcav. sc — API, cfg — параметры из формы, show(obj) — показать "
                   "результат, apply(cfg) — в форму. «Справка» — подробнее.\n")  # fmt: skip

    # --- вывод ------------------------------------------------------------------------

    @QtCore.Slot(str)
    def write(self, s: str) -> None:
        cur = self.out.textCursor()
        cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cur.insertText(s)
        self.out.setTextCursor(cur)
        self.out.ensureCursorVisible()

    def _history(self, step: int) -> None:
        if not self.history:
            return
        self._hpos = max(0, min(len(self.history), self._hpos + step))
        text = self.history[self._hpos] if self._hpos < len(self.history) else ""
        self.inp.setPlainText(text)

    # --- выполнение ---------------------------------------------------------------------

    def _namespace(self, report: Callable[[float, str], None]) -> dict[str, Any]:
        import saltcav.api as sc

        ns = self.ns
        ns.update(sc=sc, cfg=self.window.cfg, window=self.window, progress=report)
        ns["show"] = lambda obj: self._pending.append(lambda: self.window.show_object(obj))
        ns["apply"] = lambda c: self._pending.append(lambda: self.window.apply_config(c))
        return ns

    def run_input(self) -> None:
        src = self.inp.toPlainText()
        if not src.strip():
            return
        if self.run_source(src, "<консоль>", echo=True):
            self.history.append(src)
            self._hpos = len(self.history)
            self.inp.clear()

    def run_source(self, src: str, name: str, echo: bool = False,
                   path: Path | None = None) -> bool:  # fmt: skip
        """Выполнить код в фоне; False — если уже что-то выполняется."""
        if self.busy or self.window._thread is not None:
            self.write("— занято: дождитесь конца расчёта\n")
            return False
        if echo:
            self.write("".join(f"{'>>> ' if i == 0 else '... '}{ln}\n"
                               for i, ln in enumerate(src.rstrip().splitlines())))  # fmt: skip
        self.window.validate()
        self.busy = True
        self._pending = []
        stream = self.stream

        def job(report: Callable[[float, str], None]) -> str:
            ns = self._namespace(report)
            ns["__name__"] = "__console__"
            if path is not None:  # скрипт — как при запуске python скрипт.py
                ns["__file__"] = str(path)
                ns["__name__"] = "__main__"
                sys.path.insert(0, str(path.parent))
            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):  # type: ignore[type-var]
                    try:
                        code = compile(src, name, "eval")
                    except SyntaxError:
                        exec(compile(src, name, "exec"), ns)
                    else:
                        value = eval(code, ns)
                        if value is not None:
                            ns["_"] = value
                            print(repr(value))
            except Exception as exc:  # ошибка в коде пользователя — в консоль, не в окно
                if type(exc).__name__ == "Cancelled":
                    raise
                tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
                return "".join(tb[1:] if len(tb) > 1 else tb)  # без кадра консоли
            finally:
                if path is not None and sys.path and sys.path[0] == str(path.parent):
                    sys.path.pop(0)
            return ""

        def done(err: str) -> None:
            self.busy = False
            if err:
                self.write(err)
            for action in self._pending:
                try:
                    action()
                except Exception as exc:
                    self.write(f"show/apply: {type(exc).__name__}: {exc}\n")
            self._pending = []

        self.window._start(f"Консоль: {name}", job, done)
        if self.window._worker is not None:  # отмена или сбой — busy снимается в _finish
            self.window._worker.failed.connect(lambda _m: setattr(self, "busy", False))
        return True
