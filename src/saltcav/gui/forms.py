"""Формы ввода, построенные по модели конфигурации (pydantic).

Каждое поле модели превращается в редактор по типу: число — строка ввода (принимает
и запятую, и запись 3e-5), перечисление и поле с шаблоном ``^(a|b)$`` — список,
логическое — флажок, вложенная модель — группа, необязательная модель — группа с
флажком, список моделей — таблица. Подписи — описания полей из схемы, подсказка —
имя поля, ограничения и значение по умолчанию.

Проверяет не форма, а сама модель: форма собирает словарь, ``RunConfig`` его
проверяет, ошибки по пути поля подсвечиваются в форме. Поэтому окно, YAML и
``saltcav check`` не расходятся.
"""

from __future__ import annotations

import re
import textwrap
import types
from collections.abc import Callable
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PySide6 import QtCore, QtGui, QtWidgets

Path = tuple[str | int, ...]
ERROR_STYLE = "background: #fbe3e1;"
_PATTERN_CHOICES = re.compile(r"^\^\(([\w|]+)\)\$$")


# --- разбор типов ---------------------------------------------------------------------


def unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """Тип без None и признак «может быть пустым»."""
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def choices_of(info: FieldInfo, ann: Any) -> list[str] | None:
    """Допустимые значения: перечисление или шаблон вида ^(a|b|c)$."""
    if isinstance(ann, type) and issubclass(ann, Enum):
        return [str(e.value) for e in ann]
    for m in info.metadata:
        pattern = getattr(m, "pattern", None)
        if pattern:
            hit = _PATTERN_CHOICES.match(pattern)
            if hit:
                return hit.group(1).split("|")
    return None


def constraints_text(info: FieldInfo) -> str:
    parts = []
    for m in info.metadata:
        for attr, sign in (("gt", ">"), ("ge", "≥"), ("lt", "<"), ("le", "≤")):
            v = getattr(m, attr, None)
            if v is not None:
                parts.append(f"{sign} {format_number(v)}")
        if getattr(m, "min_length", None):
            parts.append(f"не меньше {m.min_length} строк")
    return ", ".join(parts)


def format_number(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        a = abs(v)
        if a == 0 or 1e-3 <= a < 1e7:
            return f"{v:.10g}"
        return f"{v:.6g}"
    return str(v)


def parse_number(text: str, integer: bool) -> Any:
    """Число из строки; запятая — десятичный разделитель. Не число — строка как есть
    (модель сообщит об ошибке)."""
    t = text.strip().replace(" ", "").replace(",", ".")
    try:
        x = float(t)
    except ValueError:
        return text
    if integer and x.is_integer():
        return int(x)
    return x


def default_of(info: FieldInfo) -> Any:
    if info.default_factory is not None:
        val = info.default_factory()  # type: ignore[call-arg]
    else:
        val = info.default
    if isinstance(val, BaseModel):
        return val.model_dump(mode="json")
    if isinstance(val, list):
        return [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in val]
    if isinstance(val, Enum):
        return val.value
    return val


def tooltip(name: str, info: FieldInfo) -> str:
    lines = [f"<b>{name}</b>"]
    if info.description:
        lines.append(info.description)
    c = constraints_text(info)
    if c:
        lines.append(f"допустимо: {c}")
    if not info.is_required():
        d = default_of(info)
        if not isinstance(d, list | dict):
            lines.append(f"по умолчанию: {format_number(d) if d is not None else 'пусто'}")
    return "<br>".join(lines)


def label_of(name: str, info: FieldInfo) -> str:
    return info.description or name


# --- редакторы скалярных полей ----------------------------------------------------------


class ScalarEditor:
    """Редактор одного значения: число, строка, выбор, флажок, пара чисел."""

    def __init__(self, name: str, info: FieldInfo, on_change: Callable[[], None]) -> None:
        self.name = name
        self.info = info
        ann, self.optional = unwrap_optional(info.annotation)
        self.ann = ann
        self.choices = choices_of(info, ann)
        self.required = info.is_required()
        tip = tooltip(name, info)
        if self.choices is not None:
            w: QtWidgets.QWidget = QtWidgets.QComboBox()
            if self.optional:
                w.addItem("")
            w.addItems(self.choices)
            w.currentIndexChanged.connect(lambda _: on_change())
        elif ann is bool:
            w = QtWidgets.QCheckBox()
            w.toggled.connect(lambda _: on_change())
        elif get_origin(ann) is tuple:
            w = QtWidgets.QLineEdit()
            w.setPlaceholderText("нижняя; верхняя — или пусто")
            w.textChanged.connect(lambda _: on_change())
        else:
            w = QtWidgets.QLineEdit()
            d = None if self.required else default_of(info)
            if d is not None:
                w.setPlaceholderText(format_number(d))
            elif self.optional:
                w.setPlaceholderText("пусто")
            w.textChanged.connect(lambda _: on_change())
        w.setToolTip(tip)
        self.widget = w

    def set(self, value: Any) -> None:
        w = self.widget
        w.blockSignals(True)
        try:
            if isinstance(w, QtWidgets.QComboBox):
                text = (
                    "" if value is None else str(value.value if isinstance(value, Enum) else value)
                )
                i = w.findText(text)
                w.setCurrentIndex(max(i, 0))
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(value))
            elif get_origin(self.ann) is tuple:
                w.setText("" if value is None else "; ".join(format_number(x) for x in value))
            else:
                w.setText("" if value is None else format_number(value))
        finally:
            w.blockSignals(False)

    def get(self) -> tuple[bool, Any]:
        """(задано ли, значение). Пустое необязательное поле не задаётся — берётся
        значение по умолчанию модели."""
        w = self.widget
        if isinstance(w, QtWidgets.QComboBox):
            t = w.currentText()
            return (t != "", t if t != "" else None)
        if isinstance(w, QtWidgets.QCheckBox):
            return True, w.isChecked()
        text = w.text().strip()
        if text == "":  # пусто — значение по умолчанию модели (обязательное — ошибка)
            return (True, None) if self.required else (False, None)
        if get_origin(self.ann) is tuple:
            parts = [p for p in re.split(r"[;\s]+", text) if p]
            return True, [parse_number(p, False) for p in parts]
        if self.ann in (int, float):
            return True, parse_number(text, self.ann is int)
        return True, text

    def set_error(self, msg: str | None) -> None:
        self.widget.setStyleSheet(ERROR_STYLE if msg else "")
        base = tooltip(self.name, self.info)
        self.widget.setToolTip(
            base + (f"<br><span style='color:#c0392b'>{msg}</span>" if msg else "")
        )


# --- таблица для списка моделей ---------------------------------------------------------


class ListTable(QtWidgets.QWidget):
    """Список моделей (ступени, прослои) — таблица с добавлением и удалением строк."""

    def __init__(
        self, name: str, info: FieldInfo, item: type[BaseModel], on_change: Callable[[], None]
    ) -> None:
        super().__init__()
        self.name, self.info, self.item = name, info, item
        self.on_change = on_change
        self.fields = list(item.model_fields.items())
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel(f"<b>{label_of(name, info)}</b>")
        title.setToolTip(tooltip(name, info))
        lay.addWidget(title)
        self.table = QtWidgets.QTableWidget(0, len(self.fields))
        heads = [textwrap.fill(label_of(f, i), 18) for f, i in self.fields]
        self.table.setHorizontalHeaderLabels(heads)
        rows = max(h.count("\n") + 1 for h in heads) if heads else 1
        self.table.horizontalHeader().setMinimumHeight(18 * rows + 8)
        for j, (fname, finfo) in enumerate(self.fields):
            hi = self.table.horizontalHeaderItem(j)
            hi.setToolTip(tooltip(fname, finfo))
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.itemChanged.connect(lambda _: self.on_change())
        lay.addWidget(self.table)
        bar = QtWidgets.QHBoxLayout()
        for text, slot in (
            ("+ строка", self.add_row),
            ("копировать", self.copy_row),
            ("удалить", self.remove_row),
            ("↑", lambda: self.move(-1)),
            ("↓", lambda: self.move(1)),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)

    def _cell_widget(self, row: int, j: int) -> None:
        _fname, finfo = self.fields[j]
        ann, optional = unwrap_optional(finfo.annotation)
        ch = choices_of(finfo, ann)
        if ch is not None:
            cb = QtWidgets.QComboBox()
            if optional:
                cb.addItem("")
            cb.addItems(ch)
            cb.currentIndexChanged.connect(lambda _: self.on_change())
            self.table.setCellWidget(row, j, cb)
        elif ann is bool:
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(row, j, it)
        else:
            self.table.setItem(row, j, QtWidgets.QTableWidgetItem(""))

    def _set_row(self, row: int, values: dict[str, Any]) -> None:
        for j, (fname, finfo) in enumerate(self.fields):
            v = values.get(fname, default_of(finfo) if not finfo.is_required() else None)
            w = self.table.cellWidget(row, j)
            if isinstance(w, QtWidgets.QComboBox):
                w.blockSignals(True)
                w.setCurrentIndex(max(w.findText("" if v is None else str(v)), 0))
                w.blockSignals(False)
            else:
                it = self.table.item(row, j)
                if unwrap_optional(finfo.annotation)[0] is bool:
                    it.setCheckState(
                        QtCore.Qt.CheckState.Checked if v else QtCore.Qt.CheckState.Unchecked
                    )
                else:
                    it.setText("" if v is None else format_number(v))

    def set(self, rows: list[dict[str, Any]]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for values in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for j in range(len(self.fields)):
                self._cell_widget(r, j)
            self._set_row(r, values)
        self.table.blockSignals(False)
        head = self.table.horizontalHeader().minimumHeight()
        self.table.setFixedHeight(min(head + 26 * (self.table.rowCount() + 1) + 6, 420))

    def get(self) -> list[dict[str, Any]]:
        out = []
        for r in range(self.table.rowCount()):
            row: dict[str, Any] = {}
            for j, (fname, finfo) in enumerate(self.fields):
                ann, _optional = unwrap_optional(finfo.annotation)
                w = self.table.cellWidget(r, j)
                if isinstance(w, QtWidgets.QComboBox):
                    t = w.currentText()
                    if t:
                        row[fname] = t
                    continue
                it = self.table.item(r, j)
                if ann is bool:
                    row[fname] = it.checkState() == QtCore.Qt.CheckState.Checked
                    continue
                text = it.text().strip()
                if text == "":
                    if finfo.is_required():
                        row[fname] = None
                    continue
                row[fname] = parse_number(text, ann is int) if ann in (int, float) else text
            out.append(row)
        return out

    def add_row(self) -> None:
        rows = self.get()
        rows.append(dict(rows[-1]) if rows else {})
        self.set(rows)
        self.on_change()

    def copy_row(self) -> None:
        rows = self.get()
        r = self.table.currentRow()
        if 0 <= r < len(rows):
            rows.insert(r + 1, dict(rows[r]))
            self.set(rows)
            self.table.setCurrentCell(r + 1, 0)
            self.on_change()

    def remove_row(self) -> None:
        rows = self.get()
        r = self.table.currentRow()
        if not 0 <= r < len(rows):
            r = len(rows) - 1  # строка не выбрана — последняя
        if r >= 0:
            rows.pop(r)
            self.set(rows)
            self.table.setCurrentCell(min(r, len(rows) - 1), 0)
            self.on_change()

    def move(self, d: int) -> None:
        rows = self.get()
        r = self.table.currentRow()
        if 0 <= r < len(rows) and 0 <= r + d < len(rows):
            rows[r], rows[r + d] = rows[r + d], rows[r]
            self.set(rows)
            self.table.setCurrentCell(r + d, 0)
            self.on_change()

    def clear_errors(self) -> None:
        for r in range(self.table.rowCount()):
            for j in range(len(self.fields)):
                it = self.table.item(r, j)
                if it is not None:
                    it.setBackground(QtGui.QBrush())
                w = self.table.cellWidget(r, j)
                if w is not None:
                    w.setStyleSheet("")

    def set_cell_error(self, row: int, field: str | None, msg: str) -> None:
        cols = (
            range(len(self.fields))
            if field is None
            else [j for j, (f, _) in enumerate(self.fields) if f == field]
        )
        for j in cols:
            if 0 <= row < self.table.rowCount():
                it = self.table.item(row, j)
                if it is not None:
                    it.setBackground(QtGui.QColor("#fbe3e1"))
                    it.setToolTip(msg)
                w = self.table.cellWidget(row, j)
                if w is not None:
                    w.setStyleSheet(ERROR_STYLE)
                    w.setToolTip(msg)


# --- форма модели -------------------------------------------------------------------------


class ModelForm(QtWidgets.QWidget):
    """Форма вложенной модели: скалярные поля, группы и таблицы."""

    def __init__(
        self, model: type[BaseModel], on_change: Callable[[], None], skip: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self.model = model
        self.on_change = on_change
        self.scalars: dict[str, ScalarEditor] = {}
        self.subforms: dict[str, ModelForm] = {}
        self.optional_groups: dict[str, QtWidgets.QGroupBox] = {}
        self.tables: dict[str, ListTable] = {}
        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        outer.addLayout(form)
        if model.__doc__:
            doc = QtWidgets.QLabel(model.__doc__.strip().split("\n\n")[0].replace("\n", " "))
            doc.setWordWrap(True)
            doc.setStyleSheet("color: #52514e;")
            form.addRow(doc)
        for name, info in model.model_fields.items():
            if name in skip:
                continue
            ann, optional = unwrap_optional(info.annotation)
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                box = QtWidgets.QGroupBox(label_of(name, info))
                box.setToolTip(tooltip(name, info))
                sub = ModelForm(ann, on_change)
                QtWidgets.QVBoxLayout(box).addWidget(sub)
                if optional:
                    box.setCheckable(True)
                    box.toggled.connect(lambda _: on_change())
                    self.optional_groups[name] = box
                self.subforms[name] = sub
                outer.addWidget(box)
            elif get_origin(ann) is list and _is_model(get_args(ann)[0]):
                tab = ListTable(name, info, get_args(ann)[0], on_change)
                self.tables[name] = tab
                outer.addWidget(tab)
            else:
                ed = ScalarEditor(name, info, on_change)
                self.scalars[name] = ed
                lab = QtWidgets.QLabel(label_of(name, info))
                lab.setWordWrap(True)
                lab.setToolTip(tooltip(name, info))
                lab.setMinimumWidth(260)
                form.addRow(lab, ed.widget)
        outer.addStretch(1)

    def set(self, data: dict[str, Any]) -> None:
        for name, ed in self.scalars.items():
            info = self.model.model_fields[name]
            if name in data:
                ed.set(data[name])
            elif isinstance(ed.widget, QtWidgets.QLineEdit):
                ed.set(None)  # пусто: подсказка показывает значение по умолчанию
            else:
                ed.set(None if info.is_required() else default_of(info))
        for name, sub in self.subforms.items():
            info = self.model.model_fields[name]
            v = data.get(name, default_of(info))
            if name in self.optional_groups:
                box = self.optional_groups[name]
                box.blockSignals(True)
                box.setChecked(v is not None)
                box.blockSignals(False)
                if v is None:
                    v = default_of_model(sub.model)
            sub.set(v or {})
        for name, tab in self.tables.items():
            info = self.model.model_fields[name]
            tab.set(data.get(name, default_of(info)) or [])

    def get(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, ed in self.scalars.items():
            given, v = ed.get()
            if given:
                out[name] = v
        for name, sub in self.subforms.items():
            if name in self.optional_groups and not self.optional_groups[name].isChecked():
                out[name] = None
            else:
                out[name] = sub.get()
        for name, tab in self.tables.items():
            out[name] = tab.get()
        return out

    def clear_errors(self) -> None:
        for ed in self.scalars.values():
            ed.set_error(None)
        for sub in self.subforms.values():
            sub.clear_errors()
        for tab in self.tables.values():
            tab.clear_errors()

    def mark_error(self, loc: Path, msg: str) -> bool:
        """Подсветить поле по пути ошибки; True — нашлось."""
        if not loc:
            return False
        head, rest = loc[0], loc[1:]
        if head in self.scalars:
            self.scalars[str(head)].set_error(msg)
            return True
        if head in self.subforms:
            return self.subforms[str(head)].mark_error(rest, msg)
        if head in self.tables:
            row = rest[0] if rest and isinstance(rest[0], int) else -1
            field = rest[1] if len(rest) > 1 and isinstance(rest[1], str) else None
            self.tables[str(head)].set_cell_error(row, field, msg)
            return True
        return False


def _is_model(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def default_of_model(model: type[BaseModel]) -> dict[str, Any]:
    """Значения по умолчанию модели (без обязательных полей)."""
    return {n: default_of(i) for n, i in model.model_fields.items() if not i.is_required()}


_RU = {
    "missing": "обязательное поле не заполнено",
    "float_parsing": "нужно число",
    "int_parsing": "нужно целое число",
    "int_from_float": "нужно целое число",
    "float_type": "нужно число",
    "int_type": "нужно целое число",
    "string_type": "нужен текст",
    "bool_parsing": "нужно да/нет",
    "extra_forbidden": "лишнее поле (такого параметра нет)",
    "string_pattern_mismatch": "недопустимое значение",
    "enum": "недопустимое значение",
    "too_short": "слишком мало строк",
    "tuple_type": "нужны два числа через «;»",
}


def ru_message(err: dict[str, Any]) -> str:
    """Сообщение pydantic по-русски (свои проверки модели уже на русском)."""
    t = err.get("type", "")
    ctx = err.get("ctx") or {}
    if t in ("greater_than", "greater_than_equal", "less_than", "less_than_equal"):
        sign = {"greater_than": ">", "greater_than_equal": "≥", "less_than": "<",
                "less_than_equal": "≤"}[t]  # fmt: skip
        bound = next(iter(ctx.values()), "")
        return f"должно быть {sign} {format_number(bound)}"
    if t in _RU:
        return _RU[t]
    return str(err.get("msg", "")).removeprefix("Value error, ")
