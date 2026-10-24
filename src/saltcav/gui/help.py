"""Справка в окне: руководство, справочник параметров (строится из модели), клавиши."""

from __future__ import annotations

import html
from importlib import resources
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from PySide6 import QtWidgets

from saltcav.core.config import RunConfig
from saltcav.gui.forms import constraints_text, default_of, format_number, unwrap_optional

KEYS = (
    ("Ctrl+N / Ctrl+O / Ctrl+S", "новый, открыть, сохранить"),
    ("Ctrl+K", "проверить параметры"),
    ("F5", "рассчитать выщелачивание"),
    ("F6", "ансамбль"),
    ("F7", "геомеханика"),
    ("F1", "справка по полю, на котором курсор"),
    ("← →, Home, End, пробел", "просмотр: шаг, начало, конец, воспроизведение"),
)


def guide_markdown() -> str:
    return (resources.files("saltcav.gui") / "help" / "guide.md").read_text(encoding="utf-8")


def anchor(path: str) -> str:
    return "p-" + path.replace("[]", "").replace(".", "-")


def _rows(model: type[BaseModel], prefix: str) -> list[str]:
    out: list[str] = []
    nested: list[tuple[str, type[BaseModel], str]] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann, _ = unwrap_optional(info.annotation)
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            nested.append((path, ann, info.description or ""))
            continue
        if get_origin(ann) is list and issubclass(get_args(ann)[0], BaseModel):
            nested.append((path + "[]", get_args(ann)[0], (info.description or "") + " (таблица)"))
            continue
        d = "обязательно" if info.is_required() else default_of(info)
        d_txt = d if isinstance(d, str) else ("пусто" if d is None else format_number(d))
        out.append(
            f"<tr><td><a name='{anchor(path)}'></a><code>{html.escape(path)}</code></td>"
            f"<td>{html.escape(info.description or '')}</td>"
            f"<td>{html.escape(constraints_text(info))}</td><td>{html.escape(str(d_txt))}</td></tr>"
        )
    for path, sub, desc in nested:
        out.append(
            f"<tr><td colspan='4' style='background:#eef3fb'><a name='{anchor(path)}'></a>"
            f"<b>{html.escape(path)}</b> — {html.escape(desc)}</td></tr>"
        )
        out += _rows(sub, path)
    return out


def parameters_html() -> str:
    """Справочник всех параметров конфигурации — из модели, всегда актуален."""
    head = (
        "<h2>Параметры конфигурации</h2><p>Имя поля — как в YAML. Пустое значение в окне "
        "означает «по умолчанию».</p><table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>поле</th><th>описание</th><th>допустимо</th><th>по умолчанию</th></tr>"
    )
    return head + "".join(_rows(RunConfig, "")) + "</table>"


def keys_html() -> str:
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in KEYS)
    return f"<h2>Клавиши</h2><table cellpadding='4'>{rows}</table>"


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Справка saltcav")
        self.resize(980, 760)
        lay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.guide = QtWidgets.QTextBrowser()
        self.guide.setMarkdown(guide_markdown())
        self.params = QtWidgets.QTextBrowser()
        self.params.setHtml(parameters_html())
        keys = QtWidgets.QTextBrowser()
        keys.setHtml(keys_html())
        self.tabs.addTab(self.guide, "Руководство")
        self.tabs.addTab(self.params, "Параметры")
        self.tabs.addTab(keys, "Клавиши")
        lay.addWidget(self.tabs)

    def show_parameter(self, path: str) -> None:
        self.tabs.setCurrentWidget(self.params)
        self.params.scrollToAnchor(anchor(path))
