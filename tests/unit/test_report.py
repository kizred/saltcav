"""Отчёт в один клик: разделы по тому, что посчитано, выводы, автономный HTML."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import saltcav.api as sc
from saltcav.cli import main
from saltcav.report import ReportInputs, build_html
from saltcav.surrogate import study

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cfg() -> Any:
    c = sc.load(ROOT / "configs" / "example_ensemble.yaml")
    return sc.set(c, {"grid.nz": 60, "grid.nphi": 4, "leaching.output_every_days": 20})


def test_leaching_only(cfg: Any, tmp_path: Path) -> None:
    p = sc.report(cfg, tmp_path / "r.html")
    text = p.read_text(encoding="utf-8")
    assert "<h2>Выводы</h2>" in text and "2. Выщелачивание" in text
    assert "Ансамбль Монте-Карло" not in text  # не считали — раздела нет
    assert "data:image/png;base64," in text
    assert not re.search(r'src="https?://', text)  # всё внутри файла
    assert "sha256" in text and "<details>" in text


def test_all_sections(cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gmsh")

    def fake(mc: Any, z: Any, radius: Any, layers: Any = None) -> Any:
        return np.array([0.9, 0.8, 3.0, -0.5])  # запас ниже 1 — вывод «опасно»

    monkeypatch.setattr(study, "mechanics_targets", fake)
    c = sc.set(cfg, {"surrogate.n_train": 5, "surrogate.adaptive": 0,
                     "mechanics.stages": [{"duration_years": 1.0}]})  # fmt: skip
    ens = sc.ensemble(c, n=6, jobs=1)
    st = sc.stability(c, ens, jobs=1)
    inp = ReportInputs(c, results=sc.run(c), ensemble=ens, mechanics=sc.mechanics(c), stability=st)
    html = build_html(inp)
    for head in ("1. Исходные данные", "4. Неопределённость", "Карты вероятностей",
                 "5. Геомеханика", "6. Устойчивость по ансамблю", "Воспроизводимость"):  # fmt: skip
        assert head in html, head
    assert 'class="bad"' in html  # P(запас < 1) = 1


def test_cli_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "rep.html"
    assert main(["report", str(ROOT / "configs" / "example.yaml"), "--out", str(out)]) == 0
    assert out.stat().st_size > 50_000
    assert "отчёт:" in capsys.readouterr().out
    assert main(["report", str(ROOT / "configs" / "example.yaml"), "--out", str(out),
                 "--compute", "nonsense"]) == 1  # fmt: skip
