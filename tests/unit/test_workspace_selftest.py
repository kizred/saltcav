"""Рабочая папка собранного приложения и самопроверка."""

from pathlib import Path

import pytest

from saltcav.workspace import (
    bundled_materials_library,
    bundled_root,
    ensure_workspace,
    frozen,
    sansmic_reference,
)


def _materials_only_skip() -> str:
    from saltcav.selftest import checks

    return ",".join(name for name, _ in checks(Path()) if name != "библиотека материалов")


def test_workspace_copy(tmp_path: Path) -> None:
    assert not frozen()
    assert (bundled_root() / "configs" / "example.yaml").exists()
    assert sansmic_reference().exists()
    ws = ensure_workspace(tmp_path / "ws")
    cfg = ws / "configs" / "example.yaml"
    assert cfg.exists() and (ws / "data" / "demo" / "well-1" / "brine.csv").exists()
    assert (ws / "examples" / "parameter_sweep.py").exists()
    cfg.write_text("# моя правка\n", encoding="utf-8")
    ensure_workspace(tmp_path / "ws")  # повторно — правки пользователя не затираются
    assert cfg.read_text(encoding="utf-8") == "# моя правка\n"


def test_selftest_bundled_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from saltcav.cli import main

    monkeypatch.chdir(tmp_path)
    assert not list(tmp_path.iterdir())
    assert main(["selftest", "--skip", _materials_only_skip()]) == 0
    out = capsys.readouterr().out
    assert "библиотека материалов" in out and "OK" in out
    assert "порода test-synthetic-rock, среда закачки test-synthetic-fluid" in out


def test_selftest_missing_materials_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from saltcav.cli import main

    missing = tmp_path / "missing-materials-library"
    monkeypatch.setattr("saltcav.workspace.bundled_materials_library", lambda: missing)

    assert main(["selftest", "--skip", _materials_only_skip()]) == 1
    out = capsys.readouterr().out
    assert "библиотека материалов" in out and "СБОЙ" in out
    assert "FileNotFoundError" in out
    assert f"поставляемый ресурс библиотеки материалов не найден: {missing}" in out


def test_selftest_missing_materials_entry_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from saltcav.cli import main

    root = tmp_path / "empty-materials-library"
    (root / "materials").mkdir(parents=True)
    (root / "fluids").mkdir()
    monkeypatch.setattr("saltcav.workspace.bundled_materials_library", lambda: root)

    assert main(["selftest", "--skip", _materials_only_skip()]) == 1
    out = capsys.readouterr().out
    assert str(root) in out
    assert "не найдена порода 'test-synthetic-rock'" in out


@pytest.mark.parametrize(
    ("category", "entry_id"),
    [("materials", "test-synthetic-rock"), ("fluids", "test-synthetic-fluid")],
)
def test_selftest_missing_materials_entry(
    category: str,
    entry_id: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from saltcav.cli import main
    from saltcav.materials import load_library

    library = load_library(bundled_materials_library())
    getattr(library, category).pop(entry_id)
    monkeypatch.setattr("saltcav.materials.load_library", lambda _: library)

    assert main(["selftest", "--skip", _materials_only_skip()]) == 1
    out = capsys.readouterr().out
    assert "СБОЙ" in out and "не найдена" in out and entry_id in out


def test_selftest_quick(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("gmsh")
    from saltcav.selftest import run_selftest

    ok = run_selftest(skip=("PARDISO", "геомеханика 3D", "3D-вид", "ансамбль"))
    out = capsys.readouterr().out
    assert ok, out
    assert "выщелачивание    OK" in out and "калибровка       OK" in out


def test_cli_output_in_ansi_pipe(tmp_path: Path) -> None:
    """Вывод в конвейер с кодировкой ANSI (как на Windows) — без UnicodeEncodeError."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "from saltcav.cli import main; main(['version']); "
         "print('χ² ≈ 1 — проверка')"],
        capture_output=True, env={**__import__("os").environ, "PYTHONIOENCODING": "cp1252"},
        check=True,
    )  # fmt: skip
    assert "χ² ≈ 1 — проверка" in out.stdout.decode("utf-8")


def test_license(capsys: pytest.CaptureFixture[str]) -> None:
    """Соглашение в поставке: UTF-8 с BOM (для Inno Setup), команда и текст."""
    from saltcav.cli import main
    from saltcav.workspace import license_text

    raw = (bundled_root() / "LICENSE.txt").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = license_text()
    assert "Sergei Kizred" in text and "коммерческой" in text and "декомпилировать" in text
    assert main(["license"]) == 0
    assert "ЛИЦЕНЗИОННОЕ СОГЛАШЕНИЕ" in capsys.readouterr().out
