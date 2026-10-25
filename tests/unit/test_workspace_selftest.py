"""Рабочая папка собранного приложения и самопроверка."""

from pathlib import Path

import pytest

from saltcav.workspace import bundled_root, ensure_workspace, frozen, sansmic_reference


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
