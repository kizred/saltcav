"""Значок программы и ярлык окна."""

import sys
from pathlib import Path

import pytest

from saltcav import __author__
from saltcav.gui.icons import icon_path
from saltcav.shortcut import linux_desktop_entry, windows_script


def test_icons_shipped() -> None:
    from PIL import Image

    png = Image.open(icon_path("png"))
    assert png.size == (512, 512) and png.mode == "RGBA"
    assert png.getpixel((0, 0))[3] == 0  # углы прозрачные
    ico = Image.open(icon_path("ico"))
    assert (256, 256) in ico.info["sizes"] and (16, 16) in ico.info["sizes"]
    assert __author__ == "Sergei Kizred"


def test_windows_script() -> None:
    s = windows_script(Path("C:/env/Scripts/saltcav-gui.exe"), Path("C:/x/saltcav.ico"),
                       Path("C:/Users/O'Brien/saltcav"), ["Desktop", "Programs"])  # fmt: skip
    assert s.count("$lnk.Save()") == 2 and "GetFolderPath('Desktop')" in s
    assert "IconLocation = 'C:/x/saltcav.ico,0'" in s
    assert "O''Brien" in s  # кавычка экранирована для PowerShell


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ярлык Linux")
def test_linux_shortcut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import saltcav.shortcut as sh

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sh, "gui_executable", lambda: Path("/env/bin/saltcav-gui"))
    (tmp_path / "Desktop").mkdir()
    made = sh.create_shortcuts(workdir=tmp_path)
    assert len(made) == 2
    text = made[0].read_text(encoding="utf-8")
    assert 'Exec="/env/bin/saltcav-gui" %f' in text and "saltcav.png" in text
    assert linux_desktop_entry(Path("/a"), Path("/b.png"), tmp_path).startswith("[Desktop Entry]")
