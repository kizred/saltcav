"""Ярлык окна saltcav со значком программы: на рабочем столе и в меню «Пуск»
(Windows) или в меню приложений (Linux). Команда: ``saltcav shortcut``.

Ярлык запускает ``saltcav-gui`` из того же окружения Python, в которое установлен
saltcav (например, .venv проекта), — поэтому после переустановки окружения ярлык
нужно создать заново.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from saltcav.gui.icons import icon_path

NAME = "saltcav"
COMMENT = "Моделирование соляных камер: выщелачивание, неопределённость, устойчивость"


def gui_executable() -> Path:
    """Запускатель окна в текущем окружении (saltcav-gui[.exe])."""
    scripts = Path(sysconfig.get_path("scripts"))
    exe = scripts / ("saltcav-gui.exe" if sys.platform == "win32" else "saltcav-gui")
    if exe.exists():
        return exe
    found = shutil.which("saltcav-gui")
    if found:
        return Path(found)
    raise FileNotFoundError("не найден saltcav-gui: установите пакет с окном "
                            '(pip install -e ".[gui]")')  # fmt: skip


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def windows_script(target: Path, icon: Path, workdir: Path, folders: list[str]) -> str:
    """Скрипт PowerShell: ярлык .lnk в папках ``folders`` (Desktop, Programs)."""
    lines = ["$sh = New-Object -ComObject WScript.Shell"]
    for f in folders:
        lines += [
            f"$dir = [Environment]::GetFolderPath({_ps_quote(f)})",
            f"$lnk = $sh.CreateShortcut((Join-Path $dir {_ps_quote(NAME + '.lnk')}))",
            f"$lnk.TargetPath = {_ps_quote(str(target))}",
            f"$lnk.WorkingDirectory = {_ps_quote(str(workdir))}",
            f"$lnk.IconLocation = {_ps_quote(str(icon) + ',0')}",
            f"$lnk.Description = {_ps_quote(COMMENT)}",
            "$lnk.Save()",
            "Write-Output $lnk.FullName",
        ]
    return "; ".join(lines)


def linux_desktop_entry(target: Path, icon: Path, workdir: Path) -> str:
    return (
        "[Desktop Entry]\nType=Application\n"
        f'Name={NAME}\nComment={COMMENT}\nExec="{target}" %f\nIcon={icon}\n'
        f"Path={workdir}\nTerminal=false\nCategories=Science;Engineering;\n"
        "MimeType=application/x-yaml;\n"
    )


def create_shortcuts(desktop: bool = True, menu: bool = True,
                     workdir: Path | None = None) -> list[Path]:  # fmt: skip
    """Создать ярлыки; вернуть пути созданных файлов. ``workdir`` — рабочая папка окна
    (по умолчанию текущая: там ищутся configs/ и data/)."""
    target = gui_executable()
    wd = (workdir or Path.cwd()).resolve()
    if sys.platform == "win32":
        folders = (["Desktop"] if desktop else []) + (["Programs"] if menu else [])
        script = windows_script(target, icon_path("ico"), wd, folders)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return [Path(p) for p in out.stdout.split()]
    if sys.platform.startswith("linux"):
        entry = linux_desktop_entry(target, icon_path("png"), wd)
        made = []
        dirs = []
        if menu:
            dirs.append(Path.home() / ".local" / "share" / "applications")
        if desktop:
            dirs.append(Path.home() / "Desktop")
        for d in dirs:
            if d.name == "Desktop" and not d.exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{NAME}.desktop"
            p.write_text(entry, encoding="utf-8")
            p.chmod(0o755)
            made.append(p)
        return made
    raise OSError("ярлык создаётся на Windows и Linux; на macOS запускайте saltcav-gui")
