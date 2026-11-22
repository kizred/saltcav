"""Версия пакета совпадает с последним выпуском в журнале."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_version_matches_latest_changelog_release() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = re.search(r"^##\s+(\d+\.\d+\.\d+)\s+—", changelog, re.MULTILINE)
    assert latest is not None, "В CHANGELOG.md не найдена запись о выпуске"
    assert version == latest.group(1)
