import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_installed_version_matches_latest_changelog_release() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest_release = re.search(r"^## ([0-9][^\s]*)\s+—", changelog, re.MULTILINE)

    assert latest_release is not None, "В CHANGELOG.md не найдена запись о выпуске"
    assert version("saltcav") == latest_release.group(1)
