"""Проверка связей названных в коде источников с библиографией."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("src/saltcav/leaching/plume.py", "SANSMIC", "https://github.com/sandialabs/sansmic"),
    (
        "src/saltcav/core/brine.py",
        "Handbook of Chemistry and Physics",
        "https://www.taylorfrancis.com/books/mono/10.1201/b17118/"
        "crc-handbook-chemistry-physics-william-haynes",
    ),
    (
        "src/saltcav/mechanics/creep.py",
        "Hampel, 1999",
        "https://www.sciencedirect.com/science/article/pii/S0013795299000113",
    ),
    (
        "src/saltcav/mechanics/creep.py",
        "Munson, 1997",
        "https://www.sciencedirect.com/science/article/abs/pii/S0148906296000472",
    ),
    (
        "src/saltcav/mechanics/criteria.py",
        "Ratigan, Van Sambeek, DeVries, Nieland, 1991",
        "https://digital.library.unt.edu/ark%3A/67531/metadc841794/m2/1/high_res_d/1013228.pdf",
    ),
    (
        "src/saltcav/surrogate/study.py",
        "Echard et al., 2011",
        "https://www.sciencedirect.com/science/article/pii/S0167473011000038",
    ),
)


@pytest.mark.parametrize(("path", "source", "url"), SOURCES)
def test_code_sources_are_listed_in_literature(path: str, source: str, url: str) -> None:
    assert source in (ROOT / path).read_text(encoding="utf-8")
    literature = (ROOT / "docs/literature.md").read_text(encoding="utf-8")
    assert f"]({url})" in literature, f"Источник отсутствует в docs/literature.md: {url}"


def test_missing_source_link_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read = Path.read_text
    path, source, url = SOURCES[0]

    def read_without_source(file: Path, *args: object, **kwargs: object) -> str:
        text = original_read(file, *args, **kwargs)
        return text.replace(url, "missing-source") if file == ROOT / "docs/literature.md" else text

    monkeypatch.setattr(Path, "read_text", read_without_source)
    with pytest.raises(AssertionError, match="Источник отсутствует"):
        test_code_sources_are_listed_in_literature(path, source, url)
