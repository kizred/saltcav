"""JSON Schema конфигурации: совпадает с моделью, принимает примеры, ловит ошибки."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from saltcav.cli import SCHEMA_PATH, main
from saltcav.core.config import config_schema

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / SCHEMA_PATH
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def _validator() -> Any:
    schema = json.loads(COMMITTED.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_committed_schema_is_up_to_date() -> None:
    """Если упал: поменялась конфигурация — выполните `saltcav schema` и закоммитьте."""
    assert json.loads(COMMITTED.read_text(encoding="utf-8")) == config_schema()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_examples_valid_and_point_to_schema(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# $schema: ../schema/saltcav-config.schema.json")
    _validator().validate(yaml.safe_load(text))


def _example() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text("utf-8"))
    return raw


def test_schema_catches_typos_and_bad_values() -> None:
    v = _validator()
    raw = _example()
    raw["grid"]["nphy"] = 12  # опечатка
    assert not v.is_valid(raw)
    raw = _example()
    raw["schedule"]["stages"][0]["rate_m3_per_h"] = -5
    assert not v.is_valid(raw)
    raw = _example()
    raw["schedule"]["stages"][0]["mode"] = "sideways"
    assert not v.is_valid(raw)


def test_every_field_is_described() -> None:
    schema = config_schema()
    missing = [
        f"{name}.{prop}"
        for name, d in schema["$defs"].items()  # type: ignore[attr-defined]
        for prop, spec in d.get("properties", {}).items()
        if "description" not in spec and "$ref" not in spec and "allOf" not in spec
    ]
    assert missing == []


def test_cli_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "s.json"
    assert main(["schema", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == config_schema()
    assert main(["schema", "--out", "-"]) == 0
    assert '"$schema"' in capsys.readouterr().out
