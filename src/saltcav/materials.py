"""Проверенные YAML-библиотеки пород и рабочих сред.

Значения и границы диапазонов хранятся только в СИ. Модуль не связан с
конфигурацией прогона и не подставляет свойства в существующие расчёты.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

__all__ = [
    "QUANTITY_UNITS",
    "LibraryEntry",
    "LibraryValidationError",
    "MaterialLibrary",
    "NumericProperty",
    "PropertySource",
    "load_library",
]

QUANTITY_UNITS: dict[str, frozenset[str]] = {
    "activation_energy": frozenset({"J/mol"}),
    "compressibility": frozenset({"1/Pa"}),
    "density": frozenset({"kg/m3"}),
    "diffusivity": frozenset({"m2/s"}),
    "dimensionless": frozenset({"1"}),
    "dynamic_viscosity": frozenset({"Pa*s"}),
    "kinematic_viscosity": frozenset({"m2/s"}),
    "length": frozenset({"m"}),
    "mass": frozenset({"kg"}),
    "mass_concentration": frozenset({"kg/m3"}),
    "pressure": frozenset({"Pa"}),
    "specific_heat_capacity": frozenset({"J/(kg*K)"}),
    "strain_rate": frozenset({"1/s"}),
    "stress": frozenset({"Pa"}),
    "temperature": frozenset({"K"}),
    "thermal_conductivity": frozenset({"W/(m*K)"}),
    "thermal_expansion": frozenset({"1/K"}),
    "time": frozenset({"s"}),
    "velocity": frozenset({"m/s"}),
}


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("ожидается число, а не логическое значение")
    return value


FiniteFloat = Annotated[float, BeforeValidator(_reject_bool), Field(allow_inf_nan=False)]
PropertyName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class PropertySource(_Strict):
    """Источник числового свойства и точное место в нём, если оно применимо."""

    reference: Annotated[
        str, Field(min_length=1, description="Полное библиографическое описание или идентификатор")
    ]
    location: (
        Annotated[
            str,
            Field(
                min_length=1, description="Страница, таблица, уравнение, раздел или строка данных"
            ),
        ]
        | None
    ) = None


class NumericProperty(_Strict):
    """Числовое свойство в СИ с допустимым включительным диапазоном."""

    quantity: Annotated[
        str, Field(min_length=1, description="Физическая величина из QUANTITY_UNITS")
    ]
    unit: Annotated[str, Field(min_length=1, description="Каноническая единица СИ")]
    range: tuple[FiniteFloat, FiniteFloat]
    value: FiniteFloat
    source: PropertySource

    @field_validator("quantity")
    @classmethod
    def _known_quantity(cls, quantity: str) -> str:
        if quantity not in QUANTITY_UNITS:
            raise ValueError(
                f"неизвестная физическая величина {quantity!r}; "
                "допустимые величины перечислены в QUANTITY_UNITS"
            )
        return quantity

    @field_validator("unit")
    @classmethod
    def _unit_matches_quantity(cls, unit: str, info: ValidationInfo) -> str:
        quantity = info.data.get("quantity")
        allowed = QUANTITY_UNITS.get(quantity) if isinstance(quantity, str) else None
        if allowed is not None and unit not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ValueError(
                f"единица {unit!r} не соответствует величине {quantity!r}; ожидается {expected}"
            )
        return unit

    @field_validator("range")
    @classmethod
    def _ordered_range(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        if bounds[0] > bounds[1]:
            raise ValueError("нижняя граница не может быть больше верхней")
        return bounds

    @field_validator("value")
    @classmethod
    def _value_in_range(cls, value: float, info: ValidationInfo) -> float:
        bounds = info.data.get("range")
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"значение {value} вне допустимого диапазона [{bounds[0]}, {bounds[1]}]"
            )
        return value


class LibraryEntry(_Strict):
    """Одна порода или рабочая среда из отдельного YAML-файла."""

    id: Annotated[
        str,
        Field(
            pattern=r"^[a-z0-9][a-z0-9_-]*$",
            description="Стабильный идентификатор, уникальный во всей библиотеке",
        ),
    ]
    name: Annotated[str, Field(min_length=1, description="Название для отображения")]
    kind: Annotated[str, Field(min_length=1, description="Вид породы или рабочей среды")]
    properties: Annotated[dict[PropertyName, NumericProperty], Field(min_length=1)]


@dataclass(frozen=True)
class MaterialLibrary:
    """Записи библиотеки по стабильным идентификаторам."""

    materials: dict[str, LibraryEntry]
    fluids: dict[str, LibraryEntry]


class LibraryValidationError(ValueError):
    """Ошибка библиотеки с именем файла и путём проблемного поля."""

    def __init__(self, path: str | Path, field: str, message: str) -> None:
        self.path = Path(path)
        self.field = field
        super().__init__(f"{self.path}: {field}: {message}")


class _DuplicateKeyError(yaml.YAMLError):
    def __init__(self, key: object) -> None:
        self.key = key
        super().__init__(f"повторяющийся ключ {key!r}")


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise _DuplicateKeyError(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _field_path(parts: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in parts) if parts else "$"


def _load_entry(path: Path) -> LibraryEntry:
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.load(fh, Loader=_UniqueKeyLoader)
    except _DuplicateKeyError as exc:
        raise LibraryValidationError(path, str(exc.key), str(exc)) from exc
    except yaml.YAMLError as exc:
        raise LibraryValidationError(path, "$", f"некорректный YAML: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise LibraryValidationError(path, "$", f"не удалось прочитать YAML: {exc}") from exc
    try:
        return LibraryEntry.model_validate(raw)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise LibraryValidationError(path, _field_path(error["loc"]), error["msg"]) from exc


def load_library(root: str | Path) -> MaterialLibrary:
    """Загрузить записи из ``root/materials`` и ``root/fluids``.

    Читаются файлы ``.yaml`` и ``.yml`` непосредственно в этих каталогах.
    Отсутствующий каталог означает пустой раздел. Идентификаторы должны быть
    уникальны не только внутри раздела, но и между породами и средами.
    """

    root = Path(root)
    if not root.is_dir():
        raise LibraryValidationError(root, "$", "ожидается каталог библиотеки")

    records: dict[str, dict[str, LibraryEntry]] = {"materials": {}, "fluids": {}}
    origins: dict[str, Path] = {}
    for category in ("materials", "fluids"):
        directory = root / category
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise LibraryValidationError(directory, "$", "ожидается каталог")
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        for path in paths:
            entry = _load_entry(path)
            if entry.id in origins:
                raise LibraryValidationError(
                    path,
                    "id",
                    f"идентификатор {entry.id!r} уже используется в {origins[entry.id]}",
                )
            records[category][entry.id] = entry
            origins[entry.id] = path

    return MaterialLibrary(materials=records["materials"], fluids=records["fluids"])
