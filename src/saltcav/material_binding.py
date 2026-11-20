"""Разрешение ссылок геологических слоёв на записи библиотеки материалов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from saltcav.core.config import LayerConfig, RunConfig
from saltcav.materials import LibraryValidationError, MaterialLibrary, load_library


class LayerMaterialError(ValueError):
    """Ошибка привязки с путём к конфигурации, слою и полю."""

    def __init__(self, config_path: Path, layer: int, field: str, message: str) -> None:
        self.config_path = config_path
        self.layer = layer
        self.field = field
        super().__init__(f"{config_path}: geology.layers.{layer}.{field}: {message}")


@dataclass(frozen=True)
class _PropertyBinding:
    property_name: str
    field_name: str
    quantity: str
    unit: str


_LAYER_PROPERTIES = (
    _PropertyBinding("critical_span", "critical_span_m", "length", "m"),
)
_PROPERTIES_BY_NAME = {binding.property_name: binding for binding in _LAYER_PROPERTIES}


def _library_error_field(exc: LibraryValidationError) -> str:
    parts = exc.field.split(".")
    if len(parts) >= 2 and parts[0] == "properties":
        binding = _PROPERTIES_BY_NAME.get(parts[1])
        if binding is not None:
            return binding.field_name
    return "material_id"


def _resolve_layer(
    layer: LayerConfig, index: int, config_path: Path, library: MaterialLibrary
) -> LayerConfig:
    material_id = layer.material_id
    if material_id is None:
        return layer
    if material_id not in library.materials:
        if material_id in library.fluids:
            raise LayerMaterialError(
                config_path,
                index,
                "material_id",
                f"id {material_id!r} найден в fluids, а не в materials; слою нужна порода",
            )
        raise LayerMaterialError(
            config_path, index, "material_id", f"неизвестный id породы {material_id!r}"
        )
    entry = library.materials[material_id]
    if entry.kind != layer.kind.value:
        raise LayerMaterialError(
            config_path,
            index,
            "kind",
            f"kind слоя {layer.kind.value!r} противоречит kind материала {entry.kind!r} "
            f"({material_id!r})",
        )

    raw = layer.model_dump(mode="json")
    sources = dict(layer.material_sources)
    applied = False
    for binding in _LAYER_PROPERTIES:
        prop = entry.properties.get(binding.property_name)
        if prop is None:
            continue
        current = getattr(layer, binding.field_name)
        if binding.field_name in layer.model_fields_set and current is not None:
            continue
        if prop.quantity != binding.quantity or prop.unit != binding.unit:
            raise LayerMaterialError(
                config_path,
                index,
                binding.field_name,
                f"materials[{material_id!r}].properties[{binding.property_name!r}] должно иметь "
                f"quantity={binding.quantity!r} и unit={binding.unit!r}; получено "
                f"quantity={prop.quantity!r}, unit={prop.unit!r}",
            )
        candidate = {**raw, binding.field_name: prop.value}
        try:
            LayerConfig.model_validate(candidate)
        except ValidationError as exc:
            detail = exc.errors()[0]["msg"]
            raise LayerMaterialError(
                config_path,
                index,
                binding.field_name,
                f"значение {prop.value} из materials[{material_id!r}]."
                f"properties[{binding.property_name!r}] не прошло проверку: {detail}",
            ) from exc
        raw = candidate
        sources[binding.field_name] = prop.source
        applied = True
    if not applied:
        return layer
    raw["material_sources"] = sources
    return LayerConfig.model_validate(raw)


def resolve_layer_materials(cfg: RunConfig, config_path: str | Path) -> RunConfig:
    """Заполнить поддерживаемые поля слоёв из библиотеки, не меняя её записи.

    Библиотека загружается только при наличии хотя бы одного ``material_id``.
    Относительный ``materials_library`` разрешается от каталога ``config_path``.
    """

    config_path = Path(config_path)
    referenced = [(i, layer) for i, layer in enumerate(cfg.geology.layers) if layer.material_id]
    if not referenced:
        return cfg
    first_index = referenced[0][0]
    if cfg.materials_library is None:
        raise LayerMaterialError(
            config_path,
            first_index,
            "material_id",
            "ссылка на материал требует верхнеуровневого поля materials_library",
        )
    root = Path(cfg.materials_library)
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    try:
        library = load_library(root)
    except LibraryValidationError as exc:
        raise LayerMaterialError(
            config_path,
            first_index,
            _library_error_field(exc),
            f"не удалось загрузить materials_library {str(root)!r}: {exc}",
        ) from exc

    layers = list(cfg.geology.layers)
    for index, layer in referenced:
        layers[index] = _resolve_layer(layer, index, config_path, library)
    geology = cfg.geology.model_copy(update={"layers": layers})
    return cfg.model_copy(update={"materials_library": str(root), "geology": geology})
