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


_LAYER_PROPERTIES = (_PropertyBinding("critical_span", "critical_span_m", "length", "m"),)
_PROPERTIES_BY_NAME = {binding.property_name: binding for binding in _LAYER_PROPERTIES}


def _library_error_field(exc: LibraryValidationError) -> str:
    parts = exc.field.split(".")
    if len(parts) >= 2 and parts[0] == "properties":
        binding = _PROPERTIES_BY_NAME.get(parts[1])
        if binding is not None:
            return binding.field_name
    return "material_id"


def _library_error_layer(cfg: RunConfig, exc: LibraryValidationError, default: int) -> int:
    parts = exc.field.split(".")
    if len(parts) >= 2 and parts[0] == "properties":
        for rule in cfg.ensemble.layer_properties:
            if rule.property == parts[1]:
                return rule.layer
    return default


def layer_property_field(property_name: str) -> str:
    """Имя расчётного поля слоя для поддерживаемого свойства материала."""
    return _PROPERTIES_BY_NAME[property_name].field_name


def sampled_layer_property_ranges(
    cfg: RunConfig,
    config_path: str | Path = Path("<RunConfig>"),
    library: MaterialLibrary | None = None,
) -> list[tuple[float, float]]:
    """Проверить правила ансамбля и вернуть их библиотечные диапазоны в СИ."""
    rules = cfg.ensemble.layer_properties
    if not rules:
        return []
    config_path = Path(config_path)
    bindings: list[_PropertyBinding] = []
    seen: set[tuple[int, str]] = set()
    for rule_index, rule in enumerate(rules):
        location = f"ensemble.layer_properties.{rule_index}"
        binding = _PROPERTIES_BY_NAME.get(rule.property)
        if binding is None:
            supported = ", ".join(_PROPERTIES_BY_NAME)
            raise LayerMaterialError(
                config_path,
                rule.layer,
                "material_id",
                f"{location}: неизвестное или неподдерживаемое свойство {rule.property!r}; "
                f"доступно: {supported}",
            )
        bindings.append(binding)
        if rule.layer >= len(cfg.geology.layers):
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"{location}: слой с таким индексом отсутствует в geology.layers",
            )
        layer = cfg.geology.layers[rule.layer]
        key = (rule.layer, rule.property)
        if key in seen:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"{location}: правило выборки свойства {rule.property!r} уже задано для этого слоя",
            )
        seen.add(key)
        if layer.material_id is None:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"{location}: выборка свойства {rule.property!r} требует material_id",
            )
        # SC-011 добавляет material_overrides; getattr сохраняет совместимость веток до merge.
        overrides = getattr(layer, "material_overrides", {})
        if rule.property in overrides:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                f"material_overrides.{rule.property}",
                f"{location}: выборка конфликтует с явным локальным переопределением "
                f"свойства {rule.property!r}",
            )
        current = getattr(layer, binding.field_name)
        if current is not None and binding.field_name not in layer.material_sources:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"{location}: выборка свойства {rule.property!r} конфликтует с явным "
                f"значением {binding.field_name}",
            )
        if rule.property == "critical_span" and cfg.uncertainty.ledge_span_log_sd > 0:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"{location}: выборка свойства 'critical_span' конфликтует с уже заданным "
                "uncertainty.ledge_span_log_sd",
            )

    if library is None:
        first = rules[0]
        if cfg.materials_library is None:
            raise LayerMaterialError(
                config_path,
                first.layer,
                bindings[0].field_name,
                f"выборка свойства {first.property!r} требует materials_library",
            )
        try:
            library = load_library(cfg.materials_library)
        except LibraryValidationError as exc:
            raise LayerMaterialError(
                config_path,
                _library_error_layer(cfg, exc, first.layer),
                _library_error_field(exc),
                f"не удалось загрузить materials_library {cfg.materials_library!r}: {exc}",
            ) from exc

    ranges: list[tuple[float, float]] = []
    for rule, binding in zip(rules, bindings, strict=True):
        layer = cfg.geology.layers[rule.layer]
        material_id = layer.material_id
        assert material_id is not None
        entry = library.materials.get(material_id)
        if entry is None:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                "material_id",
                f"неизвестный id породы {material_id!r} для выборки {rule.property!r}",
            )
        prop = entry.properties.get(rule.property)
        if prop is None:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"materials[{material_id!r}] не содержит свойство {rule.property!r} "
                "с диапазоном range для выборки",
            )
        bounds = getattr(prop, "range", None)
        if bounds is None:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"materials[{material_id!r}].properties[{rule.property!r}] "
                "не содержит диапазон range для выборки",
            )
        if prop.quantity != binding.quantity or prop.unit != binding.unit:
            raise LayerMaterialError(
                config_path,
                rule.layer,
                binding.field_name,
                f"materials[{material_id!r}].properties[{rule.property!r}] должно иметь "
                f"quantity={binding.quantity!r} и unit={binding.unit!r}; получено "
                f"quantity={prop.quantity!r}, unit={prop.unit!r}",
            )
        ranges.append(bounds)
    return ranges


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
        sampled_layer_property_ranges(cfg, config_path)
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
            _library_error_layer(cfg, exc, first_index),
            _library_error_field(exc),
            f"не удалось загрузить materials_library {str(root)!r}: {exc}",
        ) from exc

    layers = list(cfg.geology.layers)
    for index, layer in referenced:
        layers[index] = _resolve_layer(layer, index, config_path, library)
    geology = cfg.geology.model_copy(update={"layers": layers})
    resolved = cfg.model_copy(update={"materials_library": str(root), "geology": geology})
    sampled_layer_property_ranges(resolved, config_path, library)
    return resolved
