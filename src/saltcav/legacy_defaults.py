"""Bundled, validated 1.x physical defaults."""

from functools import lru_cache

from saltcav.materials import MaterialLibrary, load_library
from saltcav.workspace import bundled_materials_library


@lru_cache(maxsize=1)
def library() -> MaterialLibrary:
    return load_library(bundled_materials_library())


def value(category: str, entry: str, name: str, quantity: str, unit: str) -> float:
    entries = getattr(library(), category)
    prop = entries[entry].properties[name]
    if (prop.quantity, prop.unit) != (quantity, unit):
        raise ValueError(f"{entry}.{name}: expected {quantity} in {unit}")
    return float(prop.value)
