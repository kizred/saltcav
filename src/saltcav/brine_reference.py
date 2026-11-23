"""Reference temperature series loaded from the material library.

Density and solubility describe saturated NaCl; viscosity is tabulated for
0–6 mol/kg water. These series do not replace the 1.x model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from saltcav.materials import LibraryEntry, MaterialLibrary

_TEMPERATURES = tuple(range(20, 101, 10))
_SERIES = {
    "saturation": ("dimensionless", "1"),
    "saturated_density": ("density", "kg/m3"),
    "viscosity_6molal": ("dynamic_viscosity", "Pa*s"),
}


def _series(
    entry: LibraryEntry, name: str, axis: tuple[int, ...], quantity: str, unit: str
) -> tuple[float, ...]:
    values = []
    for point in axis:
        key = f"{name}_{point}"
        prop = entry.properties.get(key)
        if prop is None or (prop.quantity, prop.unit) != (quantity, unit):
            raise ValueError(f"{entry.id}: missing or invalid {key}")
        if prop.value <= 0 or (quantity == "dimensionless" and prop.value >= 1):
            raise ValueError(f"{entry.id}: invalid value for {key}")
        values.append(prop.value)
    return tuple(values)


class BrineReference:
    """Explicit interpolation of the fixed conditions in the reference tables."""

    def __init__(self, library: MaterialLibrary, fluid_id: str = "nacl-reference-tables") -> None:
        try:
            entry = library.fluids[fluid_id]
        except KeyError as exc:
            raise ValueError(f"unknown fluid {fluid_id!r}") from exc
        if entry.kind != "nacl_brine_tables":
            raise ValueError(f"{fluid_id}: expected nacl_brine_tables")
        self._values: dict[str, tuple[float, ...]] = {}
        for name, (quantity, unit) in _SERIES.items():
            self._values[name] = _series(entry, f"{name}_t", _TEMPERATURES, quantity, unit)
        self._viscosity: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
        for i in range(13):
            # Two conflicting transcriptions are deliberately excluded.
            omitted = {8: 30, 11: 80}.get(i)
            extra = {8: (25, 35), 11: (75, 85)}.get(i, ())
            axis = tuple(sorted(t for t in _TEMPERATURES + extra if t != omitted))
            name = f"viscosity_m_{i}_t" if i < 12 else "viscosity_6molal_t"
            self._viscosity.append((axis, _series(entry, name, axis, "dynamic_viscosity", "Pa*s")))

    def _interpolate(self, name: str, t_c: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t_c, dtype=float)
        if np.any(~np.isfinite(t)) or np.any((t < 20) | (t > 100)):
            raise ValueError("temperature outside reference range 20–100 °C")
        return np.asarray(np.interp(t, _TEMPERATURES, self._values[name]), dtype=np.float64)

    def saturation_mass_fraction(self, t_c: ArrayLike) -> NDArray[np.float64]:
        return self._interpolate("saturation", t_c)

    def saturated_density(self, t_c: ArrayLike) -> NDArray[np.float64]:
        return self._interpolate("saturated_density", t_c)

    def viscosity_6molal(self, t_c: ArrayLike) -> NDArray[np.float64]:
        return self._interpolate("viscosity_6molal", t_c)

    def viscosity_molal(self, molality: ArrayLike, t_c: ArrayLike) -> NDArray[np.float64]:
        """Viscosity in Pa*s; molality is mol NaCl per kg water, at p*."""
        m, t = np.broadcast_arrays(np.asarray(molality, dtype=float), np.asarray(t_c, dtype=float))
        if np.any(~np.isfinite(m)) or np.any((m < 0) | (m > 6)):
            raise ValueError("molality outside reference range 0–6 mol/kg")
        if np.any(~np.isfinite(t)) or np.any((t < 20) | (t > 100)):
            raise ValueError("temperature outside reference range 20–100 °C")
        rows = np.stack([np.interp(t, axis, values) for axis, values in self._viscosity])
        low = np.minimum((m * 2).astype(int), 11)
        lower = np.take_along_axis(rows, low[None, ...], axis=0)[0]
        upper = np.take_along_axis(rows, (low + 1)[None, ...], axis=0)[0]
        return np.asarray(lower + (upper - lower) * (m * 2 - low), dtype=np.float64)


class LegacyBrineTables:
    """Reproduce the tabular density and solubility calculations of version 1.x."""

    _water_t = tuple(range(0, 101, 10))
    _sol_t = (0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
    _salt_percent = (0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26)

    def __init__(self, library: MaterialLibrary, fluid_id: str = "nacl-default-1x") -> None:
        try:
            entry = library.fluids[fluid_id]
        except KeyError as exc:
            raise ValueError(f"unknown fluid {fluid_id!r}") from exc
        if entry.kind != "nacl_brine_tables":
            raise ValueError(f"{fluid_id}: expected nacl_brine_tables")
        self._water = _series(entry, "water_density", self._water_t, "density", "kg/m3")
        self._sat = _series(entry, "saturation", self._sol_t, "dimensionless", "1")
        self._density = _series(entry, "brine_density_20", self._salt_percent, "density", "kg/m3")

    @staticmethod
    def _temperature(t_c: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t_c, dtype=float)
        if np.any(~np.isfinite(t)) or np.any((t < 0) | (t > 100)):
            raise ValueError("temperature outside table range 0–100 °C")
        return t

    def water_density(self, t_c: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(np.interp(self._temperature(t_c), self._water_t, self._water))

    def saturation_mass_fraction(self, t_c: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(np.interp(self._temperature(t_c), self._sol_t, self._sat))

    def brine_density(self, w: ArrayLike, t_c: ArrayLike = 20.0) -> NDArray[np.float64]:
        w_arr = np.asarray(w, dtype=float)
        if np.any(~np.isfinite(w_arr)) or np.any((w_arr < 0) | (w_arr > 0.30)):
            raise ValueError("mass fraction outside range 0–0.30")
        axis = np.asarray(self._salt_percent, dtype=float) / 100.0
        slope = (self._density[-1] - self._density[-2]) / (axis[-1] - axis[-2])
        rho20 = np.where(
            w_arr <= axis[-1],
            np.interp(w_arr, axis, self._density),
            self._density[-1] + slope * (w_arr - axis[-1]),
        )
        return np.asarray(rho20 * self.water_density(t_c) / self._water[2])
