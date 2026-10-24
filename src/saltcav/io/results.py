"""Результаты прогона в файле .npz — всё, что нужно для просмотра по шагам.

Формат — словарь массивов NumPy (``np.savez_compressed``): читается без saltcav,
например ``np.load("hist.npz")["volume_m3"]``. Массивы по времени — первая ось,
по моментам записи (``leaching.output_every_days``). Файлы прежнего формата
(``run --out`` до просмотрщика) читаются: недостающие поля заполняются NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from saltcav import __version__
from saltcav.core import brine
from saltcav.core.config import RunConfig

FloatArray = NDArray[np.float64]
FORMAT_VERSION = 2


@dataclass(frozen=True)
class Event:
    """Событие прогона для журнала просмотрщика."""

    t: float  # с
    kind: str  # "collapse" | "shoe" | "stage"
    text: str
    depth_m: float = float("nan")


@dataclass
class Results:
    name: str
    z_edges: FloatArray  # (nz+1,), глубина узлов, м
    nphi: int
    times: FloatArray  # (nt,), с
    radius: FloatArray  # (nt, nz+1, nphi), м
    concentration: FloatArray  # (nt, nz), кг/м³
    volume: FloatArray  # (nt,), м³
    produced_salt: FloatArray  # (nt,), кг, накоплено
    dissolved_salt: FloatArray  # (nt,), кг, накоплено
    floor_depth: FloatArray  # (nt,), м
    insoluble_bulk: FloatArray  # (nt,), м³
    c_sat: float  # кг/м³
    wall_rate: FloatArray | None = None  # (nt, nz+1, nphi), м/с
    outlet_concentration: FloatArray | None = None  # (nt,)
    shoe_in_depth: FloatArray | None = None  # (nt,)
    shoe_out_depth: FloatArray | None = None
    blanket_depth: FloatArray | None = None
    stage_index: NDArray[np.int64] | None = None
    k_mult: FloatArray | None = None  # (nz+1, nphi)
    insoluble: FloatArray | None = None
    layers: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))  # (n, 2): кровля, мощность
    collapses: FloatArray = field(default_factory=lambda: np.zeros((0, 5)))
    # колонки collapses: t, глубина, вылет, сектор, прослой
    shoe_move_times: FloatArray = field(default_factory=lambda: np.zeros(0))
    stage_starts: FloatArray = field(default_factory=lambda: np.zeros(0))  # с
    n_steps: int = 0
    config_yaml: str = ""

    # --- свойства ------------------------------------------------------------

    @property
    def nt(self) -> int:
        return int(self.times.size)

    @property
    def nz(self) -> int:
        return int(self.z_edges.size - 1)

    @property
    def days(self) -> FloatArray:
        return np.asarray(self.times / 86_400.0)

    @property
    def phi(self) -> FloatArray:
        """Азимуты центров секторов, рад."""
        return np.asarray((np.arange(self.nphi) + 0.5) * 2.0 * np.pi / self.nphi)

    def frame_at(self, day: float) -> int:
        """Ближайший сохранённый момент к сутке ``day``."""
        return int(np.argmin(np.abs(self.days - day)))

    def events(self) -> list[Event]:
        """Обрушения, подъёмы башмака и смены ступеней по времени."""
        ev: list[Event] = []
        for i, t0 in enumerate(self.stage_starts):
            if i > 0:
                ev.append(Event(float(t0), "stage", f"ступень {i + 1}"))
        for t, depth, span, sector, layer in self.collapses:
            ev.append(
                Event(
                    float(t),
                    "collapse",
                    f"обрушение карниза прослоя {int(layer) + 1}, сектор {int(sector) + 1}, "
                    f"вылет {span:.1f} м",
                    float(depth),
                )
            )
        for t in self.shoe_move_times:
            ev.append(Event(float(t), "shoe", "башмак поднят над насыпью"))
        return sorted(ev, key=lambda e: e.t)

    # --- построение ----------------------------------------------------------

    @classmethod
    def from_history(cls, hist: object, cfg: RunConfig) -> Results:
        """Из :class:`saltcav.leaching.simulator.History` и конфигурации прогона."""
        from saltcav.leaching.simulator import History

        assert isinstance(hist, History)
        g = hist.grid
        props = hist.props
        coll = np.array(
            [(t, e.depth_m, e.span, e.sector, e.layer) for t, e in hist.collapses], dtype=float
        ).reshape(-1, 5)
        starts = np.cumsum([0.0] + [s.duration_days * 86_400.0 for s in cfg.schedule.stages])
        return cls(
            name=cfg.name,
            z_edges=np.asarray(g.z_edges, dtype=float),
            nphi=g.nphi,
            times=np.asarray(hist.times, dtype=float),
            radius=np.asarray(hist.radius, dtype=float),
            concentration=np.asarray(hist.concentration, dtype=float),
            volume=np.asarray(hist.volume, dtype=float),
            produced_salt=np.asarray(hist.produced_salt, dtype=float),
            dissolved_salt=np.asarray(hist.dissolved_salt, dtype=float),
            floor_depth=np.asarray(hist.floor_depth, dtype=float),
            insoluble_bulk=np.asarray(hist.insoluble_bulk, dtype=float),
            c_sat=float(brine.saturation_concentration(cfg.leaching.temperature_c)),
            wall_rate=np.asarray(hist.wall_rate, dtype=float),
            outlet_concentration=np.asarray(hist.outlet_concentration, dtype=float),
            shoe_in_depth=np.asarray(hist.shoe_in_depth, dtype=float),
            shoe_out_depth=np.asarray(hist.shoe_out_depth, dtype=float),
            blanket_depth=np.asarray(hist.blanket_depth, dtype=float),
            stage_index=np.asarray(hist.stage_index, dtype=np.int64),
            k_mult=None if props is None else np.asarray(props.k_mult, dtype=float),
            insoluble=None if props is None else np.asarray(props.insoluble, dtype=float),
            layers=np.array(
                [(lay.top_m, lay.thickness_m) for lay in cfg.geology.layers], dtype=float
            ).reshape(-1, 2),
            collapses=coll,
            shoe_move_times=np.asarray(hist.shoe_move_times, dtype=float),
            stage_starts=starts[:-1],
            n_steps=hist.n_steps,
            config_yaml=yaml.safe_dump(
                cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
        )

    # --- файл ----------------------------------------------------------------

    _OPTIONAL = (
        ("wall_rate", "wall_rate_m_s"),
        ("outlet_concentration", "outlet_concentration_kg_m3"),
        ("shoe_in_depth", "shoe_in_depth_m"),
        ("shoe_out_depth", "shoe_out_depth_m"),
        ("blanket_depth", "blanket_depth_m"),
        ("stage_index", "stage_index"),
        ("k_mult", "k_mult"),
        ("insoluble", "insoluble_fraction"),
    )

    def save(self, path: str | Path) -> None:
        arrays: dict[str, object] = {
            "format_version": np.array(FORMAT_VERSION),
            "saltcav_version": np.array(__version__),
            "name": np.array(self.name),
            "config_yaml": np.array(self.config_yaml),
            "times_s": self.times,
            "z_edges_m": self.z_edges,
            "nphi": np.array(self.nphi),
            "radius_m": self.radius,
            "concentration_kg_m3": self.concentration,
            "volume_m3": self.volume,
            "produced_salt_kg": self.produced_salt,
            "dissolved_salt_kg": self.dissolved_salt,
            "floor_depth_m": self.floor_depth,
            "insoluble_bulk_m3": self.insoluble_bulk,
            "c_sat_kg_m3": np.array(self.c_sat),
            "layers_m": self.layers,
            "collapses": self.collapses,
            "shoe_move_times_s": self.shoe_move_times,
            "stage_starts_s": self.stage_starts,
            "n_steps": np.array(self.n_steps),
        }
        for attr, key in self._OPTIONAL:
            val = getattr(self, attr)
            if val is not None:
                arrays[key] = val
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> Results:
        path = Path(path)
        with np.load(path) as d:
            radius = np.asarray(d["radius_m"], dtype=float)
            nt = radius.shape[0]

            def get(key: str, default: object) -> object:
                return d[key] if key in d else default

            nan_t = np.full(nt, np.nan)
            res = cls(
                name=str(get("name", path.stem)),
                z_edges=np.asarray(d["z_edges_m"], dtype=float),
                nphi=int(get("nphi", radius.shape[2])),  # type: ignore[call-overload]
                times=np.asarray(d["times_s"], dtype=float),
                radius=radius,
                concentration=np.asarray(d["concentration_kg_m3"], dtype=float),
                volume=np.asarray(d["volume_m3"], dtype=float),
                produced_salt=np.asarray(get("produced_salt_kg", nan_t), dtype=float),
                dissolved_salt=np.asarray(get("dissolved_salt_kg", nan_t), dtype=float),
                floor_depth=np.asarray(get("floor_depth_m", nan_t), dtype=float),
                insoluble_bulk=np.asarray(get("insoluble_bulk_m3", nan_t), dtype=float),
                c_sat=float(get("c_sat_kg_m3", brine.saturation_concentration(30.0))),  # type: ignore[arg-type]
                layers=np.asarray(get("layers_m", np.zeros((0, 2))), dtype=float).reshape(-1, 2),
                collapses=np.asarray(get("collapses", np.zeros((0, 5))), dtype=float).reshape(
                    -1, 5
                ),
                shoe_move_times=np.asarray(get("shoe_move_times_s", np.zeros(0)), dtype=float),
                stage_starts=np.asarray(get("stage_starts_s", np.zeros(0)), dtype=float),
                n_steps=int(get("n_steps", 0)),  # type: ignore[call-overload]
                config_yaml=str(get("config_yaml", "")),
            )
            for attr, key in cls._OPTIONAL:
                if key in d:
                    setattr(res, attr, np.asarray(d[key]))
        return res
