"""Геомеханический расчёт по форме камеры: конфигурация → сетка → ползучесть → сводка.

Результат сохраняется в .npz (``MechanicsRun.save``): сетка, поля по моментам и
сводка критериев — для картинок, просмотра и суррогата (этап 7).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import LayerConfig, MechanicsConfig, RunConfig
from saltcav.mechanics.creep import CreepLaw, MunsonDawson, PowerBranch
from saltcav.mechanics.criteria import Summary, element_geometry, summarize
from saltcav.mechanics.fem import AxiFEM
from saltcav.mechanics.solver import (
    YEAR,
    Material,
    MechanicsSetup,
    PressureFn,
    Stage,
    fluid_pressure,
    geothermal,
    lithostatic_gradient,
    run_mechanics,
)
from saltcav.mesh.axisym import (
    AxiMesh,
    Band,
    cavity_profiles,
    default_domain,
    domain_polygon,
    equivalent_radius,
    mesh_polygon,
)

FloatArray = NDArray[np.float64]


def _arr(x: FloatArray | None) -> FloatArray:
    return np.zeros(0) if x is None else x


def _opt(d: Any, key: str) -> Any:
    return d[key] if key in d and d[key].size else None


def creep_law(mc: MechanicsConfig) -> CreepLaw:
    branches = tuple(
        PowerBranch(b.a_per_day / 86_400.0, b.n, 1e3 * b.q_kj_per_mol) for b in mc.creep
    )
    tr = mc.transient
    transient = (
        None
        if tr is None
        else MunsonDawson(
            mu_pa=tr.mu_gpa * 1e9,
            k0=tr.k0,
            c_per_k=tr.c_per_k,
            m=tr.m,
            alpha_w=tr.alpha_w,
            beta_w=tr.beta_w,
            delta=tr.delta,
        )
    )
    return CreepLaw(branches, transient)


def setup_from_config(mc: MechanicsConfig) -> tuple[MechanicsSetup, list[PressureFn]]:
    pressures = [
        fluid_pressure(s.wellhead_pressure_mpa * 1e6, s.fluid_density_kg_m3) for s in mc.stages
    ]
    stages = tuple(
        Stage(s.duration_years * YEAR, p, s.label)
        for s, p in zip(mc.stages, pressures, strict=True)
    )
    setup = MechanicsSetup(
        e_modulus=mc.youngs_modulus_gpa * 1e9,
        nu=mc.poisson,
        lithostatic=lithostatic_gradient(mc.overburden_density_kg_m3),
        temperature_k=geothermal(mc.surface_temperature_c, mc.geothermal_gradient_c_per_km / 1e3),
        law=creep_law(mc),
        stages=stages,
        outer=mc.outer,
        supports=(("axis", 0), ("bottom", 1)) + ((("outer", 0),) if mc.outer == "roller" else ()),
        output_every_s=mc.output_every_years * YEAR,
        eta=mc.step_tolerance,
    )
    return setup, pressures


KIND_RU = {"anhydrite": "ангидрит", "clay": "глина"}


@dataclass(frozen=True)
class LayerModel:
    """Прослои для механики: полосы сетки, материалы 1, 2, … и их прочность на разрыв."""

    bands: list[Band] = field(default_factory=list)
    materials: tuple[Material, ...] = ()
    strengths: list[float] = field(default_factory=list)  # Па

    @property
    def names(self) -> list[str]:
        return ["соль", *(m.name for m in self.materials)]


def layer_model(mc: MechanicsConfig, layers: Sequence[LayerConfig] | None) -> LayerModel:
    """Нерастворимые прослои → полосы и материалы (по породе: один материал на породу)."""
    if not mc.interlayers or not layers:
        return LayerModel()
    props = {m.kind: m for m in mc.layer_materials}
    index: dict[str, int] = {}
    mats: list[Material] = []
    strengths: list[float] = []
    bands = []
    for ly in layers:
        kind = str(getattr(ly.kind, "value", ly.kind))
        if kind not in props:
            continue  # соль (halite) или порода без свойств — считается солью
        if kind not in index:
            p = props[kind]
            mats.append(
                Material(
                    p.youngs_modulus_gpa * 1e9,
                    p.poisson,
                    p.creep_factor,
                    KIND_RU.get(kind, kind),
                    p.compressive_strength_mpa * 1e6,
                )
            )
            strengths.append(p.tensile_strength_mpa * 1e6)
            index[kind] = len(mats)
        bands.append(Band(ly.top_m, ly.top_m + ly.thickness_m, index[kind]))
    return LayerModel(bands, tuple(mats), strengths)


def realization_layers(
    cfg: RunConfig, params: FloatArray | None = None, names: Sequence[str] | None = None
) -> list[LayerConfig]:
    """Прослои конфигурации; для реализации ансамбля — с её отметками и мощностями
    (``params``, ``names`` — значения и имена параметров реализации)."""
    layers = list(cfg.geology.layers)
    if params is None or names is None:
        return layers
    names = list(names)
    out = []
    for k, ly in enumerate(layers):
        upd = {}
        for key, field_ in ((f"layer{k}_top", "top_m"), (f"layer{k}_thickness", "thickness_m")):
            if key in names:
                upd[field_] = float(params[names.index(key)])
        out.append(ly.model_copy(update=upd))
    return out


def ensemble_layers(cfg: RunConfig, ens: Any) -> list[list[LayerConfig]]:
    """Прослои каждой реализации ансамбля (по её параметрам)."""
    names = list(getattr(ens, "param_names", []))
    params = np.asarray(getattr(ens, "params", np.zeros((0, 0))))
    if params.ndim != 2 or params.shape[0] == 0:
        return [list(cfg.geology.layers)] * int(getattr(ens, "n", 0))
    return [realization_layers(cfg, row, names) for row in params]


def build_mesh(
    mc: MechanicsConfig, depth: FloatArray, radius: FloatArray, bands: list[Band] | None = None
) -> AxiMesh:
    r_eq = equivalent_radius(radius)
    profiles = cavity_profiles(depth, r_eq, mc.r_min_m)
    r_out = mc.cell_radius_m if mc.outer == "roller" else None
    dom = default_domain(profiles, mc.h_wall_m, mc.r_factor, mc.height_factor, r_out)
    if bands:
        from saltcav.mesh.axisym import mesh_with_layers

        return mesh_with_layers(profiles, dom, bands)
    poly, tags = domain_polygon(profiles, dom)
    return mesh_polygon(poly, tags, dom.h_wall, dom.h_far, dom.grow_distance)


@dataclass
class MechanicsRun:
    name: str
    day: float  # момент формы, сут
    nodes: FloatArray
    tri: NDArray[np.int64]
    cavity_edges: NDArray[np.int64]
    times: FloatArray  # с
    stage: NDArray[np.int64]
    u: FloatArray  # (nt, N, 2)
    stress: FloatArray  # (nt, M, 4)
    creep_eq: FloatArray  # (nt, M)
    volume: FloatArray  # (nt,)
    volume0: float
    leaching_volume: float  # объём из модели выщелачивания на этот момент
    summary: Summary
    n_steps: int
    stage_labels: list[str]
    dilatancy_coeff: float = 0.27
    # стеновые элементы: номера и центры (для карты стенки в 3D)
    wall_elements: NDArray[np.int64] | None = None
    wall_centroids: FloatArray | None = None
    material: NDArray[np.int64] | None = None  # (M,) 0 — соль, k — прослой
    material_names: list[str] | None = None

    @property
    def dim(self) -> int:
        """2 — осесимметрия (r, y), 3 — полная 3D-постановка (x, y, z)."""
        return int(self.nodes.shape[1])

    @property
    def years(self) -> FloatArray:
        return np.asarray(self.times / YEAR)

    def save(self, path: str | Path) -> None:
        s = self.summary
        np.savez_compressed(
            path,
            kind=np.array("saltcav-mechanics"),
            name=np.array(self.name), day=np.array(self.day), nodes=self.nodes, tri=self.tri,
            cavity_edges=self.cavity_edges, times_s=self.times, stage=self.stage, u_m=self.u,
            stress_pa=self.stress, creep_eq=self.creep_eq, volume_m3=self.volume,
            volume0_m3=np.array(self.volume0), leaching_volume_m3=np.array(self.leaching_volume),
            n_steps=np.array(self.n_steps), stage_labels=np.array(self.stage_labels),
            dilatancy_coeff=np.array(self.dilatancy_coeff),
            s_years=s.years, s_convergence_pct=s.convergence_pct,
            s_convergence_rate_pct_yr=s.convergence_rate_pct_yr, s_min_fos_wall=s.min_fos_wall,
            s_min_fos=s.min_fos, s_dilatant_volume_m3=s.dilatant_volume_m3,
            s_tension_margin_mpa=s.tension_margin_mpa, s_max_wall_disp_m=s.max_wall_disp_m,
            s_fos_wall_p5=_arr(s.fos_wall_p5),
            s_tension_margin_p5_mpa=_arr(s.tension_margin_p5_mpa),
            s_layer_tension_mpa=_arr(s.layer_tension_mpa),
            s_layer_tension_ratio=_arr(s.layer_tension_ratio),
            material=np.zeros(0, dtype=np.int64) if self.material is None else self.material,
            material_names=np.array(self.material_names or []),
            wall_elements=np.zeros(0, dtype=np.int64) if self.wall_elements is None
            else self.wall_elements,
            wall_centroids=np.zeros((0, 3)) if self.wall_centroids is None
            else self.wall_centroids,
        )  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> MechanicsRun:
        with np.load(path) as d:
            if str(d.get("kind", "")) != "saltcav-mechanics":
                raise ValueError(f"{path}: это не результат saltcav mechanics")
            summ = Summary(
                d["s_years"], d["s_convergence_pct"], d["s_convergence_rate_pct_yr"],
                d["s_min_fos_wall"], d["s_min_fos"], d["s_dilatant_volume_m3"],
                d["s_tension_margin_mpa"], d["s_max_wall_disp_m"],
                d["s_fos_wall_p5"] if "s_fos_wall_p5" in d and d["s_fos_wall_p5"].size else None,
                d["s_tension_margin_p5_mpa"]
                if "s_tension_margin_p5_mpa" in d and d["s_tension_margin_p5_mpa"].size else None,
                _opt(d, "s_layer_tension_mpa"), _opt(d, "s_layer_tension_ratio"),
            )  # fmt: skip
            mat = _opt(d, "material")
            return cls(
                name=str(d["name"]), day=float(d["day"]), nodes=d["nodes"], tri=d["tri"],
                cavity_edges=d["cavity_edges"], times=d["times_s"], stage=d["stage"],
                u=d["u_m"], stress=d["stress_pa"], creep_eq=d["creep_eq"],
                volume=d["volume_m3"], volume0=float(d["volume0_m3"]),
                leaching_volume=float(d["leaching_volume_m3"]), summary=summ,
                n_steps=int(d["n_steps"]), stage_labels=[str(x) for x in d["stage_labels"]],
                dilatancy_coeff=float(d["dilatancy_coeff"]),
                wall_elements=d["wall_elements"] if "wall_elements" in d else None,
                wall_centroids=d["wall_centroids"] if "wall_centroids" in d else None,
                material=None if mat is None else mat.astype(np.int64),
                material_names=[str(x) for x in d["material_names"]]
                if mat is not None and "material_names" in d else None,
            )  # fmt: skip


def run_from_shape(
    mc: MechanicsConfig,
    depth: FloatArray,
    radius: FloatArray,
    name: str = "",
    day: float = float("nan"),
    leaching_volume: float = float("nan"),
    progress: Callable[[float], None] | None = None,
    layers: Sequence[LayerConfig] | None = None,
) -> MechanicsRun:
    """Полный расчёт по профилю R(z) или R(z, φ) (``mc.model``: axisym или 3d).

    ``layers`` — нерастворимые прослои (обычно ``cfg.geology.layers``): отдельный
    материал по ``mc.layer_materials``; без них массив — одна соль."""
    lm = layer_model(mc, layers)
    fem: Any
    if mc.model == "3d":
        from saltcav.mechanics.fem3d import FEM3D
        from saltcav.mesh.cavern3d import cavern_mesh_3d

        r_out = mc.cell_radius_m if mc.outer == "roller" else None
        mesh3, _ = cavern_mesh_3d(
            depth, radius, mc.h_wall_3d_m, mc.r_factor, mc.height_factor, mc.r_min_m, r_out,
            bands=lm.bands,
        )  # fmt: skip
        fem = FEM3D(mesh3)
        elements, boundary = mesh3.tets, mesh3.faces["cavity"]
    else:
        mesh = build_mesh(mc, depth, radius, lm.bands)
        fem = AxiFEM(mesh)
        elements, boundary = mesh.tri, mesh.edges["cavity"]
    setup, pressures = setup_from_config(mc)
    setup = replace(setup, materials=lm.materials)
    res = run_mechanics(fem, setup, progress)
    summ = summarize(fem, res, pressures, mc.dilatancy_coeff, lm.strengths)
    mat = getattr(fem.mesh, "material", None)
    geo = element_geometry(fem)
    wall = np.flatnonzero(geo.wall)
    return MechanicsRun(
        name=name,
        day=day,
        nodes=fem.mesh.nodes,
        tri=elements,
        cavity_edges=boundary,
        wall_elements=wall,
        wall_centroids=geo.centroid[wall],
        times=np.asarray(res.times),
        stage=np.asarray(res.stage, dtype=np.int64),
        u=np.asarray(res.u),
        stress=np.asarray(res.stress),
        creep_eq=np.asarray(res.creep_eq),
        volume=np.asarray(res.volume),
        volume0=res.volume0,
        leaching_volume=leaching_volume,
        summary=summ,
        n_steps=res.n_steps,
        stage_labels=[s.label for s in mc.stages],
        dilatancy_coeff=mc.dilatancy_coeff,
        material=mat,
        material_names=lm.names if mat is not None else None,
    )
