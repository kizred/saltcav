"""Геомеханический расчёт по форме камеры: конфигурация → сетка → ползучесть → сводка.

Результат сохраняется в .npz (``MechanicsRun.save``): сетка, поля по моментам и
сводка критериев — для картинок, просмотра и суррогата (этап 7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from saltcav.core.config import MechanicsConfig
from saltcav.mechanics.creep import CreepLaw, MunsonDawson, PowerBranch
from saltcav.mechanics.criteria import Summary, summarize
from saltcav.mechanics.fem import AxiFEM
from saltcav.mechanics.solver import (
    YEAR,
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
    cavity_profiles,
    default_domain,
    domain_polygon,
    equivalent_radius,
    mesh_polygon,
)

FloatArray = NDArray[np.float64]


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


def build_mesh(mc: MechanicsConfig, depth: FloatArray, radius: FloatArray) -> AxiMesh:
    r_eq = equivalent_radius(radius)
    profiles = cavity_profiles(depth, r_eq, mc.r_min_m)
    r_out = mc.cell_radius_m if mc.outer == "roller" else None
    dom = default_domain(profiles, mc.h_wall_m, mc.r_factor, mc.height_factor, r_out)
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
            )  # fmt: skip
            return cls(
                name=str(d["name"]), day=float(d["day"]), nodes=d["nodes"], tri=d["tri"],
                cavity_edges=d["cavity_edges"], times=d["times_s"], stage=d["stage"],
                u=d["u_m"], stress=d["stress_pa"], creep_eq=d["creep_eq"],
                volume=d["volume_m3"], volume0=float(d["volume0_m3"]),
                leaching_volume=float(d["leaching_volume_m3"]), summary=summ,
                n_steps=int(d["n_steps"]), stage_labels=[str(x) for x in d["stage_labels"]],
                dilatancy_coeff=float(d["dilatancy_coeff"]),
            )  # fmt: skip


def run_from_shape(
    mc: MechanicsConfig,
    depth: FloatArray,
    radius: FloatArray,
    name: str = "",
    day: float = float("nan"),
    leaching_volume: float = float("nan"),
    progress: Callable[[float], None] | None = None,
) -> MechanicsRun:
    """Полный расчёт по профилю R(z) или R(z, φ)."""
    mesh = build_mesh(mc, depth, radius)
    fem = AxiFEM(mesh)
    setup, pressures = setup_from_config(mc)
    res = run_mechanics(fem, setup, progress)
    summ = summarize(fem, res, pressures, mc.dilatancy_coeff)
    return MechanicsRun(
        name=name,
        day=day,
        nodes=mesh.nodes,
        tri=mesh.tri,
        cavity_edges=mesh.edges["cavity"],
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
    )
