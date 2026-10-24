"""Упругость и ползучесть массива вокруг камеры во времени (осесимметрия).

Постановка в приращениях от природного состояния: σ = σ₀ + Δσ, σ₀ = −p_лит(z)·I
(природное поле гидростатическое, K₀ = 1, — обычное допущение для соли; оно
уравновешено весом пород, поэтому объёмная сила в задаче для Δσ не нужна). На стенке
полости Δσ·n = −(p_кам − p_лит)·n: камера под давлением рассола меньше горного, стенка
сходится внутрь. Граничные условия: ось — u_r = 0, низ — u_z = 0, верх и внешний
радиус свободны (дальнее поле) или внешний радиус на катках (ячейка ряда камер).

Ползучесть ε_c интегрируется линеаризованной неявной схемой (θ = 1):
Δε_c = Δt·(ε̇ₙ + G·Δσ), G = ∂ε̇/∂σ, отсюда Δσ = D̃(Δε − Δt·ε̇ₙ), D̃ = (D⁻¹ + Δt·G)⁻¹.
Схема устойчива при больших шагах; шаг выбирается по точности — относительное
изменение девиатора напряжений за шаг |Δs|/σ_экв ≈ η (по умолчанию 0,1). При
установившейся ползучести напряжения почти не меняются, и шаг растёт до dt_max.

Смена давления в камере между ступенями — упругий скачок.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saltcav.mechanics.creep import CreepLaw, trace_vector, von_mises
from saltcav.mechanics.fem import QW, AxiFEM, elastic_matrix

FloatArray = NDArray[np.float64]
PressureFn = Callable[[FloatArray], FloatArray]  # p(y), Па; y = −глубина
YEAR = 365.25 * 86_400.0
G_ACC = 9.81


@dataclass(frozen=True)
class Stage:
    """Ступень эксплуатации: длительность и давление в полости p(y)."""

    duration_s: float
    cavity_pressure: PressureFn
    label: str = ""


@dataclass(frozen=True)
class MechanicsSetup:
    e_modulus: float  # Па
    nu: float
    lithostatic: PressureFn  # p_лит(y), Па
    temperature_k: Callable[[FloatArray], FloatArray]  # T(y), К
    law: CreepLaw
    stages: tuple[Stage, ...]
    outer: str = "free"  # free — дальнее поле, roller — ячейка ряда камер
    supports: tuple[tuple[str, int], ...] = (("axis", 0), ("bottom", 1))  # (граница, u_r|u_z)
    output_every_s: float = YEAR
    eta: float = 0.1  # целевое относительное изменение напряжения за шаг
    dt_first_s: float = 60.0
    dt_max_s: float = 0.25 * YEAR


@dataclass
class MechanicsResult:
    """Сохранённые моменты: поля по элементам (среднее по точкам) и узлам."""

    times: list[float] = field(default_factory=list)  # с
    stage: list[int] = field(default_factory=list)
    u: list[FloatArray] = field(default_factory=list)  # (N, 2), м
    stress: list[FloatArray] = field(default_factory=list)  # (M, 4) полное, Па, Мандель
    creep_eq: list[FloatArray] = field(default_factory=list)  # (M,), экв. ползучесть
    volume: list[float] = field(default_factory=list)  # м³, объём полостей
    n_steps: int = 0
    volume0: float = 0.0  # м³ до выемки (недеформированная граница)

    def convergence(self) -> FloatArray:
        """Потеря объёма камеры, доля от исходного."""
        return np.asarray(1.0 - np.asarray(self.volume) / self.volume0)


def _element_mean(x: FloatArray, qw: FloatArray = QW) -> FloatArray:
    w = qw / qw.sum()
    return np.asarray(np.einsum("q,eq...->e...", w, x))


def _eq_strain(e: FloatArray) -> FloatArray:
    """Эквивалентная деформация √(2/3·e:e) девиатора (запись Манделя)."""
    tr = e[..., 0] + e[..., 1] + e[..., 2]
    d = e - (tr / 3.0)[..., None] * trace_vector(e.shape[-1])
    return np.asarray(np.sqrt(2.0 / 3.0 * np.einsum("...i,...i->...", d, d)))


def run_mechanics(
    fem: AxiFEM | Any,
    setup: MechanicsSetup,
    progress: Callable[[float], None] | None = None,
) -> MechanicsResult:
    """Выемка (упругий отклик) и ползучесть по ступеням давления.

    ``progress(доля)`` вызывается после каждого шага; исключение в нём прерывает расчёт.
    """
    t_total = sum(s.duration_s for s in setup.stages)
    fem.fixed[:] = False
    for tag, comp in setup.supports:
        fem.fix(tag, comp)
    if setup.outer == "roller":
        fem.fix("outer", 0)
    elif setup.outer != "free":
        raise ValueError(f"outer: free или roller, не {setup.outer!r}")

    d = (
        fem.elastic(setup.e_modulus, setup.nu)
        if hasattr(fem, "elastic")
        else elastic_matrix(setup.e_modulus, setup.nu)
    )
    mvec = trace_vector(d.shape[0])
    qw = getattr(fem, "qw", QW)
    d_inv = np.linalg.inv(d)
    k_el = fem.stiffness(d)
    sig0 = -setup.lithostatic(fem.yq)[..., None] * mvec
    t_k = setup.temperature_k(fem.yq)

    def load(stage: Stage) -> FloatArray:
        return fem.pressure_load(
            "cavity", lambda r, y: stage.cavity_pressure(y) - setup.lithostatic(y)
        )

    res = MechanicsResult(volume0=fem.cavity_volume())
    shape_q = fem.yq.shape
    eps_c = np.zeros((*shape_q, d.shape[0]))
    zeta = np.zeros(shape_q)
    u = np.zeros(fem.ndof)
    dsig = np.zeros((*shape_q, d.shape[0]))
    f_prev = np.zeros(fem.ndof)

    def record(t: float, i_stage: int) -> None:
        res.times.append(t)
        res.stage.append(i_stage)
        res.u.append(u.reshape(fem.mesh.n_nodes, -1).copy())
        res.stress.append(_element_mean(sig0 + dsig, qw))
        res.creep_eq.append(_element_mean(_eq_strain(eps_c), qw))
        res.volume.append(fem.cavity_volume(u))

    t = 0.0
    dt = setup.dt_first_s
    next_out = 0.0
    for i_stage, stage in enumerate(setup.stages):
        # упругий отклик на выемку или смену давления
        f = load(stage)
        du = fem.solve(k_el, f - f_prev)
        f_prev = f
        u += du
        dsig += np.einsum("ij,eqj->eqi", d, fem.strain(du))
        if i_stage == 0:
            record(t, 0)
            next_out = setup.output_every_s
        else:
            record(t, i_stage)  # сразу после скачка давления (тот же момент времени)
        t_end = t + stage.duration_s
        while t < t_end - 1e-6:
            step = min(dt, t_end - t, next_out - t)
            sig = sig0 + dsig
            rate, tang, ss, fac = setup.law.evaluate(sig, t_k, zeta, step)
            d_t = np.asarray(np.linalg.inv(d_inv + step * tang), dtype=np.float64)
            k_t = fem.stiffness(d_t)
            pre = np.einsum("eqij,eqj->eqi", d_t, step * rate)
            du = fem.solve(k_t, fem.internal(pre))
            de = fem.strain(du)
            ds = np.einsum("eqij,eqj->eqi", d_t, de) - pre
            dec = de - np.einsum("ij,eqj->eqi", d_inv, ds)
            u += du
            dsig += ds
            eps_c += dec
            zeta = np.maximum(zeta + (fac - 1.0) * ss * step, 0.0)
            t += step
            res.n_steps += 1
            if progress:
                progress(t / t_total)

            seq = von_mises(sig0 + dsig)
            mask = seq > 1e5
            if np.any(mask):
                eta = von_mises(ds)[mask] / seq[mask]
                eta_p = float(np.percentile(eta, 99.5))
                grow = setup.eta / max(eta_p, 1e-12)
                new = step * float(np.clip(grow, 0.5, 2.0))
                if step < dt and grow >= 1.0:  # шаг был урезан до момента записи
                    new = max(new, dt)
                dt = float(np.clip(new, 1.0, setup.dt_max_s))
            else:
                dt = min(2 * step, setup.dt_max_s)
            if t >= next_out - 1e-6:
                record(t, i_stage)
                next_out += setup.output_every_s
        if res.times[-1] < t - 1e-6:
            record(t, i_stage)
    return res


def lithostatic_gradient(density: float, depth_offset: float = 0.0) -> PressureFn:
    """p_лит = ρ·g·(глубина + смещение) — среднее давление вышележащих пород."""

    def p(y: FloatArray) -> FloatArray:
        return np.asarray(density * G_ACC * (-np.asarray(y) + depth_offset))

    return p


def fluid_pressure(wellhead_pa: float, density: float) -> PressureFn:
    """Давление в камере: на устье + столб жидкости (рассол, газ — малая плотность)."""

    def p(y: FloatArray) -> FloatArray:
        return np.asarray(wellhead_pa + density * G_ACC * (-np.asarray(y)))

    return p


def geothermal(surface_c: float, gradient_c_per_m: float) -> Callable[[FloatArray], FloatArray]:
    def t(y: FloatArray) -> FloatArray:
        return np.asarray(273.15 + surface_c + gradient_c_per_m * (-np.asarray(y)))

    return t
