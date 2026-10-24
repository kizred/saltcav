"""Картинка геомеханики: разрез массива (запас по дилатансии, напряжения, ползучесть)
и сводка во времени."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.mechanics.creep import von_mises
from saltcav.mechanics.criteria import dilatancy_fos, principal_compressive
from saltcav.mechanics.run import MechanicsRun

INK, INK2, GRID, LINE, ACCENT, SURFACE = (
    "#0b0b0b", "#52514e", "#e4e3df", "#1c5cab", "#c0392b", "#fcfcfb",
)  # fmt: skip


def _window(run: MechanicsRun) -> tuple[float, float, float, float]:
    e = run.cavity_edges
    pts = run.nodes[np.unique(e)] if e.size else run.nodes
    r_max = float(np.max(pts[:, 0]))
    y_lo, y_hi = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    h = y_hi - y_lo
    pad = max(0.8 * h, 1.5 * r_max)
    return 2.4 * r_max, y_lo - 0.5 * pad, y_hi + 0.5 * pad, r_max


def _outline(ax: Any, run: MechanicsRun, sign: float, u: np.ndarray | None = None) -> None:
    for row in run.cavity_edges:
        p = run.nodes[row[[0, 2, 1]]]
        if u is not None:
            p = p + u[row[[0, 2, 1]]]
        ax.plot(sign * p[:, 0], p[:, 1], color=INK, lw=1.0)


MECH_FIELDS: dict[str, tuple[str, str]] = {
    "fos": ("Запас по дилатансии", "RdYlGn"),
    "vm": ("σ Мизеса, МПа", "magma_r"),
    "disp": ("Смещение, см", "Blues"),
    "creep": ("Накопленная ползучесть, %", "YlOrBr"),
    "s3": ("Наименьшее главное напряжение (сжатие), МПа", "viridis"),
}


def element_field(run: MechanicsRun, i: int, key: str) -> np.ndarray:
    """Поле по элементам в момент ``i``."""
    sig = run.stress[i]
    if key == "fos":
        return np.asarray(dilatancy_fos(sig, run.dilatancy_coeff))
    if key == "vm":
        return np.asarray(von_mises(sig) / 1e6)
    if key == "disp":
        disp = np.hypot(run.u[i][:, 0], run.u[i][:, 1])
        return np.asarray(disp[run.tri[:, :3]].mean(axis=1) * 100)
    if key == "creep":
        return np.asarray(run.creep_eq[i] * 100)
    if key == "s3":
        return np.asarray(principal_compressive(sig)[..., 2] / 1e6)
    raise KeyError(f"неизвестное поле {key!r}; есть {list(MECH_FIELDS)}")


def field_limits(run: MechanicsRun, key: str) -> tuple[float, float]:
    """Шкала, общая для всех моментов (цвета не «прыгают» при листании)."""
    if key == "fos":
        return 0.0, 4.0
    vals = np.concatenate([element_field(run, i, key) for i in range(run.times.size)])
    lo = 0.0 if key in ("vm", "disp", "creep") else float(np.percentile(vals, 0.5))
    hi = float(np.percentile(vals, 99.5))
    return lo, max(hi, lo + 1e-6)


def draw_section(
    fig: Any, ax: Any, run: MechanicsRun, i: int, left: str, right: str,
    limits: dict[str, tuple[float, float]] | None = None, deformed: float = 0.0,
) -> None:  # fmt: skip
    """Разрез через ось: слева поле ``left``, справа ``right``; ``deformed`` — масштаб
    смещений для контура камеры (0 — исходный контур)."""
    from matplotlib.tri import Triangulation

    tri_r = Triangulation(run.nodes[:, 0], run.nodes[:, 1], run.tri[:, :3])
    tri_l = Triangulation(-run.nodes[:, 0], run.nodes[:, 1], run.tri[:, :3])
    x_lim, y_lo, y_hi, _ = _window(run)
    lims = limits or {}
    for tri, key, x0 in ((tri_l, left, 0.0), (tri_r, right, 0.53)):
        label, cmap = MECH_FIELDS[key]
        lo, hi = lims[key] if key in lims else field_limits(run, key)
        m = ax.tripcolor(tri, facecolors=element_field(run, i, key), cmap=cmap, vmin=lo, vmax=hi)
        cax = ax.inset_axes((x0, -0.2, 0.47, 0.03))
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(("← " if x0 == 0 else "") + label + (" →" if x0 > 0 else ""),
                     fontsize=8, color=INK2)  # fmt: skip
        cb.ax.tick_params(labelsize=7)
    u = run.u[i] * deformed if deformed > 0 else None
    _outline(ax, run, 1, u)
    _outline(ax, run, -1, u)
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("r, м")
    ax.set_ylabel("z = −глубина, м")


def draw_series(ax3: Any, ax4: Any, run: MechanicsRun, i: int) -> None:
    """Конвергенция и запас по дилатансии во времени с отметкой момента ``i``."""
    s = run.summary
    ax3.plot(s.years, s.convergence_pct, color=LINE, lw=1.8)
    ax3.axvline(s.years[i], color=ACCENT, lw=0.8)
    ax3.set_ylabel("потеря объёма, %")
    conv_t = f"Конвергенция: {s.convergence_pct[i]:.2f} %"
    rate_t = f"скорость {s.convergence_rate_pct_yr[i]:.3f} %/год"
    ax3.set_title(f"{conv_t}, {rate_t}", loc="left", fontsize=10, color=INK)
    ax4.plot(s.years, s.min_fos_wall, color=LINE, lw=1.8, label="у стенки")
    ax4.plot(s.years, s.min_fos, color=INK2, lw=1.0, ls="--", label="во всём массиве")
    ax4.axhline(1.0, color=ACCENT, lw=0.8)
    ax4.axhline(1.5, color="#eda100", lw=0.8, ls=":")
    ax4.axvline(s.years[i], color=ACCENT, lw=0.8)
    ax4.set_ylabel("мин. запас FOS")
    ax4.set_xlabel("годы после выщелачивания")
    ax4.set_ylim(0, max(3.0, float(np.nanmax(s.min_fos_wall)) * 1.1))
    ax4.legend(frameon=False, fontsize=8)
    dv = s.dilatant_volume_m3[i]
    dv_t = f"{dv:,.0f}".replace(",", " ")
    ax4.set_title(
        f"Дилатансия: мин. запас {s.min_fos_wall[i]:.2f}, зона FOS<1 — {dv_t} м³",
        loc="left", fontsize=10, color=INK,
    )  # fmt: skip
    for ax in (ax3, ax4):
        ax.grid(color=GRID, lw=0.5)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def plot_mechanics(run: MechanicsRun, out: str | Path, index: int = -1, title: str = "") -> None:
    """Сохранить PNG: два разреза (момент ``index``) и сводку во времени."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i = index % run.times.size
    fig = plt.figure(figsize=(14, 6.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gs3 = gs[2].subgridspec(2, 1, hspace=0.35)
    ax3, ax4 = fig.add_subplot(gs3[0]), fig.add_subplot(gs3[1])
    draw_section(fig, ax1, run, i, "fos", "vm")
    ax1.set_title("Запас по дилатансии | напряжения", loc="left", fontsize=10, color=INK)
    draw_section(fig, ax2, run, i, "disp", "creep")
    ax2.set_title("Смещения | накопленная ползучесть", loc="left", fontsize=10, color=INK)
    draw_series(ax3, ax4, run, i)
    head = title or f"{run.name}: геомеханика формы на {run.day:.0f} сут"
    fig.suptitle(f"{head}   |   {run.years[i]:.1f} лет", x=0.02, ha="left", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.14, wspace=0.3)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
