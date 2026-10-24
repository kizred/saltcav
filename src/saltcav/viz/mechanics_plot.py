"""Картинка геомеханики: разрез массива (запас по дилатансии, напряжения, ползучесть)
и сводка во времени."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.mechanics.creep import von_mises
from saltcav.mechanics.criteria import dilatancy_fos
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


def plot_mechanics(run: MechanicsRun, out: str | Path, index: int = -1, title: str = "") -> None:
    """Сохранить PNG: два разреза (момент ``index``) и сводку во времени."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    i = index % run.times.size
    sig = run.stress[i]
    fos = dilatancy_fos(sig, run.dilatancy_coeff)
    vm = von_mises(sig) / 1e6
    disp = np.hypot(run.u[i][:, 0], run.u[i][:, 1])
    disp_e = disp[run.tri[:, :3]].mean(axis=1) * 100  # см
    creep = run.creep_eq[i] * 100  # %
    tri_r = Triangulation(run.nodes[:, 0], run.nodes[:, 1], run.tri[:, :3])
    tri_l = Triangulation(-run.nodes[:, 0], run.nodes[:, 1], run.tri[:, :3])
    x_lim, y_lo, y_hi, _ = _window(run)

    fig = plt.figure(figsize=(14, 6.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gs3 = gs[2].subgridspec(2, 1, hspace=0.35)
    ax3, ax4 = fig.add_subplot(gs3[0]), fig.add_subplot(gs3[1])

    def panel(ax: Any, left: np.ndarray, right: np.ndarray, lk: dict[str, Any],
              rk: dict[str, Any], lt: str, rt: str) -> None:  # fmt: skip
        ml = ax.tripcolor(tri_l, facecolors=left, **lk)
        mr = ax.tripcolor(tri_r, facecolors=right, **rk)
        _outline(ax, run, 1)
        _outline(ax, run, -1)
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("r, м")
        ax.set_ylabel("z = −глубина, м")
        for m, t, x0 in ((ml, lt, 0.0), (mr, rt, 0.53)):
            cax = ax.inset_axes((x0, -0.2, 0.47, 0.03))
            cb = fig.colorbar(m, cax=cax, orientation="horizontal")
            cb.set_label(t, fontsize=8, color=INK2)
            cb.ax.tick_params(labelsize=7)

    panel(
        ax1, fos, vm,
        {"cmap": "RdYlGn", "vmin": 0.0, "vmax": 4.0},
        {"cmap": "magma_r", "vmin": 0.0, "vmax": float(np.percentile(vm, 99.5))},
        "← запас по дилатансии", "σ Мизеса, МПа →",
    )  # fmt: skip
    ax1.set_title("Запас по дилатансии | напряжения", loc="left", fontsize=10, color=INK)
    panel(
        ax2, disp_e, creep,
        {"cmap": "Blues", "vmin": 0.0, "vmax": float(np.percentile(disp_e, 99.9))},
        {"cmap": "YlOrBr", "vmin": 0.0, "vmax": max(float(np.percentile(creep, 99.5)), 1e-6)},
        "← смещение, см", "ползучесть, % →",
    )  # fmt: skip
    ax2.set_title("Смещения | накопленная ползучесть", loc="left", fontsize=10, color=INK)

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

    head = title or f"{run.name}: геомеханика формы на {run.day:.0f} сут"
    fig.suptitle(f"{head}   |   {run.years[i]:.1f} лет", x=0.02, ha="left", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.14, wspace=0.3)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
