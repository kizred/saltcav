"""Карты вероятностей по ансамблю: докуда может дойти стенка камеры.

Слева — P(R(z) > r): вероятность, что на глубине z стенка (в любом направлении)
дальше r от оси. Изолинии 10, 50, 90 % — границы, за которые камера выходит с этой
вероятностью; по ним видно, на каком расстоянии ставить соседнюю камеру и где
кровля. Справа (3D) — радиус, который превышается с вероятностью 10 % (P90), по
азимуту и глубине: в каком направлении камера уходит дальше; с ``limit`` —
P(R(z, φ) > limit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from saltcav.stochastic.ensemble import EnsembleResult

INK, INK2, GRID, SURFACE, ACCENT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c0392b"
FloatArray = NDArray[np.float64]


def exceedance(ens: EnsembleResult, r_grid: FloatArray) -> FloatArray:
    """P(наибольший по азимуту радиус на глубине z > r): (nz+1, len(r_grid))."""
    r_max = ens.radius.max(axis=2)  # (n, nz+1)
    return np.asarray((r_max[:, :, None] > r_grid[None, None, :]).mean(axis=0))


def radius_quantile_limit(ens: EnsembleResult, p: float = 0.1) -> float:
    """Радиус, который камера превышает с вероятностью ``p`` хотя бы на одной глубине."""
    return float(np.quantile(ens.radius.max(axis=(1, 2)), 1.0 - p))


def draw_probability(fig: Any, ens: EnsembleResult, limit: float | None = None,
                     title: str = "") -> None:  # fmt: skip
    """Нарисовать карты; ``limit`` — радиус для карты по азимуту (по умолчанию на ней —
    P90 радиуса)."""
    fig.clear()
    three_d = ens.phi.size > 1
    axes = fig.subplots(1, 2 if three_d else 1, squeeze=False,
                        gridspec_kw={"width_ratios": [1.2, 1] if three_d else [1]})[0]  # fmt: skip
    z = ens.z_edges
    r_top = float(np.max(ens.radius)) * 1.05
    r_grid = np.linspace(0.5, r_top, 200)
    p = exceedance(ens, r_grid)
    wide = np.nonzero(p[:, np.searchsorted(r_grid, 1.0)] > 0.0)[0]
    z_lo, z_hi = (z[wide[0]] - 5, z[wide[-1]] + 5) if wide.size else (z[0], z[-1])

    a1 = axes[0]
    m = a1.pcolormesh(r_grid, z, 100 * p, cmap="YlOrRd", vmin=0, vmax=100, shading="auto")
    cs = a1.contour(r_grid, z, 100 * p, levels=(10, 50, 90), colors=INK, linewidths=(0.8, 1.4, 0.8))
    a1.clabel(cs, fmt="%d %%", fontsize=8)
    a1.set_ylim(z_hi, z_lo)
    a1.set_xlabel("расстояние от оси скважины r, м")
    a1.set_ylabel("глубина, м")
    a1.set_title("P(стенка дальше r) по глубине", loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(m, ax=a1, fraction=0.05)
    cb.set_label("вероятность, %", color=INK2)

    if three_d:
        a2 = axes[1]
        phi = np.rad2deg(ens.phi)
        dphi = 360.0 / phi.size
        edges = np.r_[phi - dphi / 2, phi[-1] + dphi / 2]
        zb = np.r_[z[0], 0.5 * (z[1:] + z[:-1]), z[-1]]  # границы ячеек вокруг узлов
        if limit is None:
            r90 = ens.radius_quantiles((0.9,))[0]  # (nz+1, nphi)
            m2 = a2.pcolormesh(edges, zb, r90, cmap="viridis", shading="flat")
            head, label = "P90 радиуса по азимуту", "радиус, м"
        else:
            pm = 100 * ens.prob_radius_exceeds(limit)
            m2 = a2.pcolormesh(edges, zb, pm, cmap="YlOrRd", vmin=0, vmax=100, shading="flat")
            head, label = f"P(R > {limit:g} м) по азимуту", "вероятность, %"
        a2.set_ylim(z_hi, z_lo)
        a2.set_xlim(0, 360)
        a2.set_xticks(np.arange(0, 361, 90))
        a2.set_xlabel("азимут, °")
        a2.set_title(head, loc="left", fontsize=10, color=INK)
        fig.colorbar(m2, ax=a2, fraction=0.05).set_label(label, color=INK2)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title or f"Карты вероятностей: {ens.n} реализаций", x=0.02, ha="left",
                 color=INK, fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def plot_probability(ens: EnsembleResult, out: str | Path, limit: float | None = None,
                     title: str = "") -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6.5), facecolor=SURFACE)
    draw_probability(fig, ens, limit, title)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
