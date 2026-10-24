"""Картинка ансамбля: огибающие радиуса P10–P90 по глубине и сходимость квантилей."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.stochastic.ensemble import EnsembleResult

INK, INK2, GRID, BAND, LINE, SURFACE = (
    "#0b0b0b", "#52514e", "#e4e3df", "#b7d3f6", "#1c5cab", "#fcfcfb",
)  # fmt: skip


def plot_envelope(
    ens: EnsembleResult,
    out: str | Path,
    title: str = "",
    layers: list[tuple[float, float]] | None = None,
) -> None:
    """Сохранить PNG: слева огибающие эквивалентного радиуса, справа сходимость объёма."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10.5, 6.2), facecolor=SURFACE)
    draw_envelope(fig, ens, title, layers)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_envelope(
    fig: Any,
    ens: EnsembleResult,
    title: str = "",
    layers: list[tuple[float, float]] | None = None,
) -> None:
    """Нарисовать огибающие и сходимость на фигуре ``fig`` (для окна и для файла)."""
    q = ens.equivalent_radius_quantiles((0.1, 0.5, 0.9))
    z = ens.z_edges
    wide = np.nonzero(q[2] > 1.0)[0]
    z_lo, z_hi = (z[wide[0]] - 5, z[wide[-1]] + 5) if wide.size else (z[0], z[-1])

    fig.clear()
    a1, a2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.4, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for top, th in layers or []:
        a1.axhspan(top, top + th, color="#eda100", alpha=0.3, lw=0)
    for sign in (1, -1):
        a1.fill_betweenx(z, sign * q[0], sign * q[2], color=BAND, lw=0,
                         label="P10–P90" if sign == 1 else None)  # fmt: skip
        a1.plot(sign * q[1], z, color=LINE, lw=2, label="P50" if sign == 1 else None)
    r_max = float(np.max(q[2])) * 1.1
    a1.set_xlim(-r_max, r_max)
    a1.set_ylim(z_hi, z_lo)
    a1.set_xlabel("эквивалентный радиус сечения, м")
    a1.set_ylabel("глубина, м")
    a1.set_title("Огибающие формы камеры", loc="left", color=INK, fontweight="bold")
    a1.legend(frameon=False, loc="lower left", labelcolor=INK2)

    conv = ens.convergence("volume") / 1e3
    k = np.arange(1, ens.n + 1)
    for j, (lab, ls) in enumerate((("P10", ":"), ("P50", "-"), ("P90", "--"))):
        a2.plot(k, conv[:, j], color=LINE, lw=1.8, ls=ls, label=lab)
    a2.set_xlabel("число реализаций")
    a2.set_ylabel("объём камеры, тыс. м³")
    a2.set_title("Сходимость квантилей объёма", loc="left", color=INK, fontweight="bold")
    a2.legend(frameon=False, loc="center right", labelcolor=INK2)

    fig.suptitle(title or f"Ансамбль: {ens.n} реализаций, seed {ens.seed}", x=0.06, ha="left",
                 color=INK, fontsize=13)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
