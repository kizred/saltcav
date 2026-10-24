"""Схема скважины по глубине для препроцессора: сетка, прослои, башмаки, нерастворитель
по ступеням и (если есть) контур рассчитанной камеры. Только matplotlib."""

from __future__ import annotations

from typing import Any

import numpy as np

from saltcav.core.config import LayerKind, RunConfig

INK, INK2, GRID, BLUE, RED, AMBER, BAND = (
    "#0b0b0b", "#52514e", "#e4e3df", "#1c5cab", "#c0392b", "#eda100", "#b7d3f6",
)  # fmt: skip
LAYER_COLORS = {
    LayerKind.anhydrite: "#c9a227",
    LayerKind.clay: "#8a6d4b",
    LayerKind.halite: "#d9d4c7",
}


def draw_well(
    ax: Any,
    cfg: RunConfig,
    profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Нарисовать схему на осях ``ax``; ``profile`` — (глубины узлов, радиус) камеры."""
    ax.clear()
    g = cfg.grid
    stages = cfg.schedule.stages
    n = len(stages)
    depths = [g.z_top_m, g.z_bottom_m]
    for s in stages:
        depths += [s.injection_depth_m, s.production_depth_m, s.blanket_depth_m]
    for lay in cfg.geology.layers:
        depths += [lay.top_m, lay.top_m + lay.thickness_m]
    lo = min(s.blanket_depth_m for s in stages)
    hi = g.z_bottom_m
    pad = 0.08 * (hi - lo) + 2
    top, bot = lo - pad, hi + 0.5 * pad

    r_cav = 0.0
    if profile is not None:
        z, r = profile
        r_cav = float(np.max(r)) if r.size else 0.0
        ax.fill_betweenx(z, -r, r, color=BAND, lw=0, zorder=1)
        ax.plot(r, z, color=BLUE, lw=1.2, zorder=2)
        ax.plot(-r, z, color=BLUE, lw=1.2, zorder=2)
    width = max(r_cav * 1.15, 10.0 + 4.0 * n)

    for lay in cfg.geology.layers:
        ax.axhspan(lay.top_m, lay.top_m + lay.thickness_m, color=LAYER_COLORS[lay.kind],
                   alpha=0.55, lw=0, zorder=0)  # fmt: skip
        ax.text(-width, lay.top_m, f" {lay.kind.value} {lay.thickness_m:g} м", va="bottom",
                ha="left", fontsize=7, color=INK2)  # fmt: skip
    ax.axhline(g.z_bottom_m, color=INK2, lw=0.8, ls="--")
    ax.text(width, g.z_bottom_m, "низ сетки ", va="bottom", ha="right", fontsize=7, color=INK2)

    # ступени — столбики справа от оси
    xs = np.linspace(2.0, width * 0.9, n) if n > 1 else np.array([width * 0.5])
    for k, (s, x) in enumerate(zip(stages, xs, strict=True)):
        ax.plot([x, x], [top, max(s.injection_depth_m, s.production_depth_m)], color=GRID, lw=3,
                zorder=0)  # fmt: skip
        ax.plot([x - 1.5, x + 1.5], [s.blanket_depth_m] * 2, color=AMBER, lw=2.5)
        ax.plot(x - 0.5, s.injection_depth_m, "v", color=BLUE, ms=8)
        ax.plot(x + 0.5, s.production_depth_m, "v", color=RED, ms=8)
        mode = "прямоток" if s.mode.value == "direct" else "противоток"
        ax.text(x, top, f"{k + 1}\n{s.duration_days:g} сут\n{mode}\n{s.rate_m3_per_h:g} м³/ч",
                ha="center", va="top", fontsize=7, color=INK)  # fmt: skip
    ax.plot([], [], "v", color=BLUE, label="башмак закачки")
    ax.plot([], [], "v", color=RED, label="башмак отбора")
    ax.plot([], [], color=AMBER, lw=2.5, label="нерастворитель")
    if profile is not None:
        ax.plot([], [], color=BLUE, lw=1.2, label="камера (расчёт)")
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(-width, width)
    ax.set_ylim(bot, top - 0.25 * pad)
    ax.set_ylabel("глубина, м")
    ax.set_xlabel("R, м (столбики — ступени режима)")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9, edgecolor=GRID)
    ax.set_title(f"{cfg.name}: {n} ступ., {sum(s.duration_days for s in stages):g} сут",
                 loc="left", fontsize=10, color=INK)  # fmt: skip
