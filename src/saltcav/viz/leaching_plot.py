"""Картинка одного прогона выщелачивания: профили камеры по времени, объём и рассол."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.io.results import Results

INK, INK2, GRID, LINE, ACCENT, AMBER, SURFACE = (
    "#0b0b0b", "#52514e", "#e4e3df", "#1c5cab", "#c0392b", "#eda100", "#fcfcfb",
)  # fmt: skip


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_leaching(fig: Any, res: Results, title: str = "", n_profiles: int = 5) -> None:
    fig.clear()
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.2))
    a1 = fig.add_subplot(gs[:, 0])
    a2, a3 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    z = res.z_edges
    idx = np.unique(np.linspace(0, res.nt - 1, n_profiles + 1).round().astype(int))[1:]
    cmap = __import__("matplotlib").colormaps["Blues"]
    for k, i in enumerate(idx):
        r = res.radius[i]
        req = np.sqrt(np.mean(r**2, axis=1))
        c = cmap(0.35 + 0.65 * (k + 1) / len(idx))
        a1.plot(req, z, color=c, lw=1.5, label=f"{res.days[i]:g} сут")
        a1.plot(-req, z, color=c, lw=1.5)
        if res.nphi > 1 and i == idx[-1]:
            a1.fill_betweenx(z, -r.max(axis=1), -r.min(axis=1), color=c, alpha=0.2, lw=0)
            a1.fill_betweenx(z, r.min(axis=1), r.max(axis=1), color=c, alpha=0.2, lw=0,
                             label="разброс по азимуту")  # fmt: skip
    for top, th in res.layers:
        a1.axhspan(top, top + th, color=AMBER, alpha=0.35, lw=0)
    a1.axhline(res.floor_depth[-1], color=INK2, lw=0.8, ls="--")
    wide = np.nonzero(res.radius[-1].max(axis=1) > 1.0)[0]
    if wide.size:
        pad = 0.15 * (z[wide[-1]] - z[wide[0]]) + 5
        a1.set_ylim(z[wide[-1]] + pad, z[wide[0]] - pad)
    else:
        a1.invert_yaxis()
    a1.axvline(0, color=INK2, lw=0.5, ls=":")
    a1.set_xlabel("эквивалентный радиус, м")
    a1.set_ylabel("глубина, м")
    a1.set_title("Профиль камеры (пунктир — насыпь)", loc="left", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="lower right")

    a2.plot(res.days, res.volume / 1e3, color=LINE, lw=1.6)
    for t0 in res.stage_starts[1:]:
        a2.axvline(t0 / 86_400.0, color=GRID, lw=2, zorder=0)
    a2.set_ylabel("объём, тыс. м³")
    a2.set_title(f"Объём {res.volume[-1] / 1e3:,.1f} тыс. м³".replace(",", " "),
                 loc="left", fontsize=10, color=INK)  # fmt: skip
    if res.outlet_concentration is not None:
        a3.plot(res.days, res.outlet_concentration, color=LINE, lw=1.4)
        a3.axhline(res.c_sat, color=ACCENT, lw=0.8, ls=":", label="насыщение")
        a3.set_ylabel("рассол на выходе, кг/м³")
        a3.legend(frameon=False, fontsize=8)
    a3.set_xlabel("сутки")
    for ax in (a1, a2, a3):
        _style(ax)
    fig.suptitle(title or f"{res.name}: выщелачивание, {res.days[-1]:g} сут", x=0.02,
                 ha="left", color=INK, fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def plot_leaching(res: Results, out: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7.5), facecolor=SURFACE)
    draw_leaching(fig, res, title)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
