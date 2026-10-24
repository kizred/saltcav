"""Картинка калибровки: рассол на выходе, профили сонара, апостериорные параметров."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.calibration.fit import CalibResult

INK, INK2, GRID, LINE, BAND, ACCENT, AMBER, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
    "#1c5cab",
    "#b7d3f6",
    "#c0392b",
    "#eda100",
    "#fcfcfb",
)
SURVEY = ("#1c5cab", "#2e8b57", "#8e44ad", "#d35400", "#7f8c8d", "#c0392b")


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_calibration(fig: Any, res: CalibResult, title: str = "") -> None:
    fig.clear()
    gs = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[:, 1])
    d = len(res.names)
    sub = gs[1, 0].subgridspec(1, d, wspace=0.35)
    axes_p = [fig.add_subplot(sub[0, j]) for j in range(d)]

    # 1. рассол на выходе
    if res.curve_days.size:
        a1.fill_between(
            res.curve_days,
            res.curve_lo,
            res.curve_hi,
            color=BAND,
            lw=0,
            label="90 % по апостериорному",
        )
        a1.plot(res.curve_days, res.curve_map, color=LINE, lw=1.4, label="модель (MAP)")
    if res.brine_day.size:
        f = res.brine_fit
        a1.errorbar(
            res.brine_day[f],
            res.brine_obs[f],
            yerr=res.brine_sigma[f],
            fmt="o",
            ms=3,
            color=INK,
            ecolor=INK2,
            elinewidth=0.6,
            label="замеры (подгонка)",
        )
        if (~f).any():
            a1.errorbar(
                res.brine_day[~f],
                res.brine_obs[~f],
                yerr=res.brine_sigma[~f],
                fmt="o",
                ms=3,
                mfc="white",
                color=ACCENT,
                ecolor=ACCENT,
                elinewidth=0.6,
                label="замеры (проверка)",
            )
    if res.holdout_day is not None:
        a1.axvline(res.holdout_day, color=ACCENT, lw=0.8, ls="--")
    for sd in res.sonar_days:
        a1.axvline(sd, color=GRID, lw=2, zorder=0)
    a1.set_xlabel("сутки")
    a1.set_ylabel("концентрация на выходе, кг/м³")
    rms = res.rms_fit.get("brine")
    head = "Рассол на выходе" + (f": СКО {rms:.1f} кг/м³" if rms is not None else "")
    if "brine" in res.rms_holdout:
        head += f", на проверке {res.rms_holdout['brine']:.1f}"
    a1.set_title(head, loc="left", fontsize=10, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="lower right")

    # 2. профили сонара
    for i, sd in enumerate(res.sonar_days):
        c = SURVEY[i % len(SURVEY)]
        z = res.z_edges
        a2.fill_betweenx(z, res.profile_lo[i], res.profile_hi[i], color=c, alpha=0.15, lw=0)
        a2.plot(res.profile_map[i], z, color=c, lw=1.2)
        dz, rz = res.sonar_obs[i]
        fit = bool(res.sonar_fit[i])
        a2.plot(
            rz,
            dz,
            "o",
            ms=2.5,
            color=c,
            mfc=c if fit else "white",
            label=f"{sd:g} сут" + ("" if fit else " (проверка)"),
        )
    a2.invert_yaxis()
    a2.set_xlabel("эквивалентный радиус, м")
    a2.set_ylabel("глубина, м")
    rms = res.rms_fit.get("sonar")
    head = "Сонар: точки — замер, линия — модель"
    if rms is not None:
        head += f"\nСКО {rms:.2f} м" + (
            f", на проверке {res.rms_holdout['sonar']:.2f} м" if "sonar" in res.rms_holdout else ""
        )
    a2.set_title(head, loc="left", fontsize=10, color=INK)
    if res.sonar_days.size:
        a2.legend(frameon=False, fontsize=8, loc="lower right")
        r_obs = [r for _, r in res.sonar_obs if r.size]
        rmax = max([float(np.max(res.profile_hi))] + [float(np.max(r)) for r in r_obs])
        a2.set_xlim(0, rmax * 1.08)
        zs = [d for d, _ in res.sonar_obs if d.size]
        if zs:
            lo, hi = min(float(np.min(z)) for z in zs), max(float(np.max(z)) for z in zs)
            pad = 0.1 * (hi - lo) + 2
            a2.set_ylim(hi + pad, lo - pad)

    # 3. апостериорные параметров
    th = res.theta_samples()
    q = res.quantiles()
    for j, ax in enumerate(axes_p):
        v = th[:, j]
        lg = res.log_scale[j]
        vals = np.log10(v) if lg else v
        ax.hist(vals, bins=30, color=BAND, edgecolor=LINE, lw=0.4, density=True)
        ax.axvline(np.log10(res.theta_map()[j]) if lg else res.theta_map()[j], color=LINE, lw=1.2)
        if res.names[j] in res.truth:
            t = res.truth[res.names[j]]
            ax.axvline(np.log10(t) if lg else t, color=ACCENT, lw=1.2, ls="--")
        ax.set_yticks([])
        ax.set_xlabel(("lg " if lg else "") + res.labels[j], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{q[1, j]:.3g}\n[{q[0, j]:.3g}; {q[2, j]:.3g}]", fontsize=8, color=INK)
        ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(3))
    if axes_p:
        txt = "Апостериорные: медиана [90 %]; синяя — MAP"
        if res.truth:
            txt += ", красная — истина"
        axes_p[0].text(0, 1.32, txt, transform=axes_p[0].transAxes, fontsize=10, color=INK)

    for ax in (a1, a2, *axes_p):
        _style(ax)
    m = {"map": "MAP + лаплас", "mcmc": "MCMC"}.get(res.method, res.method)
    head = title or "Калибровка по данным скважины"
    fig.suptitle(
        f"{head}: {m}, χ²/N = {res.chi2_reduced:.2f}, {res.n_runs} прогонов, {res.seconds:.0f} с",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def plot_calibration(res: CalibResult, out: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    draw_calibration(fig, res, title)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
