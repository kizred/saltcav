"""Картинка суррогата устойчивости: распределение запаса по ансамблю, проверка
суррогата, важность признаков, сходимость вероятности."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.surrogate.features import SHORT
from saltcav.surrogate.study import KEY, TARGETS, StudyResult

INK, INK2, GRID, LINE, BAND, ACCENT, AMBER, SURFACE = (
    "#0b0b0b", "#52514e", "#e4e3df", "#1c5cab", "#b7d3f6", "#c0392b", "#eda100", "#fcfcfb",
)  # fmt: skip


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_surrogate(fig: Any, res: StudyResult, title: str = "") -> None:
    fig.clear()
    (a1, a2), (a3, a4) = fig.subplots(2, 2)
    j = res.column(KEY)
    thr = res.threshold
    mu = res.mu[:, j]
    p, p_unc = res.prob_below()

    # 1. распределение запаса по всем реализациям
    lo, hi = float(np.min(mu)), float(np.max(mu))
    bins = np.linspace(min(lo, thr) - 0.05, max(hi, thr) + 0.05, 30)
    a1.hist(mu, bins=bins, color=BAND, edgecolor=LINE, lw=0.6)
    a1.axvline(thr, color=ACCENT, lw=1.5, label=f"порог {thr:g}")
    if bins[-1] > 1.5:
        a1.axvline(1.5, color=AMBER, lw=1, ls=":", label="проектный 1,5")
    p15 = res.prob_below(threshold=1.5)[0]
    a1.set_xlabel(TARGETS[KEY][0])
    a1.set_ylabel("реализаций")
    a1.legend(frameon=False, fontsize=8)
    ci = res.prob_ci(p)
    a1.set_title(f"P(запас < {thr:g}) = {p:.3f} ± {ci:.3f} (с неопр. суррогата {p_unc:.3f}); "
                 f"P(< 1,5) = {p15:.2f}", loc="left", fontsize=10, color=INK)  # fmt: skip

    # 2. проверка суррогата: вне обучения против расчёта
    cv = res.cv.get(KEY, {})
    if "mu" in cv:
        y = res.y_train[:, j]
        a2.errorbar(y, cv["mu"], yerr=1.645 * np.asarray(cv["sd"]), fmt="o", ms=3, color=LINE,
                    ecolor=BAND, lw=1)  # fmt: skip
        lo_ = min(float(np.min(y)), float(np.min(cv["mu"]))) - 0.05
        hi_ = max(float(np.max(y)), float(np.max(cv["mu"]))) + 0.05
        lim = (lo_, hi_)
        a2.plot(lim, lim, color=INK2, lw=0.8, ls="--")
        a2.axvline(thr, color=ACCENT, lw=0.8)
        a2.axhline(thr, color=ACCENT, lw=0.8)
        a2.set_xlim(lim)
        a2.set_ylim(lim)
        cov = 100 * cv["coverage90"]
        head = f"Проверка вне обучения: R² = {cv['r2']:.2f}, 90 %-интервал — {cov:.0f} %"
        if not res.explains(KEY):
            head += "\nразброс ≈ погрешности МКЭ: запас почти постоянен"
        a2.set_title(head, loc="left", fontsize=10, color=INK)
    a2.set_xlabel("расчёт МКЭ")
    a2.set_ylabel("суррогат")

    # 3. важность признаков
    rel = res.relevance
    order = np.argsort(rel)
    names = [SHORT.get(res.feature_names[i], res.feature_names[i]) for i in order]
    a3.barh(range(len(order)), rel[order] / max(float(np.max(rel)), 1e-12), color=LINE)
    a3.set_yticks(range(len(order)))
    a3.set_yticklabels(names, fontsize=8)
    a3.set_xlabel("относительная важность (изменение запаса при перемешивании признака)")
    a3.set_title("Что определяет запас", loc="left", fontsize=10, color=INK)

    # 4. сходимость вероятности по мере уточнения
    if len(res.history) <= 2:
        a4.text(0.5, 0.5, "Уточнение не понадобилось:\nвсе реализации далеко от порога\n"
                f"(|μ − порог| > {2:g}σ)", ha="center", va="center", transform=a4.transAxes,
                color=INK2)  # fmt: skip
        a4.set_xticks([])
        a4.set_yticks([])
        a4.set_title(f"Расчётов МКЭ: {res.train.size}", loc="left", fontsize=10, color=INK)
    else:
        h = np.array(res.history)
        a4.plot(h[:, 0], h[:, 1], "o-", color=LINE, ms=4)
        a4.set_xlabel("расчётов МКЭ")
        a4.set_ylabel("P(запас < порога)")
        a4.set_title(f"Уточнение у границы: {res.train.size} расчётов"
                     + (f" + {res.hf.size} в 3D" if res.hf.size else ""),
                     loc="left", fontsize=10, color=INK)  # fmt: skip
    for ax in (a1, a2, a3, a4):
        _style(ax)
    fig.suptitle(title or f"Устойчивость по ансамблю: {res.n} реализаций", x=0.02, ha="left",
                 color=INK, fontsize=12)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def plot_surrogate(res: StudyResult, out: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
    draw_surrogate(fig, res, title)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
