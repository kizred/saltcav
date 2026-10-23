"""Сравнение saltcav с SANSMIC на сценарии configs/benchmark_sansmic_direct.yaml.

Запуск из корня репозитория:

    python benchmarks/sansmic/compare.py            # таблица метрик
    python benchmarks/sansmic/compare.py --plot out.png

Эталон SANSMIC хранится в reference_direct.npz (пересчёт — run_reference.py),
поэтому сам SANSMIC для сравнения не нужен.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saltcav.core import brine
from saltcav.core.config import RunConfig, load_config
from saltcav.leaching import History, simulate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "benchmark_sansmic_direct.yaml"
REFERENCE = Path(__file__).with_name("reference_direct.npz")
MID = (1020.0, 1080.0)  # средняя часть камеры: без зоны нерастворителя и насыпи


@dataclass(frozen=True)
class Comparison:
    volume_ours: float
    volume_sansmic: float
    radius_mid_ours: float
    radius_mid_sansmic: float
    floor_ours: float
    floor_sansmic: float
    sat_out_ours: float  # средняя C/C_s на выходе за сутки 10–100
    sat_out_sansmic: float

    @property
    def volume_ratio(self) -> float:
        return self.volume_ours / self.volume_sansmic

    @property
    def radius_ratio(self) -> float:
        return self.radius_mid_ours / self.radius_mid_sansmic


def _volume_from_radius(z: np.ndarray, r: np.ndarray) -> float:
    return float(np.sum(np.pi / 3 * np.diff(z) * (r[:-1] ** 2 + r[:-1] * r[1:] + r[1:] ** 2)))


def _sg_to_saturation(sg: np.ndarray, t_c: float) -> np.ndarray:
    w = np.linspace(0.0, float(brine.saturation_mass_fraction(t_c)), 400)
    rho = brine.brine_density(w, t_c)
    rho_obs = sg * float(brine.water_density(4.0))  # sg относительно воды при 4 °C
    w_obs = np.interp(rho_obs, rho, w)
    return np.asarray(w_obs * rho_obs / float(brine.saturation_concentration(t_c)))


def compare(cfg: RunConfig | None = None, hist: History | None = None) -> Comparison:
    cfg = cfg or load_config(CONFIG)
    hist = hist or simulate(cfg)
    ref = np.load(REFERENCE)
    t_c = cfg.leaching.temperature_c

    z_o, r_o = hist.grid.z_edges, hist.radius[-1][:, 0]
    z_s, r_s = ref["z_m"], ref["radius_m"][:, -1]
    mid_o = (z_o >= MID[0]) & (z_o <= MID[1])
    mid_s = (z_s >= MID[0]) & (z_s <= MID[1])

    times = np.array(hist.times) / 86_400
    prod = np.array(hist.produced_salt)
    q = cfg.schedule.stages[0].rate_si
    c_out = np.diff(prod) / np.diff(times * 86_400) / q
    sat_o = c_out / float(brine.saturation_concentration(t_c))
    day_o = times[1:]
    sat_s = _sg_to_saturation(ref["sg_out"], t_c)
    day_s = ref["time_d"]

    return Comparison(
        volume_ours=hist.volume[-1],
        volume_sansmic=_volume_from_radius(z_s, r_s),
        radius_mid_ours=float(np.mean(r_o[mid_o])),
        radius_mid_sansmic=float(np.mean(r_s[mid_s])),
        floor_ours=hist.floor_depth[-1],
        floor_sansmic=float(z_s[-1] - ref["insol_height_m"][-1]),
        sat_out_ours=float(np.mean(sat_o[day_o >= 10])),
        sat_out_sansmic=float(np.mean(sat_s[day_s >= 10])),
    )


def _plot(cfg: RunConfig, hist: History, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.load(REFERENCE)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    ours, sans = "#2a78d6", "#eb6834"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.6), facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    days = np.array(hist.times) / 86_400
    for d, ls in ((25, ":"), (50, "--"), (100, "-")):
        i = int(np.argmin(abs(days - d)))
        j = int(np.argmin(abs(ref["time_d"] - d)))
        a1.plot(hist.radius[i][:, 0], hist.grid.z_edges, color=ours, lw=2, ls=ls)
        a1.plot(ref["radius_m"][:, j], ref["z_m"], color=sans, lw=2, ls=ls)
        a1.text(hist.radius[i][50, 0] + 0.2, 1050, f"{d} сут", color=ink2, fontsize=8)
    a1.plot([], [], color=ours, lw=2, label="saltcav")
    a1.plot([], [], color=sans, lw=2, label="SANSMIC 1.0.7")
    a1.set_ylim(1100, 1000)
    a1.set_xlabel("радиус, м")
    a1.set_ylabel("глубина, м")
    a1.set_title("Радиус стенки", loc="left", color=ink, fontweight="bold")
    a1.legend(frameon=False, loc="upper right")

    t_c = cfg.leaching.temperature_c
    q = cfg.schedule.stages[0].rate_si
    sat_o = np.diff(hist.produced_salt) / np.diff(np.array(hist.times)) / q
    sat_o /= float(brine.saturation_concentration(t_c))
    a2.plot(days[1:], sat_o, color=ours, lw=2, label="saltcav")
    a2.plot(ref["time_d"][1:], _sg_to_saturation(ref["sg_out"], t_c)[1:], color=sans, lw=2,
            label="SANSMIC")  # fmt: skip
    a2.set_ylim(0, 1.02)
    a2.set_xlabel("время, сут")
    a2.set_ylabel("C / C_s на выходе")
    a2.set_title("Насыщение рассола на выходе", loc="left", color=ink, fontweight="bold")
    a2.legend(frameon=False, loc="lower right")
    fig.suptitle("Бенчмарк: прямоток, цилиндр R = 5 м, 50 м³/ч, 100 сут", x=0.06, ha="left",
                 color=ink, fontsize=13)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, help="сохранить картинку сравнения")
    args = parser.parse_args()
    cfg = load_config(CONFIG)
    hist = simulate(cfg)
    c = compare(cfg, hist)
    rows = [
        ("объём камеры, м³", f"{c.volume_ours:.0f}", f"{c.volume_sansmic:.0f}",
         f"{c.volume_ratio:.3f}"),
        ("радиус 1020–1080 м, м", f"{c.radius_mid_ours:.2f}", f"{c.radius_mid_sansmic:.2f}",
         f"{c.radius_ratio:.3f}"),
        ("подошва, м", f"{c.floor_ours:.1f}", f"{c.floor_sansmic:.1f}", ""),
        ("C/C_s на выходе, 10–100 сут", f"{c.sat_out_ours:.2f}", f"{c.sat_out_sansmic:.2f}", ""),
    ]  # fmt: skip
    print(f"{'':30}{'saltcav':>10}{'SANSMIC':>10}{'отношение':>11}")
    for name, a, b, r in rows:
        print(f"{name:30}{a:>10}{b:>10}{r:>11}")
    if args.plot:
        _plot(cfg, hist, args.plot)
        print(f"картинка: {args.plot}")


if __name__ == "__main__":
    main()
