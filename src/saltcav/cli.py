"""Командная строка: ``saltcav <команда>``.

Сейчас доступны ``version``, ``check``, ``run``, ``view`` и ``ensemble``; команды
``mechanics``, ``calibrate``, ``report`` появятся на своих этапах.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError

from saltcav import __version__
from saltcav.core.config import RunConfig, load_config

if TYPE_CHECKING:
    from saltcav.io.results import Results


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {args.config}:\n{exc}", file=sys.stderr)
        return 1
    total = sum(s.duration_days for s in cfg.schedule.stages)
    print(
        f"OK: «{cfg.name}», сетка {cfg.grid.nz}×{cfg.grid.nphi}, "
        f"{len(cfg.schedule.stages)} ступ., {total:g} сут, "
        f"{cfg.ensemble.n_realizations} реализаций"
    )
    return 0


def _simulate_results(cfg: RunConfig, args: argparse.Namespace) -> tuple[Results, float]:
    """Прогон по конфигурации (или по реализации ансамбля) → результаты и время, с."""
    from saltcav.io.results import Results
    from saltcav.leaching import simulate

    props = None
    if getattr(args, "every", None):
        lp = cfg.leaching.model_copy(update={"output_every_days": args.every})
        cfg = cfg.model_copy(update={"leaching": lp})
    if getattr(args, "realization", None) is not None:
        from saltcav.stochastic.ensemble import make_realization

        real = make_realization(cfg, args.realization, args.n, args.seed)
        cfg, props = real.cfg, real.props
        cfg = cfg.model_copy(update={"name": f"{cfg.name} #{args.realization}"})
    t0 = time.perf_counter()
    hist = simulate(cfg, props)
    return Results.from_history(hist, cfg), time.perf_counter() - t0


def _load_source(src: Path, args: argparse.Namespace) -> Results:
    """История .npz или YAML-конфигурация (тогда — прогон)."""
    from saltcav.io.results import Results

    if src.suffix == ".npz":
        return Results.load(src)
    res, elapsed = _simulate_results(load_config(src), args)
    print(f"«{res.name}»: прогон {elapsed:.1f} с")
    return res


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        res, elapsed = _simulate_results(cfg, args)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {args.config}:\n{exc}", file=sys.stderr)
        return 1
    r = res.radius[-1]
    i_max = int(np.unravel_index(np.argmax(r), r.shape)[0])
    print(f"«{res.name}»: {res.days[-1]:g} сут, {res.n_steps} шагов, {elapsed:.2f} с")
    print(f"  объём камеры      {res.volume[-1]:12.0f} м³")
    z_max = float(res.z_edges[i_max])
    print(f"  макс. радиус      {float(np.max(r)):12.2f} м на глубине {z_max:.1f} м")
    print(f"  подошва (насыпь)  {res.floor_depth[-1]:12.1f} м, {res.insoluble_bulk[-1]:.0f} м³")
    if res.shoe_move_times.size:
        print(f"  башмак поднимался над насыпью {res.shoe_move_times.size} раз(а)")
    if res.collapses.size:
        print(f"  обрушений карнизов {res.collapses.shape[0]:11d}")
    print(f"  растворено соли   {res.dissolved_salt[-1] / 1e3:12.1f} т")
    print(f"  вынесено рассолом {res.produced_salt[-1] / 1e3:12.1f} т")
    if args.out:
        res.save(args.out)
        print(f"  результаты: {args.out}  (смотреть: saltcav view {args.out})")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Просмотр по шагам: окно, скриншот, анимация, ParaView."""
    src = Path(args.source)
    try:
        res = _load_source(src, args)
    except (OSError, KeyError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {src}:\n{exc}", file=sys.stderr)
        return 1
    from saltcav.viz.frames import available_fields

    if args.field not in available_fields(res):
        print(f"поле {args.field!r} недоступно; есть {available_fields(res)}", file=sys.stderr)
        return 1
    i = res.nt - 1 if args.day is None else res.frame_at(args.day)
    batch = any((args.screenshot, args.vtk, args.animation, args.paraview))
    try:
        if args.screenshot:
            from saltcav.viz.scene import offscreen_scene

            scene = offscreen_scene(res, args.field)
            scene.show_frame(i)
            scene.screenshot(args.screenshot)
            scene.plotter.close()
            print(f"скриншот: {args.screenshot} (сутки {res.days[i]:.0f})")
        if args.vtk:
            from saltcav.viz import render

            render(res.z_edges, res.radius[i], None, vtk=args.vtk)
            print(f"VTK: {args.vtk}")
        if args.animation:
            from saltcav.viz.scene import export_animation

            n = export_animation(res, args.animation, args.field, args.fps)
            print(f"анимация: {args.animation}, {n} кадров")
        if args.paraview:
            from saltcav.viz.scene import export_paraview

            n = export_paraview(res, args.paraview)
            print(f"ParaView: {Path(args.paraview).with_suffix('.pvd')}, {n} шагов")
        if not batch:
            from saltcav.viz.viewer import run_viewer

            return run_viewer(res, src, start=i, field=args.field)
    except (RuntimeError, ModuleNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def _cmd_ensemble(args: argparse.Namespace) -> int:
    from saltcav.stochastic.ensemble import run_ensemble

    try:
        cfg = load_config(args.config)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {args.config}:\n{exc}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    ens = run_ensemble(cfg, n=args.n, seed=args.seed, jobs=args.jobs)
    elapsed = time.perf_counter() - t0
    print(f"«{cfg.name}»: {ens.n} реализаций, seed {ens.seed}, {elapsed:.1f} с")

    def q(name: str, scale: float = 1.0, fmt: str = ".1f") -> str:
        x = ens.output(name) / scale
        p10, p50, p90 = np.nanquantile(x, (0.1, 0.5, 0.9))
        return f"{p10:{fmt}} / {p50:{fmt}} / {p90:{fmt}}"

    print("                          P10 / P50 / P90")
    print(f"  объём, тыс. м³          {q('volume', 1e3)}")
    print(f"  макс. радиус, м         {q('max_radius')}")
    print(f"  кровля (R > 1 м), м     {q('roof_depth')}")
    print(f"  подошва, м              {q('floor_depth')}")
    print(f"  смещение центра, м      {q('centroid_offset', fmt='.2f')}")
    print(f"  обрушений карнизов      {q('collapses', fmt='.0f')}")
    if args.radius_limit is not None:
        p = ens.prob("max_radius", above=args.radius_limit)
        print(f"  P(макс. радиус > {args.radius_limit:g} м) = {p:.3f}")
    if ens.n >= 20:
        conv = ens.convergence("volume")
        tail = conv[int(0.8 * ens.n) :]
        drift = np.max(np.ptp(tail, axis=0) / np.abs(conv[-1]))
        print(f"  сходимость: изменение P10/P50/P90 объёма за последние 20 % — {100 * drift:.1f} %")
    sens = ens.sensitivity(("volume", "max_radius", "floor_depth"))
    if sens:
        print("  чувствительность (Спирмен, три сильнейших):")
        for out_name, d in sens.items():
            top = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"    {out_name}: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
    for out_name, label in (("max_radius", "макс. радиусом"), ("volume", "объёмом")):
        j = int(np.nanargmax(ens.output(out_name)))
        print(f"  реализация с наибольшим {label}: #{j}")
    print(
        f"  посмотреть по шагам: saltcav view {args.config} --realization N -n {ens.n} "
        f"--seed {ens.seed}"
    )
    if args.out:
        ens.save(args.out)
        print(f"  ансамбль сохранён: {args.out}")
    if args.plot:
        from saltcav.viz.envelope import plot_envelope

        layers = [(lay.top_m, lay.thickness_m) for lay in cfg.geology.layers]
        plot_envelope(ens, args.plot, f"«{cfg.name}»: {ens.n} реализаций", layers)
        print(f"  картинка: {args.plot}")
    return 0


def _realization_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--realization", type=int, help="прогнать реализацию N ансамбля")
    p.add_argument("-n", type=int, help="размер ансамбля (как при saltcav ensemble)")
    p.add_argument("--seed", type=int, help="seed ансамбля (как при saltcav ensemble)")
    p.add_argument("--every", type=float, help="сохранять каждые N сут (по умолчанию из YAML)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saltcav", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="версия пакета")
    p_check = sub.add_parser("check", help="проверить YAML-конфигурацию")
    p_check.add_argument("config", help="путь к YAML")
    p_run = sub.add_parser("run", help="детерминированный прогон одной камеры")
    p_run.add_argument("config", help="путь к YAML")
    p_run.add_argument("--out", help="сохранить результаты в .npz (для saltcav view)")
    _realization_args(p_run)
    p_ens = sub.add_parser("ensemble", help="ансамбль Монте-Карло по неопределённости")
    p_ens.add_argument("config", help="путь к YAML")
    p_ens.add_argument("-n", type=int, help="число реализаций (по умолчанию из конфигурации)")
    p_ens.add_argument("--seed", type=int, help="seed ансамбля (по умолчанию из конфигурации)")
    p_ens.add_argument("--jobs", type=int, default=1, help="процессов; 0 — все ядра")
    p_ens.add_argument("--radius-limit", type=float, help="порог для P(макс. радиус > порог)")
    p_ens.add_argument("--out", help="сохранить ансамбль в .npz")
    p_ens.add_argument("--plot", help="сохранить огибающие в .png")
    p_view = sub.add_parser("view", help="просмотр результатов по шагам (нужен .[gui])")
    p_view.add_argument("source", help="результаты .npz из run --out или YAML (с прогоном)")
    p_view.add_argument("--day", type=float, help="начальный момент, сут (по умолчанию — конец)")
    p_view.add_argument(
        "--field",
        default="radius",
        help="раскраска: radius, rate, growth, undersat, k_mult, insoluble",
    )
    p_view.add_argument("--screenshot", help="сохранить картинку .png (без окна)")
    p_view.add_argument("--vtk", help="сохранить поверхность момента --day в .vtk")
    p_view.add_argument("--animation", help="сохранить анимацию .gif или .mp4 (без окна)")
    p_view.add_argument("--fps", type=int, default=8, help="кадров в секунду анимации")
    p_view.add_argument("--paraview", help="серия шагов для ParaView (.pvd, без окна)")
    _realization_args(p_view)

    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "ensemble":
        return _cmd_ensemble(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
