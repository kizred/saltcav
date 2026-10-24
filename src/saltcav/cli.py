"""Командная строка: ``saltcav <команда>``.

Сейчас доступны ``version``, ``check``, ``run`` и ``view``; команды ``ensemble``,
``mechanics``, ``calibrate``, ``report`` появятся на своих этапах.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from saltcav import __version__
from saltcav.core.config import load_config


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


def _cmd_run(args: argparse.Namespace) -> int:
    from saltcav.leaching import simulate  # импорт здесь: check и version не тянут модель

    try:
        cfg = load_config(args.config)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {args.config}:\n{exc}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    hist = simulate(cfg)
    elapsed = time.perf_counter() - t0
    cav = hist.cavern()
    r = cav.radius
    i_max = int(np.unravel_index(np.argmax(r), r.shape)[0])
    print(f"«{cfg.name}»: {hist.times[-1] / 86_400:g} сут, {hist.n_steps} шагов, {elapsed:.2f} с")
    print(f"  объём камеры      {hist.volume[-1]:12.0f} м³")
    z_max = float(hist.grid.z_edges[i_max])
    print(f"  макс. радиус      {float(np.max(r)):12.2f} м на глубине {z_max:.1f} м")
    print(f"  подошва (насыпь)  {hist.floor_depth[-1]:12.1f} м, {hist.insoluble_bulk[-1]:.0f} м³")
    if hist.n_shoe_moves:
        print(f"  башмак поднимался над насыпью на {hist.n_shoe_moves} шагах из {hist.n_steps}")
    print(f"  растворено соли   {hist.dissolved_salt[-1] / 1e3:12.1f} т")
    print(f"  вынесено рассолом {hist.produced_salt[-1] / 1e3:12.1f} т")
    if args.out:
        out = Path(args.out)
        np.savez_compressed(
            out,
            times_s=np.array(hist.times),
            z_edges_m=hist.grid.z_edges,
            radius_m=np.array(hist.radius),
            concentration_kg_m3=np.array(hist.concentration),
            volume_m3=np.array(hist.volume),
            produced_salt_kg=np.array(hist.produced_salt),
            dissolved_salt_kg=np.array(hist.dissolved_salt),
            salt_in_brine_kg=np.array(hist.salt_in_brine),
            floor_depth_m=np.array(hist.floor_depth),
            insoluble_bulk_m3=np.array(hist.insoluble_bulk),
        )
        print(f"  история сохранена в {out}")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """3D-вид камеры из конфигурации (с прогоном) или сохранённой истории .npz."""
    from saltcav.viz import render

    src = Path(args.source)
    try:
        if src.suffix == ".npz":
            data = np.load(src)
            times_d = data["times_s"] / 86_400
            z_edges, radius_all = data["z_edges_m"], data["radius_m"]
            floors = data["floor_depth_m"] if "floor_depth_m" in data else None
            name = src.stem
        else:
            from saltcav.leaching import simulate

            cfg = load_config(src)
            hist = simulate(cfg)
            times_d = np.array(hist.times) / 86_400
            z_edges, radius_all = hist.grid.z_edges, np.array(hist.radius)
            floors = np.array(hist.floor_depth)
            name = cfg.name
    except (OSError, KeyError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {src}:\n{exc}", file=sys.stderr)
        return 1
    i = len(times_d) - 1 if args.day is None else int(np.argmin(np.abs(times_d - args.day)))
    radius = radius_all[i]
    floor = float(floors[i]) if floors is not None else None
    vol = float(
        np.sum(
            np.pi
            / 3
            * np.diff(z_edges)[:, None]
            * (radius[:-1] ** 2 + radius[:-1] * radius[1:] + radius[1:] ** 2)
        )
        / radius.shape[1]
    )
    title = f"{name}: {times_d[i]:.0f} сут, V = {vol:.0f} м³"
    try:
        render(z_edges, radius, floor, title, screenshot=args.screenshot, vtk=args.vtk)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    for label, path in (("скриншот", args.screenshot), ("VTK", args.vtk)):
        if path:
            print(f"{label}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saltcav", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="версия пакета")
    p_check = sub.add_parser("check", help="проверить YAML-конфигурацию")
    p_check.add_argument("config", help="путь к YAML")
    p_run = sub.add_parser("run", help="детерминированный прогон одной камеры")
    p_run.add_argument("config", help="путь к YAML")
    p_run.add_argument("--out", help="сохранить историю в .npz")
    p_view = sub.add_parser("view", help="3D-вид камеры (нужен PyVista)")
    p_view.add_argument("source", help="YAML-конфигурация или история .npz из run --out")
    p_view.add_argument("--day", type=float, help="момент, сут (по умолчанию — последний)")
    p_view.add_argument("--screenshot", help="сохранить картинку .png вместо окна")
    p_view.add_argument("--vtk", help="сохранить поверхность в .vtk для ParaView")

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
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
