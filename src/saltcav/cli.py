"""Командная строка: ``saltcav <команда>``.

Команды: ``version``, ``schema``, ``gui``, ``check``, ``run``, ``view``, ``ensemble``,
``mechanics``, ``surrogate``, ``data``, ``synth``, ``calibrate``, ``report``.
"""

from __future__ import annotations

import argparse
import subprocess
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
    from saltcav.mechanics.run import MechanicsRun


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
    from saltcav.core.config import config_warnings

    for loc, msg in config_warnings(cfg):
        print(f"  внимание, {'.'.join(loc)}: {msg}")
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


SCHEMA_PATH = Path("schema") / "saltcav-config.schema.json"


def _cmd_schema(args: argparse.Namespace) -> int:
    """JSON Schema конфигурации для подсказок в PyCharm / VS Code."""
    import json

    from saltcav.core.config import config_schema

    text = json.dumps(config_schema(), ensure_ascii=False, indent=2) + "\n"
    if args.out == "-":
        print(text, end="")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"схема: {out}")
    print("в начало YAML-файла: # $schema: <путь к схеме относительно файла>")
    return 0


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


def _cmd_mechanics(args: argparse.Namespace) -> int:
    """Геомеханика по форме камеры: ползучесть и критерии устойчивости."""
    src = Path(args.source)
    try:
        if src.suffix == ".npz" and _is_mechanics(src):
            from saltcav.mechanics.run import MechanicsRun

            run = MechanicsRun.load(src)
        else:
            run = _mechanics_from_source(src, args)
    except (OSError, KeyError, ValidationError, ValueError, RuntimeError) as exc:
        print(f"Ошибка в {src}:\n{exc}", file=sys.stderr)
        return 1
    s = run.summary
    kind = "3D" if run.dim == 3 else "осесимметрия"
    print(
        f"«{run.name}» ({kind}): форма на {run.day:.0f} сут, {run.tri.shape[0]} элементов, ", end=""
    )
    print(f"{run.n_steps} шагов")
    v_fem = f"{run.volume0:,.0f}".replace(",", " ")
    v_leach = f"{run.leaching_volume:,.0f}".replace(",", " ")
    print(f"  объём полости в МКЭ {v_fem} м³ (по выщелачиванию {v_leach})")
    if run.material is not None and run.material_names:
        counts = np.bincount(run.material, minlength=len(run.material_names))
        print("  материалы: " + ", ".join(
            f"{n} — {c} эл." for n, c in zip(run.material_names, counts, strict=True)))  # fmt: skip
    head = ("годы", "потеря V, %", "%/год", "FOS у стенки: мин / 5 %", "зона FOS<1, м³")
    print("  " + "   ".join(head) + "   σ_t−p: мин / 5 %, МПа")
    marks = (int(np.argmin(np.abs(s.years - y))) for y in (1, 5, 10))
    picks = sorted({0, *marks, len(s.years) - 1})
    for k in picks:
        print(
            f"  {s.years[k]:5.1f}  {s.convergence_pct[k]:10.2f}"
            f"  {s.convergence_rate_pct_yr[k]:7.3f}"
            f"  {s.min_fos_wall[k]:13.2f} / {p5(s.fos_wall_p5, k):4.2f}"
            f"  {s.dilatant_volume_m3[k]:15.0f}"
            f"  {s.tension_margin_mpa[k]:10.1f} / {p5(s.tension_margin_p5_mpa, k):4.1f}"
        )
    if s.layer_tension_ratio is not None and s.layer_tension_mpa is not None:
        k = int(np.argmax(s.layer_tension_ratio))
        verdict = "возможна трещина (отрыв)" if s.layer_tension_ratio[k] > 1 else "ниже прочности"
        lt, lr = max(s.layer_tension_mpa[k], 0.0), s.layer_tension_ratio[k]
        print(f"  прослои: наибольшее растяжение {lt:.1f} МПа = {lr:.2f} прочности "
              f"на {s.years[k]:.1f} год — {verdict}")  # fmt: skip
    if args.out:
        run.save(args.out)
        print(f"  результат: {args.out}")
    if args.plot:
        from saltcav.viz.mechanics_plot import plot_mechanics

        idx = -1 if args.at_year is None else int(np.argmin(np.abs(s.years - args.at_year)))
        plot_mechanics(run, args.plot, idx)
        print(f"  картинка: {args.plot}")
    return 0


def p5(arr: np.ndarray | None, k: int) -> float:
    return float("nan") if arr is None else float(arr[k])


def _is_mechanics(path: Path) -> bool:
    with np.load(path) as d:
        return "kind" in d and str(d["kind"]) == "saltcav-mechanics"


def _mechanics_from_source(src: Path, args: argparse.Namespace) -> MechanicsRun:
    import yaml

    from saltcav.core.config import MechanicsConfig, RunConfig
    from saltcav.mechanics.run import run_from_shape

    res = _load_source(src, args)
    layers = None  # прослои — из конфигурации формы (у реализации — её отметки)
    shape_cfg = (
        RunConfig.model_validate(yaml.safe_load(res.config_yaml)) if res.config_yaml else None
    )
    if args.config:
        full = load_config(args.config)
        mc, layers = full.mechanics, list(full.geology.layers)
    elif shape_cfg is not None:
        mc = shape_cfg.mechanics
    else:
        mc = MechanicsConfig()
    if shape_cfg is not None:
        layers = list(shape_cfg.geology.layers)
    if args.model:
        mc = mc.model_copy(update={"model": args.model})
    if args.years is not None:
        if len(mc.stages) != 1:
            raise ValueError("--years только для одной ступени давления; иначе задайте в YAML")
        st = mc.stages[0].model_copy(update={"duration_years": args.years})
        mc = mc.model_copy(update={"stages": [st]})
    day = args.day if args.day is not None else mc.day
    i = res.nt - 1 if day is None else res.frame_at(day)
    t0 = time.perf_counter()
    total = sum(st.duration_years for st in mc.stages)

    def progress(frac: float) -> None:
        el = time.perf_counter() - t0
        print(f"\r  геомеханика: {100 * frac:5.1f} % ({frac * total:.1f} из {total:g} лет, "
              f"{el:.0f} с)", end="", flush=True)  # fmt: skip

    run = run_from_shape(
        mc, res.z_edges, res.radius[i], res.name, float(res.days[i]), float(res.volume[i]),
        progress=progress if sys.stdout.isatty() or mc.model == "3d" else None, layers=layers,
    )  # fmt: skip
    print(f"\rгеомеханика: {time.perf_counter() - t0:.1f} с" + " " * 40)
    return run


def _cmd_surrogate(args: argparse.Namespace) -> int:
    """Вероятность потери устойчивости по ансамблю через суррогат геомеханики."""
    from saltcav.stochastic.ensemble import EnsembleResult, run_ensemble
    from saltcav.surrogate.study import KEY, TARGETS, run_study

    try:
        cfg = load_config(args.config)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {args.config}:\n{exc}", file=sys.stderr)
        return 1
    pairs = (
        ("n_train", args.train),
        ("adaptive", args.adaptive),
        ("n_3d", args.n3d),
        ("fos_threshold", args.threshold),
    )
    upd = {k: v for k, v in pairs if v is not None}
    cfg = cfg.model_copy(update={"surrogate": cfg.surrogate.model_copy(update=upd)})
    t0 = time.perf_counter()
    if args.ensemble:
        ens = EnsembleResult.load(args.ensemble)
        print(f"ансамбль из {args.ensemble}: {ens.n} реализаций")
    else:
        ens = run_ensemble(cfg, n=args.n, seed=args.seed, jobs=args.jobs)
        print(f"ансамбль: {ens.n} реализаций, {time.perf_counter() - t0:.0f} с")
        if args.save_ensemble:
            ens.save(args.save_ensemble)

    def progress(frac: float, text: str) -> None:
        el = time.perf_counter() - t0
        print(f"\r  {text} ({el:.0f} с)" + " " * 10, end="", flush=True)

    from saltcav.mechanics.run import ensemble_layers

    res = run_study(cfg, ens.z_edges, ens.radius, seed=ens.seed, jobs=args.jobs, progress=progress,
                    layers=ensemble_layers(cfg, ens))  # fmt: skip
    print(f"\rсуррогат: {res.train.size} расчётов механики"
          + (f" + {res.hf.size} в 3D" if res.hf.size else "")
          + f", {time.perf_counter() - t0:.0f} с" + " " * 30)  # fmt: skip
    thr = res.threshold
    p, p_unc = res.prob_below()
    p15, _ = res.prob_below(threshold=1.5)
    print(f"  P(запас по дилатансии < {thr:g}) = {p:.3f} ± {res.prob_ci(p):.3f}"
          f"   (с неопределённостью суррогата {p_unc:.3f})")  # fmt: skip
    print(f"  P(запас < 1.5)                  = {p15:.3f}")
    print("                                         P10 / P50 / P90")
    for name, (label, _) in TARGETS.items():
        q = res.quantiles(name)
        print(f"  {label:<44.44} {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    print("  проверка вне обучения (R², доля попаданий в 90 %-интервал):")
    for name, d in res.cv.items():
        note = "" if res.explains(name) else "  — разброс ≈ погрешности расчёта, выход ≈ постоянный"
        print(f"    {name:<16} R² = {d['r2']:5.2f}, {100 * d['coverage90']:3.0f} %{note}")
    crit = res.critical(5)
    j = res.column(KEY)
    print("  наименьший запас — реализации: " + ", ".join(
        f"#{i} ({res.mu[i, j]:.2f})" for i in crit))  # fmt: skip
    if args.out:
        res.save(args.out)
        print(f"  результат: {args.out}")
    if args.plot:
        from saltcav.viz.surrogate_plot import plot_surrogate

        plot_surrogate(res, args.plot, f"«{cfg.name}»: устойчивость по {res.n} реализациям")
        print(f"  картинка: {args.plot}")
    return 0


def _load_cfg(path: str) -> RunConfig | None:
    try:
        return load_config(path)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Ошибка в {path}:\n{exc}", file=sys.stderr)
        return None


def _cmd_data(args: argparse.Namespace) -> int:
    """Прочитать и проверить набор данных скважины."""
    from saltcav.io.fielddata import load_field_data

    try:
        data = load_field_data(args.folder)
    except (OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    cfg = _load_cfg(args.config) if args.config else None
    if args.config and cfg is None:
        return 1
    print(f"«{data.name}»" + (f", {data.temperature_c:g} °C" if data.temperature_c else ""))
    if data.operations:
        print(f"  режим: {len(data.operations)} ступ., {data.duration_days:g} сут")
    if data.brine_day.size:
        b, c = data.brine_day, data.brine_conc
        print(f"  рассол: {b.size} замеров, {b[0]:g}–{b[-1]:g} сут, "
              f"{float(np.min(c)):.0f}–{float(np.max(c)):.0f} кг/м³")  # fmt: skip
    for s in data.sonar:
        az = "" if s.azimuth_deg is None else f", {np.unique(s.azimuth_deg).size} азимутов"
        z0, z1 = float(np.min(s.depth)), float(np.max(s.depth))
        print(f"  сонар {s.day:g} сут: {s.depth.size} точек, {z0:g}–{z1:g} м{az}")
    if data.layers:
        print(f"  прослоев: {len(data.layers)}")
    if data.notes:
        print(f"  примечание: {data.notes}")
    issues = data.issues(cfg)
    for msg in issues:
        print(f"  внимание: {msg}")
    if cfg is not None:
        try:
            data.apply(cfg)
        except (ValidationError, ValueError) as exc:
            print(f"  режим из данных не согласуется с {args.config}:\n{exc}", file=sys.stderr)
            return 1
    print("OK" if not issues else f"замечаний: {len(issues)}")
    return 0


def _parse_truth(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise ValueError(f"--truth {it}: нужно имя=значение")
        out[k.strip()] = float(v)
    return out


def _cmd_synth(args: argparse.Namespace) -> int:
    """Синтетический набор «как от компании» или открытый эталон SANSMIC."""
    from saltcav.calibration import sansmic_dataset, synthesize, write_truth
    from saltcav.io.fielddata import write_field_data

    if args.sansmic:
        data = sansmic_dataset()
        write_field_data(args.out, data)
        print(f"открытый набор SANSMIC: {args.out}")
        return 0
    if not args.config:
        print("нужна конфигурация (или --sansmic)", file=sys.stderr)
        return 1
    cfg = _load_cfg(args.config)
    if cfg is None:
        return 1
    try:
        truth = _parse_truth(args.truth or [])
        days = [float(v) for v in args.sonar_days.split(",")] if args.sonar_days else None
        data = synthesize(cfg, truth, brine_every_days=args.brine_every, sonar_days=days,
                          sigma_c=args.sigma_c, sigma_r=args.sigma_r, seed=args.seed,
                          name=args.name or f"синтетика «{cfg.name}»")  # fmt: skip
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    write_field_data(args.out, data, excel=args.excel)
    write_truth(args.out, truth)
    print(f"синтетический набор: {args.out} (истина — truth.yaml: "
          + ", ".join(f"{k}={v:g}" for k, v in truth.items()) + ")")  # fmt: skip
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Байесовская калибровка параметров по данным скважины."""
    from saltcav.calibration import calibrate, read_truth
    from saltcav.core.config import dump_config
    from saltcav.io.fielddata import load_field_data

    cfg = _load_cfg(args.config)
    if cfg is None:
        return 1
    try:
        data = load_field_data(args.data)
        for msg in data.issues(cfg):
            print(f"  внимание: {msg}")
        if args.steps:
            cfg = cfg.model_copy(
                update={"calibration": cfg.calibration.model_copy(update={"steps": args.steps})}
            )
        t0 = time.perf_counter()

        def progress(frac: float, text: str) -> None:
            el = time.perf_counter() - t0
            print(f"\r  {text} ({el:.0f} с)" + " " * 10, end="", flush=True)

        res = calibrate(cfg, data, method=args.method, holdout_day=args.holdout_days,
                        jobs=args.jobs, progress=progress, truth=read_truth(args.data))  # fmt: skip
    except (OSError, ValueError, ValidationError) as exc:
        print(f"\nОшибка: {exc}", file=sys.stderr)
        return 1
    m = {"map": "MAP + лаплас", "mcmc": "MCMC"}[res.method]
    print(f"\rкалибровка ({m}): {res.n_runs} прогонов, {res.seconds:.0f} с" + " " * 30)
    head = res.table_header()
    rows = res.table()
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    for r in rows:
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    acc = f"; доля приёма MCMC {res.acceptance:.2f}" if res.method == "mcmc" else ""
    print(f"  χ²/N = {res.chi2_reduced:.2f} по {res.n_fit} замерам{acc}")
    unit = {"brine": ("рассол", "кг/м³"), "sonar": ("сонар", "м")}

    def rms(d: dict[str, float]) -> str:
        return ", ".join(f"{unit[k][0]} {v:.2f} {unit[k][1]}" for k, v in d.items())

    print(f"  СКО невязок: {rms(res.rms_fit)}")
    if res.rms_holdout:
        ho = rms(res.rms_holdout)
        cov = (
            ""
            if np.isnan(res.coverage_holdout)
            else f"; в 90 %-полосе {100 * res.coverage_holdout:.0f} %"
        )
        print(f"  проверка после {res.holdout_day:g} сут: СКО {ho}{cov}")
    for note in res.notes():
        print(f"  внимание: {note}")
    if args.out:
        res.save(args.out)
        print(f"  результат: {args.out}")
    if args.plot:
        from saltcav.viz.calibration_plot import plot_calibration

        plot_calibration(res, args.plot, f"«{data.name}»")
        print(f"  картинка: {args.plot}")
    if args.write_config:
        new = res.calibrated_config(cfg, data)
        new = new.model_copy(update={"name": f"{cfg.name}-calibrated"})
        dump_config(new, args.write_config)
        print(f"  откалиброванная конфигурация: {args.write_config}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Отчёт HTML в один клик: готовые результаты + досчитанные разделы."""
    import saltcav.api as sc
    from saltcav.report import PARTS, make_report

    cfg = _load_cfg(args.config)
    if cfg is None:
        return 1
    compute = list(PARTS) if args.all else [p for p in (args.compute or "leaching").split(",") if p]
    try:
        kw: dict[str, object] = {}
        for key in ("results", "ensemble", "mechanics", "stability", "calibration"):
            path = getattr(args, key)
            if path:
                kw[key] = sc.load(path)
        if args.data:
            kw["data"] = sc.data(args.data)
            if "calibration" not in kw and args.calibrate:
                print("калибровка…")
                kw["calibration"] = sc.calibrate(cfg, args.data, jobs=args.jobs)
        t0 = time.perf_counter()

        def progress(frac: float, text: str) -> None:
            print(f"\r  {text} ({time.perf_counter() - t0:.0f} с)" + " " * 10, end="", flush=True)

        out = make_report(cfg, args.out, compute=compute, n=args.n, jobs=args.jobs,
                          progress=progress, **kw)  # type: ignore[arg-type]  # fmt: skip
    except (OSError, ValueError, ValidationError, RuntimeError) as exc:
        print(f"\nОшибка: {exc}", file=sys.stderr)
        return 1
    print(f"\rотчёт: {out} ({out.stat().st_size / 1e6:.1f} МБ, {time.perf_counter() - t0:.0f} с)"
          + " " * 20)  # fmt: skip
    print("  открыть в браузере; в PDF — «Печать → Сохранить как PDF»")
    return 0


def _cmd_shortcut(args: argparse.Namespace) -> int:
    """Ярлык окна со значком: рабочий стол и меню «Пуск» / приложений."""
    from saltcav.shortcut import create_shortcuts

    try:
        made = create_shortcuts(desktop=not args.no_desktop, menu=not args.no_menu,
                                workdir=Path(args.workdir) if args.workdir else None)  # fmt: skip
    except (OSError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    for p in made:
        print(f"ярлык: {p}")
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


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from saltcav.gui.app import main as gui_main
    except ImportError as exc:
        print(f'для окна нужен PySide6: pip install -e ".[gui]" ({exc})', file=sys.stderr)
        return 1
    return gui_main([args.file] if args.file else [])


def _realization_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--realization", type=int, help="прогнать реализацию N ансамбля")
    p.add_argument("-n", type=int, help="размер ансамбля (как при saltcav ensemble)")
    p.add_argument("--seed", type=int, help="seed ансамбля (как при saltcav ensemble)")
    p.add_argument("--every", type=float, help="сохранять каждые N сут (по умолчанию из YAML)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saltcav", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="версия пакета")
    p_schema = sub.add_parser("schema", help="JSON Schema конфигурации (подсказки в редакторе)")
    p_schema.add_argument(
        "--out",
        default=str(SCHEMA_PATH),
        help=f"куда записать (по умолчанию {SCHEMA_PATH}; - — вывод)",
    )
    p_gui = sub.add_parser("gui", help="окно: параметры, расчёт, просмотр (нужен .[gui])")
    p_gui.add_argument("file", nargs="?", help="YAML-конфигурация или результаты .npz")
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
    p_sur = sub.add_parser("surrogate", help="вероятность потери устойчивости по ансамблю")
    p_sur.add_argument("config", help="путь к YAML (разделы ensemble, mechanics, surrogate)")
    p_sur.add_argument("--ensemble", help="готовый ансамбль .npz (иначе — расчёт)")
    p_sur.add_argument("-n", type=int, help="размер ансамбля, если считать заново")
    p_sur.add_argument("--seed", type=int, help="seed ансамбля")
    p_sur.add_argument("--jobs", type=int, default=0, help="процессов (0 — все ядра)")
    p_sur.add_argument("--train", type=int, help="начальная выборка (surrogate.n_train)")
    p_sur.add_argument("--adaptive", type=int, help="уточнение у границы (surrogate.adaptive)")
    p_sur.add_argument("--n3d", type=int, help="3D-расчётов для поправки (surrogate.n_3d)")
    p_sur.add_argument("--threshold", type=float, help="порог запаса (surrogate.fos_threshold)")
    p_sur.add_argument("--save-ensemble", help="сохранить посчитанный ансамбль .npz")
    p_sur.add_argument("--out", help="сохранить результат .npz")
    p_sur.add_argument("--plot", help="сохранить картинку .png")
    p_mech = sub.add_parser("mechanics", help="геомеханика: ползучесть и устойчивость камеры")
    p_mech.add_argument(
        "source", help="результаты .npz из run --out, YAML (с прогоном) или .npz из mechanics"
    )
    p_mech.add_argument("--day", type=float, help="момент формы, сут (по умолчанию конец)")
    p_mech.add_argument("--config", help="взять раздел mechanics из этого YAML")
    p_mech.add_argument("--years", type=float, help="длительность (одна ступень давления), лет")
    p_mech.add_argument(
        "--model", choices=("axisym", "3d"), help="постановка (по умолчанию из YAML)"
    )
    p_mech.add_argument("--out", help="сохранить результат .npz")
    p_mech.add_argument("--plot", help="сохранить картинку .png")
    p_mech.add_argument(
        "--at-year", type=float, help="момент для картинки, лет (по умолчанию конец)"
    )
    _realization_args(p_mech)
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
    p_data = sub.add_parser("data", help="проверить набор данных скважины (папка CSV)")
    p_data.add_argument("folder", help="папка: operations.csv, brine.csv, sonar.csv, layers.csv")
    p_data.add_argument("--config", help="сверить с конфигурацией (сетка, режим)")
    p_syn = sub.add_parser("synth", help="синтетический набор данных или открытый SANSMIC")
    p_syn.add_argument("config", nargs="?", help="YAML: камера и режим синтетики")
    p_syn.add_argument("--out", required=True, help="папка набора")
    p_syn.add_argument("--truth", action="append", metavar="ИМЯ=ЗНАЧЕНИЕ",
                       help="скрытый истинный параметр: k0, roof_ratio, …")  # fmt: skip
    p_syn.add_argument("--brine-every", type=float, default=5.0, help="замер рассола каждые N сут")
    p_syn.add_argument("--sonar-days", help="дни сонарных съёмок через запятую")
    p_syn.add_argument("--sigma-c", type=float, default=3.0, help="шум концентрации, кг/м³")
    p_syn.add_argument("--sigma-r", type=float, default=0.3, help="шум радиуса, м")
    p_syn.add_argument("--seed", type=int, default=0, help="seed шума")
    p_syn.add_argument("--name", help="название скважины")
    p_syn.add_argument("--sansmic", action="store_true", help="открытый набор из расчёта SANSMIC")
    p_syn.add_argument("--excel", action="store_true", help="CSV как из Excel: «;» и запятая")
    p_cal = sub.add_parser("calibrate", help="подобрать параметры по данным скважины")
    p_cal.add_argument("config", help="YAML: априорные значения и раздел calibration")
    p_cal.add_argument("--data", required=True, help="папка данных скважины")
    p_cal.add_argument("--method", choices=("map", "mcmc"), help="по умолчанию из YAML")
    p_cal.add_argument("--steps", type=int, help="MCMC: шагов цепочки (calibration.steps)")
    p_cal.add_argument("--holdout-days", type=float,
                       help="подгонка по замерам до N сут, позже — проверка")  # fmt: skip
    p_cal.add_argument("--jobs", type=int, default=0, help="процессов (0 — все ядра)")
    p_cal.add_argument("--out", help="сохранить результат .npz")
    p_cal.add_argument("--plot", help="сохранить картинку .png")
    p_cal.add_argument("--write-config", help="записать откалиброванный YAML (для ансамбля)")

    p_rep = sub.add_parser("report", help="отчёт HTML в один клик (печать в PDF — из браузера)")
    p_rep.add_argument("config", help="YAML-конфигурация")
    p_rep.add_argument("--out", default="report.html", help="файл отчёта .html")
    p_rep.add_argument("--compute", help="досчитать: leaching,ensemble,mechanics,stability")
    p_rep.add_argument("--all", action="store_true", help="досчитать всё (ансамбль — минуты)")
    p_rep.add_argument("-n", type=int, help="реализаций ансамбля (по умолчанию из YAML)")
    p_rep.add_argument("--jobs", type=int, default=0, help="процессов (0 — все ядра)")
    for key, what in (("results", "выщелачивание"), ("ensemble", "ансамбль"),
                      ("mechanics", "геомеханика"), ("stability", "устойчивость"),
                      ("calibration", "калибровка")):  # fmt: skip
        p_rep.add_argument(f"--{key}", help=f"готовый результат: {what} (.npz)")
    p_rep.add_argument("--data", help="папка данных скважины (сводка и замечания)")
    p_rep.add_argument("--calibrate", action="store_true", help="откалибровать по --data")

    p_sc = sub.add_parser("shortcut", help="ярлык окна со значком (рабочий стол, меню «Пуск»)")
    p_sc.add_argument("--no-desktop", action="store_true", help="без ярлыка на рабочем столе")
    p_sc.add_argument("--no-menu", action="store_true", help="без ярлыка в меню")
    p_sc.add_argument("--workdir", help="рабочая папка окна (по умолчанию — текущая)")

    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "gui":
        return _cmd_gui(args)
    if args.command == "schema":
        return _cmd_schema(args)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "view":
        return _cmd_view(args)
    if args.command == "surrogate":
        return _cmd_surrogate(args)
    if args.command == "mechanics":
        return _cmd_mechanics(args)
    if args.command == "ensemble":
        return _cmd_ensemble(args)
    if args.command == "data":
        return _cmd_data(args)
    if args.command == "synth":
        return _cmd_synth(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "shortcut":
        return _cmd_shortcut(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
