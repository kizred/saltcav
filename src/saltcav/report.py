"""Отчёт в один клик: один файл HTML (картинки внутри), который открывается в любом
браузере и печатается в PDF.

Разделы — по тому, что посчитано: исходные данные, выщелачивание, калибровка,
ансамбль и карты вероятностей, геомеханика, устойчивость, выводы, воспроизводимость
(версия, seed, конфигурация целиком). Недостающее можно досчитать (``compute``).
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from saltcav import __version__
from saltcav.core.config import RunConfig, config_warnings

if TYPE_CHECKING:
    from saltcav.calibration.fit import CalibResult
    from saltcav.io.fielddata import FieldData
    from saltcav.io.results import Results
    from saltcav.mechanics.run import MechanicsRun
    from saltcav.stochastic.ensemble import EnsembleResult
    from saltcav.surrogate.study import StudyResult

Progress = Callable[[float, str], None]
OK, WARN, BAD = "ok", "warn", "bad"
PARTS = ("leaching", "ensemble", "mechanics", "stability")


@dataclass
class Finding:
    level: str  # ok | warn | bad
    text: str


@dataclass
class ReportInputs:
    """Что войдёт в отчёт (None — раздела нет)."""

    cfg: RunConfig
    results: Results | None = None
    ensemble: EnsembleResult | None = None
    mechanics: MechanicsRun | None = None
    stability: StudyResult | None = None
    calibration: CalibResult | None = None
    data: FieldData | None = None
    timings: dict[str, float] = field(default_factory=dict)


def compute_missing(inp: ReportInputs, compute: Sequence[str], n: int | None = None,
                    jobs: int = 0, progress: Progress | None = None) -> ReportInputs:  # fmt: skip
    """Досчитать недостающие разделы из ``compute`` (leaching, ensemble, mechanics,
    stability) по конфигурации ``inp.cfg``."""
    import saltcav.api as sc

    say = progress or (lambda f, t: None)
    todo = [p for p in PARTS if p in compute]
    unknown = set(compute) - set(PARTS)
    if unknown:
        raise ValueError(f"неизвестные разделы: {', '.join(sorted(unknown))}; есть {PARTS}")
    for k, part in enumerate(todo):
        frac = k / max(len(todo), 1)
        t0 = time.perf_counter()
        if part == "leaching" and inp.results is None:
            say(frac, "отчёт: выщелачивание")
            inp.results = sc.run(inp.cfg)
        elif part == "ensemble" and inp.ensemble is None:
            say(frac, "отчёт: ансамбль")
            inp.ensemble = sc.ensemble(inp.cfg, n=n, jobs=jobs)
        elif part == "mechanics" and inp.mechanics is None:
            say(frac, "отчёт: геомеханика")
            if inp.results is None:
                inp.results = sc.run(inp.cfg)
            inp.mechanics = sc.mechanics(inp.cfg, inp.results)
        elif part == "stability" and inp.stability is None:
            say(frac, "отчёт: устойчивость")
            if inp.ensemble is None:
                inp.ensemble = sc.ensemble(inp.cfg, n=n, jobs=jobs)
            inp.stability = sc.stability(inp.cfg, inp.ensemble, jobs=jobs)
        else:
            continue
        inp.timings[part] = time.perf_counter() - t0
    say(1.0, "отчёт: оформление")
    return inp


# --- картинки --------------------------------------------------------------------------


def _png(draw: Callable[[Any], None], size: tuple[float, float]) -> str:
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, facecolor="#fcfcfb")
    draw(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#fcfcfb")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str, caption: str) -> str:
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")  # fmt: skip


def _mechanics_png(run: MechanicsRun) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from saltcav.viz.mechanics_plot import plot_mechanics

    buf = io.BytesIO()
    plot_mechanics(run, buf)  # type: ignore[arg-type]
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- таблицы ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        if not np.isfinite(v):
            return "—"
        a = abs(float(v))
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{v:.3g}" if a < 1e-3 else f"{v:,.0f}".replace(",", " ")
        return f"{v:.4g}"
    return html.escape(str(v))


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]], cls: str = "") -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def _api_table(obj: Any) -> str:
    import saltcav.api as sc

    t = sc.table(obj)
    cols = t.columns
    return _table(cols, [[r.get(c, "") for c in cols] for r in t.rows])


# --- разделы --------------------------------------------------------------------------


def _inputs(cfg: RunConfig) -> str:
    g, geo, lp, mc = cfg.grid, cfg.geology, cfg.leaching, cfg.mechanics
    out = ["<h2>1. Исходные данные</h2>"]
    rows = [
        ("Сетка по глубине", f"{g.z_top_m:g}–{g.z_bottom_m:g} м, {g.nz} слоёв"),
        ("Секторов по азимуту", f"{g.nphi}" + (" (осесимметрия)" if g.nphi == 1 else " (3D)")),
        ("Доля нерастворимых", f"{geo.insoluble_fraction:g} (σ {geo.insoluble_fraction_sd:g})"),
        ("Разрыхление насыпи", f"{geo.bulking_factor:g}"),
        ("k0, м/с", f"{lp.k0_m_per_s:.3g}"),
        ("Кровля / стенка", f"{lp.roof_ratio:g}"),
        ("Температура рассола", f"{lp.temperature_c:g} °C"),
        ("Геомеханика", "осесимметрия" if mc.model == "axisym" else "3D"),
        ("Модуль Юнга соли", f"{mc.youngs_modulus_gpa:g} ГПа"),
        ("Ползучесть", ", ".join(f"A={b.a_per_day:g} 1/сут, n={b.n:g}" for b in mc.creep)),
    ]
    out.append(_table(("параметр", "значение"), rows, "kv"))
    st = [(i + 1, s.duration_days, "прямоток" if s.mode.value == "direct" else "противоток",
           s.rate_m3_per_h, s.injection_depth_m, s.production_depth_m, s.blanket_depth_m)
          for i, s in enumerate(cfg.schedule.stages)]  # fmt: skip
    out.append("<h3>Режим выщелачивания</h3>")
    out.append(_table(("№", "сут", "схема", "расход, м³/ч", "закачка, м", "отбор, м",
                       "нерастворитель, м"), st))  # fmt: skip
    if geo.layers:
        ru = {"anhydrite": "ангидрит", "clay": "глина", "halite": "каменная соль"}
        lay = [(ru.get(ly.kind.value, ly.kind.value), ly.top_m, ly.thickness_m, ly.top_sd_m)
               for ly in geo.layers]  # fmt: skip
        out.append("<h3>Прослои</h3>")
        out.append(_table(("порода", "кровля, м", "мощность, м", "σ отметки, м"), lay))
    ps = [(i + 1, s.duration_years, s.wellhead_pressure_mpa, s.fluid_density_kg_m3, s.label)
          for i, s in enumerate(mc.stages)]  # fmt: skip
    out.append("<h3>Эксплуатация (давление в камере)</h3>")
    out.append(_table(("№", "лет", "на устье, МПа", "плотность, кг/м³", "этап"), ps))
    return "\n".join(out)


def _leaching(res: Results, fnd: list[Finding]) -> str:
    from saltcav.viz.leaching_plot import draw_leaching

    r = res.radius[-1]
    wide = np.nonzero(r.max(axis=1) > 1.0)[0]
    roof = float(res.z_edges[wide[0]]) if wide.size else float("nan")
    ev = [e for e in res.events() if e.kind == "collapse"]
    rows = [
        ("Срок", f"{res.days[-1]:g} сут"),
        ("Объём", f"{res.volume[-1]:,.0f} м³".replace(",", " ")),
        ("Наибольший радиус", f"{r.max():.1f} м"),
        ("Кровля камеры", f"{roof:.1f} м"),
        ("Верх насыпи нерастворимых", f"{res.floor_depth[-1]:.1f} м"),
        ("Обрушений карнизов (по секторам)", f"{len(ev)}"),
    ]
    v = res.volume[-1] / 1e3
    fnd.append(Finding(OK, f"Детерминированный прогон: объём {v:,.1f} тыс. м³, наибольший "
                           f"радиус {r.max():.1f} м, кровля на {roof:.1f} м."))  # fmt: skip
    if ev:
        fnd.append(Finding(WARN, f"Карнизы прослоев обрушались ({len(ev)} событий по секторам): "
                                 "обломки в насыпи, ступенчатая стенка."))  # fmt: skip
    return "\n".join([
        "<h2>2. Выщелачивание (детерминированный прогон)</h2>",
        _table(("показатель", "значение"), rows, "kv"),
        _img(_png(lambda f: draw_leaching(f, res), (12, 7.5)), "Профиль камеры по времени, "
             "объём и концентрация рассола на выходе"),
    ])  # fmt: skip


def _calibration(cal: CalibResult, fnd: list[Finding]) -> str:
    from saltcav.viz.calibration_plot import draw_calibration

    head = cal.table_header()
    est = ", ".join(f"{lab} = {v:.3g}" for lab, v in zip(cal.labels, cal.estimate().values(),
                                                           strict=True))  # fmt: skip
    level = OK if cal.chi2_reduced <= 2.0 else WARN
    fnd.append(Finding(level, f"Калибровка по данным: {est}; χ²/N = {cal.chi2_reduced:.2f}."))
    for n in cal.notes():
        fnd.append(Finding(WARN, f"Калибровка: {n}."))
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in cal.notes())
    return "\n".join([
        "<h2>3. Калибровка по данным скважины</h2>",
        _table(head, cal.table()),
        f"<p>Метод: {'MCMC' if cal.method == 'mcmc' else 'MAP + лапласово приближение'}; "
        f"χ²/N = {_fmt(cal.chi2_reduced)} по {cal.n_fit} замерам; {cal.n_runs} прогонов.</p>",
        f"<ul class='notes'>{notes}</ul>" if notes else "",
        _img(_png(lambda f: draw_calibration(f, cal), (13, 9)), "Рассол на выходе и профили "
             "сонара: замеры и модель; апостериорные распределения параметров"),
    ])  # fmt: skip


def _ensemble(ens: EnsembleResult, cfg: RunConfig, fnd: list[Finding]) -> str:
    from saltcav.viz.envelope import draw_envelope
    from saltcav.viz.probability_plot import draw_probability

    layers = [(ly.top_m, ly.thickness_m) for ly in cfg.geology.layers]
    vol = np.quantile(ens.output("volume"), (0.1, 0.5, 0.9)) / 1e3
    rmax = np.quantile(ens.output("max_radius"), (0.1, 0.5, 0.9))
    fnd.append(Finding(OK, f"Ансамбль {ens.n} реализаций: объём P10–P90 {vol[0]:.1f}–{vol[2]:.1f} "
                           f"тыс. м³, наибольший радиус P90 {rmax[2]:.1f} м."))  # fmt: skip
    sens = ens.sensitivity(("volume", "max_radius"))
    srows = []
    for name, d in sens.items():
        for p, rho in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:4]:
            srows.append((("объём" if name == "volume" else "наибольший радиус"), p, rho))
    return "\n".join([
        "<h2>4. Неопределённость: ансамбль Монте-Карло</h2>",
        f"<p>{ens.n} реализаций, seed {ens.seed}, план «{cfg.ensemble.sampler}».</p>",
        _api_table(ens),
        _img(_png(lambda f: draw_envelope(f, ens, "", layers), (10.5, 6.2)),
             "Огибающие эквивалентного радиуса (P10–P90) и сходимость квантилей объёма"),
        _img(_png(lambda f: draw_probability(f, ens), (12, 6.5)),
             "Карты вероятностей: P(стенка дальше r) по глубине; P90 радиуса по азимуту"),
        "<h3>Что сильнее всего влияет (ранговая корреляция Спирмена)</h3>" if srows else "",
        _table(("выход", "параметр", "ρ"), srows) if srows else "",
    ])  # fmt: skip


def _mechanics(run: MechanicsRun, fnd: list[Finding]) -> str:
    s = run.summary
    p5 = s.fos_wall_p5 if s.fos_wall_p5 is not None else s.min_fos_wall
    fos = float(np.min(p5[1:])) if p5.size > 1 else float(p5[0])
    lvl = BAD if fos < 1.0 else WARN if fos < 1.5 else OK
    fnd.append(Finding(lvl, f"Геомеханика: запас по дилатансии на 95 % стенки — {fos:.2f} "
                            "(требование ≥ 1,5; < 1 — разуплотнение соли)."))  # fmt: skip
    fnd.append(Finding(OK, f"Конвергенция за {s.years[-1]:.0f} лет — "
                           f"{s.convergence_pct[-1]:.1f} % объёма, к концу "
                           f"{s.convergence_rate_pct_yr[-1]:.2f} %/год."))  # fmt: skip
    tm = s.tension_margin_p5_mpa if s.tension_margin_p5_mpa is not None else s.tension_margin_mpa
    if float(np.min(tm[1:] if tm.size > 1 else tm)) < 0:
        fnd.append(Finding(WARN, "На части стенки (5 %) напряжения вдоль стенки ниже давления в "
                                 "камере — риск проникновения рассола/газа в соль."))  # fmt: skip
    if s.layer_tension_ratio is not None and float(np.max(s.layer_tension_ratio)) > 1:
        fnd.append(Finding(WARN, f"Растяжение в прослое — {np.max(s.layer_tension_ratio):.1f} "
                                 "прочности на разрыв: возможна трещина, карниз может "
                                 "обрушиться."))  # fmt: skip
    kind = "3D" if run.dim == 3 else "осесимметрия"
    return "\n".join([
        "<h2>5. Геомеханика</h2>",
        f"<p>Форма на {run.day:.0f} сут, {kind}, {run.tri.shape[0]} элементов.</p>",
        _api_table(run),
        _img(_mechanics_png(run), "Запас по дилатансии, напряжения, смещения и ползучесть; "
             "конвергенция и запас во времени"),
    ])  # fmt: skip


def _stability(st: StudyResult, fnd: list[Finding]) -> str:
    from saltcav.surrogate.study import KEY
    from saltcav.viz.surrogate_plot import draw_surrogate

    p, _ = st.prob_below()
    p15 = st.prob_below(threshold=1.5)[0]
    lvl = BAD if p > 0.01 else WARN if p15 > 0.1 else OK
    ci = st.prob_ci(p)
    fnd.append(Finding(lvl, f"Вероятность потери устойчивости P(запас < {st.threshold:g}) "
                            f"= {p:.3f} ± {ci:.3f}; P(запас < 1,5) = {p15:.2f}."))  # fmt: skip
    if not st.explains(KEY):
        fnd.append(Finding(OK, "Разброс запаса по ансамблю сопоставим с погрешностью расчёта: "
                               "запас почти не зависит от неопределённых входов."))  # fmt: skip
    return "\n".join([
        "<h2>6. Устойчивость по ансамблю</h2>",
        f"<p>Геомеханика для {st.train.size} реализаций, остальные — суррогат "
        f"(гауссов процесс по признакам формы).</p>",
        _api_table(st),
        _img(_png(lambda f: draw_surrogate(f, st), (13, 9)),
             "Распределение запаса, проверка суррогата, важность признаков"),
    ])  # fmt: skip


def _findings(fnd: list[Finding]) -> str:
    icon = {OK: "✔", WARN: "⚠", BAD: "✖"}
    items = "".join(f'<li class="{f.level}"><span>{icon[f.level]}</span> {html.escape(f.text)}</li>'
                    for f in fnd)  # fmt: skip
    return f'<h2>Выводы</h2><ul class="findings">{items}</ul>'


CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --accent:#1c5cab; --bg:#fcfcfb;
        --ok:#2e7d32; --warn:#a86b00; --bad:#c0392b; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 15px/1.5 "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } .sub { color: var(--ink2); margin: 0 0 24px; }
h2 { font-size: 20px; margin: 36px 0 12px; padding-top: 12px; border-top: 2px solid var(--line); }
h3 { font-size: 16px; margin: 20px 0 8px; color: var(--ink2); }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: right; }
th { color: var(--ink2); font-weight: 600; text-align: right; }
td:first-child, th:first-child { text-align: left; }
table.kv td:last-child { text-align: left; }
figure { margin: 12px 0 24px; } img { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--ink2); font-size: 13px; margin-top: 4px; }
ul.findings { list-style: none; padding: 0; } ul.findings li { padding: 6px 10px; margin: 4px 0;
  border-left: 4px solid var(--ok); background: #fff; }
ul.findings li.warn { border-color: var(--warn); } ul.findings li.bad { border-color: var(--bad); }
ul.findings li span { font-weight: 700; } li.ok span { color: var(--ok); }
li.warn span { color: var(--warn); } li.bad span { color: var(--bad); }
details { margin: 8px 0; } pre { font-size: 12px; background: #fff; padding: 12px;
  border: 1px solid var(--line); overflow-x: auto; }
.toc { columns: 2; font-size: 14px; }
@media print { body { background: #fff; } main { max-width: none; padding: 0; }
  h2 { break-before: page; } h2:first-of-type { break-before: auto; }
  figure, table { break-inside: avoid; } details { display: none; } }
"""


def _favicon() -> str:
    from saltcav.gui.icons import icon_path

    try:
        return base64.b64encode(icon_path("png256").read_bytes()).decode("ascii")
    except OSError:  # pragma: no cover
        return ""


def _renumber(text: str) -> str:
    """Номера разделов подряд: без калибровки нет «3», следующий — снова «3»."""
    import re

    counter = iter(range(1, 100))
    return re.sub(r"<h2>\d+\. ", lambda m: f"<h2>{next(counter)}. ", text)


def build_html(inp: ReportInputs, title: str = "") -> str:
    """HTML отчёта по готовым результатам."""
    cfg = inp.cfg
    fnd: list[Finding] = []
    for loc, msg in config_warnings(cfg):
        fnd.append(Finding(WARN, f"Параметры ({'.'.join(loc)}): {msg}."))
    if inp.data is not None:
        for msg in inp.data.issues(cfg):
            fnd.append(Finding(WARN, f"Данные скважины: {msg}."))
    parts = [_inputs(cfg)]
    if inp.results is not None:
        parts.append(_leaching(inp.results, fnd))
    if inp.calibration is not None:
        parts.append(_calibration(inp.calibration, fnd))
    if inp.ensemble is not None:
        parts.append(_ensemble(inp.ensemble, cfg, fnd))
    if inp.mechanics is not None:
        parts.append(_mechanics(inp.mechanics, fnd))
    if inp.stability is not None:
        parts.append(_stability(inp.stability, fnd))

    yml = yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    digest = hashlib.sha256(yml.encode("utf-8")).hexdigest()[:16]
    timing = ", ".join(f"{k} {v:.0f} с" for k, v in inp.timings.items())
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    head = title or f"Камера «{cfg.name}»"
    repro = (
        "<h2>Воспроизводимость</h2>"
        f"<p>saltcav {html.escape(__version__)}, отчёт {now}; конфигурация sha256 {digest}; "
        f"seed ансамбля {cfg.ensemble.seed}"
        + (f"; время расчёта: {html.escape(timing)}" if timing else "")
        + ". Тот же код, конфигурация и seed дают тот же результат бит в бит.</p>"
        f"<details><summary>Конфигурация (YAML)</summary><pre>{html.escape(yml)}</pre></details>"
    )
    limits = (
        "<h2>Ограничения</h2><ul>"
        "<li>Параметры, не откалиброванные по данным месторождения, взяты из литературы и "
        "открытых бенчмарков (SANSMIC, WIPP, BGRa) — результаты оценочные.</li>"
        "<li>Механика считается по итоговой форме камеры; влияние механики на выщелачивание "
        "не учитывается.</li>"
        "<li>Прослои жёстко сцеплены с солью, без проскальзывания по контакту.</li></ul>"
    )
    body = _renumber("".join(parts))
    return (
        f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(head)} — отчёт saltcav</title>"
        f'<link rel="icon" href="data:image/png;base64,{_favicon()}">'
        f"<style>{CSS}</style></head><body><main>"
        f"<h1>{html.escape(head)}</h1>"
        f'<p class="sub">Отчёт saltcav {html.escape(__version__)} (автор программы — '
        f"Sergei Kizred) · {now}</p>"
        + _findings(fnd)
        + body
        + limits
        + repro
        + "</main></body></html>"
    )


def make_report(
    cfg: RunConfig,
    out: str | Path,
    *,
    results: Results | None = None,
    ensemble: EnsembleResult | None = None,
    mechanics: MechanicsRun | None = None,
    stability: StudyResult | None = None,
    calibration: CalibResult | None = None,
    data: FieldData | None = None,
    compute: Sequence[str] = ("leaching",),
    n: int | None = None,
    jobs: int = 0,
    progress: Progress | None = None,
    title: str = "",
) -> Path:
    """Собрать отчёт: готовые результаты + досчитанные разделы ``compute``."""
    inp = ReportInputs(cfg, results, ensemble, mechanics, stability, calibration, data)
    compute_missing(inp, compute, n=n, jobs=jobs, progress=progress)
    p = Path(out)
    p.write_text(build_html(inp, title), encoding="utf-8")
    return p
