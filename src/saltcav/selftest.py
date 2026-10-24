"""Самопроверка установки: все части программы на маленьких задачах (~1 мин).

``saltcav selftest`` — после установки или сборки: библиотеки (Qt, VTK, Gmsh, MKL,
scikit-learn), выщелачивание, ансамбль в нескольких процессах, геомеханика 2D и 3D,
калибровка, отчёт. Каждый пункт — «OK» или причина; код выхода 1, если что-то не так.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def _small_cfg() -> Any:
    import saltcav.api as sc
    from saltcav.workspace import bundled_root

    cfg = sc.load(bundled_root() / "configs" / "example_ensemble.yaml")
    return sc.set(cfg, {
        "grid.nz": 50, "grid.nphi": 4, "leaching.output_every_days": 40,
        "mechanics.stages": [{"duration_years": 1.0}], "mechanics.h_wall_m": 2.0,
        "mechanics.h_wall_3d_m": 5.0, "ensemble.n_realizations": 4,
    })  # fmt: skip


def checks(tmp: Path) -> list[tuple[str, Callable[[], str]]]:
    state: dict[str, Any] = {}

    def libs() -> str:
        import gmsh  # noqa: F401
        import PySide6
        import sklearn
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

        return f"Qt {PySide6.__version__}, scikit-learn {sklearn.__version__}"

    def pardiso() -> str:
        import scipy.sparse as sp

        from saltcav.mechanics.linsolve import factorize_spd

        n = 200
        a = sp.diags([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
        x = factorize_spd(a, "pardiso")(np.ones(n))
        if not np.allclose(a @ x, 1.0):
            raise RuntimeError("PARDISO: неверное решение")
        return "PARDISO (MKL)"

    def leaching() -> str:
        import saltcav.api as sc

        state["cfg"] = cfg = _small_cfg()
        state["res"] = res = sc.run(cfg)
        return f"объём {res.volume[-1]:,.0f} м³".replace(",", " ")

    def ensemble() -> str:
        import saltcav.api as sc

        state["ens"] = ens = sc.ensemble(state["cfg"], n=4, jobs=2)
        return f"{ens.n} реализаций в 2 процессах"

    def mech2d() -> str:
        import saltcav.api as sc

        state["mech"] = run = sc.mechanics(state["cfg"], state["res"])
        return f"конвергенция {run.summary.convergence_pct[-1]:.2f} % за год"

    def mech3d() -> str:
        import saltcav.api as sc

        cfg = sc.set(state["cfg"], {"mechanics.interlayers": False, "mechanics.r_factor": 4.0,
                                    "mechanics.height_factor": 1.5})  # fmt: skip
        run = sc.mechanics(cfg, state["res"], model="3d")
        conv = run.summary.convergence_pct[-1]
        return f"{run.tri.shape[0]} тетраэдров, конвергенция {conv:.2f} %"

    def calibration() -> str:
        import saltcav.api as sc
        from saltcav.calibration import synthesize
        from saltcav.core.config import CalibParamConfig
        from saltcav.io.fielddata import write_field_data

        cfg = state["cfg"]
        cc = cfg.calibration.model_copy(
            update={"parameters": [CalibParamConfig(name="k0")], "n_predictive": 2}
        )
        cfg = cfg.model_copy(update={"calibration": cc})
        write_field_data(tmp / "well", synthesize(cfg, {"k0": 2e-5}, seed=1))
        cal = sc.calibrate(cfg, tmp / "well", jobs=1)
        k0 = cal.estimate()["k0"]
        if abs(k0 / 2e-5 - 1) > 0.1:
            raise RuntimeError(f"k0 = {k0:.3g}, ожидалось 2e-5")
        return f"k0 = {k0:.3g} (истина 2e-5)"

    def report() -> str:
        import saltcav.api as sc

        p = sc.report(state["cfg"], tmp / "report.html", results=state["res"],
                      ensemble=state.get("ens"), mechanics=state.get("mech"),
                      compute=())  # fmt: skip
        return f"{p.stat().st_size / 1e6:.1f} МБ"

    def render() -> str:
        import pyvista as pv

        from saltcav.viz.surface import render as render_scene

        pv.OFF_SCREEN = True
        res = state["res"]
        render_scene(res.z_edges, res.radius[-1], float(res.floor_depth[-1]),
                     screenshot=tmp / "scene.png")  # fmt: skip
        if not (tmp / "scene.png").exists():
            raise RuntimeError("снимок не создан")
        return "3D-вид камеры (VTK, OpenGL)"

    return [
        ("библиотеки", libs), ("PARDISO", pardiso), ("выщелачивание", leaching),
        ("ансамбль", ensemble), ("геомеханика 2D", mech2d), ("геомеханика 3D", mech3d),
        ("калибровка", calibration), ("отчёт", report), ("3D-вид", render),
    ]  # fmt: skip


def run_selftest(skip: tuple[str, ...] = (), print_: Callable[[str], None] = print) -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for name, fn in checks(Path(d)):
            if name in skip:
                print_(f"  {name:<16} пропущено")
                continue
            t0 = time.perf_counter()
            try:
                info = fn()
                print_(f"  {name:<16} OK   {time.perf_counter() - t0:5.1f} с  {info}")
            except Exception as exc:  # самопроверка сообщает о сбое, а не падает
                ok = False
                print_(f"  {name:<16} СБОЙ {type(exc).__name__}: {exc}")
                if os.environ.get("SALTCAV_DEBUG"):
                    print_(traceback.format_exc())
    print_("всё в порядке" if ok else "есть сбои — см. выше")
    return ok
