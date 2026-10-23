"""Пересчитать эталон SANSMIC для configs/benchmark_sansmic_direct.yaml.

Нужен отдельный пакет SANSMIC (BSD-3, Sandia): ``pip install sansmic``.
Результат — benchmarks/sansmic/reference_direct.npz (в СИ, глубины сверху вниз).

Сценарий задаётся в родных единицах SANSMIC (ft/in/bbl) и переводится в СИ.
Метрический режим (Units.M_CM_M3) в sansmic 1.0.7 использовать нельзя:
в нём выдаваемый объём делится на 5,615 (перевод ft³→bbl), а прирост радиуса
получается примерно в 3,3 раза больше физически возможного (баланс соли не
сходится) — похоже на смешение футов и метров в скорости отступания стенки.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sansmic.model as sm  # type: ignore[import-not-found]

FT, BBL = 0.3048, 0.158987294928
OUT = Path(__file__).with_name("reference_direct.npz")


def ft(x_m: float) -> float:
    return x_m / FT


def main() -> None:
    nc = 100
    sc = sm.Scenario(
        title="saltcav benchmark: direct leach of a 5 m cylinder",
        num_cells=nc,
        geometry_format=sm.GeometryFormat.RADIUS_LIST,
        geometry_data={"radii": [ft(5.0)] * (nc + 1)},
        cavern_height=ft(100.0),
        floor_depth=ft(1100.0),
        insolubles_ratio=0.04,
        units=sm.Units.FT_IN_BBL,
        ullage_standoff=ft(5.0),
        stages=[
            dict(
                title="direct",
                simulation_mode="ordinary",
                solver_timestep=0.1,  # ч
                save_frequency=240,  # шагов = 1 сут
                injection_duration=24 * 100,  # ч
                rest_duration=0,
                inner_tbg_inside_diam=17.78 / 2.54,  # дюймы
                inner_tbg_outside_diam=19.37 / 2.54,
                outer_csg_inside_diam=24.45 / 2.54,
                outer_csg_outside_diam=27.31 / 2.54,
                brine_injection_sg=1.0,
                brine_injection_depth=ft(1090.0),
                brine_production_depth=ft(1020.0),
                brine_injection_rate=1200.0 / BBL,  # bbl/сут
                set_initial_conditions=True,
                set_cavern_sg=1.2019,
                blanket_depth=ft(1010.0),
            )
        ],
    )
    with sc.new_simulation("sansmic_reference") as sim:
        sim.run_sim()
    res = sim.results

    z = np.asarray(res.depths, float)[::-1] * FT  # SANSMIC: снизу вверх
    radius = np.asarray(res.radius, float)[::-1, :] * FT
    np.savez_compressed(
        OUT,
        time_d=np.asarray(res.time, float),
        z_m=z,
        radius_m=radius,
        volume_m3=np.asarray(res.cavern_volume, float) * BBL,
        sg_out=np.asarray(res.brine_production_sg, float),
        insol_height_m=np.asarray(res.insoluble_material_height, float) * FT,
        plume_depth_m=np.asarray(res.df_t_1D.z_plume, float) * FT,
    )
    print(f"эталон записан: {OUT}")


if __name__ == "__main__":
    main()
