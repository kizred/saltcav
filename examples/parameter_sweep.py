"""Серия расчётов: как объём и наибольший радиус камеры зависят от k0 и расхода.

Запуск:  python examples/parameter_sweep.py   (или в окне: Файл → Выполнить скрипт…)
"""

import saltcav.api as sc

if __name__ == "__main__":
    cfg = sc.load("configs/example.yaml")
    rows = []
    for k0 in (1.5e-5, 2.0e-5, 3.0e-5):
        for rate in (40, 50, 60):
            c = sc.set(cfg, {"leaching.k0_m_per_s": k0, "schedule.stages.0.rate_m3_per_h": rate})
            res = sc.run(c)
            rows.append(
                {
                    "k0, м/с": k0,
                    "расход, м³/ч": rate,
                    "объём, м³": float(res.volume[-1]),
                    "R макс, м": float(res.radius[-1].max()),
                }
            )
    table = sc.Table(rows, "Чувствительность к k0 и расходу первой ступени")
    print(table)
    table.to_csv("sweep.csv", excel=True)
    print("таблица: sweep.csv (открывается в Excel)")
