"""Цепочка «данные → калибровка → ансамбль → устойчивость» одним скриптом.

Запуск:  python examples/calibrate_then_ensemble.py   (несколько минут на всех ядрах)
"""

import saltcav.api as sc

if __name__ == "__main__":
    cfg = sc.load("configs/demo_calibration.yaml")
    cal = sc.calibrate(cfg, "data/demo/well-1", holdout_days=160)
    print(sc.table(cal))
    sc.plot(cal, "calibration.png")

    cfg = cal.calibrated_config(cfg, sc.data("data/demo/well-1"))
    sc.save(cfg, "calibrated.yaml")

    ens = sc.ensemble(cfg, n=100)
    print(sc.table(ens))
    sc.plot(ens, "ensemble.png")

    stab = sc.stability(cfg, ens)
    print(sc.table(stab))
    sc.plot(stab, "stability.png")
