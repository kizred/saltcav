"""Верификация движения стенки по уравнению Гамильтона–Якоби.

При постоянной нормальной скорости V без зависимости от ориентации:
цилиндр растёт как R + Vt, сфера остаётся сферой радиуса a + Vt.
"""

import numpy as np
import pytest

from saltcav.geometry import Cavern, CavernGrid
from saltcav.leaching import DissolutionLaw, wall_growth_rate

pytestmark = pytest.mark.verification

ISO = DissolutionLaw(k0=1e-5, exponent=1.0, roof_ratio=1.0, floor_ratio=1.0)
CS, RHO = 300.0, 2000.0
V = 1e-5 * CS / RHO  # нормальная скорость при C = 0


def _advance(cav: Cavern, t_end: float, cfl: float = 0.5) -> Cavern:
    g = cav.grid
    n = g.nz + 1
    t = 0.0
    while t < t_end:
        rate, vmax = wall_growth_rate(
            cav.radius, g.dz, np.zeros(n), CS, ISO, RHO, np.ones(n, dtype=bool)
        )
        dt = min(cfl * float(np.min(g.dz)) / vmax, t_end - t)
        cav = cav.with_radius(cav.radius + dt * rate)
        t += dt
    return cav


def test_cylinder_grows_uniformly() -> None:
    g = CavernGrid.uniform(0, 50, 50, 4)
    t = 1.0e6
    cav = _advance(Cavern.cylinder(g, 5.0), t)
    np.testing.assert_allclose(cav.radius, 5.0 + V * t, rtol=1e-12)


def _sphere_case(nz: int) -> tuple[float, float]:
    """Сфера a=20 м в области 100 м растёт на 10 м: ошибка объёма и макс. радиуса."""
    a, grow = 20.0, 10.0
    g = CavernGrid.uniform(0, 100, nz)
    zc = 50.0
    cav = Cavern.from_profile(g, lambda z, p: np.sqrt(np.clip(a**2 - (z - zc) ** 2, 0, None)))
    cav = _advance(cav, grow / V)
    b = a + grow
    vol_err = abs(cav.volume() - 4 / 3 * np.pi * b**3) / (4 / 3 * np.pi * b**3)
    rad_err = abs(cav.max_radius() - b) / b
    return vol_err, rad_err


def test_sphere_stays_sphere() -> None:
    vol_err, rad_err = _sphere_case(400)
    assert rad_err < 2e-3
    assert vol_err < 1e-2


def test_sphere_error_decreases_with_refinement() -> None:
    errs = [_sphere_case(n)[0] for n in (100, 200, 400)]
    assert errs[0] > errs[1] > errs[2]


def test_roof_climbs_at_roof_speed() -> None:
    # плоская кровля на глубине 50 м поднимается со скоростью roof_ratio·V
    law = DissolutionLaw(k0=1e-5, exponent=1.0, roof_ratio=2.0, floor_ratio=0.0)
    g = CavernGrid.uniform(0, 100, 200)  # Δz = 0.5 м
    r0 = np.where(g.z_edges >= 50.0, 20.0, 0.0)[:, None]
    cav = Cavern(g, r0)
    t_end = 10.0 / V  # кровля должна подняться на 2·10 = 20 м
    t = 0.0
    n = g.nz + 1
    while t < t_end:
        rate, vmax = wall_growth_rate(
            cav.radius, g.dz, np.zeros(n), CS, law, RHO, np.ones(n, dtype=bool)
        )
        dt = min(0.5 * 0.5 / vmax, t_end - t)
        cav = cav.with_radius(cav.radius + dt * rate)
        t += dt
    filled = g.z_edges[cav.radius[:, 0] > 10.0]  # узлы, где камера уже широкая
    assert filled.min() == pytest.approx(30.0, abs=1.5)
