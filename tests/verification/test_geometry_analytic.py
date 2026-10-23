"""Верификация геометрии против аналитических решений.

Критерий этапа 1 из паспорта: объём и площадь цилиндра, конуса и сферы
совпадают с аналитикой с относительной ошибкой < 0.1 % на рабочей сетке (nz=200).
"""

import numpy as np
import pytest

from saltcav.geometry import Cavern, CavernGrid

pytestmark = pytest.mark.verification

Z0, H = 1000.0, 100.0  # верх и высота области


def _sphere(nz: int, nphi: int = 1, a: float = 40.0) -> Cavern:
    grid = CavernGrid.uniform(Z0, Z0 + 2 * a, nz, nphi)
    zc = Z0 + a
    return Cavern.from_profile(grid, lambda z, p: np.sqrt(np.clip(a**2 - (z - zc) ** 2, 0, None)))


@pytest.mark.parametrize("nphi", [1, 8, 36])
def test_cylinder_exact(nphi: int) -> None:
    r = 25.0
    cav = Cavern.cylinder(CavernGrid.uniform(Z0, Z0 + H, 37, nphi), r)
    assert cav.volume() == pytest.approx(np.pi * r**2 * H, rel=1e-12)
    assert cav.wall_areas().sum() == pytest.approx(2 * np.pi * r * H, rel=1e-12)
    assert cav.surface_area() == pytest.approx(2 * np.pi * r * H + 2 * np.pi * r**2, rel=1e-12)
    np.testing.assert_allclose(cav.wall_angles(), np.pi / 2)


def test_cone_exact() -> None:
    # вершина вверху, основание внизу: расширяется книзу → «кровельная» стенка, ψ < 90°
    r_base = 30.0
    grid = CavernGrid.uniform(Z0, Z0 + H, 17)
    cav = Cavern.from_profile(grid, lambda z, p: r_base * (z - Z0) / H)
    slant = np.hypot(H, r_base)
    assert cav.volume() == pytest.approx(np.pi * r_base**2 * H / 3, rel=1e-12)
    assert cav.wall_areas().sum() == pytest.approx(np.pi * r_base * slant, rel=1e-12)
    np.testing.assert_allclose(cav.wall_angles(), np.arccos(r_base / slant))
    assert cav.overhang_mask().all()


def test_inverted_cone_is_floor_like() -> None:
    grid = CavernGrid.uniform(Z0, Z0 + H, 10)
    cav = Cavern.from_profile(grid, lambda z, p: 30.0 * (Z0 + H - z) / H)
    assert np.all(cav.wall_angles() > np.pi / 2)
    assert not cav.overhang_mask().any()


@pytest.mark.parametrize("nphi", [1, 12])
def test_sphere_working_grid(nphi: int) -> None:
    a = 40.0
    cav = _sphere(200, nphi, a)
    assert cav.volume() == pytest.approx(4 / 3 * np.pi * a**3, rel=1e-3)
    assert cav.surface_area() == pytest.approx(4 * np.pi * a**2, rel=1e-3)


def test_sphere_volume_converges() -> None:
    a = 40.0
    exact = 4 / 3 * np.pi * a**3
    errs = [abs(_sphere(n, a=a).volume() - exact) / exact for n in (50, 100, 200, 400)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.8), orders  # второй порядок по Δz


def test_sphere_angles_span_roof_to_floor() -> None:
    psi = np.rad2deg(_sphere(200).wall_angles()[:, 0])
    assert psi[0] < 10 and psi[-1] > 170  # верх — кровля, низ — подошва
    assert np.all(np.diff(psi) > 0)  # монотонно от кровли к подошве
    assert abs(psi[100] - 90) < 1  # экватор — вертикаль
