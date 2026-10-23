import numpy as np
import pytest

from saltcav.core.config import load_config
from saltcav.geometry import Cavern, CavernGrid
from tests.unit.test_config import EXAMPLE


def test_grid_from_config() -> None:
    cfg = load_config(EXAMPLE)
    g = CavernGrid.from_config(cfg.grid)
    assert g.nz == cfg.grid.nz
    assert g.z_edges[0] == cfg.grid.z_top_m and g.z_edges[-1] == cfg.grid.z_bottom_m
    assert g.dz.sum() == pytest.approx(cfg.grid.z_bottom_m - cfg.grid.z_top_m)


def test_grid_validation() -> None:
    with pytest.raises(ValueError):
        CavernGrid(np.array([1.0, 1.0, 2.0]), 1)
    with pytest.raises(ValueError):
        CavernGrid(np.array([1.0, 2.0]), 0)


def test_radius_shape_and_sign_checked() -> None:
    g = CavernGrid.uniform(0, 10, 5, 4)
    with pytest.raises(ValueError):
        Cavern(g, np.ones((5, 4)))
    with pytest.raises(ValueError):
        Cavern(g, -np.ones((6, 4)))


def test_cavern_is_immutable_and_copies_input() -> None:
    g = CavernGrid.uniform(0, 10, 5)
    r = np.ones(6)
    cav = Cavern(g, r)
    r[:] = 5.0
    assert cav.radius.max() == 1.0
    with pytest.raises(ValueError):
        cav.radius[0, 0] = 2.0


def test_layer_volumes_sum_and_axisymmetric_sectors() -> None:
    g1 = CavernGrid.uniform(0, 50, 20, 1)
    g8 = CavernGrid.uniform(0, 50, 20, 8)

    def prof(z: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 5 + 3 * np.sin(z / 10)

    v1 = Cavern.from_profile(g1, prof).volume()
    v8 = Cavern.from_profile(g8, prof).volume()
    assert v8 == pytest.approx(v1, rel=1e-12)


def test_normals_unit_and_consistent_with_angle() -> None:
    g = CavernGrid.uniform(0, 30, 30, 3)
    rng = np.random.default_rng(0)
    cav = Cavern(g, rng.uniform(1, 10, (31, 3)))
    nr, nz = cav.wall_normals()
    np.testing.assert_allclose(nr**2 + nz**2, 1.0)
    np.testing.assert_allclose(np.arccos(-nz), cav.wall_angles())  # вверх = −z
    assert np.all(nr > 0)  # нормаль всегда наружу от оси


def test_centroid_offset() -> None:
    g = CavernGrid.uniform(0, 10, 4, 4)
    sym = Cavern.cylinder(g, 10.0)
    np.testing.assert_allclose(sym.centroid_offset(), 0.0, atol=1e-12)
    r = np.full((5, 4), 10.0)
    r[:, 0] = 20.0  # сектор вокруг φ = 45° вырос
    off = Cavern(g, r).centroid_offset()
    assert np.all(off[:, 0] > 0) and np.all(off[:, 1] > 0)
    np.testing.assert_allclose(off[:, 0], off[:, 1])


def test_equivalent_radius() -> None:
    g = CavernGrid.uniform(0, 10, 2, 2)
    cav = Cavern(g, np.array([[3.0, 4.0]] * 3))
    np.testing.assert_allclose(cav.equivalent_radius(), np.sqrt(12.5))
