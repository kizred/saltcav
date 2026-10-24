"""3D МКЭ (тетраэдры P2): базис, сетка, упругость сферы, совпадение с осесимметрией."""

import numpy as np
import pytest

from saltcav.mechanics.fem3d import FEM3D, elastic_matrix6, lagrange_basis
from saltcav.mesh.cavern3d import cavity_runs, ring_points, sector_area

pytestmark = pytest.mark.verification
E, NU = 25e9, 0.25
G = E / (2 * (1 + NU))


def test_lagrange_basis_is_nodal() -> None:
    ref = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, 0],
         [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]], dtype=float,
    )  # fmt: skip
    n, dn = lagrange_basis(ref, ref)
    np.testing.assert_allclose(n, np.eye(10), atol=1e-12)
    pts = np.random.default_rng(0).random((5, 3)) / 3
    n, dn = lagrange_basis(ref, pts)
    np.testing.assert_allclose(n.sum(axis=1), 1.0)
    np.testing.assert_allclose(dn.sum(axis=1), 0.0, atol=1e-12)


def test_ring_area_and_runs() -> None:
    row = np.array([10.0, 20.0, 10.0, 20.0])
    xy = ring_points(row, 32, 0.5)
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    assert area == pytest.approx(sector_area(row))
    r = np.array([[0.1], [5.0], [6.0], [0.1], [0.1], [4.0], [0.1]])
    assert cavity_runs(r, 0.5) == [(0, 3), (4, 6)]


@pytest.fixture(scope="module")
def sphere_fem() -> FEM3D:
    pytest.importorskip("gmsh")
    from saltcav.mesh.cavern3d import cavern_mesh_3d

    a = 10.0
    z = np.linspace(-a, a, 41) + 1000.0
    r = np.sqrt(np.maximum(a * a - (z - 1000.0) ** 2, 0.0))
    mesh, _ = cavern_mesh_3d(z, r, h_wall=2.0, r_factor=10, height_factor=4, r_min=0.3)
    return FEM3D(mesh)


def test_spherical_cavity_3d(sphere_fem: FEM3D) -> None:
    a, p = 10.0, 10e6
    fem = sphere_fem
    fem.fixed[:] = False
    fem.fix("bottom", 1)
    u = fem.solve(
        fem.stiffness(elastic_matrix6(E, NU)), fem.pressure_load("cavity", lambda r, z: p + 0 * r)
    )
    cav = fem.mesh.boundary_nodes("cavity")
    x = fem.mesh.nodes[cav] - np.array([0.0, 0.0, -1000.0])
    rr = np.linalg.norm(x, axis=1)
    ok = rr > 0.9 * a  # без закрытия у полюсов
    ur = np.einsum("ij,ij->i", u.reshape(-1, 3)[cav], x) / rr
    assert ur[ok].mean() == pytest.approx(p * a / (4 * G), rel=0.02)
    # объём: вписанные кольца с сохранением площади сечений
    assert fem.cavity_volume() == pytest.approx(4 / 3 * np.pi * a**3, rel=0.03)
    dv = fem.cavity_volume(u) - fem.cavity_volume()
    assert dv == pytest.approx(4 * np.pi * a**2 * p * a / (4 * G), rel=0.05)


def test_3d_matches_axisym_creep() -> None:
    """Симметричная камера: 3D и осесимметричная постановки дают одну конвергенцию."""
    pytest.importorskip("gmsh")
    from saltcav.core.config import MechanicsConfig, PressureStageConfig
    from saltcav.mechanics.run import run_from_shape

    z = np.linspace(1000.0, 1040.0, 41)
    r = 12.0 * np.sin(np.pi * (z - 1000.0) / 40.0) ** 0.5 + 0.1
    base = {
        "stages": [PressureStageConfig(duration_years=2.0)],
        "output_every_years": 1.0,
        "transient": None,
        "r_factor": 6.0,
        "height_factor": 2.5,
    }
    ax = run_from_shape(MechanicsConfig(**base, h_wall_m=1.0), z, r)
    d3 = run_from_shape(MechanicsConfig(**base, model="3d", h_wall_3d_m=2.5), z, r)
    assert d3.dim == 3 and ax.dim == 2
    assert d3.volume0 == pytest.approx(ax.volume0, rel=0.02)
    np.testing.assert_allclose(
        d3.summary.convergence_pct[1:], ax.summary.convergence_pct[1:], rtol=0.04
    )
    np.testing.assert_allclose(d3.summary.min_fos_wall, ax.summary.min_fos_wall, rtol=0.08)
