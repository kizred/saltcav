"""Этап 6: верификация осесимметричного МКЭ и ползучести по аналитическим решениям."""

import numpy as np
import pytest

pytest.importorskip("gmsh")

from saltcav.mechanics.creep import CreepLaw, PowerBranch
from saltcav.mechanics.fem import AxiFEM, elastic_matrix
from saltcav.mechanics.solver import YEAR, MechanicsSetup, Stage, run_mechanics
from saltcav.mesh.axisym import mesh_polygon

pytestmark = pytest.mark.verification

E, NU = 25e9, 0.25
G = E / (2 * (1 + NU))


def _annulus(a: float, b: float, h: float, size: float) -> AxiFEM:
    poly = np.array([[a, 0.0], [b, 0.0], [b, h], [a, h]])
    mesh = mesh_polygon(poly, ["bottom", "outer", "top", "cavity"], size, 4 * size, b)
    return AxiFEM(mesh)


def test_lame_thick_cylinder_converges() -> None:
    """Толстостенный цилиндр под внутренним давлением, плоская деформация (Ламе)."""
    a, b, p = 1.0, 5.0, 10e6
    errors = []
    for size in (0.4, 0.2, 0.1):
        fem = _annulus(a, b, 1.0, size)
        fem.fix("bottom", 1)
        fem.fix("top", 1)
        u = fem.solve(
            fem.stiffness(elastic_matrix(E, NU)),
            fem.pressure_load("cavity", lambda r, y: p + 0 * r),
        )
        r = fem.mesh.nodes[:, 0]
        exact = p * a**2 / (b**2 - a**2) * ((1 + NU) * (1 - 2 * NU) * r + (1 + NU) * b**2 / r) / E
        errors.append(np.max(np.abs(u[0::2] - exact)) / exact.max())
        assert np.max(np.abs(u[1::2])) < 1e-3 * exact.max()
    assert errors[-1] < 1e-4
    assert errors[0] / errors[-1] > 10  # сходимость с измельчением


def test_spherical_cavity_in_infinite_medium() -> None:
    """Сферическая полость: u(a) = p·a/(4G); граница области на 40a."""
    a, big, p = 1.0, 40.0, 10e6
    th = np.linspace(np.pi / 2, 0.0, 41)
    arc = np.c_[a * np.cos(th), a * np.sin(th)]
    poly = np.vstack([[a, 0.0], [big, 0.0], [big, big], [0.0, big], arc[:-1]])
    tags = ["bottom", "outer", "top", "axis"] + ["cavity"] * (len(arc) - 1)
    fem = AxiFEM(mesh_polygon(poly, tags, 0.05, 6.0, 20.0))
    fem.fix("bottom", 1)
    fem.fix("axis", 0)
    u = fem.solve(
        fem.stiffness(elastic_matrix(E, NU)), fem.pressure_load("cavity", lambda r, y: p + 0 * r)
    )
    nodes = fem.mesh.boundary_nodes("cavity")
    x = fem.mesh.nodes[nodes]
    ur = np.einsum("ij,ij->i", u.reshape(-1, 2)[nodes], x) / np.hypot(x[:, 0], x[:, 1])
    assert ur.mean() == pytest.approx(p * a / (4 * G), rel=2e-3)
    assert ur.std() / ur.mean() < 1e-3
    # объём полости — полусфера по Паппу–Гюльдену (половина сферы, симметрия)
    assert fem.cavity_volume() == pytest.approx(2 / 3 * np.pi * a**3, rel=1e-3)


def _mms_fields(r: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bub = r * (1 - r) * y * (1 - y)
    return bub * (1 + r + 0.5 * y), bub * (2 - y + r**2)


def _mms_stress(r: np.ndarray, y: np.ndarray, h: float = 1e-5) -> np.ndarray:
    ur = lambda rr, yy: _mms_fields(rr, yy)[0]  # noqa: E731
    uz = lambda rr, yy: _mms_fields(rr, yy)[1]  # noqa: E731
    err = (ur(r + h, y) - ur(r - h, y)) / (2 * h)
    ezz = (uz(r, y + h) - uz(r, y - h)) / (2 * h)
    ett = ur(r, y) / r
    grz = (ur(r, y + h) - ur(r, y - h)) / (2 * h) + (uz(r + h, y) - uz(r - h, y)) / (2 * h)
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    tr = err + ezz + ett
    return np.stack(
        [lam * tr + 2 * G * err, lam * tr + 2 * G * ezz, lam * tr + 2 * G * ett, G * grz], axis=-1
    )


def _mms_body(r: np.ndarray, y: np.ndarray, h: float = 1e-4) -> np.ndarray:
    s = _mms_stress(r, y)
    dsr = (_mms_stress(r + h, y)[..., 0] - _mms_stress(r - h, y)[..., 0]) / (2 * h)
    dsrz_z = (_mms_stress(r, y + h)[..., 3] - _mms_stress(r, y - h)[..., 3]) / (2 * h)
    dsrz_r = (_mms_stress(r + h, y)[..., 3] - _mms_stress(r - h, y)[..., 3]) / (2 * h)
    dsz = (_mms_stress(r, y + h)[..., 1] - _mms_stress(r, y - h)[..., 1]) / (2 * h)
    br = -(dsr + dsrz_z + (s[..., 0] - s[..., 2]) / r)
    bz = -(dsrz_r + dsz + s[..., 3] / r)
    return np.stack([br, bz], axis=-1)


@pytest.mark.parametrize(("bbar", "order"), [(False, 2.7), (True, 1.7)])
def test_manufactured_solution_order(bbar: bool, order: float) -> None:
    """Метод производственных решений, ошибка перемещений в L2: P2 — порядок 3;
    с B̄ (объёмная деформация постоянна в элементе) — порядок 2, плата за отсутствие
    запирания при несжимаемой ползучести."""
    sizes, errs = (0.2, 0.1, 0.05), []
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for size in sizes:
        mesh = mesh_polygon(poly, ["bottom", "outer", "top", "axis"], size, size, 1.0)
        fem = AxiFEM(mesh, bbar)
        for tag in ("bottom", "outer", "top", "axis"):
            fem.fix(tag, 0)
            fem.fix(tag, 1)
        u = fem.solve(fem.stiffness(elastic_matrix(E, NU)), fem.body_load(_mms_body))
        uh = np.einsum("qi,eia->eqa", fem.n, u.reshape(-1, 2)[fem.mesh.tri])
        ex = np.stack(_mms_fields(fem.rq, fem.yq), axis=-1)
        errs.append(np.sqrt(np.sum(fem.wq * np.sum((uh - ex) ** 2, axis=-1))))
    rates = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(2.0)
    assert rates.min() > order, rates


def test_steady_creep_thick_cylinder() -> None:
    """Установившаяся степенная ползучесть толстостенного цилиндра (плоская деформация):
    скорость стенки u̇(a) = (3/4)·A·(√3/2)^(n−1)·Kⁿ/a, K = 2Δp/(n(a^(−2/n) − b^(−2/n)))."""
    a, b, po, pi, n, amp = 1.0, 8.0, 20e6, 5e6, 5.0, 1e-10
    fem = _annulus(a, b, 0.5, 0.08)
    setup = MechanicsSetup(
        E, NU, lambda y: po + 0 * y, lambda y: 300.0 + 0 * y,
        CreepLaw((PowerBranch(amp, n, 0.0),), transient=None),
        (Stage(10 * YEAR, lambda y: pi + 0 * y),),
        supports=(("bottom", 1), ("top", 1)),
    )  # fmt: skip
    res = run_mechanics(fem, setup)
    cav = fem.mesh.boundary_nodes("cavity")
    ua = np.array([u[cav, 0].mean() for u in res.u])
    vel = np.diff(ua[-3:]) / np.diff(res.times[-3:])
    k = 2 * (po - pi) / 1e6 / (n * (a ** (-2 / n) - b ** (-2 / n)))
    exact = 0.75 * amp * (np.sqrt(3) / 2) ** (n - 1) * k**n / a
    np.testing.assert_allclose(-vel, exact, rtol=0.01)
    assert res.n_steps < 200  # при установившейся ползучести шаг растёт


def test_bbar_prevents_locking() -> None:
    """Без B̄ несжимаемая ползучесть P2 «запирается» — скорость занижена в разы."""
    a, b, po, pi, n, amp = 1.0, 8.0, 20e6, 5e6, 5.0, 1e-10

    def wall_velocity(bbar: bool) -> float:
        poly = np.array([[a, 0.0], [b, 0.0], [b, 0.5], [a, 0.5]])
        fem = AxiFEM(mesh_polygon(poly, ["bottom", "outer", "top", "cavity"], 0.1, 0.4, b), bbar)
        setup = MechanicsSetup(
            E, NU, lambda y: po + 0 * y, lambda y: 300.0 + 0 * y,
            CreepLaw((PowerBranch(amp, n, 0.0),), transient=None),
            (Stage(5 * YEAR, lambda y: pi + 0 * y),), supports=(("bottom", 1), ("top", 1)),
        )  # fmt: skip
        res = run_mechanics(fem, setup)
        cav = fem.mesh.boundary_nodes("cavity")
        ua = [u[cav, 0].mean() for u in res.u[-2:]]
        return float(-(ua[1] - ua[0]) / (res.times[-1] - res.times[-2]))

    assert wall_velocity(False) < 0.8 * wall_velocity(True)
