"""Этап 6: сетка, закон ползучести, критерии, сводный расчёт и команда mechanics."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from saltcav.mechanics.creep import BGRA, CreepLaw, MunsonDawson, von_mises
from saltcav.mechanics.criteria import (
    dilatancy_fos,
    invariants,
    principal_compressive,
    wall_tangential,
)
from saltcav.mechanics.fem import SQRT2
from saltcav.mesh.axisym import cavity_profiles, default_domain, domain_polygon, equivalent_radius

ROOT = Path(__file__).resolve().parents[2]
MPA = 1e6


def _sig(sr: float, sz: float, st: float, srz: float = 0.0) -> np.ndarray:
    return np.array([sr, sz, st, SQRT2 * srz]) * MPA


# --- сетка ---------------------------------------------------------------------------


def test_equivalent_radius_keeps_area() -> None:
    r = np.array([[1.0, 3.0, 1.0, 3.0]])
    req = equivalent_radius(r)
    assert np.pi * req[0] ** 2 == pytest.approx(np.mean(np.pi * r**2))


def test_cavity_profiles_split_and_merge() -> None:
    depth = np.arange(10.0)
    r = np.array([0.1, 0.1, 5, 6, 0.1, 0.1, 4, 0.1, 3, 0.1])
    prof = cavity_profiles(depth, r, r_min=0.5)
    # 2–3 — первая полость; узел 7 — одиночный, полости 6 и 8 сливаются через перешеек
    assert len(prof) == 2
    np.testing.assert_array_equal(prof[0][:, 1], [1, 2, 3, 4])
    assert prof[0][0, 0] == 0 and prof[0][-1, 0] == 0  # концы на оси
    np.testing.assert_allclose(prof[1][1:-1, 0], [4, 0.5, 3])
    assert cavity_profiles(depth, np.full(10, 0.15)) == []


def test_domain_polygon_is_ccw_and_tagged() -> None:
    depth = np.arange(100.0, 111.0)
    r = np.r_[0.1, np.full(9, 5.0), 0.1]
    prof = cavity_profiles(depth, r)
    dom = default_domain(prof, h_wall=1.0)
    poly, tags = domain_polygon(prof, dom)
    assert len(tags) == poly.shape[0]
    x, y = poly[:, 0], poly[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0  # против часовой
    assert set(tags) == {"bottom", "outer", "top", "axis", "cavity"}
    assert dom.depth_top < 100 and dom.depth_bottom > 110 and dom.r_out == pytest.approx(40.0)
    with pytest.raises(ValueError):
        default_domain([])


def test_cavern_mesh_volume_matches_profile() -> None:
    pytest.importorskip("gmsh")
    from saltcav.mechanics.fem import AxiFEM
    from saltcav.mesh.axisym import cavern_mesh

    depth = np.linspace(1000.0, 1040.0, 41)
    r = 10.0 * np.sin(np.pi * (depth - 1000.0) / 40.0) + 0.1
    mesh, _ = cavern_mesh(depth, r, h_wall=1.0)
    rr = np.where(r >= 0.5, r, 0.0)
    frustum = np.sum(np.pi / 3 * np.diff(depth) * (rr[:-1] ** 2 + rr[:-1] * rr[1:] + rr[1:] ** 2))
    assert AxiFEM(mesh).cavity_volume() == pytest.approx(frustum, rel=1e-9)


# --- ползучесть -------------------------------------------------------------------------


def test_bgra_rate_magnitude() -> None:
    """BGRa при 10 МПа и 40 °C — порядка 1e-10 1/с (≈ 0,6 %/год)."""
    rate = BGRA.rate(np.array(10 * MPA), np.array(313.15))
    assert 5e-11 < rate < 5e-10


def test_creep_tangent_matches_finite_difference() -> None:
    law = CreepLaw(transient=None)
    sig = _sig(-20, -12, -15, 3)
    t = np.array(320.0)
    rate, tang, _, _ = law.evaluate(sig, t, np.array(0.0))
    h = 1e3
    fd = np.empty((4, 4))
    for j in range(4):
        dp = np.zeros(4)
        dp[j] = h
        fd[:, j] = (
            law.evaluate(sig + dp, t, np.array(0.0))[0]
            - law.evaluate(sig - dp, t, np.array(0.0))[0]
        ) / (2 * h)
    np.testing.assert_allclose(tang, fd, rtol=1e-5, atol=1e-30)
    assert abs(rate[:3].sum()) < 1e-12 * np.abs(rate).max()  # течение несжимаемо
    assert np.allclose(tang, tang.T)


def test_munson_dawson_hardening_and_recovery() -> None:
    md = MunsonDawson()
    s, t = np.array(10 * MPA), np.array(313.0)
    lim = md.limit(s, t)
    assert 1e-3 < lim < 3e-2
    assert md.factor(s, t, np.array(0.0)) > 100  # начало: быстрая неустановившаяся
    assert md.factor(s, t, lim) == pytest.approx(1.0)
    assert md.factor(s, t, 2 * lim) < 1.0  # возврат
    # за один большой шаг ζ не перескакивает предел: F усреднён по шагу
    law = CreepLaw()
    sig = _sig(-25, -15, -15)
    seq = von_mises(sig)
    _, _, ss, f = law.evaluate(sig, t, np.array(0.0), dt=1e9)
    assert (f - 1) * ss * 1e9 == pytest.approx(float(md.limit(seq, t)), rel=1e-9)


# --- критерии -------------------------------------------------------------------------


def test_principal_and_invariants() -> None:
    sig = _sig(-20, -10, -15, 5)
    p = principal_compressive(sig)
    assert p[0] >= p[1] >= p[2]
    assert p.sum() == pytest.approx(45 * MPA)
    i1, j2 = invariants(sig)
    assert j2 == pytest.approx(von_mises(sig) / np.sqrt(3))
    assert i1 == pytest.approx(45 * MPA)


def test_dilatancy_fos_reference_states() -> None:
    assert dilatancy_fos(_sig(-10, -10, -10)) == 10.0  # гидростатика — без повреждения
    # одноосное сжатие: I1 = σ, √J2 = σ/√3 → FOS = 0,27·√3
    assert dilatancy_fos(_sig(0, -10, 0)) == pytest.approx(0.27 * np.sqrt(3))


def test_wall_tangential_stress() -> None:
    sig = np.array([_sig(-12, -20, -18)])
    vertical = np.array([[0.0, -1.0]])  # вертикальная стенка: касательная вдоль z
    assert wall_tangential(sig, vertical)[0] == pytest.approx(18 * MPA)  # min(σ_z, σ_θ)
    horizontal = np.array([[1.0, 0.0]])  # кровля: касательная вдоль r
    assert wall_tangential(sig, horizontal)[0] == pytest.approx(12 * MPA)


# --- сводный расчёт и команда ---------------------------------------------------------


def _fast_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "example_mechanics.yaml").read_text("utf-8"))
    raw["mechanics"]["h_wall_m"] = 3.0
    raw["mechanics"]["stages"][0]["duration_years"] = 1
    raw["mechanics"]["stages"][1]["duration_years"] = 1
    path = tmp_path / "fast.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_cli_mechanics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    from saltcav.cli import main
    from saltcav.mechanics.run import MechanicsRun

    cfg = _fast_config(tmp_path)
    out, png = tmp_path / "m.npz", tmp_path / "m.png"
    assert main(["mechanics", str(cfg), "--out", str(out), "--plot", str(png)]) == 0
    text = capsys.readouterr().out
    assert "потеря V" in text
    run = MechanicsRun.load(out)
    s = run.summary
    assert run.volume0 == pytest.approx(run.leaching_volume, rel=0.01)
    assert np.all(np.diff(s.convergence_pct) >= -1e-9)  # камера только сходится
    assert s.convergence_pct[-1] > 0
    # газ при низком давлении: запас по дилатансии меньше, чем под рассолом
    gas = run.stage == 1
    assert s.min_fos_wall[gas].min() < s.min_fos_wall[~gas][1:].min()
    assert png.stat().st_size > 10_000
    assert main(["mechanics", str(out), "--plot", str(tmp_path / "again.png")]) == 0
    assert main(["mechanics", str(cfg), "--years", "1"]) == 1  # две ступени — нельзя
