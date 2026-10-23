import numpy as np
import pytest

from saltcav.core import brine
from saltcav.leaching.brine_column import plume_transport_step, transport_step
from saltcav.leaching.plume import plume_fluxes

T = 30.0
CS = float(brine.saturation_concentration(T))
N = 40
DZ = np.ones(N)
R_WIDE = np.full(N, 20.0)


def _plume(c: np.ndarray, k_src: int, r: np.ndarray = R_WIDE, alpha: float = 0.09):  # type: ignore[no-untyped-def]
    return plume_fluxes(c, DZ, r, k_src, q0=0.03, r_tube=0.1, alpha=alpha, temperature_c=T)


def test_volume_and_salt_closure() -> None:
    c = np.full(N, CS)
    p = _plume(c, 30)
    assert p.q_deposit == pytest.approx(0.03 + p.entrained.sum())
    assert p.c_deposit * p.q_deposit == pytest.approx(np.dot(p.entrained, c))
    assert p.entrained.sum() > 0.03  # вовлекает больше, чем закачано


def test_fresh_jet_turns_and_rises_in_uniform_brine() -> None:
    p = _plume(np.full(N, CS), 30)
    assert p.k_turn == 30  # медленная струя (≈1 м/с) разворачивается в слое башмака
    assert p.k_deposit == 0  # смесь легче рассола везде — всплывает до верха колонны


def test_fast_jet_penetrates_deeper() -> None:
    # длина фонтана ~ 2.5·M0^(3/4)/B0^(1/2): узкая труба, ~11 м/с — несколько метров
    c = np.full(N, CS)
    slow = _plume(c, 20)
    fast = plume_fluxes(c, DZ, R_WIDE, 20, q0=0.03, r_tube=0.03, alpha=0.09, temperature_c=T)
    assert fast.k_turn >= slow.k_turn + 3


def test_pure_plume_entrainment_magnitude() -> None:
    # чистый плюм: Q ≈ (6α/5)(9α/10)^(1/3) π^(2/3) B^(1/3) z^(5/3), B = g'·Q0
    c = np.full(N, CS)
    p = _plume(c, 30)
    rho_s, rho_w = float(brine.density_from_concentration(CS, T)), float(brine.water_density(T))
    b0 = 9.81 * (rho_s - rho_w) / rho_s * 0.03
    a = 0.09
    q_theory = (
        (6 * a / 5) * (9 * a / 10) ** (1 / 3) * np.pi ** (2 / 3) * b0 ** (1 / 3) * 30 ** (5 / 3)
    )
    assert p.q_deposit == pytest.approx(q_theory, rel=0.3)


def test_plume_stops_below_fresher_layer() -> None:
    # сверху слой почти пресной воды: смесь тяжелее его и в него не входит
    c = np.r_[np.full(10, 0.02 * CS), np.full(N - 10, CS)]
    p = _plume(c, N - 5)
    assert p.k_deposit >= 9  # допускается «перелёт» по импульсу на один слой
    assert np.all(p.entrained[:9] == 0.0)
    assert p.c_deposit > 0.9 * CS  # смесь почти насыщена — тяжелее пресного слоя


def test_no_entrainment_without_alpha() -> None:
    p = _plume(np.full(N, CS), 30, alpha=0.0)
    assert p.entrained.sum() == 0.0
    assert p.c_deposit == 0.0  # пресная вода отдаётся как есть
    assert p.k_deposit == 0


def test_narrow_cavity_limits_entrainment() -> None:
    wide = _plume(np.full(N, CS), 30)
    narrow = _plume(np.full(N, CS), 30, r=np.full(N, 0.12))
    assert narrow.entrained.sum() < 0.1 * wide.entrained.sum()


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        _plume(np.full(N, CS), N)
    with pytest.raises(ValueError):
        plume_fluxes(np.full(N, CS), DZ, R_WIDE, 5, q0=0.0, r_tube=0.1, alpha=0.1, temperature_c=T)


# --- перенос с плюмом ---------------------------------------------------------


def _case(seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    c = np.sort(rng.uniform(50, CS, N))
    v_old = rng.uniform(5, 80, N)
    v_new = v_old * rng.uniform(1.0, 1.01, N)
    dm = rng.uniform(0, 50, N)
    e = rng.uniform(0, 0.01, N)
    return c, v_old, v_new, dm, e


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(("k_dep", "k_out"), [(0, 35), (12, 3), (20, 20)])
def test_plume_transport_conserves_salt(seed: int, k_dep: int, k_out: int) -> None:
    c, v_old, v_new, dm, e = _case(seed)
    dt, q = 3600.0, 0.03
    res = plume_transport_step(c, v_old, v_new, dm, q, e, k_dep, k_out, dt)
    before = np.dot(c, v_old) + dm.sum()
    after = np.dot(res.c, v_new) + res.produced_salt
    assert after == pytest.approx(before, rel=1e-12)
    assert np.all(res.c >= 0)


def test_plume_transport_bounded_by_inputs() -> None:
    # без источников соли концентрации остаются в пределах начальных (принцип максимума)
    c, v_old, _, _, e = _case(5)
    res = plume_transport_step(c, v_old, v_old, np.zeros(N), 0.03, e, 0, N - 1, 1.0e5)
    assert res.c.min() >= -1e-9 and res.c.max() <= c.max() + 1e-9


def test_reduces_to_path_transport_without_entrainment() -> None:
    # без вовлечения, с отдачей у башмака и без роста объёмов это тот же перенос от
    # закачки к отбору (с ростом объёмов новая схема заполняет прирост встречным
    # течением от соседей, прежняя — нет, поэтому сравниваем при постоянных объёмах)
    c, v_old, _, dm, _ = _case(7)
    for k_in, k_out in [(30, 5), (5, 30)]:
        a = transport_step(c, v_old, v_old, dm, 0.02, k_in, k_out, 1800.0)
        b = plume_transport_step(c, v_old, v_old, dm, 0.02, np.zeros(N), k_in, k_out, 1800.0)
        np.testing.assert_allclose(b.c, a.c, rtol=1e-10)
        assert b.produced_salt == pytest.approx(a.produced_salt, rel=1e-10)
