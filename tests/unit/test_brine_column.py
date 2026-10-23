import numpy as np
import pytest

from saltcav.leaching import column_step, stabilize, transport_step

# --- плотностная устойчивость -------------------------------------------------


def test_stable_profile_unchanged() -> None:
    c = np.array([100.0, 150.0, 150.0, 300.0])
    np.testing.assert_array_equal(stabilize(c, np.ones(4)), c)


def test_two_layer_inversion_mixes_by_volume() -> None:
    c = stabilize(np.array([300.0, 0.0]), np.array([1.0, 3.0]))
    np.testing.assert_allclose(c, 75.0)


def test_fresh_water_at_bottom_rises_through_column() -> None:
    # пресная вода в нижнем слое под насыщенным рассолом — смешивается со всей колонной
    c = stabilize(np.array([300.0, 300.0, 300.0, 0.0]), np.ones(4))
    np.testing.assert_allclose(c, 225.0)


def test_stabilize_random_properties() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        c = rng.uniform(0, 320, n)
        v = rng.uniform(0.1, 50, n)
        s = stabilize(c, v)
        assert np.all(np.diff(s) >= -1e-12)  # не убывает с глубиной
        assert np.dot(s, v) == pytest.approx(np.dot(c, v), rel=1e-12)  # масса сохранена


# --- перенос ------------------------------------------------------------------


def _random_case(seed: int) -> tuple:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    n = 30
    c = np.sort(rng.uniform(100, 310, n))
    v_old = rng.uniform(5, 50, n)
    v_new = v_old * rng.uniform(1.0, 1.02, n)
    dm = rng.uniform(0, 500, n)
    return c, v_old, v_new, dm


@pytest.mark.parametrize(("k_in", "k_out"), [(25, 10), (5, 28), (12, 12)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_salt_mass_conserved_exactly(k_in: int, k_out: int, seed: int) -> None:
    c, v_old, v_new, dm = _random_case(seed)
    dt = 3600.0
    q_in = 0.05
    res = column_step(c, v_old, v_new, dm, q_in, k_in, k_out, dt)
    before = np.dot(c, v_old) + dm.sum()
    after = np.dot(res.c, v_new) + res.produced_salt
    assert after == pytest.approx(before, rel=1e-12)
    assert res.q_out == pytest.approx(q_in - (v_new.sum() - v_old.sum()) / dt)


def test_layers_outside_path_untouched_without_sources() -> None:
    c = np.linspace(200, 300, 10)
    v = np.full(10, 5.0)
    res = transport_step(c, v, v, np.zeros(10), 0.01, k_in=7, k_out=4, dt=100.0)
    np.testing.assert_allclose(res.c[:4], c[:4], rtol=1e-14)
    np.testing.assert_allclose(res.c[8:], c[8:], rtol=1e-14)
    # на пути убыло ровно столько соли, сколько ушло с рассолом
    assert np.dot(res.c[4:8], v[4:8]) == pytest.approx(np.dot(c[4:8], v[4:8]) - res.produced_salt)
    assert res.c[7] < c[7]  # у башмака закачки — разбавление пресной водой
    assert res.c[4] > c[4]  # поток вверх приносит более плотный рассол снизу


def test_single_layer_flush_matches_exponential() -> None:
    # хорошо перемешанный слой, промывка пресной водой: C = C0·exp(−Qt/V)
    v = np.array([100.0])
    q, t_end, c0 = 0.01, 20_000.0, 300.0
    exact = c0 * np.exp(-q * t_end / v[0])
    errs = []
    for n in (100, 200, 400, 800):
        c = np.array([c0])
        dt = t_end / n
        for _ in range(n):
            c = transport_step(c, v, v, np.zeros(1), q, 0, 0, dt).c
        errs.append(abs(c[0] - exact) / exact)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 1.0) < 0.1)  # неявный Эйлер — первый порядок


def test_large_step_stays_bounded() -> None:
    # неявная схема: при огромном шаге концентрация не уходит в минус и не осциллирует
    c = np.full(20, 300.0)
    v = np.full(20, 0.07)  # затравка у скважины, слой ~1 м
    res = column_step(c, v, v, np.zeros(20), 0.0139, 19, 10, dt=1.0e5)
    assert np.all(res.c >= 0) and np.all(res.c <= 300.0)


def test_growth_larger_than_injection_rejected() -> None:
    v = np.ones(3)
    with pytest.raises(ValueError):
        transport_step(np.zeros(3), v, 2 * v, np.zeros(3), 1e-3, 2, 0, 1.0)


def test_steady_state_with_saturating_source() -> None:
    # колонна с постоянным притоком соли выходит на стационар: отбор = источник
    n = 10
    v = np.full(n, 10.0)
    dm_rate = np.full(n, 0.5)  # кг/с на слой
    q = 0.02
    c = np.zeros(n)
    dt = 500.0
    for _ in range(2000):
        res = column_step(c, v, v, dm_rate * dt, q, n - 1, 0, dt)
        c = res.c
    assert res.produced_salt / dt == pytest.approx(dm_rate.sum(), rel=1e-6)
    # концентрация уходящего рассола = источник / расход
    c_out = res.produced_salt / (dt * res.q_out)
    assert c_out == pytest.approx(dm_rate.sum() / q, rel=1e-6)
    assert np.all(np.diff(c) >= 0)  # профиль устойчив
