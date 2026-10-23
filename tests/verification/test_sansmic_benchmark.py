"""Сверка с SANSMIC (ворота Г2): прямоток в цилиндре R = 5 м, 50 м³/ч, 100 сут.

Эталон — benchmarks/sansmic/reference_direct.npz (SANSMIC 1.0.7, родные единицы
ft/in/bbl). Модели разные (закон растворения, плюм, нерастворимые), поэтому
сверяются интегральные величины с допусками, а не профили точка в точку.
"""

import pytest

from benchmarks.sansmic.compare import Comparison, compare

pytestmark = pytest.mark.verification


@pytest.fixture(scope="module")
def cmp() -> Comparison:
    return compare()


def test_volume_matches(cmp: Comparison) -> None:
    assert cmp.volume_ratio == pytest.approx(1.0, abs=0.10)


def test_mid_radius_matches(cmp: Comparison) -> None:
    assert cmp.radius_ratio == pytest.approx(1.0, abs=0.10)


def test_outlet_saturation_close(cmp: Comparison) -> None:
    assert abs(cmp.sat_out_ours - cmp.sat_out_sansmic) < 0.10


def test_floor_rise_close(cmp: Comparison) -> None:
    # разные модели насыпи (у нас B = 1,5 и погребение слоями) — допуск 5 м
    assert abs(cmp.floor_ours - cmp.floor_sansmic) < 5.0
