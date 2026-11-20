from saltcav.stability_probe import clamp_si


def test_clamp_si_within_range() -> None:
    assert clamp_si(1.5e6, 1.0e6, 2.0e6) == 1.5e6


def test_clamp_si_below_low() -> None:
    assert clamp_si(0.5e6, 1.0e6, 2.0e6) == 1.0e6


def test_clamp_si_above_high() -> None:
    assert clamp_si(2.5e6, 1.0e6, 2.0e6) == 2.0e6
