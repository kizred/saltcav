from saltcav.verifier_probe import format_si


def test_format_si_default_digits() -> None:
    assert format_si(12.5, "m") == "12.500 m"


def test_format_si_rounds_value() -> None:
    assert format_si(1.23456, "m", digits=4) == "1.2346 m"


def test_format_si_supports_another_si_unit() -> None:
    assert format_si(42.0, "kg", digits=1) == "42.0 kg"
