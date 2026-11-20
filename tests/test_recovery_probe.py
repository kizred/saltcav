from saltcav.recovery_probe import is_positive


def test_is_positive_returns_true_for_positive_number() -> None:
    assert is_positive(1.0) is True


def test_is_positive_returns_false_for_negative_number() -> None:
    assert is_positive(-1.0) is False


def test_is_positive_returns_false_for_zero() -> None:
    assert is_positive(0.0) is False
