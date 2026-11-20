"""Ограничение значений СИ-величин заданным диапазоном."""


def clamp_si(value: float, low: float, high: float) -> float:
    """Ограничить значение СИ-величины диапазоном [low, high]."""
    return max(low, min(value, high))
