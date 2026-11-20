"""Форматирование значений в единицах СИ."""


def format_si(value: float, unit: str, digits: int = 3) -> str:
    """Вернуть значение с заданным числом знаков после запятой и единицей СИ."""
    return f"{value:.{digits}f} {unit}"
