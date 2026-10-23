"""Воспроизводимые генераторы случайных чисел для ансамблей.

Один seed ансамбля → детерминированный seed каждой реализации через
:class:`numpy.random.SeedSequence`. Реализация i всегда получает один и тот же
поток чисел, независимо от числа процессов и порядка запуска.
"""

from __future__ import annotations

import numpy as np


def realization_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Независимые генераторы для n реализаций ансамбля."""
    if n < 0:
        raise ValueError("n должно быть неотрицательным")
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(s) for s in children]


def realization_rng(seed: int, index: int) -> np.random.Generator:
    """Генератор реализации index — тот же, что в :func:`realization_rngs`."""
    if index < 0:
        raise ValueError("index должен быть неотрицательным")
    child = np.random.SeedSequence(seed, spawn_key=(index,))
    return np.random.default_rng(child)
