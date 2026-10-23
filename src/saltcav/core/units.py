"""Перевод промысловых единиц в СИ и обратно.

Правило проекта: внутри расчётов только СИ (м, с, кг, Па, кг/м³).
Функции здесь вызываются только на границах — при чтении данных и выводе.
"""

from __future__ import annotations

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0
BARREL_M3 = 0.158987294928  # нефтяной баррель, м³


def m3_per_hour_to_si(q: float) -> float:
    """Расход м³/ч → м³/с."""
    return q / SECONDS_PER_HOUR


def si_to_m3_per_hour(q: float) -> float:
    """Расход м³/с → м³/ч."""
    return q * SECONDS_PER_HOUR


def days_to_seconds(t: float) -> float:
    """Время сут → с."""
    return t * SECONDS_PER_DAY


def seconds_to_days(t: float) -> float:
    """Время с → сут."""
    return t / SECONDS_PER_DAY


def mpa_to_pa(p: float) -> float:
    """Давление МПа → Па."""
    return p * 1.0e6


def pa_to_mpa(p: float) -> float:
    """Давление Па → МПа."""
    return p / 1.0e6


def g_per_l_to_kg_per_m3(c: float) -> float:
    """Концентрация г/л → кг/м³ (численно совпадают)."""
    return c


def bbl_to_m3(v: float) -> float:
    """Объём баррели → м³ (для данных SPR/SANSMIC)."""
    return v * BARREL_M3


def m3_to_bbl(v: float) -> float:
    """Объём м³ → баррели."""
    return v / BARREL_M3
