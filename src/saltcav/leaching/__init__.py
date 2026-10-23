"""Модель выщелачивания: стратификация, струя, растворение, нерастворимые, карнизы (этапы 2–4)."""

from saltcav.leaching.brine_column import TransportResult, column_step, stabilize, transport_step

__all__ = ["TransportResult", "column_step", "stabilize", "transport_step"]
