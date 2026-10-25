"""Данные скважины: чтение CSV (в т. ч. из Excel), проверка, перенос в конфигурацию."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.core import brine
from saltcav.core.config import load_config
from saltcav.io.fielddata import (
    FieldData,
    SonarSurvey,
    concentration_from_density,
    load_field_data,
    write_field_data,
)

ROOT = Path(__file__).resolve().parents[2]


def _data() -> FieldData:
    return FieldData(
        name="тест",
        temperature_c=25.0,
        operations=[
            {
                "day_start": "0",
                "day_end": "30",
                "mode": "direct",
                "rate_m3_per_h": "50",
                "injection_depth_m": "1090",
                "production_depth_m": "1020",
                "blanket_depth_m": "1010",
            },
        ],
        brine_day=np.array([5.0, 10.0, 20.0]),
        brine_conc=np.array([200.0, 220.5, 240.25]),
        sonar=[
            SonarSurvey(
                30.0,
                np.array([1020.0, 1020.0, 1030.0]),
                np.array([0.0, 90.0, 0.0]),
                np.array([5.0, 7.0, 6.0]),
            )
        ],
        layers=[{"kind": "anhydrite", "top_m": 1050.0, "thickness_m": 1.0}],
    )


@pytest.mark.parametrize("excel", [False, True])
def test_roundtrip(tmp_path: Path, excel: bool) -> None:
    write_field_data(tmp_path, _data(), excel=excel)
    if excel:
        assert ";" in (tmp_path / "brine.csv").read_text(encoding="utf-8-sig")
    d = load_field_data(tmp_path)
    assert d.name == "тест" and d.temperature_c == 25.0
    assert np.allclose(d.brine_conc, [200.0, 220.5, 240.25])
    s = d.sonar[0]
    assert s.azimuth_deg is not None and np.allclose(s.radius, [5, 7, 6])
    _, req = s.equivalent_profile()
    assert np.allclose(req, [np.sqrt((25 + 49) / 2), 6.0])
    assert d.layers[0]["kind"] == "anhydrite"


def test_excel_cp1251_and_density(tmp_path: Path) -> None:
    """Выгрузка из Excel: «;», десятичная запятая, cp1251, лишний столбец, плотность."""
    rho = float(brine.density_from_concentration(250.0, 20.0))
    text = f"day;density_kg_m3;комментарий\n1;{rho:.3f};норма\n2;{rho:.3f};\n".replace(".", ",")
    (tmp_path / "brine.csv").write_bytes(text.encode("cp1251"))
    d = load_field_data(tmp_path)
    assert d.brine_conc == pytest.approx([250.0, 250.0], abs=0.3)
    assert concentration_from_density(np.array([998.2]), 20.0)[0] == pytest.approx(0.0, abs=0.5)


def test_issues(tmp_path: Path) -> None:
    d = _data()
    d.operations.append({**d.operations[0], "day_start": "35", "day_end": "60"})
    d.brine_conc[1] = 500.0
    d.brine_day[2] = 7.0
    msgs = " | ".join(d.issues(load_config(ROOT / "configs" / "benchmark_sansmic_direct.yaml")))
    assert "разрыв" in msgs and "выше насыщения" in msgs and "не возрастает" in msgs


def test_missing_columns(tmp_path: Path) -> None:
    (tmp_path / "sonar.csv").write_text("day,depth_m\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="radius_m"):
        load_field_data(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="нет данных"):
        load_field_data(empty)


def test_apply_to_config() -> None:
    cfg = load_config(ROOT / "configs" / "benchmark_sansmic_direct.yaml")
    out = _data().apply(cfg)
    assert out.schedule.stages[0].duration_days == 30
    assert out.geology.layers[0].top_m == 1050
    assert out.leaching.temperature_c == 25


@pytest.mark.parametrize("folder", ["demo/well-1", "open/sansmic-direct"])
def test_shipped_datasets_are_clean(folder: str) -> None:
    d = load_field_data(ROOT / "data" / folder)
    cfg = load_config(
        ROOT
        / "configs"
        / ("demo_calibration.yaml" if "demo" in folder else "benchmark_sansmic_direct.yaml")
    )
    assert d.issues(cfg) == []
    d.apply(cfg)


def test_template(tmp_path: Path) -> None:
    from saltcav.io.fielddata import write_template

    cfg = load_config(ROOT / "configs" / "demo_calibration.yaml")
    out = write_template(tmp_path / "скв-1", cfg)
    assert (out / "ПРОЧТИ.txt").exists()
    assert (out / "brine.csv").read_text(encoding="utf-8-sig").startswith("day;")
    d = load_field_data(out)  # пустые таблицы — не ошибка; режим и прослои — из cfg
    assert d.brine_day.size == 0 and not d.sonar
    assert len(d.operations) == 2 and d.layers[0]["kind"] == "anhydrite"
    assert d.apply(cfg).schedule == cfg.schedule
    with pytest.raises(ValueError, match="не пуста"):
        write_template(out, cfg)
    empty = write_template(tmp_path / "пусто")  # без конфигурации — только заголовки
    with pytest.raises(ValueError, match="только заголовки"):
        load_field_data(empty)
