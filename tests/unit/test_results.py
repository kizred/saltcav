"""Результаты прогона для просмотра по шагам: запись истории, файл, события, поля."""

from pathlib import Path

import numpy as np
import pytest

from saltcav.core.config import load_config
from saltcav.io.results import Results
from saltcav.leaching import simulate
from saltcav.viz.frames import (
    FIELDS,
    available_fields,
    field_range,
    frame_surface,
    node_field,
    section_profile,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGES = ROOT / "configs" / "example_ledges.yaml"


@pytest.fixture(scope="module")
def res() -> Results:
    cfg = load_config(LEDGES)
    return Results.from_history(simulate(cfg), cfg)


def test_history_records_viewer_fields(res: Results) -> None:
    nt = res.nt
    assert res.wall_rate is not None and res.wall_rate.shape == res.radius.shape
    assert np.all(res.wall_rate[0] == 0) and np.all(res.wall_rate >= 0)
    assert res.k_mult is not None
    # прослой не растворяется: скорость в его узлах ноль на всех шагах
    assert np.all(res.wall_rate[:, res.k_mult[:, 0] == 0] == 0)
    for arr in (res.outlet_concentration, res.shoe_in_depth, res.shoe_out_depth):
        assert arr is not None and arr.shape == (nt,)
    # концентрация на выходе — концентрация слоя башмака отбора
    assert res.shoe_out_depth is not None and res.outlet_concentration is not None
    zc = 0.5 * (res.z_edges[:-1] + res.z_edges[1:])
    k = np.searchsorted(zc, res.shoe_out_depth[-1])
    assert res.outlet_concentration[-1] == pytest.approx(res.concentration[-1][k])
    assert res.stage_index is not None and res.stage_index[-1] == 1  # две ступени


def test_events_sorted_and_complete(res: Results) -> None:
    ev = res.events()
    assert [e.t for e in ev] == sorted(e.t for e in ev)
    kinds = {e.kind for e in ev}
    assert {"collapse", "stage"} <= kinds
    assert sum(e.kind == "collapse" for e in ev) == res.collapses.shape[0] > 0
    assert res.frame_at(res.days[3] + 0.1) == 3


def test_save_load_roundtrip(res: Results, tmp_path: Path) -> None:
    path = tmp_path / "r.npz"
    res.save(path)
    back = Results.load(path)
    assert back.name == res.name and back.nphi == res.nphi and back.n_steps == res.n_steps
    for attr in ("radius", "concentration", "wall_rate", "collapses", "layers", "k_mult"):
        np.testing.assert_array_equal(getattr(back, attr), getattr(res, attr))
    assert "leaching" in back.config_yaml


def test_load_old_format(res: Results, tmp_path: Path) -> None:
    """Файлы run --out до просмотрщика: только геометрия и баланс."""
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        times_s=res.times,
        z_edges_m=res.z_edges,
        radius_m=res.radius,
        concentration_kg_m3=res.concentration,
        volume_m3=res.volume,
    )
    old = Results.load(path)
    assert old.wall_rate is None and old.events() == []
    assert available_fields(old) == ["radius", "growth", "undersat"]
    assert np.all(np.isnan(old.floor_depth))


def test_node_fields(res: Results) -> None:
    i = res.nt // 2
    for key in available_fields(res):
        f = node_field(res, i, key)
        assert f.shape == (res.nz + 1, res.nphi), key
        assert np.all(np.isfinite(f)), key
        lo, hi = field_range(res, key)
        assert hi > lo
    np.testing.assert_allclose(node_field(res, i, "growth"), res.radius[i] - res.radius[0])
    u = node_field(res, i, "undersat")
    assert np.all((u >= 0) & (u <= 100))
    with pytest.raises(KeyError):
        node_field(res, i, "stress")
    assert set(available_fields(res)) == set(FIELDS)


def test_rate_scale_ignores_start_spike(res: Results) -> None:
    lo, hi = field_range(res, "rate")
    peak = max(float(node_field(res, i, "rate").max()) for i in range(res.nt))
    assert lo == 0 and hi < peak  # первые сутки у ствола не задают шкалу


def test_surface_and_section(res: Results) -> None:
    pts, quads, vals = frame_surface(res, res.nt - 1, "radius", n_ring=16)
    assert vals.shape == (pts.shape[0],) and quads.max() < pts.shape[0]
    np.testing.assert_allclose(vals, np.hypot(pts[:, 0], pts[:, 1]), atol=1e-9)
    z, r1, r2 = section_profile(res, res.nt - 1, 0)
    assert z.shape == r1.shape == r2.shape
    np.testing.assert_array_equal(r1, r2)  # одна ось симметрии (nphi = 1)
