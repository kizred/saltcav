from pathlib import Path

import numpy as np
import pytest

from saltcav.viz import cavity_window, floor_disc, ring_radius, wall_surface

Z = np.linspace(1000.0, 1010.0, 6)


def test_ring_radius_constant_and_periodic() -> None:
    _, r = ring_radius(np.array([5.0]), 12)
    np.testing.assert_allclose(r, 5.0)
    _, r = ring_radius(np.array([1.0, 2.0, 3.0, 4.0]), 64)
    assert r.min() >= 1.0 and r.max() <= 4.0
    # значение в центре сектора совпадает с сектором
    _, r4 = ring_radius(np.array([1.0, 2.0, 3.0, 4.0]), 8)
    np.testing.assert_allclose(r4[1::2], [1.0, 2.0, 3.0, 4.0])


def test_wall_surface_shape_and_radius() -> None:
    radius = np.full((6, 3), 7.0)
    pts, quads = wall_surface(Z, radius, n_ring=24)
    assert pts.shape == (6 * 24, 3) and quads.shape == (5 * 24, 4)
    np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1]), 7.0)
    np.testing.assert_allclose(sorted(set(-pts[:, 2])), Z)  # z вверх = −глубина
    assert quads.max() < pts.shape[0]


def test_floor_disc_and_window() -> None:
    radius = np.c_[np.linspace(0.15, 10.0, 6)]
    ring = floor_disc(Z, radius, 1005.0, n_ring=10)
    assert ring.shape == (10, 3)
    np.testing.assert_allclose(ring[:, 2], -1005.0)
    keep = cavity_window(Z, radius, threshold=1.0, margin=0.0)
    assert not keep[0] and keep[-1]


def test_render_screenshot(tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    from saltcav.viz import render

    radius = np.c_[[0.15, 5.0, 8.0, 8.0, 5.0, 0.15]]
    out, vtk = tmp_path / "c.png", tmp_path / "c.vtk"
    render(Z, radius, z_floor=1008.0, title="тест", screenshot=out, vtk=vtk)
    assert out.stat().st_size > 1000 and vtk.exists()


def test_cli_view_from_saved_history(tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    from saltcav.cli import main
    from tests.unit.test_config import EXAMPLE

    npz, png = tmp_path / "h.npz", tmp_path / "v.png"
    assert main(["run", str(EXAMPLE), "--out", str(npz)]) == 0
    assert main(["view", str(npz), "--day", "100", "--screenshot", str(png)]) == 0
    assert png.stat().st_size > 1000


def test_cli_view_bad_source() -> None:
    from saltcav.cli import main

    assert main(["view", "no_such_file.yaml", "--screenshot", "x.png"]) == 1
