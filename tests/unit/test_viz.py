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


# --- просмотр по шагам -----------------------------------------------------------


@pytest.fixture(scope="module")
def results_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from saltcav.cli import main

    path = tmp_path_factory.mktemp("res") / "ledges.npz"
    cfg = Path(__file__).resolve().parents[2] / "configs" / "example_ledges.yaml"
    assert main(["run", str(cfg), "--out", str(path)]) == 0
    return path


def test_cli_view_batch_exports(results_npz: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    pytest.importorskip("imageio")
    from saltcav.cli import main

    png, gif, pvd = tmp_path / "s.png", tmp_path / "a.gif", tmp_path / "pv" / "series.pvd"
    pvd.parent.mkdir()
    args = ["view", str(results_npz), "--field", "rate", "--day", "60"]
    assert main([*args, "--screenshot", str(png), "--animation", str(gif), "--fps", "4"]) == 0
    assert png.stat().st_size > 1000 and gif.stat().st_size > 1000
    assert main(["view", str(results_npz), "--paraview", str(pvd)]) == 0
    text = pvd.read_text(encoding="utf-8")
    n = text.count("<DataSet")
    assert n > 10 and len(list((tmp_path / "pv" / "series").glob("*.vtp"))) == n
    assert main(["view", str(results_npz), "--field", "stress", "--screenshot", str(png)]) == 1


def test_scene_switches_frames_and_fields(results_npz: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    from saltcav.io.results import Results
    from saltcav.viz.frames import available_fields
    from saltcav.viz.scene import offscreen_scene

    res = Results.load(results_npz)
    scene = offscreen_scene(res, "radius", (400, 300))
    n_pts = scene.wall.n_points
    for i, key in zip(range(0, res.nt, 5), available_fields(res), strict=False):
        scene.set_field(key)
        scene.show_frame(i)
        assert scene.wall.n_points == n_pts  # те же точки, другие координаты
        np.testing.assert_allclose(
            np.hypot(scene.wall.points[:, 0], scene.wall.points[:, 1]).max(),
            res.radius[i][scene.window].max(),
            rtol=1e-6,
        )
    scene.set_clip(True)
    scene.show_frame(res.nt - 1)
    scene.screenshot(tmp_path / "clip.png")
    scene.plotter.close()
    assert (tmp_path / "clip.png").stat().st_size > 1000


def test_viewer_window(results_npz: Path) -> None:
    """Окно просмотрщика: шаги, поля, воспроизведение, переход к событию."""
    import os
    import sys

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("нет дисплея (запуск: xvfb-run pytest)")
    pytest.importorskip("pyvistaqt")
    qtw = pytest.importorskip("PySide6.QtWidgets")
    from saltcav.io.results import Results
    from saltcav.viz.viewer import build_window

    app = qtw.QApplication.instance() or qtw.QApplication([])
    res = Results.load(results_npz)
    win = build_window(res, results_npz, start=0, field="undersat")
    win.show()
    app.processEvents()
    assert win.i == 0 and win.field.currentData() == "undersat"
    win.go(res.nt + 5)
    assert win.i == res.nt - 1 and win.slider.value() == res.nt - 1
    win.toggle_play()  # с конца — начинает сначала
    assert win.i == 0 and win.timer.isActive()
    win._tick()
    assert win.i == 1
    win.toggle_play()
    assert not win.timer.isActive()
    collapse_row = next(k for k, e in enumerate(win.events) if e.kind == "collapse")
    win.ev_list.itemDoubleClicked.emit(win.ev_list.item(collapse_row))
    assert res.times[win.i] >= win.events[collapse_row].t
    for k in range(win.series.count()):
        win.series.setCurrentIndex(k)
    win.field.setCurrentIndex(0)
    app.processEvents()
    win.close()
