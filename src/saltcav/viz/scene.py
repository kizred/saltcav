"""3D-сцена камеры по шагам: одна сцена для окна просмотрщика, анимации и скриншотов.

Сцена создаётся один раз; :meth:`CavernScene.show_frame` только меняет точки и
значения поля в уже добавленных объектах — переключение шага мгновенное.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saltcav.io.results import Results
from saltcav.viz.frames import FIELDS, display_window, field_range, frame_surface
from saltcav.viz.surface import _cyrillic_font, _pyvista, floor_disc, ring_radius

INK = "#0b0b0b"
BACKGROUND = "#fcfcfb"
N_RING = 72
TUBE_GAP = 0.8  # м, колонны на картинке разнесены, чтобы не сливались


def _disc(pv: Any, ring: np.ndarray) -> Any:
    return pv.PolyData(ring, np.r_[ring.shape[0], np.arange(ring.shape[0])])


class CavernScene:
    """Стенка с раскраской по полю, верх насыпи, нерастворитель, колонны и башмаки."""

    def __init__(self, plotter: Any, res: Results, field: str = "radius") -> None:
        self.pv = _pyvista()
        self.plotter = plotter
        self.res = res
        self.window = display_window(res)
        self.field = field
        self.index = 0
        self.clip_half = False
        self._font = _cyrillic_font()
        plotter.set_background(BACKGROUND)

        pts, quads, vals = frame_surface(res, 0, field, self.window, N_RING)
        faces = np.hstack([np.full((quads.shape[0], 1), 4), quads]).ravel()
        self.wall = self.pv.PolyData(pts, faces)
        self.wall["field"] = vals
        self._wall_actor: Any = None
        self._bar_title = ""
        self._add_wall()

        z_top = float(res.z_edges[self.window][0])
        self._z_top = z_top
        self.floor = _disc(self.pv, self._ring(res.floor_depth[0], 0))
        self.blanket = _disc(self.pv, self._ring(z_top, 0))
        plotter.add_mesh(self.floor, color="#c9c6bb", name="floor")
        plotter.add_mesh(self.blanket, color="#eda100", opacity=0.35, name="blanket")
        # колонны: закачка (синяя) и отбор (красная), шарик — башмак
        self.tube_in = self.pv.Line((-TUBE_GAP, 0.0, -z_top), (-TUBE_GAP, 0.0, -z_top - 1.0))
        self.tube_out = self.pv.Line((TUBE_GAP, 0.0, -z_top), (TUBE_GAP, 0.0, -z_top - 1.0))
        self._ball = self.pv.Sphere(radius=0.9)
        self.shoe_in = self._ball.copy()
        self.shoe_out = self._ball.copy()
        for mesh, color, name in (
            (self.tube_in, "#1c5cab", "tube_in"),
            (self.tube_out, "#c0392b", "tube_out"),
        ):
            plotter.add_mesh(mesh, color=color, line_width=4, name=name)
        plotter.add_mesh(self.shoe_in, color="#1c5cab", name="shoe_in")
        plotter.add_mesh(self.shoe_out, color="#c0392b", name="shoe_out")
        self._text: Any = None

        plotter.show_axes()
        plotter.camera_position = "xz"
        plotter.camera.elevation = 20
        plotter.camera.azimuth = 30
        self.show_frame(0)
        plotter.reset_camera()

    # --- построение -----------------------------------------------------------

    def _ring(self, depth: float, i: int) -> np.ndarray:
        z, r = self.res.z_edges, self.res.radius[i]
        if not np.isfinite(depth):
            depth = float(z[-1])
        depth = float(np.clip(depth, z[0], z[-1]))
        ring = floor_disc(z, r, depth, N_RING)
        if self.res.nphi == 1:
            theta, rr = ring_radius(np.array([np.interp(depth, z, r[:, 0])]), N_RING)
            ring[:, 0], ring[:, 1] = rr * np.cos(theta), rr * np.sin(theta)
        return ring

    def _add_wall(self) -> None:
        info = FIELDS[self.field]
        lo, hi = field_range(self.res, self.field, self.window)
        if self._wall_actor is not None:
            self.plotter.remove_actor(self._wall_actor)
            if self._bar_title in self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar(self._bar_title)
        title = info.unit  # название поля — в заголовке сцены
        self._bar_title = title
        bar: dict[str, Any] = {"title": title, "color": INK, "vertical": True, "n_labels": 5}
        mesh = self.wall.clip("y", invert=False) if self.clip_half else self.wall
        self._wall_actor = self.plotter.add_mesh(
            mesh,
            scalars="field",
            cmap=info.cmap,
            clim=(lo, hi),
            opacity=1.0 if self.clip_half else 0.85,
            smooth_shading=True,
            show_edges=False,
            scalar_bar_args=bar,
            name="wall",
        )
        if self._font:
            sb = self.plotter.scalar_bar
            for prop in (sb.GetTitleTextProperty(), sb.GetLabelTextProperty()):
                prop.SetFontFamily(4)  # VTK_FONT_FILE
                prop.SetFontFile(self._font)

    def set_field(self, field: str) -> None:
        self.field = field
        self.show_frame(self.index, rebuild=True)

    def set_clip(self, on: bool) -> None:
        self.clip_half = on
        self.show_frame(self.index, rebuild=True)

    # --- шаг ------------------------------------------------------------------

    def title(self, i: int) -> str:
        r = self.res
        info = FIELDS[self.field]
        vol = f"{r.volume[i]:,.0f}".replace(",", " ")
        return (
            f"{r.name}   сутки {r.days[i]:.0f}   (кадр {i + 1}/{r.nt})\n"
            f"V = {vol} м³\nцвет: {info.label.lower()}, {info.unit}"
        )

    def show_frame(self, i: int, rebuild: bool = False) -> None:
        r = self.res
        i = int(np.clip(i, 0, r.nt - 1))
        self.index = i
        pts, _, vals = frame_surface(r, i, self.field, self.window, N_RING)
        self.wall.points = pts
        self.wall["field"] = vals
        if rebuild or self.clip_half:
            self._add_wall()

        self.floor.points = self._ring(r.floor_depth[i], i)
        bd = r.blanket_depth[i] if r.blanket_depth is not None else self._z_top
        self.blanket.points = self._ring(max(float(bd), self._z_top), i)
        top = -self._z_top
        for line, ball, depth in (
            (self.tube_in, self.shoe_in, None if r.shoe_in_depth is None else r.shoe_in_depth[i]),
            (
                self.tube_out,
                self.shoe_out,
                None if r.shoe_out_depth is None else r.shoe_out_depth[i],
            ),
        ):
            p = np.array(line.points)
            p[0, 2] = top
            p[1, 2] = top if depth is None or not np.isfinite(depth) else -float(depth)
            line.points = p
            ball.points = np.asarray(self._ball.points) + p[1]

        if self._text is not None:
            self.plotter.remove_actor(self._text)
        kw: dict[str, Any] = {"font_size": 10, "color": INK, "position": "upper_left"}
        if self._font:
            kw["font_file"] = self._font
        self._text = self.plotter.add_text(self.title(i), **kw)
        self.plotter.render()

    # --- запись ---------------------------------------------------------------

    def screenshot(self, path: str | Path) -> None:
        self.plotter.screenshot(str(path))


def offscreen_scene(
    res: Results, field: str = "radius", size: tuple[int, int] = (1100, 900)
) -> CavernScene:
    pv = _pyvista()
    return CavernScene(pv.Plotter(off_screen=True, window_size=size), res, field)


def export_animation(
    res: Results,
    path: str | Path,
    field: str = "radius",
    fps: int = 8,
    frames: list[int] | None = None,
) -> int:
    """Анимация по шагам в .gif или .mp4 (нужен imageio, для mp4 — imageio-ffmpeg)."""
    path = Path(path)
    scene = offscreen_scene(res, field, (960, 780))
    pl = scene.plotter
    if path.suffix.lower() == ".gif":
        pl.open_gif(str(path), fps=fps)
    else:
        pl.open_movie(str(path), framerate=fps, quality=7)
    idx = list(range(res.nt)) if frames is None else frames
    for i in idx:
        scene.show_frame(i)
        pl.write_frame()
    pl.close()
    return len(idx)


def export_paraview(res: Results, pvd_path: str | Path) -> int:
    """Серия .vtp по шагам и файл .pvd со временем в сутках — открывается в ParaView."""
    from saltcav.viz.frames import available_fields

    pv = _pyvista()
    pvd = Path(pvd_path).with_suffix(".pvd")
    folder = pvd.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    keys = available_fields(res)
    window = display_window(res)
    rows = []
    for i in range(res.nt):
        pts, quads, _ = frame_surface(res, i, "radius", window, N_RING)
        faces = np.hstack([np.full((quads.shape[0], 1), 4), quads]).ravel()
        mesh = pv.PolyData(pts, faces)
        for key in keys:
            info = FIELDS[key]
            mesh[f"{key} [{info.unit}]"] = frame_surface(res, i, key, window, N_RING)[2]
        name = f"{folder.name}_{i:04d}.vtp"
        mesh.save(str(folder / name))
        rows.append(f'    <DataSet timestep="{res.days[i]:.6g}" file="{folder.name}/{name}"/>')
    pvd.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
        + "\n".join(rows)
        + "\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    return res.nt
