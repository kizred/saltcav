"""3D-поверхность камеры из R(z, φ) и её показ через PyVista.

Геометрия строится на чистом NumPy (тестируется без графики); PyVista нужен только
для показа и записи: ``pip install -e ".[viz]"``.

Оси: x, y — горизонталь, z — вверх (z = −глубина), азимут φ отсчитывается от оси x.
Радиус между центрами секторов интерполируется по кругу — поверхность гладкая,
хотя в модели радиус внутри сектора постоянный.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def ring_radius(radius_row: FloatArray, n_ring: int) -> tuple[FloatArray, FloatArray]:
    """Углы кольца (n_ring,) и радиус на них по значениям в центрах секторов."""
    r = np.asarray(radius_row, dtype=float)
    theta = 2.0 * np.pi * np.arange(n_ring) / n_ring
    if r.size == 1:
        return theta, np.full(n_ring, float(r[0]))
    centers = (np.arange(r.size) + 0.5) * 2.0 * np.pi / r.size
    # периодическая интерполяция: продлеваем по кругу на сектор с каждой стороны
    c_ext = np.r_[centers[-1] - 2.0 * np.pi, centers, centers[0] + 2.0 * np.pi]
    r_ext = np.r_[r[-1], r, r[0]]
    return theta, np.interp(theta, c_ext, r_ext)


def wall_surface(
    z_edges: FloatArray, radius: FloatArray, n_ring: int = 72
) -> tuple[FloatArray, IntArray]:
    """Точки (N, 3) и четырёхугольники (M, 4) поверхности стенки.

    ``radius`` — (nz+1, nphi) в узлах глубины ``z_edges`` (глубина вниз).
    """
    z = np.asarray(z_edges, dtype=float)
    r = np.asarray(radius, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if r.shape[0] != z.size:
        raise ValueError("число строк радиуса должно совпадать с числом узлов z")
    pts = np.empty((z.size, n_ring, 3))
    for i in range(z.size):
        theta, rr = ring_radius(r[i], n_ring)
        pts[i, :, 0] = rr * np.cos(theta)
        pts[i, :, 1] = rr * np.sin(theta)
        pts[i, :, 2] = -z[i]
    idx = np.arange(z.size * n_ring).reshape(z.size, n_ring)
    a = idx[:-1, :]
    b = np.roll(idx[:-1, :], -1, axis=1)
    c = np.roll(idx[1:, :], -1, axis=1)
    d = idx[1:, :]
    quads = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
    return pts.reshape(-1, 3), quads.astype(np.int64)


def floor_disc(
    z_edges: FloatArray, radius: FloatArray, z_floor: float, n_ring: int = 72
) -> FloatArray:
    """Контур верха насыпи (n_ring, 3) на глубине z_floor."""
    z = np.asarray(z_edges, dtype=float)
    r = np.asarray(radius, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    row = np.array([np.interp(z_floor, z, r[:, j]) for j in range(r.shape[1])])
    theta, rr = ring_radius(row, n_ring)
    return np.stack([rr * np.cos(theta), rr * np.sin(theta), np.full(n_ring, -z_floor)], axis=1)


def cavity_window(
    z_edges: FloatArray, radius: FloatArray, threshold: float = 1.0, margin: float = 5.0
) -> NDArray[np.bool_]:
    """Узлы для показа: где камера шире threshold, плюс margin метров ствола вокруг."""
    z = np.asarray(z_edges, dtype=float)
    r = np.asarray(radius, dtype=float)
    r_max = r.max(axis=1) if r.ndim == 2 else r
    wide = np.nonzero(r_max > threshold)[0]
    if wide.size == 0:
        return np.ones(z.size, dtype=bool)
    return np.asarray((z >= z[wide[0]] - margin) & (z <= z[wide[-1]] + margin))


def _cyrillic_font() -> str | None:
    """TTF с кириллицей: встроенный шрифт VTK её не содержит. Берём DejaVu из matplotlib."""
    try:
        import matplotlib
    except ImportError:  # pragma: no cover
        return None
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return str(path) if path.exists() else None


def _pyvista() -> Any:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError('для 3D-просмотра нужен PyVista: pip install -e ".[viz]"') from exc
    return pv


def build_scene(
    z_edges: FloatArray,
    radius: FloatArray,
    z_floor: float | None = None,
    title: str = "",
    off_screen: bool = False,
) -> Any:
    """Плоттер PyVista со стенкой (цвет — радиус) и верхом насыпи."""
    pv = _pyvista()
    keep = cavity_window(z_edges, radius)  # ствол скважины далеко над камерой не нужен
    z_show = np.asarray(z_edges)[keep]
    r_show = np.asarray(radius)[keep]
    pts, quads = wall_surface(z_show, r_show)
    faces = np.hstack([np.full((quads.shape[0], 1), 4), quads]).ravel()
    wall = pv.PolyData(pts, faces)
    wall["R, m"] = np.hypot(pts[:, 0], pts[:, 1])

    plotter = pv.Plotter(off_screen=off_screen, window_size=(1100, 900))
    plotter.set_background("#fcfcfb")
    plotter.add_mesh(
        wall,
        scalars="R, m",
        cmap="Blues",
        opacity=0.8,  # полупрозрачно: видна насыпь внутри
        smooth_shading=True,
        scalar_bar_args={"title": "R, m", "color": "#0b0b0b", "vertical": True},
    )
    if z_floor is not None and z_floor < float(np.asarray(z_edges)[-1]):
        ring = floor_disc(z_edges, radius, z_floor)
        disc = pv.PolyData(ring, np.r_[ring.shape[0], np.arange(ring.shape[0])])
        plotter.add_mesh(disc, color="#c9c6bb", label="верх насыпи")
    if title:
        font = _cyrillic_font()
        if font:
            plotter.add_text(title, font_size=11, color="#0b0b0b", font_file=font)
        else:  # pragma: no cover
            plotter.add_text(title, font_size=11, color="#0b0b0b")
    plotter.show_axes()
    plotter.camera_position = "xz"
    plotter.camera.elevation = 20
    plotter.camera.azimuth = 30
    return plotter


def render(
    z_edges: FloatArray,
    radius: FloatArray,
    z_floor: float | None = None,
    title: str = "",
    screenshot: str | Path | None = None,
    vtk: str | Path | None = None,
) -> None:
    """Показать камеру в окне, сохранить скриншот и/или поверхность в VTK (для ParaView)."""
    pv = _pyvista()
    if vtk is not None:
        pts, quads = wall_surface(z_edges, radius)
        faces = np.hstack([np.full((quads.shape[0], 1), 4), quads]).ravel()
        mesh = pv.PolyData(pts, faces)
        mesh["radius_m"] = np.hypot(pts[:, 0], pts[:, 1])
        mesh.save(str(vtk))
    if screenshot is not None:
        plotter = build_scene(z_edges, radius, z_floor, title, off_screen=True)
        plotter.screenshot(str(screenshot))
        plotter.close()
    if screenshot is None and vtk is None:  # pragma: no cover - интерактивное окно
        build_scene(z_edges, radius, z_floor, title).show()
