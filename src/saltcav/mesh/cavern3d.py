"""3D-сетка массива вокруг камеры R(z, φ): тетраэдры P2 (Gmsh, ядро OpenCASCADE).

Полость строится как набор колец по глубине: в каждом узле глубины — замкнутый
сплайн через радиусы секторов; соседние кольца соединяются линейчатыми
поверхностями (как усечённые конусы модели выщелачивания). Радиус кольца
масштабируется так, чтобы площадь сечения совпала с площадью в модели
выщелачивания (секторы — круговые сектора): объём камеры сохраняется.
Узлы с R < r_min считаются массивом; полость закрывается плоским кругом радиуса r_min
у соседнего узла — как перешеек ствола.

Область — цилиндр радиусом r_factor·R_max с запасом height_factor·размер сверху и
снизу. Границы помечены: ``cavity``, ``outer``, ``top``, ``bottom``.
Прослои — горизонтальные слои цилиндра со своим материалом (``Mesh3D.material``),
сетка у кромок карнизов сгущается.
Координаты: x, y — горизонталь, z — вверх (z = −глубина).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_TET10, _TRI6 = 11, 9


@dataclass(frozen=True)
class Mesh3D:
    nodes: FloatArray  # (N, 3)
    tets: IntArray  # (M, 10), порядок узлов Gmsh
    faces: dict[str, IntArray] = field(default_factory=dict)  # (F, 6), порядок Gmsh
    tet_ref: FloatArray = field(default_factory=lambda: np.zeros((10, 3)))  # опорные узлы
    tri_ref: FloatArray = field(default_factory=lambda: np.zeros((6, 2)))
    material: IntArray | None = None  # (M,) 0 — соль, k ≥ 1 — прослой

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.tets.shape[0])

    def boundary_nodes(self, tag: str) -> IntArray:
        f = self.faces.get(tag)
        return np.unique(f) if f is not None and f.size else np.zeros(0, dtype=np.int64)


def sector_area(radius_row: FloatArray) -> float:
    """Площадь сечения в модели выщелачивания: круговые сектора равного угла."""
    r = np.asarray(radius_row, dtype=float)
    return float(np.pi * np.mean(r**2))


def ring_points(radius_row: FloatArray, n_points: int, r_min: float) -> FloatArray:
    """Точки кольца (n, 2): периодическая интерполяция по секторам, площадь как в модели."""
    from saltcav.viz.surface import ring_radius

    r = np.maximum(np.asarray(radius_row, dtype=float), r_min)
    theta, rr = ring_radius(r, n_points)
    xy = np.stack([rr * np.cos(theta), rr * np.sin(theta)], axis=1)
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    target = max(sector_area(r), np.pi * r_min**2)
    return np.asarray(xy * np.sqrt(target / area))


def cavity_runs(radius: FloatArray, r_min: float) -> list[tuple[int, int]]:
    """Отрезки узлов глубины [k_top, k_bot] каждой полости, включая замыкающие узлы."""
    r_max = np.asarray(radius).max(axis=1)
    open_ = r_max >= r_min
    lone = np.zeros_like(open_)
    lone[1:-1] = ~open_[1:-1] & open_[:-2] & open_[2:]
    open_ |= lone  # одиночный узел массива — перешеек, полость не разделяет
    runs, k, n = [], 0, open_.size
    while k < n:
        if not open_[k]:
            k += 1
            continue
        k0 = k
        while k < n and open_[k]:
            k += 1
        runs.append((max(k0 - 1, 0), min(k, n - 1)))
    return runs


@dataclass(frozen=True)
class Domain3D:
    r_out: float
    depth_top: float
    depth_bottom: float
    h_wall: float
    h_far: float
    grow_distance: float


def cavern_mesh_3d(
    depth: FloatArray,
    radius: FloatArray,
    h_wall: float = 3.0,
    r_factor: float = 8.0,
    height_factor: float = 3.0,
    r_min: float = 0.5,
    r_out: float | None = None,
    points_per_sector: int = 2,
    bands: list[Any] | None = None,
) -> tuple[Mesh3D, Domain3D]:
    """Сетка массива вокруг камеры R(z, φ) (nphi ≥ 1)."""
    gmsh = _gmsh()
    z = np.asarray(depth, dtype=float)
    r = np.asarray(radius, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    runs = cavity_runs(r, r_min)
    if not runs:
        raise ValueError("нет полости: камера уже ствола скважины")
    r_cav = float(r[runs[0][0] : runs[-1][1] + 1].max())
    top, bot = float(z[runs[0][0]]), float(z[runs[-1][1]])
    size = max(r_cav, bot - top)
    margin = height_factor * size
    ro = r_factor * r_cav if r_out is None else r_out
    dom = Domain3D(
        r_out=ro,
        depth_top=max(top - margin, 0.0),
        depth_bottom=bot + margin,
        h_wall=h_wall,
        h_far=max(h_wall, 0.25 * min(ro, margin + bot - top)),
        grow_distance=max(4 * h_wall, (0.25 * min(ro, margin + bot - top) - h_wall) / 0.4),
    )
    n_ring = max(16, points_per_sector * r.shape[1]) if r.shape[1] > 1 else 16

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cavern3d")
        occ = gmsh.model.occ
        solids = []
        for k0, k1 in runs:
            wires = []
            for i in range(k0, k1 + 1):
                xy = ring_points(r[i], n_ring, r_min)
                pts = [occ.addPoint(float(x), float(y), -float(z[i])) for x, y in xy]
                curve = occ.addSpline([*pts, pts[0]])
                wires.append(occ.addWire([curve]))
            solids += occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        cyl = occ.addCylinder(0, 0, -dom.depth_bottom, 0, 0, dom.depth_bottom - dom.depth_top, ro)
        body, _ = occ.cut([(3, cyl)], solids, removeTool=False)
        slabs: list[tuple[list[tuple[int, int]], int]] = []
        thick = []
        for b in bands or []:
            top, bot = max(b.top, dom.depth_top), min(b.bottom, dom.depth_bottom)
            if bot - top < 1e-6:
                continue
            s = occ.addCylinder(0, 0, -bot, 0, 0, bot - top, ro)
            cut, _ = occ.cut([(3, s)], solids, removeTool=False)
            if cut:
                slabs.append((cut, b.material))
                thick.append(bot - top)
        occ.remove(solids, recursive=True)
        vol_mat: dict[int, int] = {}
        if slabs:
            out, out_map = occ.fragment(body, [dt for cut, _ in slabs for dt in cut])
            vol_mat = {t: 0 for d, t in out if d == 3}
            k = len(body)
            for cut, mat in slabs:
                for _ in cut:
                    for d, t in out_map[k]:
                        if d == 3:
                            vol_mat[t] = mat
                    k += 1
            body = [(3, t) for t in vol_mat]
        occ.synchronize()

        tags: dict[str, list[int]] = {"cavity": [], "outer": [], "top": [], "bottom": []}
        outer_surf = set()
        for _, s in gmsh.model.getBoundary(body, combined=True, oriented=False):
            s = abs(s)
            outer_surf.add(s)
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
            reach = max(abs(x0), abs(x1), abs(y0), abs(y1))
            if abs(z1 - z0) < 1e-6 and abs(z0 + dom.depth_top) < 1e-6 and reach > 0.6 * ro:
                tags["top"].append(s)
            elif abs(z1 - z0) < 1e-6 and abs(z0 + dom.depth_bottom) < 1e-6 and reach > 0.6 * ro:
                tags["bottom"].append(s)
            elif reach > 0.6 * ro:
                tags["outer"].append(s)
            else:
                tags["cavity"].append(s)
        for name, ids in tags.items():
            gmsh.model.addPhysicalGroup(2, ids, name=name)
        if vol_mat:
            for m in sorted(set(vol_mat.values())):
                ids = [t for t, mm in vol_mat.items() if mm == m]
                gmsh.model.addPhysicalGroup(3, ids, name="salt" if m == 0 else f"layer{m}")
        else:
            gmsh.model.addPhysicalGroup(3, [t for _, t in body], name="salt")

        mf = gmsh.model.mesh.field
        fields = []

        def threshold(kind: str, ids: list[int], size_min: float, reach: float) -> None:
            if not ids:
                return
            fd = mf.add("Distance")
            mf.setNumbers(fd, kind, ids)
            ft = mf.add("Threshold")
            mf.setNumber(ft, "InField", fd)
            mf.setNumber(ft, "SizeMin", size_min)
            mf.setNumber(ft, "SizeMax", dom.h_far)
            mf.setNumber(ft, "DistMin", 0.5 * size_min)
            mf.setNumber(ft, "DistMax", reach)
            fields.append(ft)

        threshold("SurfacesList", tags["cavity"], h_wall, dom.grow_distance)
        if vol_mat:
            layer_vols = [(3, t) for t, m in vol_mat.items() if m > 0]
            iface = sorted(
                {abs(s) for _, s in gmsh.model.getBoundary(layer_vols, oriented=False)} - outer_surf
            )
            tips = []
            for _, c in gmsh.model.getBoundary([(2, s) for s in iface], combined=False,
                                               oriented=False):  # fmt: skip
                x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, abs(c))
                if max(abs(x0), abs(x1), abs(y0), abs(y1)) < 0.6 * ro:
                    tips.append(abs(c))
            t_min = min(thick)
            # в 3D мельче h/2 у кромок не дробим: число тетраэдров растёт как 1/h³
            s_if = min(dom.h_far, max(h_wall, 2.0 * t_min))
            threshold("SurfacesList", iface, s_if, 4.0 * s_if)
            s_tip = min(h_wall, max(t_min / 2, h_wall / 2))
            threshold("CurvesList", sorted(set(tips)), s_tip, 3.0 * h_wall)
        f_min = mf.add("Min")
        mf.setNumbers(f_min, "FieldsList", fields)
        mf.setAsBackgroundMesh(f_min)
        for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber(f"Mesh.{opt}", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT, параллельный
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)  # прямые рёбра: якобиан постоянный
        gmsh.model.mesh.generate(3)
        mesh = _extract(gmsh, tags, vol_mat or None)
    finally:
        gmsh.finalize()
    return mesh, dom


def _extract(
    gmsh: Any, tags: dict[str, list[int]], vol_mat: dict[int, int] | None = None
) -> Mesh3D:
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
    index[ntags] = np.arange(ntags.size)
    blocks, mats = [], []
    for v in vol_mat if vol_mat is not None else [-1]:
        types, _, el_nodes = gmsh.model.mesh.getElements(3, v)
        for t, nd in zip(types, el_nodes, strict=True):
            if t == _TET10:
                blk = np.asarray(nd, dtype=np.int64).reshape(-1, 10)
                blocks.append(blk)
                mats.append(np.full(blk.shape[0], 0 if vol_mat is None else vol_mat[v]))
    tets = index[np.vstack(blocks)]
    faces: dict[str, IntArray] = {}
    for name, ids in tags.items():
        parts = []
        for s in ids:
            ftypes, _, fnodes = gmsh.model.mesh.getElements(2, s)
            for ft, nd in zip(ftypes, fnodes, strict=True):
                if ft == _TRI6:
                    parts.append(index[np.asarray(nd, dtype=np.int64).reshape(-1, 6)])
        faces[name] = np.vstack(parts) if parts else np.zeros((0, 6), dtype=np.int64)
    tet_ref = np.asarray(gmsh.model.mesh.getElementProperties(_TET10)[4], dtype=float).reshape(
        -1, 3
    )
    tri_ref = np.asarray(gmsh.model.mesh.getElementProperties(_TRI6)[4], dtype=float).reshape(-1, 2)
    # используются только узлы тетраэдров (служебные узлы геометрии — нет)
    used = np.unique(tets)
    remap = np.full(xyz.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return Mesh3D(
        nodes=xyz[used],
        tets=remap[tets],
        faces={k: remap[v] for k, v in faces.items()},
        tet_ref=tet_ref,
        tri_ref=tri_ref,
        material=np.concatenate(mats).astype(np.int64) if vol_mat is not None else None,
    )


def _gmsh() -> Any:
    try:
        import gmsh
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise RuntimeError('для сетки нужен Gmsh: pip install -e ".[mechanics]"') from exc
    return gmsh
