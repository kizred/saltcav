"""Осесимметричная сетка массива вокруг камеры: профиль R(z) → треугольники P2 (Gmsh).

Координаты сетки: r — расстояние от оси, y = −глубина (вверх). Область — прямоугольник
[0, r_out] × [y_низ, y_верх] без полостей камеры. Полость прилегает к оси: её граница —
профиль r = R(z) и отрезки оси. Ствол скважины (R < r_min) считается заполненным —
это масштаб, несущественный для массива.

Форма 3D-камеры приводится к осесимметричной по эквивалентному радиусу
R_eq(z) = √(среднее по φ R²) — площадь каждого сечения сохраняется.

Границы помечены: ``axis``, ``cavity``, ``outer``, ``top``, ``bottom``.

Нерастворимые прослои (ангидрит, глина) — горизонтальные полосы от стенки камеры до
внешней границы со своим номером материала (``AxiMesh.material``: 0 — соль). Сетка
согласована по границам полос; у кромок карнизов (где полоса выходит на стенку) она
сгущается до трети мощности прослоя, вдоль границ полос — до двух мощностей.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TAGS = ("axis", "cavity", "outer", "top", "bottom")
_TRI6, _LINE3 = 9, 8  # типы элементов Gmsh


@dataclass(frozen=True)
class AxiMesh:
    """Сетка из треугольников P2 в плоскости (r, y)."""

    nodes: FloatArray  # (N, 2): r, y
    tri: IntArray  # (M, 6): вершины 0, 1, 2, середины рёбер 01, 12, 20
    edges: dict[str, IntArray] = field(default_factory=dict)  # (E, 3): концы, середина
    material: IntArray | None = None  # (M,) номер материала: 0 — соль, k ≥ 1 — прослой

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.tri.shape[0])

    def boundary_nodes(self, tag: str) -> IntArray:
        e = self.edges.get(tag)
        return np.unique(e) if e is not None and e.size else np.zeros(0, dtype=np.int64)


def equivalent_radius(radius: FloatArray) -> FloatArray:
    """R_eq(z) = √(среднее по секторам R²) — сохраняет площадь сечения."""
    r = np.asarray(radius, dtype=float)
    if r.ndim == 1:
        return r.copy()
    return np.asarray(np.sqrt(np.mean(r**2, axis=1)))


def cavity_profiles(depth: FloatArray, r: FloatArray, r_min: float = 0.5) -> list[FloatArray]:
    """Полости вдоль оси: для каждой — точки (r, глубина) от оси у кровли до оси у подошвы.

    Узлы с R < r_min (ствол, нерастворимый прослой) считаются массивом; полость
    начинается и заканчивается на оси в соседнем с ней узле массива. Одиночный узел
    массива между полостями (прослой тоньше шага сетки) не может разделить их — там
    остаётся перешеек радиусом r_min.
    """
    depth = np.asarray(depth, dtype=float)
    r = np.array(r, dtype=float)
    open_ = r >= r_min
    lone = np.zeros_like(open_)
    lone[1:-1] = ~open_[1:-1] & open_[:-2] & open_[2:]
    r[lone] = r_min
    open_ |= lone
    out: list[FloatArray] = []
    k = 0
    n = r.size
    while k < n:
        if not open_[k]:
            k += 1
            continue
        k0 = k
        while k < n and open_[k]:
            k += 1
        k1 = k - 1
        top = max(k0 - 1, 0)
        bot = min(k1 + 1, n - 1)
        pts = [(0.0, depth[top])] if top < k0 else [(0.0, depth[k0])]
        pts += [(float(r[i]), float(depth[i])) for i in range(k0, k1 + 1)]
        pts += [(0.0, depth[bot])] if bot > k1 else [(0.0, depth[k1])]
        out.append(np.array(pts))
    return out


@dataclass(frozen=True)
class Band:
    """Горизонтальный прослой в сетке: глубины кровли и подошвы, номер материала ≥ 1."""

    top: float
    bottom: float
    material: int
    extent: float | None = None  # радиус, до которого прослой в сетке; None — до границы


@dataclass(frozen=True)
class DomainSpec:
    """Размеры области и сетки."""

    r_out: float  # м, внешний радиус
    depth_top: float  # м, верх области
    depth_bottom: float  # м, низ области
    h_wall: float  # м, размер элемента у стенки
    h_far: float  # м, размер элемента на внешних границах
    grow_distance: float  # м, на каком расстоянии от стенки размер достигает h_far


def default_domain(
    profiles: list[FloatArray],
    h_wall: float = 1.0,
    r_factor: float = 8.0,
    height_factor: float = 3.0,
    r_out: float | None = None,
) -> DomainSpec:
    """Область по размерам камеры: радиус r_factor·R_max, сверху и снизу height_factor·H."""
    if not profiles:
        raise ValueError("нет полости: камера уже ствола скважины")
    pts = np.vstack(profiles)
    r_max = float(np.max(pts[:, 0]))
    top, bot = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    height = bot - top
    size = max(r_max, height)
    margin = height_factor * size
    ro = r_factor * r_max if r_out is None else r_out
    if ro <= r_max:
        raise ValueError(f"внешний радиус {ro:.1f} м не больше радиуса камеры {r_max:.1f} м")
    return DomainSpec(
        r_out=ro,
        depth_top=max(top - margin, 0.0),
        depth_bottom=bot + margin,
        h_wall=h_wall,
        h_far=max(h_wall, 0.15 * min(ro, margin + height)),
        grow_distance=2.0 * size,
    )


def domain_polygon(profiles: list[FloatArray], dom: DomainSpec) -> tuple[FloatArray, list[str]]:
    """Замкнутый контур области (против часовой стрелки в (r, y)) и тег каждого отрезка."""

    def y(depth: float) -> float:
        return -float(depth)

    pts: list[tuple[float, float]] = [(0.0, y(dom.depth_bottom)), (dom.r_out, y(dom.depth_bottom))]
    tags = ["bottom"]
    pts.append((dom.r_out, y(dom.depth_top)))
    tags.append("outer")
    pts.append((0.0, y(dom.depth_top)))
    tags.append("top")
    # вниз по оси, обходя полости сверху вниз
    for prof in sorted(profiles, key=lambda p: float(p[0, 1])):
        if not (dom.depth_top < prof[0, 1] and prof[-1, 1] < dom.depth_bottom):
            raise ValueError("полость выходит за область")
        pts.append((0.0, y(prof[0, 1])))
        tags.append("axis")
        for rr, dd in prof[1:]:
            pts.append((float(rr), y(dd)))
            tags.append("cavity")
    tags.append("axis")  # от последней полости к низу оси (замыкание)
    poly = np.array(pts)
    # убрать повторяющиеся соседние точки (полость, начинающаяся на верхней границе и т. п.)
    keep = np.r_[True, np.any(np.abs(np.diff(poly, axis=0)) > 1e-9, axis=1)]
    tags_kept = [t for t, k in zip(tags, np.r_[keep[1:], True], strict=True) if k]
    poly = poly[keep]
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
        tags_kept = tags_kept[:-1]
    return poly, tags_kept


def mesh_polygon(
    poly: FloatArray,
    tags: list[str],
    h_wall: float,
    h_far: float,
    grow_distance: float,
    refine_tags: tuple[str, ...] = ("cavity",),
) -> AxiMesh:
    """Сетка P2 внутри многоугольника: мельче у отрезков ``refine_tags``."""
    gmsh = _gmsh()
    if len(tags) != poly.shape[0]:
        raise ValueError("тегов должно быть столько же, сколько отрезков")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("saltcav")
        geo = gmsh.model.geo
        p_ids = [geo.addPoint(float(x), float(yv), 0.0, h_far) for x, yv in poly]
        n = len(p_ids)
        l_ids = [geo.addLine(p_ids[i], p_ids[(i + 1) % n]) for i in range(n)]
        loop = geo.addCurveLoop(l_ids)
        surf = geo.addPlaneSurface([loop])
        geo.synchronize()
        by_tag: dict[str, list[int]] = {}
        for lid, t in zip(l_ids, tags, strict=True):
            by_tag.setdefault(t, []).append(lid)
        for t, ids in by_tag.items():
            gmsh.model.addPhysicalGroup(1, ids, name=t)
        gmsh.model.addPhysicalGroup(2, [surf], name="salt")

        fine = [lid for t in refine_tags for lid in by_tag.get(t, [])]
        if fine:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", fine)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
            f_thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_wall)
            gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.5 * h_wall)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMax", grow_distance)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)  # середины рёбер на прямых
        gmsh.model.mesh.generate(2)
        return _extract(gmsh, by_tag)
    finally:
        gmsh.finalize()


def _occ_polygon(occ: Any, pts: FloatArray) -> int:
    keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)]
    pts = pts[keep]
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    p_ids = [occ.addPoint(float(x), float(y), 0.0) for x, y in pts]
    n = len(p_ids)
    lines = [occ.addLine(p_ids[i], p_ids[(i + 1) % n]) for i in range(n)]
    return int(occ.addPlaneSurface([occ.addCurveLoop(lines)]))


def mesh_with_layers(profiles: list[FloatArray], dom: DomainSpec, bands: list[Band]) -> AxiMesh:
    """Сетка P2 с прослоями: OpenCASCADE, полосы — отдельные поверхности (материалы)."""
    gmsh = _gmsh()
    y_top, y_bot = -dom.depth_top, -dom.depth_bottom
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("saltcav-layers")
        occ = gmsh.model.occ
        rect = occ.addRectangle(0.0, y_bot, 0.0, dom.r_out, y_top - y_bot)
        cav = [(2, _occ_polygon(occ, np.c_[p[:, 0], -p[:, 1]])) for p in profiles]
        body, _ = occ.cut([(2, rect)], cav, removeObject=True, removeTool=False)
        pieces: list[tuple[list[tuple[int, int]], int]] = []
        thick = []
        for b in bands:
            top, bot = max(b.top, dom.depth_top), min(b.bottom, dom.depth_bottom)
            if bot - top < 1e-6:
                continue
            width = dom.r_out if b.extent is None else min(dom.r_out, b.extent)
            r = occ.addRectangle(0.0, -bot, 0.0, width, bot - top)
            cut, _ = occ.cut([(2, r)], cav, removeObject=True, removeTool=False)
            if cut:
                pieces.append((cut, b.material))
                thick.append(bot - top)
        occ.remove(cav, recursive=True)
        tools = [dt for cut, _ in pieces for dt in cut]
        out, out_map = occ.fragment(body, tools)
        occ.synchronize()
        surf_mat = {t: 0 for d, t in out if d == 2}
        k = len(body)
        for cut, mat in pieces:
            for _ in cut:
                for d, t in out_map[k]:
                    if d == 2:
                        surf_mat[t] = mat
                k += 1
        surfaces = [(2, t) for t in surf_mat]

        eps = 1e-6 * max(dom.r_out, y_top - y_bot)
        by_tag: dict[str, list[int]] = {t: [] for t in TAGS}
        outer_curves = set()
        for _, c in gmsh.model.getBoundary(surfaces, combined=True, oriented=False):
            c = abs(c)
            outer_curves.add(c)
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            if x1 < eps:
                by_tag["axis"].append(c)
            elif x0 > dom.r_out - eps:
                by_tag["outer"].append(c)
            elif y0 > y_top - eps:
                by_tag["top"].append(c)
            elif y1 < y_bot + eps:
                by_tag["bottom"].append(c)
            else:
                by_tag["cavity"].append(c)
        layer_surf = [(2, t) for t, m in surf_mat.items() if m > 0]
        iface = sorted(
            {abs(c) for _, c in gmsh.model.getBoundary(layer_surf, combined=False, oriented=False)}
            - outer_curves
        )
        # кромки карнизов — точки, где граница прослоя выходит на стенку камеры
        cav_pts = {abs(p) for _, p in gmsh.model.getBoundary([(1, c) for c in by_tag["cavity"]],
                                                             combined=False)}  # fmt: skip
        tips = [abs(pt) for _, pt in gmsh.model.getBoundary([(1, c) for c in iface], combined=False)
                if abs(pt) in cav_pts]  # fmt: skip
        for t, ids in by_tag.items():
            if ids:
                gmsh.model.addPhysicalGroup(1, ids, name=t)
        for m in sorted(set(surf_mat.values())):
            ids = [t for t, mm in surf_mat.items() if mm == m]
            gmsh.model.addPhysicalGroup(2, ids, name="salt" if m == 0 else f"layer{m}")

        t_min = min(thick) if thick else dom.h_wall
        mf = gmsh.model.mesh.field
        fields = []

        def threshold(kind: str, ids: list[int], size_min: float, reach: float) -> None:
            if not ids:
                return
            fd = mf.add("Distance")
            mf.setNumbers(fd, kind, ids)
            if kind == "CurvesList":
                mf.setNumber(fd, "Sampling", 200)
            ft = mf.add("Threshold")
            mf.setNumber(ft, "InField", fd)
            mf.setNumber(ft, "SizeMin", size_min)
            mf.setNumber(ft, "SizeMax", dom.h_far)
            mf.setNumber(ft, "DistMin", 0.5 * size_min)
            mf.setNumber(ft, "DistMax", reach)
            fields.append(ft)

        # стенка — как без прослоев; границы полос и кромки карнизов — мелко, но с
        # быстрым ростом, чтобы не мельчить сетку вдоль всего прослоя
        threshold("CurvesList", by_tag["cavity"], dom.h_wall, dom.grow_distance)
        s_if = min(dom.h_far, max(dom.h_wall, 2.0 * t_min))
        threshold("CurvesList", iface, s_if, 6.0 * s_if)
        s_tip = min(dom.h_wall, max(t_min / 3.0, dom.h_wall / 4.0))
        threshold("PointsList", sorted(set(tips)), s_tip, max(4.0 * dom.h_wall, 20.0 * s_tip))
        f_min = mf.add("Min")
        mf.setNumbers(f_min, "FieldsList", fields)
        mf.setAsBackgroundMesh(f_min)
        for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber(f"Mesh.{opt}", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.model.mesh.generate(2)
        return _extract(gmsh, by_tag, surf_mat)
    finally:
        gmsh.finalize()


def _extract(
    gmsh: Any, by_tag: dict[str, list[int]], surf_mat: dict[int, int] | None = None
) -> AxiMesh:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags] = np.arange(tags.size)
    blocks, mats = [], []
    surfaces = [(2, t) for t in surf_mat] if surf_mat is not None else [(2, -1)]
    for _, s in surfaces:
        types, _, el_nodes = gmsh.model.mesh.getElements(2, s)
        for t, nd in zip(types, el_nodes, strict=True):
            if t == _TRI6:
                blk = np.asarray(nd, dtype=np.int64).reshape(-1, 6)
                blocks.append(blk)
                mats.append(np.full(blk.shape[0], 0 if surf_mat is None else surf_mat[s]))
    tri = index[np.vstack(blocks)]
    material = np.concatenate(mats).astype(np.int64) if surf_mat is not None else None
    edges: dict[str, IntArray] = {}
    for t, curves in by_tag.items():
        parts = []
        for c in curves:
            ctypes, _, cnodes = gmsh.model.mesh.getElements(1, c)
            for ct, nd in zip(ctypes, cnodes, strict=True):
                if ct == _LINE3:
                    parts.append(index[np.asarray(nd, dtype=np.int64).reshape(-1, 3)])
        edges[t] = np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.int64)
    nodes = xyz[:, :2].copy()
    nodes[np.abs(nodes[:, 0]) < 1e-12, 0] = 0.0
    tri = _orient(nodes, tri)
    if surf_mat is not None:  # служебные узлы геометрии (не в элементах) — убрать
        used = np.unique(tri)
        remap = np.full(nodes.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        nodes, tri = nodes[used], remap[tri]
        edges = {k: remap[v] for k, v in edges.items()}
    return AxiMesh(nodes=nodes, tri=tri, edges=edges, material=material)


def _orient(nodes: FloatArray, tri: IntArray) -> IntArray:
    """Все треугольники — против часовой стрелки (положительный якобиан)."""
    p = nodes[tri[:, :3]]
    area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (
        p[:, 1, 1] - p[:, 0, 1]
    )
    neg = area2 < 0
    t = tri.copy()
    # меняем вершины 1 и 2, середины 01 ↔ 20, 12 остаётся
    t[neg] = tri[neg][:, [0, 2, 1, 5, 4, 3]]
    return t


def cavern_mesh(
    depth: FloatArray,
    radius: FloatArray,
    h_wall: float = 1.0,
    r_factor: float = 8.0,
    height_factor: float = 3.0,
    r_out: float | None = None,
    r_min: float = 0.5,
    bands: list[Band] | None = None,
) -> tuple[AxiMesh, DomainSpec]:
    """Сетка массива вокруг камеры по R(z) или R(z, φ) (тогда — эквивалентный радиус).
    ``bands`` — прослои; без них — одна соль (геометрическое ядро Gmsh)."""
    r_eq = equivalent_radius(radius)
    profiles = cavity_profiles(depth, r_eq, r_min)
    dom = default_domain(profiles, h_wall, r_factor, height_factor, r_out)
    if bands:
        return mesh_with_layers(profiles, dom, bands), dom
    poly, tags = domain_polygon(profiles, dom)
    mesh = mesh_polygon(poly, tags, dom.h_wall, dom.h_far, dom.grow_distance)
    return mesh, dom


def _gmsh() -> Any:
    try:
        import gmsh
    except (ImportError, OSError) as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError('для сетки нужен Gmsh: pip install -e ".[mechanics]"') from exc
    return gmsh
