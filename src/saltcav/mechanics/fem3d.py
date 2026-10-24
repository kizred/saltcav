"""3D МКЭ: тетраэдры P2, перемещения (u_x, u_y, u_z), запись Манделя из 6 компонент
[ε_xx, ε_yy, ε_zz, √2ε_yz, √2ε_xz, √2ε_xy], B̄ против запирания (как в осесимметрии).

Интерфейс совпадает с :class:`saltcav.mechanics.fem.AxiFEM`, поэтому решатель
ползучести (:mod:`saltcav.mechanics.solver`) один для обеих постановок.

Объём данных: элементные матрицы не хранятся, а пересчитываются блоками; шаблон
разреженной матрицы строится один раз (сборка — ``bincount`` по готовым индексам).
Линейная система — PARDISO (``pypardiso``, если установлен) или SuperLU.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from saltcav.mesh.cavern3d import Mesh3D

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SQRT2 = float(np.sqrt(2.0))
M6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
P6 = np.eye(6) - np.outer(M6, M6) / 3.0

# квадратура тетраэдра 2-й степени (4 точки), веса — доли объёма 1/6
_A, _B = 0.5854101966249685, 0.1381966011250105
TQP = np.array([[_B, _B, _B], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A]])
TQW = np.full(4, 1.0 / 24.0)
# квадратура треугольника 4-й степени (6 точек, Дунавант), веса — доли площади 1/2
_T = [(0.445948490915965, 0.223381589678011), (0.091576213509771, 0.109951743655322)]
FQP = np.array(
    [(a, a) for a, _ in _T] + [(1 - 2 * a, a) for a, _ in _T] + [(a, 1 - 2 * a) for a, _ in _T]
)
FQW = 0.5 * np.array([w for _, w in _T] * 3)


def _monomials3(p: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    one, zero = np.ones_like(x), np.zeros_like(x)
    v = np.stack([one, x, y, z, x * x, y * y, z * z, x * y, y * z, x * z], axis=-1)
    d = np.stack(
        [
            np.stack([zero, one, zero, zero, 2 * x, zero, zero, y, zero, z], axis=-1),
            np.stack([zero, zero, one, zero, zero, 2 * y, zero, x, z, zero], axis=-1),
            np.stack([zero, zero, zero, one, zero, zero, 2 * z, zero, y, x], axis=-1),
        ],
        axis=-1,
    )
    return v, d


def _monomials2(p: FloatArray) -> tuple[FloatArray, FloatArray]:
    x, y = p[..., 0], p[..., 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    v = np.stack([one, x, y, x * x, y * y, x * y], axis=-1)
    d = np.stack(
        [
            np.stack([zero, one, zero, 2 * x, zero, y], axis=-1),
            np.stack([zero, zero, one, zero, 2 * y, x], axis=-1),
        ],
        axis=-1,
    )
    return v, d


def lagrange_basis(ref: FloatArray, pts: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Квадратичный лагранжев базис по опорным узлам ``ref``: значения (q, n) и
    производные (q, n, dim) в точках ``pts``. Порядок узлов — как в ``ref`` (Gmsh)."""
    mono = _monomials3 if ref.shape[1] == 3 else _monomials2
    vand, _ = mono(ref)
    coef = np.linalg.inv(vand)  # столбец j — коэффициенты функции узла j
    v, d = mono(pts)
    return v @ coef, np.einsum("qmk,mn->qnk", d, coef)


def elastic_matrix6(e_modulus: float, nu: float) -> FloatArray:
    k = e_modulus / (3 * (1 - 2 * nu))
    g = e_modulus / (2 * (1 + nu))
    return np.asarray(k * np.outer(M6, M6) + 2 * g * P6)


@dataclass
class FEM3D:
    """Геометрия точек интегрирования, сборка и граничные условия (тетраэдры P2)."""

    mesh: Mesh3D
    bbar: bool = True
    chunk: int = 4000
    solver: str = "auto"  # auto | pardiso | superlu
    fixed: NDArray[np.bool_] = field(init=False)

    # совместимость с решателем: «y» — вертикальная координата (вверх), «r» — радиус
    mvec: FloatArray = field(init=False, default_factory=lambda: M6.copy())
    qw: FloatArray = field(init=False, default_factory=lambda: TQW.copy())

    def __post_init__(self) -> None:
        m = self.mesh
        n, dn = lagrange_basis(m.tet_ref, TQP)  # (q, 10), (q, 10, 3)
        xyz = m.nodes[m.tets]  # (M, 10, 3)
        jac = np.einsum("eia,qib->eqab", xyz, dn)  # d(x)/d(ξ)
        det = np.linalg.det(jac)
        if np.any(det <= 0):
            raise ValueError("вывернутый элемент 3D-сетки")
        inv = np.linalg.inv(jac)
        self.n = n
        self.dndx = np.einsum("qia,eqab->eqib", dn, inv)  # (M, q, 10, 3)
        pts = np.einsum("qi,eia->eqa", n, xyz)
        self.xq = pts
        self.yq = pts[..., 2]  # вертикальная координата
        self.rq = np.hypot(pts[..., 0], pts[..., 1])
        self.wq = TQW[None, :] * det
        self.vol_mean: FloatArray | None = None
        if self.bbar:
            # среднее по элементу от div N (для B̄): (M, 10, 3)
            w = self.wq / self.wq.sum(axis=1, keepdims=True)
            self.vol_mean = np.einsum("eq,eqia->eia", w, self.dndx)
        dofs = np.empty((m.n_elements, 30), dtype=np.int64)
        for c in range(3):
            dofs[:, c::3] = 3 * m.tets + c
        self.dofs = dofs
        self.ndof = 3 * m.n_nodes
        self.fixed = np.zeros(self.ndof, dtype=bool)
        self._owner: dict[tuple[int, ...], int] | None = None
        self._factor: Callable[[FloatArray], FloatArray] | None = None
        self._factor_n = 0
        self._pardiso: list[Any] = []
        self.factorizations = 0
        self.iterations: list[int] = []
        self.rtol = 1e-7
        self._pattern()

    # --- B-матрицы ------------------------------------------------------------------

    def bmat(self, sl: slice) -> FloatArray:
        """B (c, q, 6, 30) для элементов среза ``sl``."""
        g = self.dndx[sl]  # (c, q, 10, 3)
        c, q = g.shape[:2]
        b = np.zeros((c, q, 6, 30))
        gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
        b[..., 0, 0::3] = gx
        b[..., 1, 1::3] = gy
        b[..., 2, 2::3] = gz
        b[..., 3, 1::3] = gz / SQRT2
        b[..., 3, 2::3] = gy / SQRT2
        b[..., 4, 0::3] = gz / SQRT2
        b[..., 4, 2::3] = gx / SQRT2
        b[..., 5, 0::3] = gy / SQRT2
        b[..., 5, 1::3] = gx / SQRT2
        if self.vol_mean is not None:
            vol = np.zeros((c, q, 30))
            for k in range(3):
                vol[..., k::3] = g[..., k]
            mean = np.zeros((c, 30))
            for k in range(3):
                mean[:, k::3] = self.vol_mean[sl][..., k]
            b += (M6[None, None, :, None] / 3.0) * (mean[:, None, :] - vol)[:, :, None, :]
        return b

    def _chunks(self) -> list[slice]:
        ne = self.mesh.n_elements
        return [slice(i, min(i + self.chunk, ne)) for i in range(0, ne, self.chunk)]

    # --- сборка -----------------------------------------------------------------------

    def _pattern(self) -> None:
        """Шаблон CSR: для каждого элементного вклада — индекс в data."""
        key = (self.dofs[:, :, None] * self.ndof + self.dofs[:, None, :]).ravel()
        uniq, inv = np.unique(key, return_inverse=True)
        del key
        self._inv = inv.astype(np.int32)
        del inv
        r = uniq // self.ndof
        c = uniq % self.ndof
        indptr = np.zeros(self.ndof + 1, dtype=np.int64)
        np.add.at(indptr, r + 1, 1)
        self._indptr = np.cumsum(indptr)
        self._indices = c.astype(np.int32)
        self._nnz = uniq.size

    def stiffness(self, d: FloatArray) -> sp.csr_matrix:
        data = np.zeros(self._nnz)
        for sl in self._chunks():
            b = self.bmat(sl)
            dd = d if d.ndim == 2 else d[sl]
            db = np.matmul(dd, b)  # (c, q, 6, 30), пакетное BLAS-умножение
            bw = b * self.wq[sl][:, :, None, None]
            ke = np.matmul(bw.transpose(0, 1, 3, 2), db).sum(axis=1)  # (c, 30, 30)
            idx = self._inv[sl.start * 900 : sl.stop * 900]
            data += np.bincount(idx, ke.ravel(), minlength=self._nnz)
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.ndof, self.ndof))

    def internal(self, sig: FloatArray) -> FloatArray:
        f = np.zeros(self.ndof)
        for sl in self._chunks():
            b = self.bmat(sl)
            fe = np.matmul(b.transpose(0, 1, 3, 2), (sig[sl] * self.wq[sl][..., None])[..., None])
            fe = fe[..., 0].sum(axis=1)
            f += np.bincount(self.dofs[sl].ravel(), fe.ravel(), minlength=self.ndof)
        return f

    def strain(self, u: FloatArray) -> FloatArray:
        out = np.empty((*self.wq.shape, 6))
        for sl in self._chunks():
            out[sl] = np.matmul(self.bmat(sl), u[self.dofs[sl]][:, None, :, None])[..., 0]
        return out

    def elastic(self, e_modulus: float, nu: float) -> FloatArray:
        return elastic_matrix6(e_modulus, nu)

    # --- граничные условия -------------------------------------------------------------

    def fix(self, tag: str, component: int) -> None:
        """component: 0 — радиальная (катки на цилиндре), 1 — вертикальная, 2 — все."""
        nodes = self.mesh.boundary_nodes(tag)
        if tag == "axis":  # оси нет: жёсткое смещение и поворот снимаются точками внизу
            return
        if tag == "outer" and component == 0:
            # u·n = 0 на цилиндре: закрепляем обе горизонтальные компоненты — приближение
            # катков для ячейки ряда камер
            self.fixed[3 * nodes] = True
            self.fixed[3 * nodes + 1] = True
            return
        comp = 2 if component == 1 else component
        self.fixed[3 * nodes + comp] = True
        if tag == "bottom" and component == 1:
            self._pin_bottom(nodes)

    def _pin_bottom(self, nodes: IntArray) -> None:
        """Снять горизонтальное смещение и поворот вокруг вертикали: точка в центре дна
        (x, y) и точка на краю дна (y)."""
        p = self.mesh.nodes[nodes]
        c = nodes[np.argmin(np.hypot(p[:, 0], p[:, 1]))]
        e = nodes[np.argmax(p[:, 0] - np.abs(p[:, 1]))]
        self.fixed[3 * c] = self.fixed[3 * c + 1] = True
        self.fixed[3 * e + 1] = True

    def fix_nodes(self, nodes: IntArray, component: int) -> None:
        self.fixed[3 * np.asarray(nodes) + component] = True

    # --- нагрузки --------------------------------------------------------------------

    def _face_geometry(
        self, tag: str, u: FloatArray | None = None
    ) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
        f = self.mesh.faces.get(tag)
        if f is None or f.size == 0:
            z = np.zeros((0,))
            return np.zeros((0, 6), dtype=np.int64), z, z, z
        n, dn = lagrange_basis(self.mesh.tri_ref, FQP)  # (p, 6), (p, 6, 2)
        xyz = self.mesh.nodes[f]  # (F, 6, 3)
        if u is not None:  # деформированная граница (для объёма полости)
            xyz = xyz + u.reshape(-1, 3)[f]
        pts = np.einsum("pi,fia->fpa", n, xyz)
        t1 = np.einsum("pi,fia->fpa", dn[..., 0], xyz)
        t2 = np.einsum("pi,fia->fpa", dn[..., 1], xyz)
        nda = np.cross(t1, t2)  # нормаль × якобиан площади
        # нормаль направить из массива наружу (в полость): от центра тетраэдра к грани
        return f, n, pts, self._orient(f, nda)

    def _orient(self, f: IntArray, nda: FloatArray) -> FloatArray:
        """Ориентировать нормали граней наружу из массива (по соседнему тетраэдру)."""
        owner = self._face_owner()
        elems = np.array([owner[tuple(sorted(face[:3]))] for face in f])
        cen = self.mesh.nodes[self.mesh.tets[elems, :4]].mean(axis=1)
        fc = self.mesh.nodes[f[:, :3]].mean(axis=1)
        sign = np.where(np.einsum("fa,fa->f", nda[:, 0], fc - cen) > 0, 1.0, -1.0)
        return np.asarray(nda * sign[:, None, None])

    def _face_owner(self) -> dict[tuple[int, ...], int]:
        if self._owner is None:
            t = self.mesh.tets[:, :4]
            d: dict[tuple[int, ...], int] = {}
            for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                tri = np.sort(t[:, [a, b, c]], axis=1)
                d.update(zip(map(tuple, tri.tolist()), range(t.shape[0]), strict=True))
            self._owner = d
        return self._owner

    def pressure_load(
        self, tag: str, q: Callable[[FloatArray, FloatArray], FloatArray]
    ) -> FloatArray:
        """Давление q(r, z) на границе ``tag`` (q > 0 давит на массив): f = −∫ q n N dA."""
        f, n, pts, nda = self._face_geometry(tag)
        out = np.zeros(self.ndof)
        if f.size == 0:
            return out
        qv = q(np.hypot(pts[..., 0], pts[..., 1]), pts[..., 2])
        load = -(qv * FQW[None, :])[..., None] * nda  # (F, p, 3)
        fe = np.einsum("pi,fpa->fia", n, load)
        for c in range(3):
            np.add.at(out, 3 * f + c, fe[..., c])
        return out

    def body_load(self, fb: Callable[[FloatArray], FloatArray]) -> FloatArray:
        b = fb(self.xq)  # (M, q, 3)
        fe = np.einsum("qi,eqa,eq->eia", self.n, b, self.wq)
        out = np.zeros(self.ndof)
        for c in range(3):
            np.add.at(out, 3 * self.mesh.tets + c, fe[..., c])
        return out

    # --- решение ------------------------------------------------------------------------

    def solve(self, k: sp.csr_matrix, f: FloatArray) -> FloatArray:
        """K u = f. Разложение матрицы — дорогая часть (секунды); касательная матрица
        ползучести меняется от шага к шагу медленно, поэтому старое разложение служит
        предобусловливателем для сопряжённых градиентов, а пересчитывается, только
        если итераций стало много."""
        free = np.flatnonzero(~self.fixed)
        u = np.zeros(self.ndof)
        kff = sp.csr_matrix(k[free][:, free])
        if self._factor is None or self._factor_n != free.size:
            self._refactor(kff)
        b = f[free]
        x, it = self._pcg(kff, b)
        if x is None:
            self._refactor(kff)
            x, it = self._pcg(kff, b)
            if x is None:  # pragma: no cover - разложение точное, сюда не попадаем
                raise RuntimeError("линейная система не решилась")
        self.iterations.append(it)
        u[free] = x
        return u

    def _refactor(self, kff: sp.csr_matrix) -> None:
        self._factor = None  # старое разложение освобождается до нового (память)
        self._factor = _factorize_spd(kff, self.solver, self._pardiso)
        self._factor_n = kff.shape[0]
        self.factorizations += 1

    def _pcg(
        self, a: sp.csr_matrix, b: FloatArray, max_it: int = 12
    ) -> tuple[FloatArray | None, int]:
        """Сопряжённые градиенты с предобусловливателем — разложением ``_factor``."""
        solve = self._factor
        assert solve is not None
        nb = float(np.linalg.norm(b))
        if nb == 0:
            return np.zeros_like(b), 0
        x = solve(b)
        r = b - a @ x
        z = solve(r)
        p = z.copy()
        rz = float(r @ z)
        for it in range(1, max_it + 1):
            if np.linalg.norm(r) <= self.rtol * nb:
                return x, it - 1
            ap = a @ p
            alpha = rz / float(p @ ap)
            x += alpha * p
            r -= alpha * ap
            z = solve(r)
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return (x, max_it) if np.linalg.norm(r) <= self.rtol * nb else (None, max_it)

    def cavity_volume(self, u: FloatArray | None = None, tag: str = "cavity") -> float:
        """Объём полостей по (деформированной) границе: V = −(1/3)∮ x·n dA, n — из массива.

        Нормаль и площадь берутся на деформированной границе: при неподвижной нормали
        изменение объёма занижается втрое (вклад площади — 2/3)."""
        f, _, pts, nda = self._face_geometry(tag, u)
        if f.size == 0:
            return 0.0
        return float(-np.sum(np.einsum("fpa,fpa->fp", pts, nda) * FQW[None, :]) / 3.0)


def _factorize_spd(
    k: sp.csr_matrix, solver: str, holder: list[Any]
) -> Callable[[FloatArray], FloatArray]:
    """Разложение симметричной положительно определённой матрицы → функция решения.

    ``holder`` хранит один экземпляр PARDISO на задачу: при новом разложении память
    прежнего освобождается, а не копится."""
    if solver in ("auto", "pardiso"):
        if sys.platform.startswith("linux"):
            # Gmsh на Linux собран с GNU OpenMP (libgomp), MKL по умолчанию — с Intel
            # OpenMP (libiomp5); две среды OpenMP в одном процессе падают. Переводим MKL
            # на libgomp — до первой загрузки MKL.
            os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        try:
            import pypardiso
        except ImportError:
            if solver == "pardiso":
                raise
        else:
            if holder:
                ps = holder[0]
                ps.free_memory(everything=False)
            else:
                ps = pypardiso.PyPardisoSolver()
                ps.set_matrix_type(2)  # вещественная симметричная положительно определённая
                holder.append(ps)
            upper = sp.triu(k, format="csr")
            ps.factorize(upper)
            return lambda b: np.asarray(ps.solve(upper, b))
    lu = spla.splu(sp.csc_matrix(k))
    return lambda b: np.asarray(lu.solve(b))
