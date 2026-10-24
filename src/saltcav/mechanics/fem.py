"""Осесимметричный МКЭ: треугольники P2, перемещения (u_r, u_z).

Деформации и напряжения в записи Манделя: [ε_r, ε_z, ε_θ, √2·ε_rz] — тогда
σ:ε = σᵀε, а матрицы упругости и касательные симметричны без множителей у сдвига.
Интегралы — по меридиональному сечению с весом r (множитель 2π опущен везде
одинаково, кроме объёма полости).

Всё векторизовано по элементам и точкам интегрирования: матрицы B хранятся для
всех точек, произведения — пакетные матричные (BLAS), шаблон разреженной матрицы
строится один раз. Линейные системы — :class:`~saltcav.mechanics.linsolve.ReusedFactorSolver`
(одно разложение на много шагов ползучести, решение до невязки 1e-10).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from saltcav.mechanics.linsolve import ReusedFactorSolver, csr_pattern
from saltcav.mesh.axisym import AxiMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SQRT2 = float(np.sqrt(2.0))
M_VEC = np.array([1.0, 1.0, 1.0, 0.0])  # след в записи Манделя
P_DEV = np.eye(4) - np.outer(M_VEC, M_VEC) / 3.0

# 7-точечная квадратура Дунаванта, точная для полиномов 5-й степени
_A1, _B1 = 0.059715871789770, 0.470142064105115
_A2, _B2 = 0.797426985353087, 0.101286507323456
QP = np.array(
    [
        (1 / 3, 1 / 3),
        (_B1, _B1), (_A1, _B1), (_B1, _A1),
        (_B2, _B2), (_A2, _B2), (_B2, _A2),
    ]
)  # fmt: skip
QW = 0.5 * np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)

# 3-точечный Гаусс на ребре s ∈ [−1, 1]
EP = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
EW = np.array([5 / 9, 8 / 9, 5 / 9])


def shape_p2(xi: FloatArray, eta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Функции формы P2 (..., 6) и производные по (ξ, η) (..., 6, 2)."""
    l1 = 1.0 - xi - eta
    n = np.stack(
        [l1 * (2 * l1 - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * l1 * xi, 4 * xi * eta,
         4 * eta * l1],
        axis=-1,
    )  # fmt: skip
    dxi = np.stack([-(4 * l1 - 1), 4 * xi - 1, 0 * xi, 4 * (l1 - xi), 4 * eta, -4 * eta], axis=-1)
    deta = np.stack([-(4 * l1 - 1), 0 * xi, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (l1 - eta)], axis=-1)
    return n, np.stack([dxi, deta], axis=-1)


def shape_line3(s: FloatArray) -> FloatArray:
    """Функции формы квадратичного ребра: концы 0, 1, середина 2."""
    return np.stack([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s**2], axis=-1)


def elastic_matrix(e_modulus: float, nu: float) -> FloatArray:
    """Матрица упругости 4×4 в записи Манделя."""
    k = e_modulus / (3 * (1 - 2 * nu))
    g = e_modulus / (2 * (1 + nu))
    return np.asarray(k * np.outer(M_VEC, M_VEC) + 2 * g * P_DEV)


@dataclass
class AxiFEM:
    """Геометрия точек интегрирования, сборка и граничные условия для одной сетки."""

    mesh: AxiMesh
    bbar: bool = True
    solver: str = "superlu"  # в 2D SuperLU быстрее: у pypardiso накладные расходы на каждое решение

    def __post_init__(self) -> None:
        m = self.mesh
        xy = m.nodes[m.tri]  # (M, 6, 2)
        n, dn = shape_p2(QP[:, 0], QP[:, 1])  # (q, 6), (q, 6, 2)
        jac = np.einsum("eia,qib->eqab", xy, dn)  # (M, q, 2, 2): d(r,y)/d(ξ,η)
        det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
        if np.any(det <= 0):
            raise ValueError("вырожденный или вывернутый элемент сетки")
        inv = np.empty_like(jac)
        inv[..., 0, 0] = jac[..., 1, 1] / det
        inv[..., 1, 1] = jac[..., 0, 0] / det
        inv[..., 0, 1] = -jac[..., 0, 1] / det
        inv[..., 1, 0] = -jac[..., 1, 0] / det
        dndx = np.einsum("qia,eqab->eqib", dn, inv)  # (M, q, 6, 2): dN/dr, dN/dy
        self.n = n
        self.rq = np.einsum("qi,ei->eq", n, xy[..., 0])
        self.yq = np.einsum("qi,ei->eq", n, xy[..., 1])
        self.wq = QW[None, :] * det * self.rq  # вес: w·detJ·r
        ne, nq = det.shape
        b = np.zeros((ne, nq, 4, 12))
        b[..., 0, 0::2] = dndx[..., 0]
        b[..., 1, 1::2] = dndx[..., 1]
        b[..., 2, 0::2] = n[None, :, :] / self.rq[..., None]
        b[..., 3, 0::2] = dndx[..., 1] / SQRT2
        b[..., 3, 1::2] = dndx[..., 0] / SQRT2
        if self.bbar:
            # B̄: объёмная деформация — среднее по элементу (P2/P0). Ползучесть соли
            # несжимаема, и при больших шагах касательная почти несжимаема; без этого
            # P2 «запирается» и сходимость камеры занижена в разы.
            vol = b[..., 0, :] + b[..., 1, :] + b[..., 2, :]  # (M, q, 12)
            mean = np.einsum("eq,eqk->ek", self.wq, vol) / self.wq.sum(axis=1)[:, None]
            b += (M_VEC[None, None, :, None] / 3.0) * (mean[:, None, :] - vol)[:, :, None, :]
        self.b = b
        dofs = np.empty((ne, 12), dtype=np.int64)
        dofs[:, 0::2] = 2 * m.tri
        dofs[:, 1::2] = 2 * m.tri + 1
        self.dofs = dofs
        self.ndof = 2 * m.n_nodes
        self._inv, self._indices, self._indptr = csr_pattern(dofs, self.ndof)
        self._nnz = int(self._indices.size)
        self.bt = np.ascontiguousarray(b.transpose(0, 1, 3, 2))  # Bᵀ (M, q, 12, 4)
        self.btw = self.bt * self.wq[:, :, None, None]
        self.fixed = np.zeros(self.ndof, dtype=bool)
        self.linear = ReusedFactorSolver(self.solver)

    # --- граничные условия -----------------------------------------------------

    def fix(self, tag: str, component: int) -> None:
        """Закрепить компоненту (0 — u_r, 1 — u_z) на узлах границы ``tag``."""
        nodes = self.mesh.boundary_nodes(tag)
        self.fixed[2 * nodes + component] = True

    def fix_nodes(self, nodes: IntArray, component: int) -> None:
        self.fixed[2 * np.asarray(nodes) + component] = True

    # --- сборка -----------------------------------------------------------------

    def stiffness(self, d: FloatArray) -> sp.csr_matrix:
        """K = Σ Bᵀ D B·w; ``d`` — (4, 4) или (M, q, 4, 4)."""
        db = np.matmul(d, self.b)  # (M, q, 4, 12)
        ke = np.matmul(self.btw, db).sum(axis=1)  # (M, 12, 12)
        data = np.bincount(self._inv, ke.ravel(), minlength=self._nnz)
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.ndof, self.ndof))

    def internal(self, sig: FloatArray) -> FloatArray:
        """∫ Bᵀ σ: вектор узловых сил по напряжениям в точках (M, q, 4)."""
        fe = np.matmul(self.btw, sig[..., None])[..., 0].sum(axis=1)  # (M, 12)
        f = np.bincount(self.dofs.ravel(), fe.ravel(), minlength=self.ndof)
        return np.asarray(f, dtype=np.float64)

    def strain(self, u: FloatArray) -> FloatArray:
        """Деформации в точках (M, q, 4) по узловым перемещениям."""
        return np.asarray(np.matmul(self.b, u[self.dofs][:, None, :, None])[..., 0])

    def pressure_load(
        self, tag: str, q: Callable[[FloatArray, FloatArray], FloatArray]
    ) -> FloatArray:
        """Давление q(r, y) на границе ``tag`` (q > 0 давит на массив): f = −∫ q n N r ds."""
        e = self.mesh.edges.get(tag)
        f = np.zeros(self.ndof)
        if e is None or e.size == 0:
            return f
        xy = self.mesh.nodes[e]  # (E, 3, 2)
        n = shape_line3(EP)  # (3, 3)
        dn = np.stack([EP - 0.5, EP + 0.5, -2 * EP], axis=-1)  # dN/ds (3, 3)
        pts = np.einsum("pi,eia->epa", n, xy)
        tang = np.einsum("pi,eia->epa", dn, xy)  # dx/ds
        # обход области — против часовой стрелки, внешняя нормаль справа: (t_y, −t_x)
        normal_ds = np.stack([tang[..., 1], -tang[..., 0]], axis=-1)  # n·|dx/ds|
        qv = q(pts[..., 0], pts[..., 1])
        load = -(qv * EW[None, :] * pts[..., 0])[..., None] * normal_ds  # (E, p, 2)
        fe = np.einsum("pi,epa->eia", n, load)  # (E, 3, 2)
        np.add.at(f, 2 * e, fe[..., 0])
        np.add.at(f, 2 * e + 1, fe[..., 1])
        return f

    def body_load(self, fb: Callable[[FloatArray, FloatArray], FloatArray]) -> FloatArray:
        """∫ N·b r dA; ``fb(r, y)`` → (..., 2)."""
        b = fb(self.rq, self.yq)  # (M, q, 2)
        fe = np.einsum("qi,eqa,eq->eia", self.n, b, self.wq)
        f = np.zeros(self.ndof)
        np.add.at(f, 2 * self.mesh.tri, fe[..., 0])
        np.add.at(f, 2 * self.mesh.tri + 1, fe[..., 1])
        return f

    # --- решение ------------------------------------------------------------------

    def solve(self, k: sp.csr_matrix, f: FloatArray) -> FloatArray:
        """K u = f при нулевых перемещениях в закреплённых степенях свободы."""
        free = np.flatnonzero(~self.fixed)
        u = np.zeros(self.ndof)
        kff = sp.csr_matrix(k[free][:, free])
        u[free] = self.linear.solve(kff, f[free])
        return u

    # --- величины на границе ------------------------------------------------------

    def cavity_volume(self, u: FloatArray | None = None, tag: str = "cavity") -> float:
        """Объём полостей, м³ (по деформированной границе, если задано u).

        Для замкнутого контура по Паппу–Гюльдену V = π ∮ r² dy; вклад оси нулевой.
        Контур полости обходится по часовой стрелке (область — против), отсюда знак.
        """
        e = self.mesh.edges.get(tag)
        if e is None or e.size == 0:
            return 0.0
        xy = self.mesh.nodes[e].copy()
        if u is not None:
            xy = xy + u.reshape(-1, 2)[e]
        n = shape_line3(EP)
        dn = np.stack([EP - 0.5, EP + 0.5, -2 * EP], axis=-1)
        r = np.einsum("pi,ei->ep", n, xy[..., 0])
        dy = np.einsum("pi,ei->ep", dn, xy[..., 1])
        return float(-np.pi * np.sum(r**2 * dy * EW[None, :]))
