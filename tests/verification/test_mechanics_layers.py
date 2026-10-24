"""Прослои в геомеханике: отдельный материал, согласованная сетка, предел текучести."""

import numpy as np
import pytest

pytest.importorskip("gmsh")

from saltcav.core.config import (
    LayerConfig,
    LayerMaterialConfig,
    MechanicsConfig,
    PressureStageConfig,
)
from saltcav.mechanics.creep import von_mises
from saltcav.mechanics.run import layer_model, realization_layers, run_from_shape
from saltcav.mesh.axisym import Band, cavern_mesh

Z = np.linspace(1000.0, 1040.0, 41)
R = 12.0 * np.sin(np.pi * (Z - 1000.0) / 40.0) ** 0.5 + 0.1
R[20:22] = 7.0  # карниз на 1020–1021 м
LAYER = [LayerConfig(kind="anhydrite", top_m=1020.0, thickness_m=1.0)]
BASE = {
    "stages": [PressureStageConfig(duration_years=2.0)],
    "output_every_years": 1.0,
    "transient": None,
    "r_factor": 6.0,
    "height_factor": 2.5,
}


def test_mesh_conforms_to_layers() -> None:
    mesh, _ = cavern_mesh(Z, R, h_wall=1.0, bands=[Band(1020.0, 1021.0, 1), Band(1030, 1030.5, 2)])
    assert mesh.material is not None and set(np.unique(mesh.material)) == {0, 1, 2}
    depth = -mesh.nodes[mesh.tri[:, :3]].mean(axis=1)[:, 1]
    in1 = (depth > 1020.0) & (depth < 1021.0)
    assert np.all(mesh.material[in1] == 1) and np.all(depth[mesh.material == 1] > 1020.0)
    assert np.all(depth[mesh.material == 1] < 1021.0)
    # у кромки карниза сетка мельче, чем у гладкой стенки
    p = mesh.nodes[mesh.tri[:, :3]]
    h = np.max(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2), axis=1)
    cen = p.mean(axis=1)
    tip = np.hypot(cen[:, 0] - 7.0, cen[:, 1] + 1020.5) < 1.0
    assert np.median(h[tip]) < 0.5
    for tag in ("axis", "cavity", "outer", "top", "bottom"):
        assert mesh.edges[tag].size, tag


def test_salt_like_layer_changes_nothing() -> None:
    """Прослой со свойствами соли — тот же результат, что и без прослоя."""
    salt_like = LayerMaterialConfig(
        kind="anhydrite", youngs_modulus_gpa=25.0, poisson=0.25, creep_factor=1.0,
        compressive_strength_mpa=1e6, tensile_strength_mpa=1.0,
    )  # fmt: skip
    mc = MechanicsConfig(**BASE, layer_materials=[salt_like])
    plain = run_from_shape(mc, Z, R).summary
    with_l = run_from_shape(mc, Z, R, layers=LAYER).summary
    np.testing.assert_allclose(with_l.convergence_pct[1:], plain.convergence_pct[1:], rtol=0.02)
    # критерии соли считаются только по соли (кромка «прослоя» исключена) — сравниваем
    # смещения стенки, они от разметки материалов не зависят
    np.testing.assert_allclose(with_l.max_wall_disp_m, plain.max_wall_disp_m, rtol=0.03)


def test_stiff_layer_and_yield_limit() -> None:
    mc = MechanicsConfig(**BASE)
    plain = run_from_shape(mc, Z, R)
    run = run_from_shape(mc, Z, R, layers=LAYER)
    assert run.material is not None and run.material_names == ["соль", "ангидрит"]
    s = run.summary
    # жёсткий несползающий прослой держит камеру: конвергенция меньше
    assert s.convergence_pct[-1] < 0.95 * plain.summary.convergence_pct[-1]
    assert s.layer_tension_mpa is not None and s.layer_tension_ratio is not None
    assert s.layer_tension_ratio.shape == s.years.shape
    # напряжения в прослое не выше предела текучести (80 МПа) — с запасом на перескок
    lay = run.material == 1
    assert float(np.max(von_mises(run.stress[-1][lay]))) < 1.1 * 80e6
    # без прослоев — прежний результат (совместимость)
    assert plain.material is None and plain.summary.layer_tension_mpa is None


def test_switch_off_and_realization_layers(tmp_path: object) -> None:
    mc = MechanicsConfig(**BASE, interlayers=False)
    assert layer_model(mc, LAYER).bands == []
    halite = [LayerConfig(kind="halite", top_m=1020.0, thickness_m=1.0)]
    assert layer_model(MechanicsConfig(**BASE), halite).bands == []
    from saltcav.core.config import load_config

    cfg = load_config("configs/example_ensemble.yaml")
    names = ["k0", "layer0_top", "layer0_thickness", "layer1_top", "layer1_thickness"]
    got = realization_layers(cfg, np.array([1.0, 1041.2, 0.8, 1074.0, 0.4]), names)
    assert (got[0].top_m, got[0].thickness_m, got[1].top_m) == (1041.2, 0.8, 1074.0)


def test_save_load_roundtrip(tmp_path: object) -> None:
    from pathlib import Path

    from saltcav.mechanics.run import MechanicsRun

    mc = MechanicsConfig(**{**BASE, "stages": [PressureStageConfig(duration_years=1.0)]})
    run = run_from_shape(mc, Z, R, layers=LAYER)
    p = Path(str(tmp_path)) / "m.npz"
    run.save(p)
    back = MechanicsRun.load(p)
    assert back.material_names == run.material_names
    assert np.array_equal(back.material, run.material)
    assert back.summary.layer_tension_ratio is not None
    from saltcav.viz.mechanics_plot import material_edges, plot_mechanics

    assert len(material_edges(back)) > 10
    plot_mechanics(back, Path(str(tmp_path)) / "m.png")


def test_3d_mesh_has_layers() -> None:
    from saltcav.mesh.cavern3d import cavern_mesh_3d

    mesh, _ = cavern_mesh_3d(Z, R, h_wall=4.0, r_factor=5.0, height_factor=2.0,
                             bands=[Band(1020.0, 1021.0, 1)])  # fmt: skip
    assert mesh.material is not None and set(np.unique(mesh.material)) == {0, 1}
    depth = -mesh.nodes[mesh.tets[:, :4]].mean(axis=1)[:, 2]
    assert np.all((depth[mesh.material == 1] > 1020.0) & (depth[mesh.material == 1] < 1021.0))
    for tag in ("cavity", "outer", "top", "bottom"):
        assert mesh.faces[tag].size, tag


@pytest.mark.slow
def test_3d_matches_axisym_with_layer() -> None:
    """Симметричная камера с карнизом: 3D и осесимметрия согласуются и с прослоем."""
    ax = run_from_shape(MechanicsConfig(**BASE, h_wall_m=1.0), Z, R, layers=LAYER).summary
    d3 = run_from_shape(
        MechanicsConfig(**BASE, model="3d", h_wall_3d_m=2.5), Z, R, layers=LAYER
    ).summary
    np.testing.assert_allclose(d3.convergence_pct[1:], ax.convergence_pct[1:], rtol=0.04)
    assert d3.fos_wall_p5 is not None and ax.fos_wall_p5 is not None
    np.testing.assert_allclose(d3.fos_wall_p5, ax.fos_wall_p5, rtol=0.1)
