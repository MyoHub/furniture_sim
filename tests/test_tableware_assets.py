"""Tests for cup and tableware MuJoCo assets."""

from __future__ import annotations

import pytest
from furniture_sim import get_asset_root, resolve_asset

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    ("xml_rel", "body_name", "expected_mass"),
    [
        ("cup.xml", "cuproot0", 0.250),
        ("cup.xml", "cuproot1", 0.260),
        ("tableware.xml", "bowlroot", 0.300),
        ("tableware.xml", "plateroot", 0.350),
        ("tableware.xml", "spoonroot", 0.045),
    ],
)
def test_cup_and_tableware_masses_and_groups(
    xml_rel: str, body_name: str, expected_mass: float
) -> None:
    model = mujoco.MjModel.from_xml_path(str(resolve_asset(xml_rel)))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0
    assert model.body_mass[bid] == pytest.approx(expected_mass, abs=1e-4)

    body_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == bid]
    assert any(int(model.geom_group[i]) == 1 for i in body_geoms)
    assert any(int(model.geom_group[i]) == 4 for i in body_geoms)
    for gid in body_geoms:
        if int(model.geom_group[gid]) == 1:
            assert int(model.geom_contype[gid]) == 0
            assert int(model.geom_conaffinity[gid]) == 0
        elif int(model.geom_group[gid]) == 4:
            assert int(model.geom_contype[gid]) == 1
            assert int(model.geom_conaffinity[gid]) == 1

    body_sites = [i for i in range(model.nsite) if int(model.site_bodyid[i]) == bid]
    assert len(body_sites) >= 2
    assert all(int(model.site_group[s]) == 3 for s in body_sites)


@pytest.mark.parametrize(
    ("asset_file", "body_file", "drop_z"),
    [
        ("cup/cup_asset.xml", "cup/cup_body0.xml", 0.07),
        ("cup/cup_asset.xml", "cup/cup_body1.xml", 0.07),
        ("tableware/tableware_asset.xml", "tableware/bowl_body.xml", 0.05),
        ("tableware/tableware_asset.xml", "tableware/plate_body.xml", 0.03),
    ],
)
def test_hollow_vessels_retain_dropped_sphere(
    asset_file: str,
    body_file: str,
    drop_z: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(get_asset_root())
    scene_xml = f"""<mujoco>
        <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 4"/>
        <include file="{asset_file}"/>
        <worldbody>
            <geom type="plane" size="1 1 0.05" group="4" contype="1" conaffinity="1"/>
            <body name="vessel" pos="0 0 0">
                <include file="{body_file}"/>
            </body>
            <body name="marble" pos="0.008 0 {drop_z}">
                <freejoint/>
                <geom type="sphere" size="0.010" mass="0.015" group="4" contype="1" conaffinity="1"/>
            </body>
        </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 250)
    marble_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "marble")]
    assert float(np.hypot(marble_pos[0], marble_pos[1])) < 0.07
    assert 0.008 < float(marble_pos[2]) < drop_z


def test_cup_and_tableware_settle_on_simple_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(get_asset_root())
    scene_xml = """<mujoco>
        <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 4"/>
        <include file="simpleTable/simpleTable_asset.xml"/>
        <include file="cup/cup_asset.xml"/>
        <include file="tableware/tableware_asset.xml"/>
        <worldbody>
            <geom type="plane" size="2 2 0.05" group="4" contype="1" conaffinity="1"/>
            <include file="simpleTable/simpleWoodTable_body.xml"/>
            <body name="mug0" pos="-0.25 0 0.765"><freejoint/><include file="cup/cup_body0.xml"/></body>
            <body name="mug1" pos="-0.08 0 0.765"><freejoint/><include file="cup/cup_body1.xml"/></body>
            <body name="bowl" pos="0.10 0 0.765"><freejoint/><include file="tableware/bowl_body.xml"/></body>
            <body name="plate" pos="0.32 0 0.765"><freejoint/><include file="tableware/plate_body.xml"/></body>
            <body name="spoon" pos="0.32 0.16 0.765"><freejoint/><include file="tableware/spoon_body.xml"/></body>
        </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 200)
    assert not np.isnan(data.qpos).any()
    for bname in ("mug0", "mug1", "bowl", "plate", "spoon"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        assert float(data.xpos[bid, 2]) > 0.74
