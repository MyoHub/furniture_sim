"""Tests for standardized interaction and placement site affordances."""

from __future__ import annotations

import pytest
from furniture_sim import (
    AFFORDANCE_TYPES,
    AffordanceInfo,
    get_affordances,
    load_affordance_catalog,
    resolve_asset,
)

ALL_MODELS = [
    "bin.xml",
    "boxcabinet.xml",
    "counters.xml",
    "dishwasher.xml",
    "hingecabinet.xml",
    "kettle.xml",
    "lightButtons.xml",
    "microwave.xml",
    "oven.xml",
    "simpleTable.xml",
    "slidecabinet.xml",
    "studyTable.xml",
    "ventionTable.xml",
    "kitchen/kitchen1.xml",
    "kitchen/kitchen2.xml",
    "kitchen/kitchen3.xml",
]


def test_static_affordance_catalog_schema() -> None:
    catalog = load_affordance_catalog()
    assert tuple(catalog["affordance_types"]) == AFFORDANCE_TYPES
    entries = get_affordances()
    assert len(entries) == len(catalog["sites"])
    assert all(isinstance(e, AffordanceInfo) and e.pos is None for e in entries)
    for atype in AFFORDANCE_TYPES:
        filtered = get_affordances(affordance_type=atype)
        assert len(filtered) >= 1
        assert all(e.affordance_type == atype for e in filtered)

    with pytest.raises(ValueError, match="Unknown affordance_type"):
        get_affordances(affordance_type="invalid_type")


@pytest.mark.parametrize("rel_path", ALL_MODELS)
def test_every_model_exposes_valid_affordance_sites(rel_path: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")

    model = mujoco.MjModel.from_xml_path(str(resolve_asset(rel_path)))
    data = mujoco.MjData(model)

    local_affs = get_affordances(model)
    world_affs = get_affordances(model, data)
    assert len(local_affs) >= 1
    assert len(local_affs) == len(world_affs)

    for aff in world_affs:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, aff.site_name)
        assert sid >= 0
        assert aff.pos is not None
        np.testing.assert_allclose(aff.pos, data.site_xpos[sid], atol=1e-9)
        if aff.affordance_type in ("open", "turn", "toggle"):
            assert aff.joint_name is not None
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, aff.joint_name)
            assert jid >= 0


def test_mjspec_attached_prefixes_and_convention_sites_resolve() -> None:
    mujoco = pytest.importorskip("mujoco")
    parent = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    child = mujoco.MjSpec.from_file(str(resolve_asset("boxcabinet.xml")))
    frame = parent.worldbody.add_frame(pos=[0.5, 0.0, 1.0])
    parent.attach(child, prefix="cab0/", frame=frame)
    custom_body = parent.worldbody.add_body(name="mug", pos=[0.0, 0.0, 0.2])
    custom_body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], mass=0.2)
    custom_body.add_site(name="grasp_mug_handle_site", pos=[0.04, 0.0, 0.0])

    model = parent.compile()
    data = mujoco.MjData(model)
    affs = {a.site_name: a for a in get_affordances(model, data)}

    assert "cab0/boxdoor_site" in affs
    assert affs["cab0/boxdoor_site"].joint_name == "cab0/boxdoorhinge"
    assert "cab0/place_zone_boxcabinet_site" in affs
    assert "grasp_mug_handle_site" in affs
    assert affs["grasp_mug_handle_site"].affordance_type == "grasp"
