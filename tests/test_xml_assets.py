"""MuJoCo XML asset compilation, composability, and geometry regression tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from furniture_sim import get_asset_root, resolve_asset

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

ALL_MODELS = [
    "bin.xml",
    "boxcabinet.xml",
    "counters.xml",
    "cup.xml",
    "dishwasher.xml",
    "hingecabinet.xml",
    "kettle.xml",
    "lightButtons.xml",
    "microwave.xml",
    "oven.xml",
    "simpleTable.xml",
    "slidecabinet.xml",
    "studyTable.xml",
    "tableware.xml",
    "ventionTable.xml",
    "kitchen/kitchen1.xml",
    "kitchen/kitchen2.xml",
    "kitchen/kitchen3.xml",
]

ASSET_FAMILIES = [
    (
        "backwall",
        "backwall/backwall_asset.xml",
        ["backwall/backwall_body.xml", "backwall/sidewall_body.xml"],
    ),
    ("bin", "bin/bin_asset.xml", ["bin/bin_body.xml"]),
    (
        "boxcabinet",
        "boxcabinet/boxcabinet_assets.xml",
        ["boxcabinet/boxcabinet_body.xml"],
    ),
    (
        "counters",
        "counters/counters_asset.xml",
        ["counters/sink_counter_body.xml", "counters/box_counter_body.xml"],
    ),
    (
        "cup",
        "cup/cup_asset.xml",
        ["cup/cup_body0.xml", "cup/cup_body1.xml"],
    ),
    (
        "dishwasher",
        "dishwasher/dishwasher_asset.xml",
        ["dishwasher/dishwasher_body.xml"],
    ),
    (
        "hingecabinet",
        "hingecabinet/hingecabinet_asset.xml",
        ["hingecabinet/hingecabinet_body.xml"],
    ),
    ("kettle", "kettle/kettle_asset.xml", ["kettle/kettle_body0.xml"]),
    (
        "lightButtons",
        "lightButtons/lightButtons_assets.xml",
        ["lightButtons/lights_body.xml", "lightButtons/buttons_body.xml"],
    ),
    ("microwave", "microwave/microwave_asset.xml", ["microwave/microwave_body0.xml"]),
    ("oven", "oven/oven_asset.xml", ["oven/oven_body.xml"]),
    (
        "simpleTable",
        "simpleTable/simpleTable_asset.xml",
        ["simpleTable/simpleWoodTable_body.xml"],
    ),
    (
        "slidecabinet",
        "slidecabinet/slidecabinet_asset.xml",
        ["slidecabinet/slidecabinet_body.xml"],
    ),
    (
        "studyTable",
        "studyTable/studyTable_asset.xml",
        ["studyTable/studyTable_body.xml"],
    ),
    (
        "tableware",
        "tableware/tableware_asset.xml",
        [
            "tableware/bowl_body.xml",
            "tableware/plate_body.xml",
            "tableware/spoon_body.xml",
        ],
    ),
    (
        "ventionTable",
        "ventionTable/ventionTable_asset.xml",
        ["ventionTable/ventionTable_body.xml"],
    ),
]


@pytest.mark.parametrize("rel_path", ALL_MODELS)
def test_model_loads_and_steps(rel_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    xml_path = resolve_asset(rel_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 25)
    assert not np.isnan(data.qpos).any()
    assert not np.isnan(data.qvel).any()

    # Also verify loading from XML string (exercises compiler meshdir/texturedir).
    monkeypatch.chdir(xml_path.parent)
    model_from_str = mujoco.MjModel.from_xml_string(
        xml_path.read_text(encoding="utf-8")
    )
    assert model_from_str.ngeom == model.ngeom


def test_all_furniture_families_compose_in_single_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All 14 furniture families must coexist without duplicate names or zero-mass errors."""
    root = get_asset_root()
    monkeypatch.chdir(root)

    includes = "\n".join(
        f'    <include file="{asset}"/>' for _, asset, _ in ASSET_FAMILIES
    )
    bodies = []
    for idx, (name, _, body_files) in enumerate(ASSET_FAMILIES):
        inner = "\n".join(f'            <include file="{bf}"/>' for bf in body_files)
        bodies.append(
            f'        <body name="slot_{name}" pos="{idx * 3.0} 0 0">\n{inner}\n        </body>'
        )
    scene_xml = (
        f"<mujoco>\n{includes}\n    <worldbody>\n"
        + "\n".join(bodies)
        + "\n    </worldbody>\n</mujoco>"
    )

    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nbody > len(ASSET_FAMILIES)


def test_dynamic_bin_has_inertia_under_group4_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = get_asset_root()
    monkeypatch.chdir(root)
    scene_xml = """<mujoco>
        <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
        <include file="bin/bin_asset.xml"/>
        <worldbody>
            <body name="free_bin" pos="0 0 0.5">
                <freejoint/>
                <include file="bin/bin_body.xml"/>
            </body>
        </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "free_bin")
    assert model.body_mass[bid] > 0.0


@pytest.mark.parametrize(("name", "asset_file", "body_files"), ASSET_FAMILIES)
def test_asset_includes_declare_radian_angles(
    name: str,
    asset_file: str,
    body_files: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Including an asset+body into a default (degree) scene must preserve radian angles."""
    monkeypatch.chdir(get_asset_root())
    inner = "\n".join(f'<include file="{bf}"/>' for bf in body_files)
    xml_default = f'<mujoco><include file="{asset_file}"/><worldbody><body>{inner}</body></worldbody></mujoco>'
    xml_radian = (
        f'<mujoco><compiler angle="radian"/><include file="{asset_file}"/>'
        f"<worldbody><body>{inner}</body></worldbody></mujoco>"
    )
    m_def = mujoco.MjModel.from_xml_string(xml_default)
    m_rad = mujoco.MjModel.from_xml_string(xml_radian)
    np.testing.assert_allclose(
        m_def.geom_quat, m_rad.geom_quat, atol=1e-6, err_msg=name
    )
    if m_rad.njnt > 0:
        np.testing.assert_allclose(
            m_def.jnt_range, m_rad.jnt_range, atol=1e-6, err_msg=name
        )


@pytest.mark.parametrize("idx", range(4))
def test_microwave_variants_geometry_and_mass(
    idx: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(get_asset_root())
    xml = f"""<mujoco>
        <include file="microwave/microwave_asset.xml"/>
        <worldbody><body><include file="microwave/microwave_body{idx}.xml"/></body></worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    total_mass = float(model.body_mass.sum())
    assert 2.0 <= total_mass <= 6.0, (
        f"Unexpected mass {total_mass} kg for microwave_body{idx}"
    )

    # Top of visual geoms (group 1) and collision geoms (group 4) should match closely.
    viz_top = (
        max(
            float(data.geom_xpos[i, 2] + model.geom_rbound[i])
            for i in range(model.ngeom)
            if model.geom_group[i] == 1
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        )
        if idx in (2, 3)
        else None
    )
    if viz_top is not None:
        viz_max_z = max(
            float(data.geom_xpos[i, 2] + model.geom_size[i, 2])
            for i in range(model.ngeom)
            if model.geom_group[i] == 1
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        )
        col_max_z = max(
            float(data.geom_xpos[i, 2] + model.geom_size[i, 2])
            for i in range(model.ngeom)
            if model.geom_group[i] == 4
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        )
        assert abs(viz_max_z - col_max_z) < 1e-3


@pytest.mark.parametrize(
    "body_xml",
    [
        "simpleTable/simpleWoodTable_body.xml",
        "simpleTable/simpleMarbleTable_body.xml",
        "simpleTable/simpleGraniteTable_body.xml",
    ],
)
def test_simple_table_collision_matches_visual_top(
    body_xml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(get_asset_root())
    xml = f"""<mujoco>
        <include file="simpleTable/simpleTable_asset.xml"/>
        <worldbody><include file="{body_xml}"/></worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    viz_x = max(
        model.geom_size[i, 0] for i in range(model.ngeom) if model.geom_group[i] == 1
    )
    col_x = max(
        model.geom_size[i, 0] for i in range(model.ngeom) if model.geom_group[i] == 4
    )
    assert abs(float(viz_x) - float(col_x)) < 1e-6


def test_xml_cylinder_sizes_and_material_ranges() -> None:
    root = get_asset_root()
    for xml_file in sorted(root.rglob("*.xml")):
        elem_root = ET.parse(xml_file).getroot()
        rel = xml_file.relative_to(root).as_posix()
        for el in elem_root.iter():
            if (
                el.tag == "geom"
                and el.get("type") == "cylinder"
                and "size" in el.attrib
            ):
                parts = el.attrib["size"].split()
                assert len(parts) == 2, (
                    f"{rel}: cylinder geom has {len(parts)} size values ({el.attrib['size']})"
                )
            if el.tag == "material":
                if "reflectance" in el.attrib:
                    r = float(el.attrib["reflectance"])
                    assert 0.0 <= r <= 1.0, f"{rel}: reflectance={r} out of [0, 1]"
                if "rgba" in el.attrib:
                    vals = [float(v) for v in el.attrib["rgba"].split()]
                    assert all(0.0 <= v <= 1.0 for v in vals), (
                        f"{rel}: rgba={el.attrib['rgba']} out of [0, 1]"
                    )
