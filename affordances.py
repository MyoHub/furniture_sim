"""Interaction and placement affordance catalog and MuJoCo site query helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mujoco

AFFORDANCE_TYPES: tuple[str, ...] = (
    "grasp",
    "open",
    "turn",
    "toggle",
    "place_zone",
)

_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("place_zone_", "place_zone"),
    ("grasp_", "grasp"),
    ("open_", "open"),
    ("turn_", "turn"),
    ("toggle_", "toggle"),
)


@dataclass(frozen=True)
class AffordanceInfo:
    """Semantic affordance descriptor for a MuJoCo ``<site>``."""

    site_name: str
    affordance_type: str
    object_category: str
    joint_name: str | None = None
    pos: tuple[float, float, float] | None = None


def load_affordance_catalog() -> dict[str, Any]:
    """Load the static site affordance catalog from ``affordances.json``."""
    catalog_path = Path(__file__).resolve().parent / "affordances.json"
    with catalog_path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _split_attached_name(full_name: str) -> tuple[str, str]:
    if "/" in full_name:
        prefix, base = full_name.rsplit("/", 1)
        return f"{prefix}/", base
    return "", full_name


def _infer_prefix_affordance(
    prefix: str, base_name: str
) -> tuple[str, str] | None:
    for site_prefix, aff_type in _PREFIX_RULES:
        if base_name.startswith(site_prefix):
            stem = base_name[len(site_prefix) :].removesuffix("_site")
            if prefix:
                raw_cat = prefix.rstrip("/").rsplit("/", 1)[-1]
                cat = re.sub(r"\d+$", "", raw_cat) or stem or "object"
            else:
                cat = stem or "object"
            return aff_type, cat
    return None


def _resolve_joint_name(
    model: mujoco.MjModel,
    site_id: int,
    prefix: str,
    catalog_joint: str | None,
) -> str | None:
    import mujoco

    if catalog_joint:
        for candidate in (f"{prefix}{catalog_joint}", catalog_joint):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if jid >= 0:
                return candidate
        return catalog_joint

    body_id = int(model.site_bodyid[site_id])
    jnt_num = int(model.body_jntnum[body_id])
    jnt_adr = int(model.body_jntadr[body_id])
    for offset in range(jnt_num):
        jid = jnt_adr + offset
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            jname = model.joint(jid).name
            if jname:
                return jname
    return None


def _site_position(
    model: mujoco.MjModel,
    data: mujoco.MjData | None,
    site_id: int,
) -> tuple[float, float, float]:
    if data is None:
        vec = model.site_pos[site_id]
        return (float(vec[0]), float(vec[1]), float(vec[2]))

    import mujoco
    import numpy as np

    if (
        model.nbody > 1
        and not np.any(data.xpos[1:])
        and not np.any(data.site_xpos)
    ):
        mujoco.mj_kinematics(model, data)

    vec = data.site_xpos[site_id]
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def get_affordances(
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    affordance_type: str | None = None,
) -> list[AffordanceInfo]:
    """Return site affordances from ``affordances.json`` or a compiled ``MjModel``.

    Args:
        model: Optional compiled ``MjModel``. When ``None``, returns static
            entries from ``affordances.json`` with ``pos=None``. When provided,
            resolves catalog sites (including ``MjSpec.attach`` prefixes such as
            ``cab0/boxdoor_site``) and prefix-convention sites (``grasp_*``,
            ``open_*``, ``turn_*``, ``toggle_*``, ``place_zone_*``).
        data: Optional ``MjData``. When provided alongside ``model``, populates
            ``pos`` with world-frame ``site_xpos`` instead of body-local ``site_pos``.
        affordance_type: Optional filter from :data:`AFFORDANCE_TYPES`.
    """
    if affordance_type is not None and affordance_type not in AFFORDANCE_TYPES:
        raise ValueError(
            f"Unknown affordance_type {affordance_type!r}; expected one of {AFFORDANCE_TYPES}"
        )

    catalog = load_affordance_catalog()
    sites_map: dict[str, dict[str, Any]] = catalog.get("sites", {})

    if model is None:
        results: list[AffordanceInfo] = []
        for sname, entry in sites_map.items():
            atype = str(entry["affordance_type"])
            if affordance_type is not None and atype != affordance_type:
                continue
            results.append(
                AffordanceInfo(
                    site_name=sname,
                    affordance_type=atype,
                    object_category=str(entry["object_category"]),
                    joint_name=entry.get("joint_name"),
                    pos=None,
                )
            )
        return results

    results = []
    for site_id in range(model.nsite):
        full_name = model.site(site_id).name
        if not full_name:
            continue

        prefix, base_name = _split_attached_name(full_name)
        entry = sites_map.get(base_name)
        if entry is not None:
            atype = str(entry["affordance_type"])
            obj_cat = str(entry["object_category"])
            jname = _resolve_joint_name(
                model, site_id, prefix, entry.get("joint_name")
            )
        else:
            inferred = _infer_prefix_affordance(prefix, base_name)
            if inferred is None:
                continue
            atype, obj_cat = inferred
            jname = _resolve_joint_name(model, site_id, prefix, None)

        if affordance_type is not None and atype != affordance_type:
            continue

        results.append(
            AffordanceInfo(
                site_name=full_name,
                affordance_type=atype,
                object_category=obj_cat,
                joint_name=jname,
                pos=_site_position(model, data, site_id),
            )
        )

    return results
