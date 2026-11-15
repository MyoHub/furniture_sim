"""Randomly generate and preview randomized MuJoCo kitchen scenes.

Usage:
    python kitchen/random_kitchen.py -m kitchen3.xml
    python kitchen/random_kitchen.py -m kitchen3.xml --headless --episodes 2 --steps 10
"""

from __future__ import annotations

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

DESC = """Script to randomly generate a kitchen scene.
USAGE:
    python random_kitchen.py -m kitchen3.xml
"""

KITCHEN_DIR = Path(__file__).resolve().parent
REPO_DIR = KITCHEN_DIR.parent


class Sim:
    """Lightweight simulation container pairing ``MjModel`` and ``MjData``."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)


def euler2quat(euler: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles (radians) to ``[w, x, y, z]`` quaternions."""
    arr = np.asarray(euler, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3, got shape {arr.shape}")
    out = np.empty(arr.shape[:-1] + (4,), dtype=np.float64)
    for idx in np.ndindex(arr.shape[:-1]):
        mujoco.mju_euler2Quat(out[idx], arr[idx], "xyz")
    return out


def resolve_model_path(model_path: str | os.PathLike[str]) -> Path:
    """Resolve ``model_path`` relative to CWD, ``kitchen/``, or package root."""
    raw = Path(model_path)
    candidates = (
        [raw] if raw.is_absolute() else [raw, KITCHEN_DIR / raw, REPO_DIR / raw]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"File {model_path} does not exist")


def _prepare_in_memory_xml(xml_text: str, model_dir: Path) -> str:
    """Resolve relative ``<include>`` paths so XML text compiles from any CWD."""
    root = ET.fromstring(xml_text)
    for comp in root.iter("compiler"):
        for attr in ("meshdir", "texturedir"):
            if comp.get(attr) in ("", ".", ".."):
                del comp.attrib[attr]
    for inc in root.iter("include"):
        inc_file = inc.get("file")
        if inc_file and not Path(inc_file).is_absolute():
            inc.set("file", (model_dir / inc_file).resolve().as_posix())
    return ET.tostring(root, encoding="unicode")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body {name!r} not found in model")
    return bid


def get_sim(
    model_path: str | os.PathLike[str] | None = None,
    model_xmlstr: str | None = None,
) -> Sim:
    """Build a :class:`Sim` from ``model_path`` or ``model_xmlstr``."""
    if model_path is not None:
        resolved = resolve_model_path(model_path)
        model = mujoco.MjModel.from_xml_path(str(resolved))
    elif model_xmlstr is not None:
        model = mujoco.MjModel.from_xml_string(model_xmlstr)
    else:
        raise TypeError("Both model_path and model_xmlstr can't be None")
    return Sim(model)


def randomize_appliances(
    model_path: str | os.PathLike[str],
    np_random: np.random.RandomState,
) -> str:
    """Return in-memory XML text with randomized microwave and kettle variants."""
    resolved = resolve_model_path(model_path)
    model_xml = resolved.read_text(encoding="utf-8")

    micro_opt = int(np_random.randint(low=0, high=4))
    model_xml = re.sub(
        r"microwave_body\d\.xml",
        f"microwave_body{micro_opt}.xml",
        model_xml,
    )
    kettle_opt = int(np_random.randint(low=0, high=8))
    model_xml = re.sub(
        r"kettle_body\d\.xml",
        f"kettle_body{kettle_opt}.xml",
        model_xml,
    )
    return _prepare_in_memory_xml(model_xml, resolved.parent)


def randomize_textures(
    model_or_xml: str | os.PathLike[str],
    np_random: np.random.RandomState,
) -> Sim:
    """Compile a :class:`Sim` with randomized wood, stone, and metal textures."""
    if isinstance(model_or_xml, os.PathLike) or (
        isinstance(model_or_xml, str) and "<" not in model_or_xml
    ):
        resolved = resolve_model_path(model_or_xml)
        xml_text = _prepare_in_memory_xml(
            resolved.read_text(encoding="utf-8"),
            resolved.parent,
        )
    else:
        xml_text = model_or_xml

    spec = mujoco.MjSpec.from_string(xml_text)
    tex_opt = int(np_random.randint(low=0, high=5))
    for tex in spec.textures:
        if tex.file:
            tex.file = re.sub(
                r"(wood|stone|metal)\d\.png",
                rf"\g<1>{tex_opt}.png",
                tex.file,
            )

    sim = Sim(spec.compile())
    sim.forward()
    return sim


def randomize_layout(sim: Sim, np_random: np.random.RandomState) -> None:
    """Randomize counter and wall/cabinet appliance placements in place."""
    body_offset = {
        "slidecabinet": {"pos": [0.0, 0.0, 0.0], "euler": [0.0, 0.0, 0.0]},
        "hingecabinet": {"pos": [0.0, 0.0, 0.0], "euler": [0.0, 0.0, 0.0]},
        "microwave": {"pos": [0.0, 0.0, -0.2], "euler": [0.0, 0.0, 0.0]},
        "kettle": {"pos": [0.0, 0.0, 0.0], "euler": [0.0, 0.0, 0.0]},
    }

    layout = {
        "sink": {
            "L": [-1.620, 0.0, 0.0],
            "R": [0.0, 0.0, 0.0],
        },
        "island": {
            "L": [-0.020, 0.0, 0.0],
            "R": [1.92, 0.0, 0.0],
        },
    }
    counter_loc = ["L", "R"]

    app_xy = {
        # Front panel
        "FL": [-0.5, 0.28],
        "FR": [0.4, 0.28],
        # Left panel
        "LL": [-1.0, -1.0],
        "LR": [-1.0, -0.25],
        # Right panel
        "RL": [1.0, -0.25],
        "RR": [1.0, -1.0],
    }
    app_z = {
        "T": [2.6],
        "M": [2.2],
        "B": [1.8],
    }
    app_xyz = {
        # Front panel
        "FLT": {
            "pos": app_xy["FL"] + app_z["T"],
            "euler": [0.0, 0.0, 0.0],
            "accept": True,
        },
        "FRT": {
            "pos": app_xy["FR"] + app_z["T"],
            "euler": [0.0, 0.0, 0.0],
            "accept": True,
        },
        "FLM": {
            "pos": app_xy["FL"] + app_z["M"],
            "euler": [0.0, 0.0, 0.0],
            "accept": False,
        },
        "FRM": {
            "pos": app_xy["FR"] + app_z["M"],
            "euler": [0.0, 0.0, 0.0],
            "accept": False,
        },
        "FLB": {
            "pos": app_xy["FL"] + app_z["B"],
            "euler": [0.0, 0.0, 0.0],
            "accept": False,
        },
        "FRB": {
            "pos": app_xy["FR"] + app_z["B"],
            "euler": [0.0, 0.0, 0.0],
            "accept": False,
        },
        # Left panel
        "LLT": {
            "pos": app_xy["LL"] + app_z["T"],
            "euler": [0.0, 0.0, 1.57],
            "accept": True,
        },
        "LRT": {
            "pos": app_xy["LR"] + app_z["T"],
            "euler": [0.0, 0.0, 1.57],
            "accept": True,
        },
        "LLM": {
            "pos": app_xy["LL"] + app_z["M"],
            "euler": [0.0, 0.0, 1.57],
            "accept": True,
        },
        "LRM": {
            "pos": app_xy["LR"] + app_z["M"],
            "euler": [0.0, 0.0, 1.57],
            "accept": False,
        },
        "LLB": {
            "pos": app_xy["LL"] + app_z["B"],
            "euler": [0.0, 0.0, 1.57],
            "accept": True,
        },
        "LRB": {
            "pos": app_xy["LR"] + app_z["B"],
            "euler": [0.0, 0.0, 1.57],
            "accept": True,
        },
        # Right panel
        "RLT": {
            "pos": app_xy["RL"] + app_z["T"],
            "euler": [0.0, 0.0, -1.57],
            "accept": True,
        },
        "RRT": {
            "pos": app_xy["RR"] + app_z["T"],
            "euler": [0.0, 0.0, -1.57],
            "accept": True,
        },
        "RLM": {
            "pos": app_xy["RL"] + app_z["M"],
            "euler": [0.0, 0.0, -1.57],
            "accept": False,
        },
        "RRM": {
            "pos": app_xy["RR"] + app_z["M"],
            "euler": [0.0, 0.0, -1.57],
            "accept": True,
        },
        "RLB": {
            "pos": app_xy["RL"] + app_z["B"],
            "euler": [0.0, 0.0, -1.57],
            "accept": True,
        },
        "RRB": {
            "pos": app_xy["RR"] + app_z["B"],
            "euler": [0.0, 0.0, -1.57],
            "accept": True,
        },
    }
    app_loc = list(app_xyz)

    # Randomize counter layouts
    opt = int(np_random.randint(low=0, high=2))
    island_grid = counter_loc[opt]
    sim.model.body_pos[_body_id(sim.model, "island")] = layout["island"][island_grid]

    sink_grid = counter_loc[1 - opt]
    sim.model.body_pos[_body_id(sim.model, "sink")] = layout["sink"][sink_grid]
    for side in ("L", "R"):
        app_xyz[sink_grid + side + "B"]["accept"] = False

    # Randomize wall/countertop appliances
    for body_name in ("slidecabinet", "hingecabinet", "microwave"):
        while True:
            sel_grid = app_loc[int(np_random.randint(low=0, high=len(app_loc)))]
            if app_xyz[sel_grid]["accept"]:
                break

        bid = _body_id(sim.model, body_name)
        sim.model.body_pos[bid] = np.asarray(app_xyz[sel_grid]["pos"]) + np.asarray(
            body_offset[body_name]["pos"]
        )
        sim.model.body_quat[bid] = euler2quat(
            np.asarray(app_xyz[sel_grid]["euler"])
            + np.asarray(body_offset[body_name]["euler"])
        )
        app_xyz[sel_grid]["accept"] = False
        if sel_grid in ("LRT", "FLT"):
            app_xyz["LRT"]["accept"] = app_xyz["FLT"]["accept"] = False
        if sel_grid in ("FRT", "RLT"):
            app_xyz["FRT"]["accept"] = app_xyz["RLT"]["accept"] = False


def randomize_visuals(sim: Sim, np_random: np.random.RandomState) -> None:
    """Randomize geom RGB tint while keeping opaque alpha."""
    sim.model.geom_rgba[:, :3] = np_random.uniform(
        low=0.4,
        high=0.8,
        size=(sim.model.ngeom, 3),
    )
    sim.model.geom_rgba[:, 3] = 1.0


def randomize_scene(
    model_path: str | os.PathLike[str],
    seed: int = 0,
    *,
    headless: bool = False,
    episodes: int | None = None,
    steps: int = 1000,
    visuals: bool = False,
    step_delay: float = 0.0,
) -> Sim:
    """Generate and optionally display randomized kitchen scenes."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if episodes is not None and episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")

    target_episodes = (1 if headless else 0) if episodes is None else episodes
    np_random = np.random.RandomState(seed)

    sim: Sim | None = None
    episode_idx = 0
    while target_episodes == 0 or episode_idx < target_episodes:
        episode_idx += 1
        randomized_xml = randomize_appliances(model_path, np_random)
        sim = randomize_textures(randomized_xml, np_random)

        if headless:
            for _ in range(steps):
                randomize_layout(sim, np_random)
                if visuals:
                    randomize_visuals(sim, np_random)
                sim.forward()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                for _ in range(steps):
                    if not viewer.is_running():
                        return sim
                    randomize_layout(sim, np_random)
                    if visuals:
                        randomize_visuals(sim, np_random)
                    sim.forward()
                    viewer.sync()
                    if step_delay > 0.0:
                        time.sleep(step_delay)

    if sim is None:
        randomized_xml = randomize_appliances(model_path, np_random)
        sim = randomize_textures(randomized_xml, np_random)
    return sim


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path or filename of kitchen XML to load (e.g. kitchen3.xml).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible randomization.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an interactive MuJoCo viewer window.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of appliance/texture episodes (default: 1 if --headless, else loop until closed).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Layout randomizations per episode.",
    )
    parser.add_argument(
        "--visuals",
        action="store_true",
        help="Also randomize geom RGB colors on each layout step.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional sleep duration (seconds) between interactive viewer updates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    randomize_scene(
        model_path=args.model_path,
        seed=args.seed,
        headless=args.headless,
        episodes=args.episodes,
        steps=args.steps,
        visuals=args.visuals,
        step_delay=args.step_delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
