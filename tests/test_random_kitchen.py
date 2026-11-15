"""Tests for test_sims.py and kitchen/random_kitchen.py."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_repo_module(pkg_mod: str, rel_file: str) -> ModuleType:
    try:
        return importlib.import_module(pkg_mod)
    except ModuleNotFoundError:
        path = REPO_ROOT / rel_file
        spec = importlib.util.spec_from_file_location(pkg_mod, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_mod] = module
        spec.loader.exec_module(module)
        return module


@pytest.mark.parametrize("rel_file", ["test_sims.py", "kitchen/random_kitchen.py"])
def test_scripts_compile_without_syntax_warnings(rel_file: str) -> None:
    source_path = REPO_ROOT / rel_file
    source = source_path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, str(source_path), "exec")


def test_test_sims_loads_all_16_models() -> None:
    pytest.importorskip("mujoco")
    ts = _load_repo_module("furniture_sim.test_sims", "test_sims.py")
    assert "boxcabinet.xml" in ts.model_paths
    assert len(ts.model_paths) == 16

    case = ts.TestSims()
    case.test_sims()
    case.test_get_sim_xmlstr_and_errors()


def test_euler2quat_matches_mujoco_compiler() -> None:
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    rk = _load_repo_module(
        "furniture_sim.kitchen.random_kitchen", "kitchen/random_kitchen.py"
    )

    samples = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.57],
            [0.0, 0.0, -1.57],
            [0.35, -0.45, 1.1],
        ]
    )
    batch_quats = rk.euler2quat(samples)
    assert batch_quats.shape == (4, 4)

    for euler, q_got in zip(samples, batch_quats):
        xml = (
            "<mujoco><compiler angle='radian'/><worldbody>"
            f"<body euler='{euler[0]} {euler[1]} {euler[2]}'>"
            "<geom size='0.1'/></body></worldbody></mujoco>"
        )
        m = mujoco.MjModel.from_xml_string(xml)
        q_ref = m.body_quat[1]
        assert np.allclose(q_got, q_ref, atol=1e-12) or np.allclose(
            q_got, -q_ref, atol=1e-12
        )

    with pytest.raises(ValueError, match="size 3"):
        rk.euler2quat([0.0, 1.0])


@pytest.mark.parametrize(
    "arg",
    [
        "kitchen1.xml",
        "kitchen/kitchen2.xml",
        "kitchen3.xml",
    ],
)
def test_random_kitchen_runs_from_arbitrary_cwd_without_disk_writes(
    arg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    rk = _load_repo_module(
        "furniture_sim.kitchen.random_kitchen", "kitchen/random_kitchen.py"
    )
    kitchen_dir = Path(rk.KITCHEN_DIR)
    before_files = set(kitchen_dir.iterdir())

    monkeypatch.chdir(tmp_path)
    sim = rk.randomize_scene(arg, seed=11, headless=True, episodes=1, steps=3)
    sim.step()

    after_files = set(kitchen_dir.iterdir())
    assert after_files == before_files
    assert list(tmp_path.iterdir()) == []


def test_random_kitchen_determinism_and_isolated_rng() -> None:
    pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    rk = _load_repo_module(
        "furniture_sim.kitchen.random_kitchen", "kitchen/random_kitchen.py"
    )

    np.random.seed(999)
    expected_next = np.random.RandomState(999).uniform(size=4)

    sim_a = rk.randomize_scene(
        "kitchen3.xml",
        seed=42,
        headless=True,
        episodes=2,
        steps=6,
        visuals=True,
    )
    actual_next = np.random.uniform(size=4)
    assert np.allclose(actual_next, expected_next), "Global np.random state was mutated"

    sim_b = rk.randomize_scene(
        "kitchen3.xml",
        seed=42,
        headless=True,
        episodes=2,
        steps=6,
        visuals=True,
    )
    sim_c = rk.randomize_scene(
        "kitchen3.xml",
        seed=43,
        headless=True,
        episodes=2,
        steps=6,
        visuals=True,
    )

    assert np.array_equal(sim_a.model.body_pos, sim_b.model.body_pos)
    assert np.array_equal(sim_a.model.body_quat, sim_b.model.body_quat)
    assert np.array_equal(sim_a.model.geom_rgba, sim_b.model.geom_rgba)
    assert np.allclose(sim_a.data.xpos, sim_b.data.xpos)
    assert not np.array_equal(sim_a.model.geom_rgba, sim_c.model.geom_rgba)


def test_random_kitchen_cli_and_validation() -> None:
    pytest.importorskip("mujoco")
    rk = _load_repo_module(
        "furniture_sim.kitchen.random_kitchen", "kitchen/random_kitchen.py"
    )

    rc = rk.main(
        [
            "-m",
            "kitchen3.xml",
            "-s",
            "5",
            "--headless",
            "--episodes",
            "2",
            "--steps",
            "4",
            "--visuals",
        ]
    )
    assert rc == 0

    with pytest.raises(FileNotFoundError):
        rk.randomize_scene("missing_kitchen.xml", headless=True)

    with pytest.raises(ValueError, match="steps"):
        rk.randomize_scene("kitchen3.xml", headless=True, steps=0)
