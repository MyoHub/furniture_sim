"""Package, asset resolution, and wheel hygiene tests."""

from __future__ import annotations

from pathlib import Path

import furniture_sim
import pytest
from furniture_sim import __version__, get_asset_root, resolve_asset

TOP_LEVEL_MODELS = [
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


def test_version_exposed() -> None:
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    assert furniture_sim.__version__ == __version__


def test_asset_root_exists() -> None:
    root = get_asset_root()
    assert root.exists()
    assert (root / "README.md").exists()


@pytest.mark.parametrize("rel_path", TOP_LEVEL_MODELS)
def test_can_resolve_all_known_xmls(rel_path: str) -> None:
    xml_path = resolve_asset(rel_path)
    assert xml_path.is_file()
    # PathLike input must also work.
    assert resolve_asset(Path(rel_path)) == xml_path


def test_resolve_asset_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_asset("nonexistent_model.xml")


def test_resolve_asset_rejects_absolute_and_parent_traversal() -> None:
    root = get_asset_root()
    with pytest.raises(ValueError, match="relative"):
        resolve_asset(str(root / "simpleTable.xml"))

    with pytest.raises(ValueError, match="relative"):
        resolve_asset("/simpleTable.xml")

    with pytest.raises(ValueError, match="escapes"):
        resolve_asset("../outside.xml")


def test_installed_package_excludes_build_and_test_artifacts() -> None:
    """When installed into site-packages, build/dist/tests metadata must not leak."""
    root = get_asset_root()
    if "site-packages" not in root.parts:
        pytest.skip("Editable/source checkout legitimately contains repo files")
    assert __version__ != "0.0.0+unknown"
    assert not (root / "build").exists()
    assert not (root / "dist").exists()
    assert not (root / "tests").exists()
    assert not (root / ".github").exists()
    assert not (root / "pyproject.toml").exists()
    assert not list(root.glob("*.egg-info"))


@pytest.mark.parametrize("rel_path", TOP_LEVEL_MODELS)
def test_packaged_xmls_compile_in_mujoco(rel_path: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_path = resolve_asset(rel_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nbody >= 2
