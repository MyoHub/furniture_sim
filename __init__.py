"""furniture_sim asset package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path

try:
    __version__ = version("furniture-sim")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "get_asset_root", "resolve_asset"]


def get_asset_root() -> Path:
    """Return absolute path to packaged furniture_sim assets."""
    return Path(__file__).resolve().parent


def resolve_asset(rel_path: str | PathLike[str]) -> Path:
    """Resolve an asset path relative to the furniture_sim package root."""
    rel = Path(rel_path)
    if rel.is_absolute() or rel.anchor:
        raise ValueError(f"Asset path must be relative, got: {rel_path!r}")
    root = get_asset_root()
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Asset path escapes furniture_sim package root: {rel_path!r}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Asset not found in furniture_sim package: {rel_path} ({candidate})"
        )
    return candidate
