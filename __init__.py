"""furniture_sim asset package."""

from __future__ import annotations

from pathlib import Path


def get_asset_root() -> Path:
    """Return absolute path to packaged furniture_sim assets."""
    return Path(__file__).resolve().parent


def resolve_asset(rel_path: str) -> Path:
    """Resolve an asset path relative to the furniture_sim package root."""
    candidate = get_asset_root() / rel_path
    if not candidate.exists():
        raise FileNotFoundError(
            f"Asset not found in furniture_sim package: {rel_path} ({candidate})"
        )
    return candidate

