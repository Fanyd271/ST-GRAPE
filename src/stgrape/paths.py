"""Portable data discovery and protection of the published reference archive."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Find a source checkout, without relying on a particular directory name."""
    override = os.environ.get("STGRAPE_DATA_DIR")
    if override and (Path(override).expanduser() / "manifest.json").is_file():
        return Path(override).expanduser().resolve().parent
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
                return candidate
    raise FileNotFoundError("Run from the release checkout or set STGRAPE_DATA_DIR.")


def data_root() -> Path:
    """Return the archive location; an environment override supports wheel installs."""
    override = os.environ.get("STGRAPE_DATA_DIR")
    return Path(override).expanduser().resolve() if override else project_root() / "data"


def ensure_output_path(path: Path) -> Path:
    """Prevent experiments and analysis from modifying released reference data."""
    resolved = path.expanduser().resolve()
    try:
        archive = data_root().resolve()
    except FileNotFoundError:
        return resolved
    if resolved == archive or archive in resolved.parents:
        raise ValueError("Outputs must be outside the immutable data/ archive.")
    return resolved
