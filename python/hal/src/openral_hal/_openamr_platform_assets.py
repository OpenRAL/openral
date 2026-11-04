"""Fetch ``openamrobot_description`` from ``openAMRobot/openamr-platform-sw``.

``robots/openamr_v2/openamr_v2.urdf`` names the mobile base's meshes
``package://openamrobot_description/meshes/...``. That package lives at
``ros2/src/openamrobot_description`` in the public platform repo (MIT code; the
CAD-derived meshes are CERN-OHL-P-2.0, a permissive licence). It is fetched as a
pinned clone under ``$OPENRAL_CACHE_DIR/openamr_platform_sw/<sha>/``, the same
convention as ``_openarm_description_assets``, so the 19 MB of base STLs stay out of git.
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_clone import fetch_pinned_clone

__all__ = ["ensure_openamrobot_description"]

# The platform commit openamr-upperbody-sw pins in its
# config/upstream_pins.yaml (upperbody main @ 17cf693, 2026-11-02).
_PINNED_SHA: str = "131e500fc4e5afc9f648cbd7dbf5b416280761fb"
_REPO_URL: str = "https://github.com/openAMRobot/openamr-platform-sw.git"
_PACKAGE_REL: str = "ros2/src/openamrobot_description"


def ensure_openamrobot_description() -> Path:
    """Return the ``openamrobot_description`` package root, fetching it if needed.

    Idempotent: a later call reuses the cached clone.

    Raises:
        ROSConfigError: ``git`` is missing, the clone/checkout fails, or the
            package has no ``package.xml``.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    repo_dir = base / "openamr_platform_sw" / _PINNED_SHA
    pkg = repo_dir / _PACKAGE_REL
    if not (pkg / "package.xml").is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        fetch_pinned_clone(_REPO_URL, _PINNED_SHA, repo_dir, what="openamr-platform-sw")
    if not (pkg / "package.xml").is_file():
        raise ROSConfigError(f"openamr-platform-sw clone at {repo_dir} has no {_PACKAGE_REL}.")
    return pkg
