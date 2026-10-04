"""Fetch the upstream ``enactic/openarm_description`` ROS package.

``robots/openarm/openarm.urdf`` references its meshes as
``package://openarm_description/assets/...``. That package is public
(Enactic, Apache-2.0) — every one of the URDF's 22 meshes exists, byte-identical,
in the pinned commit below — so a host without a ROS workspace that builds it
fetches it here instead: a pinned shallow clone under
``$OPENRAL_CACHE_DIR/openarm_description/<sha>/``, the same convention as the
v2 MJCF (``_openarm_v2_assets``). The clone root *is* the package root
(``package.xml`` at the top), which is what a ``package://`` lookup needs.
"""

from __future__ import annotations

import os
from pathlib import Path

from openral_core.exceptions import ROSConfigError

from openral_hal._pinned_clone import fetch_pinned_clone

__all__ = ["ensure_openarm_description"]

# ``main`` at 2026-10-02 ("c103041"). Pinned, not HEAD, so the meshes the Isaac
# import and the Foxglove 3D panel render cannot drift under us.
_PINNED_SHA: str = "c10304158f917579491b02682e02f903b948d74d"
_REPO_URL: str = "https://github.com/enactic/openarm_description.git"


def ensure_openarm_description() -> Path:
    """Return the ``openarm_description`` package root, fetching it if needed.

    Idempotent: a later call reuses the cached clone.

    Raises:
        ROSConfigError: ``git`` is missing, the clone/checkout fails, or the
            tree has no ``package.xml``.
    """
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    repo_dir = base / "openarm_description" / _PINNED_SHA
    if not (repo_dir / "package.xml").is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        fetch_pinned_clone(_REPO_URL, _PINNED_SHA, repo_dir, what="openarm_description")
    if not (repo_dir / "package.xml").is_file():
        raise ROSConfigError(f"openarm_description clone at {repo_dir} has no package.xml.")
    return repo_dir
