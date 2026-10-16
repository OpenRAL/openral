"""Make fetched public ROS description packages resolvable by ROS tools.

A vendored URDF names its meshes ``package://<pkg>/...``. A sourced ROS workspace
that builds ``<pkg>`` resolves them; a host without one cannot, unless ``<pkg>``
is a known public package (``PUBLIC_ROS_PACKAGES``), which OpenRAL fetches as a
pinned clone into the openral cache.

The Isaac importer is handed that clone's path directly. ROS tools are not:
``foxglove_bridge`` answers a viewer's ``fetchAsset`` through ``resource_retriever``,
which finds a package only through the ament index on ``AMENT_PREFIX_PATH``. On
Spark it logged ``Package [openarm_description] does not exist`` for every mesh,
and Foxglove's 3D panel drew the TF axes with no robot. ``public_package_overlay``
builds a minimal ament prefix registering those packages, for the caller to
prepend to that process's ``AMENT_PREFIX_PATH``.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path

from openral_hal._openarm_description_assets import ensure_openarm_description

__all__ = ["PUBLIC_ROS_PACKAGES", "fetch_public_package", "public_package_overlay"]

#: Public description packages fetched on demand: ``{package: fetcher}``, each
#: fetcher returning the package root (the directory holding ``package.xml``).
PUBLIC_ROS_PACKAGES: dict[str, Callable[[], Path]] = {
    "openarm_description": ensure_openarm_description,
}

_INDEX = Path("share") / "ament_index" / "resource_index" / "packages"


def fetch_public_package(pkg: str) -> Path | None:
    """The root of the known public ``pkg``, fetched if needed; ``None`` for any other.

    Raises:
        ROSConfigError: the fetch fails.

    Example:
        >>> fetch_public_package("no_such_description") is None
        True
    """
    fetch = PUBLIC_ROS_PACKAGES.get(pkg)
    return None if fetch is None else fetch()


def _on_ament_index(pkg: str) -> bool:
    prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
    return any((Path(p) / _INDEX / pkg).is_file() for p in prefixes)


def public_package_overlay(urdf_xml: str) -> Path | None:
    """An ament prefix registering the public packages ``urdf_xml`` needs, else ``None``.

    Covers every ``package://<pkg>/`` the URDF names that is in
    ``PUBLIC_ROS_PACKAGES`` and on no ``AMENT_PREFIX_PATH`` entry (a sourced
    workspace always wins). The prefix lives at ``$OPENRAL_CACHE_DIR/ament_overlay``:
    ``share/<pkg>`` links to the fetched clone and the ament index marks it, which
    is all ``ament_index`` lookups (and so ``package://`` resolution) read.
    Idempotent. Prepend the result to a process's ``AMENT_PREFIX_PATH``.

    Raises:
        ROSConfigError: a needed package cannot be fetched.
        OSError: the overlay cannot be written.

    Example:
        >>> public_package_overlay('<robot name="r"/>') is None
        True
    """
    pkgs = sorted(set(re.findall(r"package://([^/\"'<>\s]+)/", urdf_xml)))
    missing = [p for p in pkgs if p in PUBLIC_ROS_PACKAGES and not _on_ament_index(p)]
    if not missing:
        return None
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    prefix = base / "ament_overlay"
    (prefix / _INDEX).mkdir(parents=True, exist_ok=True)
    for pkg in missing:
        root = PUBLIC_ROS_PACKAGES[pkg]()
        link = prefix / "share" / pkg
        if not (link.is_symlink() and link.resolve() == root.resolve()):
            # Swap the link in atomically: concurrent launches share this cache.
            tmp = link.with_name(f".{pkg}.{uuid.uuid4().hex}")
            tmp.symlink_to(root, target_is_directory=True)
            tmp.replace(link)
        (prefix / _INDEX / pkg).touch()
    return prefix
