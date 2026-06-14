"""Robot manifest adapters — register ``robots/<id>/robot.yaml`` factories.

The eval layer needs a :class:`openral_core.RobotDescription` to perform
the rSkill compatibility check inside :class:`openral_sim.SimRunner`.
This module wires every ``robot.yaml`` shipped under the top-level
``robots/`` directory to the :data:`openral_sim.ROBOTS` registry.

Search path
-----------
A robot id ``<id>`` is resolved by looking, in order, at:

1. ``$OPENRAL_ROBOTS_DIR/<id>/robot.yaml`` (if the env var is set).
2. ``<repo_root>/robots/<id>/robot.yaml`` — the in-tree, ``robots/`` layout.

The repo root is discovered by walking up from this file until a directory
containing ``robots/`` and ``pyproject.toml`` is found.

Adding a new robot is a one-step change: drop a new
``robots/<id>/robot.yaml`` (and a sibling ``README.md``) into the tree.
The directory scan below picks it up at import time — no Python edit
required.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError

from openral_sim.registry import ROBOTS


def _find_repo_root() -> Path:
    """Walk up from this file looking for a ``robots/`` sibling of ``pyproject.toml``.

    Returns:
        The absolute path to the OpenRAL repository root.

    Raises:
        ROSConfigError: If no suitable parent directory is found.
    """
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "robots").is_dir():
            return ancestor
    raise ROSConfigError(
        f"could not locate OpenRAL repo root from {here} — set $OPENRAL_ROBOTS_DIR to override"
    )


def _robots_search_dir() -> Path:
    """Return the directory to scan for ``<id>/robot.yaml`` entries.

    Honours ``$OPENRAL_ROBOTS_DIR`` for tests / out-of-tree manifests;
    otherwise falls back to ``<repo_root>/robots``.
    """
    override = os.environ.get("OPENRAL_ROBOTS_DIR")
    if override:
        return Path(override)
    return _find_repo_root() / "robots"


def _discover_robot_ids() -> list[str]:
    """List every ``<id>`` such that ``<search_dir>/<id>/robot.yaml`` exists.

    The returned list is sorted to make the registration order deterministic
    across machines (helps when reasoning about ``ROBOTS.names()`` in tests).
    """
    search_dir = _robots_search_dir()
    if not search_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in search_dir.iterdir()
        if entry.is_dir() and (entry / "robot.yaml").is_file()
    )


def _resolve_manifest(robot_id: str) -> Path:
    """Return the absolute path to ``<robot_id>/robot.yaml`` on the search path."""
    override = os.environ.get("OPENRAL_ROBOTS_DIR")
    if override:
        candidate = Path(override) / robot_id / "robot.yaml"
        if candidate.is_file():
            return candidate
    in_tree = _find_repo_root() / "robots" / robot_id / "robot.yaml"
    if in_tree.is_file():
        return in_tree
    raise ROSConfigError(
        f"robot manifest for {robot_id!r} not found (looked at $OPENRAL_ROBOTS_DIR and {in_tree})"
    )


def _make_factory(robot_id: str) -> Callable[[], RobotDescription]:
    """Build a ``() -> RobotDescription`` factory for the given robot id.

    The factory loads the YAML lazily on first call and caches the result on
    a closure-level attribute so repeated calls do not re-parse.
    """
    cache: dict[str, RobotDescription] = {}

    def factory() -> RobotDescription:
        if robot_id not in cache:
            cache[robot_id] = RobotDescription.from_yaml(str(_resolve_manifest(robot_id)))
        return cache[robot_id]

    factory.__name__ = f"_load_{robot_id}_manifest"
    factory.__qualname__ = factory.__name__
    return factory


for _id in _discover_robot_ids():
    ROBOTS.register(_id)(_make_factory(_id))
