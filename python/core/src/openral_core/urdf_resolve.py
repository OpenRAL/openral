"""Resolve a ``RobotDescription.urdf_path`` to a concrete filesystem path.

Lifted verbatim (behaviour-preserving) from
``packages/openral_rskill_ros/launch/sim_e2e.launch.py`` so the sim launch and
the offline collision-lowering tool share one resolver (CLAUDE.md §1.13).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

__all__ = ["resolve_urdf_path"]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_urdf_path(value: str, *, repo_root: Path | None = None) -> str | None:
    """Resolve a ``urdf_path`` manifest value to an on-disk file, or ``None``.

    Formats:
      * ``python:<module>:<attribute>`` — import ``<module>``, read ``<attribute>``
        (typically ``URDF_PATH``) from the upstream ``robot_descriptions`` package.
      * absolute path — used verbatim.
      * relative path — resolved against ``repo_root`` (default: the repo root).

    Returns ``None`` (with a stderr warning) when the reference is malformed,
    the import fails, or the resolved file does not exist.

    Example:
        >>> resolve_urdf_path("/nonexistent.urdf") is None
        True
    """
    root = repo_root or _REPO_ROOT
    if value.startswith("python:"):
        try:
            _, module_name, attr_name = value.split(":", 2)
        except ValueError:
            print(
                f"[urdf_resolve] {value!r} malformed — expected python:<module>:<attribute>.",
                file=sys.stderr,
            )
            return None
        try:
            module = importlib.import_module(module_name)
            path = str(getattr(module, attr_name))
        except (ImportError, AttributeError) as exc:
            print(
                f"[urdf_resolve] {value!r} failed to resolve: {type(exc).__name__}: {exc}.",
                file=sys.stderr,
            )
            return None
    elif os.path.isabs(value):
        path = value
    else:
        path = str(root / value)
    if not os.path.isfile(path):
        print(
            f"[urdf_resolve] {value!r} resolved to {path!r} but the file does not exist.",
            file=sys.stderr,
        )
        return None
    return path
