"""Shared ``deploy_e2e.launch.py`` loader and launch walker for this package's launch tests.

The launch file lives outside the package's importable Python tree (it is
installed to ``share/openral_rskill_ros/launch/`` by ament_python), so a
normal ``from openral_rskill_ros.launch.deploy_e2e import …`` is not available.
Load the source file directly — the same pattern
``test_franka_scene_attach.launch.py`` follows for its launch-side imports.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def import_launch_module(launch_file: Path) -> ModuleType:
    """Load ``launch_file`` (e.g. ``deploy_e2e.launch.py``) as a Python module via importlib."""
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", launch_file)
    assert spec is not None and spec.loader is not None, f"failed to spec {launch_file}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walk_launch(entity: Any, context: Any, started: list[Any]) -> None:
    """Visit ``entity`` as ``LaunchService`` does, recording processes instead of starting them.

    Mirrors ``launch``'s ``visit_all_entities_and_collect_futures``: each entity's
    ``visit(context)`` runs (launch arguments, ``OpaqueFunction`` callbacks, includes,
    groups, conditions), depth-first, in launch order. Every ``ExecuteLocal`` whose
    condition holds is appended to ``started`` and not executed, and a ``TimerAction``'s
    actions count as started (they would be, once it fires), so it is safe to walk a
    hardware launch on any host. An exception raised by any entity — a refusing gate —
    propagates; ``started`` then holds exactly the processes the launch reached first.
    ``os.environ`` (``LaunchContext.environment``) is restored afterwards.

    Example:
        >>> from launch import LaunchContext, LaunchDescription
        >>> from launch.actions import ExecuteProcess
        >>> started: list[Any] = []
        >>> walk_launch(LaunchDescription([ExecuteProcess(cmd=["true"])]), LaunchContext(), started)
        >>> len(started)
        1
    """
    saved_env = dict(os.environ)
    try:
        _walk(entity, context, started)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _walk(entity: Any, context: Any, started: list[Any]) -> None:
    from launch.actions import ExecuteLocal, TimerAction

    deferred = isinstance(entity, ExecuteLocal | TimerAction)
    if deferred and entity.condition is not None and not entity.condition.evaluate(context):
        return
    if isinstance(entity, ExecuteLocal):
        started.append(entity)
        return
    children = entity.actions if isinstance(entity, TimerAction) else entity.visit(context)
    for child in children or ():
        _walk(child, context, started)
