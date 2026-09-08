"""Shared ``deploy_e2e.launch.py`` loader for this package's launch-graph tests.

The launch file lives outside the package's importable Python tree (it is
installed to ``share/openral_rskill_ros/launch/`` by ament_python), so a
normal ``from openral_rskill_ros.launch.deploy_e2e import …`` is not available.
Load the source file directly — the same pattern
``test_franka_scene_attach.launch.py`` follows for its launch-side imports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def import_launch_module(launch_file: Path) -> ModuleType:
    """Load ``launch_file`` (e.g. ``deploy_e2e.launch.py``) as a Python module via importlib."""
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", launch_file)
    assert spec is not None and spec.loader is not None, f"failed to spec {launch_file}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
