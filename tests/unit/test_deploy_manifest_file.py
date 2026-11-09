"""The deploy launch writes the robot manifest for the dashboard (ADR-0129).

A deploy with the reasoner off still needs a manifest, so the launch builds one from
the robot, the resolved runtime and the in-tree rSkills, and hands its path to
`openral dashboard --manifest`. The real launch file, loaded by path.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pytest
from openral_core import DeployRuntime, RobotDescription, RobotManifest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAUNCH = _ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> Any:  # reason: a launch file loaded by path is untyped
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_manifest", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_manifest"] = module
    spec.loader.exec_module(module)
    return module


def test_launch_writes_the_manifest_without_the_reasoner(
    launch_module: Any,  # reason: see the fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_DASHBOARD_WRITE_CONTROLS", "1")
    robot = RobotDescription.from_yaml(str(_ROOT / "robots" / "so101_follower" / "robot.yaml"))
    path = pathlib.Path(
        launch_module._write_robot_manifest(
            robot, DeployRuntime(enable_reasoner=False), bridge_on=True, unit_serial="SN-1"
        )
    )
    try:
        manifest = RobotManifest.model_validate_json(path.read_text())
    finally:
        path.unlink()
    assert (manifest.robot_id, manifest.unit_serial) == ("so101_follower", "SN-1")
    ids = [a.id for a in manifest.actions]
    assert ids[:2] == ["estop", "estop_reset"] and "prompt" not in ids  # no reasoner
    assert "skill:OpenRAL/rskill-smolvla-so101-eraser_place-bf16" in ids
    assert "viz3d" in {p.id for p in manifest.panels}
