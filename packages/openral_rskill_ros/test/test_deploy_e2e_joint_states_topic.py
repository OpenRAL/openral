"""The deploy launch hands the runtime node the HAL-derived JointState topic.

A real ros2_control arm's ``/joint_states`` is the ``joint_state_broadcaster``'s
full-rate stream; the in-process Python nodes read the HAL's rate-limited
``/<hal_node>/joint_states`` instead. This used to be a per-scene opt-in only the
OpenArm bench set; it is now derived for every ros2_control robot, and the
``preload_rskill_revision`` launch arg reaches the runner the same way.

Real ``RobotDescription`` manifests (UR5e ros2_control, SO-100 serial), real
``LaunchContext``, real ``compose_runtime_graph``. No mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_core")
pytest.importorskip("openral_safety")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"


def _runtime_params(robot_id: str, hal_mode: str, **extra: str) -> dict[str, object]:
    """Compose the real launch graph and return the runtime node's evaluated params."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_js", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / robot_id / "robot.yaml")
    cfg["hal_package"] = "openral_hal_node"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = f"openral_hal_{robot_id}"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = hal_mode
    for leg in ("slam", "nav2", "octomap", "object_detector", "dashboard"):
        cfg[f"enable_{leg}"] = "false"
    cfg.update(extra)
    entities = list(module.compose_runtime_graph(ctx))
    runtime = next(
        e
        for e in entities
        if isinstance(e, Node) and getattr(e, "_Node__node_executable", None) == "runtime_node"
    )
    merged: dict[str, object] = {}
    for params in evaluate_parameters(ctx, runtime._Node__parameters):  # type: ignore[attr-defined]
        if isinstance(params, dict):
            merged.update(params)
    return merged


def test_real_ros2_control_arm_gets_the_hal_republish() -> None:
    """UR5e (ros2_control, no scene pin) reads ``/openral_hal_ur5e/joint_states``."""
    params = _runtime_params("ur5e", "real")
    assert params["joint_states_topic"] == "/openral_hal_ur5e/joint_states"


def test_serial_arm_and_sim_keep_the_default() -> None:
    """SO-100 publishes /joint_states itself; a sim HAL too. ``""`` = /joint_states."""
    assert _runtime_params("so100_follower", "real")["joint_states_topic"] == ""
    assert _runtime_params("ur5e", "sim")["joint_states_topic"] == ""


def test_preload_revision_reaches_the_runtime_node() -> None:
    params = _runtime_params(
        "so100_follower",
        "real",
        preload_rskill_id="rskill-smolvla-so101-eraser_place-bf16",
        preload_rskill_revision="v1.2.0",
    )
    assert params["preload_rskill_id"] == "rskill-smolvla-so101-eraser_place-bf16"
    assert params["preload_rskill_revision"] == "v1.2.0"
