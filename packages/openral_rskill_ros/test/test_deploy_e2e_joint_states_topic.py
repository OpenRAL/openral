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
from typing import Any, cast

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


def _compose(robot_id: str, hal_mode: str, **extra: str) -> tuple[Any, list[Any]]:
    """Compose the real launch graph; return the context and its entities."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

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
    return ctx, list(module.compose_runtime_graph(ctx))


def _runtime_params(robot_id: str, hal_mode: str, **extra: str) -> dict[str, object]:
    """Compose the real launch graph and return the runtime node's evaluated params."""
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    ctx, entities = _compose(robot_id, hal_mode, **extra)
    runtime = next(
        e
        for e in entities
        if isinstance(e, Node) and getattr(e, "_Node__node_executable", None) == "runtime_node"
    )
    merged: dict[str, object] = {}
    for params in evaluate_parameters(ctx, runtime._Node__parameters):
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


def _kernel(robot_id: str, hal_mode: str) -> tuple[dict[str, str], dict[str, object]]:
    """The safety kernel's evaluated remappings and parameters."""
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters

    ctx, entities = _compose(robot_id, hal_mode)
    (kernel,) = [
        e
        for e in entities
        if isinstance(e, Node)
        and getattr(e, "_Node__node_executable", None) == "safety_kernel_node"
    ]
    remaps = {
        perform_substitutions(ctx, list(src)): perform_substitutions(ctx, list(dst))
        for src, dst in (kernel._Node__remappings or [])
    }
    merged: dict[str, object] = {}
    for params in evaluate_parameters(ctx, kernel._Node__parameters):
        if isinstance(params, dict):
            merged.update(params)
    return remaps, merged


def test_real_openarm_kernel_reads_the_manifest_named_republish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kernel's measured state comes from the stream that spells its joint names.

    It folds ``/joint_states`` by ``collision_joint_names`` (the manifest's names). On
    the real OpenArm the broadcaster's global ``/joint_states`` says
    ``openarm_left_joint1`` for ``left_joint1``: no name matched, the state never
    filled, and every slot row was refused ``state_unavailable`` (Orin cell,
    2026-10-16). The HAL's ``~/joint_states`` republishes under the manifest's names.
    """
    from openral_core import RobotDescription

    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "orin")
    remaps, params = _kernel("openarm", "real")
    assert remaps == {"/joint_states": "/openral_hal_openarm/joint_states"}
    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    names = list(cast("tuple[str, ...]", params["collision_joint_names"]))
    assert names == [j.name for j in desc.joints]
    assert "openarm_left_joint1" not in names


def test_sim_and_serial_kernels_keep_the_global_joint_states() -> None:
    """A HAL that publishes ``/joint_states`` itself (sim, serial) needs no remap."""
    assert _kernel("openarm", "sim")[0] == {}
    assert _kernel("so100_follower", "real")[0] == {}


def test_preload_revision_reaches_the_runtime_node() -> None:
    params = _runtime_params(
        "so100_follower",
        "real",
        preload_rskill_id="rskill-smolvla-so101-eraser_place-bf16",
        preload_rskill_revision="v1.2.0",
    )
    assert params["preload_rskill_id"] == "rskill-smolvla-so101-eraser_place-bf16"
    assert params["preload_rskill_revision"] == "v1.2.0"
