"""The segmenter and the object tracker reach ACTIVE through the poll-based autostart.

The defect this pins (issue #418, Spark trials r3po_shadow1 and bagtest2_po): both nodes
were driven by the event chain in ``_autostart_lifecycle`` (CONFIGURE on process start,
ACTIVATE on the ``configuring -> inactive`` transition event). On Jazzy a node's
``change_state`` reply can be dropped (``failed to send response (timeout)``); the
launch then waits for it until shutdown, never sends ACTIVATE, and the node sits
INACTIVE for the whole trial: the HAL's grasp fits were refused with "segmenter node is
not active", and the tracker published nothing. ``tools/lifecycle_autostart.py`` reads
``get_state`` instead of trusting the reply.

Real ``LaunchContext`` + real ``compose_runtime_graph``, same harness as
``test_deploy_e2e_estop_sources.py``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_core")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO"),
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
_AUTOSTART_SCRIPT = str(_REPO_ROOT / "tools" / "lifecycle_autostart.py")
_NODES = ("openral_segmenter", "openral_object_tracker")


def _compose() -> list[Any]:
    """An OpenArm Isaac sim deploy (head_zed depth) with the segmenter and the tracker
    (tracking) on."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    spec = importlib.util.spec_from_file_location("deploy_e2e_tracking_autostart", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml")
    cfg["hal_package"] = "openral_hal_node"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = "openral_hal_test"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(ctx)
    cfg["hal_mode"] = "sim"
    cfg["enable_dashboard"] = "false"
    cfg["enable_octomap"] = "true"
    cfg["enable_object_detector"] = "false"
    cfg["enable_vision_attachment"] = "true"
    cfg["object_tracking"] = "tracking"
    # An Isaac scene: its rendered depth shows the robot, so the self-filter runs and
    # renders the tracker's robot masks (a MuJoCo ray-cast hides the robot: no masks, and
    # the launch refuses tracking there).
    cfg["deploy_config"] = str(_REPO_ROOT / "scenes" / "deploy" / "isaac_openarm_warehouse.yaml")
    cfg["grounding_wait_s"] = "8.0"
    return list(module.compose_runtime_graph(ctx))


def _autostart_targets(entities: list[Any]) -> list[str]:
    from launch.actions import ExecuteProcess

    targets = []
    for entity in entities:
        if isinstance(entity, ExecuteProcess):
            parts = [sub.text for group in entity.cmd for sub in group if hasattr(sub, "text")]
            if _AUTOSTART_SCRIPT in parts:
                targets.append(parts[parts.index("--node") + 1])
    return targets


def _event_chain_targets(entities: list[Any]) -> list[str]:
    """Node names an ``OnStateTransition`` (the event-driven ACTIVATE) is registered for."""
    from launch.actions import RegisterEventHandler
    from launch_ros.event_handlers import OnStateTransition

    names = []
    for entity in entities:
        if isinstance(entity, RegisterEventHandler) and isinstance(
            entity.event_handler, OnStateTransition
        ):
            node = vars(entity.event_handler).get("_OnStateTransition__target_lifecycle_node")
            names.append(str(getattr(node, "_Node__node_name", "")))
    return names


def test_segmenter_and_tracker_use_the_poll_based_autostart() -> None:
    entities = _compose()
    targets = _autostart_targets(entities)
    chained = _event_chain_targets(entities)
    for name in _NODES:
        assert f"/{name}" in targets, f"/{name} is not driven by lifecycle_autostart.py"
        assert name not in chained, f"{name} still has the event-driven ACTIVATE chain"
