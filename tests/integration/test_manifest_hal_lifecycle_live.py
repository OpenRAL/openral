"""Live exercise of the public ``ManifestHALLifecycleNode`` (issue #191, ADR-0032).

The unified, ``robot.yaml``-driven node is the generic lifecycle node the
per-robot HAL packages collapse into. No existing integration test brings it up
on its own (the panda_mobile / openarm tests cover the two *bespoke* nodes), so
this one closes that gap: it constructs ``ManifestHALLifecycleNode`` directly,
points it at a real manifest, and drives
``UNCONFIGURED → INACTIVE → ACTIVE → INACTIVE → UNCONFIGURED``.

It asserts:

* the node configures (builds the HAL through ``openral_hal.build_hal`` and
  ``connect()``s it) and activates;
* a real ``/joint_states`` stream flows at the configured rate, carrying the
  manifest's joints;
* a manifest that declares a ``hal.parameters`` block (ADR-0029) still brings
  the node up cleanly — ``build_hal`` threads the defaults and drops the keys
  the sim HAL does not accept, rather than crashing.

Real ``RobotDescription`` (``robots/franka_panda``), real derived
``MujocoArmHAL``, real rclpy executor + publishers — no mocks (CLAUDE.md §1.11).

Gates: ``ROS_DISTRO`` env + ``rclpy`` + ``mujoco`` (the franka sim HAL derives a
``MujocoArmHAL``). When any is missing the test ``pytest.skip``s cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)  # custom IDL must be colcon-built, not just ROS sourced (skip cleanly otherwise)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — this test requires a sourced ROS 2 install.",
)

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("mujoco")  # franka sim derives a MujocoArmHAL

try:
    from openral_hal.lifecycle import ManifestHALLifecycleNode
except ImportError:  # pragma: no cover - exercised only on no-rclpy hosts
    pytest.skip(
        "rclpy unavailable; ManifestHALLifecycleNode not importable.", allow_module_level=True
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_YAML = REPO_ROOT / "robots" / "franka_panda" / "robot.yaml"
SO100_YAML = REPO_ROOT / "robots" / "so100_follower" / "robot.yaml"

_COUNT_WINDOW_S = 2.0
_MIN_JOINT_STATES = 25  # 50 Hz × 2 s = 100 nominal; loose floor for startup slack


def _run_lifecycle_and_count(robot_yaml: Path, *, rate_hz: float = 50.0) -> tuple[int, list[str]]:
    """Bring the manifest node up on ``robot_yaml`` (sim), count /joint_states.

    Drives the full transition cycle and returns the number of joint-state
    messages observed during ``_COUNT_WINDOW_S`` plus the last joint-name list.
    """
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node: Any = ManifestHALLifecycleNode("test_manifest_hal_live")
    node.set_parameters(
        [
            Parameter("robot_yaml", value=str(robot_yaml)),
            Parameter("hal_mode", value="sim"),
            Parameter("publish_rate_hz", value=rate_hz),
        ]
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    sub_node = Node("test_manifest_hal_live_listener")
    executor.add_node(sub_node)

    state = {"count": 0, "names": []}

    def _cb(msg: JointState) -> None:
        state["count"] += 1
        state["names"] = list(msg.name)

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    sub_node.create_subscription(JointState, "/joint_states", _cb, qos)

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        assert str(node.trigger_configure()).endswith("SUCCESS"), "configure failed"
        assert str(node.trigger_activate()).endswith("SUCCESS"), "activate failed"
        time.sleep(_COUNT_WINDOW_S)
        observed = state["count"]
        names = state["names"]
        assert str(node.trigger_deactivate()).endswith("SUCCESS"), "deactivate failed"
        assert str(node.trigger_cleanup()).endswith("SUCCESS"), "cleanup failed"
    finally:
        executor.shutdown()
        node.destroy_node()
        sub_node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2.0)
    return observed, names


def test_manifest_node_brings_up_real_robot_and_streams_joint_states() -> None:
    """The public node configures + activates a real robot and streams joints."""
    observed, names = _run_lifecycle_and_count(FRANKA_YAML)
    assert observed >= _MIN_JOINT_STATES, f"expected steady /joint_states, got {observed}"
    # franka_panda manifest declares 7 arm joints + the gripper joint.
    assert len(names) >= 7, names
    assert any(n.startswith("panda_joint") for n in names), names


def test_migrated_so100_brings_up_via_manifest_node() -> None:
    """SO-100, migrated off its bespoke node (issue #191 Phase 2), comes up live.

    `openral deploy sim` drives the SO-100 through this generic node now; the bare
    MuJoCo twin (`MujocoArmHAL.from_description`) streams its 6 joints.
    """
    observed, names = _run_lifecycle_and_count(SO100_YAML)
    assert observed >= _MIN_JOINT_STATES, f"expected steady /joint_states, got {observed}"
    assert "gripper" in names and "shoulder_pan" in names, names


def test_manifest_node_tolerates_hal_parameters_block(tmp_path: Path) -> None:
    """A manifest with a ``hal.parameters`` block still brings the node up live.

    ``build_hal`` threads ``hal.parameters.defaults`` into the constructor and
    drops keys the derived sim ``MujocoArmHAL`` does not accept (here a real
    ``robot_ip`` transport default), so the node configures cleanly instead of
    raising — proving the ADR-0029 seam is safe on the live path.
    """
    manifest = yaml.safe_load(FRANKA_YAML.read_text())
    manifest["hal"] = {
        **manifest.get("hal", {}),
        "parameters": {"defaults": {"robot_ip": "192.168.1.10"}},
    }
    patched = tmp_path / "robot.yaml"
    patched.write_text(yaml.safe_dump(manifest, sort_keys=False))

    observed, names = _run_lifecycle_and_count(patched)
    assert observed >= _MIN_JOINT_STATES, f"expected steady /joint_states, got {observed}"
    assert any(n.startswith("panda_joint") for n in names), names
