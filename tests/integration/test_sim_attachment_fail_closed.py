# SPDX-License-Identifier: Apache-2.0
"""The sim bridge never certifies "nothing carried" for a twin whose grasp it cannot measure.

Issue #752: when ``SimAttachmentEvidenceTracker`` refused a robot, ``SimSensorBridge`` kept
heartbeating an empty, fresh ``/openral/attachment_state``, so the safety kernel read
"nothing carried" while the twin could close its gripper on a payload. Now:

* a twin that drives a gripper (``twin_drives_gripper``) and whose tracker refuses has the
  stream withdrawn, so the kernel's attached check fails closed;
* a twin with no gripper keeps the empty heartbeat, which is the truth there;
* the SO-101, one of the robots the issue listed, arms.

Real components (CLAUDE.md §1.11): a real ``rclpy`` ``LifecycleNode``, the real
``SimSensorBridge`` and ``SimAttachedHAL`` over the native-MuJoCo ``so101_box`` scene, the
real ``robots/so101_follower/robot.yaml``. The two refusal cases edit that manifest in
memory (no fixed jaw; no gripper at all), the way a robot the tracker cannot model looks.
Skips without ROS 2 + colcon-built ``openral_msgs``, ``openral_sim`` or ``mujoco``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — this test constructs a real rclpy LifecycleNode.",
)


@pytest.fixture
def setup_bridge() -> Iterator[Callable[[Any], Any]]:
    """Build the bridge for a (possibly edited) SO-101 description and arm its attachment state."""
    pytest.importorskip("openral_sim")
    pytest.importorskip("mujoco")

    import rclpy
    from openral_hal.sim_attached import SimAttachedHAL
    from openral_hal.sim_bringup import build_sim_env_from_yaml
    from openral_hal.sim_sensor_bridge import SimSensorBridge
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    made: list[tuple[Any, Any, Any]] = []

    def build(description: Any) -> Any:
        env, seed = build_sim_env_from_yaml(
            "scenes/sim/so101_tube_insertion.yaml", robot_id_fallback="so101_follower"
        )
        hal = SimAttachedHAL(env, _so101(), env_reset_seed=seed, env_action_dim=6)
        hal.connect()
        node = LifecycleNode(f"test_sim_attachment_fail_closed_{len(made)}")
        bridge = SimSensorBridge(
            node,
            hal,
            description,
            viewer_enabled=False,
            attachment_heartbeat=True,
            attachment_kernel_checked=True,
        )
        made.append((bridge, node, hal))
        bridge._setup_attachment_state()
        return bridge

    try:
        yield build
    finally:
        for bridge, node, _hal in made:
            bridge.teardown()
            node.destroy_node()
        rclpy.shutdown()


def _so101() -> Any:
    from openral_core import RobotDescription

    return RobotDescription.from_yaml("robots/so101_follower/robot.yaml")


def test_a_twin_the_tracker_models_arms(setup_bridge: Callable[[Any], Any]) -> None:
    bridge = setup_bridge(_so101())
    assert bridge._attachment_tracker is not None
    assert bridge._attachment_pub is not None and bridge._attachment_timer is not None


def test_a_gripper_twin_the_tracker_refuses_withdraws_the_stream(
    setup_bridge: Callable[[Any], Any],
) -> None:
    desc = _so101()
    joints = [j.model_copy(update={"fixed_jaw_link": None}) for j in desc.joints]
    bridge = setup_bridge(desc.model_copy(update={"joints": joints}))
    assert bridge._attachment_tracker is None
    assert bridge._attachment_pub is None and bridge._attachment_timer is None


def test_a_twin_without_a_gripper_keeps_the_empty_heartbeat(
    setup_bridge: Callable[[Any], Any],
) -> None:
    desc = _so101()
    joints = [
        j.model_copy(update={"role": "unknown", "fixed_jaw_link": None})
        if j.role == "gripper"
        else j
        for j in desc.joints
    ]
    assert desc.sim is not None
    sim = desc.sim.model_copy(update={"grippers": []})
    bridge = setup_bridge(desc.model_copy(update={"joints": joints, "sim": sim}))
    assert bridge._attachment_tracker is None
    assert bridge._attachment_pub is not None and bridge._attachment_timer is not None
