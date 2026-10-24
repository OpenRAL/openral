# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the HAL publishes a ``KIND_GRASP_OUTCOME`` per missed close (issue #353).

Real rclpy, real ``VisionAttachmentBridge`` on the bimanual OpenArm manifest with
``grasp_target_enabled``, real ``openral_msgs`` on the wire, the real grasp trigger
fed real ``Action`` commands and ``JointState`` reads at 20 Hz. A peer node
subscribes ``/openral/failure/hal`` and decodes every ``FailureTrigger`` through the
``FailureEvidence`` union.

Asserts, on the left hand (the right hand stays open throughout and must publish
nothing):

* a close that ends on nothing with no grasp declaration is one ``miss`` at
  ``SEVERITY_WARN`` with no target (an undeclared close wakes nobody);
* the same close under a live ``GraspDeclaration`` naming the left jaw link is one
  ``miss`` at ``SEVERITY_FAIL``, carrying the target, the declaration's skill and trace,
  and the running count;
* the next close that stalls on an object (ATTACH) is one ``attach`` at
  ``SEVERITY_INFO`` with the misses it recovered from;
* a later first-try ATTACH publishes nothing.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_grasp_outcome_live.py
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_vision_attachment_optical_frame_live import _wait_until

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first.",
)

_REPO = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
_LEFT_JAW = "openarm_left_finger_pair"
_TRACE = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
# OpenArm's closure calibration: closed 0.0, rest 0.0086, stall gap 0.08. A jaw that
# reaches its rest offset closed on nothing; one held 0.2 rad short stalled on an object.
_EMPTY_Q = 0.0086
_HELD_Q = 0.2
_OPEN_Q = 0.5
_OPEN_CMD = 0.7
_SETTLE_S = 1.0


def test_a_missed_close_publishes_one_grasp_outcome_and_the_recovery_one_more() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    from openral_core import (
        Action,
        ControlMode,
        FailureEvidence,
        GraspDeclaration,
        GraspOutcomeEvidence,
        JointState,
        RobotDescription,
    )
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import FailureTrigger
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from pydantic import TypeAdapter
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    adapter: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)

    rclpy.init()
    node = Node("test_grasp_outcome_hal")
    peer = Node("test_grasp_outcome_peer")
    outcomes: list[tuple[Any, GraspOutcomeEvidence]] = []
    lock = threading.Lock()

    def on_failure(msg: Any) -> None:
        if int(msg.kind) != FailureTrigger.KIND_GRASP_OUTCOME:
            return
        evidence = adapter.validate_json(msg.evidence_json)
        assert isinstance(evidence, GraspOutcomeEvidence)
        with lock:
            outcomes.append((msg, evidence))

    bus_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=50,
    )
    peer.create_subscription(FailureTrigger, "/openral/failure/hal", on_failure, bus_qos)
    declaration_pub = peer.create_publisher(
        GraspDeclarationMsg,
        "/openral/grasp_declaration",
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )

    bridge = VisionAttachmentBridge(
        node,
        description,
        config=VisionAttachmentConfig(
            camera="head_zed",
            depth_topic="/test_grasp_outcome/depth",
            camera_info_topic="/test_grasp_outcome/camera_info",
            service_name="/openral/perception/segment_in_view_grasp_outcome_itest",
            deadline_s=0.5,
            grasp_target_enabled=True,
        ),
    )
    bridge.setup()
    # The left jaw's command and measured position; the right hand stays open.
    state = {"command": _OPEN_CMD, "q": _OPEN_Q}

    def feed() -> None:
        bridge.observe_command(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_names=["left_gripper", "right_gripper"],
                joint_targets=[[state["command"], _OPEN_CMD]],
            )
        )
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[state["q"], _OPEN_Q],
                effort=[0.0, 0.0],
                stamp_ns=time.monotonic_ns(),
            )
        )

    node.create_timer(0.05, feed)
    executor = MultiThreadedExecutor()
    for each in (node, peer):
        executor.add_node(each)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    def count() -> int:
        with lock:
            return len(outcomes)

    def close_and_open(q: float) -> None:
        """Command a close, hold the jaw at ``q`` until it settles, then open it."""
        state.update(command=0.0, q=q)
        time.sleep(_SETTLE_S)
        state.update(command=_OPEN_CMD, q=_OPEN_Q)
        time.sleep(_SETTLE_S)

    try:
        time.sleep(0.5)  # the trigger sees an open jaw first

        # 1. An undeclared close on nothing: one WARN miss, no target.
        close_and_open(_EMPTY_Q)
        assert _wait_until(lambda: count() == 1, timeout_s=5.0), outcomes
        msg, ev = outcomes[0]
        assert (int(msg.severity), ev.outcome) == (FailureTrigger.SEVERITY_WARN, "miss")
        assert (ev.gripper_joint, ev.jaw_link, ev.hand_link) == (
            "left_gripper",
            _LEFT_JAW,
            "openarm_left_link7",
        )
        assert (ev.consecutive_misses, ev.target_id, ev.region_armed) == (1, "", False)
        assert ev.short_of_command is not None and ev.short_of_command < 0.08
        assert msg.header.frame_id == "hal" and msg.rskill_id == ""

        # 2. The same close under a declaration naming the left hand: a FAIL miss.
        declaration = GraspDeclaration(
            target_id="cell:pick_box",
            contact_links=(_LEFT_JAW,),
            rskill_id="openral/itest-grasp-outcome",
            trace_id=_TRACE,
            timeout_s=60.0,
            stamp_ns=int(node.get_clock().now().nanoseconds),
        )
        wire = GraspDeclarationMsg()
        declaration.fill_idl(wire)
        declaration_pub.publish(wire)
        assert _wait_until(
            lambda: (
                bridge._grasp_target is not None
                and bridge._grasp_target.tracker.declaration is not None
            ),
            timeout_s=5.0,
        ), "the bridge never received the declaration"
        close_and_open(_EMPTY_Q)
        assert _wait_until(lambda: count() == 2, timeout_s=5.0), outcomes
        msg, ev = outcomes[1]
        assert (int(msg.severity), ev.outcome) == (FailureTrigger.SEVERITY_FAIL, "miss")
        assert (ev.consecutive_misses, ev.target_id, ev.region_armed) == (
            2,
            "cell:pick_box",
            False,
        )
        assert (msg.rskill_id, msg.trace_id) == ("openral/itest-grasp-outcome", _TRACE)

        # 3. A close that stalls on an object ends the run: one INFO attach.
        close_and_open(_HELD_Q)
        assert _wait_until(lambda: count() == 3, timeout_s=5.0), outcomes
        msg, ev = outcomes[2]
        assert (int(msg.severity), ev.outcome) == (FailureTrigger.SEVERITY_INFO, "attach")
        assert (ev.consecutive_misses, ev.target_id) == (2, "cell:pick_box")

        # 4. A first-try ATTACH (no miss before it) publishes nothing.
        close_and_open(_HELD_Q)
        time.sleep(1.0)
        assert count() == 3, outcomes[3:]
        assert all(ev.jaw_link == _LEFT_JAW for _, ev in outcomes), "the open hand spoke"
    finally:
        # Stop the executor first: a timer callback waiting on the bridge lock must not
        # run on entities the teardown destroyed.
        executor.shutdown()
        spin.join(timeout=5.0)
        with suppress(Exception):
            bridge.teardown()
        for each in (peer, node):
            each.destroy_node()
        rclpy.shutdown()
