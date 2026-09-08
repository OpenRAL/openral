# SPDX-License-Identifier: Apache-2.0
"""The OpenArm HIL transport, checked without the cell.

``tests.hil._openarm_ros_transport.OpenArmHILTransport`` is the only thing between
``OpenArmRealHAL`` and a physical arm, so its behaviour shouldn't first be exercised on a
powered robot. Everything here runs on any host with ROS 2: real publishers, a real
subscriber, real messages over DDS — just no controllers on the other end.

The graph is confined to its own DDS domain with LOCALHOST discovery — not ceremony for a
test publishing on ``/*_controller/joint_trajectory``: on 2026-09-05 a sim left on domain 0
with subnet multicast reached a live bimanual OpenArm on another host (#227). A test whose
whole subject is arm command topics must not repeat that.

Load-bearing case: ``test_a_partial_joint_state_is_reported_missing`` — ``state()`` zero-fills
a joint it has never heard from, and ``OpenArmRealHAL`` turns that into a full 16-DoF vector,
so a half-populated ``/joint_states`` reads downstream as a plausible pose with zeros in it.
The motion test refuses to command anything until ``wait_for_every_joint`` passes; this
proves that gate actually detects the condition.
"""

from __future__ import annotations

import time

import pytest

rclpy = pytest.importorskip("rclpy", reason="the HIL transport needs a ROS 2 install")
pytest.importorskip("trajectory_msgs", reason="the HIL transport needs trajectory_msgs")

from openral_hal.openarm_real import OpenArmRealHAL  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, QoSReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import JointState as RosJointState  # noqa: E402
from trajectory_msgs.msg import JointTrajectory  # noqa: E402

from tests.hil._openarm_ros_transport import OpenArmHILTransport  # noqa: E402

_TEST_DOMAIN = "92"
_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)


@pytest.fixture
def isolated_ros(monkeypatch: pytest.MonkeyPatch):
    """A private DDS domain, LOCALHOST-only, torn down after the test."""
    monkeypatch.setenv("ROS_DOMAIN_ID", _TEST_DOMAIN)
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    context = rclpy.Context()
    context.init()
    yield context
    context.try_shutdown()


@pytest.fixture
def rig(isolated_ros):
    """A transport plus the HAL metadata it is built from."""
    node = Node("openarm_transport_test", context=isolated_ros)
    hal = OpenArmRealHAL(require_can_links=False)
    transport = OpenArmHILTransport(
        node,
        hal.ros2_control_joint_names(),
        command_topics=list(hal.command_topics()),
        time_from_start_s=0.8,
    )
    yield transport, hal, node, isolated_ros
    transport.close()
    node.destroy_node()


class TestConstruction:
    def test_it_must_be_built_from_the_ros2_control_namespace(self, rig) -> None:
        """16 names, and they are the URDF ones `/joint_states` is keyed by."""
        _, hal, _, _ = rig
        names = hal.ros2_control_joint_names()
        assert len(names) == 16
        assert names[0] == "openarm_left_joint1"

    def test_a_wrong_joint_count_is_refused(self, isolated_ros) -> None:
        node = Node("openarm_transport_bad", context=isolated_ros)
        hal = OpenArmRealHAL(require_can_links=False)
        try:
            with pytest.raises(ValueError, match="16 joint names"):
                OpenArmHILTransport(
                    node, ["only", "three", "names"], command_topics=list(hal.command_topics())
                )
        finally:
            node.destroy_node()

    def test_a_zero_trajectory_window_is_refused(self, isolated_ros) -> None:
        # time_from_start = 0 asks a JointTrajectoryController to teleport,
        # which is the unbounded-rate command this transport exists to avoid.
        node = Node("openarm_transport_bad2", context=isolated_ros)
        hal = OpenArmRealHAL(require_can_links=False)
        try:
            with pytest.raises(ValueError, match="time_from_start_s"):
                OpenArmHILTransport(
                    node,
                    hal.ros2_control_joint_names(),
                    command_topics=list(hal.command_topics()),
                    time_from_start_s=0.0,
                )
        finally:
            node.destroy_node()


class TestPublish:
    def test_a_hal_command_crosses_dds_as_a_joint_trajectory(self, rig) -> None:
        """The whole point: a HAL publish becomes a real message on the wire."""
        transport, hal, node, _ = rig
        topics = list(hal.command_topics())
        names = hal.ros2_control_joint_names()

        received: list[JointTrajectory] = []
        node.create_subscription(JointTrajectory, topics[0], received.append, _QOS)

        payload = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77]
        deadline = 5.0
        start = time.monotonic()
        while not received and time.monotonic() - start < deadline:
            transport.publish(topics[0], {"joint_names": names[0:7], "joint_targets": [payload]})
            transport.spin_once(timeout_sec=0.05)
        assert received, "no JointTrajectory crossed DDS within 5 s"

        msg = received[-1]
        assert list(msg.joint_names) == names[0:7]
        assert [round(v, 6) for v in msg.points[0].positions] == payload
        # The rate bound: 0.8 s, not the production 0.1 s.
        assert msg.points[0].time_from_start.sec == 0
        assert msg.points[0].time_from_start.nanosec == 800_000_000

    def test_only_the_last_step_of_a_chunk_is_commanded(self, rig) -> None:
        """A multi-step chunk collapses to its final target, as ALOHA's does."""
        transport, hal, node, _ = rig
        topics = list(hal.command_topics())
        names = hal.ros2_control_joint_names()
        received: list[JointTrajectory] = []
        node.create_subscription(JointTrajectory, topics[1], received.append, _QOS)
        start = time.monotonic()
        while not received and time.monotonic() - start < 5.0:
            transport.publish(
                topics[1], {"joint_names": [names[7]], "joint_targets": [[0.1], [0.2], [0.3]]}
            )
            transport.spin_once(timeout_sec=0.05)
        assert received, "no gripper JointTrajectory crossed DDS within 5 s"
        assert [round(v, 6) for v in received[-1].points[0].positions] == [0.3]

    def test_an_unknown_topic_raises_rather_than_dropping_the_command(self, rig) -> None:
        # Silently dropping is the ADR-0102 failure mode all over again.
        transport, _, _, _ = rig
        with pytest.raises(ValueError, match="unknown topic"):
            transport.publish("/not/a/controller", {"joint_names": [], "joint_targets": [[0.0]]})

    def test_a_width_mismatch_raises(self, rig) -> None:
        transport, hal, _, _ = rig
        with pytest.raises(ValueError, match="names 2 joints"):
            transport.publish(
                next(iter(hal.command_topics())),
                {"joint_names": ["a", "b"], "joint_targets": [[1.0]]},
            )


class TestStateGate:
    def test_no_joint_state_means_every_joint_is_missing(self, rig) -> None:
        transport, _, _, _ = rig
        assert len(transport.missing_joints()) == 16
        assert transport.wait_for_every_joint(deadline_s=0.4) is False

    def test_state_zero_fills_an_unheard_joint(self, rig) -> None:
        """Why the gate exists: absence is indistinguishable from 0.0 downstream."""
        transport, _, _, _ = rig
        assert transport.state()["position"] == [0.0] * 16

    def test_a_partial_joint_state_is_reported_missing(self, rig) -> None:
        """Half a `/joint_states` must not read as a complete pose.

        This is the condition that would let the motion test compute
        "measured + delta" against fiction and slew the cell toward home.
        """
        transport, hal, node, _ = rig
        names = hal.ros2_control_joint_names()
        pub = node.create_publisher(RosJointState, "/joint_states", _QOS)

        partial = RosJointState()
        partial.name = names[0:8]  # left side only — the right arm never reports
        partial.position = [0.5] * 8

        start = time.monotonic()
        while transport.missing_joints() and time.monotonic() - start < 5.0:
            pub.publish(partial)
            transport.spin_once(timeout_sec=0.05)
            if len(transport.seen_joints()) >= 8:
                break
        assert len(transport.seen_joints()) == 8, "the partial state should have landed"
        assert transport.missing_joints() == names[8:16]
        assert transport.wait_for_every_joint(deadline_s=0.4) is False, (
            "a partial /joint_states must NOT satisfy the motion gate"
        )
        # …and the zero-fill that makes it dangerous:
        assert transport.state()["position"][8:] == [0.0] * 8

    def test_a_complete_joint_state_opens_the_gate(self, rig) -> None:
        transport, hal, node, _ = rig
        names = hal.ros2_control_joint_names()
        pub = node.create_publisher(RosJointState, "/joint_states", _QOS)

        full = RosJointState()
        full.name = list(names)
        full.position = [float(i) / 100.0 for i in range(16)]

        start = time.monotonic()
        opened = False
        while not opened and time.monotonic() - start < 5.0:
            pub.publish(full)
            opened = transport.wait_for_every_joint(deadline_s=0.2)
        assert opened, "a complete /joint_states should satisfy the gate"
        assert transport.missing_joints() == []
        assert transport.state()["position"][15] == pytest.approx(0.15)
