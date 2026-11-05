"""The HAL advertises ``/openral/policy_state`` only once its backend produces one (#474).

The reasoner's palette admits a ``use_world_state_policy_state`` skill only while the
topic has a publisher, so the HAL must not advertise it on a backend that never fills
``ProprioFrame.policy_state``. Real ``HALLifecycleNodeBase`` + real ``ProprioSnapshot``
frames on a live rclpy context; skipped without a ROS environment.
"""

from __future__ import annotations

import time

import pytest
from openral_core import JointState

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("std_msgs.msg")

from openral_hal.proprio_snapshot import ProprioFrame, ProprioSnapshot  # noqa: E402

_TOPIC = "/openral/policy_state"


def _frame(policy_state: tuple[float, ...] | None) -> ProprioFrame:
    return ProprioFrame(
        state=JointState(name=["j0"], position=[0.0], velocity=[0.0], stamp_ns=1),
        base_pose=(0.0, 0.0, 0.0),
        base_pose_6dof=None,
        base_twist=(0.0,) * 6,
        policy_state=policy_state,
    )


def test_policy_state_topic_is_advertised_on_the_first_policy_state_frame() -> None:
    from openral_hal.lifecycle import HALLifecycleNodeBase
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Float32MultiArray

    rclpy.init()
    node = None
    peer = None
    try:
        node = HALLifecycleNodeBase("openral_policy_state_lazy_test")
        peer = rclpy.create_node("openral_policy_state_lazy_peer")
        received: list[list[float]] = []
        peer.create_subscription(
            Float32MultiArray, _TOPIC, lambda msg: received.append(list(msg.data)), 10
        )
        # What on_activate sets up; the publisher itself is deferred.
        node._policy_state_qos = QoSProfile(depth=10)
        node._proprio = ProprioSnapshot()

        node._proprio.set(_frame(None))
        node._publish_policy_state()
        assert node.count_publishers(_TOPIC) == 0, (
            "a backend that produces no policy state must not advertise the topic"
        )

        node._proprio.set(_frame((0.5,) * 61))
        node._publish_policy_state()
        assert node.count_publishers(_TOPIC) == 1

        # The first sample can go out before discovery matches the peer; the next
        # new frame must arrive.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            node._proprio.set(_frame((0.5,) * 61))
            node._publish_policy_state()
            rclpy.spin_once(peer, timeout_sec=0.05)
        assert received and len(received[-1]) == 61
    finally:
        if peer is not None:
            peer.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
