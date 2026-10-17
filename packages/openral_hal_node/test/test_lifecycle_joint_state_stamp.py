"""A sim frame's ``/joint_states`` stamp is its sim time, on every republish (issue #418).

The HAL node publishes ``/clock`` from the same post-step frames it republishes as joint
states. With the sim slower than real time one frame is republished several times; the
old stamp (node now minus the frame's WALL age) walked each republish further into the
past, and the robot self-filter posed the arm at the wrong instant (sf_dr2: stamps
0.07-2.0 s early, ghost voxels of the arm's own surface in the octomap).
"""

from __future__ import annotations

import time
from typing import cast

import pytest

rclpy = pytest.importorskip("rclpy")

from openral_core.schemas import JointState
from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION, GalaxeaA1HAL
from openral_hal.lifecycle import ManifestHALLifecycleNode
from openral_hal.proprio_snapshot import ProprioFrame, ProprioSnapshot
from openral_hal.protocol import HAL


def test_a_republished_sim_frame_keeps_its_sim_time_stamp() -> None:
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState as RosJointState

    sim_ns = 27_283_333_333
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_t_sim_stamp")
        listener = rclpy.create_node("t_sim_stamp_listener")
        received: list[RosJointState] = []
        listener.create_subscription(
            RosJointState, "/t_sim_stamp_joint_states", received.append, 10
        )
        try:
            node._publisher = node.create_publisher(RosJointState, "/t_sim_stamp_joint_states", 10)
            # The node drives /clock from these frames (sim-attached HAL, use_sim_time).
            node._clock_pub = node.create_publisher(Clock, "/t_sim_stamp_clock", 10)
            node._hal = cast(HAL, GalaxeaA1HAL())
            names = [joint.name for joint in GALAXEA_A1_DESCRIPTION.joints]
            snapshot = ProprioSnapshot()
            snapshot.set(
                ProprioFrame(
                    # Read 0.3 s ago on the wall clock: an aged frame of a slow sim.
                    state=JointState(
                        name=names,
                        position=[0.25] * len(names),
                        stamp_ns=time.time_ns() - 300_000_000,
                    ),
                    base_pose=(0.0, 0.0, 0.0),
                    base_pose_6dof=None,
                    base_twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    sim_time_ns=sim_ns,
                )
            )
            node._proprio = snapshot
            node._publish_joint_state()
            time.sleep(0.2)  # a later republish of the same frame: older on the wall clock
            node._publish_joint_state()
            deadline = time.monotonic() + 2.0
            while len(received) < 2 and time.monotonic() < deadline:
                rclpy.spin_once(listener, timeout_sec=0.1)
            assert len(received) == 2
            stamps = [m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec for m in received]
            assert stamps == [sim_ns, sim_ns]
        finally:
            node.destroy_node()
            listener.destroy_node()
    finally:
        rclpy.try_shutdown()
