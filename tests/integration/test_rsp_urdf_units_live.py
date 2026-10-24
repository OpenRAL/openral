# SPDX-License-Identifier: Apache-2.0
"""The real robot_state_publisher poses the SO-101 jaw where the manifest's model does.

The HAL's ``~/urdf_joint_states`` copy (``urdf_named_joint_state`` with
``urdf_joint_map``) is what the deploy launch remaps robot_state_publisher onto. Here the
real ``robot_state_publisher`` binary is fed that copy for a gripper fraction, and its
``gripper <- jaw`` transform on /tf is compared with the manifest's mimic chain — the
SO-101 jaw was posed at the fraction read as radians (deploy trial g19).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``, listed in ``scripts/ros_live_tests.sh``. Locally::

    source /opt/ros/jazzy/setup.bash && just ros2-build
    source install/setup.bash
    just test-ros-live -k rsp_urdf_units
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from openral_core import RobotDescription
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.schemas import FixedAttachment, MimicJoint

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live robot_state_publisher + rclpy — set OPENRAL_TEST_ROS_LIVE=1 and source "
    "install/setup.bash first.",
)

_ROBOT = Path(__file__).resolve().parents[2] / "robots" / "so101_follower"
_TOPIC = "/test_rsp_urdf_units/urdf_joint_states"


def test_robot_state_publisher_poses_the_jaw_by_the_mimic_map() -> None:
    import rclpy
    import tf2_ros
    from openral_core.kinematics import edge_chain as _edge_chain
    from openral_core.kinematics import joint_motion as _joint_motion
    from openral_hal.lifecycle import urdf_named_joint_state
    from openral_hal.resolver import urdf_joint_map
    from sensor_msgs.msg import JointState

    if shutil.which("ros2") is None:
        pytest.skip("no ros2 CLI on PATH")
    robot = RobotDescription.from_yaml(str(_ROBOT / "robot.yaml"))
    urdf = (_ROBOT / "so101_follower.urdf").read_text()
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    fraction = 0.6
    with tempfile.TemporaryDirectory() as tmp:
        params = Path(tmp) / "rsp.yaml"
        params.write_text(
            yaml.safe_dump({"/rsp_units_test": {"ros__parameters": {"robot_description": urdf}}})
        )
        rsp = subprocess.Popen(
            [
                "ros2",
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "-r",
                "__node:=rsp_units_test",
                "--params-file",
                str(params),
                "-r",
                f"joint_states:={_TOPIC}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rclpy.init()
        try:
            node = rclpy.create_node("test_rsp_urdf_units")
            buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(buffer, node)
            pub = node.create_publisher(JointState, _TOPIC, 10)
            names = [j.name for j in robot.joints]
            seen = None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and seen is None:
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = names
                msg.position = [fraction if n == "gripper" else 0.0 for n in names]
                pub.publish(urdf_named_joint_state(msg, m.names, m.scales, m.offsets))
                rclpy.spin_once(node, timeout_sec=0.1)
                try:
                    seen = buffer.lookup_transform("gripper", "jaw", rclpy.time.Time())
                except tf2_ros.TransformException:
                    continue
            assert seen is not None, "robot_state_publisher published no gripper <- jaw"
            t, q = seen.transform.translation, seen.transform.rotation
            tf_pose = homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            chain = _edge_chain(robot, "gripper", "jaw")
            assert chain is not None
            model = np.eye(4)
            for edge, origin in chain:
                assert not isinstance(edge, FixedAttachment), edge  # joint_motion poses joints only
                value = edge.value_at(fraction) if isinstance(edge, MimicJoint) else fraction
                model = model @ origin @ _joint_motion(edge, value)
            np.testing.assert_allclose(tf_pose, model, atol=1e-5)
            node.destroy_node()
        finally:
            rclpy.shutdown()
            rsp.terminate()
            rsp.wait(timeout=10)
