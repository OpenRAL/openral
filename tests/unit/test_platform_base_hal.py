"""``PlatformBaseHAL`` against live ROS topics.

The platform stack is outside OpenRAL, so the test node plays it at the process
boundary: it publishes ``nav_msgs/Odometry`` and listens on the twist topic. The HAL
under test is real, on a real ``rclpy`` node. Skipped when rclpy or the message
packages are not importable.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from openral_core import Action, ControlMode, RobotDescription, ROSConfigError
from openral_core.exceptions import ROSEStopRequested, ROSPerceptionStale

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav_msgs.msg")
pytest.importorskip("geometry_msgs.msg")

from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from openral_hal.platform_base import PlatformBaseHAL  # noqa: E402

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openamr_v2" / "robot.yaml"


def _base_description() -> RobotDescription:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    base = [j for j in robot.joints if j.name in (robot.base_joints or [])]
    return robot.model_copy(update={"joints": base, "end_effectors": []})


@pytest.fixture
def rig(ros_init: None) -> Iterator[tuple[Any, PlatformBaseHAL, Any, list[Any]]]:
    node = rclpy.create_node("platform_base_hal_test")
    hal = PlatformBaseHAL(
        description=_base_description(),
        odom_topic="/test_platform/odom",
        cmd_vel_topic="/test_platform/cmd_vel",
        staleness_limit_s=0.5,
    )
    hal.attach_ros_node(node)
    hal.connect()
    odom_pub = node.create_publisher(Odometry, "/test_platform/odom", 10)
    got: list[Any] = []
    node.create_subscription(Twist, "/test_platform/cmd_vel", got.append, 10)
    yield node, hal, odom_pub, got
    hal.disconnect()
    node.destroy_node()


def _spin(node: object, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def _odom(x: float, y: float, yaw: float, vx: float, wz: float) -> Odometry:
    msg = Odometry()
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    msg.twist.twist.linear.x, msg.twist.twist.angular.z = vx, wz
    return msg


def _twist(vx: float, vy: float, wz: float) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)]
    )


def test_state_is_the_odometry_pose(rig: tuple[Any, PlatformBaseHAL, Any, list[Any]]) -> None:
    node, hal, odom_pub, _ = rig
    with pytest.raises(ROSPerceptionStale):
        hal.read_state()  # nothing received yet
    for _ in range(5):
        odom_pub.publish(_odom(1.5, -0.5, 0.7, 0.2, 0.1))
        _spin(node, 0.05)
    state = hal.read_state()
    assert state.name == ["base_x", "base_y", "base_yaw"]
    assert state.position == pytest.approx([1.5, -0.5, 0.7])
    assert hal.base_twist[0] == pytest.approx(0.2)
    assert hal.base_twist[5] == pytest.approx(0.1)
    _spin(node, 0.6)
    with pytest.raises(ROSPerceptionStale):
        hal.read_state()  # odometry stopped: stale, never the last value forever


def test_body_twist_reaches_the_platform_and_vy_is_refused(
    rig: tuple[Any, PlatformBaseHAL, Any, list[Any]],
) -> None:
    node, hal, _, got = rig
    _spin(node, 0.2)  # discovery
    hal.send_action(_twist(0.25, 0.0, 0.4))
    _spin(node, 0.3)
    assert got, "no Twist reached the platform topic"
    assert (got[-1].linear.x, got[-1].angular.z) == pytest.approx((0.25, 0.4))
    with pytest.raises(ROSConfigError, match="differential"):
        hal.send_action(_twist(0.0, 0.1, 0.0))


def test_estop_commands_zero_twist_and_raises(
    rig: tuple[Any, PlatformBaseHAL, Any, list[Any]],
) -> None:
    node, hal, _, got = rig
    _spin(node, 0.2)
    hal.send_action(_twist(0.3, 0.0, 0.0))
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    _spin(node, 0.3)
    assert (got[-1].linear.x, got[-1].angular.z) == (0.0, 0.0)
