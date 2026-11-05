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
    # Joint rates are odom-frame: body vx=0.2 at yaw 0.7 splits across base_x/base_y.
    assert state.velocity == pytest.approx([0.2 * math.cos(0.7), 0.2 * math.sin(0.7), 0.1])
    # Wall-clock stamp (the lifecycle republisher's domain), within a second of now.
    assert abs(state.stamp_ns - time.time_ns()) < 1_000_000_000
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


def _composite(node: Any) -> Any:  # reason: ComposedHAL, imported lazily
    """Base + held lift, composed over the openamr_v2 manifest narrowed to those joints."""
    from openral_hal.composed import ComposedHAL

    robot = RobotDescription.from_yaml(str(_MANIFEST))
    keep = {*(robot.base_joints or []), "lift_joint"}
    description = robot.model_copy(
        update={"joints": [j for j in robot.joints if j.name in keep], "end_effectors": []}
    )
    hal = ComposedHAL(
        description=description,
        parts=[
            {
                "entry": "openral_hal.platform_base:PlatformBaseHAL",
                "joints": list(robot.base_joints or []),
                "params": {
                    "odom_topic": "/test_platform/odom",
                    "cmd_vel_topic": "/test_platform/cmd_vel",
                },
            },
            {
                "entry": "openral_hal.composed:HeldJointsHAL",
                "joints": ["lift_joint"],
                "params": {"positions": {"lift_joint": 0.1}},
            },
        ],
        staleness_limit_s=0.2,
    )
    hal.attach_ros_node(node)
    hal.connect()
    return hal


def test_a_failed_base_stop_is_not_reported_as_a_clean_stop(ros_init: None) -> None:
    """The base's zero-twist publish fails: that error propagates, not ROSEStopRequested.

    Raising the success-shaped ROSEStopRequested over it sent the lifecycle node down its
    "hardware estop completed" path with the base never told to stop.
    """
    node = rclpy.create_node("composed_estop_test")
    try:
        hal = _composite(node)
        base = hal._parts[0]
        node.destroy_publisher(base._pub)  # the platform publisher is gone
        with pytest.raises(Exception) as caught:  # reason: rclpy's own error type
            hal.estop()
        assert not isinstance(caught.value, ROSEStopRequested)
        assert any("PlatformBaseHAL" in note for note in caught.value.__notes__)
    finally:
        node.destroy_node()


def test_a_tick_that_leaves_the_base_alone_does_not_need_fresh_odometry(
    ros_init: None,
) -> None:
    """``last_applied_action`` fills uncommanded joints from the last read, not a new one.

    Re-reading every part on each slot tick tied the arms to base odometry: one EKF
    pause past the staleness window failed every arm tick.
    """
    from openral_core import Action

    node = rclpy.create_node("composed_tick_test")
    try:
        hal = _composite(node)
        odom_pub = node.create_publisher(Odometry, "/test_platform/odom", 10)
        for _ in range(5):
            odom_pub.publish(_odom(2.0, 1.0, 0.5, 0.0, 0.0))
            _spin(node, 0.05)
        hal.read_state()
        time.sleep(0.3)  # odometry is now past the 0.2 s window
        with pytest.raises(ROSPerceptionStale):
            hal.read_state()
        common = {"horizon": 1, "tick_index": 1, "tick_group_size": 2, "runner_session_id": 3}
        hal.send_action(  # a held-lift slot and a zero twist: no base joint target
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=[[0.0, 0.0, 0.0, 0.1]],
                joint_names=["lift_joint"],
                **common,
            )
        )
        hal.send_action(_twist(0.0, 0.0, 0.0).model_copy(update=common))
        applied = hal.last_applied_action
        assert applied is not None and applied.joint_targets is not None
        assert applied.joint_targets[-1] == pytest.approx([2.0, 1.0, 0.5, 0.1])
    finally:
        node.destroy_node()


def test_the_odom_timer_survives_missing_odometry(
    rig: tuple[Any, PlatformBaseHAL, Any, list[Any]],
) -> None:
    """No platform odometry yet: the bridge skips /odom instead of killing the executor.

    ``base_pose`` raises ``ROSPerceptionStale``; out of an rclpy timer callback that ended
    ``spin`` and the HAL node, with the base on its last twist.
    """
    from openral_hal.mobile_base_bridge import MobileBaseBridge

    node, hal, _, _ = rig
    seen: list[Any] = []
    node.create_subscription(Odometry, "/odom", seen.append, 10)
    bridge = MobileBaseBridge(node, hal, _base_description())
    bridge.setup()
    try:
        with pytest.raises(ROSPerceptionStale):
            _ = hal.base_pose
        bridge._publish_odom()  # the timer's body; must not raise
        _spin(node, 0.2)
        assert seen == []
    finally:
        bridge.teardown()


def test_effort_is_unmeasured_never_zero(ros_init: None) -> None:
    """Odometry and a held lift measure no effort: empty, never 0.0 (a torque observer
    reads 0.0 as an unloaded joint), and the composite keeps that absence."""
    node = rclpy.create_node("composed_effort_test")
    try:
        hal = _composite(node)
        odom_pub = node.create_publisher(Odometry, "/test_platform/odom", 10)
        for _ in range(5):
            odom_pub.publish(_odom(0.0, 0.0, 0.0, 0.0, 0.0))
            _spin(node, 0.05)
        assert hal._parts[0].read_state().effort == []
        assert hal.read_state().effort == []
    finally:
        node.destroy_node()


def test_disconnect_releases_the_ros_endpoints(
    rig: tuple[Any, PlatformBaseHAL, Any, list[Any]],
) -> None:
    """The lifecycle node outlives the HAL; a kept subscription would feed a dead one."""
    node, hal, _, _ = rig
    assert node.count_publishers("/test_platform/cmd_vel") == 1
    subs = node.count_subscribers("/test_platform/odom")
    hal.disconnect()
    assert node.count_publishers("/test_platform/cmd_vel") == 0
    assert node.count_subscribers("/test_platform/odom") == subs - 1
    hal.disconnect()  # idempotent
