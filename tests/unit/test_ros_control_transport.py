# SPDX-License-Identifier: Apache-2.0
"""A real ros2_control HAL must actually reach its controllers, and know when it hasn't.

Two defects motivated this file, both found on live OpenArm hardware (2026-09-08) and both
invisible to the suite as it stood:

1. `RosControlHAL` was only ever given a transport by HIL test helpers. In production
   `build_hal` runs before any ROS node exists, so `publish_fn` stayed `_default_publish` — a
   `log.debug` no-op — and every real ros2_control robot (OpenArm, UR5e, UR10e, Franka,
   Sawyer) silently dropped commands and read zeros. Only the serial arms (SO-100/SO-101,
   whose HAL owns its own bus) worked, which is why nothing failed.

2. `_last_state_time` was written only in `connect()` and never on message arrival, so
   `read_state()` raised `ROSPerceptionStale` for any call more than `staleness_limit_s`
   after connect — against a flawless 750 Hz `/joint_states`. The existing UR tests *rewind*
   that field to fake staleness, which passes precisely because nothing else touches it, so
   the bug was load-bearing for its own coverage.

The staleness check is a safety control on the actuation path (CLAUDE.md §1.1): it is what
stops a policy acting on a dead robot's last known pose. These tests pin both directions —
fresh data must not trip it, and genuinely stale data must.
"""

from __future__ import annotations

import importlib.util
import time

import pytest
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale
from openral_core.schemas import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_hal.ros_control import RosControlHAL

requires_rclpy = pytest.mark.skipif(
    importlib.util.find_spec("rclpy") is None
    or importlib.util.find_spec("trajectory_msgs") is None,
    reason="rclpy / trajectory_msgs not importable — source a ROS 2 install",
)


def _description(n_joints: int = 2) -> RobotDescription:
    return RobotDescription(
        name="bench_arm",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=f"j{i}",
                joint_type=JointType.REVOLUTE,
                parent_link="base" if i == 0 else f"link{i}",
                child_link=f"link{i + 1}",
            )
            for i in range(n_joints)
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
    )


def _hal(**kw: object) -> RosControlHAL:
    return RosControlHAL(_description(), controller_name="joint_trajectory_controller", **kw)  # type: ignore[arg-type]  # reason: kwargs are the documented optional transport knobs


# ── The generalisation surface ────────────────────────────────────────────────


def test_a_plain_hal_reports_one_command_topic_and_its_manifest_joint_names() -> None:
    """What a transport is built from. A robot that overrides neither still works."""
    hal = _hal()
    assert hal.command_topics() == ["/joint_trajectory_controller/joint_trajectory"]
    assert hal.ros2_control_joint_names() == ["j0", "j1"]
    assert hal.joint_state_topic == "/joint_states"


def test_openarm_overrides_still_win_over_the_new_base_defaults() -> None:
    """The base gained these names after OpenArm already defined them.

    If the base ever shadowed the override, the bimanual arm would be commanded on one topic
    instead of four and fifteen of its sixteen joints would never be addressed.
    """
    from openral_hal.openarm_real import OpenArmRealHAL

    hal = OpenArmRealHAL(require_can_links=False)
    assert len(hal.command_topics()) == 4
    assert hal.ros2_control_joint_names()[0] == "openarm_left_joint1"


def test_send_action_carries_joint_names_so_the_transport_needs_no_mapping() -> None:
    """A transport that kept its own copy of the mapping could drift from the HAL's."""
    sent: list[tuple[str, dict[str, object]]] = []
    hal = _hal()
    hal.attach_transport(lambda t, m: sent.append((t, m)), lambda: {}, None)
    hal.connect()
    hal.send_action(
        Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.1, 0.2]])
    )
    assert sent[0][1]["joint_names"] == ["j0", "j1"]


# ── The watchdog ──────────────────────────────────────────────────────────────


def test_fresh_data_never_goes_stale_however_long_the_hal_has_been_connected() -> None:
    """The actual bug: age was measured from connect(), so this raised after 0.5 s."""
    now = time.monotonic
    hal = _hal(staleness_limit_s=0.5)
    hal.attach_transport(lambda t, m: None, lambda: {}, now)  # always "just arrived"
    hal.connect()
    hal._last_state_time -= 60.0  # a long-lived connection, as on a real rig
    state = hal.read_state()  # must not raise
    assert len(state.position) == 2


def test_a_transport_that_stops_delivering_trips_the_watchdog() -> None:
    """The other direction: the check must still fail closed on a dead robot."""
    frozen = time.monotonic() - 5.0
    hal = _hal(staleness_limit_s=0.5)
    hal.attach_transport(lambda t, m: None, lambda: {}, lambda: frozen)
    hal.connect()
    with pytest.raises(ROSPerceptionStale):
        hal.read_state()


def test_a_transport_that_has_never_received_anything_is_stale_not_fresh() -> None:
    """`last_arrival` is 0.0 before the first message.

    Treated as an epoch timestamp that would be ~2^31 s old it happens to raise anyway, but
    only by accident of the clock; assert the intent so a future refactor cannot turn "no data
    has ever arrived" into "fresh".
    """
    hal = _hal(staleness_limit_s=0.5)
    hal.attach_transport(lambda t, m: None, lambda: {}, lambda: 0.0)
    hal.connect()
    with pytest.raises(ROSPerceptionStale):
        hal.read_state()


def test_without_a_transport_the_pre_existing_connect_clock_still_applies() -> None:
    """Unchanged for SimTransport-injected unit tests, which have no arrival clock."""
    hal = _hal(staleness_limit_s=0.5)
    hal.connect()
    hal._last_state_time -= 1.0
    with pytest.raises(ROSPerceptionStale):
        hal.read_state()


# ── The transport itself (real rclpy, no mocks) ───────────────────────────────


@requires_rclpy
def test_transport_refuses_a_hal_it_could_not_drive() -> None:
    """Failing here is far cheaper than a silent no-op on hardware."""
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_refuse", context=ctx)
    try:
        with pytest.raises(ROSConfigError):
            RosControlTransport(node, command_topics=[], joint_names=["j0"])
        with pytest.raises(ROSConfigError):
            RosControlTransport(node, command_topics=["/c/joint_trajectory"], joint_names=[])
    finally:
        node.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_transport_merges_state_by_name_and_stamps_arrival() -> None:
    """Controllers publish independently and in no fixed order.

    The bimanual case is the sharp one: each message carries only one limb's joints, so
    index-based reading would scramble the arm the moment two controllers interleave.
    """
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as RosJointState

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_merge", context=ctx)
    try:
        tr = RosControlTransport(
            node, command_topics=["/c/joint_trajectory"], joint_names=["j0", "j1", "j2"]
        )
        assert tr.last_arrival() == 0.0
        assert tr.missing_joints() == ["j0", "j1", "j2"]

        # Two controllers, reversed order, each owning a different joint.
        msg = RosJointState()
        msg.name = ["j2", "j0"]
        msg.position = [0.3, 0.1]
        tr._on_joint_state(msg)

        state = tr.state()
        assert state["position"] == [0.1, 0.0, 0.3]  # projected into HAL order, j1 unseen
        assert state["name"] == ["j0", "j1", "j2"]
        assert tr.missing_joints() == ["j1"]
        assert tr.last_arrival() > 0.0
    finally:
        node.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_transport_publishes_the_chunks_last_step_to_the_named_controller() -> None:
    """End-to-end over real DDS: HAL action in, JointTrajectory out."""
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_pub", context=ctx)
    sink = Node("t_sink", context=ctx)
    received: list[JointTrajectory] = []
    try:
        hal = _hal()
        # Built from the HAL's own report, exactly as the lifecycle node does.
        (topic,) = hal.command_topics()
        tr = RosControlTransport(
            node,
            command_topics=hal.command_topics(),
            joint_names=hal.ros2_control_joint_names(),
        )
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        sink.create_subscription(
            JointTrajectory,
            topic,
            received.append,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        hal.attach_transport(tr.publish, tr.state, tr.last_arrival)
        hal.connect()
        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=2,
                joint_targets=[[0.1, 0.2], [0.3, 0.4]],
            )
        )

        # Bound to this test's own context, not the default one: rclpy.spin_once
        # would reach for a default context that was never initialised.
        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(sink)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            executor.spin_once(timeout_sec=0.05)
        executor.shutdown()

        assert received, "no JointTrajectory reached the controller topic"
        traj = received[0]
        assert list(traj.joint_names) == ["j0", "j1"]
        # Only the final step: a trajectory point is an absolute target the
        # controller interpolates toward, not a step to replay.
        assert [round(p, 3) for p in traj.points[0].positions] == [0.3, 0.4]
    finally:
        node.destroy_node()
        sink.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_transport_rejects_a_topic_the_hal_never_declared() -> None:
    """Contract drift between HAL and transport must be loud, not a dropped command."""
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_drift", context=ctx)
    try:
        tr = RosControlTransport(node, command_topics=["/declared"], joint_names=["j0"])
        with pytest.raises(ROSConfigError):
            tr.publish("/undeclared", {"joint_names": ["j0"], "joint_targets": [[0.0]]})
    finally:
        node.destroy_node()
        ctx.shutdown()


def test_attach_transport_rejects_a_value_where_a_clock_was_expected() -> None:
    """`transport.last_stamp` is a property on some transports.

    Passing it unwrapped hands the HAL a float that never advances. Caught at wire-up
    because the alternative is a `TypeError` from inside `read_state` on a connected robot.
    """
    hal = _hal()
    with pytest.raises(ROSConfigError, match="callable"):
        hal.attach_transport(lambda t, m: None, lambda: {}, 123.4)  # type: ignore[arg-type]  # reason: the mistake under test
