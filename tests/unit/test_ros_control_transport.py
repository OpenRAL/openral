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
    ActionSpec,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_hal.ros_control import ControllerKind, RosControlHAL

requires_rclpy = pytest.mark.skipif(
    importlib.util.find_spec("rclpy") is None
    or importlib.util.find_spec("trajectory_msgs") is None,
    reason="rclpy / trajectory_msgs not importable — source a ROS 2 install",
)


def _description(n_joints: int = 2, *, control_freq_hz: float | None = 30.0) -> RobotDescription:
    return RobotDescription(
        name="bench_arm",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        action_spec=(
            None
            if control_freq_hz is None
            else ActionSpec(dim=n_joints, control_freq_hz=control_freq_hz)
        ),
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
        # A real HAL has no staleness default: it reads the manifest's window.
        safety=SafetyEnvelope(joint_state_staleness_limit_s=0.5),
    )


def _hal(control_freq_hz: float | None = 30.0, **kw: object) -> RosControlHAL:
    return RosControlHAL(
        _description(control_freq_hz=control_freq_hz),
        controller_name="joint_trajectory_controller",
        **kw,  # type: ignore[arg-type]  # reason: kwargs are the documented optional transport knobs
    )


# ── The generalisation surface ────────────────────────────────────────────────


def test_a_plain_hal_reports_one_command_topic_and_its_manifest_joint_names() -> None:
    """What a transport is built from. A robot that overrides neither still works."""
    hal = _hal()
    assert hal.command_topics() == ["/joint_trajectory_controller/joint_trajectory"]
    assert hal.ros2_control_joint_names() == ["j0", "j1"]
    assert hal.joint_state_topic == "/joint_states"


def test_a_plain_hal_declares_its_controller_kind_rather_than_leaving_it_implied() -> None:
    """The wire format is part of the contract, not a transport-side assumption.

    A topic name says nothing about the message type its controller accepts, and publishing
    the wrong one is silent: DDS declines to deliver it, so the arm does not move and nothing
    logs an error. Declaring the kind is what lets the transport pick a publisher type instead
    of guessing one.
    """
    assert _hal().command_bindings() == {
        "/joint_trajectory_controller/joint_trajectory": ControllerKind.JOINT_TRAJECTORY
    }


def test_command_topics_is_derived_from_the_bindings_so_the_two_cannot_drift() -> None:
    """One source of truth. A HAL overrides `command_bindings`; the topic list follows."""

    class TwoControllers(RosControlHAL):
        def command_bindings(self) -> dict[str, ControllerKind]:
            return {
                "/arm/joint_trajectory": ControllerKind.JOINT_TRAJECTORY,
                "/gripper/commands": ControllerKind.FORWARD_COMMAND,
            }

    hal = TwoControllers(_description(), controller_name="unused")
    assert hal.command_topics() == ["/arm/joint_trajectory", "/gripper/commands"]


def test_every_controller_kind_maps_to_a_message_type() -> None:
    """Adding an enum member without teaching the transport would be a silent wrong type.

    `_message_type` is the single place that turns a declared kind into a publisher, so an
    unmapped member must fail here rather than reach a controller.
    """
    from openral_hal.ros_control_transport import _message_type

    for kind in ControllerKind:
        try:
            assert _message_type(kind) is not None
        except ImportError:  # reason: per-kind, and only the ROS-less lane
            pytest.skip("ROS 2 message packages not importable — source a ROS 2 install")


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


def test_send_action_carries_the_deadline_the_control_rate_implies() -> None:
    """`time_from_start` is one control period per step, not the transport's 100 ms (#303)."""
    sent: list[tuple[str, dict[str, object]]] = []
    hal = _hal(control_freq_hz=50.0)
    hal.attach_transport(lambda t, m: sent.append((t, m)), lambda: {}, None)
    hal.connect()
    hal.send_action(
        Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.1, 0.2]])
    )
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=4,
            joint_targets=[[0.1, 0.2]] * 4,
        )
    )
    assert sent[0][1]["time_from_start_s"] == pytest.approx(0.02)
    assert sent[1][1]["time_from_start_s"] == pytest.approx(0.08)


@pytest.mark.parametrize("rate", [None, 0.0, -30.0])
def test_a_manifest_without_a_positive_rate_cannot_build_the_hal(rate: float | None) -> None:
    """No deadline fallback exists, so the HAL refuses at construction and names the field."""
    with pytest.raises(ROSConfigError, match=r"action_spec\.control_freq_hz"):
        _hal(control_freq_hz=rate)


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


def test_read_state_is_stamped_with_the_samples_arrival_not_the_read() -> None:
    """The stamp says how old the sample is, so a stamp-pairing consumer can refuse it.

    The robot self-filter pairs each cloud with the joint state nearest its capture stamp
    and trusts a 0.1 s skew bound; a read-time stamp hid a sample up to the staleness
    limit old.
    """
    arrived = time.monotonic() - 0.3
    hal = _hal(staleness_limit_s=0.5)
    hal.attach_transport(lambda t, m: None, lambda: {}, lambda: arrived)
    hal.connect()
    before = time.time_ns()
    state = hal.read_state()
    age_s = (before - state.stamp_ns) / 1e9
    assert 0.29 <= age_s <= 0.35


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
                horizon=3,
                joint_targets=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
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
        # Every row is a point, one control period apart (30 Hz manifest), so
        # the controller interpolates through the chunk's shape rather than
        # driving a straight line to its last row (issue #303).
        assert [[round(p, 3) for p in pt.positions] for pt in traj.points] == [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
        deadlines = [
            pt.time_from_start.sec + pt.time_from_start.nanosec / 1e9 for pt in traj.points
        ]
        assert deadlines == pytest.approx([1 / 30, 2 / 30, 3 / 30], abs=1e-6)
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


# ── Who gets a transport, and what happens to a payload it can't express ──────


def test_membership_is_structural_not_by_ancestry() -> None:
    """The lifecycle node gates on this Protocol, so its shape decides who gets wired.

    Gating on `isinstance(hal, RosControlHAL)` instead would skip a ros2_control robot that
    reimplements the same fan-out on `HALBase` rather than inheriting it, leaving it with no
    transport and no error to say so.
    """
    from openral_hal.openarm_real import OpenArmRealHAL
    from openral_hal.ros_control_transport import RosControlDrivable

    assert isinstance(_hal(), RosControlDrivable)
    assert isinstance(OpenArmRealHAL(require_can_links=False), RosControlDrivable)

    class OptsInWithoutInheriting:
        """Any HAL that grows the members qualifies, whatever its base class."""

        def command_bindings(self) -> dict[str, ControllerKind]:
            return {"/c/joint_trajectory": ControllerKind.JOINT_TRAJECTORY}

        def ros2_control_joint_names(self) -> list[str]:
            return ["j0"]

        @property
        def joint_state_topic(self) -> str:
            return "/joint_states"

        def attach_transport(self, publish_fn, state_fn, stamp_fn=None) -> None:  # type: ignore[no-untyped-def]  # reason: structural stand-in
            return None

    assert isinstance(OptsInWithoutInheriting(), RosControlDrivable)


@pytest.mark.parametrize(
    "entry_point",
    [
        "openral_hal.ur_real:UR5eRealHAL",
        "openral_hal.ur_real:UR10eRealHAL",
        "openral_hal.franka_panda_real:FrankaPandaRealHAL",
        "openral_hal.sawyer_real:SawyerRealHAL",
    ],
)
def test_every_single_controller_real_adapter_is_drivable(entry_point: str) -> None:
    """The four single-controller real arms must all reach the production transport.

    `FrankaPandaRealHAL` and `SawyerRealHAL` used to *wrap* a `RosControlHAL` and forward
    only the five HAL Protocol methods, so they exposed none of the `RosControlDrivable`
    surface. `_attach_ros_control_transport` skipped them, and a real Panda or Sawyer deploy
    published every command into `_default_publish` — the exact no-op the transport docstring
    says it fixed for "OpenArm, UR5e, UR10e, Franka, Sawyer". Constructed through the same
    `hal.real` entry-point string the manifests carry, so a rename shows up here too.
    """
    from openral_hal.resolver import _import_object
    from openral_hal.ros_control_transport import RosControlDrivable

    hal = _import_object(entry_point)()  # type: ignore[operator]  # reason: entry point resolves to a HAL class
    assert isinstance(hal, RosControlDrivable), entry_point
    assert hal.command_topics() == [f"/{hal.controller_name}/joint_trajectory"]  # type: ignore[attr-defined]  # reason: every listed adapter pins controller_name
    assert hal.ros2_control_joint_names() == [j.name for j in hal.description.joints]


def test_a_hal_missing_the_surface_is_not_drivable() -> None:
    """The serial arms own their own bus and must not be handed a ros2_control transport."""
    from openral_hal.ros_control_transport import RosControlDrivable

    class SerialArm:
        pass

    assert not isinstance(SerialArm(), RosControlDrivable)


@requires_rclpy
def test_an_unexpressible_payload_raises_instead_of_vanishing() -> None:
    """A command shape this transport cannot publish must not be dropped quietly.

    A HAL that sends `{"position": float}` rather than `joint_targets` is asking for a
    controller format this transport does not build. Skipping that at runtime would be a
    silent no-op on the actuation path — the failure this whole transport exists to end —
    so it raises and names the keys instead.
    """
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_unexpressible", context=ctx)
    try:
        tr = RosControlTransport(node, command_topics=["/g/command"], joint_names=["g0"])
        with pytest.raises(ROSConfigError, match="cannot express"):
            tr.publish("/g/command", {"position": 0.5, "stamp_ns": 0})
        # An empty chunk is a different thing: the HAL had nothing to send.
        tr.publish("/g/command", {"joint_targets": [], "joint_names": ["g0"]})
        # A chunk with no deadline is refused rather than driven at some
        # arbitrary speed — the 100 ms constant that used to fill this gap is
        # the defect of issue #303.
        with pytest.raises(ROSConfigError, match="time_from_start_s"):
            tr.publish("/g/command", {"joint_targets": [[0.1]], "joint_names": ["g0"]})
    finally:
        node.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_a_forward_command_topic_publishes_float64multiarray_not_a_trajectory() -> None:
    """The declared kind decides the publisher's type, which is the whole point.

    A `ForwardCommandController` subscribes to `std_msgs/Float64MultiArray`. Publishing a
    `trajectory_msgs/JointTrajectory` at it is not an error anyone sees — the types simply
    never match, so the controller receives nothing and the robot stands still.
    """
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_forward_kind", context=ctx)
    try:
        tr = RosControlTransport(
            node,
            command_topics=["/forward_position_controller/commands"],
            joint_names=["j0", "j1"],
            command_kinds={"/forward_position_controller/commands": ControllerKind.FORWARD_COMMAND},
        )
        pub = tr._pubs["/forward_position_controller/commands"]
        assert pub.msg_type is Float64MultiArray
        # And the same HAL payload still publishes cleanly through the other format.
        tr.publish(
            "/forward_position_controller/commands",
            {"joint_targets": [[0.1, 0.2]], "joint_names": ["j0", "j1"]},
        )
    finally:
        node.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_a_kind_naming_no_declared_topic_is_rejected_at_wire_up() -> None:
    """A kind that governs nothing is a typo, and a typo here picks the wrong message type.

    Silently ignoring it would leave the mistyped topic on the JointTrajectory default, which
    is exactly the silent mismatch the declaration exists to prevent.
    """
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.node import Node

    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_stray_kind", context=ctx)
    try:
        with pytest.raises(ROSConfigError, match="not in command_topics"):
            RosControlTransport(
                node,
                command_topics=["/arm/joint_trajectory"],
                joint_names=["j0"],
                command_kinds={"/arm/joint_trajctory": ControllerKind.FORWARD_COMMAND},
            )
    finally:
        node.destroy_node()
        ctx.shutdown()


@requires_rclpy
def test_a_stalled_controller_manager_cannot_stretch_the_stop_past_its_budget() -> None:
    """Discovery, the switch and its confirmation share one `timeout_s` deadline.

    A `switch_controller` that accepts the request and never answers is the worst case
    for the e-stop path: the old per-phase timeouts let the stop take a full window for
    the switch **and** another for confirmation, so the lifecycle node's FATAL came late.
    A real `switch_controller` service that sleeps past the budget stands in for that
    manager; the stop must still return inside `timeout_s` and report the miss.
    """
    import threading

    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    try:
        from controller_manager_msgs.srv import SwitchController
        from std_srvs.srv import Trigger
    except ImportError:
        pytest.skip("controller_manager_msgs / std_srvs not importable — source ros2_control")

    budget_s = 1.0
    ctx = rclpy.Context()
    ctx.init()
    node = Node("t_budget", context=ctx)
    manager = Node("t_budget_manager", context=ctx)

    def _stall(_request: object, response: object) -> object:
        time.sleep(budget_s * 2)
        return response

    manager.create_service(SwitchController, "/cm/switch_controller", _stall)
    manager.create_service(Trigger, "/stall/stop", _stall)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(manager)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        tr = RosControlTransport(
            node,
            command_topics=["/c/joint_trajectory"],
            joint_names=["j0"],
            controller_names=["c"],
            trigger_services=["/stall/stop"],
            controller_manager="/cm",
        )
        # Let discovery settle first so the budget below is spent on the stall, not on DDS.
        # Both services are probed: an undiscovered `/stall/stop` would fail as "not
        # available" instead of exercising the stalled-call path measured below.
        probe = node.create_client(SwitchController, "/cm/switch_controller")
        assert probe.wait_for_service(timeout_sec=10.0), "stalled manager never discovered"
        stop_probe = node.create_client(Trigger, "/stall/stop")
        assert stop_probe.wait_for_service(timeout_sec=10.0), "stalled stop never discovered"
        t0 = time.monotonic()
        report = tr.deactivate_controllers(["c"], timeout_s=budget_s)
        elapsed = time.monotonic() - t0
        assert not report.ok
        assert "did not answer" in report.detail
        assert elapsed < budget_s * 1.3, f"stop took {elapsed:.2f}s against a {budget_s}s budget"

        t0 = time.monotonic()
        trig = tr.call_trigger("/stall/stop", timeout_s=budget_s)
        elapsed = time.monotonic() - t0
        assert not trig.success
        assert "did not answer" in trig.message
        assert elapsed < budget_s * 1.3, f"trigger took {elapsed:.2f}s against {budget_s}s"
        tr.close()
    finally:
        # The stalled callbacks are still sleeping on the spinner; let them drain before the
        # nodes go away, so teardown never races a callback that holds the service.
        executor.shutdown(timeout_sec=budget_s * 6)
        spinner.join(timeout=budget_s * 6)
        node.destroy_node()
        manager.destroy_node()
        ctx.shutdown()
