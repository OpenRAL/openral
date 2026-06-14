"""Unit tests for the HAL Protocol and RosControlHAL adapter.

All tests run without a live ROS 2 installation.  Instead of MagicMock, a
``SimTransport`` is injected as the publish/state transport.  SimTransport
records every published message and applies ``joint_targets`` back to its
internal state, so ``send_action`` → ``read_state`` forms a real closed loop.

Test sequence mirrors the kickoff doc lifecycle:
  connect → read_state → send_action → estop → disconnect
"""

from __future__ import annotations

import time

import pytest
from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
    ROSSafetyViolation,
    SafetyEnvelope,
)
from openral_hal import HAL, RosControlHAL, SimTransport

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_description(
    n_joints: int = 3,
    supported_modes: list[ControlMode] | None = None,
) -> RobotDescription:
    """Return a minimal RobotDescription with *n_joints* revolute joints."""
    joints = [
        JointSpec(
            name=f"j{i}",
            joint_type=JointType.REVOLUTE,
            parent_link="base_link" if i == 0 else f"link_{i - 1}",
            child_link=f"link_{i}",
        )
        for i in range(n_joints)
    ]
    return RobotDescription(
        name="test_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=joints,
        capabilities=RobotCapabilities(
            supported_control_modes=(
                [ControlMode.JOINT_POSITION] if supported_modes is None else supported_modes
            ),
        ),
        safety=SafetyEnvelope(),
    )


def _make_transport(n_joints: int = 3) -> SimTransport:
    """Return a fresh SimTransport for *n_joints* joints."""
    return SimTransport(n_joints=n_joints)


def _make_hal(
    n_joints: int = 3,
    supported_modes: list[ControlMode] | None = None,
    transport: SimTransport | None = None,
    staleness_limit_s: float = 0.5,
) -> tuple[RosControlHAL, SimTransport]:
    """Return a ``(hal, transport)`` pair wired for closed-loop testing.

    The returned ``transport`` can be used to inspect published messages and
    verify that ``read_state()`` reflects the positions from ``send_action()``.
    """
    t = transport or _make_transport(n_joints)
    hal = RosControlHAL(
        _make_description(n_joints=n_joints, supported_modes=supported_modes),
        controller_name="joint_trajectory_controller",
        publish_fn=t.publish,
        state_fn=t.state,
        staleness_limit_s=staleness_limit_s,
    )
    return hal, t


def _joint_position_action(n_joints: int = 3, horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * n_joints for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


class TestHALProtocol:
    def test_ros_control_hal_satisfies_protocol(self) -> None:
        """RosControlHAL must satisfy the HAL runtime-checkable Protocol."""
        hal, _ = _make_hal()
        assert isinstance(hal, HAL)

    def test_protocol_attributes_present(self) -> None:
        hal, _ = _make_hal()
        assert hasattr(hal, "description")
        assert hasattr(hal, "connect")
        assert hasattr(hal, "disconnect")
        assert hasattr(hal, "read_state")
        assert hasattr(hal, "send_action")
        assert hasattr(hal, "estop")


# ── Construction ──────────────────────────────────────────────────────────────


class TestRosControlHALConstruction:
    def test_raises_on_empty_joints(self) -> None:
        desc = _make_description(n_joints=0)
        # Pydantic won't stop us having 0 joints — but HAL should refuse
        with pytest.raises(ROSConfigError, match="no joints"):
            RosControlHAL(desc, controller_name="ctrl")

    def test_default_command_topic(self) -> None:
        hal, _ = _make_hal()
        assert hal._command_topic == "/joint_trajectory_controller/joint_trajectory"

    def test_custom_command_topic(self) -> None:
        hal = RosControlHAL(
            _make_description(),
            controller_name="ctrl",
            command_topic="/custom/topic",
        )
        assert hal._command_topic == "/custom/topic"

    def test_not_connected_after_init(self) -> None:
        hal, _ = _make_hal()
        assert hal._connected is False


# ── Lifecycle: connect / disconnect ───────────────────────────────────────────


class TestLifecycle:
    def test_connect_sets_connected(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        assert hal._connected is True

    def test_connect_twice_raises(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSRuntimeError, match="already connected"):
            hal.connect()

    def test_disconnect_clears_connected(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        hal.disconnect()
        assert hal._connected is False

    def test_disconnect_idempotent(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        hal.disconnect()
        hal.disconnect()  # must not raise
        assert hal._connected is False

    def test_disconnect_without_connect_is_noop(self) -> None:
        hal, _ = _make_hal()
        hal.disconnect()  # must not raise


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_raises_when_not_connected(self) -> None:
        hal, _ = _make_hal()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()

    def test_returns_joint_state_with_correct_names(self) -> None:
        hal, _ = _make_hal(n_joints=3)
        hal.connect()
        state = hal.read_state()
        assert isinstance(state, JointState)
        assert state.name == ["j0", "j1", "j2"]

    def test_zeroed_positions_on_fresh_transport(self) -> None:
        hal, _ = _make_hal(n_joints=2)
        hal.connect()
        state = hal.read_state()
        assert state.position == [0.0, 0.0]

    def test_state_fn_values_are_used(self) -> None:
        state_data = {"position": [1.0, 2.0, 3.0], "velocity": [0.1, 0.2, 0.3]}
        hal, _ = _make_hal(n_joints=3)
        # Override state_fn with a custom lambda to test the DI mechanism directly
        hal._state_fn = lambda: state_data  # type: ignore[assignment]
        hal.connect()
        state = hal.read_state()
        assert state.position == [1.0, 2.0, 3.0]
        assert state.velocity == [0.1, 0.2, 0.3]

    def test_stamp_ns_is_positive(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        state = hal.read_state()
        assert state.stamp_ns > 0

    def test_raises_perception_stale_when_too_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hal, _ = _make_hal(staleness_limit_s=0.001)
        hal.connect()
        # Wind the clock forward so the state appears stale
        original_monotonic = time.monotonic
        monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 10.0)
        with pytest.raises(ROSPerceptionStale):
            hal.read_state()


# ── send_action ───────────────────────────────────────────────────────────────


class TestSendAction:
    def test_raises_when_not_connected(self) -> None:
        hal, _ = _make_hal()
        action = _joint_position_action()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(action)

    def test_calls_publish_fn(self) -> None:
        hal, transport = _make_hal()
        hal.connect()
        action = _joint_position_action(n_joints=3)
        hal.send_action(action)
        assert transport.call_count == 1
        last = transport.last_call
        assert last is not None
        topic, msg = last
        assert topic == "/joint_trajectory_controller/joint_trajectory"
        assert msg["control_mode"] == ControlMode.JOINT_POSITION

    def test_raises_on_unsupported_control_mode(self) -> None:
        hal, _ = _make_hal(supported_modes=[ControlMode.JOINT_POSITION])
        hal.connect()
        action = Action(
            control_mode=ControlMode.CARTESIAN_POSE,
            horizon=1,
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="control_mode"):
            hal.send_action(action)

    def test_raises_on_joint_count_mismatch(self) -> None:
        hal, _ = _make_hal(n_joints=3)
        hal.connect()
        # Send an action with only 2 joint values for a 3-joint robot
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0, 0.0]],  # wrong length
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="3 joints"):
            hal.send_action(action)

    def test_allows_any_mode_when_supported_is_empty(self) -> None:
        """If supported_control_modes is empty, any mode is accepted."""
        hal, _ = _make_hal(supported_modes=[])
        hal.connect()
        action = Action(
            control_mode=ControlMode.CARTESIAN_POSE,
            horizon=1,
            stamp_ns=time.time_ns(),
        )
        hal.send_action(action)  # must not raise

    def test_multi_step_chunk(self) -> None:
        hal, transport = _make_hal(n_joints=3)
        hal.connect()
        action = _joint_position_action(n_joints=3, horizon=5)
        hal.send_action(action)
        assert transport.call_count == 1
        last = transport.last_call
        assert last is not None
        _, msg = last
        assert msg["horizon"] == 5


# ── estop ─────────────────────────────────────────────────────────────────────


class TestEstop:
    def test_estop_raises_e_stop_requested(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()

    def test_estop_disconnects_hal(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert hal._connected is False

    def test_read_state_fails_after_estop(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()

    def test_send_action_fails_after_estop(self) -> None:
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        action = _joint_position_action()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(action)

    def test_estop_exception_is_safety_violation(self) -> None:
        """ROSEStopRequested must be a ROSSafetyViolation (never silently caught)."""
        hal, _ = _make_hal()
        hal.connect()
        with pytest.raises(ROSSafetyViolation):
            hal.estop()


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_estop(self) -> None:
        """connect → read_state → send_action → estop lifecycle."""
        hal, transport = _make_hal(n_joints=2)

        hal.connect()
        assert hal._connected is True

        state = hal.read_state()
        assert len(state.name) == 2

        action = _joint_position_action(n_joints=2)
        hal.send_action(action)
        assert transport.call_count == 1

        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert hal._connected is False

    def test_connect_read_send_disconnect(self) -> None:
        """connect → read_state → send_action → disconnect (clean shutdown)."""
        hal, transport = _make_hal(n_joints=3)

        hal.connect()
        hal.read_state()
        action = _joint_position_action(n_joints=3)
        hal.send_action(action)
        assert transport.call_count == 1
        hal.disconnect()

        assert hal._connected is False
        with pytest.raises(ROSRuntimeError):
            hal.read_state()


# ── Closed-loop: send_action → read_state reflects commanded positions ────────


class TestClosedLoop:
    def test_send_action_updates_read_state_positions(self) -> None:
        """Positions commanded via send_action must appear in read_state."""
        hal, _ = _make_hal(n_joints=3)
        hal.connect()

        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[1.0, 2.0, 3.0]],
            stamp_ns=time.time_ns(),
        )
        hal.send_action(action)
        state = hal.read_state()
        assert state.position == [1.0, 2.0, 3.0]

    def test_multi_step_chunk_applies_last_waypoint(self) -> None:
        """For a multi-step trajectory the final waypoint is the resting state."""
        hal, _ = _make_hal(n_joints=2)
        hal.connect()

        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=3,
            joint_targets=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            stamp_ns=time.time_ns(),
        )
        hal.send_action(action)
        state = hal.read_state()
        assert state.position == pytest.approx([0.5, 0.6])

    def test_sequential_actions_update_state(self) -> None:
        """Each successive send_action updates the state independently."""
        hal, transport = _make_hal(n_joints=2)
        hal.connect()

        for i, targets in enumerate([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]):
            action = Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[targets],
                stamp_ns=time.time_ns(),
            )
            hal.send_action(action)
            state = hal.read_state()
            assert state.position == pytest.approx(targets), f"step {i}"

        assert transport.call_count == 3
