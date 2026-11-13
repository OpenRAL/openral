"""Unit tests for SO100FollowerHAL — SO-100 follower arm adapter.

Exercises the real lerobot code path via ``SO100DigitalTwin``, a genuine lerobot
``Robot`` subclass with no serial port or physical hardware; every lerobot import
resolves to the real installed package (no ``sys.modules`` mocking).
"""

from __future__ import annotations

import math
import sys
import time
from unittest.mock import patch

import pytest
from openral_core import (
    Action,
    ControlMode,
    JointType,
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
    ROSSafetyViolation,
)
from openral_hal.protocol import (
    EStopObservableHAL,
    EStopRecovery,
    LifecycleEStopHAL,
    ResettableLifecycleEStopHAL,
)
from openral_hal.ros_control import DownstreamStopReporting
from openral_hal.so100_follower import (
    _SO100_JOINT_NAMES,
    SO100_DESCRIPTION,
    SO100FollowerHAL,
    _deg_to_rad,
    _hold_pose_on_torque_enable,
    _rad_to_deg,
)
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig

from tests.unit.fakes.fake_feetech_bus import SO_FOLLOWER_MOTORS, FakeFeetechBus

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def twin() -> SO100DigitalTwin:
    """Fresh SO100DigitalTwin with default positions (all zeros, gripper=50)."""
    return SO100DigitalTwin(SO100DigitalTwinConfig())


@pytest.fixture()
def hal(twin: SO100DigitalTwin) -> SO100FollowerHAL:
    """Connected SO100FollowerHAL backed by a digital twin."""
    h = SO100FollowerHAL(robot=twin)
    h.connect()
    return h


def _make_action(positions: list[float] | None = None) -> Action:
    pos = positions or [0.0] * 6
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[pos],
        stamp_ns=time.time_ns(),
    )


# ── SO100_DESCRIPTION ─────────────────────────────────────────────────────────


class TestSO100Description:
    def test_name(self) -> None:
        assert SO100_DESCRIPTION.name == "so100_follower"

    def test_six_joints(self) -> None:
        assert len(SO100_DESCRIPTION.joints) == 6

    def test_joint_names_match_lerobot_order(self) -> None:
        names = [j.name for j in SO100_DESCRIPTION.joints]
        assert names == _SO100_JOINT_NAMES

    def test_gripper_is_prismatic(self) -> None:
        gripper = next(j for j in SO100_DESCRIPTION.joints if j.name == "gripper")
        assert gripper.joint_type == JointType.PRISMATIC

    def test_supports_joint_position(self) -> None:
        assert ControlMode.JOINT_POSITION in SO100_DESCRIPTION.capabilities.supported_control_modes

    def test_embodiment_tag(self) -> None:
        assert "so100_follower" in SO100_DESCRIPTION.capabilities.embodiment_tags

    def test_hal_real(self) -> None:
        assert SO100_DESCRIPTION.hal.real == "openral_hal.so100_follower:SO100FollowerHAL"
        assert SO100_DESCRIPTION.hal.sim is None  # derives MujocoArmHAL from sim: block


# ── Unit conversions ──────────────────────────────────────────────────────────


class TestConversions:
    def test_deg_to_rad_zero(self) -> None:
        assert _deg_to_rad(0.0) == pytest.approx(0.0)

    def test_deg_to_rad_180(self) -> None:
        assert _deg_to_rad(180.0) == pytest.approx(math.pi)

    def test_rad_to_deg_pi(self) -> None:
        assert _rad_to_deg(math.pi) == pytest.approx(180.0)

    def test_roundtrip(self) -> None:
        for deg in [0.0, 45.0, -90.0, 180.0]:
            assert _rad_to_deg(_deg_to_rad(deg)) == pytest.approx(deg)


# ── SO100DigitalTwin ──────────────────────────────────────────────────────────


class TestSO100DigitalTwin:
    """Verify the digital twin satisfies the lerobot Robot protocol."""

    def test_default_positions(self, twin: SO100DigitalTwin) -> None:
        twin.connect(calibrate=False)
        obs = twin.get_observation()
        assert obs["shoulder_pan.pos"] == pytest.approx(0.0)
        assert obs["gripper.pos"] == pytest.approx(50.0)

    def test_custom_initial_positions(self) -> None:
        cfg = SO100DigitalTwinConfig(initial_positions={"shoulder_pan": 45.0})
        t = SO100DigitalTwin(cfg)
        t.connect(calibrate=False)
        assert t.get_observation()["shoulder_pan.pos"] == pytest.approx(45.0)
        # Un-specified joints fall back to defaults
        assert t.get_observation()["gripper.pos"] == pytest.approx(50.0)

    def test_send_action_updates_state(self, twin: SO100DigitalTwin) -> None:
        twin.connect(calibrate=False)
        twin.send_action({"shoulder_pan.pos": 90.0, "gripper.pos": 0.0})
        obs = twin.get_observation()
        assert obs["shoulder_pan.pos"] == pytest.approx(90.0)
        assert obs["gripper.pos"] == pytest.approx(0.0)

    def test_six_observation_keys(self, twin: SO100DigitalTwin) -> None:
        twin.connect(calibrate=False)
        obs = twin.get_observation()
        assert len(obs) == 6
        assert set(obs.keys()) == {f"{n}.pos" for n in _SO100_JOINT_NAMES}

    def test_connect_twice_raises(self, twin: SO100DigitalTwin) -> None:
        twin.connect(calibrate=False)
        # lerobot's check_if_already_connected raises DeviceAlreadyConnectedError(ConnectionError)
        with pytest.raises(ConnectionError):
            twin.connect(calibrate=False)

    def test_is_calibrated(self, twin: SO100DigitalTwin) -> None:
        assert twin.is_calibrated is True

    def test_observation_features_schema(self, twin: SO100DigitalTwin) -> None:
        assert set(twin.observation_features.keys()) == {f"{n}.pos" for n in _SO100_JOINT_NAMES}
        assert all(v is float for v in twin.observation_features.values())

    def test_action_features_schema(self, twin: SO100DigitalTwin) -> None:
        assert set(twin.action_features.keys()) == {f"{n}.pos" for n in _SO100_JOINT_NAMES}


# ── Construction ──────────────────────────────────────────────────────────────


class TestSO100FollowerHALConstruction:
    def test_default_port(self) -> None:
        hal = SO100FollowerHAL()
        assert hal._port == "/dev/ttyUSB0"

    def test_custom_port(self) -> None:
        hal = SO100FollowerHAL(port="/dev/ttyACM0")
        assert hal._port == "/dev/ttyACM0"

    def test_not_connected_after_init(self) -> None:
        hal = SO100FollowerHAL()
        assert hal._connected is False

    def test_description_is_so100(self) -> None:
        hal = SO100FollowerHAL()
        assert hal.description.name == "so100_follower"

    def test_injected_robot_stored(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        assert hal._injected_robot is twin


# ── Lifecycle ─────────────────────────────────────────────────────────────────


class TestSO100FollowerHALLifecycle:
    def test_connect_sets_connected(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        assert hal._connected is True

    def test_connect_twice_raises(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        with pytest.raises(ROSRuntimeError, match="already connected"):
            hal.connect()

    def test_disconnect_clears_connected(self, hal: SO100FollowerHAL) -> None:
        hal.disconnect()
        assert hal._connected is False

    def test_disconnect_idempotent(self, hal: SO100FollowerHAL) -> None:
        hal.disconnect()
        hal.disconnect()  # must not raise

    def test_disconnect_without_connect_is_noop(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        hal.disconnect()  # must not raise

    def test_lerobot_not_installed_raises_config_error(self) -> None:
        """When lerobot is missing and no robot is injected, connect() raises ROSConfigError."""
        hal = SO100FollowerHAL()  # no robot= injection → will try to import lerobot
        with (
            patch.dict(sys.modules, {"lerobot.robots.so_follower": None}),
            pytest.raises(ROSConfigError, match="lerobot"),
        ):
            hal.connect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestSO100FollowerHALReadState:
    def test_raises_when_not_connected(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()

    def test_returns_six_joints(self, hal: SO100FollowerHAL) -> None:
        state = hal.read_state()
        assert len(state.name) == 6
        assert len(state.position) == 6

    def test_joint_names_in_lerobot_order(self, hal: SO100FollowerHAL) -> None:
        state = hal.read_state()
        assert state.name == _SO100_JOINT_NAMES

    def test_zero_degrees_converted_to_zero_radians(self, hal: SO100FollowerHAL) -> None:
        """Default twin has shoulder_pan=0.0 deg → 0.0 rad."""
        state = hal.read_state()
        assert state.position[0] == pytest.approx(0.0)

    def test_degrees_converted_to_radians(self) -> None:
        """Twin with shoulder_pan=10.0 deg → ~0.1745 rad after read_state."""
        cfg = SO100DigitalTwinConfig(initial_positions={"shoulder_pan": 10.0})
        twin = SO100DigitalTwin(cfg)
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        state = hal.read_state()
        assert state.position[0] == pytest.approx(_deg_to_rad(10.0))

    def test_gripper_normalised_to_0_1(self, hal: SO100FollowerHAL) -> None:
        """Default twin has gripper=50.0 → 0.5 after normalisation."""
        state = hal.read_state()
        gripper_idx = _SO100_JOINT_NAMES.index("gripper")
        assert state.position[gripper_idx] == pytest.approx(0.5)

    def test_stamp_ns_is_positive(self, hal: SO100FollowerHAL) -> None:
        state = hal.read_state()
        assert state.stamp_ns > 0


def _record_sends(twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every ``twin.send_action`` call while keeping the twin's real behavior."""
    calls: list[object] = []
    original = twin.send_action

    def _recording(action: dict[str, float]) -> dict[str, float]:
        calls.append(action)
        sent_back: dict[str, float] = original(action)
        return sent_back

    monkeypatch.setattr(twin, "send_action", _recording)
    return calls


# ── send_action ───────────────────────────────────────────────────────────────


class TestSO100FollowerHALSendAction:
    def test_raises_when_not_connected(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(_make_action())

    def test_sends_zero_action(self, hal: SO100FollowerHAL) -> None:
        hal.send_action(_make_action())  # must not raise

    def test_radians_converted_to_degrees(
        self, hal: SO100FollowerHAL, twin: SO100DigitalTwin
    ) -> None:
        """pi/2 rad for shoulder_pan → 90.0 deg stored in twin._positions."""
        positions = [math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.5]
        hal.send_action(_make_action(positions))
        assert twin._positions["shoulder_pan"] == pytest.approx(90.0, abs=1e-4)

    def test_gripper_scaled_to_lerobot_range(
        self, hal: SO100FollowerHAL, twin: SO100DigitalTwin
    ) -> None:
        """Gripper 0.5 (normalised) → 50.0 in lerobot's [0, 100] range."""
        hal.send_action(_make_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]))
        assert twin._positions["gripper"] == pytest.approx(50.0)

    def test_gripper_open_full(self, hal: SO100FollowerHAL, twin: SO100DigitalTwin) -> None:
        """Gripper 1.0 → 100.0 in lerobot units."""
        hal.send_action(_make_action([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        assert twin._positions["gripper"] == pytest.approx(100.0)

    def test_gripper_closed(self, hal: SO100FollowerHAL, twin: SO100DigitalTwin) -> None:
        """Gripper 0.0 → 0.0 in lerobot units."""
        hal.send_action(_make_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert twin._positions["gripper"] == pytest.approx(0.0)

    def test_raises_on_wrong_control_mode(self, hal: SO100FollowerHAL) -> None:
        action = Action(
            control_mode=ControlMode.CARTESIAN_POSE,
            horizon=1,
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="JOINT_POSITION"):
            hal.send_action(action)

    def test_raises_on_missing_joint_targets(self, hal: SO100FollowerHAL) -> None:
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="joint_targets"):
            hal.send_action(action)

    def test_raises_on_joint_count_mismatch(self, hal: SO100FollowerHAL) -> None:
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0, 0.0]],  # wrong length
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="6 joints"):
            hal.send_action(action)

    def test_action_reflected_in_next_read_state(
        self, hal: SO100FollowerHAL, twin: SO100DigitalTwin
    ) -> None:
        """send_action updates twin state; subsequent read_state reflects it."""
        target_rad = math.pi / 4  # 45 degrees
        hal.send_action(_make_action([target_rad, 0.0, 0.0, 0.0, 0.0, 0.0]))
        state = hal.read_state()
        assert state.position[0] == pytest.approx(target_rad, abs=1e-4)


# ── estop ─────────────────────────────────────────────────────────────────────


class TestSO100FollowerHALEstop:
    def test_estop_raises(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            hal.estop()

    def test_estop_is_safety_violation(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSSafetyViolation):
            hal.estop()

    def test_lifecycle_estop_disconnects(self, hal: SO100FollowerHAL) -> None:
        # No motor bus to prove the torque-off: the session is closed, nothing kept.
        assert isinstance(hal, LifecycleEStopHAL)
        assert hal.estop_recovery is EStopRecovery.RESETTABLE
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert hal._connected is False
        assert hal._robot is None
        assert hal._estop_robot is None

    def test_read_state_fails_after_estop(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()


class _TwinWithFailingClose(SO100DigitalTwin):
    """A twin whose ``disconnect`` raises, as lerobot does when the port is gone."""

    def disconnect(self) -> None:
        raise ConnectionError("serial port vanished")


def _twin_on_bus(bus: FakeFeetechBus, *, failing_close: bool = False) -> SO100DigitalTwin:
    """A digital twin with a serial-boundary fake bus attached as ``robot.bus``."""
    cls = _TwinWithFailingClose if failing_close else SO100DigitalTwin
    twin = cls(SO100DigitalTwinConfig())
    twin.bus = bus
    return twin


def _estop(hal: SO100FollowerHAL) -> str:
    with pytest.raises(ROSEStopRequested) as info:
        hal.estop()
    return str(info.value)


class TestSO100FollowerHALDownstreamStopReport:
    """Issue #526: the e-stop reports torque off per servo and never swallows a failure."""

    def test_reports_nothing_before_the_first_estop(self) -> None:
        hal = SO100FollowerHAL(robot=_twin_on_bus(FakeFeetechBus()))
        assert isinstance(hal, DownstreamStopReporting)
        assert hal.last_stop_report is None

    def test_torque_off_on_every_servo_is_acknowledged(self) -> None:
        bus = FakeFeetechBus()
        bus.port_handler.is_using = True  # a control-loop transaction cut short
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
        hal.connect()

        message = _estop(hal)

        report = hal.last_stop_report
        assert report is not None and report.stopped
        assert "downstream stop acknowledged" in message
        assert report.fields()["downstream_stop"] == "acknowledged"
        assert report.vendor_stop == "torque_off (IEC 60204-1 category 0)"
        assert dict(report.controller_states) == dict.fromkeys(SO_FOLLOWER_MOTORS, "torque_off")
        assert bus.torque == dict.fromkeys(SO_FOLLOWER_MOTORS, 0)
        assert [w[1] for w in bus.writes] == list(SO_FOLLOWER_MOTORS)
        assert [r[1] for r in bus.reads] == list(SO_FOLLOWER_MOTORS)
        assert bus.port_handler.cleared == 1
        assert hal._connected is False
        assert hal._robot is None

    def test_a_dark_servo_is_unacknowledged_and_the_others_are_still_stopped(self) -> None:
        bus = FakeFeetechBus(dark=frozenset({"elbow_flex"}))
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
        hal._robot = hal._injected_robot  # bypass the connect-time ping, which would refuse
        hal._connected = True

        message = _estop(hal)

        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "NOT acknowledged" in message
        fields = report.fields()
        assert fields["downstream_stop"] == "unacknowledged"
        assert "elbow_flex: ConnectionError" in fields["downstream_stop_detail"]
        assert report.controller_states["elbow_flex"] == "torque_unknown"
        others = [m for m in SO_FOLLOWER_MOTORS if m != "elbow_flex"]
        assert all(report.controller_states[m] == "torque_off" for m in others)
        assert all(bus.torque[m] == 0 for m in others)

    def test_a_write_that_does_not_take_fails_the_read_back(self) -> None:
        bus = FakeFeetechBus(stuck_torque=frozenset({"gripper"}))
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
        hal.connect()

        _estop(hal)

        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert report.controller_states["gripper"] == "torque_on"
        assert "gripper: Torque_Enable reads back 1" in report.detail

    def test_an_unreadable_servo_is_not_counted_as_stopped(self) -> None:
        bus = FakeFeetechBus(unreadable=frozenset({"wrist_roll"}))
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
        hal.connect()

        _estop(hal)

        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert report.controller_states["wrist_roll"] == "torque_unknown"

    def test_a_proven_stop_keeps_the_bus_open_and_closes_it_on_disconnect(self) -> None:
        bus = FakeFeetechBus()
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus, failing_close=True))
        hal.connect()

        _estop(hal)

        report = hal.last_stop_report
        assert report is not None and report.stopped
        assert "bus kept open read-only" in report.detail
        assert hal._estop_robot is not None
        hal.disconnect()  # the failing close is logged, never raised: torque is proven off
        assert hal._estop_robot is None

    def test_a_failed_close_with_no_bus_to_verify_is_unacknowledged(self) -> None:
        twin = _TwinWithFailingClose(SO100DigitalTwinConfig())
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()

        _estop(hal)

        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "no motor bus" in report.detail
        assert "disconnect failed" in report.detail

    def test_a_robot_without_a_motor_bus_is_never_reported_stopped(
        self, hal: SO100FollowerHAL
    ) -> None:
        _estop(hal)
        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "no motor bus" in report.detail

    def test_estop_while_disconnected_is_unacknowledged(self, twin: SO100DigitalTwin) -> None:
        hal = SO100FollowerHAL(robot=twin)
        _estop(hal)
        report = hal.last_stop_report
        assert report is not None and not report.stopped
        assert "not connected" in report.detail

    def test_reconnect_clears_the_previous_report(self) -> None:
        hal = SO100FollowerHAL(robot=_twin_on_bus(FakeFeetechBus()))
        hal.connect()
        _estop(hal)
        assert hal.last_stop_report is not None
        hal.connect()
        assert hal.last_stop_report is None


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestSO100FollowerHALFullLifecycle:
    def test_connect_read_send_estop(self, twin: SO100DigitalTwin) -> None:
        """connect → read_state → send_action → estop."""
        hal = SO100FollowerHAL(robot=twin)

        hal.connect()
        assert hal._connected

        state = hal.read_state()
        assert len(state.name) == 6

        hal.send_action(_make_action([0.0] * 6))

        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert not hal._connected

    def test_connect_read_send_disconnect(self, twin: SO100DigitalTwin) -> None:
        """connect → read_state → send_action → clean disconnect."""
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        hal.read_state()
        hal.send_action(_make_action([0.0] * 6))
        hal.disconnect()
        assert not hal._connected

    def test_multiple_send_action_accumulate(self, twin: SO100DigitalTwin) -> None:
        """Each send_action updates twin state; read_state always reflects latest."""
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()

        steps = [math.pi / 6, math.pi / 4, math.pi / 3]
        for rad in steps:
            hal.send_action(_make_action([rad, 0.0, 0.0, 0.0, 0.0, 0.0]))
            state = hal.read_state()
            assert state.position[0] == pytest.approx(rad, abs=1e-4)

        hal.disconnect()


# ── reset_to_pose (real-arm starting-pose ramp) ───────────────────────────────


class TestSO100FollowerHALResetToPose:
    """The slow-ramp real-hardware counterpart of the sim arms' qpos snap."""

    def test_reaches_target_pose(
        self, hal: SO100FollowerHAL, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "sleep", lambda _s: None)  # ramp without wall time
        target = [0.2, -0.4, 0.6, 0.3, -0.1, 0.8]
        hal.reset_to_pose(target)
        state = hal.read_state()
        for got, want in zip(state.position, target, strict=True):
            assert got == pytest.approx(want, abs=1e-3)

    def test_ramp_is_interpolated_not_snapped(
        self, twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The twin must receive many intermediate waypoints, not one snap."""
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        seen: list[float] = []
        original = twin.send_action

        def _recording(action: dict[str, float]) -> dict[str, float]:
            seen.append(action["shoulder_pan.pos"])
            sent_back: dict[str, float] = original(action)
            return sent_back

        monkeypatch.setattr(twin, "send_action", _recording)
        hal.reset_to_pose([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        # 1 rad at 0.5 rad/s → 2 s → ~60 steps at 30 Hz; monotone ramp.
        assert len(seen) >= 30
        assert seen == sorted(seen)
        # First waypoint is a small step from 0, NOT the full target.
        assert seen[0] < math.degrees(1.0) / 2

    def test_noop_when_already_at_pose(
        self, hal: SO100FollowerHAL, twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        current = hal.read_state().position
        calls = _record_sends(twin, monkeypatch)
        hal.reset_to_pose(current)
        assert calls == []

    def test_wrong_length_raises(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSConfigError, match="expects 6"):
            hal.reset_to_pose([0.0, 0.0, 0.0])

    def test_not_connected_raises(self, twin: SO100DigitalTwin) -> None:
        h = SO100FollowerHAL(robot=twin)
        with pytest.raises(ROSRuntimeError, match="reset_to_pose"):
            h.reset_to_pose([0.0] * 6)

    @pytest.mark.parametrize(
        "bad",
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],  # gripper above [0, 1]
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.1],  # gripper below [0, 1]
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # shoulder_pan beyond pi
            [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, float("inf"), 0.0, 0.0, 0.0, 0.5],
        ],
    )
    def test_out_of_limit_or_nonfinite_pose_refused_before_any_write(
        self,
        hal: SO100FollowerHAL,
        twin: SO100DigitalTwin,
        monkeypatch: pytest.MonkeyPatch,
        bad: list[float],
    ) -> None:
        """The ResetToPose contract promises joint-limit protection."""
        calls = _record_sends(twin, monkeypatch)
        with pytest.raises(ROSConfigError, match="reset_to_pose"):
            hal.reset_to_pose(bad)
        assert calls == []

    def test_limits_come_from_the_loaded_manifest(
        self, twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """so101_follower reuses this class: its tighter manifest limits apply."""
        from pathlib import Path

        from openral_core import RobotDescription

        repo = Path(__file__).resolve().parents[2]
        desc = RobotDescription.from_yaml(str(repo / "robots/so101_follower/robot.yaml"))
        h = SO100FollowerHAL(robot=twin, description=desc)
        h.connect()
        calls = _record_sends(twin, monkeypatch)
        with pytest.raises(ROSConfigError, match="shoulder_pan"):
            h.reset_to_pose([2.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # inside pi, outside 1.9199
        assert calls == []

    def test_large_move_keeps_the_ramp_speed_bound(
        self, twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 6 rad move used to be squeezed into the 6 s cap (1 rad/s)."""
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        hal = SO100FollowerHAL(robot=twin)
        hal.connect()
        hal.reset_to_pose([-3.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        seen: list[float] = []
        original = twin.send_action

        def _recording(action: dict[str, float]) -> dict[str, float]:
            seen.append(action["shoulder_pan.pos"])
            sent_back: dict[str, float] = original(action)
            return sent_back

        monkeypatch.setattr(twin, "send_action", _recording)
        hal.reset_to_pose([3.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        # 6 rad at <= 0.5 rad/s needs >= 12 s, i.e. >= 360 steps at 30 Hz.
        assert len(seen) >= 360


class TestJointValuesToLerobot:
    """The SINGLE unit conversion both send_action and the reset ramp share."""

    def test_gripper_and_arm_units(self) -> None:
        from openral_hal.so100_follower import _joint_values_to_lerobot

        out = _joint_values_to_lerobot([math.pi, 0.0, -math.pi / 2, 0.0, 0.0, 0.5])
        assert out["shoulder_pan.pos"] == pytest.approx(180.0)
        assert out["elbow_flex.pos"] == pytest.approx(-90.0)
        assert out["gripper.pos"] == pytest.approx(50.0)  # [0, 1] → [0, 100]

    def test_send_action_and_ramp_agree(
        self, hal: SO100FollowerHAL, twin: SO100DigitalTwin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the ramp used to re-implement the conversion inline, so a
        calibration change wouldn't apply to it. Final ramp waypoint must be
        byte-identical to a send_action of the same pose."""
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        target = [0.2, -0.4, 0.6, 0.3, -0.1, 0.8]

        sent: list[dict[str, float]] = []
        original = twin.send_action

        def _record(a: dict[str, float]) -> dict[str, float]:
            sent.append(a)
            sent_back: dict[str, float] = original(a)
            return sent_back

        monkeypatch.setattr(twin, "send_action", _record)

        hal.reset_to_pose(target)
        ramp_final = sent[-1]

        hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                joint_targets=[target],
            )
        )
        assert sent[-1] == ramp_final


def test_so100_envelope_requires_a_deadman_like_its_manifests() -> None:
    """The in-code literal must agree with robots/so10{0,1}_follower/robot.yaml."""
    from openral_hal.so100_follower import SO100_DESCRIPTION

    assert SO100_DESCRIPTION.safety.deadman_required is True


class TestHoldPoseOnTorqueEnable:
    """Torque-on holds the current pose instead of driving to a stale goal (#545 follow-up)."""

    def test_goal_is_synced_to_present_before_every_torque_on(self) -> None:
        bus = FakeFeetechBus(torque=dict.fromkeys(SO_FOLLOWER_MOTORS, 0))
        _hold_pose_on_torque_enable(bus)
        bus.enable_torque()  # what lerobot's configure() calls on leaving torque_disabled()
        assert bus.goal_at_enable == bus.present
        assert set(bus.torque.values()) == {1}

    def test_a_single_motor_enable_syncs_only_that_motor(self) -> None:
        bus = FakeFeetechBus(torque=dict.fromkeys(SO_FOLLOWER_MOTORS, 0))
        _hold_pose_on_torque_enable(bus)
        bus.enable_torque("gripper")
        assert bus.goal_at_enable == {"gripper": bus.present["gripper"]}
        assert bus.goal["shoulder_pan"] == 0

    def test_a_goal_that_does_not_take_refuses_torque_on(self) -> None:
        bus = FakeFeetechBus(
            torque=dict.fromkeys(SO_FOLLOWER_MOTORS, 0), stuck_goal=frozenset({"elbow_flex"})
        )
        _hold_pose_on_torque_enable(bus)
        with pytest.raises(ROSRuntimeError, match="elbow_flex"):
            bus.enable_torque()
        assert bus.torque["elbow_flex"] == 0
        assert ("Torque_Enable", "elbow_flex", 1) not in bus.writes


# ── E-stop: read-only observation and in-process release ──────────────────────


def _proven_stop(bus: FakeFeetechBus | None = None) -> tuple[SO100FollowerHAL, FakeFeetechBus]:
    """A connected HAL on a fake bus, e-stopped with torque proven off on every servo."""
    bus = bus or FakeFeetechBus()
    hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
    hal.connect()
    _estop(hal)
    assert hal.last_stop_report is not None and hal.last_stop_report.stopped
    return hal, bus


class TestSO100FollowerHALEstopObserveAndReset:
    """The arm stays observable while e-stopped and a release resumes it holding its pose."""

    def test_implements_both_estop_extensions(self, hal: SO100FollowerHAL) -> None:
        assert isinstance(hal, EStopObservableHAL)
        assert isinstance(hal, ResettableLifecycleEStopHAL)

    def test_reads_during_a_proven_stop_never_write(self) -> None:
        hal, bus = _proven_stop()
        writes = len(bus.writes)
        state = hal.read_state_estopped()
        assert list(state.name) == list(SO_FOLLOWER_MOTORS)
        assert len(bus.writes) == writes
        assert bus.torque == dict.fromkeys(SO_FOLLOWER_MOTORS, 0)
        # Commands stay refused: the normal paths still see a disconnected HAL.
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(_make_action())

    def test_an_unproven_stop_offers_no_reads_and_no_release(self) -> None:
        bus = FakeFeetechBus(dark=frozenset({"elbow_flex"}))
        hal = SO100FollowerHAL(robot=_twin_on_bus(bus))
        hal._robot = hal._injected_robot  # bypass the connect-time ping, which would refuse
        hal._connected = True
        _estop(hal)
        assert hal.last_stop_report is not None and not hal.last_stop_report.stopped
        with pytest.raises(ROSRuntimeError, match="no read-only session"):
            hal.read_state_estopped()
        with pytest.raises(ROSRuntimeError, match="restart the lifecycle node"):
            hal.reset_estop()
        assert hal._connected is False

    def test_release_holds_the_present_pose_and_resumes(self) -> None:
        hal, bus = _proven_stop()
        bus.present = {m: 1500 + 37 * i for i, m in enumerate(SO_FOLLOWER_MOTORS)}  # moved by hand
        bus.goal = dict.fromkeys(SO_FOLLOWER_MOTORS, 0)  # a stale goal: a full-speed jump

        hal.reset_estop()

        # Every servo got Goal := Present before its torque came on: no motion.
        assert bus.goal_at_enable == bus.present
        assert bus.torque == dict.fromkeys(SO_FOLLOWER_MOTORS, 1)
        assert hal._connected is True and hal._estop_robot is None
        assert hal.last_stop_report is None
        hal.read_state()
        hal.send_action(_make_action())

    def test_a_goal_that_does_not_take_keeps_the_arm_limp_and_observable(self) -> None:
        hal, bus = _proven_stop(FakeFeetechBus(stuck_goal=frozenset({"wrist_flex"})))
        with pytest.raises(ROSRuntimeError, match="torque cut again"):
            hal.reset_estop()
        assert bus.torque == dict.fromkeys(SO_FOLLOWER_MOTORS, 0)
        assert hal._connected is False
        hal.read_state_estopped()  # still observable
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(_make_action())

    def test_a_restart_after_a_proven_stop_frees_the_bus_first(self) -> None:
        hal, _ = _proven_stop()
        hal.connect()
        assert hal._connected is True and hal._estop_robot is None

    def test_a_failed_rearm_whose_cut_is_unproven_closes_the_session(self) -> None:
        # The re-arm fails (a goal that does not take) and the follow-up torque cut is
        # not proven (a servo that stops answering): only a proven stop may stay
        # releasable, so the session closes and the stop is recorded unacknowledged.
        hal, bus = _proven_stop()
        bus.stuck_goal = frozenset({"wrist_flex"})
        bus.unreadable = frozenset({"gripper"})
        with pytest.raises(ROSRuntimeError, match="NOT proven"):
            hal.reset_estop()
        assert hal._estop_robot is None
        assert hal.last_stop_report is not None and not hal.last_stop_report.stopped
        with pytest.raises(ROSRuntimeError, match="restart the lifecycle node"):
            hal.reset_estop()

    def test_a_second_stop_cuts_torque_through_the_kept_session(self) -> None:
        hal, bus = _proven_stop()
        writes = len(bus.writes)
        _estop(hal)
        assert len(bus.writes) > writes  # torque off written again, not skipped
        assert hal.last_stop_report is not None and hal.last_stop_report.stopped
        assert hal._estop_robot is not None
        hal.read_state_estopped()

    def test_an_estop_while_the_watcher_arms_aborts_the_rearm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The watcher is a subprocess (the process-boundary double): it arms outside the
        # bus lock, and a stop that lands meanwhile wins.
        from types import SimpleNamespace

        hal, bus = _proven_stop()
        bus.port_handler.baudrate = 1_000_000  # type: ignore[attr-defined]  # reason: fake port
        # lerobot's MotorsBus.motors values carry the servo id the watcher is armed with.
        monkeypatch.setattr(
            FakeFeetechBus,
            "motors",
            property(
                lambda self: {
                    n: SimpleNamespace(id=i + 1) for i, n in enumerate(SO_FOLLOWER_MOTORS)
                }
            ),
        )
        hal._injected_robot = None  # take the real-bus path, which arms a watcher

        stopped: list[bool] = []

        class _Watcher:
            def stop(self, *, clean: bool) -> None:
                stopped.append(clean)

        def spawn(*_args: object, **_kw: object) -> _Watcher:
            with pytest.raises(ROSEStopRequested):
                hal.estop()
            return _Watcher()

        monkeypatch.setattr("openral_hal.so100_follower.FeetechTorqueWatcher.spawn", spawn)
        with pytest.raises(ROSRuntimeError, match="arrived during the re-arm"):
            hal.reset_estop()
        assert stopped == [True]
        assert hal._connected is False
        assert bus.torque == dict.fromkeys(SO_FOLLOWER_MOTORS, 0)

    def test_a_command_waiting_for_the_bus_is_refused_after_a_stop(self) -> None:
        import threading

        hal = SO100FollowerHAL(robot=_twin_on_bus(FakeFeetechBus()))
        hal.connect()
        errors: list[BaseException] = []

        def send() -> None:
            try:
                hal.send_action(_make_action())
            except BaseException as exc:  # reason: collected for the assertion
                errors.append(exc)

        with hal._bus_lock:  # a transaction in flight
            t = threading.Thread(target=send)
            t.start()
            time.sleep(0.1)  # the command has passed its connected check, waits for the bus
            hal._connected = False  # what estop() does first, before taking the lock
        t.join(2)
        assert errors and isinstance(errors[0], ROSRuntimeError)
        assert "not connected" in str(errors[0])


def test_a_composed_robot_stays_observable_through_its_parts_stop() -> None:
    """``ComposedHAL`` forwards ``read_state_estopped``: an SO-101 part read read-only."""
    from openral_hal.composed import ComposedHAL

    assert hasattr(ComposedHAL, "read_state_estopped")
    hal, _ = _proven_stop()
    composed = ComposedHAL.__new__(ComposedHAL)
    composed._parts = [hal]
    composed.description = hal.description
    state = composed.read_state_estopped()
    assert list(state.name) == [j.name for j in hal.description.joints]
