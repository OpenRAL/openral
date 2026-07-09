"""Unit tests for SO100FollowerHAL — SO-100 follower arm adapter.

These tests exercise the real lerobot code path via ``SO100DigitalTwin``, a
genuine lerobot ``Robot`` subclass that implements the full Robot interface
without any serial port or physical hardware.  No ``sys.modules`` mocking is
used; every lerobot import resolves to the real installed package.
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
from openral_hal.so100_follower import (
    _SO100_JOINT_NAMES,
    SO100_DESCRIPTION,
    SO100FollowerHAL,
    _deg_to_rad,
    _rad_to_deg,
)
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig

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
            pytest.raises(ROSConfigError, match="lerobot"),  # type: ignore[dict-item]
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

    def test_estop_disconnects(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert hal._connected is False
        assert hal._robot is None

    def test_read_state_fails_after_estop(self, hal: SO100FollowerHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()


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
