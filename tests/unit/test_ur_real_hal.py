"""Unit tests for ``openral_hal.ur_real`` — UR5eRealHAL / UR10eRealHAL.

These tests exercise the real-hardware UR HAL adapters against an injected
:class:`~openral_hal.sim_transport.SimTransport` (a closed-loop in-memory
``ros2_control`` transport) — no ``rclpy``, no MuJoCo, no live robot.  Per
CLAUDE.md §5.4: real component or ``pytest.skip``; ``SimTransport`` is the
real RTDE-shaped fixture used here, not a mock.

The conformance test
``tests/unit/test_hal_protocol_conformance.py::HAL_BUILDERS`` already covers
the full HAL Protocol surface for these classes; the cases below pin
behaviours specific to the real-HW wrapper (manifest pinning, controller /
topic defaults, deadman topic, license metadata in the YAML).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSPerceptionStale
from openral_core.schemas import RobotDescription
from openral_hal.sim_transport import SimTransport
from openral_hal.ur import UR5e_DESCRIPTION, UR10e_DESCRIPTION
from openral_hal.ur_real import (
    UR5e_REAL_DESCRIPTION,
    UR5eRealHAL,
    UR10e_REAL_DESCRIPTION,
    UR10eRealHAL,
)

# ── Manifest pinning ─────────────────────────────────────────────────────────


class TestRealDescriptions:
    def test_ur5e_real_description_pins_real_hal(self) -> None:
        assert UR5e_REAL_DESCRIPTION.sdk_kind == "closed"
        assert UR5e_REAL_DESCRIPTION.hal.real == "openral_hal.ur_real:UR5eRealHAL"

    def test_ur10e_real_description_pins_real_hal(self) -> None:
        assert UR10e_REAL_DESCRIPTION.sdk_kind == "closed"
        assert UR10e_REAL_DESCRIPTION.hal.real == "openral_hal.ur_real:UR10eRealHAL"

    def test_ur5e_real_inherits_kinematics_from_sim(self) -> None:
        """Real and sim manifests must share kinematics and safety envelope.

        The only difference is ``sdk_kind`` (ADR-0031: ``hal`` is shared) — the
        production path is ``ros2_control`` + ``ur_robot_driver`` and the sim
        path is MuJoCo, but the robot itself is the same.
        """
        sim_joints = [
            (j.name, j.position_limits, j.velocity_limit, j.effort_limit)
            for j in UR5e_DESCRIPTION.joints
        ]
        real_joints = [
            (j.name, j.position_limits, j.velocity_limit, j.effort_limit)
            for j in UR5e_REAL_DESCRIPTION.joints
        ]
        assert sim_joints == real_joints
        assert UR5e_REAL_DESCRIPTION.safety == UR5e_DESCRIPTION.safety
        assert UR5e_REAL_DESCRIPTION.capabilities == UR5e_DESCRIPTION.capabilities
        assert UR5e_REAL_DESCRIPTION.embodiment_kind == UR5e_DESCRIPTION.embodiment_kind

    def test_ur10e_real_inherits_kinematics_from_sim(self) -> None:
        sim_joints = [
            (j.name, j.position_limits, j.velocity_limit, j.effort_limit)
            for j in UR10e_DESCRIPTION.joints
        ]
        real_joints = [
            (j.name, j.position_limits, j.velocity_limit, j.effort_limit)
            for j in UR10e_REAL_DESCRIPTION.joints
        ]
        assert sim_joints == real_joints
        assert UR10e_REAL_DESCRIPTION.safety == UR10e_DESCRIPTION.safety
        assert UR10e_REAL_DESCRIPTION.capabilities == UR10e_DESCRIPTION.capabilities


# ── HAL behaviour ─────────────────────────────────────────────────────────────


def _make_ur5e(robot_ip: str = "192.0.2.10") -> tuple[UR5eRealHAL, SimTransport]:
    transport = SimTransport(n_joints=6)
    hal = UR5eRealHAL(
        robot_ip=robot_ip,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, transport


def _make_ur10e(robot_ip: str = "192.0.2.11") -> tuple[UR10eRealHAL, SimTransport]:
    transport = SimTransport(n_joints=6)
    hal = UR10eRealHAL(
        robot_ip=robot_ip,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    return hal, transport


class TestURRealHALDefaults:
    def test_ur5e_records_robot_ip(self) -> None:
        hal, _ = _make_ur5e(robot_ip="192.168.1.42")
        assert hal.robot_ip == "192.168.1.42"

    def test_ur5e_default_command_topic_is_ur_driver(self) -> None:
        """Pinned to ``ur_robot_driver``'s scaled trajectory controller."""
        hal, _ = _make_ur5e()
        assert hal._command_topic == "/scaled_joint_trajectory_controller/joint_trajectory"

    def test_ur5e_default_joint_state_topic(self) -> None:
        hal, _ = _make_ur5e()
        assert hal._joint_state_topic == "/joint_states"

    def test_deadman_topic_is_ur_safety_mode(self) -> None:
        hal, _ = _make_ur5e()
        assert hal.deadman_topic == "/io_and_status_controller/safety_mode"

    def test_ur10e_default_command_topic_is_ur_driver(self) -> None:
        hal, _ = _make_ur10e()
        assert hal._command_topic == "/scaled_joint_trajectory_controller/joint_trajectory"


class TestURRealHALRoundTrip:
    """Closed-loop send→state via SimTransport — same pattern as test_hal.py."""

    def test_ur5e_send_action_publishes_trajectory(self) -> None:
        hal, transport = _make_ur5e()
        hal.connect()
        try:
            target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                )
            )
            assert transport.call_count == 1
            topic, msg = transport.calls[0]
            assert topic == "/scaled_joint_trajectory_controller/joint_trajectory"
            assert msg["joint_targets"] == [target]
            # SimTransport applies the last waypoint to its internal state, so
            # a follow-up read_state should see the commanded position.
            state = hal.read_state()
            assert list(state.position) == target
        finally:
            hal.disconnect()

    def test_ur10e_send_action_publishes_trajectory(self) -> None:
        hal, transport = _make_ur10e()
        hal.connect()
        try:
            target = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                )
            )
            assert transport.call_count == 1
            topic, _ = transport.calls[0]
            assert topic == "/scaled_joint_trajectory_controller/joint_trajectory"
        finally:
            hal.disconnect()


class TestURRealHALStaleness:
    def test_ur5e_read_state_raises_when_state_stale(self) -> None:
        hal, _ = _make_ur5e()
        hal.connect()
        try:
            hal._last_state_time -= hal._staleness_limit_s + 1.0
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
        finally:
            hal.disconnect()


# ── YAML ↔ HAL ───────────────────────────────────────────────────────────────


class TestRobotManifestMetadata:
    """The YAML manifests record the BSD-3 license posture (CLAUDE.md §7.4)."""

    @pytest.mark.parametrize(
        "manifest_path",
        ["robots/ur5e/robot.yaml", "robots/ur10e/robot.yaml"],
    )
    def test_ur_yaml_records_driver_license(self, manifest_path: str) -> None:
        desc = RobotDescription.from_yaml(str(Path(manifest_path)))
        meta = desc.onboard_compute
        assert meta.get("driver_license") == "BSD-3-Clause"
        assert meta.get("controller_name") == "scaled_joint_trajectory_controller"
