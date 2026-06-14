"""Sim tests for :class:`openral_hal.UR10eHAL` against real MuJoCo physics.

Mirrors the UR5e test suite but verifies the **distinct safety / capability
envelope** of the larger UR10e (higher payload, slower shoulder, larger
torque limits).  The kinematic structure is identical so closed-loop
convergence tests are a smaller subset.
"""

from __future__ import annotations

import time

import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    from robot_descriptions import ur10e_mj_description as _ur10e_desc

    _ = _ur10e_desc.MJCF_PATH  # triggers lazy clone / cache lookup
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointType,
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_hal import UR10e_DESCRIPTION, UR10eHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"UR10e MJCF unavailable: {_MJCF_ERROR}",
    ),
]


class TestUR10eDescription:
    def test_canonical_description_shape(self) -> None:
        desc = UR10e_DESCRIPTION
        assert desc.name == "ur10e"
        assert desc.embodiment_kind == EmbodimentKind.MANIPULATOR.value
        assert len(desc.joints) == 6
        assert all(j.joint_type == JointType.REVOLUTE.value for j in desc.joints)

    def test_payload_and_reach_match_datasheet(self) -> None:
        ee = UR10e_DESCRIPTION.end_effectors[0]
        assert ee.max_payload_kg == pytest.approx(12.5)
        assert ee.workspace_radius_m == pytest.approx(1.30, abs=1e-3)

    def test_velocity_limits_are_stricter_than_ur5e(self) -> None:
        # The UR10e shoulder is slower than the UR5e shoulder (120°/s vs
        # 180°/s).  The schema must reflect the datasheet difference.
        joints = {j.name: j for j in UR10e_DESCRIPTION.joints}
        shoulder_pan = joints["shoulder_pan_joint"].velocity_limit
        shoulder_lift = joints["shoulder_lift_joint"].velocity_limit
        assert shoulder_pan is not None and shoulder_pan < 3.142  # < pi rad/s
        assert shoulder_lift is not None and shoulder_lift < 3.142

    def test_effort_limits_match_datasheet(self) -> None:
        joints = {j.name: j for j in UR10e_DESCRIPTION.joints}
        # UR10e shoulder torque limit is 330 Nm per the datasheet.
        assert joints["shoulder_pan_joint"].effort_limit == pytest.approx(330.0)

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = UR10e_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_embodiment_tags_include_ur10e(self) -> None:
        tags = UR10e_DESCRIPTION.capabilities.embodiment_tags
        assert "ur10e" in tags
        assert "ur" in tags  # Shared family tag for skill matching


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> UR10eHAL:
    return UR10eHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: UR10eHAL) -> UR10eHAL:
    hal.connect()
    yield hal
    hal.disconnect()


# ── Protocol + lifecycle ──────────────────────────────────────────────────────


# ── UR10e-specific lifecycle tests ────────────────────────────────────────────
#
# Shared protocol compliance test (test_satisfies_hal_protocol) is consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only UR10e-specific lifecycle tests here.


class TestUR10eLifecycle:
    def test_connect_loads_six_actuators(self, hal: UR10eHAL) -> None:
        hal.connect()
        try:
            assert hal._model is not None
            assert hal._model.nu == 6
        finally:
            hal.disconnect()

    def test_send_action_requires_connect(self, hal: UR10eHAL) -> None:
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[0.0] * 6],
                    stamp_ns=time.time_ns(),
                )
            )

    def test_estop_clears_state(self, connected_hal: UR10eHAL) -> None:
        with pytest.raises(ROSEStopRequested):
            connected_hal.estop()
        assert connected_hal._connected is False


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    def test_position_command_converges(self, connected_hal: UR10eHAL) -> None:
        target = [0.3, -0.4, 0.5, -0.6, 0.3, -0.3]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position == pytest.approx(target, abs=1e-3)

    def test_returns_to_zero(self, connected_hal: UR10eHAL) -> None:
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.5] * 6],
                stamp_ns=time.time_ns(),
            )
        )
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0] * 6],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        for q in state.position:
            assert abs(q) < 1e-3


# ── Action validation ─────────────────────────────────────────────────────────


class TestActionValidation:
    def test_rejects_unsupported_mode(self, connected_hal: UR10eHAL) -> None:
        with pytest.raises(ROSConfigError, match="control_mode"):
            connected_hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_VELOCITY,
                    horizon=1,
                    stamp_ns=time.time_ns(),
                )
            )

    def test_rejects_dimension_mismatch(self, connected_hal: UR10eHAL) -> None:
        with pytest.raises(ROSConfigError, match="6 joints"):
            connected_hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[0.0] * 7],
                    stamp_ns=time.time_ns(),
                )
            )
