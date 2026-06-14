"""Sim tests for :class:`openral_hal.FrankaPandaHAL` against real MuJoCo.

The Panda HAL exposes 8 joints: 7 revolute arm joints + 1 synthetic gripper
channel reported in ``[0, 1]`` (0 = closed, 1 = fully open).  Internally the
gripper is driven by a single tendon-coupled actuator that mirrors both
finger ``qpos`` values.  These tests cover both arm and gripper paths against
real MuJoCo physics — no mocks.
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
    from robot_descriptions import panda_mj_description as _panda_desc

    _ = _panda_desc.MJCF_PATH  # triggers lazy clone / cache lookup
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
    ROSSafetyViolation,
)
from openral_hal import (
    FRANKA_PANDA_DESCRIPTION,
    FrankaPandaHAL,
)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"Panda MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# ── Schema-level checks ───────────────────────────────────────────────────────


class TestPandaDescription:
    def test_canonical_description_shape(self) -> None:
        desc = FRANKA_PANDA_DESCRIPTION
        assert desc.name == "franka_panda"
        assert desc.embodiment_kind == EmbodimentKind.MANIPULATOR.value
        # 7 arm joints + 1 normalised gripper channel
        assert len(desc.joints) == 8

    def test_arm_joints_are_revolute(self) -> None:
        arm = FRANKA_PANDA_DESCRIPTION.joints[:7]
        assert all(j.joint_type == JointType.REVOLUTE.value for j in arm)
        assert [j.name for j in arm] == [f"panda_joint{i}" for i in range(1, 8)]

    def test_gripper_is_prismatic_normalised_channel(self) -> None:
        gripper = FRANKA_PANDA_DESCRIPTION.joints[7]
        assert gripper.name == "panda_gripper"
        assert gripper.joint_type == JointType.PRISMATIC.value
        assert gripper.position_limits == (0.0, 1.0)

    def test_endeffector_is_parallel_gripper_70n(self) -> None:
        ee = FRANKA_PANDA_DESCRIPTION.end_effectors[0]
        assert ee.kind == "parallel_gripper"
        assert ee.max_grip_force_n == pytest.approx(70.0)

    def test_arm_torque_limits_match_datasheet(self) -> None:
        joints = {j.name: j for j in FRANKA_PANDA_DESCRIPTION.joints}
        # Joints 1..4 = 87 Nm, 5..7 = 12 Nm per Franka datasheet.
        for n in range(1, 5):
            assert joints[f"panda_joint{n}"].effort_limit == pytest.approx(87.0)
        for n in range(5, 8):
            assert joints[f"panda_joint{n}"].effort_limit == pytest.approx(12.0)

    def test_safety_envelope_requires_deadman(self) -> None:
        assert FRANKA_PANDA_DESCRIPTION.safety.deadman_required is True

    def test_embodiment_tags_include_panda(self) -> None:
        tags = FRANKA_PANDA_DESCRIPTION.capabilities.embodiment_tags
        assert "franka_panda" in tags
        assert "franka" in tags


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> FrankaPandaHAL:
    return FrankaPandaHAL(gravity_enabled=False, settle_steps=1500)


@pytest.fixture()
def connected_hal(hal: FrankaPandaHAL) -> FrankaPandaHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _make_action(targets: list[float]) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[targets],
        stamp_ns=time.time_ns(),
    )


# ── Franka-specific lifecycle ─────────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only Franka-specific tests here.


class TestFrankaLifecycle:
    def test_connect_loads_panda_model(self, hal: FrankaPandaHAL) -> None:
        """Franka-specific: verify 8 actuators and 9 qpos in menagerie XML."""
        hal.connect()
        try:
            assert hal._model is not None
            # 7 arm + 1 gripper actuators
            assert hal._model.nu == 8
            # 7 arm + 2 finger qpos
            assert hal._model.nq == 9
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_eight_joints(self, connected_hal: FrankaPandaHAL) -> None:
        state = connected_hal.read_state()
        assert len(state.name) == 8
        assert state.name == [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]
        assert len(state.position) == 8

    def test_initial_arm_positions_are_zero(self, connected_hal: FrankaPandaHAL) -> None:
        state = connected_hal.read_state()
        for q in state.position[:7]:
            assert abs(q) < 1e-6

    def test_initial_gripper_is_closed(self, connected_hal: FrankaPandaHAL) -> None:
        # MJCF default qpos has fingers at 0 → normalised gripper = 0.0
        state = connected_hal.read_state()
        assert state.position[7] == pytest.approx(0.0, abs=1e-6)


# ── Closed-loop arm control ───────────────────────────────────────────────────


class TestClosedLoopArm:
    def test_arm_target_converges(self, connected_hal: FrankaPandaHAL) -> None:
        # All 7 arm targets within their joint position limits, gripper closed.
        target = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.0]
        connected_hal.send_action(_make_action(target))
        state = connected_hal.read_state()
        assert state.position[:7] == pytest.approx(target[:7], abs=1e-3)

    def test_action_within_panda_joint_limits(self, connected_hal: FrankaPandaHAL) -> None:
        # Each target chosen to be inside the per-joint range from the
        # Franka datasheet (joint 4 is asymmetric; we pick a safe value).
        target = [-1.0, -1.0, 1.0, -1.5, 1.0, 2.0, -1.0, 0.0]
        connected_hal.send_action(_make_action(target))
        state = connected_hal.read_state()
        for i in range(7):
            assert state.position[i] == pytest.approx(target[i], abs=1e-3)


# ── Closed-loop gripper control ───────────────────────────────────────────────


class TestClosedLoopGripper:
    def test_open_gripper(self, connected_hal: FrankaPandaHAL) -> None:
        target = [0.0] * 7 + [1.0]  # fully open
        connected_hal.send_action(_make_action(target))
        state = connected_hal.read_state()
        # MuJoCo's tendon-coupled actuator may not push the fingers all the
        # way to 0.04 m within settle_steps; a generous tolerance is fine
        # because the closed→open round-trip below verifies actuation.
        assert state.position[7] > 0.5

    def test_close_gripper(self, connected_hal: FrankaPandaHAL) -> None:
        # Open then close; verify the gripper transitions in the correct
        # direction.
        connected_hal.send_action(_make_action([0.0] * 7 + [1.0]))
        opened = connected_hal.read_state().position[7]

        connected_hal.send_action(_make_action([0.0] * 7 + [0.0]))
        closed = connected_hal.read_state().position[7]

        assert opened > closed
        assert closed == pytest.approx(0.0, abs=1e-3)

    def test_gripper_value_clamped_to_range(self, connected_hal: FrankaPandaHAL) -> None:
        # An out-of-range gripper value (> 1.0) is clamped, not rejected,
        # because the action *vector* dimensions are valid; the HAL is
        # responsible for mapping gripper [0,1] → actuator range.
        target = [0.0] * 7 + [1.5]
        connected_hal.send_action(_make_action(target))
        state = connected_hal.read_state()
        # Should reach the open extreme (~1.0), not overshoot.
        assert state.position[7] <= 1.0 + 1e-6


# ── Action validation ─────────────────────────────────────────────────────────


class TestActionValidation:
    def test_rejects_unsupported_mode(self, connected_hal: FrankaPandaHAL) -> None:
        bad = Action(
            control_mode=ControlMode.JOINT_TORQUE,
            horizon=1,
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="control_mode"):
            connected_hal.send_action(bad)

    def test_rejects_seven_dim_action(self, connected_hal: FrankaPandaHAL) -> None:
        # Forgetting the gripper channel should fail; Panda needs 8 values.
        with pytest.raises(ROSConfigError, match="8 joints"):
            connected_hal.send_action(_make_action([0.0] * 7))


# ── Safety ────────────────────────────────────────────────────────────────────


class TestSafety:
    def test_estop_raises_safety_violation(self, connected_hal: FrankaPandaHAL) -> None:
        with pytest.raises(ROSSafetyViolation):
            connected_hal.estop()

    def test_estop_releases_state(self, hal: FrankaPandaHAL) -> None:
        hal.connect()
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert hal._connected is False
        with pytest.raises(ROSRuntimeError, match="not connected"):
            hal.read_state()
