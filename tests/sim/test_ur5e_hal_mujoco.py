"""Sim tests for :class:`openral_hal.UR5eHAL` against real MuJoCo physics.

These tests load the ``mujoco_menagerie`` UR5e MJCF (via
``robot_descriptions``) and exercise the full HAL lifecycle — connect →
read_state → send_action → estop / disconnect — against a real ``mj_step``
loop.  No mocks; the closed-loop behaviour comes from MuJoCo's own
position-controlled actuators.

Gravity is disabled in the closed-loop test so the joint positions converge
exactly to the commanded pose; the staleness / lifecycle tests run in default
gravity-on configuration to mirror production.
"""

from __future__ import annotations

import math
import time

import pytest

# ``MUJOCO_GL=osmesa`` (set in CI to render without a display) makes
# ``import mujoco`` eagerly load the OSMesa renderer, which crashes on hosts
# without the OSMesa OpenGL stack.  Our HAL never renders — we only need the
# physics — so we treat any failure during the import as a skip.
try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# robot_descriptions clones mujoco_menagerie (~650 MB) from GitHub the first
# time a model is loaded; on CI runners with restricted network or first-run
# cold caches the clone may fail or time out.  Treat any failure as a skip so
# the suite stays green on flaky environments and only enforces correctness
# when the MJCF is actually reachable.
try:
    from robot_descriptions import ur5e_mj_description as _ur5e_desc

    _ = _ur5e_desc.MJCF_PATH  # triggers lazy clone / cache lookup
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointState,
    JointType,
    ROSConfigError,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_hal import UR5e_DESCRIPTION, UR5eHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"UR5e MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestUR5eDescription:
    def test_canonical_description_shape(self) -> None:
        desc = UR5e_DESCRIPTION
        assert desc.name == "ur5e"
        assert desc.embodiment_kind == EmbodimentKind.MANIPULATOR.value
        assert len(desc.joints) == 6
        assert all(j.joint_type == JointType.REVOLUTE.value for j in desc.joints)

    def test_joint_names_match_mujoco_menagerie(self) -> None:
        names = [j.name for j in UR5e_DESCRIPTION.joints]
        assert names == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = UR5e_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_safety_envelope_has_deadman(self) -> None:
        # Per CLAUDE.md §1.1 — the UR5e safety envelope must require deadman
        # because the arm has enough mass to injure operators.
        assert UR5e_DESCRIPTION.safety.deadman_required is True

    def test_embodiment_tags_include_ur5e(self) -> None:
        assert "ur5e" in UR5e_DESCRIPTION.capabilities.embodiment_tags


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> UR5eHAL:
    """Fresh UR5e HAL with gravity off and enough settle steps for the
    position controllers to converge to the commanded pose."""
    return UR5eHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: UR5eHAL) -> UR5eHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _zero_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * 6 for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── UR5e-specific lifecycle tests ─────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only UR5e-specific tests here.


class TestUR5eLifecycle:
    def test_connect_loads_mujoco_model(self, hal: UR5eHAL) -> None:
        """UR5e-specific: verify 6-DoF actuator count in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            # 6-DoF UR5e
            assert hal._model.nu == 6
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_six_joints(self, connected_hal: UR5eHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 6
        assert state.name == [j.name for j in UR5e_DESCRIPTION.joints]
        assert len(state.position) == 6
        assert len(state.velocity) == 6
        assert state.stamp_ns > 0

    def test_initial_positions_are_zero(self, connected_hal: UR5eHAL) -> None:
        state = connected_hal.read_state()
        for q in state.position:
            assert abs(q) < 1e-6

    def test_perception_stale_raises_when_old(self, monkeypatch) -> None:
        hal = UR5eHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
        hal.connect()
        try:
            real_monotonic = time.monotonic
            monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10.0)
            with pytest.raises(ROSPerceptionStale):
                hal.read_state()
        finally:
            hal.disconnect()


# ── send_action ───────────────────────────────────────────────────────────────


class TestSendAction:
    def test_rejects_wrong_joint_count(self, connected_hal: UR5eHAL) -> None:
        """UR5e-specific: verify 6-joint contract."""
        # 5 values for a 6-joint robot.
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 5],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="6 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No UR5e-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge."""

    def test_send_action_drives_joints_to_target(self, connected_hal: UR5eHAL) -> None:
        target = [0.5, -0.5, 0.5, -0.5, 0.5, 0.5]
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

    def test_multi_step_chunk_settles_at_last_waypoint(self, connected_hal: UR5eHAL) -> None:
        chunk = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.2, 0.1, 0.0, 0.0, 0.0],
        ]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=3,
                joint_targets=chunk,
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position == pytest.approx(chunk[-1], abs=1e-3)

    def test_sequential_actions_advance_state(self, connected_hal: UR5eHAL) -> None:
        targets = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
        ]
        for tgt in targets:
            connected_hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[tgt],
                    stamp_ns=time.time_ns(),
                )
            )
            state = connected_hal.read_state()
            assert state.position == pytest.approx(tgt, abs=1e-3)

    def test_target_within_position_limits(self, connected_hal: UR5eHAL) -> None:
        # The elbow joint is restricted to [-pi, pi] in the MJCF and the
        # OpenRAL JointSpec.  A target inside this range must be reachable.
        state_before = connected_hal.read_state()
        target = [0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state_after = connected_hal.read_state()
        # Elbow moved to π/2; other joints stayed.
        assert state_after.position[2] == pytest.approx(math.pi / 2, abs=1e-3)
        for i in (0, 1, 3, 4, 5):
            assert abs(state_after.position[i] - state_before.position[i]) < 1e-3


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: UR5eHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 6

            target = [0.2, -0.2, 0.2, -0.2, 0.2, 0.2]
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position == pytest.approx(target, abs=1e-3)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()
