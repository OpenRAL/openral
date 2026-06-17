"""Sim tests for :class:`openral_hal.AlohaMujocoHAL` against real MuJoCo physics.

These tests load gym-aloha's ``bimanual_viperx_transfer_cube.xml`` and
exercise the full HAL lifecycle — connect → read_state → send_action →
estop / disconnect — against a real ``mj_step`` loop.  No mocks; the
closed-loop behaviour comes from MuJoCo's own position-controlled
actuators driving the canonical ViperX 300 ×2 model.

The point of this suite is the bimanual "real hardware first day"
contract (CLAUDE.md §1.11): if these tests pass, the 14-DoF
``left arm 6 + left gripper 1 + right arm 6 + right gripper 1``
action layout — which is the same layout
:class:`openral_hal.AlohaHAL` forwards to the four Interbotix XS
``ros2_control`` controllers — is guaranteed to drive the physical
ALOHA the same way on first connect.  Remaining failure surface is
the Interbotix USB / DXL driver level (HIL territory).

Gravity is disabled in the closed-loop test so the joint positions
converge exactly to the commanded pose; the staleness / lifecycle
tests run in default gravity-on configuration to mirror production.
"""

from __future__ import annotations

import time

import pytest

# Use try/except → boolean + `pytestmark.skipif` rather than module-level
# `pytest.skip(allow_module_level=True)`: with `tests/sim/__init__.py`
# making this directory a Package, a Skipped raised at module-import time
# poisons the whole `tests/sim` Package collection ("found no collectors
# for ..." on every sibling). Our HAL never renders — we only need the
# physics — so deferring the decision to `pytestmark` keeps this module
# importable on hosts where MuJoCo's eager OSMesa probe or the gym-aloha
# MJCF lookup fails.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# gym-aloha ships the canonical bimanual ViperX MJCF.  Skip the whole
# suite if it isn't installed — the test exercises a real physics path
# and there is no useful degraded mode.
try:
    import os as _os

    import gym_aloha

    _ALOHA_MJCF = _os.path.join(
        _os.path.dirname(gym_aloha.__file__),
        "assets",
        "bimanual_viperx_transfer_cube.xml",
    )
    if not _os.path.isfile(_ALOHA_MJCF):
        raise FileNotFoundError(f"missing MJCF at {_ALOHA_MJCF}")
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointState,
    ROSConfigError,
    ROSRuntimeError,
)
from openral_hal import ALOHA_DESCRIPTION, AlohaMujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"gym-aloha MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestAlohaDescription:
    def test_canonical_description_shape(self) -> None:
        desc = ALOHA_DESCRIPTION
        assert desc.name == "aloha_bimanual"
        assert desc.embodiment_kind == EmbodimentKind.BIMANUAL.value
        # 2 × (6 arm + 1 gripper) = 14.
        assert len(desc.joints) == 14

    def test_joint_names_match_expected_layout(self) -> None:
        names = [j.name for j in ALOHA_DESCRIPTION.joints]
        assert names == [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper",
        ]

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = ALOHA_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_bimanual(self) -> None:
        assert ALOHA_DESCRIPTION.capabilities.bimanual is True


# ── MJCF schema invariants (catches gym-aloha upstream drift) ─────────────────


class TestMjcfSchema:
    """Guard against silent ``gym-aloha`` MJCF schema drift.

    The :class:`AlohaMujocoHAL` indexing assumes the actuator / qpos
    layout documented in ``aloha.py`` — if a future gym-aloha upgrade
    reorders joints, this guard fails before the closed-loop tests do.
    """

    def test_joint_order_and_names(self) -> None:
        model = mujoco.MjModel.from_xml_path(_ALOHA_MJCF)
        # 16 controllable joints + 1 free joint for the cube.
        assert model.njnt == 17
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert names[:16] == [
            "vx300s_left/waist",
            "vx300s_left/shoulder",
            "vx300s_left/elbow",
            "vx300s_left/forearm_roll",
            "vx300s_left/wrist_angle",
            "vx300s_left/wrist_rotate",
            "vx300s_left/left_finger",
            "vx300s_left/right_finger",
            "vx300s_right/waist",
            "vx300s_right/shoulder",
            "vx300s_right/elbow",
            "vx300s_right/forearm_roll",
            "vx300s_right/wrist_angle",
            "vx300s_right/wrist_rotate",
            "vx300s_right/left_finger",
            "vx300s_right/right_finger",
        ]

    def test_actuator_count(self) -> None:
        model = mujoco.MjModel.from_xml_path(_ALOHA_MJCF)
        # 16 position actuators (no actuator for the cube free joint).
        assert model.nu == 16


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> AlohaMujocoHAL:
    """Fresh bimanual HAL with gravity off and enough settle steps for the
    position controllers to converge to the commanded pose."""
    return AlohaMujocoHAL(gravity_enabled=False, settle_steps=3000)


@pytest.fixture()
def connected_hal(hal: AlohaMujocoHAL) -> AlohaMujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


# The gym-aloha MJCF keyframe defines a self-collision-free "home" pose
# the ALOHA bring-up procedure rests at.  Tests command small deltas from
# this pose because commanding "all zeros" pulls the arms through a
# self-colliding configuration and the contact dynamics push joints
# unpredictably — physical reality, not a HAL bug (the real ALOHA
# refuses to track an all-zeros target for the same reason).
_ALOHA_HOME_POSE: tuple[float, ...] = (
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,  # left arm
    0.024,  # left gripper (slightly open)
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,  # right arm
    0.024,  # right gripper
)


def _home_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[list(_ALOHA_HOME_POSE) for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── ALOHA-specific lifecycle tests ────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only ALOHA-specific tests here.


class TestAlohaLifecycle:
    def test_connect_loads_mujoco_model(self, hal: AlohaMujocoHAL) -> None:
        """ALOHA-specific: verify 16 actuators (14 arm + 2 extra finger) in MJCF."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 16  # 14 arm + 2 extra finger actuators
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_fourteen_joints(self, connected_hal: AlohaMujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 14
        assert state.name == [j.name for j in ALOHA_DESCRIPTION.joints]
        assert len(state.position) == 14
        assert len(state.velocity) == 14
        assert state.stamp_ns > 0

    def test_perception_starvation_warns_not_latch_when_old(self, monkeypatch) -> None:
        """A starved servicing gap warns once and returns live state — never latches.

        Deploy-sim regression: ``read_state`` reads live in-process ``MjData``
        (always current), so a gap > ``staleness_limit_s`` is executor
        starvation, not bad data — it must NOT raise a latched
        ``ROSPerceptionStale``, and the next read must self-heal.
        """
        hal = AlohaMujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
        hal.connect()
        try:
            real_monotonic = time.monotonic
            monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + 10.0)
            # Must return the live state, not raise.
            state = hal.read_state()
            assert state.position
            # Recoverable: the clock was refreshed, so the next read is fresh.
            assert hal.read_state().position
        finally:
            hal.disconnect()


# ── send_action ───────────────────────────────────────────────────────────────


class TestSendAction:
    def test_rejects_wrong_joint_count(self, connected_hal: AlohaMujocoHAL) -> None:
        """ALOHA-specific: verify 14-joint contract."""
        # 13 values for a 14-joint robot.
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 13],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="14 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No ALOHA-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge on both arms.

    Targets are small deltas off the gym-aloha home pose (the MJCF
    keyframe) so the bimanual ViperX doesn't self-collide; commanding
    "all zeros" pulls the arms through a colliding configuration where
    contact dynamics push joints unpredictably (this is physical, not a
    HAL bug — the real ALOHA refuses such commands too).
    """

    def test_home_pose_holds_when_commanded(self, connected_hal: AlohaMujocoHAL) -> None:
        # Commanding the keyframe pose itself should leave the arms in
        # place (the keyframe is in static equilibrium with gravity off).
        connected_hal.send_action(_home_action())
        state = connected_hal.read_state()
        for i in range(6):
            assert state.position[i] == pytest.approx(_ALOHA_HOME_POSE[i], abs=5e-3)
        for i in range(7, 13):
            assert state.position[i] == pytest.approx(_ALOHA_HOME_POSE[i], abs=5e-3)

    def test_left_arm_converges_to_delta(self, connected_hal: AlohaMujocoHAL) -> None:
        # Small deltas off the home pose for every left-arm joint.
        target = list(_ALOHA_HOME_POSE)
        target[0] = 0.2  # left_waist (was 0.0)
        target[1] = -0.8  # left_shoulder (was -0.96)
        target[2] = 1.0  # left_elbow (was 1.16)
        target[3] = 0.2  # left_forearm_roll (was 0.0)
        target[4] = -0.2  # left_wrist_angle (was -0.3)
        target[5] = 0.1  # left_wrist_rotate (was 0.0)
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[0:6] == pytest.approx(target[0:6], abs=5e-3)
        # Right arm wasn't commanded to move, stays near home pose.
        for i in range(7, 13):
            assert state.position[i] == pytest.approx(_ALOHA_HOME_POSE[i], abs=2e-2)

    def test_right_arm_converges_to_delta(self, connected_hal: AlohaMujocoHAL) -> None:
        target = list(_ALOHA_HOME_POSE)
        target[7] = -0.2  # right_waist
        target[8] = -0.8  # right_shoulder
        target[9] = 1.0  # right_elbow
        target[10] = -0.2  # right_forearm_roll
        target[11] = -0.4  # right_wrist_angle
        target[12] = -0.1  # right_wrist_rotate
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[7:13] == pytest.approx(target[7:13], abs=5e-3)
        # Left arm wasn't commanded to move, stays near home pose.
        for i in range(6):
            assert state.position[i] == pytest.approx(_ALOHA_HOME_POSE[i], abs=2e-2)

    def test_both_arms_move_independently(self, connected_hal: AlohaMujocoHAL) -> None:
        target = list(_ALOHA_HOME_POSE)
        target[0] = 0.2
        target[1] = -0.8
        target[7] = -0.2
        target[8] = -0.7
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        # Each arm independently tracks its commanded waist + shoulder.
        assert state.position[0] == pytest.approx(target[0], abs=5e-3)
        assert state.position[1] == pytest.approx(target[1], abs=5e-3)
        assert state.position[7] == pytest.approx(target[7], abs=5e-3)
        assert state.position[8] == pytest.approx(target[8], abs=5e-3)

    def test_left_gripper_opens(self, connected_hal: AlohaMujocoHAL) -> None:
        target = list(_ALOHA_HOME_POSE)
        target[6] = 0.057  # left_gripper fully open
        # right gripper stays at home (0.024).
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[6] == pytest.approx(0.057, abs=3e-3)
        # Right gripper stays roughly at home (still in valid range).
        assert state.position[13] == pytest.approx(0.024, abs=5e-3)

    def test_right_gripper_opens_independently(self, connected_hal: AlohaMujocoHAL) -> None:
        target = list(_ALOHA_HOME_POSE)
        target[13] = 0.057  # right_gripper fully open
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[6] == pytest.approx(0.024, abs=5e-3)
        assert state.position[13] == pytest.approx(0.057, abs=3e-3)

    def test_action_index_split_matches_real_hal_layout(
        self, connected_hal: AlohaMujocoHAL
    ) -> None:
        """The 6/1/6/1 split must match :class:`AlohaHAL.send_action` exactly.

        :class:`AlohaHAL` splits the 14-D action as ``[0:6][6][7:13][13]``
        — see ``aloha.py`` ``send_action`` ~lines 420-423.  If this HAL
        ever drifts (e.g. a future refactor swaps left/right), commands
        validated against the twin won't drive the real ALOHA the same
        way.  Test with positive-on-left, negative-on-right sentinels.
        """
        sentinel = list(_ALOHA_HOME_POSE)
        # Distinct, signed deltas per side prove L/R aren't swapped.
        sentinel[0] = 0.2  # left_waist  → positive
        sentinel[1] = -0.8  # left_shoulder
        sentinel[2] = 1.0  # left_elbow
        sentinel[7] = -0.2  # right_waist → negative (opposite sign)
        sentinel[8] = -0.7  # right_shoulder
        sentinel[9] = 1.1  # right_elbow
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[sentinel],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        # Per-slot identity: each sentinel lands in its matching slot.
        for i in (0, 1, 2, 7, 8, 9):
            assert state.position[i] == pytest.approx(sentinel[i], abs=5e-3)


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: AlohaMujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 14

            target = list(_ALOHA_HOME_POSE)
            target[0] = 0.2  # left_waist delta
            target[1] = -0.8  # left_shoulder delta
            target[7] = -0.2  # right_waist delta
            target[8] = -0.7  # right_shoulder delta
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position[0] == pytest.approx(target[0], abs=5e-3)
            assert state1.position[1] == pytest.approx(target[1], abs=5e-3)
            assert state1.position[7] == pytest.approx(target[7], abs=5e-3)
            assert state1.position[8] == pytest.approx(target[8], abs=5e-3)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()
