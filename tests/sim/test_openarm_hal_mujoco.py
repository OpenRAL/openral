"""Sim tests for :class:`openral_hal.OpenArmMujocoHAL` against real MuJoCo physics.

These tests load the upstream ``enactic/openarm_mujoco`` **v2**
bimanual MJCF (via :mod:`openral_hal._openarm_v2_assets`) and
exercise the full HAL lifecycle — connect → read_state →
send_action → estop / disconnect — against a real ``mj_step``
loop.  No mocks; the closed-loop behaviour comes from MuJoCo's own
native ``<position>`` actuators driving the 16-DoF bimanual rig
with per-joint kp/kv baked into the MJCF.

The point of this suite is the OpenArm "real hardware first day"
contract (CLAUDE.md §1.11): if these tests pass, the 16-DoF
``left arm 7 + left gripper 1 + right arm 7 + right gripper 1``
action layout, the joint indexing (driven finger per side, follower
finger via ``<equality>`` constraint, hidden from the public
surface), and the ``RobotDescription`` round-trip are guaranteed to
match what a future ``OpenArmRealHAL`` (wrapping lerobot's OpenArm
driver) will see when the physical arm arrives.

Gravity is disabled in the closed-loop tests so the joint positions
converge exactly to the commanded pose; the lifecycle / read_state
tests run gravity-on to mirror production.
"""

from __future__ import annotations

import os
import time

import pytest

# Use try/except → boolean + `pytestmark.skipif` rather than module-level
# `pytest.skip(allow_module_level=True)`: with `tests/sim/__init__.py`
# making this directory a Package, a Skipped raised at module-import time
# poisons the whole `tests/sim` Package collection ("found no collectors
# for ..." on every sibling). Our HAL never renders — we only need the
# physics — so deferring the decision to `pytestmark` keeps this module
# importable on hosts where MuJoCo's eager OSMesa probe or the OpenArm v2
# MJCF clone fails.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# The v2 MJCF is fetched by ``_openarm_v2_assets.ensure_openarm_v2_mjcf``
# on first use (the ``robot_descriptions`` package still pins to a
# pre-v2 commit).  Skip the suite cleanly on hosts where ``git``
# isn't on the PATH or the clone can't complete (e.g. air-gapped CI).
try:
    from openral_hal._openarm_v2_assets import ensure_openarm_v2_mjcf

    _OPENARM_MJCF: str = ensure_openarm_v2_mjcf()
    if not os.path.isfile(_OPENARM_MJCF):
        raise FileNotFoundError(f"missing MJCF at {_OPENARM_MJCF}")
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _OPENARM_MJCF = ""  # type: ignore[assignment]
    _MJCF_ERROR = str(exc)

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointState,
    JointType,
    ROSConfigError,
    ROSRuntimeError,
)
from openral_hal import OPENARM_DESCRIPTION, OpenArmMujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"OpenArm v2 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


_EXPECTED_OPENARM_JOINT_ORDER: list[str] = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
    "right_gripper",
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestOpenArmDescription:
    def test_canonical_description_shape(self) -> None:
        desc = OPENARM_DESCRIPTION
        assert desc.name == "openarm_v2"
        assert desc.embodiment_kind == EmbodimentKind.BIMANUAL.value
        # 7 arm + 1 gripper per side = 16
        assert len(desc.joints) == 16

    def test_joint_names_match_expected_layout(self) -> None:
        names = [j.name for j in OPENARM_DESCRIPTION.joints]
        assert names == _EXPECTED_OPENARM_JOINT_ORDER

    def test_all_joints_are_revolute(self) -> None:
        # v2 grippers are hinge joints (unlike v1's prismatic).
        for j in OPENARM_DESCRIPTION.joints:
            assert j.joint_type == JointType.REVOLUTE.value, j.name

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = OPENARM_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_bimanual(self) -> None:
        assert OPENARM_DESCRIPTION.capabilities.bimanual is True

    def test_embodiment_tags_include_openarm(self) -> None:
        tags = OPENARM_DESCRIPTION.capabilities.embodiment_tags
        assert "openarm" in tags
        assert "openarm_v2" in tags
        assert "enactic" in tags


# ── MJCF schema invariants (catches upstream drift) ──────────────────────────


class TestUpstreamSchema:
    """Guard against silent ``enactic/openarm_mujoco`` v2 schema drift.

    The :class:`OpenArmMujocoHAL` indexing relies on the v2 MJCF's
    18-joint / 16-actuator layout in a fixed order (left arm 7 +
    left fingers 2 + right arm 7 + right fingers 2).  If a future
    upstream upgrade reorders joints or flips an actuator's mode,
    these guards fail before the closed-loop tests do.
    """

    def test_joint_count_and_order(self) -> None:
        model = mujoco.MjModel.from_xml_path(_OPENARM_MJCF)
        # 18 joints — 16 actuated + 2 follower fingers (one per
        # gripper) coupled via an ``<equality>`` constraint.
        assert model.njnt == 18
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert names == [
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
            "openarm_left_finger_joint1",  # driven (act[7])
            "openarm_left_finger_joint2",  # follower (no actuator)
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
            "openarm_right_finger_joint1",  # driven (act[15])
            "openarm_right_finger_joint2",  # follower (no actuator)
        ]

    def test_actuators_are_position_mode(self) -> None:
        """Every v2 actuator is ``<position>`` mode with symmetric L/R gains.

        v1 had asymmetric LEFT/RIGHT finger gains (gain=1 vs 100)
        that made proportional control on the left side impossible.
        v2 fixed that.  This test fails loudly if upstream
        regresses.
        """
        model = mujoco.MjModel.from_xml_path(_OPENARM_MJCF)
        assert model.nu == 16
        # Standard MuJoCo position actuator compiles to
        # ``gainprm[0] == -biasprm[1]``.
        for i in range(model.nu):
            gain = float(model.actuator_gainprm[i][0])
            bias_p = float(model.actuator_biasprm[i][1])
            assert gain > 1.0, f"act[{i}] gain={gain} suggests torque mode"
            assert bias_p == pytest.approx(-gain, abs=1e-3), (
                f"act[{i}] gainprm={gain}, biasprm[1]={bias_p}; "
                "expected position actuator (gainprm == -biasprm[1])"
            )
        # Left + right finger actuator gains MUST be equal (the v1
        # asymmetric-gain bug fix).
        left_finger_kp = float(model.actuator_gainprm[7][0])
        right_finger_kp = float(model.actuator_gainprm[15][0])
        assert left_finger_kp == pytest.approx(right_finger_kp, abs=1e-3), (
            f"left finger kp={left_finger_kp} vs right kp={right_finger_kp}: "
            "v1 asymmetric-gain bug has regressed?"
        )

    def test_actuator_joint_alignment(self) -> None:
        model = mujoco.MjModel.from_xml_path(_OPENARM_MJCF)
        # Actuator i drives the expected MJCF joint name.
        expected_mjcf_actuator_joints = [
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
            "openarm_left_finger_joint1",
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
            "openarm_right_finger_joint1",
        ]
        for i in range(model.nu):
            driven_jnt = int(model.actuator_trnid[i, 0])
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driven_jnt)
            assert jnt_name == expected_mjcf_actuator_joints[i], (
                f"act[{i}] drives {jnt_name!r}, expected {expected_mjcf_actuator_joints[i]!r}"
            )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> OpenArmMujocoHAL:
    """Fresh OpenArm v2 HAL with gravity off and enough settle steps
    for the MJCF's native position-actuator PD to converge."""
    return OpenArmMujocoHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: OpenArmMujocoHAL) -> OpenArmMujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _zero_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * 16 for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── OpenArm-specific lifecycle tests ──────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only OpenArm-specific tests here.


class TestOpenArmLifecycle:
    def test_connect_loads_mujoco_model(self, hal: OpenArmMujocoHAL) -> None:
        """OpenArm-specific: verify 16 position actuators in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 16  # 16 position actuators in v2
        finally:
            hal.disconnect()

    def test_connect_seeds_ctrl_from_qpos(self, hal: OpenArmMujocoHAL) -> None:
        # connect() pre-loads ctrl with the current qpos so the v2
        # position actuators hold the rest pose on the first mj_step
        # rather than yanking toward ctrl=0.  Same pattern the
        # upstream ``mujoco_launch.py`` uses.  Post-ADR-0023 this is
        # driven by ``OPENARM_DESCRIPTION.sim.seed_ctrl_from_qpos=True``
        # — see openral_hal._mujoco_arm.MujocoArmHAL.connect.
        hal.connect()
        try:
            assert hal._data is not None
            for name in hal._joint_names:
                qpos_idx = hal._joint_qpos_addr[name]
                act_idx = hal._actuator_index[name]
                assert hal._data.ctrl[act_idx] == pytest.approx(
                    float(hal._data.qpos[qpos_idx]), abs=1e-6
                )
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_sixteen_joints(self, connected_hal: OpenArmMujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 16
        assert state.name == [j.name for j in OPENARM_DESCRIPTION.joints]
        assert len(state.position) == 16
        assert len(state.velocity) == 16
        assert state.stamp_ns > 0

    def test_perception_starvation_warns_not_latch_when_old(self, monkeypatch) -> None:
        """A starved servicing gap warns once and returns live state — never latches.

        Deploy-sim regression: ``read_state`` reads live in-process ``MjData``
        (always current), so a gap > ``staleness_limit_s`` is executor
        starvation, not bad data — it must NOT raise a latched
        ``ROSPerceptionStale``, and the next read must self-heal.
        """
        hal = OpenArmMujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
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
    def test_rejects_wrong_joint_count(self, connected_hal: OpenArmMujocoHAL) -> None:
        """OpenArm-specific: verify 16-joint contract."""
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 15],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="16 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No OpenArm-specific estop behavior to test.


# ── Closed-loop physics (v2 native PD — exact convergence) ───────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  v2's native ``<position>``
    actuators with per-class kp/kv track step inputs cleanly, so
    every closed-loop test asserts **exact** convergence to the
    commanded pose (unlike the v1 era, which required direction-only
    bounds on the wrist joints because the software PD couldn't
    brake the link inertia inside the actuator's saturated torque
    budget).
    """

    def test_hold_zero_pose(self, connected_hal: OpenArmMujocoHAL) -> None:
        connected_hal.send_action(_zero_action())
        state = connected_hal.read_state()
        for i, q in enumerate(state.position):
            assert abs(q) < 5e-3, f"joint {state.name[i]!r} drifted to {q:.4f}"

    # 0.15 rad is the largest magnitude that fits inside every arm
    # joint's MJCF ctrlrange — ``left_joint2`` is bounded to
    # [-3.32, +0.17] and ``right_joint2`` (mirrored) to [-0.17, +3.32],
    # so the narrowest positive / negative slot can absorb is 0.174 rad.
    # Pick a value comfortably inside both to avoid the actuator-side
    # clamp masking convergence success.
    _OPENARM_SAFE_ARM_TARGET: float = 0.15

    def test_left_arm_per_joint_convergence(self, connected_hal: OpenArmMujocoHAL) -> None:
        for slot in range(7):
            target = [0.0] * 16
            target[slot] = self._OPENARM_SAFE_ARM_TARGET
            connected_hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state = connected_hal.read_state()
            assert state.position[slot] == pytest.approx(self._OPENARM_SAFE_ARM_TARGET, abs=1e-2), (
                f"left-arm slot {slot} ({state.name[slot]!r}) failed to converge: "
                f"read {state.position[slot]!r}"
            )

    def test_right_arm_per_joint_convergence(self, connected_hal: OpenArmMujocoHAL) -> None:
        for slot in range(8, 15):
            target = [0.0] * 16
            target[slot] = self._OPENARM_SAFE_ARM_TARGET
            connected_hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state = connected_hal.read_state()
            assert state.position[slot] == pytest.approx(self._OPENARM_SAFE_ARM_TARGET, abs=1e-2), (
                f"right-arm slot {slot} ({state.name[slot]!r}) failed to converge: "
                f"read {state.position[slot]!r}"
            )

    def test_both_arms_converge_simultaneously(self, connected_hal: OpenArmMujocoHAL) -> None:
        """Multi-joint convergence — v2's per-class PD has enough
        damping for simultaneous-target moves without the saturation
        artifacts that plagued the v1 era."""
        target = [0.0] * 16
        # Modest deltas inside every joint's forcerange budget.
        target[0] = -0.3
        target[1] = -0.2
        target[3] = 0.5  # left_joint4 is positive-only
        target[8] = 0.3
        target[9] = 0.2
        target[11] = 0.5  # right_joint4 is positive-only
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        for i in (0, 1, 3, 8, 9, 11):
            assert state.position[i] == pytest.approx(target[i], abs=1e-2), (
                f"slot {i} ({state.name[i]!r}) read {state.position[i]!r}, expected {target[i]!r}"
            )

    def test_left_gripper_opens(self, connected_hal: OpenArmMujocoHAL) -> None:
        # v2 left gripper ctrlrange is [0, 0.7854] rad (closed → open).
        target = [0.0] * 16
        target[7] = 0.5
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[7] == pytest.approx(0.5, abs=1e-2)
        # Right gripper stays closed.
        assert state.position[15] == pytest.approx(0.0, abs=1e-2)

    def test_right_gripper_opens(self, connected_hal: OpenArmMujocoHAL) -> None:
        # v2 right gripper ctrlrange is [-0.7854, 0] rad (mirrored).
        target = [0.0] * 16
        target[15] = -0.5
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[7] == pytest.approx(0.0, abs=1e-2)
        assert state.position[15] == pytest.approx(-0.5, abs=1e-2)

    def test_follower_finger_tracks_driver(self, connected_hal: OpenArmMujocoHAL) -> None:
        """The v2 MJCF couples ``finger_joint2`` to ``finger_joint1``
        via an ``<equality>`` constraint per side.  Commanding the
        driven finger must move the follower in tandem — a wiring
        regression check that surfaces if a future v2 update drops
        the equality block."""
        target = [0.0] * 16
        target[7] = 0.4  # left gripper
        target[15] = -0.4  # right gripper
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        # Reach into the MJCF state for the follower finger qpos
        # (left_finger_joint2 = qpos[8]; right_finger_joint2 = qpos[17]).
        assert connected_hal._data is not None
        left_driver = float(connected_hal._data.qpos[7])
        left_follower = float(connected_hal._data.qpos[8])
        right_driver = float(connected_hal._data.qpos[16])
        right_follower = float(connected_hal._data.qpos[17])
        assert left_driver == pytest.approx(left_follower, abs=1e-2), (
            f"left finger driver {left_driver!r} vs follower {left_follower!r} — "
            "equality constraint missing?"
        )
        assert right_driver == pytest.approx(right_follower, abs=1e-2), (
            f"right finger driver {right_driver!r} vs follower {right_follower!r} — "
            "equality constraint missing?"
        )

    def test_per_slot_identity_wiring(self, connected_hal: OpenArmMujocoHAL) -> None:
        """Wiring check — every action slot drives exactly its joint.

        Sweeps the 16 slots one at a time with a fresh HAL per
        iteration (no state accumulation) and asserts the commanded
        joint converges to the commanded value while every other
        slot stays at zero.  Alternates signs to catch sign flips,
        L/R swaps, and off-by-one wiring errors.
        """
        # left_joint4 / right_joint4 are positive-only; left gripper is
        # positive-only [0, 0.7854]; right gripper is negative-only
        # [-0.7854, 0].  Pin those four slots to their valid sign;
        # alternate the rest.
        positive_only = {3, 7, 11}
        negative_only = {15}
        for slot in range(16):
            if slot in positive_only:
                expected_sign = 1.0
            elif slot in negative_only:
                expected_sign = -1.0
            else:
                expected_sign = 1.0 if slot % 2 == 0 else -1.0
            magnitude = (
                0.3
                if "gripper" in OPENARM_DESCRIPTION.joints[slot].name
                else self._OPENARM_SAFE_ARM_TARGET
            )
            expected = expected_sign * magnitude
            hal = OpenArmMujocoHAL(gravity_enabled=False, settle_steps=2000)
            hal.connect()
            try:
                target = [0.0] * 16
                target[slot] = expected
                hal.send_action(
                    Action(
                        control_mode=ControlMode.JOINT_POSITION,
                        horizon=1,
                        joint_targets=[target],
                        stamp_ns=time.time_ns(),
                    )
                )
                state = hal.read_state()
                assert state.position[slot] == pytest.approx(expected, abs=1e-2), (
                    f"slot {slot} ({state.name[slot]!r}) read {state.position[slot]!r}, "
                    f"expected {expected!r} — wiring mismatch?"
                )
                # Other slots stay near zero.
                for other in range(16):
                    if other == slot:
                        continue
                    assert abs(state.position[other]) < 2e-2, (
                        f"slot {other} ({state.name[other]!r}) drifted to "
                        f"{state.position[other]} when only slot {slot} was commanded"
                    )
            finally:
                hal.disconnect()


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: OpenArmMujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 16

            target = [0.0] * 16
            target[0] = -0.3  # left_joint1
            target[3] = 0.5  # left_joint4
            target[7] = 0.4  # left gripper open
            target[8] = 0.3  # right_joint1
            target[11] = 0.5  # right_joint4
            target[15] = -0.4  # right gripper open
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            for i in (0, 3, 7, 8, 11, 15):
                assert state1.position[i] == pytest.approx(target[i], abs=1e-2), (
                    f"slot {i} ({state1.name[i]!r}) read {state1.position[i]!r}, "
                    f"expected {target[i]!r}"
                )
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()
