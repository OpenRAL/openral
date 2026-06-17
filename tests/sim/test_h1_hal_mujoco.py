"""Sim tests for :class:`openral_hal.H1MujocoHAL` against real MuJoCo physics.

These tests load the ``mujoco_menagerie`` Unitree H1 MJCF (via
``robot_descriptions``) and exercise the full HAL lifecycle — connect →
read_state → send_action → estop / disconnect — against a real
``mj_step`` loop.  No mocks; the closed-loop behaviour comes from
MuJoCo's own position-controlled actuators driving the 19-DoF
humanoid.

The point of this suite is the H1 "real hardware first day" contract
(CLAUDE.md §1.11): if these tests pass, the 19-DoF joint-position
action layout, lifecycle, and ``RobotDescription`` joint order are
guaranteed to match what a future ``H1RealHAL`` over ``unitree_sdk2``
will see when the physical robot arrives.  Balance + walking remain
out of scope — they live in CLAUDE.md §6.2 (M2 C++ S0 cerebellum) and
no Python HAL twin can validate them.

Gravity is disabled in every test because without an S0 balance
controller the floating-base humanoid falls over in <1 s; with
gravity off the joints converge to their commanded targets and the
contract assertions are deterministic.
"""

from __future__ import annotations

import time

import pytest

# ``MUJOCO_GL=osmesa`` (set in CI to render without a display) makes
# ``import mujoco`` eagerly load the OSMesa renderer, which crashes on
# hosts without the OSMesa OpenGL stack.  Our HAL never renders — we only
# need the physics — so we treat any failure during the import as a skip.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

# robot_descriptions clones mujoco_menagerie (~650 MB) from GitHub the
# first time a model is loaded; on CI runners with restricted network
# or first-run cold caches the clone may fail or time out.  Treat any
# failure as a skip so the suite stays green on flaky environments and
# only enforces correctness when the MJCF is actually reachable.
try:
    from robot_descriptions import h1_mj_description as _h1_desc

    _ = _h1_desc.MJCF_PATH  # triggers lazy clone / cache lookup
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
    ROSRuntimeError,
)
from openral_hal import H1_DESCRIPTION, H1MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"H1 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# Expected joint name order — matches the menagerie MJCF and the
# in-code ``_H1_JOINT_NAMES`` tuple.  Keep these two ordered lists
# adjacent so any future drift surfaces in one diff.  H1 menagerie
# omits the ``_joint`` suffix convention used by G1.
_EXPECTED_H1_JOINT_ORDER: list[str] = [
    # Left leg (5)
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    # Right leg (5)
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    # Torso (1)
    "torso",
    # Left arm (4)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    # Right arm (4)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestH1Description:
    def test_canonical_description_shape(self) -> None:
        desc = H1_DESCRIPTION
        assert desc.name == "h1"
        assert desc.embodiment_kind == EmbodimentKind.HUMANOID.value
        assert len(desc.joints) == 19
        assert all(j.joint_type == JointType.REVOLUTE.value for j in desc.joints)

    def test_joint_names_match_menagerie_order(self) -> None:
        names = [j.name for j in H1_DESCRIPTION.joints]
        assert names == _EXPECTED_H1_JOINT_ORDER

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = H1_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_advertise_humanoid_traits(self) -> None:
        caps = H1_DESCRIPTION.capabilities
        assert "bipedal" in caps.locomotion
        assert caps.bimanual is True

    def test_embodiment_tags_include_h1(self) -> None:
        assert "h1" in H1_DESCRIPTION.capabilities.embodiment_tags
        assert "humanoid" in H1_DESCRIPTION.capabilities.embodiment_tags

    def test_safety_envelope_requires_deadman(self) -> None:
        # Per CLAUDE.md §1.1 — the H1 has enough mass + reach to injure
        # an operator; a deadman is mandatory at the safety envelope.
        assert H1_DESCRIPTION.safety.deadman_required is True

    def test_floating_base_is_not_in_description(self) -> None:
        # The MJCF floating base is implicit world state, not a
        # controllable joint — it must NOT appear in the description.
        names = [j.name for j in H1_DESCRIPTION.joints]
        # H1 menagerie's free joint is unnamed (None) — just verify the
        # description has only the 19 actuated joints.
        assert len(names) == 19


# ── Menagerie XML schema invariants (catches upstream drift) ──────────────────


class TestMenagerieSchema:
    """Guard against silent ``mujoco_menagerie`` schema drift.

    The :class:`H1MujocoHAL` indexing assumes the floating-base + 19
    actuated-joint order documented in ``h1.py``.  If a future
    menagerie upgrade reorders joints, these guards fail before the
    closed-loop tests do and point at the right place.
    """

    def test_joint_count_and_floating_base(self) -> None:
        model = mujoco.MjModel.from_xml_path(_h1_desc.MJCF_PATH)
        # 1 free joint (floating base, name=None in this MJCF) +
        # 19 hinge joints.
        assert model.njnt == 20
        assert int(model.jnt_type[0]) == 0  # mjJNT_FREE

    def test_actuated_joint_order_matches_description(self) -> None:
        model = mujoco.MjModel.from_xml_path(_h1_desc.MJCF_PATH)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)
        ]
        assert names == _EXPECTED_H1_JOINT_ORDER

    def test_actuator_count_and_alignment(self) -> None:
        model = mujoco.MjModel.from_xml_path(_h1_desc.MJCF_PATH)
        assert model.nu == 19
        # Actuator name + driven-joint name must match 1:1 in order.
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            driven_jnt = int(model.actuator_trnid[i, 0])
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driven_jnt)
            assert act_name == jnt_name
            # And actuator index i drives the i-th actuated joint
            # (qpos addr = 7 + i because of the floating base).
            assert jnt_name == _EXPECTED_H1_JOINT_ORDER[i]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> H1MujocoHAL:
    """Fresh H1 HAL with gravity off and enough settle steps for the
    position controllers to converge to the commanded pose.

    Gravity is **always** disabled in these tests — without an S0
    cerebellum the floating-base humanoid falls in <1 s, which would
    couple every closed-loop assertion to body dynamics.
    """
    return H1MujocoHAL(gravity_enabled=False, settle_steps=3000)


@pytest.fixture()
def connected_hal(hal: H1MujocoHAL) -> H1MujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _zero_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * 19 for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── H1-specific lifecycle tests ───────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only H1-specific tests here.


class TestH1Lifecycle:
    def test_connect_loads_mujoco_model(self, hal: H1MujocoHAL) -> None:
        """H1-specific: verify 19 actuated joints + floating base in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 19  # 19 actuated joints
            assert hal._model.njnt == 20  # +1 floating base
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_19_joints(self, connected_hal: H1MujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 19
        assert state.name == [j.name for j in H1_DESCRIPTION.joints]
        assert len(state.position) == 19
        assert len(state.velocity) == 19
        assert state.stamp_ns > 0

    def test_initial_positions_are_zero(self, connected_hal: H1MujocoHAL) -> None:
        # The menagerie H1 MJCF has no keyframe; every actuated joint
        # defaults to qpos=0 (upright neutral pose).
        state = connected_hal.read_state()
        for q in state.position:
            assert abs(q) < 1e-3

    def test_perception_starvation_warns_not_latch_when_old(self, monkeypatch) -> None:
        """A starved servicing gap warns once and returns live state — never latches.

        Deploy-sim regression: ``read_state`` reads live in-process ``MjData``
        (always current), so a gap > ``staleness_limit_s`` is executor
        starvation, not bad data — it must NOT raise a latched
        ``ROSPerceptionStale``, and the next read must self-heal.
        """
        hal = H1MujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
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
    def test_rejects_wrong_joint_count(self, connected_hal: H1MujocoHAL) -> None:
        """H1-specific: verify 19-joint contract."""
        # 18 values for a 19-joint robot.
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 18],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="19 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No H1-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge
    across all 19 joints when gravity is off (without an S0 cerebellum
    the floating base falls instantly with gravity on — see the suite
    docstring)."""

    def test_send_action_holds_zero_pose(self, connected_hal: H1MujocoHAL) -> None:
        # Commanding zero on every actuator should leave every joint at
        # zero (the menagerie's default rest pose with gravity off).
        connected_hal.send_action(_zero_action())
        state = connected_hal.read_state()
        for i, q in enumerate(state.position):
            assert abs(q) < 5e-3, f"joint {state.name[i]!r} drifted to {q:.4f}"

    def test_left_arm_converges_to_target(self, connected_hal: H1MujocoHAL) -> None:
        target = [0.0] * 19
        # Left-arm joint indices: 11..14 (after legs 0..9 and torso 10).
        target[11] = 0.4  # left_shoulder_pitch
        target[12] = 0.3  # left_shoulder_roll
        target[13] = 0.2  # left_shoulder_yaw
        target[14] = 0.6  # left_elbow
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[11:15] == pytest.approx(target[11:15], abs=5e-3)
        # Right arm (15..18) and legs (0..9) stay at zero.
        for i in list(range(0, 11)) + list(range(15, 19)):
            assert abs(state.position[i]) < 5e-3, f"joint {state.name[i]!r} moved"

    def test_right_arm_converges_to_target(self, connected_hal: H1MujocoHAL) -> None:
        target = [0.0] * 19
        # Right-arm joint indices: 15..18.
        target[15] = 0.4  # right_shoulder_pitch
        target[16] = -0.3  # right_shoulder_roll (opposite sign to left)
        target[17] = -0.2  # right_shoulder_yaw
        target[18] = 0.6  # right_elbow
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[15:19] == pytest.approx(target[15:19], abs=5e-3)
        for i in range(0, 15):
            assert abs(state.position[i]) < 5e-3

    def test_legs_converge_to_target(self, connected_hal: H1MujocoHAL) -> None:
        # Small leg deflection — exercise the indexing for the lower
        # body without driving the robot into self-collision.  Left leg
        # = indices 0..4, right leg = indices 5..9.  Stay well inside
        # the menagerie limits (hip_yaw / hip_roll only ±0.43).
        target = [0.0] * 19
        target[2] = -0.2  # left_hip_pitch (range ±1.57)
        target[3] = 0.4  # left_knee (range -0.26..2.05)
        target[7] = -0.2  # right_hip_pitch
        target[8] = 0.4  # right_knee
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[2] == pytest.approx(-0.2, abs=5e-3)
        assert state.position[3] == pytest.approx(0.4, abs=5e-3)
        assert state.position[7] == pytest.approx(-0.2, abs=5e-3)
        assert state.position[8] == pytest.approx(0.4, abs=5e-3)

    def test_torso_converges_to_target(self, connected_hal: H1MujocoHAL) -> None:
        # Torso is a single joint at index 10 (yaw only on the H1, not
        # a 3-DoF waist like the G1).
        target = [0.0] * 19
        target[10] = 0.3
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[10] == pytest.approx(0.3, abs=5e-3)

    def test_sentinel_per_slot_identity_left_vs_right(self, connected_hal: H1MujocoHAL) -> None:
        """Per-slot identity — each sentinel value lands in the matching
        output slot, with opposite signs on the two sides so a future
        L/R swap surfaces immediately.
        """
        sentinel = [0.0] * 19
        # Mirror left/right with opposite signs where the joint ranges
        # support it (shoulder_roll is asymmetric: left ∈ [-0.34, 3.11]
        # vs right ∈ [-3.11, 0.34], so we mirror the sign accordingly).
        sentinel[11] = 0.3  # left_shoulder_pitch
        sentinel[15] = -0.3  # right_shoulder_pitch (mirrored)
        sentinel[12] = 0.5  # left_shoulder_roll (positive into range)
        sentinel[16] = -0.5  # right_shoulder_roll (positive in left,
        # negative on right is positive in their respective ranges)
        sentinel[13] = 0.4  # left_shoulder_yaw
        sentinel[17] = -0.4  # right_shoulder_yaw
        sentinel[14] = 0.6  # left_elbow
        sentinel[18] = 0.6  # right_elbow (elbow only flexes positive)
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[sentinel],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        for i in range(11, 19):
            assert state.position[i] == pytest.approx(sentinel[i], abs=5e-3)

    def test_multi_step_chunk_settles_at_last_waypoint(self, connected_hal: H1MujocoHAL) -> None:
        chunk = [[0.0] * 19 for _ in range(3)]
        # Step the left elbow through three intermediate waypoints.
        chunk[0][14] = 0.2
        chunk[1][14] = 0.4
        chunk[2][14] = 0.6
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=3,
                joint_targets=chunk,
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[14] == pytest.approx(0.6, abs=5e-3)


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: H1MujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 19

            target = [0.0] * 19
            target[14] = 0.3  # left elbow
            target[18] = 0.3  # right elbow
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position[14] == pytest.approx(0.3, abs=5e-3)
            assert state1.position[18] == pytest.approx(0.3, abs=5e-3)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    # NOTE: no "every-joint commanded to the same value" convergence
    # test — same physical reality as the G1 suite documents.  The H1's
    # asymmetric arm conventions mean any uniform-target pose is
    # self-colliding for one side or the other; the per-section
    # convergence tests above already prove the 19-DoF action contract
    # without driving the body into self-collision.
