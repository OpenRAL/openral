"""Sim tests for :class:`openral_hal.G1MujocoHAL` against real MuJoCo physics.

These tests load the ``mujoco_menagerie`` Unitree G1 MJCF (via
``robot_descriptions``) and exercise the full HAL lifecycle — connect →
read_state → send_action → estop / disconnect — against a real
``mj_step`` loop.  No mocks; the closed-loop behaviour comes from
MuJoCo's own position-controlled actuators driving the 29-DoF humanoid.

The point of this suite is the G1 "real hardware first day" contract
(CLAUDE.md §1.11): if these tests pass, the 29-DoF joint-position
action layout, lifecycle, and ``RobotDescription`` joint order are
guaranteed to match what a future ``G1RealHAL`` over ``unitree_sdk2``
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
    from robot_descriptions import g1_mj_description as _g1_desc

    _ = _g1_desc.MJCF_PATH  # triggers lazy clone / cache lookup
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
from openral_hal import G1_DESCRIPTION, G1MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"G1 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# Expected joint name order — matches the menagerie MJCF and the
# in-code ``_G1_JOINT_NAMES`` tuple.  Keep these two ordered lists
# adjacent so any future drift surfaces in one diff.
_EXPECTED_G1_JOINT_ORDER: list[str] = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestG1Description:
    def test_canonical_description_shape(self) -> None:
        desc = G1_DESCRIPTION
        assert desc.name == "g1"
        assert desc.embodiment_kind == EmbodimentKind.HUMANOID.value
        assert len(desc.joints) == 29
        assert all(j.joint_type == JointType.REVOLUTE.value for j in desc.joints)

    def test_joint_names_match_menagerie_order(self) -> None:
        names = [j.name for j in G1_DESCRIPTION.joints]
        assert names == _EXPECTED_G1_JOINT_ORDER

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = G1_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_advertise_humanoid_traits(self) -> None:
        caps = G1_DESCRIPTION.capabilities
        assert "bipedal" in caps.locomotion
        assert caps.bimanual is True

    def test_embodiment_tags_include_g1(self) -> None:
        assert "g1" in G1_DESCRIPTION.capabilities.embodiment_tags
        assert "humanoid" in G1_DESCRIPTION.capabilities.embodiment_tags

    def test_safety_envelope_requires_deadman(self) -> None:
        # Per CLAUDE.md §1.1 — the G1 has enough mass + reach to injure
        # an operator; a deadman is mandatory at the safety envelope.
        assert G1_DESCRIPTION.safety.deadman_required is True

    def test_floating_base_is_not_in_description(self) -> None:
        # The MJCF floating_base_joint is implicit world state, not a
        # controllable joint — it must NOT appear in the description.
        names = [j.name for j in G1_DESCRIPTION.joints]
        assert "floating_base_joint" not in names


# ── Menagerie XML schema invariants (catches upstream drift) ──────────────────


class TestMenagerieSchema:
    """Guard against silent ``mujoco_menagerie`` schema drift.

    The :class:`G1MujocoHAL` indexing assumes the floating-base + 29
    actuated-joint order documented in ``g1.py`` — if a future
    menagerie upgrade reorders joints or splits the floating base into
    something else, these guards fail before the closed-loop tests do
    and the breakage points at the right place.
    """

    def test_joint_count_and_floating_base(self) -> None:
        model = mujoco.MjModel.from_xml_path(_g1_desc.MJCF_PATH)
        # 1 free joint (floating base) + 29 hinge joints.
        assert model.njnt == 30
        first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)
        assert first_name == "floating_base_joint"
        assert int(model.jnt_type[0]) == 0  # mjJNT_FREE

    def test_actuated_joint_order_matches_description(self) -> None:
        model = mujoco.MjModel.from_xml_path(_g1_desc.MJCF_PATH)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)
        ]
        assert names == _EXPECTED_G1_JOINT_ORDER

    def test_actuator_count_and_alignment(self) -> None:
        model = mujoco.MjModel.from_xml_path(_g1_desc.MJCF_PATH)
        assert model.nu == 29
        # Actuator name + driven-joint name must match 1:1 in order.
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            driven_jnt = int(model.actuator_trnid[i, 0])
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driven_jnt)
            assert act_name == jnt_name
            # And actuator index i drives the i-th actuated joint
            # (qpos addr = 7 + i because of the floating base).
            assert jnt_name == _EXPECTED_G1_JOINT_ORDER[i]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> G1MujocoHAL:
    """Fresh G1 HAL with gravity off and enough settle steps for the
    position controllers to converge to the commanded pose.

    Gravity is **always** disabled in these tests — without an S0
    cerebellum the floating-base humanoid falls in <1 s, which would
    couple every closed-loop assertion to body dynamics.
    """
    return G1MujocoHAL(gravity_enabled=False, settle_steps=3000)


@pytest.fixture()
def connected_hal(hal: G1MujocoHAL) -> G1MujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _zero_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * 29 for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── G1-specific lifecycle tests ───────────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only G1-specific tests here.


class TestG1Lifecycle:
    def test_connect_loads_mujoco_model(self, hal: G1MujocoHAL) -> None:
        """G1-specific: verify 29 actuated joints + floating base in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 29  # 29 actuated joints
            assert hal._model.njnt == 30  # +1 floating base
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_29_joints(self, connected_hal: G1MujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 29
        assert state.name == [j.name for j in G1_DESCRIPTION.joints]
        assert len(state.position) == 29
        assert len(state.velocity) == 29
        assert state.stamp_ns > 0

    def test_initial_positions_are_zero(self, connected_hal: G1MujocoHAL) -> None:
        # The menagerie MJCF defaults every actuated joint to qpos=0
        # (the upright neutral pose); no keyframe applied in connect.
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
        hal = G1MujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
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
    def test_rejects_wrong_joint_count(self, connected_hal: G1MujocoHAL) -> None:
        """G1-specific: verify 29-joint contract."""
        # 28 values for a 29-joint robot.
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 28],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="29 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No G1-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge
    across all 29 joints when gravity is off (without an S0 cerebellum
    the floating base falls instantly with gravity on — see the suite
    docstring)."""

    def test_send_action_holds_zero_pose(self, connected_hal: G1MujocoHAL) -> None:
        # Commanding zero on every actuator should leave every joint at
        # zero (the menagerie's default rest pose with gravity off).
        connected_hal.send_action(_zero_action())
        state = connected_hal.read_state()
        for i, q in enumerate(state.position):
            assert abs(q) < 5e-3, f"joint {state.name[i]!r} drifted to {q:.4f}"

    def test_left_arm_converges_to_target(self, connected_hal: G1MujocoHAL) -> None:
        target = [0.0] * 29
        # Left-arm joint indices in the 29-D action (per
        # _EXPECTED_G1_JOINT_ORDER): 15..21.
        target[15] = 0.4  # left_shoulder_pitch
        target[16] = 0.3  # left_shoulder_roll
        target[17] = 0.2  # left_shoulder_yaw
        target[18] = 0.6  # left_elbow
        target[19] = 0.1  # left_wrist_roll
        target[20] = -0.2  # left_wrist_pitch
        target[21] = 0.15  # left_wrist_yaw
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        # Left arm follows the command.
        assert state.position[15:22] == pytest.approx(target[15:22], abs=5e-3)
        # Right arm (indices 22..28) and legs (0..11) stay at zero.
        for i in list(range(0, 12)) + list(range(22, 29)):
            assert abs(state.position[i]) < 5e-3, f"joint {state.name[i]!r} moved"

    def test_right_arm_converges_to_target(self, connected_hal: G1MujocoHAL) -> None:
        target = [0.0] * 29
        # Right-arm joint indices: 22..28.
        target[22] = 0.4  # right_shoulder_pitch
        target[23] = -0.3  # right_shoulder_roll (opposite sign to left)
        target[24] = -0.2  # right_shoulder_yaw
        target[25] = 0.6  # right_elbow
        target[26] = -0.1  # right_wrist_roll
        target[27] = 0.2  # right_wrist_pitch
        target[28] = -0.15  # right_wrist_yaw
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[22:29] == pytest.approx(target[22:29], abs=5e-3)
        for i in range(0, 22):
            assert abs(state.position[i]) < 5e-3

    def test_legs_converge_to_target(self, connected_hal: G1MujocoHAL) -> None:
        # Small leg deflection — exercise the indexing for the lower
        # body without driving the robot into self-collision.  Left leg
        # = indices 0..5, right leg = indices 6..11.
        target = [0.0] * 29
        target[0] = -0.2  # left_hip_pitch
        target[3] = 0.4  # left_knee
        target[6] = -0.2  # right_hip_pitch
        target[9] = 0.4  # right_knee
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[0] == pytest.approx(-0.2, abs=5e-3)
        assert state.position[3] == pytest.approx(0.4, abs=5e-3)
        assert state.position[6] == pytest.approx(-0.2, abs=5e-3)
        assert state.position[9] == pytest.approx(0.4, abs=5e-3)

    def test_waist_converges_to_target(self, connected_hal: G1MujocoHAL) -> None:
        # Waist joints are indices 12..14.
        target = [0.0] * 29
        target[12] = 0.3  # waist_yaw
        target[13] = 0.2  # waist_roll
        target[14] = -0.2  # waist_pitch
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[12:15] == pytest.approx(target[12:15], abs=5e-3)

    def test_sentinel_per_slot_identity_left_vs_right(self, connected_hal: G1MujocoHAL) -> None:
        """Per-slot identity — each sentinel value lands in the matching
        output slot, with opposite signs on the two sides so a future
        L/R swap surfaces immediately.
        """
        sentinel = [0.0] * 29
        # Mirror left/right with opposite signs everywhere the joint
        # ranges allow it (most arm joints).
        for left_idx, right_idx, val in [
            (15, 22, 0.3),  # shoulder_pitch
            (16, 23, 0.2),  # shoulder_roll (left positive, right negative due to mirror)
            (17, 24, 0.4),  # shoulder_yaw
            (18, 25, 0.5),  # elbow
            (19, 26, 0.1),  # wrist_roll
            (20, 27, 0.15),  # wrist_pitch
            (21, 28, 0.25),  # wrist_yaw
        ]:
            sentinel[left_idx] = val
            sentinel[right_idx] = -val if right_idx != 25 else val  # elbow only flexes positive
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[sentinel],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        for i in range(15, 29):
            assert state.position[i] == pytest.approx(sentinel[i], abs=5e-3)

    def test_multi_step_chunk_settles_at_last_waypoint(self, connected_hal: G1MujocoHAL) -> None:
        chunk = [[0.0] * 29 for _ in range(3)]
        # Step the left elbow through three intermediate waypoints.
        chunk[0][18] = 0.2
        chunk[1][18] = 0.4
        chunk[2][18] = 0.6
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=3,
                joint_targets=chunk,
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[18] == pytest.approx(0.6, abs=5e-3)


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: G1MujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 29

            target = [0.0] * 29
            target[18] = 0.3  # left elbow
            target[25] = 0.3  # right elbow
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position[18] == pytest.approx(0.3, abs=5e-3)
            assert state1.position[25] == pytest.approx(0.3, abs=5e-3)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    # NOTE: there's intentionally no "every-joint commanded to the same
    # value" convergence test.  The G1 has asymmetric joint conventions
    # (e.g. right_shoulder_roll positive rotates the arm INTO the
    # torso while left_shoulder_roll positive rotates it AWAY), so any
    # uniform-target pose is self-colliding for one side or the other
    # and the contact forces pin a subset of joints — physical
    # reality, not a HAL bug, exactly the same situation ALOHA hits
    # at all-zeros (see test_aloha_bimanual_hal_mujoco.py's home-pose
    # comment).  The per-section convergence tests above already
    # prove the 29-DoF action contract end-to-end on a per-joint
    # basis without driving the body into self-collision.
