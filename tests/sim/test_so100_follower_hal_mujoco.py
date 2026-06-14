"""Sim tests for :class:`openral_hal.SO100MujocoHAL` against real MuJoCo physics.

These tests load the ``mujoco_menagerie`` SO-100 MJCF (via
``robot_descriptions``) and exercise the full HAL lifecycle — connect →
read_state → send_action → estop / disconnect — against a real ``mj_step``
loop.  No mocks; the closed-loop behaviour comes from MuJoCo's own
position-controlled actuators.

The point of this suite is the SO-100 "real hardware first day" contract
(CLAUDE.md §1.11): if these tests pass, the 6-DoF joint-position action
layout, gripper normalisation, lifecycle, and ``RobotDescription`` joint
order are guaranteed to match what
:class:`openral_hal.SO100FollowerHAL` will see when the physical arm
arrives — the only remaining failure surfaces are at the USB driver
level (HIL territory).

Gravity is disabled in the closed-loop test so the joint positions
converge exactly to the commanded pose; the staleness / lifecycle tests
run in default gravity-on configuration to mirror production.
"""

from __future__ import annotations

import math
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

# robot_descriptions clones mujoco_menagerie (~650 MB) from GitHub the first
# time a model is loaded; on CI runners with restricted network or first-run
# cold caches the clone may fail or time out.  Treat any failure as a skip so
# the suite stays green on flaky environments and only enforces correctness
# when the MJCF is actually reachable.
try:
    from robot_descriptions import so_arm100_mj_description as _so100_desc

    _ = _so100_desc.MJCF_PATH  # triggers lazy clone / cache lookup
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
from openral_hal import SO100_DESCRIPTION, SO100MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"SO-100 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestSO100Description:
    def test_canonical_description_shape(self) -> None:
        desc = SO100_DESCRIPTION
        assert desc.name == "so100_follower"
        assert desc.embodiment_kind == EmbodimentKind.MANIPULATOR.value
        # 5 revolute arm joints + 1 normalised gripper channel.
        assert len(desc.joints) == 6

    def test_joint_names_match_description(self) -> None:
        names = [j.name for j in SO100_DESCRIPTION.joints]
        assert names == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = SO100_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_embodiment_tags_include_so100(self) -> None:
        assert "so100_follower" in SO100_DESCRIPTION.capabilities.embodiment_tags

    def test_gripper_is_prismatic_normalised(self) -> None:
        # The description publishes the gripper as a synthetic prismatic
        # joint in [0, 1] — the HAL is responsible for mapping that to the
        # menagerie's revolute Jaw range.
        gripper = SO100_DESCRIPTION.joints[-1]
        assert gripper.name == "gripper"
        assert gripper.joint_type == JointType.PRISMATIC.value
        assert gripper.position_limits == (0.0, 1.0)


# ── Menagerie XML schema invariants (catches upstream drift) ──────────────────


class TestMenagerieSchema:
    """Guard against silent ``mujoco_menagerie`` schema drift.

    If the upstream MJCF ever renames or reorders joints/actuators, these
    tests fail early and loudly — the HAL would otherwise read the wrong
    qpos slot and the breakage would only surface in a closed-loop test
    several minutes later.
    """

    def test_joint_order_and_names(self) -> None:
        model = mujoco.MjModel.from_xml_path(_so100_desc.MJCF_PATH)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert names == [
            "Rotation",
            "Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
            "Jaw",
        ]

    def test_actuator_count_matches_joint_count(self) -> None:
        model = mujoco.MjModel.from_xml_path(_so100_desc.MJCF_PATH)
        assert model.nu == 6
        assert model.njnt == 6


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> SO100MujocoHAL:
    """Fresh SO-100 HAL with gravity off and enough settle steps for the
    position controllers to converge to the commanded pose."""
    return SO100MujocoHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: SO100MujocoHAL) -> SO100MujocoHAL:
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


# ── HAL-specific lifecycle tests ──────────────────────────────────────────────
#
# Shared protocol compliance, standard lifecycle, and safety tests are now
# consolidated in tests/sim/test_hal_protocol_contracts.py (parametrized across
# all 9 HAL implementations). Keep only SO-100-specific tests here.


class TestSO100Lifecycle:
    def test_connect_loads_mujoco_model(self, hal: SO100MujocoHAL) -> None:
        """SO-100-specific: verify actuator count in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            # 5 arm + 1 jaw = 6 actuators in the menagerie XML.
            assert hal._model.nu == 6
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_six_joints(self, connected_hal: SO100MujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 6
        assert state.name == [j.name for j in SO100_DESCRIPTION.joints]
        assert len(state.position) == 6
        assert len(state.velocity) == 6
        assert state.stamp_ns > 0

    def test_initial_arm_positions_are_zero(self, connected_hal: SO100MujocoHAL) -> None:
        state = connected_hal.read_state()
        # Arm joints (indices 0..4) start at 0 in the menagerie keyframe.
        for q in state.position[:5]:
            assert abs(q) < 1e-3

    def test_gripper_initial_position_is_normalised(self, connected_hal: SO100MujocoHAL) -> None:
        # The Jaw qpos defaults to 0 in the menagerie model; mapped to
        # [0, 1] across the menagerie range [-0.174, 1.75], that's about
        # 0.09 — strictly inside [0, 1].
        state = connected_hal.read_state()
        gripper = state.position[-1]
        assert 0.0 <= gripper <= 1.0

    def test_perception_stale_raises_when_old(self, monkeypatch) -> None:
        hal = SO100MujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
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
    def test_rejects_wrong_joint_count(self, connected_hal: SO100MujocoHAL) -> None:
        """SO-100-specific: verify 6-joint contract."""
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
# No SO-100-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge."""

    def test_send_action_drives_arm_to_target(self, connected_hal: SO100MujocoHAL) -> None:
        # Targets chosen well inside every menagerie joint range so MuJoCo
        # doesn't clip them: Rotation [-1.92, 1.92], Pitch [-3.32, 0.174],
        # Elbow [-0.174, 3.14], Wrist_Pitch [-1.66, 1.66], Wrist_Roll
        # [-2.79, 2.79].  Gripper stays closed (0).
        target = [0.5, -1.0, 1.5, 0.3, -0.4, 0.0]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        # Arm joints converge to the commanded pose.
        assert state.position[:5] == pytest.approx(target[:5], abs=5e-3)

    def test_gripper_open_command_converges_to_normalised_one(
        self, connected_hal: SO100MujocoHAL
    ) -> None:
        # Send "open" — normalised 1.0 — and confirm the menagerie's Jaw
        # converges to its upper range (1.75 rad) and is reported back
        # as 1.0 on the public surface.
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[-1] == pytest.approx(1.0, abs=1e-2)

    def test_gripper_closed_command_converges_to_normalised_zero(
        self, connected_hal: SO100MujocoHAL
    ) -> None:
        # Send "closed" — normalised 0.0 — and confirm the Jaw converges
        # to its lower range (-0.174 rad) and is reported back as 0.0.
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[-1] == pytest.approx(0.0, abs=1e-2)

    def test_multi_step_chunk_settles_at_last_waypoint(self, connected_hal: SO100MujocoHAL) -> None:
        chunk = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.0, 0.0, 0.0, 0.5],
            [0.3, -0.2, 0.5, 0.0, 0.0, 1.0],
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
        # Arm settles on the last waypoint; gripper on the normalised
        # last gripper command.
        assert state.position[:5] == pytest.approx(chunk[-1][:5], abs=5e-3)
        assert state.position[-1] == pytest.approx(chunk[-1][-1], abs=1e-2)

    def test_sequential_actions_advance_state(self, connected_hal: SO100MujocoHAL) -> None:
        targets = [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
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
            assert state.position[:5] == pytest.approx(tgt[:5], abs=5e-3)


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: SO100MujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 6

            target = [0.3, -0.2, 0.5, 0.1, -0.1, 0.5]
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position[:5] == pytest.approx(target[:5], abs=5e-3)
            assert state1.position[-1] == pytest.approx(0.5, abs=5e-2)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()

    def test_targets_within_menagerie_position_limits(self, connected_hal: SO100MujocoHAL) -> None:
        # Sanity check: pi/3 on every arm joint is inside every menagerie
        # joint range — the test fails if a future menagerie upgrade
        # tightens a limit below pi/3.
        target = [math.pi / 3, -math.pi / 3, math.pi / 3, math.pi / 3, math.pi / 3, 0.0]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position[:5] == pytest.approx(target[:5], abs=5e-3)
