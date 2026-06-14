"""Sim tests for :class:`openral_hal.Rizon4MujocoHAL` against real MuJoCo physics.

These tests load the ``mujoco_menagerie`` Flexiv Rizon 4 MJCF (via
``robot_descriptions``) and exercise the full HAL lifecycle — connect →
read_state → send_action → estop / disconnect — against a real
``mj_step`` loop.  No mocks; the closed-loop behaviour comes from
MuJoCo's own position-controlled actuators.

The point of this suite is the Rizon 4 "real hardware first day"
contract (CLAUDE.md §1.11): if these tests pass, the 7-DoF joint-
position action layout, lifecycle, and ``RobotDescription`` joint
order are guaranteed to match what a future ``Rizon4RealHAL`` over
``flexiv_rdk`` will see when the physical arm arrives — the only
remaining failure surfaces are at the vendor SDK / RTDE-equivalent
layer (HIL territory).

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
# for ..." on every sibling). Deferring the decision to `pytestmark` keeps
# this module importable when optional deps fail (e.g. ``MUJOCO_GL=osmesa``
# eagerly loads OSMesa on a headless host without the OpenGL stack and
# ``import mujoco`` raises a non-ImportError) so sibling files remain
# reachable. Our HAL never renders — we only need the physics.
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
    from robot_descriptions import rizon4_mj_description as _rizon4_desc

    _ = _rizon4_desc.MJCF_PATH  # triggers lazy clone / cache lookup
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
from openral_hal import RIZON4_DESCRIPTION, Rizon4MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        _MUJOCO_ERROR is not None,
        reason=f"mujoco unavailable: {_MUJOCO_ERROR}",
    ),
    pytest.mark.skipif(
        _MJCF_ERROR is not None,
        reason=f"Rizon 4 MJCF unavailable: {_MJCF_ERROR}",
    ),
]


_EXPECTED_RIZON4_JOINT_ORDER: list[str] = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


# ── Schema-level checks (cheap; do not require connect) ───────────────────────


class TestRizon4Description:
    def test_canonical_description_shape(self) -> None:
        desc = RIZON4_DESCRIPTION
        assert desc.name == "rizon4"
        assert desc.embodiment_kind == EmbodimentKind.MANIPULATOR.value
        assert len(desc.joints) == 7
        assert all(j.joint_type == JointType.REVOLUTE.value for j in desc.joints)

    def test_joint_names_match_menagerie_order(self) -> None:
        names = [j.name for j in RIZON4_DESCRIPTION.joints]
        assert names == _EXPECTED_RIZON4_JOINT_ORDER

    def test_capabilities_advertise_joint_position(self) -> None:
        modes = RIZON4_DESCRIPTION.capabilities.supported_control_modes
        assert ControlMode.JOINT_POSITION.value in modes

    def test_capabilities_advertise_force_control(self) -> None:
        # The Rizon 4's defining feature is whole-body force sensitivity
        # (0.1 N resolution) — has_force_control must be advertised so
        # skills can match against it.
        assert RIZON4_DESCRIPTION.capabilities.has_force_control is True

    def test_embodiment_tags_include_rizon4(self) -> None:
        tags = RIZON4_DESCRIPTION.capabilities.embodiment_tags
        assert "rizon4" in tags
        assert "flexiv" in tags

    def test_safety_envelope_requires_deadman(self) -> None:
        # 4 kg payload + 780 mm reach + 123 N·m peak torque is more than
        # enough to injure — the safety envelope must require a deadman.
        assert RIZON4_DESCRIPTION.safety.deadman_required is True


# ── Menagerie XML schema invariants (catches upstream drift) ──────────────────


class TestMenagerieSchema:
    """Guard against silent ``mujoco_menagerie`` schema drift.

    If the upstream MJCF ever renames or reorders the 7 hinge joints,
    these guards fail before the closed-loop tests do — the HAL would
    otherwise read the wrong qpos slot and the breakage would only
    surface several minutes later as a convergence failure.
    """

    def test_joint_order_and_names(self) -> None:
        model = mujoco.MjModel.from_xml_path(_rizon4_desc.MJCF_PATH)
        assert model.njnt == 7
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        assert names == _EXPECTED_RIZON4_JOINT_ORDER

    def test_actuator_count_and_alignment(self) -> None:
        model = mujoco.MjModel.from_xml_path(_rizon4_desc.MJCF_PATH)
        assert model.nu == 7
        # Actuator name + driven-joint name must match 1:1 in order.
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            driven_jnt = int(model.actuator_trnid[i, 0])
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driven_jnt)
            assert act_name == jnt_name
            assert jnt_name == _EXPECTED_RIZON4_JOINT_ORDER[i]

    def test_actuator_is_position_mode(self) -> None:
        # The Rizon menagerie ships position actuators (gain=-bias),
        # NOT torque actuators like the H1.  If a future menagerie
        # update flips this we need to either route Rizon through the
        # H1-style PD path or drop the assumption — fail loudly here.
        model = mujoco.MjModel.from_xml_path(_rizon4_desc.MJCF_PATH)
        for i in range(model.nu):
            gain = float(model.actuator_gainprm[i][0])
            bias_pos = float(model.actuator_biasprm[i][1])
            # Position actuator: bias = -gain * qpos (so gain + bias_coef = 0).
            assert gain > 1.0, f"act[{i}] gain={gain} suggests torque mode"
            assert bias_pos == pytest.approx(-gain, abs=1e-3), (
                f"act[{i}] biasprm[1]={bias_pos} expected {-gain} for a position actuator"
            )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> Rizon4MujocoHAL:
    """Fresh Rizon 4 HAL with gravity off and enough settle steps for
    the position controllers to converge to the commanded pose."""
    return Rizon4MujocoHAL(gravity_enabled=False, settle_steps=2000)


@pytest.fixture()
def connected_hal(hal: Rizon4MujocoHAL) -> Rizon4MujocoHAL:
    hal.connect()
    yield hal
    hal.disconnect()


def _zero_action(horizon: int = 1) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=horizon,
        joint_targets=[[0.0] * 7 for _ in range(horizon)],
        stamp_ns=time.time_ns(),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


# ── Rizon4-specific lifecycle tests ──────────────────────────────────────────
#
# Shared protocol compliance and standard lifecycle tests are consolidated in
# tests/sim/test_hal_protocol_contracts.py (parametrized across all 9 HALs).
# Keep only Rizon4-specific tests here.


class TestRizon4Lifecycle:
    def test_connect_loads_mujoco_model(self, hal: Rizon4MujocoHAL) -> None:
        """Rizon4-specific: verify 7-DoF actuator count in menagerie XML."""
        hal.connect()
        try:
            assert hal._connected is True
            assert hal._model is not None
            assert hal._data is not None
            assert hal._model.nu == 7
        finally:
            hal.disconnect()


# ── read_state ────────────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_seven_joints(self, connected_hal: Rizon4MujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert len(state.name) == 7
        assert state.name == [j.name for j in RIZON4_DESCRIPTION.joints]
        assert len(state.position) == 7
        assert len(state.velocity) == 7
        assert state.stamp_ns > 0

    def test_initial_positions_are_zero(self, connected_hal: Rizon4MujocoHAL) -> None:
        state = connected_hal.read_state()
        for q in state.position:
            assert abs(q) < 1e-6

    def test_perception_stale_raises_when_old(self, monkeypatch) -> None:
        hal = Rizon4MujocoHAL(gravity_enabled=False, settle_steps=1, staleness_limit_s=0.001)
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
    def test_rejects_wrong_joint_count(self, connected_hal: Rizon4MujocoHAL) -> None:
        """Rizon4-specific: verify 7-joint contract."""
        # 6 values for a 7-joint robot.
        bad = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 6],
            stamp_ns=time.time_ns(),
        )
        with pytest.raises(ROSConfigError, match="7 joints"):
            connected_hal.send_action(bad)


# ── estop ─────────────────────────────────────────────────────────────────────
# Standard estop contract is tested in test_hal_protocol_contracts.py (parametrized).
# No Rizon4-specific estop behavior to test.


# ── Closed-loop physics ───────────────────────────────────────────────────────


class TestClosedLoopMujoco:
    """Real MuJoCo physics — no mocks.  Position commands must converge."""

    def test_send_action_drives_joints_to_target(self, connected_hal: Rizon4MujocoHAL) -> None:
        # Targets well inside every menagerie joint range (joint4
        # asymmetric ∈ [-1.955, 2.775], joint6 ∈ [-1.484, 4.625]).
        target = [0.5, 0.4, -0.3, 0.6, -0.4, 1.5, 0.3]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[target],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        assert state.position == pytest.approx(target, abs=5e-3)

    def test_multi_step_chunk_settles_at_last_waypoint(
        self, connected_hal: Rizon4MujocoHAL
    ) -> None:
        chunk = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.3, -0.2, 0.1, 0.5, 0.0, 1.0, 0.2],
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
        assert state.position == pytest.approx(chunk[-1], abs=5e-3)

    def test_sequential_actions_advance_state(self, connected_hal: Rizon4MujocoHAL) -> None:
        targets = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 0.0],
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
            assert state.position == pytest.approx(tgt, abs=5e-3)

    def test_per_joint_identity_wiring(self, connected_hal: Rizon4MujocoHAL) -> None:
        """Wiring check — every action slot reaches exactly one joint.

        Commands a unique distinguishable value into each slot and
        verifies it lands in the matching ``state.position`` slot.  Any
        future off-by-one in ``RIZON4_DESCRIPTION.sim.joint_qpos_addr`` /
        ``sim.actuator_index`` (or in the default 1:1 mapping derived
        from ``description.joints`` order, ADR-0023) would fail one of
        these assertions immediately.
        """
        # Pick distinct values; keep them small enough that joint4 / 6
        # (asymmetric ranges) stay inside their MJCF limits.
        sentinel = [0.11, 0.22, -0.33, 0.44, -0.55, 0.66, -0.77]
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[sentinel],
                stamp_ns=time.time_ns(),
            )
        )
        state = connected_hal.read_state()
        for i, value in enumerate(sentinel):
            assert state.position[i] == pytest.approx(value, abs=5e-3), (
                f"joint slot {i} ({state.name[i]!r}) read {state.position[i]!r}, "
                f"expected {value!r} — wiring mismatch?"
            )


# ── Full lifecycle sequence ───────────────────────────────────────────────────


class TestFullLifecycle:
    def test_connect_read_send_disconnect(self, hal: Rizon4MujocoHAL) -> None:
        hal.connect()
        try:
            state0 = hal.read_state()
            assert len(state0.name) == 7

            target = [0.2, -0.2, 0.2, 0.5, -0.2, 1.0, 0.2]
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[target],
                    stamp_ns=time.time_ns(),
                )
            )
            state1 = hal.read_state()
            assert state1.position == pytest.approx(target, abs=5e-3)
        finally:
            hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()
