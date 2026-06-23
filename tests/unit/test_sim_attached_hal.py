"""Hermetic unit tests for :class:`openral_hal.sim_attached.SimAttachedHAL`.

CLAUDE.md §1.11 — real schemas + real Protocol exercise. The
:class:`SimRollout` Protocol boundary is satisfied by the test-only
:class:`tests.unit.fakes.fake_sim_env.FakeSimEnv` (a narrow recorder,
not a mock — see its docstring for which Protocol slice it
implements). The HAL Protocol is structural; what's under test is:

1. `pack_action_for_env` translates each supported ControlMode + row
   width correctly (and rejects the unsupported combinations cleanly).
2. `SimAttachedHAL` satisfies the HAL Protocol: connect/disconnect/
   read_state/send_action/estop semantics hold against a fake env.
3. `mujoco_handles()` forwards the env's handles when the env exposes
   them, returns None when it doesn't.

The robocasa-attached end-to-end exercise lives in
`tests/integration/test_panda_mobile_hal_lifecycle.py`.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from openral_core import (
    Action,
    ClockOrigin,
    ControlMode,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_hal.sim_attached import (
    SimAttachedHAL,
    pack_action_for_env,
)

from tests.unit.fakes.fake_sim_env import FakeSimEnv

# ── pack_action_for_env ──────────────────────────────────────────────


def _two_dof_description() -> RobotDescription:
    return RobotDescription(
        name="panda_mobile_stub",
        embodiment_kind="mobile_manipulator",
        joints=[
            JointSpec(
                name="base_x",
                joint_type=JointType.PRISMATIC,
                parent_link="world",
                child_link="base_x_link",
                sim_joint_name="mobilebase0_joint_mobile_forward",
            ),
            JointSpec(
                name="base_y",
                joint_type=JointType.PRISMATIC,
                parent_link="base_x_link",
                child_link="base_y_link",
                sim_joint_name="mobilebase0_joint_mobile_side",
            ),
            JointSpec(
                name="base_yaw",
                joint_type=JointType.REVOLUTE,
                parent_link="base_y_link",
                child_link="base_link",
                sim_joint_name="mobilebase0_joint_mobile_yaw",
            ),
            *[
                JointSpec(
                    name=f"panda_joint{i}",
                    joint_type=JointType.REVOLUTE,
                    parent_link=(f"panda_link{i - 1}" if i > 1 else "base_link"),
                    child_link=f"panda_link{i}",
                    sim_joint_name=f"robot0_joint{i}",
                )
                for i in range(1, 8)
            ],
        ],
        capabilities=RobotCapabilities(embodiment_tags=["panda_mobile"]),
        safety=SafetyEnvelope(),
        base_joints=["base_x", "base_y", "base_yaw"],
    )


def test_pack_action_body_twist_maps_planar_components() -> None:
    """BODY_TWIST [vx, vy, vz, wx, wy, wz] → env [vx, vy, wz, 0, 0, ...]."""
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.5, -0.25, 0.0, 0.0, 0.0, 0.75]],
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert out.shape == (11,)
    assert out.dtype == np.float32
    assert float(out[0]) == 0.5
    assert float(out[1]) == -0.25
    assert float(out[2]) == 0.75
    # Arm + gripper slots stay zero.
    assert np.all(out[3:] == 0.0)


def test_pack_action_joint_position_arm_only_fills_arm_slots() -> None:
    """JOINT_POSITION 7-vec lands at slots [3:10]."""
    arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chunk = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[arm],
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert out.shape == (11,)
    # Base slots stay 0.
    assert np.all(out[:3] == 0.0)
    assert list(out[3:10]) == pytest.approx(arm)
    # Gripper stays 0.
    assert float(out[10]) == 0.0


def test_pack_action_joint_position_full_fills_both() -> None:
    """JOINT_POSITION 10-vec drops the first 10 slots verbatim."""
    full = [0.5, 0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    chunk = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[full],
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert list(out[:10]) == pytest.approx(full)


def test_pack_action_cartesian_delta_packs_arm_slots() -> None:
    """ADR-0028c — CARTESIAN_DELTA fills slots [3:9] (arm OSC); base + gripper zero."""
    chunk = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.01, 0.02, 0.03, 0.1, 0.2, 0.3)],
        ee_name="panda_hand",
        frame_id="panda_link0",
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert out.shape == (11,)
    # Base (0-2) zero, arm OSC (3-8) populated, gripper (10) zero.
    assert list(out[:3]) == [0.0, 0.0, 0.0]
    assert list(out[3:9]) == pytest.approx([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])
    assert out[10] == 0.0


def test_pack_action_gripper_position_packs_last_slot() -> None:
    """ADR-0028c — GRIPPER_POSITION fills the trailing slot; arm + base zero."""
    chunk = Action(
        control_mode=ControlMode.GRIPPER_POSITION,
        horizon=1,
        gripper=[0.42],
        ee_name="panda_gripper",
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert out.shape == (11,)
    assert out[-1] == pytest.approx(0.42)
    assert all(v == 0.0 for v in out[:-1])


def test_pack_action_rejects_other_modes() -> None:
    """An unsupported mode raises ROSConfigError with the actual mode in the message."""
    chunk = Action(
        control_mode=ControlMode.JOINT_VELOCITY,
        horizon=1,
        joint_targets=[[0.1, 0.2, 0.3]],
    )
    with pytest.raises(ROSConfigError, match="JOINT_VELOCITY"):
        pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)


def test_pack_action_rejects_wrong_row_width() -> None:
    """Wrong row width for the declared mode is caught at Action construction.

    ADR-0028c — ``Action.body_twist: list[tuple[float, ..., 6]]`` enforces
    the 6-tuple shape at Pydantic-validation time, so a 3-wide body twist
    is rejected before ever reaching ``pack_action_for_env``. The HAL's
    runtime guard inside ``pack_action_for_env`` survives as defence in
    depth for paths that bypass schema validation.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[0.1, 0.2, 0.3]],  # 3 instead of 6 — Pydantic rejects
        )


# ── SimAttachedHAL — uses the FakeSimEnv at tests/unit/fakes/ ──────────


def test_sim_attached_hal_connect_resets_env_and_probes_action_dim() -> None:
    """`connect` resets the env at the requested seed and lets the next
    `send_action` route the chunk into the probed action_dim slot count.
    """
    env = FakeSimEnv(action_dim=11)
    hal = SimAttachedHAL(env, _two_dof_description(), env_reset_seed=42)
    hal.connect()
    assert env.reset_calls == [42]
    # Exercise the public contract: a JOINT_POSITION send_action call
    # after connect produces an env-shaped action vector. (BODY_TWIST
    # takes the direct-qpos path and skips env.step — covered
    # separately by ``test_body_twist_advances_mujoco_qpos``.)
    chunk = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * 7],
    )
    hal.send_action(chunk)
    assert env.last_action is not None
    assert env.last_action.shape == (11,)


def test_sim_attached_hal_send_action_packs_and_steps_env() -> None:
    """JOINT_POSITION still flows through ``env.step`` (BODY_TWIST does not — see below)."""
    env = FakeSimEnv(action_dim=11)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    chunk = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    )
    hal.send_action(chunk)
    assert env.step_calls == 1
    assert env.last_action is not None
    assert env.last_action.shape == (11,)
    # Arm targets land in slots [3:10].
    assert float(env.last_action[3]) == pytest.approx(0.1, abs=1e-6)
    assert float(env.last_action[9]) == pytest.approx(0.7, abs=1e-6)


def test_sim_attached_hal_estop_drops_subsequent_sends() -> None:
    """Estop is mode-agnostic — gates JOINT_POSITION env.step + BODY_TWIST qpos write."""
    env = FakeSimEnv(action_dim=11)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    hal.estop()
    chunk = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * 7],
    )
    hal.send_action(chunk)
    # Estopped: env.step was NOT called.
    assert env.step_calls == 0
    # Resetting the latch reenables actuation.
    hal.reset_estop()
    hal.send_action(chunk)
    assert env.step_calls == 1


def test_sim_attached_hal_read_state_without_handles_returns_zeros() -> None:
    """When the env can't supply MJCF handles, JointState is zero-padded.

    The shape must still match `description.joints` so the safety
    supervisor's per-row width check passes downstream.
    """
    env = FakeSimEnv(handles=None)
    desc = _two_dof_description()
    hal = SimAttachedHAL(env, desc)
    hal.connect()
    state = hal.read_state()
    assert len(state.position) == len(desc.joints)
    assert state.name == [j.name for j in desc.joints]
    assert all(p == 0.0 for p in state.position)


def test_sim_attached_hal_read_state_before_connect_raises() -> None:
    env = FakeSimEnv()
    hal = SimAttachedHAL(env, _two_dof_description())
    with pytest.raises(ROSRuntimeError, match="before connect"):
        hal.read_state()


def test_sim_attached_hal_send_action_before_connect_raises() -> None:
    env = FakeSimEnv()
    hal = SimAttachedHAL(env, _two_dof_description())
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.0] * 6],
    )
    with pytest.raises(ROSRuntimeError, match="before connect"):
        hal.send_action(chunk)


def test_sim_attached_hal_mujoco_handles_forwards_env_handles() -> None:
    """When the env exposes (model, data), the HAL forwards them verbatim."""
    sentinel_model = object()
    sentinel_data = object()
    env = FakeSimEnv(handles=(sentinel_model, sentinel_data))
    hal = SimAttachedHAL(env, _two_dof_description())
    handles = hal.mujoco_handles()
    assert handles is not None
    assert handles[0] is sentinel_model
    assert handles[1] is sentinel_data


# ── BODY_TWIST direct-qpos integrator (issue B fix) ──────────────────────────


_PLANAR_BASE_MJCF = """
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="mobilebase0_joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="mobilebase0_joint_mobile_side"    type="slide" axis="0 1 0"/>
      <joint name="mobilebase0_joint_mobile_yaw"     type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.2 0.2 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _build_planar_base_mjcf() -> tuple[object, object]:
    """Construct a real MuJoCo MjModel/MjData from a 3-joint MJCF.

    Mirrors the panda_mobile base layout (forward + side + yaw)
    that ``description.base_joints`` resolves through ``sim_joint_name``.
    Real MuJoCo — no faking — per CLAUDE.md §1.11.
    """
    import mujoco  # reason: optional dep, only needed when this test runs

    model = mujoco.MjModel.from_xml_string(_PLANAR_BASE_MJCF)
    data = mujoco.MjData(model)
    return model, data


def _qpos_addr(model: object, joint_name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])  # type: ignore[attr-defined]


def test_body_twist_advances_mujoco_qpos() -> None:
    """BODY_TWIST writes base qpos directly and advances the logical sim timestep."""
    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(
        action_dim=11,
        handles=(model, data),
        has_sim_clock=True,
        sim_time_from_mujoco=True,
    )
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()

    fwd_addr = _qpos_addr(model, "mobilebase0_joint_mobile_forward")
    side_addr = _qpos_addr(model, "mobilebase0_joint_mobile_side")
    yaw_addr = _qpos_addr(model, "mobilebase0_joint_mobile_yaw")
    assert float(data.qpos[fwd_addr]) == pytest.approx(0.0)

    # Drive forward at 1 m/s for one 0.05 s tick → forward qpos += 0.05.
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    hal.send_action(chunk)
    assert float(data.qpos[fwd_addr]) == pytest.approx(0.05, abs=1e-9)
    assert float(data.qpos[side_addr]) == pytest.approx(0.0, abs=1e-9)
    assert float(data.qpos[yaw_addr]) == pytest.approx(0.0, abs=1e-9)
    # Direct-qpos is still one simulation timestep. This is the deploy-sim
    # /clock contract: Nav2 BODY_TWIST streams must not freeze sim time just
    # because the base bypasses env.step.
    assert float(data.time) == pytest.approx(0.05, abs=1e-12)
    assert hal.sim_time_ns() == 50_000_000
    # Crucial: env.step was NOT called — the integrator bypasses the
    # robocasa composite controller (which doesn't honour planar
    # velocities in slots 0-2 on this scene).
    assert env.step_calls == 0


def test_body_twist_negative_x_moves_backward() -> None:
    """User's "move back 1 meter" → 20 ticks at vx=-1.0 advances by -1.0 m."""
    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(action_dim=11, handles=(model, data))
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()
    fwd_addr = _qpos_addr(model, "mobilebase0_joint_mobile_forward")

    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    for _ in range(20):
        hal.send_action(chunk)
    # 20 * (-1.0 * 0.05) = -1.0 m.
    assert float(data.qpos[fwd_addr]) == pytest.approx(-1.0, abs=1e-9)


def test_body_twist_rotates_body_to_world_by_yaw() -> None:
    """Body-frame vx with non-zero yaw lands as a world-frame xy delta."""
    import math

    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(action_dim=11, handles=(model, data))
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()

    fwd_addr = _qpos_addr(model, "mobilebase0_joint_mobile_forward")
    side_addr = _qpos_addr(model, "mobilebase0_joint_mobile_side")
    yaw_addr = _qpos_addr(model, "mobilebase0_joint_mobile_yaw")

    # Set yaw = +π/2 (robot facing world +y).
    data.qpos[yaw_addr] = math.pi / 2.0

    # Body-frame "forward" (vx=1) should advance world-frame +y.
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    hal.send_action(chunk)
    assert float(data.qpos[fwd_addr]) == pytest.approx(0.0, abs=1e-9)
    assert float(data.qpos[side_addr]) == pytest.approx(0.05, abs=1e-9)


def test_body_twist_yaw_wraps_to_pm_pi() -> None:
    """Long sessions don't accumulate yaw drift — wrap to [-π, π]."""
    import math

    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(action_dim=11, handles=(model, data))
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()
    yaw_addr = _qpos_addr(model, "mobilebase0_joint_mobile_yaw")

    # Pre-set yaw near +π so one wz tick wraps it past.
    data.qpos[yaw_addr] = math.pi - 0.01
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],  # wz = 1 rad/s
    )
    hal.send_action(chunk)
    yaw_after = float(data.qpos[yaw_addr])
    # 0.01 + 0.05 = 0.06 over +π → wrapped to roughly -π + 0.04.
    assert -math.pi <= yaw_after <= math.pi
    assert yaw_after == pytest.approx(-math.pi + 0.04, abs=1e-9)


def test_body_twist_refreshes_base_pose_6dof_for_odom() -> None:
    """Regression: a BODY_TWIST must refresh ``base_pose_6dof`` (the /odom source).

    The bug this guards against: ``_apply_body_twist_to_qpos`` writes the
    base qpos and calls ``refresh_obs``, but the refresh-merge dropped the
    ``raw_proprio`` block — so ``base_pose_6dof`` (which the panda_mobile
    lifecycle node publishes as ``/odom`` + ``odom → base_link``) kept
    returning the connect-time pose. The base physically moved while
    ``/odom`` reported it standing still, breaking Nav2's feedback loop:
    on a "move backwards" command the robot drove in circles, never
    seeing progress toward the goal. With the merge in place, ``/odom``
    tracks the base again.
    """
    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(
        action_dim=11,
        handles=(model, data),
        emit_proprio=True,
        base_joint_names=(
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_yaw",
        ),
        base_z=0.7,
    )
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()

    # Connect-time pose: base at origin, platform height 0.70.
    pose0 = hal.base_pose_6dof()
    assert pose0 is not None
    (x0, y0, z0), _quat0 = pose0
    assert (x0, y0, z0) == pytest.approx((0.0, 0.0, 0.7), abs=1e-9)

    # Back up at 1 m/s for 10 ticks → forward qpos = -0.5 m.
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    for _ in range(10):
        hal.send_action(chunk)

    pose1 = hal.base_pose_6dof()
    assert pose1 is not None
    (x1, y1, z1), _quat1 = pose1
    # The /odom-facing pose tracks the moved base — NOT frozen at 0.0.
    assert x1 == pytest.approx(-0.5, abs=1e-9)
    assert y1 == pytest.approx(0.0, abs=1e-9)
    assert z1 == pytest.approx(0.7, abs=1e-9)


def test_base_twist_latches_command_for_odom() -> None:
    """``base_twist`` (the /odom twist source) tracks the last BODY_TWIST."""
    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(action_dim=11, handles=(model, data))
    hal = SimAttachedHAL(env, _two_dof_description(), body_twist_dt_s=0.05)
    hal.connect()

    # Idle before any command.
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[-0.3, 0.1, 0.0, 0.0, 0.0, 0.2]],
        )
    )
    # vz / wx / wy are forced to 0 (planar base); vx, vy, wz pass through.
    assert hal.base_twist == pytest.approx((-0.3, 0.1, 0.0, 0.0, 0.0, 0.2))

    # A non-BODY_TWIST action clears the latch — the base is no longer
    # velocity-commanded, so /odom must not report a stale velocity.
    hal.send_action(
        Action(
            control_mode=ControlMode.CARTESIAN_DELTA,
            horizon=1,
            cartesian_delta=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
    )
    assert hal.base_twist == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_body_twist_rejects_non_planar_components() -> None:
    """Non-zero linear-z / angular-x / angular-y in a 6-vec → ROSConfigError."""
    model, data = _build_planar_base_mjcf()
    env = FakeSimEnv(action_dim=11, handles=(model, data))
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]],  # non-zero vz
    )
    with pytest.raises(ROSConfigError, match="holonomic planar"):
        hal.send_action(chunk)


def test_body_twist_without_mujoco_handles_steps_env() -> None:
    """BODY_TWIST on a non-MuJoCo env (handles=None) with base dims routes through
    ``env.step`` (ADR-0045): the scene integrates the base inside the step, so the
    HAL packs ``(vx, vy, wz)`` into the FINAL three action slots (zeroing the
    arm/gripper) instead of the MuJoCo direct-qpos path."""
    env = FakeSimEnv(action_dim=11, handles=None)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    hal.send_action(
        Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[[0.5, -0.2, 0.0, 0.0, 0.0, 0.3]],  # vx, vy, _, _, _, wz
        )
    )
    assert env.last_action is not None
    # Base twist in the final 3 slots; arm/gripper slots held at zero.
    assert tuple(round(float(x), 3) for x in env.last_action[-3:]) == (0.5, -0.2, 0.3)
    assert all(float(x) == 0.0 for x in env.last_action[:-3])
    # Latched for the /odom publisher (vx, vy, vz, wx, wy, wz).
    assert hal.base_twist == (0.5, -0.2, 0.0, 0.0, 0.0, 0.3)


def test_body_twist_without_base_dims_raises_runtime_error() -> None:
    """A non-MuJoCo env too small to carry a base twist (``env_action_dim < 3``,
    i.e. the robot declares no planar base) → ROSRuntimeError, never silent."""
    from openral_core.exceptions import ROSRuntimeError

    env = FakeSimEnv(action_dim=2, handles=None)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    chunk = Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    with pytest.raises(ROSRuntimeError, match="at least 3 slots"):
        hal.send_action(chunk)


# ── ADR-0048 Phase 1 — sim_time_ns accessor + cross-reset offset ─────────


def _joint_position_chunk() -> Action:
    """A 7-DoF arm-only JOINT_POSITION chunk the FakeSimEnv steps with."""
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * 7],
    )


def test_sim_time_ns_none_for_clockless_backend() -> None:
    """A wrapped rollout without a sim clock → SimAttachedHAL.sim_time_ns is None."""
    env = FakeSimEnv(action_dim=11, has_sim_clock=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    assert hal.sim_time_ns() is None
    assert hal.clock_authority().origin is ClockOrigin.HOST_WALL
    # Stepping does not conjure a clock.
    hal.send_action(_joint_position_chunk())
    assert hal.sim_time_ns() is None


def test_sim_time_ns_advances_with_steps() -> None:
    """sim_time_ns is monotonic non-decreasing across several env.step calls."""
    env = FakeSimEnv(action_dim=11, has_sim_clock=True, sim_dt_ns=20_000_000)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    samples = [hal.sim_time_ns()]
    for _ in range(5):
        hal.send_action(_joint_position_chunk())
        samples.append(hal.sim_time_ns())
    assert all(s is not None for s in samples)
    # Non-decreasing (the fake adds sim_dt_ns per step) and never backwards.
    for prev, cur in pairwise(samples):
        assert cur >= prev  # type: ignore[operator]  # reason: None ruled out above
    assert samples[-1] == 20_000_000 * 5  # deterministic fake clock
    authority = hal.clock_authority()
    assert authority.origin is ClockOrigin.SIMULATION
    assert authority.publishes_ros_clock is True


def test_sim_time_ns_does_not_rewind_across_reset() -> None:
    """The cross-reset offset keeps sim_time_ns monotonic across env.reset.

    The FakeSimEnv rewinds its per-episode clock to 0 on ``reset`` (modelling
    robocasa's ``MjData.time``). Drive the ADR-0036 auto-reset via the
    ``_episode_done`` latch and assert the HAL-published value never goes back.
    """
    env = FakeSimEnv(action_dim=11, has_sim_clock=True, sim_dt_ns=20_000_000)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for _ in range(4):
        hal.send_action(_joint_position_chunk())
    pre_reset = hal.sim_time_ns()
    assert pre_reset == 20_000_000 * 4

    # Simulate the prior step having reported terminal so the next send_action
    # resets the env (which rewinds the fake's per-episode clock to 0) before
    # stepping. The offset must absorb the pre-reset elapsed time.
    hal._episode_done = True  # white-box latch set (mirrors the idle_step test)
    hal.send_action(_joint_position_chunk())
    post_reset = hal.sim_time_ns()

    assert post_reset is not None and pre_reset is not None
    # No rewind: offset(=80ms from the finished episode) + 20ms fresh step.
    assert post_reset >= pre_reset
    assert post_reset == 20_000_000 * 4 + 20_000_000


def test_sim_time_ns_monotonic_across_reconnect() -> None:
    """A re-connect (configure→cleanup cycle) folds elapsed time into the offset."""
    env = FakeSimEnv(action_dim=11, has_sim_clock=True, sim_dt_ns=20_000_000)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for _ in range(3):
        hal.send_action(_joint_position_chunk())
    before_reconnect = hal.sim_time_ns()
    assert before_reconnect == 20_000_000 * 3

    hal.connect()  # idempotent re-reset rewinds the fake clock to 0
    after_reconnect = hal.sim_time_ns()
    assert after_reconnect is not None and before_reconnect is not None
    # The re-connect reset folded the 60ms elapsed into the offset; the fresh
    # episode starts at 0, so the published value is unchanged (no rewind).
    assert after_reconnect >= before_reconnect
    assert after_reconnect == 20_000_000 * 3
