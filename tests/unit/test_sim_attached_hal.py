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

import json
import logging
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import structlog
from openral_core import (
    Action,
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    ClockOrigin,
    ControlMode,
    JointSpec,
    JointType,
    Pose6D,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SceneSpec,
    TaskSpec,
)
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_hal import sim_attached
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


def test_deploy_sim_ignores_success_and_terminal_without_resetting() -> None:
    env = FakeSimEnv(
        action_dim=11,
        step_info={"is_success": True},
        step_terminated=True,
    )
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for tick in (1, 2):
        hal.send_action(
            Action(
                control_mode=ControlMode.CARTESIAN_DELTA,
                cartesian_delta=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                tick_index=tick,
                tick_group_size=2,
            )
        )
        hal.send_action(
            Action(
                control_mode=ControlMode.GRIPPER_POSITION,
                gripper=[-1.0],
                tick_index=tick,
                tick_group_size=2,
            )
        )
    assert env.step_calls == 2
    assert env.reset_calls == [None]
    # The HAL READS the backend's success predicate (for the
    # ``sim.task_success`` signal) but never ACTS on it: no reset, no
    # terminal synthesis, no effect on the step count. ``step_info``'s
    # ``is_success`` is likewise carried, never interpreted.
    assert env.task_success_value is None
    assert hal.task_success() is None


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
    """CARTESIAN_DELTA fills slots [3:9] (arm OSC); base + gripper zero."""
    chunk = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        horizon=1,
        cartesian_delta=[(0.01, 0.02, 0.03, 0.1, 0.2, 0.3)],
        cartesian_delta_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        ee_name="panda_hand",
        frame_id="panda_link0",
    )
    out = pack_action_for_env(chunk, _two_dof_description(), env_action_dim=11)
    assert out.shape == (11,)
    # Base (0-2) zero, arm OSC (3-8) populated, gripper (10) zero.
    assert list(out[:3]) == [0.0, 0.0, 0.0]
    assert list(out[3:9]) == pytest.approx([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])
    assert out[10] == 0.0


def test_multi_slot_tick_commits_one_atomic_env_step() -> None:
    env = FakeSimEnv(action_dim=11)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    cart = Action(
        control_mode=ControlMode.CARTESIAN_DELTA,
        cartesian_delta=[(0.2, -0.1, 0.3, 0.0, 0.1, -0.2)],
        tick_index=1,
        tick_group_size=2,
    )
    grip = Action(
        control_mode=ControlMode.GRIPPER_POSITION,
        gripper=[-1.0],
        tick_index=1,
        tick_group_size=2,
    )
    hal.send_action(cart)
    assert env.step_calls == 0
    assert hal.last_committed_tick == 0
    hal.send_action(grip)
    assert env.step_calls == 1
    assert hal.last_committed_tick == 1
    assert env.last_action is not None
    assert list(env.last_action[3:9]) == pytest.approx(list(cart.cartesian_delta[0]))
    assert env.last_action[-1] == pytest.approx(-1.0)


def test_pack_action_gripper_position_packs_last_slot() -> None:
    """GRIPPER_POSITION fills the trailing slot; arm + base zero."""
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

    ``Action.body_twist: list[tuple[float, ..., 6]]`` enforces
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


def test_sim_attached_hal_notifies_post_step_observer_once_per_env_step() -> None:
    env = FakeSimEnv(action_dim=11)
    hal = SimAttachedHAL(env, _two_dof_description())
    observed_steps: list[int] = []

    def observe() -> None:
        observed_steps.append(env.step_calls)

    hal.add_post_step_observer(observe)
    hal.add_post_step_observer(observe)
    hal.connect()

    hal.send_action(_joint_position_chunk())

    assert observed_steps == [1]


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


_ATTACHMENT_MJCF = """
<mujoco model="attached_objects">
  <worldbody>
    <body name="payload_a">
      <geom type="box" size="0.1 0.02 0.02"/>
      <body name="payload_a_child">
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
    <body name="payload_b">
      <geom type="sphere" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


def _attached_object(object_id: str, body_name: str) -> AttachedCollisionObject:
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.1, 0.02, 0.02)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            )
        ],
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.1),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="panda_link7",
        ),
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"mujoco_body:{body_name}",
        stamp_ns=1,
    )


def test_multiple_attached_bodies_are_resolved_atomically() -> None:
    """Multipart and multiple payload bodies enter one exact depth mask."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_ATTACHMENT_MJCF)
    data = mujoco.MjData(model)
    hal = SimAttachedHAL(
        FakeSimEnv(handles=(model, data)),
        _two_dof_description(),
    )

    hal.update_attached_objects(
        [
            _attached_object("baguette_seed1", "payload_a"),
            _attached_object("cabinet_handle_tool", "payload_b"),
        ]
    )

    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in hal.read_attached_body_ids()
    }
    assert names == {"payload_a", "payload_a_child", "payload_b"}
    assert [obj.object_id for obj in hal.read_attached_objects()] == [
        "baguette_seed1",
        "cabinet_handle_tool",
    ]


def test_invalid_attachment_update_preserves_previous_snapshot() -> None:
    """Unknown sim identity fails before replacing the active payload set."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_ATTACHMENT_MJCF)
    data = mujoco.MjData(model)
    hal = SimAttachedHAL(
        FakeSimEnv(handles=(model, data)),
        _two_dof_description(),
    )
    valid = _attached_object("baguette_seed1", "payload_a")
    hal.update_attached_objects([valid])

    with pytest.raises(ROSConfigError, match="unknown MuJoCo body"):
        hal.update_attached_objects([_attached_object("cabinet_handle_tool", "missing_payload")])

    assert hal.read_attached_objects() == [valid]


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
    ``env.step``: the scene integrates the base inside the step, so the
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


# ── Phase 1 — sim_time_ns accessor + cross-reset offset ──────────────────


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


def test_sim_time_ns_does_not_rewind_across_lifecycle_reconnect() -> None:
    """The cross-reset offset keeps sim_time_ns monotonic across reconnect.

    The FakeSimEnv rewinds its per-episode clock to 0 on ``reset`` (modelling
    robocasa's ``MjData.time``). A lifecycle reconnect may explicitly reset;
    deploy-sim action/terminal handling never does.
    """
    env = FakeSimEnv(action_dim=11, has_sim_clock=True, sim_dt_ns=20_000_000)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for _ in range(4):
        hal.send_action(_joint_position_chunk())
    pre_reset = hal.sim_time_ns()
    assert pre_reset == 20_000_000 * 4

    hal.connect()
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


# ── Task-success signal (observability only) ─────────────────────────────
#
# `deploy sim` suppresses the backend's own per-step task evaluation, so
# until this signal existed nothing in the deploy stack said whether the
# scene's task (e.g. "cup placed in the sink") had actually been completed
# — a validation run could prove attach/detach but not placement. These
# tests pin the emitted lines' shape and, critically, that reading the
# predicate changes no control flow.


class _CaptureProcessor:
    """Real ``structlog`` processor that buffers events for assertion.

    Same pattern as `tests/unit/test_diagnostics_phase_timer.py`: a real
    processor in the real pipeline (not a mock logger), dropping the event
    at the end so test logs don't pollute pytest output.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(
        self, logger: object, method: str, event_dict: dict[str, object]
    ) -> dict[str, object]:
        del logger, method
        name = str(event_dict.pop("event", ""))
        self.events.append((name, dict(event_dict)))
        raise structlog.DropEvent

    def named(self, event: str) -> list[dict[str, object]]:
        """Every captured payload logged under ``event``."""
        return [payload for name, payload in self.events if name == event]


@pytest.fixture
def cap() -> Any:
    """Install a fresh capture processor; restore structlog defaults after."""
    proc = _CaptureProcessor()
    structlog.reset_defaults()
    structlog.configure(processors=[proc])
    try:
        yield proc
    finally:
        structlog.reset_defaults()


def _success_env(**kwargs: Any) -> FakeSimEnv:
    """A FakeSimEnv carrying a real SceneSpec/TaskSpec + a sim clock.

    Real ``openral_core`` schemas and a real scene id from the shipped
    RoboCasa catalogue; the task id is the synthesised deploy-sim noop
    suffix ``openral_hal.sim_bringup`` builds (CLAUDE.md §1.11 — no
    ``"foo"`` placeholders).
    """
    return FakeSimEnv(
        action_dim=11,
        has_sim_clock=True,
        sim_dt_ns=20_000_000,
        scene=SceneSpec(id="robocasa/PickPlaceCounterToSink", backend="mujoco"),
        task=TaskSpec(
            id="robocasa/PickPlaceCounterToSink/_hal_deploy_noop",
            scene_id="robocasa/PickPlaceCounterToSink",
        ),
        **kwargs,
    )


def test_task_success_accessor_forwards_backend_verdict() -> None:
    """`task_success()` forwards the wrapped rollout's predicate verbatim."""
    env = _success_env()
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    assert hal.task_success() is None  # backend reports "no verdict"
    env.task_success_value = False
    assert hal.task_success() is False
    env.task_success_value = True
    assert hal.task_success() is True


def test_task_success_logs_the_false_to_true_transition(cap: _CaptureProcessor) -> None:
    """One `sim.task_success` line at the rising edge, with sim time + ids."""
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    hal.send_action(_joint_position_chunk())
    assert cap.named("sim.task_success") == []  # still False → silent

    env.task_success_value = True
    hal.send_action(_joint_position_chunk())

    lines = cap.named("sim.task_success")
    assert len(lines) == 1
    line = lines[0]
    assert line["success"] is True
    assert line["previous"] is False
    assert line["first_success"] is True
    assert line["scene_id"] == "robocasa/PickPlaceCounterToSink"
    assert line["task_id"] == "robocasa/PickPlaceCounterToSink/_hal_deploy_noop"
    assert line["sim_time_ns"] == 20_000_000 * 2
    assert line["sim_time_s"] == 0.04
    assert line["step"] == 2


def test_task_success_logs_both_edges_and_marks_only_the_first_success(
    cap: _CaptureProcessor,
) -> None:
    """RoboCasa success is not latched — a task undone must be visible too."""
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for value in (True, False, True):
        env.task_success_value = value
        hal.send_action(_joint_position_chunk())

    lines = cap.named("sim.task_success")
    assert [bool(line["success"]) for line in lines] == [True, False, True]
    # `first_success` marks the first rising edge only, so "did it ever
    # complete" stays answerable after a later regression.
    assert [bool(line["first_success"]) for line in lines] == [True, False, False]


def test_task_success_repeated_true_reads_log_once(cap: _CaptureProcessor) -> None:
    """A held-True predicate logs on the edge, not once per 20 Hz step."""
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    env.task_success_value = True
    for _ in range(10):
        hal.send_action(_joint_position_chunk())
    assert len(cap.named("sim.task_success")) == 1


def test_task_success_already_true_at_connect_is_the_baseline_not_a_transition(
    cap: _CaptureProcessor,
) -> None:
    """A scene that starts satisfied logs no rising edge, but still a final verdict."""
    env = _success_env(task_success_value=True)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    hal.send_action(_joint_position_chunk())
    assert cap.named("sim.task_success") == []
    hal.disconnect()
    final = cap.named("sim.task_success_final")[0]
    assert final["success"] is True
    # It was never OBSERVED to complete — the scene was handed over that
    # way — so the signal does not claim a first-success moment it never saw.
    assert final["ever_succeeded"] is False
    assert final["transitions"] == 0


def test_task_success_final_line_states_the_terminal_verdict(cap: _CaptureProcessor) -> None:
    """`disconnect` closes the session with one terminal success statement."""
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    env.task_success_value = True
    hal.send_action(_joint_position_chunk())
    env.task_success_value = False
    hal.send_action(_joint_position_chunk())
    hal.disconnect()

    finals = cap.named("sim.task_success_final")
    assert len(finals) == 1
    final = finals[0]
    # Ended NOT successful, but did succeed once — both facts survive.
    assert final["success"] is False
    assert final["ever_succeeded"] is True
    assert final["first_success_sim_time_ns"] == 20_000_000
    assert final["transitions"] == 2
    assert final["steps"] == 2
    assert final["scene_id"] == "robocasa/PickPlaceCounterToSink"

    hal.disconnect()  # idempotent — no second terminal line
    assert len(cap.named("sim.task_success_final")) == 1


def test_task_success_silent_for_a_backend_without_a_predicate(cap: _CaptureProcessor) -> None:
    """No predicate → no lines at all. `None` is never reported as failure."""
    env = _success_env()  # task_success_value stays None
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    for _ in range(3):
        hal.send_action(_joint_position_chunk())
    hal.disconnect()
    assert cap.named("sim.task_success") == []
    assert cap.named("sim.task_success_final") == []


def test_task_success_probe_failure_latches_off_without_breaking_actuation(
    cap: _CaptureProcessor,
) -> None:
    """A raising predicate warns once, stops being polled, and steps go on."""
    env = _success_env(task_success_error=RuntimeError("no target object in scene"))
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()  # the connect-time seed read is the one that raises
    calls_after_connect = env.task_success_calls
    for _ in range(3):
        hal.send_action(_joint_position_chunk())

    assert env.step_calls == 3  # actuation unaffected
    assert env.task_success_calls == calls_after_connect  # polling latched off
    warnings = cap.named("sim.task_success_probe_failed")
    assert len(warnings) == 1
    assert warnings[0]["error_type"] == "RuntimeError"
    assert hal.task_success() is None


def test_task_success_reset_reseeds_without_logging_a_regression(cap: _CaptureProcessor) -> None:
    """A reconnect re-seeds the witness; the new episode's False is not a flip."""
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    env.task_success_value = True
    hal.send_action(_joint_position_chunk())
    assert len(cap.named("sim.task_success")) == 1

    env.task_success_value = False  # the reset re-randomises the scene
    hal.connect()
    assert len(cap.named("sim.task_success")) == 1  # no true→false line
    # …and the fresh episode's own rising edge is once again a first success.
    env.task_success_value = True
    hal.send_action(_joint_position_chunk())
    lines = cap.named("sim.task_success")
    assert len(lines) == 2
    assert lines[-1]["first_success"] is True
    assert lines[-1]["step"] == 1


def test_task_success_logger_reaches_the_openral_otel_bridge_namespace() -> None:
    """The signal's logger is a stdlib DESCENDANT of the bridged ``openral``.

    `openral_observability.logging.install_structlog_bridge` attaches the
    OTel `LoggingHandler` to `logging.getLogger("openral")`, and stdlib
    ancestry is dotted — a record logged under this module's own
    `openral_hal.sim_attached` name would never propagate there, so the
    dashboard's event log would never see the signal. Asserted against real
    stdlib propagation, with a real handler.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    bridged = logging.getLogger("openral")
    handler = _Collector()
    previous_level = bridged.level
    bridged.addHandler(handler)
    bridged.setLevel(logging.INFO)
    try:
        logging.getLogger(sim_attached._TASK_SUCCESS_LOGGER).info("sim.task_success")
    finally:
        bridged.removeHandler(handler)
        bridged.setLevel(previous_level)

    assert [r.getMessage() for r in records] == ["sim.task_success"]


def test_task_success_line_is_mirrored_to_stdout_as_json(
    cap: _CaptureProcessor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verdict also lands on stdout, where `ros2 launch` captures it.

    The structlog record is exported to the collector but never printed
    (nothing configures a stdout handler in a launched HAL subprocess), so
    the line would be absent from exactly the artifact a validation run
    greps. The mirror is `<event> <json>`, matching the
    `sim.estop_ground_truth_snapshot` line the sibling sensor bridge writes.
    """
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    env.task_success_value = True
    hal.send_action(_joint_position_chunk())
    hal.disconnect()

    printed = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("sim.task_success")
    ]
    assert len(printed) == 2
    event, _, blob = printed[0].partition(" ")
    assert event == "sim.task_success"
    payload = json.loads(blob)
    assert payload["success"] is True
    assert payload["first_success"] is True
    assert payload["scene_id"] == "robocasa/PickPlaceCounterToSink"

    final_event, _, final_blob = printed[1].partition(" ")
    assert final_event == "sim.task_success_final"
    assert json.loads(final_blob)["ever_succeeded"] is True


def test_task_success_final_re_reads_the_predicate_at_disconnect(
    cap: _CaptureProcessor,
) -> None:
    """The terminal verdict is a fresh read, not the last polled value.

    The MuJoCo BODY_TWIST path advances the sim by writing base qpos
    directly rather than through `_step_and_cache`, so the predicate can
    move without a poll; the closing statement must reflect the env, not a
    stale cache.
    """
    env = _success_env(task_success_value=False)
    hal = SimAttachedHAL(env, _two_dof_description())
    hal.connect()
    hal.send_action(_joint_position_chunk())
    # Flip WITHOUT stepping — nothing polled this transition.
    env.task_success_value = True
    assert cap.named("sim.task_success") == []

    hal.disconnect()
    final = cap.named("sim.task_success_final")[0]
    assert final["success"] is True
