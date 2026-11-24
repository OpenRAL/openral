# SPDX-License-Identifier: Apache-2.0
"""Sim tests: NVIDIA WBC-AGILE velocity policy as the H2 twin's sim-only S0 (real MuJoCo).

``H2MujocoHAL(s0_controller="agile_velocity")`` runs the pinned AGILE export
(`nvidia/agile_wbc_unitree_h2`, the bundle the real H2's InferenceController loads) on
the derived official H2 MJCF with the free base **unpinned** and gravity on. Real
policy, real MJCF, real ``mj_step``; skips with the reason when MuJoCo, onnxruntime or
the downloaded bundle is missing (a SHA-256 mismatch fails instead).

Thresholds are set below what was measured on this twin (2026-11-24, robots/h2/README.md
"Sim S0"): standing 10 s drifts ~0.01 m with < 1° tilt; vx = 0.4 m/s walks ~2.1 m in
10 s (~0.21 m/s — the checkpoint's own forward deficit, within its reported training
error, not a twin bug); wz = 0.5 rad/s turns ~4.3 rad in 10 s.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

from openral_core import Action, ControlMode, ROSConfigError
from openral_core.exceptions import ROSCapabilityMismatch, ROSSafetyViolation
from openral_hal import H2_DESCRIPTION, H2MujocoHAL

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]

_JOINTS = [j.name for j in H2_DESCRIPTION.joints]
_DT = 0.02  # one control period at the manifest's 50 Hz


@pytest.fixture
def hal() -> Iterator[H2MujocoHAL]:
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed (just sync --group sim)")
    pytest.importorskip("safetensors", reason="safetensors not installed")
    try:
        robot = H2MujocoHAL(s0_controller="agile_velocity")
    except ROSConfigError as exc:
        if "does not exist" in str(exc):  # not downloaded; a pin mismatch must fail
            pytest.skip(f"H2 AGILE bundle not downloaded: {exc}")
        raise
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()


def _twist(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> Action:
    return Action(control_mode=ControlMode.BODY_TWIST, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)])


def _data(hal: H2MujocoHAL) -> mujoco.MjData:
    handles = hal.mujoco_handles()
    assert handles is not None
    return handles[1]


def _torso_tilt_deg(hal: H2MujocoHAL) -> float:
    model, data = hal.mujoco_handles()  # type: ignore[misc]  # reason: connected
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    return math.degrees(math.acos(min(1.0, float(data.xmat[torso][8]))))


def _run(
    hal: H2MujocoHAL, seconds: float, action: Action | None
) -> tuple[float, float, float, float]:
    """Drive for ``seconds``; return body-frame (forward, lateral) displacement, yaw, max tilt."""
    x0, y0, yaw0 = hal.base_pose
    prev, turned, tilt = yaw0, 0.0, 0.0
    for _ in range(round(seconds / _DT)):
        if action is None:
            assert hal.idle_step(_DT) is True
        else:
            hal.send_action(action)
        yaw = hal.base_pose[2]
        turned += math.remainder(yaw - prev, math.tau)
        prev = yaw
        tilt = max(tilt, _torso_tilt_deg(hal))
        assert not hal.s0_fallen
    x1, y1, _ = hal.base_pose
    dx, dy = x1 - x0, y1 - y0
    forward = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    lateral = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
    return forward, lateral, turned, tilt


class TestConfig:
    def test_rejects_gravity_off(self) -> None:
        with pytest.raises(ROSConfigError, match="gravity_enabled=True"):
            H2MujocoHAL(s0_controller="agile_velocity", gravity_enabled=False)

    def test_rejects_unknown_controller(self) -> None:
        with pytest.raises(ROSConfigError, match="not one of"):
            H2MujocoHAL(s0_controller="gr00t_wbc")  # type: ignore[arg-type]  # reason: invalid on purpose

    def test_default_is_the_glide(self) -> None:
        assert H2MujocoHAL(gravity_enabled=False).s0_controller == "none"


class TestObservationMapping:
    def test_network_sees_the_twin_state(self, hal: H2MujocoHAL) -> None:
        """The policy's own history outputs echo what the twin fed it, element by element.

        The export's feedback tensors hold its processed observation terms (Isaac Lab
        `projected_gravity`, `base_ang_vel` x 0.2, `joint_pos_rel`, `joint_vel_rel` x 0.05
        of the 14 policy joints), so comparing their newest frame against MuJoCo's state
        checks the quaternion order, the angular-velocity frame and the joint mapping.
        """
        hal.send_action(_twist(vx=0.4, wz=0.3))
        for _ in range(60):
            hal.send_action(_twist(vx=0.4, wz=0.3))
        twin = hal._twin
        policy = twin._policy  # type: ignore[attr-defined]  # reason: the AGILE twin
        # The twin steps the policy on the first physics step of each period: re-run
        # one tick on the current state so the outputs match the state compared.
        twin._policy_tick()  # type: ignore[attr-defined]  # reason: the AGILE twin
        out = policy.last_outputs
        data = _data(hal)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, data.qpos[3:7])  # the free-joint quat the policy was fed
        np.testing.assert_allclose(
            out["h_policy_projected_gravity_out"][-1].ravel(),
            rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            out["h_policy_base_ang_vel_out"][-1].ravel(), 0.2 * data.qvel[3:6], atol=1e-5
        )
        idx = [_JOINTS.index(n) for n in policy.action_joint_names]
        q = np.array(hal.read_state().position)[idx]
        default = np.array([policy.default_joint_pos[n] for n in policy.action_joint_names])
        np.testing.assert_allclose(
            out["h_policy_controlled_joint_pos_out"][-1].ravel(), q - default, atol=1e-5
        )
        dq = np.array(hal.read_state().velocity)[idx]
        np.testing.assert_allclose(
            out["h_policy_controlled_joint_vel_out"][-1].ravel(), 0.05 * dq, atol=1e-4
        )
        np.testing.assert_allclose(
            out["h_policy_velocity_commands_out"][-1].ravel(), [0.4, 0.0, 0.3], atol=1e-6
        )


class TestBehaviour:
    def test_stands_10s_at_zero_twist(self, hal: H2MujocoHAL) -> None:
        forward, lateral, turned, tilt = _run(hal, 10.0, _twist())
        assert math.hypot(forward, lateral) < 0.2  # measured ~0.01 m
        assert abs(turned) < 0.2
        assert tilt < 10.0  # measured < 1°
        assert _data(hal).qpos[2] > 0.85  # stands at ~0.94 m

    def test_walks_forward_at_0p4(self, hal: H2MujocoHAL) -> None:
        _run(hal, 1.0, None)
        forward, lateral, _turned, tilt = _run(hal, 10.0, _twist(vx=0.4))
        # Measured ~2.1 m (~0.21 m/s): >= 30 % of the command, never past it.
        assert 1.2 < forward < 4.4
        assert abs(lateral) < 1.0
        assert tilt < 20.0
        assert hal.base_twist == pytest.approx((0.4, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_turns_at_0p5(self, hal: H2MujocoHAL) -> None:
        _run(hal, 1.0, None)
        forward, lateral, turned, _tilt = _run(hal, 10.0, _twist(wz=0.5))
        assert 2.5 < turned < 5.5  # measured ~4.3 rad of the commanded 5.0
        assert math.hypot(forward, lateral) < 1.0

    def test_upper_body_tracks_s1_targets_while_standing(self, hal: H2MujocoHAL) -> None:
        names = ["left_shoulder_pitch_joint", "right_elbow_joint", "head_yaw_joint"]
        row = [0.0] * len(_JOINTS)
        for name, value in zip(names, (-0.4, 0.6, 0.3), strict=True):
            row[_JOINTS.index(name)] = value
        arm = Action(
            control_mode=ControlMode.JOINT_POSITION, joint_targets=[row], joint_names=names
        )
        for _ in range(100):
            hal.send_action(arm)
        assert not hal.s0_fallen
        measured = np.array(hal.read_state().position)
        for name, value in zip(names, (-0.4, 0.6, 0.3), strict=True):
            # The training gains are soft (shoulder pitch kp 90): gravity leaves a
            # steady-state PD error of ~0.1 rad on the shoulder.
            assert abs(measured[_JOINTS.index(name)] - value) < 0.15, name

    def test_s0_joint_command_is_refused(self, hal: H2MujocoHAL) -> None:
        knee = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0] * len(_JOINTS)],
            joint_names=["left_knee_joint"],
        )
        with pytest.raises(ROSCapabilityMismatch, match="left_knee_joint"):
            hal.send_action(knee)
        whole = Action(
            control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0] * len(_JOINTS)]
        )
        with pytest.raises(ROSCapabilityMismatch, match="whole-vector"):
            hal.send_action(whole)

    def test_fall_cuts_torque_and_refuses_until_reset(self, hal: H2MujocoHAL) -> None:
        model, data = hal.mujoco_handles()  # type: ignore[misc]  # reason: connected
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        data.xfrc_applied[torso, :3] = (0.0, 3000.0, 0.0)  # far beyond recovery
        for _ in range(15):
            hal.idle_step(_DT)
        data.xfrc_applied[torso, :] = 0.0
        for _ in range(100):
            hal.idle_step(_DT)
        assert hal.s0_fallen
        assert np.all(data.ctrl == 0.0)
        with pytest.raises(ROSSafetyViolation, match="fell"):
            hal.send_action(_twist())
        policy = hal._twin._policy  # type: ignore[attr-defined]  # reason: the AGILE twin
        hal.reset_to_pose([float(policy.default_joint_pos[n]) for n in _JOINTS])
        assert not hal.s0_fallen
        _run(hal, 3.0, None)
