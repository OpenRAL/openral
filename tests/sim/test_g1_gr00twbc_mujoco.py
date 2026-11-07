"""Sim tests: NVIDIA GR00T Decoupled WBC as the G1 twin's S0 controller (real MuJoCo).

``G1MujocoHAL(s0_controller="gr00t_wbc")`` runs the pinned GR00T WBC Balance /
Walk ONNX policies on the stock Menagerie G1 MJCF with the free base
**unpinned** and gravity on. These tests drive the real HAL against real
``mj_step`` physics and the real pinned weights (no mocks); they skip with the
reason when MuJoCo, the Menagerie MJCF, ``onnxruntime`` or the weights
(downloaded from a pinned GR00T-WholeBodyControl commit) are unavailable.

Behaviour thresholds come from the bake-off run on the same Menagerie MJCF
(robots/g1/README.md "S0 controllers").
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

try:
    from robot_descriptions import g1_mj_description as _g1_desc

    _ = _g1_desc.MJCF_PATH
    _MJCF_ERROR: str | None = None
except Exception as exc:
    _MJCF_ERROR = str(exc)

from openral_core import Action, ControlMode, ROSConfigError
from openral_core.exceptions import ROSCapabilityMismatch, ROSSafetyViolation
from openral_hal import G1_DESCRIPTION, G1MujocoHAL
from openral_hal._g1_gr00t_wbc import (
    S0_DEFAULT_ANGLES,
    Gr00tWbcAssets,
    Gr00tWbcController,
    ensure_gr00t_wbc_assets,
    gr00t_wbc_observation,
)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(_MJCF_ERROR is not None, reason=f"G1 MJCF unavailable: {_MJCF_ERROR}"),
]

_JOINTS = [j.name for j in G1_DESCRIPTION.joints]
_ARM_JOINTS = _JOINTS[15:]
_CONTROL_DT = 0.02


@pytest.fixture(scope="module")
def assets() -> Gr00tWbcAssets:
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed (just sync --group sim)")
    try:
        return ensure_gr00t_wbc_assets()
    except ROSConfigError as exc:
        pytest.skip(f"GR00T WBC weights unavailable: {exc}")


@pytest.fixture
def hal(assets: Gr00tWbcAssets) -> Iterator[G1MujocoHAL]:
    del assets  # ensures the weights are cached before the HAL fetches them
    robot = G1MujocoHAL(s0_controller="gr00t_wbc")
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()


def _twist(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> Action:
    return Action(control_mode=ControlMode.BODY_TWIST, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)])


def _arm_action(arm_targets: list[float]) -> Action:
    row = [0.0] * 15 + list(arm_targets)  # full-dof row, placed by name (ADR-0102)
    return Action(
        control_mode=ControlMode.JOINT_POSITION, joint_targets=[row], joint_names=_ARM_JOINTS
    )


def _stand(hal: G1MujocoHAL, seconds: float) -> None:
    for _ in range(round(seconds / _CONTROL_DT)):
        assert hal.idle_step(_CONTROL_DT) is True


def _pelvis_height(hal: G1MujocoHAL) -> float:
    handles = hal.mujoco_handles()
    assert handles is not None
    return float(handles[1].qpos[2])


class TestConfig:
    def test_rejects_gravity_off(self) -> None:
        with pytest.raises(ROSConfigError, match="gravity_enabled=True"):
            G1MujocoHAL(s0_controller="gr00t_wbc", gravity_enabled=False)

    def test_rejects_contradicting_walking_flag(self) -> None:
        with pytest.raises(ROSConfigError, match="set only one"):
            G1MujocoHAL(s0_controller="gr00t_wbc", walking_enabled=True)

    def test_rejects_unknown_controller(self) -> None:
        with pytest.raises(ROSConfigError, match="not one of"):
            G1MujocoHAL(s0_controller="agile")  # type: ignore[arg-type]  # reason: invalid on purpose

    def test_default_is_the_glide(self) -> None:
        assert G1MujocoHAL(gravity_enabled=False).s0_controller == "none"


class TestObservation:
    def test_frame_layout_matches_hand_built_frame(self) -> None:
        rng = np.random.default_rng(7)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        omega = rng.normal(size=3)
        q = rng.normal(size=29) * 0.3
        dq = rng.normal(size=29)
        last = rng.normal(size=15).astype(np.float32)
        frame = gr00t_wbc_observation(
            command=(0.4, -0.2, 0.6),
            base_height=0.7,
            torso_rpy=(0.1, -0.05, 0.02),
            base_quat_wxyz=quat,
            base_ang_vel_body=omega,
            joint_pos=q,
            joint_vel=dq,
            last_action=last,
        )
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        gravity = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        default29 = np.concatenate([S0_DEFAULT_ANGLES, np.zeros(14)])
        expected = np.concatenate(
            [
                [0.8, -0.4, 0.3, 0.7, 0.1, -0.05, 0.02],
                omega * 0.5,
                gravity,
                q - default29,
                dq * 0.05,
                last,
            ]
        ).astype(np.float32)
        assert frame.shape == (86,)
        np.testing.assert_allclose(frame, expected, atol=1e-6)

    def test_history_is_six_frames_oldest_first(self, assets: Gr00tWbcAssets) -> None:
        controller = Gr00tWbcController(assets)
        frames = [np.full(86, float(k), dtype=np.float32) for k in range(1, 4)]
        for frame in frames[:-1]:
            controller.policy_input(frame)
        batch = controller.policy_input(frames[-1])
        assert batch.shape == (1, 516)
        blocks = batch.reshape(6, 86)[:, 0].tolist()
        assert blocks == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]  # zero-filled, then oldest -> newest


def _reference_input(history: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    """The bake-off port of upstream ``run_mujoco_gear_wbc.py``'s observation, verbatim."""
    out: NDArray[np.float32] = np.concatenate(history)[None].astype(np.float32)
    return out


def _reference_frame(
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    cmd: tuple[float, float, float],
    action: NDArray[np.float32],
) -> NDArray[np.float32]:
    o = np.zeros(86, np.float32)
    o[0:3] = np.asarray(cmd, np.float32) * np.array([2.0, 2.0, 0.5], np.float32)
    o[3] = 0.74
    o[7:10] = qvel[3:6] * 0.5
    w, x, y, z = qpos[3:7]
    o[10:13] = [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)]
    default = np.zeros(29)
    default[:15] = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0]
    o[13:42] = qpos[7:36] - default
    o[42:71] = qvel[6:35] * 0.05
    o[71:86] = action
    return o


class TestOnnxParity:
    def test_controller_matches_reference_port_on_fixed_states(
        self, assets: Gr00tWbcAssets
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]  # reason: no py.typed

        rng = np.random.default_rng(3)
        model = mujoco.MjModel.from_xml_path(_g1_desc.MJCF_PATH)
        data = mujoco.MjData(model)
        controller = Gr00tWbcController(assets)
        sessions = {
            "walk": ort.InferenceSession(str(assets.walk_onnx), providers=["CPUExecutionProvider"]),
            "balance": ort.InferenceSession(
                str(assets.balance_onnx), providers=["CPUExecutionProvider"]
            ),
        }
        history: list[NDArray[np.float32]] = [np.zeros(86, np.float32)] * 6
        action = np.zeros(15, np.float32)
        for k, cmd in enumerate([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.2, 0.4)] * 2):
            data.qpos[:] = model.qpos0
            data.qpos[7:] += rng.uniform(-0.2, 0.2, 29)
            quat = np.array([1.0, *rng.uniform(-0.1, 0.1, 3)])
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            data.qvel[:] = rng.uniform(-0.5, 0.5, model.nv)
            targets = controller.step(data, command=cmd)
            history = [*history[1:], _reference_frame(data.qpos, data.qvel, cmd, action)]
            net = "walk" if np.linalg.norm(cmd) > 0.05 else "balance"
            action = sessions[net].run(None, {"input": _reference_input(history)})[0][0]
            np.testing.assert_allclose(
                targets, action * 0.25 + S0_DEFAULT_ANGLES, atol=1e-5, err_msg=f"step {k}"
            )
            assert controller.used_walk is (net == "walk")


class TestBehaviour:
    def test_stands_20s_without_falling(self, hal: G1MujocoHAL) -> None:
        x0, y0, _ = hal.base_pose
        _stand(hal, 20.0)
        x1, y1, _ = hal.base_pose
        assert not hal.s0_fallen
        assert _pelvis_height(hal) > 0.7
        assert math.hypot(x1 - x0, y1 - y0) < 0.2  # bake-off drift ~0.06 m

    def test_walks_forward_at_commanded_speed(self, hal: G1MujocoHAL) -> None:
        _stand(hal, 1.0)
        x0, y0, _ = hal.base_pose
        for _ in range(round(5.0 / _CONTROL_DT)):
            hal.send_action(_twist(vx=0.5))
        x1, y1, _ = hal.base_pose
        assert not hal.s0_fallen
        assert math.hypot(x1 - x0, y1 - y0) / 5.0 > 0.3
        assert hal.base_twist == pytest.approx((0.5, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_arms_track_joint_targets_while_walking(self, hal: G1MujocoHAL) -> None:
        arms = [0.3, 0.3, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3, -0.3, 0.0, 0.8, 0.0, 0.0, 0.0]
        _stand(hal, 1.0)
        for _ in range(150):  # 6 s: a twist tick then an arm tick
            hal.send_action(_twist(vx=0.4))
            hal.send_action(_arm_action(arms))
        assert not hal.s0_fallen
        measured = np.array(hal.read_state().position[15:])
        assert np.abs(measured - np.array(arms)).max() < 0.08
        # The arm tick did not stop the held walk command.
        assert hal.base_twist[0] == pytest.approx(0.4)

    def test_s0_joint_command_is_refused(self, hal: G1MujocoHAL) -> None:
        knee = Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.0] * 29],
            joint_names=["left_knee_joint"],
        )
        with pytest.raises(ROSCapabilityMismatch, match="left_knee_joint"):
            hal.send_action(knee)
        whole = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0] * 29])
        with pytest.raises(ROSCapabilityMismatch, match="whole-vector"):
            hal.send_action(whole)

    def test_recovers_from_300n_lateral_pushes(self, hal: G1MujocoHAL) -> None:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        _stand(hal, 2.0)
        for k in range(3):
            yaw = hal.base_pose[2]
            sign = 1.0 if k % 2 == 0 else -1.0
            lateral = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            data.xfrc_applied[torso, :3] = sign * 300.0 * lateral
            _stand(hal, 0.1)  # 100 ms impulse (bake-off protocol)
            data.xfrc_applied[torso, :] = 0.0
            _stand(hal, 3.0)
            assert not hal.s0_fallen, f"fell after push {k}"
        assert _pelvis_height(hal) > 0.7

    def test_fall_cuts_torque_and_refuses_until_reset(self, hal: G1MujocoHAL) -> None:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        data.xfrc_applied[torso, :3] = (0.0, 1500.0, 0.0)  # far beyond recovery
        _stand(hal, 0.3)
        data.xfrc_applied[torso, :] = 0.0
        _stand(hal, 2.0)
        assert hal.s0_fallen
        assert np.all(data.ctrl == 0.0)
        with pytest.raises(ROSSafetyViolation, match="fell"):
            hal.send_action(_twist())
        hal.reset_to_pose([*S0_DEFAULT_ANGLES.tolist(), *[0.0] * 14])
        assert not hal.s0_fallen
        _stand(hal, 3.0)
        assert not hal.s0_fallen
