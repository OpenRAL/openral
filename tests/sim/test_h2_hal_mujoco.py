"""Sim tests for ``openral_hal.H2MujocoHAL`` against real MuJoCo physics.

These tests load Unitree's official H2 MJCF (``unitree_mujoco`` @ ``1eb6642e``,
fetched pinned with per-file SHA-256 by ``openral_hal._unitree_h2_assets``) and
exercise the full HAL lifecycle — connect → read_state → send_action → idle_step
→ estop / disconnect — against a real ``mj_step`` loop. No mocks.

The point is the H2 "real hardware first day" contract (CLAUDE.md §1.11): the
31-DoF action layout is unitree_sdk2 motor order, the qpos addressing survives
the MJCF's tree order (head between waist and arms), and the software PD over
the torque ``motor`` actuators tracks position targets. The twin runs with
gravity ON and the free joint pinned upright (ADR-0087), like the G1 glide.

Upstream drift guards pin facts the manifest documents: the URDF joint limits,
the SDK motor order, the URDF/MJCF mass difference and the URDF/MJCF wrist-pitch
disagreement — a silent upstream change fails here first.
"""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pytest

# ``MUJOCO_GL=osmesa`` makes ``import mujoco`` eagerly load OSMesa, which
# crashes on hosts without it; the HAL never renders, so any failure skips.
try:
    import mujoco
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointState,
    JointType,
    RobotDescription,
    ROSConfigError,
    ROSEStopRequested,
    ROSRuntimeError,
)
from openral_hal import H2_DESCRIPTION, H2MujocoHAL

# The pinned assets download from raw.githubusercontent.com on first use; an
# offline host skips (the fetch is fail-closed, never faked).
try:
    from openral_hal._unitree_h2_assets import ensure_h2_mjcf, ensure_h2_urdf

    _MJCF_PATH: Path | None = ensure_h2_mjcf()
    _URDF_PATH: Path | None = ensure_h2_urdf()
    _ASSET_ERROR: str | None = None
except ROSConfigError as exc:
    from openral_hal._pinned_files import is_download_failure

    if not is_download_failure(exc):
        raise  # a SHA-256 mismatch or cache error is a real failure, never a skip
    _MJCF_PATH = _URDF_PATH = None
    _ASSET_ERROR = str(exc)

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
    pytest.mark.skipif(_ASSET_ERROR is not None, reason=f"H2 assets unavailable: {_ASSET_ERROR}"),
]

_REPO = Path(__file__).resolve().parents[2]

# unitree_sdk2 @ 8e1b9275 example/h2/low_level/h2_ankle_swing_example.cpp
# ``enum H2JointIndex`` (H2_NUM_MOTOR = 31), with the ``_joint`` suffix the
# assets use. Ankle ROLL precedes pitch (index 4 / 5).
_SDK_ORDER: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]
_IDX = {name: i for i, name in enumerate(_SDK_ORDER)}

# Joint friction loss in the MJCF is 0.2 N·m; at the softest PD stiffness
# (20 N·m/rad, wrist pitch/yaw) that is a 0.01 rad dead band, so convergence is
# asserted to 1.5e-2 rad — still far below any real indexing error (≥ 0.1 rad).
_TOL = 1.5e-2


def _joint_action(target: list[float]) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[target],
        stamp_ns=time.time_ns(),
    )


def _twist_action(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, *, vz: float = 0.0) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST,
        horizon=1,
        body_twist=[(vx, vy, vz, 0.0, 0.0, wz)],
        frame_id="base_link",
        stamp_ns=time.time_ns(),
    )


def _model() -> mujoco.MjModel:
    assert _MJCF_PATH is not None
    return mujoco.MjModel.from_xml_path(str(_MJCF_PATH))


# ── Description (no MuJoCo state) ─────────────────────────────────────────────


class TestH2Description:
    def test_canonical_description_shape(self) -> None:
        assert H2_DESCRIPTION.name == "h2"
        assert H2_DESCRIPTION.embodiment_kind == EmbodimentKind.HUMANOID.value
        assert len(H2_DESCRIPTION.joints) == 31
        assert all(j.joint_type == JointType.REVOLUTE.value for j in H2_DESCRIPTION.joints)

    def test_joint_order_is_sdk_motor_order(self) -> None:
        assert [j.name for j in H2_DESCRIPTION.joints] == _SDK_ORDER

    def test_manifest_is_the_runtime_source(self) -> None:
        desc = RobotDescription.from_yaml(str(_REPO / "robots" / "h2" / "robot.yaml"))
        assert [j.name for j in desc.joints] == _SDK_ORDER
        assert desc.sim is not None and H2_DESCRIPTION.sim is not None
        assert desc.sim.joint_qpos_addr == H2_DESCRIPTION.sim.joint_qpos_addr
        assert desc.sim.joint_qvel_addr == H2_DESCRIPTION.sim.joint_qvel_addr

    def test_sim_only_no_real_hal(self) -> None:
        assert H2_DESCRIPTION.hal.sim == "openral_hal.h2:H2MujocoHAL"
        assert H2_DESCRIPTION.hal.real is None

    def test_humanoid_tags_and_modes(self) -> None:
        caps = H2_DESCRIPTION.capabilities
        assert {"h2", "unitree_h2", "humanoid", "mobile_base"} <= set(caps.embodiment_tags)
        assert ControlMode.JOINT_POSITION.value in caps.supported_control_modes
        assert ControlMode.BODY_TWIST.value in caps.supported_control_modes
        assert caps.bimanual is True

    def test_safety_envelope_is_conservative(self) -> None:
        safety = H2_DESCRIPTION.safety
        assert safety.deadman_required is True
        assert safety.max_joint_speed_factor <= 0.5
        assert safety.max_base_linear_speed_m_s == 0.5
        assert safety.max_base_angular_speed_rad_s == 0.5

    def test_no_unverified_sensors_declared(self) -> None:
        # Camera / LiDAR model, intrinsics and rates are unverified upstream.
        assert H2_DESCRIPTION.sensors == []
        assert H2_DESCRIPTION.capabilities.has_vision is False


# ── Upstream asset drift guards ───────────────────────────────────────────────


class TestUpstreamAssets:
    def test_mjcf_has_floating_base_plus_31_hinges(self) -> None:
        model = _model()
        assert model.njnt == 32
        assert int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
        assert model.nu == 31

    def test_actuators_are_torque_motors_in_sdk_order(self) -> None:
        model = _model()
        driven = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in range(model.nu)
        ]
        assert driven == _SDK_ORDER
        # ``<motor>``: unit gain, no bias — ctrl is N·m, hence the software PD.
        assert all(float(model.actuator_gainprm[i, 0]) == 1.0 for i in range(model.nu))
        assert all(int(model.actuator_biastype[i]) == 0 for i in range(model.nu))

    def test_ctrlrange_is_the_urdf_effort_limit(self) -> None:
        model = _model()
        for i, joint in enumerate(H2_DESCRIPTION.joints):
            low, high = model.actuator_ctrlrange[i]
            effort = joint.effort_limit
            assert effort is not None, joint.name
            assert (float(low), float(high)) == pytest.approx((-effort, effort)), joint.name

    def test_qpos_qvel_maps_follow_the_mjcf_tree(self) -> None:
        model = _model()
        sim = H2_DESCRIPTION.sim
        assert sim is not None and sim.joint_qpos_addr and sim.joint_qvel_addr
        for name in _SDK_ORDER:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert sim.joint_qpos_addr[name] == int(model.jnt_qposadr[jid]), name
            assert sim.joint_qvel_addr[name] == int(model.jnt_dofadr[jid]), name
        # The tree puts the head before the arms: description index != qpos - 7.
        assert sim.joint_qpos_addr["head_pitch_joint"] == 22
        assert sim.joint_qpos_addr["left_shoulder_pitch_joint"] == 24

    def test_urdf_limits_are_the_description_limits(self) -> None:
        assert _URDF_PATH is not None
        urdf = {j.get("name"): j for j in ET.parse(_URDF_PATH).getroot().iter("joint")}
        for joint in H2_DESCRIPTION.joints:
            limit = urdf[joint.name].find("limit")
            assert limit is not None, joint.name
            assert joint.position_limits == (
                float(limit.get("lower", "nan")),
                float(limit.get("upper", "nan")),
            ), joint.name
            assert joint.effort_limit == float(limit.get("effort", "nan")), joint.name
            assert joint.velocity_limit == float(limit.get("velocity", "nan")), joint.name

    def test_urdf_and_mjcf_masses_differ_as_documented(self) -> None:
        # robots/h2/README.md: URDF 75.59 kg vs MJCF 67.04 kg of explicit
        # <inertial> (the MJCF pelvis has none; MuJoCo infers ~4.2 kg from its
        # mesh, compiling to 71.29 kg). Not papered over — pinned.
        assert _URDF_PATH is not None and _MJCF_PATH is not None
        urdf_mass = sum(
            float(m.get("value", "0")) for m in ET.parse(_URDF_PATH).getroot().iter("mass")
        )
        mjcf_explicit = sum(
            float(i.get("mass", "0")) for i in ET.parse(_MJCF_PATH).getroot().iter("inertial")
        )
        assert urdf_mass == pytest.approx(75.587, abs=1e-3)
        assert mjcf_explicit == pytest.approx(67.040, abs=1e-3)
        assert float(sum(_model().body_mass)) == pytest.approx(71.288, abs=1e-2)

    def test_urdf_and_mjcf_disagree_on_wrist_pitch_origin(self) -> None:
        # unitree_ros H2.urdf puts the wrist-pitch joint 0.079 m past wrist-roll;
        # unitree_mujoco h2_mujoco.xml puts it at 0.038 m (the wrist-yaw offset
        # absorbs the difference). The manifest follows the URDF (safety-kernel
        # FK); the twin runs the MJCF. If upstream reconciles, update both.
        assert _URDF_PATH is not None
        urdf = {j.get("name"): j for j in ET.parse(_URDF_PATH).getroot().iter("joint")}
        origin = urdf["left_wrist_pitch_joint"].find("origin")
        assert origin is not None
        urdf_x = float(origin.get("xyz", "nan").split()[0])
        model = _model()
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_pitch_link")
        mjcf_x = float(model.body_pos[body][0])
        assert urdf_x == pytest.approx(0.079007)
        assert mjcf_x == pytest.approx(0.037994)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def hal() -> H2MujocoHAL:
    """Gravity-on pinned twin with enough settle steps for the PD to converge."""
    return H2MujocoHAL(settle_steps=3000)


# ── Lifecycle / state ─────────────────────────────────────────────────────────


class TestReadState:
    def test_returns_jointstate_with_31_joints(self, connected_hal: H2MujocoHAL) -> None:
        state = connected_hal.read_state()
        assert isinstance(state, JointState)
        assert state.name == _SDK_ORDER
        assert len(state.position) == len(state.velocity) == 31
        assert max(abs(p) for p in state.position) < 1e-9  # no keyframe: all zero

    def test_base_lowered_so_feet_touch_the_floor(self, connected_hal: H2MujocoHAL) -> None:
        data, model = connected_hal._data, connected_hal._model
        assert data is not None and model is not None
        lowest = min(
            float(data.geom_xpos[g][2]) - float(model.geom_size[g][0])
            for g in range(model.ngeom)
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        )
        assert lowest == pytest.approx(0.0, abs=1e-6)
        assert float(data.qpos[2]) == pytest.approx(1.0425, abs=1e-3)

    def test_disconnect_then_read_raises(self, hal: H2MujocoHAL) -> None:
        hal.connect()
        hal.disconnect()
        with pytest.raises(ROSRuntimeError):
            hal.read_state()


class TestClosedLoopMujoco:
    """Software PD over the torque motors tracks position targets (gravity on, base pinned)."""

    def test_left_arm_converges_and_only_it_moves(self, connected_hal: H2MujocoHAL) -> None:
        target = [0.0] * 31
        arm = {
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0.3,
            "left_shoulder_yaw_joint": 0.2,
            "left_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.3,
            "left_wrist_pitch_joint": 0.2,
            "left_wrist_yaw_joint": 0.3,
        }
        for name, value in arm.items():
            target[_IDX[name]] = value
        connected_hal.send_action(_joint_action(target))
        state = connected_hal.read_state()
        for name, value in arm.items():
            assert state.position[_IDX[name]] == pytest.approx(value, abs=_TOL), name
        for name in _SDK_ORDER:
            if name not in arm:
                assert abs(state.position[_IDX[name]]) < _TOL, f"{name} moved"

    def test_right_arm_converges(self, connected_hal: H2MujocoHAL) -> None:
        target = [0.0] * 31
        arm = {
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.3,
            "right_shoulder_yaw_joint": -0.2,
            "right_elbow_joint": 0.6,
        }
        for name, value in arm.items():
            target[_IDX[name]] = value
        connected_hal.send_action(_joint_action(target))
        state = connected_hal.read_state()
        for name, value in arm.items():
            assert state.position[_IDX[name]] == pytest.approx(value, abs=_TOL), name

    def test_legs_waist_head_converge(self, connected_hal: H2MujocoHAL) -> None:
        target = [0.0] * 31
        pose = {
            "left_hip_pitch_joint": -0.2,
            "left_knee_joint": 0.4,
            "left_ankle_roll_joint": 0.1,
            "left_ankle_pitch_joint": -0.2,
            "right_hip_pitch_joint": -0.2,
            "right_knee_joint": 0.4,
            "right_ankle_roll_joint": -0.1,
            "right_ankle_pitch_joint": -0.2,
            "waist_yaw_joint": 0.3,
            "waist_roll_joint": 0.1,
            "waist_pitch_joint": 0.2,
            "head_pitch_joint": 0.3,
            "head_yaw_joint": 0.5,
        }
        for name, value in pose.items():
            target[_IDX[name]] = value
        connected_hal.send_action(_joint_action(target))
        state = connected_hal.read_state()
        for name, value in pose.items():
            assert state.position[_IDX[name]] == pytest.approx(value, abs=_TOL), name

    def test_ankle_roll_and_pitch_land_in_their_own_slots(self, connected_hal: H2MujocoHAL) -> None:
        # SDK order is roll (4) then pitch (5) — the G1 is the other way round.
        target = [0.0] * 31
        target[4] = 0.15  # left_ankle_roll_joint
        target[5] = -0.4  # left_ankle_pitch_joint
        connected_hal.send_action(_joint_action(target))
        assert connected_hal._data is not None
        data = connected_hal._data
        roll = mujoco.mj_name2id(
            connected_hal._model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll_joint"
        )
        pitch = mujoco.mj_name2id(
            connected_hal._model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch_joint"
        )
        assert connected_hal._model is not None
        assert float(data.qpos[connected_hal._model.jnt_qposadr[roll]]) == pytest.approx(
            0.15, abs=_TOL
        )
        assert float(data.qpos[connected_hal._model.jnt_qposadr[pitch]]) == pytest.approx(
            -0.4, abs=_TOL
        )

    def test_multi_step_chunk_settles_at_last_waypoint(self, connected_hal: H2MujocoHAL) -> None:
        elbow = _IDX["left_elbow_joint"]
        chunk = [[0.0] * 31 for _ in range(3)]
        for row, value in zip(chunk, (0.2, 0.4, 0.6), strict=True):
            row[elbow] = value
        connected_hal.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=3,
                joint_targets=chunk,
                stamp_ns=time.time_ns(),
            )
        )
        assert connected_hal.read_state().position[elbow] == pytest.approx(0.6, abs=_TOL)

    def test_idle_step_holds_the_commanded_pose(self, connected_hal: H2MujocoHAL) -> None:
        target = [0.0] * 31
        target[_IDX["left_elbow_joint"]] = 0.8
        target[_IDX["head_yaw_joint"]] = -0.4
        connected_hal.send_action(_joint_action(target))
        before = connected_hal.read_state().position
        for _ in range(20):
            assert connected_hal.idle_step(0.1) is True
        after = connected_hal.read_state().position
        assert max(abs(a - b) for a, b in zip(after, before, strict=True)) < 5e-3

    def test_reset_to_pose_becomes_the_hold_target(self, connected_hal: H2MujocoHAL) -> None:
        pose = [0.0] * 31
        pose[_IDX["right_elbow_joint"]] = 0.7
        connected_hal.reset_to_pose(pose)
        for _ in range(10):
            connected_hal.idle_step(0.1)
        state = connected_hal.read_state()
        assert state.position[_IDX["right_elbow_joint"]] == pytest.approx(0.7, abs=_TOL)

    def test_wrong_width_action_rejected(self, connected_hal: H2MujocoHAL) -> None:
        with pytest.raises(ROSConfigError):
            connected_hal.send_action(_joint_action([0.0] * 29))


class TestEstop:
    def test_estop_zeroes_torque_and_disconnects(self, hal: H2MujocoHAL) -> None:
        hal.connect()
        hal.send_action(_joint_action([0.0] * 31))
        data = hal._data
        assert data is not None
        with pytest.raises(ROSEStopRequested):
            hal.estop()
        assert float(abs(data.ctrl).max()) == 0.0  # motors de-energised
        assert hal.idle_step(0.1) is False  # never steps an e-stopped twin
        with pytest.raises(ROSRuntimeError):
            hal.read_state()


# ── BODY_TWIST glide (ADR-0087) ───────────────────────────────────────────────


class TestBodyTwistGlide:
    @pytest.fixture()
    def glide_hal(self) -> Iterator[H2MujocoHAL]:
        hal = H2MujocoHAL(settle_steps=50)
        hal.connect()
        yield hal
        hal.disconnect()

    def test_forward_twist_advances_base(self, glide_hal: H2MujocoHAL) -> None:
        x0, y0, yaw0 = glide_hal.base_pose
        glide_hal.send_action(_twist_action(vx=0.5))
        x1, y1, yaw1 = glide_hal.base_pose
        assert x1 - x0 == pytest.approx(0.5 * 0.05, rel=1e-6)
        assert (y1, yaw1) == pytest.approx((y0, yaw0), abs=1e-9)

    def test_twist_follows_heading(self, glide_hal: H2MujocoHAL) -> None:
        for _ in range(32):  # 32 * 1.0 rad/s * 0.05 s ≈ 92°
            glide_hal.send_action(_twist_action(wz=1.0))
        _, _, yaw = glide_hal.base_pose
        assert yaw == pytest.approx(math.pi / 2, abs=0.15)
        x0, y0, _ = glide_hal.base_pose
        glide_hal.send_action(_twist_action(vx=0.5))
        x1, y1, _ = glide_hal.base_pose
        assert y1 > y0 + 0.02
        assert abs(x1 - x0) < 0.01

    def test_base_stays_upright_and_on_the_floor(self, glide_hal: H2MujocoHAL) -> None:
        for _ in range(300):
            assert glide_hal.idle_step() is True
        glide_hal.send_action(_twist_action(vx=0.3))
        data = glide_hal._data
        assert data is not None
        assert float(data.qpos[2]) == pytest.approx(1.0425, abs=1e-3)
        assert tuple(float(v) for v in data.qpos[4:6]) == pytest.approx((0.0, 0.0), abs=1e-9)

    def test_non_planar_twist_rejected(self, glide_hal: H2MujocoHAL) -> None:
        with pytest.raises(ROSConfigError):
            glide_hal.send_action(_twist_action(vx=0.1, vz=0.5))

    def test_twist_holds_the_joint_pose_and_latches(self, glide_hal: H2MujocoHAL) -> None:
        target = [0.0] * 31
        target[_IDX["left_elbow_joint"]] = 0.5
        glide_hal.send_action(_joint_action(target))
        for _ in range(10):  # 50 settle steps is 50 ms; let the PD finish converging
            glide_hal.idle_step(0.1)
        glide_hal.send_action(_twist_action(vx=0.4, wz=0.2))
        assert glide_hal.base_twist == pytest.approx((0.4, 0.0, 0.0, 0.0, 0.0, 0.2))
        elbow = glide_hal.read_state().position[_IDX["left_elbow_joint"]]
        assert elbow == pytest.approx(0.5, abs=_TOL)  # the glide keeps the arm's hold target
        glide_hal.send_action(_joint_action(target))
        assert glide_hal.base_twist == pytest.approx((0.0,) * 6)


class TestPinnedFetchFailsClosed:
    def test_wrong_digest_installs_nothing(self, tmp_path: Path) -> None:
        """The real pinned scene.xml under a wrong SHA-256 is refused, not cached."""
        from openral_hal._pinned_files import fetch_pinned_file
        from openral_hal._unitree_h2_assets import UNITREE_MUJOCO_SHA

        url = (
            "https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/"
            f"{UNITREE_MUJOCO_SHA}/unitree_robots/h2/scene.xml"
        )
        from openral_hal._pinned_files import is_download_failure

        dest = tmp_path / "scene.xml"
        with pytest.raises(ROSConfigError) as caught:
            fetch_pinned_file(url, "0" * 64, dest, what="H2 scene.xml")
        if is_download_failure(caught.value):
            pytest.skip(f"raw.githubusercontent.com unreachable: {caught.value}")
        assert "failed SHA-256 verification" in str(caught.value)
        assert list(tmp_path.iterdir()) == []
