"""Sim tests for ``openral_hal.OpenAMRv2MujocoHAL`` against real MuJoCo physics.

The twin is built from the vendored ``robots/openamr_v2/openamr_v2.urdf``
(meshes from the pinned openarm_description / openamr-platform-sw clones) and
driven through ``build_hal``, as ``deploy sim`` does. No mocks: the base,
lift, arms and grippers move under the MJCF's own position actuators.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

from openral_core import Action, ControlMode, RobotDescription, ROSConfigError
from openral_hal import build_hal
from openral_hal.openamr_v2 import OpenAMRv2MujocoHAL

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openamr_v2" / "robot.yaml"

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_MANIFEST))


@pytest.fixture
def hal(description: RobotDescription) -> Iterator[OpenAMRv2MujocoHAL]:
    try:
        hal = build_hal(description, mode="sim")
        assert isinstance(hal, OpenAMRv2MujocoHAL)
        hal.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamr_v2 twin unavailable: {exc}")
    yield hal
    hal.disconnect()


def _twist(vx: float, wz: float, vy: float = 0.0) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)]
    )


def test_body_twist_drives_the_base_on_an_arc(hal: OpenAMRv2MujocoHAL) -> None:
    """40 x 0.05 s of (0.2 m/s, 0.5 rad/s) lands on the exact arc."""
    for _ in range(40):
        hal.send_action(_twist(0.2, 0.5))
    x, y, yaw = hal.base_pose
    t = 2.0
    assert yaw == pytest.approx(0.5 * t, abs=1e-6)
    assert x == pytest.approx(0.2 / 0.5 * math.sin(0.5 * t), abs=1e-4)
    assert y == pytest.approx(0.2 / 0.5 * (1 - math.cos(0.5 * t)), abs=1e-4)
    assert hal.base_twist == (0.2, 0.0, 0.0, 0.0, 0.0, 0.5)


def test_lateral_twist_is_refused_on_a_differential_base(hal: OpenAMRv2MujocoHAL) -> None:
    with pytest.raises(ROSConfigError, match="differential"):
        hal.send_action(_twist(0.0, 0.0, vy=0.1))
    assert hal.base_pose == (0.0, 0.0, 0.0)


def test_joint_targets_converge_and_the_lift_holds(hal: OpenAMRv2MujocoHAL) -> None:
    """Lift to 0.30 m, bend both arms, open both grippers; the base stays put."""
    names = [j.name for j in hal.description.joints]
    target = list(hal.read_state().position)
    for name, value in {
        "lift_joint": 0.30,
        "left_joint2": -0.5,
        "left_joint4": 1.0,
        "right_joint2": 0.5,
        "right_joint4": 1.0,
        "left_gripper": 0.6,
        "right_gripper": -0.6,
    }.items():
        target[names.index(name)] = value
    action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[target])
    for _ in range(300):  # 10 s at 30 Hz
        hal.send_action(action)
    reached = hal.read_state().position
    for name, want, got in zip(names, target, reached, strict=True):
        assert got == pytest.approx(want, abs=5e-3), name


def test_lift_holds_under_gravity(description: RobotDescription) -> None:
    """With gravity on, the stiff lift drive holds 0.35 m within 3 mm under both arms."""
    hal = build_hal(description, mode="sim", transport={"gravity_enabled": True})
    try:
        hal.connect()
    except ROSConfigError as exc:
        pytest.skip(f"openamr_v2 twin unavailable: {exc}")
    try:
        names = [j.name for j in description.joints]
        target = list(hal.read_state().position)
        target[names.index("lift_joint")] = 0.35
        action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[target])
        for _ in range(300):
            hal.send_action(action)
        assert hal.read_state().position[names.index("lift_joint")] == pytest.approx(0.35, abs=3e-3)
    finally:
        hal.disconnect()


def test_every_manifest_camera_renders(hal: OpenAMRv2MujocoHAL) -> None:
    images = hal.read_images()
    for sensor in hal.description.sensors:
        if sensor.modality == "rgb":
            assert sensor.intrinsics is not None
            image = images[sensor.name]
            assert isinstance(image, np.ndarray)
            assert image.shape == (
                sensor.intrinsics.height,
                sensor.intrinsics.width,
                3,
            )


def _openarm_slot_tick(
    description: RobotDescription, policy_action: np.ndarray[Any, Any], tick: int
) -> list[Action]:
    """One OpenArm π0.5 tick through the runner's real codec + slot dispatch."""
    import importlib.util

    from openral_rskill._policy_io import PolicyIOCodec

    from tests.unit.test_openarm_skill_on_openamr_v2 import openarm_pi05_manifest

    src = (
        Path(__file__).resolve().parents[2]
        / "packages/openral_rskill_ros/openral_rskill_ros/rskill_runner_node.py"
    )
    spec = importlib.util.spec_from_file_location("_sim_rskill_runner_amr", src)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = openarm_pi05_manifest()
    assert manifest.action_contract is not None
    actions = runner._policy_action_to_actions(
        policy_action,
        codec=PolicyIOCodec.from_manifest(manifest, description),
        slots=manifest.action_contract.slots,
        description=description,
        cartesian_delta_scale=None,
    )
    assert isinstance(actions, list) and len(actions) == 4
    for action in actions:
        action.tick_index = tick
        action.runner_session_id = 0x5EED
    return actions


def test_openarm_slot_tick_holds_base_and_lift(hal: OpenAMRv2MujocoHAL) -> None:
    """An OpenArm 4-slot tick drives arms + grippers; the base and lift hold their targets."""
    names = [j.name for j in hal.description.joints]
    raised = list(hal.read_state().position)
    raised[names.index("lift_joint")] = 0.30
    lift = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[raised])
    for _ in range(150):
        hal.send_action(lift)
    for _ in range(20):
        hal.send_action(_twist(0.2, 0.0))
    pose_before = hal.base_pose
    # Left j2 -0.5, j4 1.0, gripper 0.6; right j2 0.5, j4 1.0, gripper -0.6.
    policy = np.zeros(16, dtype=np.float32)
    policy[[1, 3, 7, 9, 11, 15]] = [-0.5, 1.0, 0.6, 0.5, 1.0, -0.6]
    for tick in range(1, 301):
        for action in _openarm_slot_tick(hal.description, policy, tick):
            hal.send_action(action)
    state = hal.read_state().position
    for k, joint in enumerate(
        [f"left_joint{i}" for i in range(1, 8)]
        + ["left_gripper"]
        + [f"right_joint{i}" for i in range(1, 8)]
        + ["right_gripper"]
    ):
        assert state[names.index(joint)] == pytest.approx(float(policy[k]), abs=5e-3), joint
    assert state[names.index("lift_joint")] == pytest.approx(0.30, abs=5e-3)
    assert hal.base_pose == pytest.approx(pose_before, abs=1e-3)


def test_a_slot_group_leaving_an_arm_uncommanded_is_refused(hal: OpenAMRv2MujocoHAL) -> None:
    """Holding is for the base and lift only: a group without the right arm is refused."""
    group = _openarm_slot_tick(hal.description, np.zeros(16, dtype=np.float32), 1)
    del group[2]  # the right-arm JOINT_POSITION slot
    for action in group:
        action.tick_group_size = 3
    with pytest.raises(ROSConfigError, match="right_joint1"):
        for action in group:
            hal.send_action(action)


def _room_wall_range(x: float, y: float, heading: float) -> float:
    """Distance from ``(x, y)`` along ``heading`` to the walled room's inner faces (+/-9.9 m)."""
    c, s = math.cos(heading), math.sin(heading)
    hits = [(9.9 * math.copysign(1.0, c) - x) / c] if abs(c) > 1e-9 else []
    hits += [(9.9 * math.copysign(1.0, s) - y) / s] if abs(s) > 1e-9 else []
    return min(hits)


def test_scan_from_the_lidar_mount_matches_the_walled_room(description: RobotDescription) -> None:
    """The ray-cast /scan fan rides lidar_link (0.33 m ahead, turned by yaw pi) as the base drives.

    Before the fix the fan was cast from the world origin at heading 0 whatever the
    base did (the base joints carry no ``sim_joint_name``, so the RoboCasa default
    names resolved nothing), and its first hit was the robot's own lidar at 0.33 m.
    """
    from openral_hal._openamr_v2_assets import compose_openamr_v2_walls_mjcf
    from openral_sim.backends.robocasa import synthesize_laser_scan_2d

    try:
        xml, meshdir = compose_openamr_v2_walls_mjcf()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamr_v2 twin unavailable: {exc}")
    scene = meshdir.parent / "test_walls_scene.xml"
    scene.write_text(xml)
    hal = OpenAMRv2MujocoHAL(description=description, mjcf_path=str(scene))
    hal.connect()
    try:
        for _ in range(60):  # 1.5 m forward
            hal.send_action(_twist(0.5, 0.0))
        for _ in range(40):  # then a quarter turn left
            hal.send_action(_twist(0.0, math.pi / 4))
        lidar = description.lidar_sensor
        assert lidar is not None and lidar.range_max_m is not None
        model, data = hal.mujoco_handles()  # type: ignore[misc]  # reason: connected twin
        ranges = synthesize_laser_scan_2d(
            model=model,
            data=data,
            n_beams=360,
            max_range_m=lidar.range_max_m,
            origin_body_name=lidar.parent_frame,
            mount_xyz_rpy=lidar.static_transform_xyz_rpy,
        )
        x, y, yaw = hal.base_pose
        lx, ly, lyaw = x + 0.33 * math.cos(yaw), y + 0.33 * math.sin(yaw), yaw + math.pi
        for beam in (0, 90, 180, 270):  # cardinal beams miss the pillars from here
            angle = math.radians(beam - 180)
            assert ranges[beam] == pytest.approx(_room_wall_range(lx, ly, lyaw + angle), abs=0.02)
        assert float(ranges.min()) > 1.0, "a beam hit the robot itself"
    finally:
        hal.disconnect()


def test_a_ramp_row_naming_the_arms_holds_the_base_and_lift(hal: OpenAMRv2MujocoHAL) -> None:
    """The starting-pose ramp of a 16-D OpenArm pose names the 16 joints, zero-padded.

    The twin holds the base and lift it leaves out at their targets instead of
    driving them to the padding.
    """
    names = [j.name for j in hal.description.joints]
    raised = list(hal.read_state().position)
    raised[names.index("lift_joint")] = 0.30
    lift = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[raised])
    for _ in range(150):
        hal.send_action(lift)
    for _ in range(20):
        hal.send_action(_twist(0.2, 0.3))
    pose_before = hal.base_pose
    arm = [n for n in names if n not in ("base_x", "base_y", "base_yaw", "lift_joint")]
    row = [0.0] * len(names)
    row[names.index("left_joint4")] = 0.8
    ramp = Action(
        control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row], joint_names=arm
    )
    for _ in range(150):
        hal.send_action(ramp)
    state = hal.read_state().position
    assert state[names.index("left_joint4")] == pytest.approx(0.8, abs=5e-3)
    assert state[names.index("lift_joint")] == pytest.approx(0.30, abs=5e-3)
    assert hal.base_pose == pytest.approx(pose_before, abs=1e-3)


def test_a_replayed_ramp_tick_is_refused(hal: OpenAMRv2MujocoHAL) -> None:
    """A subset-named ramp row keeps its tick, so the watermark refuses its replay."""
    from openral_core.exceptions import ROSRuntimeError

    names = [j.name for j in hal.description.joints]
    arm = [n for n in names if n not in ("base_x", "base_y", "base_yaw", "lift_joint")]
    ramp = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * len(names)],
        joint_names=arm,
        tick_index=5,
        runner_session_id=9,
    )
    hal.send_action(ramp)
    assert hal.last_committed_tick == 5
    with pytest.raises(ROSRuntimeError):
        hal.send_action(ramp)


def test_a_mobile_manipulation_tick_moves_base_and_arms_together(
    hal: OpenAMRv2MujocoHAL,
) -> None:
    """A twist slot is staged with the arm slots: nothing moves until the tick is whole."""
    from openral_core.exceptions import ROSRuntimeError

    names = [j.name for j in hal.description.joints]
    policy = np.zeros(16, dtype=np.float32)
    policy[1] = -0.5
    for tick in range(1, 101):
        arms = _openarm_slot_tick(hal.description, policy, tick)
        twist = _twist(0.2, 0.0)
        twist.tick_index, twist.runner_session_id = tick, 0x5EED
        group = [twist, *arms]
        for action in group:
            action.tick_group_size = len(group)
        pose = hal.base_pose
        for action in group[:-1]:
            hal.send_action(action)
        assert hal.base_pose == pose  # staged, not applied
        hal.send_action(group[-1])
        assert hal.last_committed_tick == tick
    assert hal.base_pose[0] > 0.5  # 100 ticks at 0.2 m/s x 0.05 s
    assert hal.read_state().position[names.index("left_joint2")] == pytest.approx(-0.5, abs=5e-3)
    with pytest.raises(ROSRuntimeError):  # the twist slot's tick is watermarked too
        hal.send_action(group[0])


def test_an_ungrouped_ticked_twist_advances_the_watermark(hal: OpenAMRv2MujocoHAL) -> None:
    from openral_core.exceptions import ROSRuntimeError

    twist = _twist(0.1, 0.0)
    twist.tick_index, twist.runner_session_id = 7, 3
    hal.send_action(twist)
    assert hal.last_committed_tick == 7
    with pytest.raises(ROSRuntimeError):
        hal.send_action(twist)
