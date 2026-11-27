"""Sim tests for ``openral_hal.OpenAMRobotV2MujocoHAL`` against real MuJoCo physics.

The twin is built from the vendored ``robots/openamrobot_v2/openamrobot_v2.urdf``
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
from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openamrobot_v2" / "robot.yaml"

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_MANIFEST))


@pytest.fixture
def hal(description: RobotDescription) -> Iterator[OpenAMRobotV2MujocoHAL]:
    try:
        hal = build_hal(description, mode="sim")
        assert isinstance(hal, OpenAMRobotV2MujocoHAL)
        hal.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamrobot_v2 twin unavailable: {exc}")
    yield hal
    hal.disconnect()


def _twist(vx: float, wz: float, vy: float = 0.0) -> Action:
    return Action(
        control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)]
    )


def _truth(hal: OpenAMRobotV2MujocoHAL) -> tuple[float, float, float]:
    """The base's simulated ground-truth ``(x, y, yaw)``: its unactuated planar joints."""
    names = [j.name for j in hal.description.joints]
    state = hal.read_state().position
    x, y, yaw = (state[names.index(n)] for n in ("base_x", "base_y", "base_yaw"))
    return (x, y, yaw)


def _stop(hal: OpenAMRobotV2MujocoHAL, seconds: float = 1.0) -> None:
    period = hal.sim_dt_per_tick_s
    assert period is not None
    for _ in range(round(seconds / period)):
        hal.send_action(_twist(0.0, 0.0))


def test_body_twist_drives_the_base_on_its_wheels(hal: OpenAMRobotV2MujocoHAL) -> None:
    """4 s of (0.2 m/s, 0.5 rad/s): the wheels reach the twist under upstream's ramp.

    The commanded yaw rate ramps in at 1 rad/s^2 (upstream's DiffDrive limit), so the
    base turns 0.5 * (4 - 0.25) rad if it rolls without slipping; the twin's wheel
    scrub costs some of that, and wheel odometry stays within a few cm of the truth.
    """
    period = hal.sim_dt_per_tick_s
    assert period is not None
    t0 = hal.sim_time_ns()
    assert t0 is not None
    ticks = round(4.0 / period)
    for _ in range(ticks):
        hal.send_action(_twist(0.2, 0.5))
    elapsed = hal.sim_time_ns()
    assert elapsed is not None and (elapsed - t0) / 1e9 == pytest.approx(ticks * period, abs=1e-6)
    vx, vy, _vz, _wx, _wy, wz = hal.base_twist
    assert vx == pytest.approx(0.2, rel=0.03) and vy == 0.0 and wz == pytest.approx(0.5, rel=0.03)
    x, y, yaw = _truth(hal)
    assert yaw == pytest.approx(0.5 * (4.0 - 0.25), rel=0.15)
    assert math.hypot(x, y) > 0.5
    ox, oy, oyaw = hal.base_pose
    assert math.hypot(ox - x, oy - y) < 0.05 and abs(oyaw - yaw) < 0.1


def test_the_wheels_stop_after_the_command_timeout(hal: OpenAMRobotV2MujocoHAL) -> None:
    """A twist is held 0.5 s of sim time without a new one, then ramped to a stop."""
    period = hal.sim_dt_per_tick_s
    assert period is not None
    for _ in range(round(1.0 / period)):
        hal.send_action(_twist(0.2, 0.0))
    assert hal.base_twist[0] == pytest.approx(0.2, rel=0.03)
    twin = hal._twin
    for _ in range(round(1.5 / period)):  # no command: idle slices, as between /cmd_vel
        twin.idle_step(period)
    still = _truth(hal)
    for _ in range(round(1.0 / period)):
        twin.idle_step(period)
    assert _truth(hal) == pytest.approx(still, abs=1e-3)
    assert hal.base_twist[0] == pytest.approx(0.0, abs=1e-3)


def test_lateral_twist_is_refused_on_a_differential_base(hal: OpenAMRobotV2MujocoHAL) -> None:
    with pytest.raises(ROSConfigError, match="differential"):
        hal.send_action(_twist(0.0, 0.0, vy=0.1))
    assert hal.base_pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def _raise_lift(hal: OpenAMRobotV2MujocoHAL, height: float) -> None:
    """Ramp the lift to ``height`` at its rated 0.018 m/s, as the real lift's controller does.

    A step command throws the carriage at the actuator's 3 kN limit and its reaction
    lifts the 90 kg base off its wheels.
    """
    names = [j.name for j in hal.description.joints]
    period = hal.sim_dt_per_tick_s
    assert period is not None
    row = list(hal.read_state().position)
    start = row[names.index("lift_joint")]
    for k in range(1, math.ceil(abs(height - start) / (0.018 * period)) + 30):
        row[names.index("lift_joint")] = start + math.copysign(
            min(abs(height - start), 0.018 * period * k), height - start
        )
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
        )


def test_joint_targets_converge_and_the_lift_holds(hal: OpenAMRobotV2MujocoHAL) -> None:
    """Lift to 0.15 m, bend both arms, open both grippers; the base stays put."""
    names = [j.name for j in hal.description.joints]
    _raise_lift(hal, 0.15)
    target = list(hal.read_state().position)
    for name, value in {
        "lift_joint": 0.15,
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
        pytest.skip(f"openamrobot_v2 twin unavailable: {exc}")
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


def test_every_manifest_camera_renders(hal: OpenAMRobotV2MujocoHAL) -> None:
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

    from tests.unit.test_openarm_skill_on_openamrobot_v2 import openarm_pi05_manifest

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


def test_openarm_slot_tick_holds_base_and_lift(hal: OpenAMRobotV2MujocoHAL) -> None:
    """An OpenArm 4-slot tick drives arms + grippers; the base and lift hold their targets."""
    names = [j.name for j in hal.description.joints]
    _raise_lift(hal, 0.30)
    for _ in range(20):
        hal.send_action(_twist(0.2, 0.0))
    _stop(hal)
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


def test_a_slot_group_leaving_an_arm_uncommanded_is_refused(hal: OpenAMRobotV2MujocoHAL) -> None:
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
    from openral_hal._openamrobot_v2_assets import compose_openamrobot_v2_walls_mjcf
    from openral_sim.backends.robocasa import synthesize_laser_scan_2d

    try:
        xml, meshdir = compose_openamrobot_v2_walls_mjcf()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamrobot_v2 twin unavailable: {exc}")
    scene = meshdir.parent / "test_walls_scene.xml"
    scene.write_text(xml)
    hal = OpenAMRobotV2MujocoHAL(description=description, mjcf_path=str(scene))
    hal.connect()
    try:
        for _ in range(90):  # ~1 m forward under the 0.5 m/s^2 ramp
            hal.send_action(_twist(0.5, 0.0))
        for _ in range(60):  # then a left turn
            hal.send_action(_twist(0.0, math.pi / 4))
        _stop(hal)
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
        x, y, yaw = _truth(hal)  # the scan is cast from where the body is, not odometry
        lx, ly, lyaw = x + 0.33 * math.cos(yaw), y + 0.33 * math.sin(yaw), yaw + math.pi
        for beam in (0, 90, 180, 270):  # cardinal beams miss the pillars from here
            angle = math.radians(beam - 180)
            assert ranges[beam] == pytest.approx(_room_wall_range(lx, ly, lyaw + angle), abs=0.02)
        assert float(ranges.min()) > 1.0, "a beam hit the robot itself"
    finally:
        hal.disconnect()


def test_a_ramp_row_naming_the_arms_holds_the_base_and_lift(hal: OpenAMRobotV2MujocoHAL) -> None:
    """The starting-pose ramp of a 16-D OpenArm pose names the 16 joints, zero-padded.

    The twin holds the base and lift it leaves out at their targets instead of
    driving them to the padding.
    """
    names = [j.name for j in hal.description.joints]
    _raise_lift(hal, 0.30)
    for _ in range(20):
        hal.send_action(_twist(0.2, 0.3))
    _stop(hal)
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


def test_a_replayed_ramp_tick_is_refused(hal: OpenAMRobotV2MujocoHAL) -> None:
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
    hal: OpenAMRobotV2MujocoHAL,
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
    assert hal.base_pose[0] > 0.5  # 100 ticks of 1/30 s at 0.2 m/s, less the 0.4 s ramp
    assert hal.read_state().position[names.index("left_joint2")] == pytest.approx(-0.5, abs=5e-3)
    with pytest.raises(ROSRuntimeError):  # the twist slot's tick is watermarked too
        hal.send_action(group[0])


def test_an_ungrouped_ticked_twist_advances_the_watermark(hal: OpenAMRobotV2MujocoHAL) -> None:
    from openral_core.exceptions import ROSRuntimeError

    twist = _twist(0.1, 0.0)
    twist.tick_index, twist.runner_session_id = 7, 3
    hal.send_action(twist)
    assert hal.last_committed_tick == 7
    with pytest.raises(ROSRuntimeError):
        hal.send_action(twist)


def test_a_joint_row_that_would_move_the_base_is_refused(hal: OpenAMRobotV2MujocoHAL) -> None:
    """The base has no actuator: a row moving it is refused, not silently dropped."""
    names = [j.name for j in hal.description.joints]
    row = list(hal.read_state().position)
    row[names.index("base_x")] += 0.5
    with pytest.raises(ROSConfigError, match="BODY_TWIST"):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
        )
    assert _truth(hal)[0] == pytest.approx(0.0, abs=1e-3)


def test_reset_to_pose_stops_the_wheels_and_restarts_odometry(
    hal: OpenAMRobotV2MujocoHAL,
) -> None:
    """A snapped base reports the snap as its odometry and does not drive on."""
    period = hal.sim_dt_per_tick_s
    assert period is not None
    for _ in range(round(1.0 / period)):
        hal.send_action(_twist(0.2, 0.0))
    names = [j.name for j in hal.description.joints]
    pose = list(hal.read_state().position)
    for name, value in (("base_x", 1.0), ("base_y", -0.5), ("base_yaw", 0.3)):
        pose[names.index(name)] = value
    hal.reset_to_pose(pose)
    assert hal.base_pose == pytest.approx((1.0, -0.5, 0.3), abs=1e-6)
    twin = hal._twin
    for _ in range(round(0.5 / period)):  # inside the command timeout: no held twist left
        twin.idle_step(period)
    assert _truth(hal) == pytest.approx((1.0, -0.5, 0.3), abs=5e-3)
    assert hal.base_pose == pytest.approx(_truth(hal), abs=5e-3)


def test_gravity_off_is_refused(description: RobotDescription) -> None:
    """Without gravity the wheels cannot grip: connect refuses instead of a still base."""
    hal = OpenAMRobotV2MujocoHAL(description=description, gravity_enabled=False)
    with pytest.raises(ROSConfigError) as info:
        hal.connect()
    if "gravity" not in str(info.value):  # an offline host refuses the MJCF first
        pytest.skip(f"twin unavailable: {info.value}")
    assert hal._twin.model is None  # a refused load leaves the rollout closed


def test_the_wheel_angles_reach_the_urdf_joint_states(hal: OpenAMRobotV2MujocoHAL) -> None:
    """The wheels are no manifest joints, but robot_state_publisher needs their angles."""
    before = hal.urdf_extra_joint_positions
    assert set(before) == {"left_wheel_joint", "right_wheel_joint"}
    for _ in range(30):
        hal.send_action(_twist(0.2, 0.0))
    after = hal.urdf_extra_joint_positions
    assert after["left_wheel_joint"] > before["left_wheel_joint"]
    assert after["right_wheel_joint"] > before["right_wheel_joint"]
