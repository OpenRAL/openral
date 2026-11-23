# SPDX-License-Identifier: Apache-2.0
"""Galaxea R1 Pro MuJoCo twin: tracking, grippers, base twist and cameras (#740).

Drives the real ``MujocoArmHAL`` that ``build_hal`` derives from
``robots/r1pro/robot.yaml`` (``hal.sim: null`` plus its ``sim:`` block) on the
MJCF ``behavior:r1pro`` builds from BEHAVIOR-1K's pinned URDF, gravity on. The
robot is moved only through ``send_action``; ``read_state`` / ``base_pose`` /
``read_images`` are the only read-backs.

No mocks (CLAUDE.md §1.11): when ``mujoco`` / ``huggingface_hub`` is missing or
the BEHAVIOR dataset cannot be downloaded (``AssetFetchError``), the module skips.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("huggingface_hub")

import numpy as np
from openral_core import Action, ControlMode, RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._sim_attachment_evidence import SimAttachmentEvidenceTracker
from openral_hal.resolver import build_hal

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_TRACK_TOL_RAD = 0.05
_GRIPPER_TOL = 0.05
_SETTLE_SIM_S = 3.0
_TORSO = [0.3, -0.6, -0.3, 0.2]
#: joint1..joint7 per arm, inside every manifest limit and away from them.
_LEFT_ARM = [-1.0, 1.0, 0.3, -1.0, 0.3, 0.3, -0.3]
_RIGHT_ARM = [-1.0, -1.0, -0.3, -1.0, -0.3, -0.3, 0.3]


@pytest.fixture
def connected() -> Iterator[tuple[RobotDescription, MujocoArmHAL]]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "r1pro" / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"BEHAVIOR-1K R1 Pro assets not fetchable on this host: {exc}")
    hal = build_hal(desc, mode="sim")
    assert isinstance(hal, MujocoArmHAL)
    hal.connect()
    try:
        yield desc, hal
    finally:
        hal.disconnect()


def _send_for(hal: MujocoArmHAL, action: Action, sim_s: float) -> None:
    tick_s = hal.sim_dt_per_tick_s
    assert tick_s is not None and tick_s > 0.0
    for _ in range(round(sim_s / tick_s)):
        hal.send_action(action)


@pytest.mark.parametrize("gripper", [1.0, 0.0], ids=["open", "closed"])
def test_tracks_torso_arms_and_grippers(
    connected: tuple[RobotDescription, MujocoArmHAL], gripper: float
) -> None:
    """Torso and arm joints settle within 0.05 rad, the grippers within 0.05, gravity on.

    ``finger_joint2`` has no actuator: it follows ``finger_joint1`` through the MJCF
    equality, so it reads the matching slide opening (metres) whatever is commanded.
    """
    desc, hal = connected
    target = [*_TORSO, *_LEFT_ARM, gripper, 0.0, *_RIGHT_ARM, gripper, 0.0]
    assert [j.name for j in desc.joints] == list(hal.read_state().name)
    action = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target], horizon=1)
    _send_for(hal, action, _SETTLE_SIM_S)
    got = dict(zip(hal.read_state().name, hal.read_state().position, strict=True))
    for joint, commanded in zip(desc.joints, target, strict=True):
        # The mirrored finger reads its slide in metres.
        want = 0.05 * gripper if joint.name.endswith("finger_joint2") else commanded
        tol = _GRIPPER_TOL if joint.role == "gripper" else _TRACK_TOL_RAD
        have = got[joint.name]
        assert abs(have - want) < tol, f"{joint.name}: commanded {want:.3f}, settled {have:.3f}"


def test_body_twist_moves_the_base(connected: tuple[RobotDescription, MujocoArmHAL]) -> None:
    """One second of a holonomic twist moves the base by about that much; arms hold."""
    _desc, hal = connected
    before = list(hal.read_state().position)
    twist = Action(
        control_mode=ControlMode.BODY_TWIST,
        body_twist=[(0.4, 0.2, 0.0, 0.0, 0.0, 0.0)],
        horizon=1,
    )
    _send_for(hal, twist, 1.0)
    x, y, yaw = hal.base_pose
    assert x == pytest.approx(0.4, abs=0.02)
    assert y == pytest.approx(0.2, abs=0.02)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    turn = twist.model_copy(update={"body_twist": [(0.0, 0.0, 0.0, 0.0, 0.0, 0.3)]})
    _send_for(hal, turn, 1.0)
    assert hal.base_pose[2] == pytest.approx(0.3, abs=0.01)
    after = list(hal.read_state().position)
    assert max(abs(a - b) for a, b in zip(after, before, strict=True)) < _TRACK_TOL_RAD
    assert math.isfinite(sum(after))


def test_cameras_render(connected: tuple[RobotDescription, MujocoArmHAL]) -> None:
    """Head and wrist cameras render at their manifest intrinsics and see the scene."""
    desc, hal = connected
    frames = hal.read_images()
    for sensor in desc.sensors:
        intr = sensor.intrinsics
        assert intr is not None
        frame = frames[sensor.name]
        assert frame.shape == (intr.height, intr.width, 3)
        assert frame.dtype == np.uint8
        assert float(frame.std()) > 5.0, f"{sensor.name} frame is blank"


def test_manifest_joints_name_mjcf_joints_within_range(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """Every manifest joint names its MJCF joint, and its limits sit inside that range."""
    import mujoco

    desc, hal = connected
    model = hal._model
    assert model is not None
    for joint in desc.joints:
        assert joint.sim_joint_name, f"{joint.name} has no sim_joint_name"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.sim_joint_name)
        assert jid >= 0, f"{joint.sim_joint_name} is not an MJCF joint"
        lo, hi = (float(v) for v in model.jnt_range[jid])
        assert joint.position_limits is not None, f"{joint.name} has no position limits"
        low, high = joint.position_limits
        if joint.urdf_curve:  # a normalised jaw channel: compare its URDF/MJCF slide value
            xs, ys = zip(*joint.urdf_curve, strict=True)
            low, high = (float(np.interp(v, xs, ys)) for v in (low, high))
        assert lo - 1e-6 <= low <= high <= hi + 1e-6, f"{joint.name}: {low, high} vs {lo, hi}"


def test_attachment_evidence_arms_one_hand_per_arm(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """Sim grasp evidence arms both R1 Pro hands, each on its own fixed gripper link.

    ``*_gripper_link`` is no joint's child; the tracker resolves it as the MJCF body
    of the same name, which carries both of that hand's fingers.
    """
    desc, hal = connected
    model = hal._model
    assert model is not None
    tracker = SimAttachmentEvidenceTracker(model, desc)
    assert tracker.attach_links == ("left_gripper_link", "right_gripper_link")
    for hand in tracker._hands:
        assert model.body(hand.attach_body_id).name == hand.attach_link
        side = hand.attach_link.split("_")[0]
        assert {model.body(root).name for root in hand.body_groups.values()} == {
            f"{side}_gripper_finger_link1",
            f"{side}_gripper_finger_link2",
        }


def test_a_named_gripper_action_drives_only_that_hand(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """A lone ``GRIPPER_POSITION`` naming a gripper joint closes that hand only."""
    desc, hal = connected
    open_row = [*_TORSO, *_LEFT_ARM, 1.0, 0.0, *_RIGHT_ARM, 1.0, 0.0]
    _send_for(
        hal,
        Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[open_row], horizon=1),
        _SETTLE_SIM_S,
    )
    before = dict(zip(hal.read_state().name, hal.read_state().position, strict=True))
    close_left = Action(
        control_mode=ControlMode.GRIPPER_POSITION,
        gripper=[0.0],
        ee_name="left_gripper_finger_joint1",
    )
    _send_for(hal, close_left, _SETTLE_SIM_S)
    after = dict(zip(hal.read_state().name, hal.read_state().position, strict=True))
    for joint in desc.joints:
        # Both left slides close (joint1 normalised, joint2 metres); the rest hold.
        want = 0.0 if joint.name.startswith("left_gripper_finger") else before[joint.name]
        tol = _GRIPPER_TOL if joint.role == "gripper" else _TRACK_TOL_RAD
        assert abs(after[joint.name] - want) < tol, f"{joint.name}: {after[joint.name]:.3f}"


@pytest.mark.parametrize("follower", [0.0, 0.05])
def test_pose_reset_keeps_the_jaws_in_step(
    connected: tuple[RobotDescription, MujocoArmHAL], follower: float
) -> None:
    """A reset snaps both slides from the leader; the follower's own entry cannot undo it."""
    _desc, hal = connected
    hal.reset_to_pose([*_TORSO, *_LEFT_ARM, 0.5, follower, *_RIGHT_ARM, 0.5, follower])
    got = dict(zip(hal.read_state().name, hal.read_state().position, strict=True))
    for side in ("left", "right"):
        assert got[f"{side}_gripper_finger_joint1"] == pytest.approx(0.5, abs=1e-6)
        assert got[f"{side}_gripper_finger_joint2"] == pytest.approx(0.025, abs=1e-6)
