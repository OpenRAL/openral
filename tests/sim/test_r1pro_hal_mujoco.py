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
