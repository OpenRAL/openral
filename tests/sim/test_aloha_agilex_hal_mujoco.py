# SPDX-License-Identifier: Apache-2.0
"""aloha-agilex MuJoCo twin: closed-loop tracking, both grippers, three cameras (#740).

Drives the real ``MujocoArmHAL.from_description`` built purely from
``robots/aloha_agilex/robot.yaml`` against the MJCF derived from RoboTwin's
pinned aloha-agilex URDF (``robotwin:aloha_agilex``), gravity on. The arms
move only through ``send_action``; ``read_state`` / ``read_images`` are the only
read-backs.

No mocks (CLAUDE.md §1.11): when ``mujoco`` is missing or the RoboTwin archive
cannot be read (``AssetFetchError``), the module skips with the reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

import numpy as np
from openral_core import Action, ControlMode, RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._sim_attachment_evidence import _resolve_gripper_body_groups, _sim_joint_body_id

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_TRACK_TOL_RAD = 0.05
_GRIPPER_TOL = 0.05
_SETTLE_SIM_S = 2.0
#: joint1..joint6 per arm, inside every manifest limit and away from them; near
#: RoboTwin's working pose (``lerobot/robotwin_unified`` state mean j2 1.1, j3 0.8).
_LEFT_ARM = [0.3, 1.1, 0.8, -0.3, 0.2, 0.3]
_RIGHT_ARM = [-0.3, 1.1, 0.8, 0.3, -0.2, -0.3]


@pytest.fixture
def connected() -> Iterator[tuple[RobotDescription, MujocoArmHAL]]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "aloha_agilex" / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"RoboTwin embodiments archive not readable on this host: {exc}")
    hal = MujocoArmHAL.from_description(desc, gravity_enabled=True)
    hal.connect()
    try:
        yield desc, hal
    finally:
        hal.disconnect()


def _drive(hal: MujocoArmHAL, target: list[float]) -> list[float]:
    """Hold ``target`` for ``_SETTLE_SIM_S`` of sim time; return the final positions."""
    tick_s = hal.sim_dt_per_tick_s
    assert tick_s is not None and tick_s > 0.0
    for _ in range(round(_SETTLE_SIM_S / tick_s)):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[target], horizon=1)
        )
    return list(hal.read_state().position)


@pytest.mark.parametrize(
    ("left", "right"), [(1.0, 0.0), (0.0, 1.0)], ids=["left_open", "right_open"]
)
def test_tracks_joint_targets_and_grippers(
    connected: tuple[RobotDescription, MujocoArmHAL], left: float, right: float
) -> None:
    """Arm joints settle within 0.05 rad; each normalised gripper within 0.05."""
    desc, hal = connected
    target = [*_LEFT_ARM, left, *_RIGHT_ARM, right]
    assert [j.name for j in desc.joints] == list(hal.read_state().name)
    got = _drive(hal, target)
    for joint, want, have in zip(desc.joints, target, got, strict=True):
        tol = _GRIPPER_TOL if joint.role == "gripper" else _TRACK_TOL_RAD
        assert abs(have - want) < tol, f"{joint.name}: commanded {want:.3f}, settled {have:.3f}"


def test_cameras_render(connected: tuple[RobotDescription, MujocoArmHAL]) -> None:
    """``top`` and both wrist cameras render at their intrinsics and see the scene."""
    desc, hal = connected
    _drive(hal, [*_LEFT_ARM, 1.0, *_RIGHT_ARM, 1.0])
    frames = hal.read_images()
    for sensor in desc.sensors:
        frame = frames[sensor.name]
        assert sensor.intrinsics is not None
        assert frame.shape == (sensor.intrinsics.height, sensor.intrinsics.width, 3)
        assert frame.dtype == np.uint8
        assert float(frame.std()) > 5.0, f"{sensor.name} camera frame is blank"


def test_manifest_resolves_each_hand_and_jaw(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """Each arm's hand and jaw resolve to twin bodies through ``sim_joint_name``.

    The single-gripper ``SimAttachmentEvidenceTracker`` refuses a bimanual robot
    (two gripper parent links), so deploy runs this twin without automatic grasp
    evidence; this pins the per-arm wiring that a bimanual tracker would use.
    """
    desc, hal = connected
    model = hal._model
    assert model is not None
    grippers = [j for j in desc.joints if j.role == "gripper"]
    by_child = {j.child_link: j for j in desc.joints}
    hands = {model.body(_sim_joint_body_id(model, by_child[g.parent_link])).name for g in grippers}
    assert hands == {"fl_link6", "fr_link6"}
    jaws = {
        model.body(root).name for root in _resolve_gripper_body_groups(model, grippers).values()
    }
    assert jaws == {"fl_link7", "fl_link8", "fr_link7", "fr_link8"}  # + mimic fingers
