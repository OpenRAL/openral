# SPDX-License-Identifier: Apache-2.0
"""WidowX-250s MuJoCo twin: closed-loop tracking and the top camera (#740).

Drives the real ``MujocoArmHAL.from_description`` built purely from
``robots/widowx/robot.yaml`` against the Menagerie ``trossen_wx250s`` MJCF
(``rd:widow_mj_description``), gravity on. The arm is moved only through
``send_action``; ``read_state`` / ``read_images`` are the only read-backs.

No mocks (CLAUDE.md §1.11): when ``mujoco`` is missing or the MJCF cannot be
downloaded (``AssetFetchError``), the module skips with the reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

import numpy as np
from openral_core import Action, ControlMode, RobotDescription
from openral_core.assets import AssetFetchError, resolve_asset
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._sim_attachment_evidence import SimAttachmentEvidenceTracker

pytestmark = pytest.mark.sim

_ROOT = Path(__file__).resolve().parents[2]
_TRACK_TOL_RAD = 0.05
_GRIPPER_TOL = 0.05
_SETTLE_SIM_S = 2.0
#: waist..wrist_rotate, inside every manifest limit and away from them.
_ARM_TARGET = [0.4, -0.3, 0.3, 0.3, 0.5, -0.4]


@pytest.fixture
def connected() -> Iterator[tuple[RobotDescription, MujocoArmHAL]]:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / "widowx" / "robot.yaml"))
    assert desc.assets.mjcf is not None
    try:
        resolve_asset(desc.assets.mjcf, "mjcf")
    except AssetFetchError as exc:
        pytest.skip(f"Menagerie WidowX MJCF not fetchable on this host: {exc}")
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


@pytest.mark.parametrize("gripper", [1.0, 0.5, 0.0], ids=["open", "half", "closed"])
def test_tracks_joint_targets_and_gripper(
    connected: tuple[RobotDescription, MujocoArmHAL], gripper: float
) -> None:
    """Arm joints settle within 0.05 rad; the normalised gripper within 0.05."""
    desc, hal = connected
    target = [*_ARM_TARGET, gripper]
    assert [j.name for j in desc.joints] == list(hal.read_state().name)
    got = _drive(hal, target)
    for joint, want, have in zip(desc.joints, target, got, strict=True):
        tol = _GRIPPER_TOL if joint.role == "gripper" else _TRACK_TOL_RAD
        assert abs(have - want) < tol, f"{joint.name}: commanded {want:.3f}, settled {have:.3f}"


def test_top_camera_renders_the_arm(connected: tuple[RobotDescription, MujocoArmHAL]) -> None:
    """The manifest's ``top`` camera renders at its intrinsics and sees the scene."""
    desc, hal = connected
    frame = hal.read_images()["top"]
    intr = desc.sensors[0].intrinsics
    assert intr is not None
    assert frame.shape == (intr.height, intr.width, 3)
    assert frame.dtype == np.uint8
    assert float(frame.std()) > 5.0, "top camera frame is blank"


def test_reset_to_pose_snaps_both_fingers(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """A snapped gripper moves the equality-mirrored right finger with the left one.

    ``right_finger`` (qpos 7) mirrors ``left_finger`` (qpos 6) through an MJCF joint
    equality; a snap that wrote only the left finger left the pair violating it, and
    the solver yanked the right finger across on the next steps.
    """
    _, hal = connected
    for gripper in (1.0, 0.0, 0.5):
        hal.reset_to_pose([*_ARM_TARGET, gripper])
        data = hal._data
        assert data is not None
        left, right = float(data.qpos[6]), float(data.qpos[7])
        assert right == pytest.approx(-left, abs=1e-9), f"gripper={gripper}"
        assert float(data.qvel[7]) == 0.0
        assert hal.read_state().position[6] == pytest.approx(gripper, abs=1e-6)


def test_attachment_evidence_resolves_the_gripper(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """The twin's grasp evidence resolves the hand body and both jaws.

    Without ``sim_joint_name`` on the joints the tracker cannot find the gripper's
    parent body and deploy disables attachment evidence for the twin.
    """
    desc, hal = connected
    model = hal._model
    assert model is not None
    tracker = SimAttachmentEvidenceTracker(model, desc)
    (hand,) = tracker._hands
    assert model.body(hand.attach_body_id).name == "wx250s/gripper_link"
    jaws = {model.body(root).name for root in hand.body_groups.values()}
    assert jaws == {"wx250s/left_finger_link", "wx250s/right_finger_link"}


def test_attachment_evidence_names_the_kernel_links(
    connected: tuple[RobotDescription, MujocoArmHAL],
) -> None:
    """Grasp evidence names the links the kernel's MJCF-lowered model has.

    The WidowX kernel model is lowered from the MJCF bodies (``wx250s/*``), not the
    manifest's ``gripper_link`` / ``finger_pair``; the kernel rejects a set naming a
    link it lacks, so the tracker binds its hand to the sim bodies the kernel knows.
    """
    lowering = pytest.importorskip("openral_safety.mjcf_lowering")
    desc, hal = connected
    model = hal._model
    assert model is not None
    params = lowering.lower_collision_params(model, [j.name for j in desc.joints])
    kernel_links = set(params["collision_link_names"])
    tracker = SimAttachmentEvidenceTracker(model, desc)
    assert tracker.bind_kernel_links(kernel_links) == []
    (hand,) = tracker._hands
    assert hand.attach_link == "wx250s/gripper_link"
    assert set(hand.touch_links) >= {"wx250s/left_finger_link", "wx250s/right_finger_link"}
    assert {hand.attach_link, *hand.touch_links} <= kernel_links
    # A kernel model with neither naming still refuses, naming the manifest links.
    assert tracker.bind_kernel_links({"base_link"}) == [
        "wx250s/gripper_link",
        *hand.touch_links,
    ]
