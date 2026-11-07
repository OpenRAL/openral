"""The fitted Metal geometry does not stop the poses the real arm is recorded in.

The kernel's self-collision model for ``robots/makermods_metal`` (box + exact
hull per link, lowered from the derived ``makermods:metal`` MJCF) must not trip
where the real arm demonstrably was: the Metal pick-place start pose and every
recorded ``observation.state`` of the four fixture episodes of
HappyEthan/metal_pick_place (``tests/unit/fixtures/metal_pick_place_ep*.npz``,
degrees + gripper stroke, converted by the scene's own
``policy_action_to_ctrl``). The refusal side (folds into the base / upper arm,
the station's arms meeting in the middle) is pinned with mesh evidence in
``test_collision_geometry_zero_pose._REAL_COLLISIONS``. Judged by the kernel's
own predicates (``_tripping_pairs``). Real fixtures only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from openral_core import RobotDescription

from tests.unit.test_collision_geometry_zero_pose import _tripping_pairs

policy_env = pytest.importorskip(
    "openral_sim.backends.metal_pick_place.env", reason="needs openral_sim (metal_pick_place)"
)

_ROOT = Path(__file__).resolve().parents[2]
_EPISODES = sorted((_ROOT / "tests" / "unit" / "fixtures").glob("metal_pick_place_ep*.npz"))
_FINGER_TRAVEL_M = 0.05  # sim.grippers[].ctrl_range span: fraction = joint8 / 0.05
# MetalPickPlaceOptions defaults: the dataset's units and closed-gripper reading.
_START_DEG = (0.74, 0.04, -0.05, 0.49, -0.10, -1.38)


def _load(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROOT / "robots" / name / "robot.yaml"))


def _q(robot: RobotDescription, policy_row: npt.NDArray[np.float32]) -> dict[str, float]:
    ctrl = policy_env.policy_action_to_ctrl(
        policy_row, joint_units="degrees", gripper_scale=100.0, gripper_offset_m=0.00125
    )
    names = [j.name for j in robot.joints]
    return {
        **dict(zip(names[:6], map(float, ctrl[:6]), strict=True)),
        names[6]: float(ctrl[6]) / _FINGER_TRAVEL_M,
    }


@pytest.mark.parametrize("gripper", [0.0, 1.0], ids=["closed", "open"])
def test_the_pick_place_start_pose_does_not_trip(gripper: float) -> None:
    robot = _load("makermods_metal")
    q = {f"joint{i + 1}": float(np.radians(v)) for i, v in enumerate(_START_DEG)}
    assert not _tripping_pairs(robot, q | {"gripper": gripper})


def test_the_station_start_pose_does_not_trip() -> None:
    robot = _load("makermods_metal_station")
    q = {
        f"{side}_joint{i + 1}": float(np.radians(v))
        for side in ("left", "right")
        for i, v in enumerate(_START_DEG)
    }
    assert not _tripping_pairs(robot, q | {"left_gripper": 0.0, "right_gripper": 0.0})


def test_no_recorded_state_of_the_real_arm_trips() -> None:
    robot = _load("makermods_metal")
    assert len(_EPISODES) == 4
    stops: list[tuple[str, int, list[tuple[str, str, float]]]] = []
    frames = 0
    for path in _EPISODES:
        for i, row in enumerate(np.load(path)["state"]):
            frames += 1
            if trips := _tripping_pairs(robot, _q(robot, row)):
                stops.append((path.stem, i, trips))
    assert frames > 3000
    assert not stops, f"{len(stops)}/{frames} recorded states would stop: {stops[:5]}"
