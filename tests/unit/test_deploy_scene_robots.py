"""A deploy scene that lists several robots (``DeployScene.robots``, openral-pro#99)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import DeployScene, NavGoal, PlaceRegion, PlaceZone, Pose6D, SceneRobot
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import scene_for_robot, yaw_to_quat_xyzw
from pydantic import ValidationError


def _at(x: float, y: float, yaw: float = 0.0, frame: str = "world") -> Pose6D:
    return Pose6D(xyz=(x, y, 0.0), quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame)


def _aisle(**extra: Any) -> dict[str, Any]:
    """Two AMRs at opposite ends of a north-south aisle, a goal and a shelf slot in `world`."""
    return {
        "scene": {"id": "aisle", "backend": "isaacsim"},
        "robots": [
            {"name": "south", "robot_id": "openamrobot_v2", "base_pose": _at(0.4, -1.1)},
            {
                "name": "north",
                "robot_id": "openamrobot_v2",
                "base_pose": _at(-0.4, 2.4, -math.pi / 2),
            },
        ],
        "nav_goals": [NavGoal(name="s5", pose=_at(0.4, 1.1))],
        "place_zones": [
            PlaceZone(
                name="slot_5",
                region=PlaceRegion(
                    frame_id="world",
                    half_extents=(0.1, 0.1, 0.05),
                    pose=_at(1.0, 1.1, math.pi / 2),
                ),
            )
        ],
        **extra,
    }


def test_each_robot_gets_its_own_scene_with_the_shared_poses_in_its_map() -> None:
    scene = DeployScene.model_validate(_aisle())
    south, north = scene_for_robot(scene, "south"), scene_for_robot(scene, "north")
    assert (south.robot_id, south.robots, south.base_pose) == (
        "openamrobot_v2",
        [],
        scene.robots[0].base_pose,
    )
    # south faces +x from (0.4, -1.1): the goal is 2.2 m to its left.
    assert south.nav_goals[0].pose.frame_id == "map"
    assert south.nav_goals[0].pose.xyz == pytest.approx((0.0, 2.2, 0.0), abs=1e-9)
    # north faces -y from (-0.4, 2.4): the goal is 1.3 m ahead, 0.8 m to its left.
    assert north.nav_goals[0].pose.xyz == pytest.approx((1.3, 0.8, 0.0), abs=1e-9)
    assert north.nav_goals[0].pose.quat_xyzw == pytest.approx(yaw_to_quat_xyzw(math.pi / 2))
    zone = north.place_zones[0].region
    assert (zone.frame_id, zone.pose.frame_id) == ("map", "map")
    assert zone.pose.xyz == pytest.approx((1.3, 1.4, 0.0), abs=1e-9)  # 1.3 m ahead, 1.4 m left
    assert zone.pose.quat_xyzw == pytest.approx(yaw_to_quat_xyzw(math.pi), abs=1e-9)
    # Each robot's scene validates as a plain one-robot scene.
    DeployScene.model_validate(north.model_dump())
    assert scene_for_robot(scene, "north", map_frame="odom").nav_goals[0].pose.frame_id == "odom"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"robots": _aisle()["robots"][:1]}, "two or more"),
        ({"robots": [_aisle()["robots"][0]] * 2}, "unique"),
        ({"robot_id": "openamrobot_v2"}, "one-robot fields"),
        ({"nav_goals": [NavGoal(name="s5", pose=_at(0, 0, frame="map"))]}, "shared 'world'"),
    ],
)
def test_a_several_robot_scene_is_refused_when_ambiguous(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        DeployScene.model_validate(_aisle(**change))


def test_a_robot_starts_in_the_shared_frame_and_has_a_plain_name() -> None:
    with pytest.raises(ValidationError, match="world"):
        SceneRobot(name="a", robot_id="x", base_pose=_at(0, 0, frame="map"))
    with pytest.raises(ValidationError):
        SceneRobot(name="North-1", robot_id="x", base_pose=_at(0, 0))


def test_an_unknown_robot_name_is_refused() -> None:
    with pytest.raises(ROSConfigError, match="no robot 'east'"):
        scene_for_robot(DeployScene.model_validate(_aisle()), "east")


def test_a_one_robot_scene_is_unchanged() -> None:
    scene = DeployScene.model_validate(
        {
            "scene": {"id": "aisle", "backend": "isaacsim"},
            "robot_id": "openamrobot_v2",
            "nav_goals": [NavGoal(name="s6", pose=_at(0, 0, frame="map"))],
        }
    )
    assert scene.robots == []
    assert scene.nav_goals[0].pose.frame_id == "map"


def test_sim_run_scenes_stay_one_robot() -> None:
    from openral_core import SimScene

    task = {"id": "restock", "scene_id": "aisle", "instruction": "restock"}
    with pytest.raises(ValidationError, match="runs one robot"):
        SimScene.model_validate(_aisle(task=task))


def test_a_robot_graph_reads_its_own_scene_in_every_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$OPENRAL_SCENE_ROBOT`` (set by ``deploy sim`` per graph): one robot, everywhere."""
    from openral_core import load_scene_strict
    from openral_core.scene_paths import SCENE_ROBOT_ENV
    from openral_hal.sim_bringup import _load_scene_for_hal

    raw = DeployScene.model_validate(_aisle()).model_dump(mode="json", exclude_defaults=True)
    path = tmp_path / "aisle.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert len(DeployScene.from_yaml(str(path)).robots) == 2  # the parent sees every robot
    monkeypatch.setenv(SCENE_ROBOT_ENV, "north")
    north = DeployScene.model_validate(_aisle()).robots[1]
    for scene in (DeployScene.from_yaml(str(path)), load_scene_strict(str(path), DeployScene)):
        assert scene.robots == []
        assert scene.base_pose == north.base_pose
        assert scene.nav_goals[0].pose.frame_id == "map"
        assert scene.nav_goals[0].pose.xyz == pytest.approx((1.3, 0.8, 0.0), abs=1e-9)
    hal_scene = _load_scene_for_hal(str(path))  # the HAL keeps the scene, robot and start pose
    assert (hal_scene.robot_id, hal_scene.base_pose) == (north.robot_id, north.base_pose)
