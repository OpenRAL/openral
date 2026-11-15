"""Named navigation goals (ADR-0133): the scene declares them, the deploy forwards them,
the reasoner resolves and shows them.

The fixture is the OpenAMRobot 2.0 warehouse scene's own ``nav_goals``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from openral_cli.deploy_sim import resolve_launch_invocation
from openral_core import DeployScene, NavGoal, Pose6D, ResolvePlaceQuery, load_scene_strict
from openral_core.exceptions import ROSConfigError
from openral_reasoner import format_resolve_place_result, nav_goal_places
from openral_reasoner.context import ContextRenderer
from openral_world_state import SpatialMemory
from pydantic import ValidationError

_REPO = Path(__file__).resolve().parents[2]
_WAREHOUSE = _REPO / "scenes/deploy/isaac_openamrobot_v2_warehouse.yaml"
_PICK_ORANGE = _REPO / "scenes/deploy/isaac_leisaac_pick_orange.yaml"


def _goals() -> list[NavGoal]:
    return load_scene_strict(str(_WAREHOUSE), expected=DeployScene).nav_goals


def _argv(config: Path) -> list[str]:
    return resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    ).argv_template


def test_the_warehouse_scene_declares_its_goals_in_the_map_frame() -> None:
    goals = _goals()
    assert [g.name for g in goals] == ["aisle_start", "shelf_1_props"]
    assert all(g.pose.frame_id == "map" for g in goals)


@pytest.mark.parametrize(
    ("xyz", "quat"),
    [((0, 0, 1.2), (0, 0, 0, 1)), ((0, 0, 0), (0.3, 0, 0, 0.954)), ((0, 0, 0), (0, 0, 0, 0))],
    ids=["off_the_floor", "tilted", "not_a_rotation"],
)
def test_a_goal_is_a_planar_pose(xyz: tuple[float, ...], quat: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError, match=r"planar|unit quaternion"):
        NavGoal(name="s1", pose=Pose6D(xyz=xyz, quat_xyzw=quat, frame_id="map"))


def test_deploy_refuses_goals_outside_the_robots_map_frame(tmp_path: Path) -> None:
    raw = yaml.safe_load(_WAREHOUSE.read_text())
    raw["nav_goals"][0]["pose"]["frame_id"] = "world"
    config = tmp_path / "world_frame_goals.yaml"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ROSConfigError, match="not in 'openamrobot_v2''s map frame 'map'"):
        _argv(config)


def test_deploy_refuses_goals_when_nothing_navigates(tmp_path: Path) -> None:
    raw = yaml.safe_load(_WAREHOUSE.read_text())
    raw["runtime"] = {**raw.get("runtime", {}), "enable_nav2": False}
    config = tmp_path / "no_nav2.yaml"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ROSConfigError, match="no navigation stack"):
        _argv(config)


def test_goal_names_and_aliases_are_unique_in_a_scene() -> None:
    raw = yaml.safe_load(_WAREHOUSE.read_text())
    raw["nav_goals"][1]["aliases"] = ["aisle_start"]
    with pytest.raises(ValidationError, match="nav_goals names/aliases must be unique"):
        DeployScene.model_validate(raw)


def test_deploy_forwards_the_goals_to_the_reasoner() -> None:
    (raw,) = [a.split(":=", 1)[1] for a in _argv(_WAREHOUSE) if a.startswith("nav_goals_json:=")]
    assert [g["name"] for g in json.loads(raw)] == ["aisle_start", "shelf_1_props"]
    assert not [a for a in _argv(_PICK_ORANGE) if a.startswith("nav_goals_json:=")]


def test_deploy_refuses_goals_on_a_robot_that_cannot_drive(tmp_path: Path) -> None:
    goals = yaml.safe_load(_WAREHOUSE.read_text())["nav_goals"]
    config = tmp_path / "arm_with_goals.yaml"
    config.write_text(
        yaml.safe_dump({**yaml.safe_load(_PICK_ORANGE.read_text()), "nav_goals": goals})
    )
    with pytest.raises(ROSConfigError, match="no mobile base"):
        _argv(config)


def test_resolve_place_returns_a_seeded_goal_by_name_alias_or_id() -> None:
    memory = SpatialMemory()
    for place in nav_goal_places(_goals()):
        memory.upsert_node(place)
    want = _goals()[1].pose
    for reference in ("shelf_1_props", "ycb_shelf", "nav_goal:shelf_1_props", "YCB_shelf"):
        result = memory.resolve_place(ResolvePlaceQuery(reference=reference))
        assert result.node_id == "nav_goal:shelf_1_props", reference
        assert result.goal == want


def test_an_exact_name_beats_a_longer_name_that_contains_it() -> None:
    """Review of #608: with the aliases packed into one label, 's1' tied 's1_back' at 0.7
    and the first-inserted won."""
    pose = Pose6D(xyz=(1, 2, 0), quat_xyzw=(0, 0, 0, 1), frame_id="map")
    memory = SpatialMemory()
    for place in nav_goal_places(
        [NavGoal(name="s1_back", pose=pose), NavGoal(name="s1", aliases=("shelf_1",), pose=pose)]
    ):
        memory.upsert_node(place)
    for reference in ("s1", "shelf_1"):
        assert memory.resolve_place(ResolvePlaceQuery(reference=reference)).node_id == (
            "nav_goal:s1"
        ), reference


def test_resolve_place_tells_the_llm_the_heading() -> None:
    memory = SpatialMemory()
    for place in nav_goal_places(_goals()):
        memory.upsert_node(place)
    result = memory.resolve_place(ResolvePlaceQuery(reference="ycb_shelf"))
    text = format_resolve_place_result("ycb_shelf", result)
    assert "yaw=-90deg" in text and "quat_xyzw=(0.0000,0.0000,-0.7071,0.7071)" in text
    assert "declared station" in text


def test_the_context_lists_the_goals_with_their_heading() -> None:
    memory = SpatialMemory()
    for place in nav_goal_places(_goals()):
        memory.upsert_node(place)
    renderer = ContextRenderer()
    renderer.set_spatial_memory(memory.to_scene_graph())
    lines = renderer._render_nav_goals().splitlines()
    assert lines[0].startswith("nav_goals (declared stations")
    assert lines[1] == (
        "- aisle_start id=nav_goal:aisle_start map@(+0.000,+0.000) yaw=0deg "
        'quat_xyzw=(0.0000,0.0000,0.0000,1.0000) "the spawn, at the start of the aisle"'
    )
    assert lines[2].startswith(
        "- shelf_1_props id=nav_goal:shelf_1_props aliases=ycb_shelf map@(+5.100,-3.000) "
        "yaw=-90deg quat_xyzw=(0.0000,0.0000,-0.7071,0.7071)"
    )
