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
from openral_reasoner import nav_goal_places
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


def test_a_goal_outside_the_map_frame_is_refused() -> None:
    pose = Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="world")
    with pytest.raises(ValidationError, match="'map' frame"):
        NavGoal(name="s1", pose=pose)


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


def test_resolve_place_returns_a_seeded_goal_by_name_alias_or_description() -> None:
    memory = SpatialMemory()
    for place in nav_goal_places(_goals()):
        memory.upsert_node(place)
    want = _goals()[1].pose
    for reference in ("shelf_1_props", "ycb_shelf", "nav_goal:shelf_1_props", "Shelf_1"):
        result = memory.resolve_place(ResolvePlaceQuery(reference=reference))
        assert result.node_id == "nav_goal:shelf_1_props", reference
        assert result.goal == want


def test_the_context_lists_the_goals_with_their_heading() -> None:
    memory = SpatialMemory()
    for place in nav_goal_places(_goals()):
        memory.upsert_node(place)
    renderer = ContextRenderer()
    renderer.set_spatial_memory(memory.to_scene_graph())
    text = renderer._render_nav_goals()
    assert text.startswith("nav_goals[map]: ")
    assert "id=nav_goal:shelf_1_props@(+5.10,-3.00,+0.00) yaw=-90deg" in text
    assert "aisle_start: the spawn, at the start of the aisle id=nav_goal:aisle_start" in text
