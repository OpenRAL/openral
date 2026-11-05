# SPDX-License-Identifier: Apache-2.0
"""The reasoner can send OpenAMRobot 2.0 to a pose through Nav2.

The path the LLM drives, minus the LLM: OpenAMR's real manifest and the in-tree rSkill
registry build the reasoner's palette (``build_tool_palette``); the palette becomes the
provider tool list (``_tool_palette_to_anthropic_tools``); a tool call shaped like a
provider's reply is decoded by the reasoner's own ``_decode_tool_payload`` into the
``ExecuteRskillTool`` that ``reasoner_node`` sends as an ``ExecuteRskill`` goal. The live
leg (that goal driving Nav2 in the walled MuJoCo room) is in robots/openamr_v2/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openral_core import ExecuteRskillTool, RobotDescription, RSkillManifest
from openral_reasoner.palette import ToolPalette, build_tool_palette
from openral_reasoner.tool_use import (
    _decode_tool_payload,
    _skill_id_to_tool_name,
    _tool_palette_to_anthropic_tools,
)

_REPO = Path(__file__).resolve().parents[2]
_NAV2 = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"


@pytest.fixture(scope="module")
def palette() -> ToolPalette:
    description = RobotDescription.from_yaml(str(_REPO / "robots" / "openamr_v2" / "robot.yaml"))
    manifests = [
        RSkillManifest.from_yaml(str(p)) for p in sorted((_REPO / "rskills").glob("*/rskill.yaml"))
    ]
    return build_tool_palette(
        installed_skills=manifests, robot_capabilities=description.capabilities
    )


def test_nav2_is_in_openamrs_palette_with_its_pose_schema(palette: ToolPalette) -> None:
    assert _NAV2 in palette.execute_rskill_ids
    tools = {t["name"]: t for t in _tool_palette_to_anthropic_tools(palette)}
    tool = tools[_skill_id_to_tool_name(_NAV2)]
    goal_schema = tool["input_schema"]["properties"]["goal_params_json"]  # type: ignore[index]
    assert "pose" in goal_schema["properties"]


def test_a_pose_tool_call_decodes_to_an_execute_rskill_goal(palette: ToolPalette) -> None:
    pose = {
        "pose": {
            "header": {"frame_id": "map"},
            "pose": {
                "position": {"x": 7.5, "y": 3.75, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.2297, "w": 0.9733},
            },
        }
    }
    call = _decode_tool_payload(
        tool_name=_skill_id_to_tool_name(_NAV2),
        arguments={"prompt": "drive behind the pillar", "goal_params_json": pose},
        palette=palette,
    )
    assert isinstance(call, ExecuteRskillTool)
    assert call.rskill_id == _NAV2
    assert json.loads(call.goal_params_json) == pose
