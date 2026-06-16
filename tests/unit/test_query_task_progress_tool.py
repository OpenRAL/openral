"""Tests for the reward-monitor ``query_task_progress`` reasoner tool (ADR-0057).

Schema / wiring only (no GPU, no ROS): the tool parses via the ReasonerToolCall
union, and the LLM sees it in the palette exactly when a reward monitor is
available (``task_progress_available``). The live ZMQ scoring path is validated
on a GPU host (rskills/robometer-4b/PHASE0.md Phase 3).

Run with:
    uv run pytest tests/unit/test_query_task_progress_tool.py -v
"""

from __future__ import annotations

import pytest


def test_query_task_progress_tool_schema_round_trips() -> None:
    """QueryTaskProgressTool parses via the ReasonerToolCall discriminated union."""
    from openral_core import QueryTaskProgressTool, ReasonerToolCall
    from pydantic import TypeAdapter

    parsed = TypeAdapter(ReasonerToolCall).validate_python(
        {"tool": "query_task_progress", "window_s": 5.0, "task": "Put the cube in the bowl"}
    )
    assert isinstance(parsed, QueryTaskProgressTool)
    assert parsed.window_s == 5.0
    assert parsed.task == "Put the cube in the bowl"
    # window_s has a default; task is optional (reuses the active goal).
    default = TypeAdapter(ReasonerToolCall).validate_python({"tool": "query_task_progress"})
    assert default.window_s == 8.0
    assert default.task == ""


def test_query_task_progress_window_must_be_positive() -> None:
    from openral_core import QueryTaskProgressTool
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueryTaskProgressTool(window_s=0.0)


def test_query_task_progress_palette_gated_on_availability() -> None:
    """The LLM sees query_task_progress only when a reward monitor is available."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    off = [d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette())]
    on = [
        d["name"]
        for d in _tool_palette_to_anthropic_tools(ToolPalette(task_progress_available=True))
    ]
    assert "query_task_progress" not in off
    assert "query_task_progress" in on


def test_query_task_progress_independent_of_scene_query() -> None:
    """query_task_progress and query_scene are independently provisioned (ADR-0057)."""
    from openral_reasoner.palette import ToolPalette
    from openral_reasoner.tool_use import _tool_palette_to_anthropic_tools

    names = [
        d["name"] for d in _tool_palette_to_anthropic_tools(ToolPalette(scene_query_available=True))
    ]
    assert "query_scene" in names
    assert "query_task_progress" not in names
