"""Unit tests for :data:`openral_core.ReasonerToolCall` (ADR-0018 F4).

Real Pydantic — no mocks. Tests cover the round-trip through the
discriminated union, the rejection of unknown discriminators, and the
field bounds enforced on every variant.
"""

from __future__ import annotations

import pytest
from openral_core import (
    DecomposeMissionTool,
    EmitPromptTool,
    ExecuteRskillTool,
    LifecycleTransitionTool,
    MemorySearchTool,
    MemoryWriteTool,
    ReasonerToolCall,
    ReloadGstPipelineTool,
)
from pydantic import TypeAdapter, ValidationError

ADAPTER = TypeAdapter(ReasonerToolCall)


def test_execute_skill_round_trip() -> None:
    """ExecuteRskillTool survives JSON round-trip via the union adapter."""
    src = ExecuteRskillTool(
        rskill_id="openral/rskill-pick-cube-so100",
        prompt="pick the red cube",
        deadline_s=5.0,
        rationale="user asked to pick the red cube",
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, ExecuteRskillTool)
    assert decoded == src


def test_memory_write_round_trip_and_decode() -> None:
    """MemoryWriteTool decodes via the union by its `memory_write` discriminator."""
    src = MemoryWriteTool(
        op="supersede",
        section="object_locations",
        content="water bottle in the fridge",
        importance=0.9,
        target="water bottle on the counter",
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, MemoryWriteTool)
    assert decoded == src
    assert decoded.tool == "memory_write"


def test_memory_write_requires_content_unless_delete() -> None:
    with pytest.raises(ValidationError, match=r"requires non-empty .content."):
        MemoryWriteTool(op="add", section="lessons", content="")
    # delete may omit content but needs a target:
    ok = MemoryWriteTool(op="delete", section="open_tasks", target="water the plants")
    assert ok.op == "delete"


def test_memory_write_requires_target_for_update_supersede_delete() -> None:
    for op in ("update", "supersede", "delete"):
        with pytest.raises(ValidationError, match=r"requires a .target. entry"):
            MemoryWriteTool(op=op, section="preferences", content="x")


def test_memory_search_round_trip_and_decode() -> None:
    src = MemorySearchTool(query="where was the mug", section="object_locations", limit=3)
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, MemorySearchTool)
    assert decoded == src and decoded.tool == "memory_search"


def test_memory_section_is_closed() -> None:
    with pytest.raises(ValidationError):
        MemoryWriteTool(op="add", section="not_a_section", content="x")  # type: ignore[arg-type]


def test_reload_gst_pipeline_round_trip() -> None:
    """ReloadGstPipelineTool round-trips and preserves the YAML body."""
    yaml_body = "sensor_id: wrist_rgb\nbackend: gstreamer\nbackend_params: {source: testsrc}\n"
    src = ReloadGstPipelineTool(sensor_id="wrist_rgb", pipeline_yaml=yaml_body)
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, ReloadGstPipelineTool)
    assert decoded.pipeline_yaml == yaml_body


def test_lifecycle_transition_round_trip() -> None:
    """LifecycleTransitionTool round-trips with the canonical transition set."""
    src = LifecycleTransitionTool(node="/openral/hal/so100", transition="activate")
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, LifecycleTransitionTool)
    assert decoded.transition == "activate"


def test_emit_prompt_round_trip() -> None:
    """EmitPromptTool round-trips and enforces absolute target_topic."""
    src = EmitPromptTool(
        target_topic="/openral/prompt",
        text="all clear; continue",
        metadata_json='{"priority": 10}',
    )
    decoded = ADAPTER.validate_json(src.model_dump_json())
    assert isinstance(decoded, EmitPromptTool)
    assert decoded.target_topic.startswith("/")


def test_emit_prompt_rejects_relative_topic() -> None:
    """Target topic must be absolute (start with /)."""
    with pytest.raises(ValidationError):
        EmitPromptTool(target_topic="prompt", text="x")


def test_lifecycle_transition_rejects_unknown_transition() -> None:
    """The transition Literal is closed to the canonical four."""
    with pytest.raises(ValidationError):
        LifecycleTransitionTool(node="/x", transition="shutdown")  # type: ignore[arg-type]


def test_execute_skill_rejects_empty_skill_id() -> None:
    """rskill_id has min_length=1."""
    with pytest.raises(ValidationError):
        ExecuteRskillTool(rskill_id="", prompt="x")


def test_execute_skill_rejects_negative_deadline() -> None:
    """deadline_s is ge=0; zero means use the manifest default."""
    with pytest.raises(ValidationError):
        ExecuteRskillTool(rskill_id="x", deadline_s=-1.0)


def test_decompose_mission_round_trip_populate_and_subdivide() -> None:
    """DecomposeMissionTool decodes via the union in both modes (#123)."""
    populate = ADAPTER.validate_json(
        '{"tool": "decompose_mission", "subtasks": ["clear table", "wipe surface"]}'
    )
    assert isinstance(populate, DecomposeMissionTool)
    assert populate.subtasks == ["clear table", "wipe surface"]
    assert populate.target_task_id == ""  # empty → populate the whole queue
    subdivide = ADAPTER.validate_json(
        DecomposeMissionTool(subtasks=["a", "b"], target_task_id="t2").model_dump_json()
    )
    assert isinstance(subdivide, DecomposeMissionTool)
    assert subdivide.target_task_id == "t2"


def test_decompose_mission_trims_and_drops_blank_subtasks() -> None:
    """Blank/whitespace subtasks are dropped; at least one must survive."""
    call = DecomposeMissionTool(subtasks=["  pick  ", "", "   ", "place "])
    assert call.subtasks == ["pick", "place"]
    with pytest.raises(ValidationError):
        DecomposeMissionTool(subtasks=["  ", ""])  # nothing survives
    with pytest.raises(ValidationError):
        DecomposeMissionTool(subtasks=[])  # min_length=1


def test_unknown_tool_kind_is_rejected() -> None:
    """A payload whose discriminator is unknown fails validation."""
    payload = '{"tool": "drive_to_bar", "x": 1}'
    with pytest.raises(ValidationError):
        ADAPTER.validate_json(payload)


def test_variants_are_frozen() -> None:
    """Every variant is frozen=True so the LLM can't mutate routed calls."""
    src = EmitPromptTool(target_topic="/openral/prompt", text="x")
    with pytest.raises(ValidationError):
        src.text = "y"  # type: ignore[misc]


def test_variants_forbid_extra_fields() -> None:
    """extra='forbid' stops a smuggled field from reaching the dispatch."""
    with pytest.raises(ValidationError):
        ExecuteRskillTool.model_validate(
            {
                "tool": "execute_rskill",
                "rskill_id": "x",
                "prompt": "y",
                "deadline_s": 0.0,
                "stash": "should-not-survive",
            },
        )
