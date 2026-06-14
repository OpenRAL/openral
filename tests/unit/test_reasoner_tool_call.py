"""Unit tests for :data:`openral_core.ReasonerToolCall` (ADR-0018 F4).

Real Pydantic — no mocks. Tests cover the round-trip through the
discriminated union, the rejection of unknown discriminators, and the
field bounds enforced on every variant.
"""

from __future__ import annotations

import pytest
from openral_core import (
    EmitPromptTool,
    ExecuteRskillTool,
    LifecycleTransitionTool,
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
