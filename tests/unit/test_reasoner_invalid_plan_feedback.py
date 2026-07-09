"""The reasoner degrades gracefully on a malformed tool call, and learns.

A weak / cheap LLM (e.g. a free gemma routed through an OpenAI-compatible
endpoint) can return tool-call ``arguments`` that aren't a single clean JSON
object — trailing tokens, two concatenated objects, or a bare list. Before this
fix ``OpenAICompatibleToolUseClient.select_tool`` ran an unguarded
``json.loads`` whose raw ``JSONDecodeError`` escaped ``ReasonerCore.tick``'s
``except ROSPlanningError`` and killed the whole reasoner node mid-mission.

These tests pin:
- the guard converts a malformed / non-object payload into
  :class:`ROSReasonerInvalidPlan` (a :class:`ROSPlanningError` the tick loop
  already handles), instead of a raw ``JSONDecodeError`` / ``dict()`` crash; and
- :func:`reflect_on_invalid_plan` plus the ``## EXECUTION`` buffer carry the
  decode error back into the next prompt so the model fixes its call.

The only double is the ``openai`` SDK itself — a network boundary (CLAUDE.md
§1.11) — faked with a tiny object graph mirroring its real response shape.

Run with:
    uv run pytest tests/unit/test_reasoner_invalid_plan_feedback.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openral_core import RobotCapabilities
from openral_core.exceptions import ROSPlanningError, ROSReasonerInvalidPlan
from openral_reasoner.context import (
    ContextRenderer,
    ExecutionEventRecord,
    reflect_on_invalid_plan,
)
from openral_reasoner.palette import build_tool_palette
from openral_reasoner.tool_use import OpenAICompatibleToolUseClient


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, *, arguments: str) -> None:
    """Patch ``openai.OpenAI`` so ``create()`` returns one tool call with ``arguments``.

    Mirrors the real SDK response graph
    (``response.choices[0].message.tool_calls[0].function.{name,arguments}``)
    with the LLM's argument string under our control.
    """

    def _create(**_kwargs: object) -> SimpleNamespace:
        function = SimpleNamespace(name="emit_prompt", arguments=arguments)
        tool_call = SimpleNamespace(function=function)
        message = SimpleNamespace(tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    import openai  # reason: network-boundary double

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


def _empty_palette() -> object:
    return build_tool_palette(
        installed_skills=[],
        robot_capabilities=RobotCapabilities(embodiment_tags=["franka_panda"]),
    )


def test_malformed_json_arguments_raise_invalid_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two concatenated objects → json.loads raises "Extra data" (the exact live crash).
    _install_fake_openai(monkeypatch, arguments='{"prompt": "hi"}{"prompt": "again"}')
    client = OpenAICompatibleToolUseClient(
        model_id="gemma-3-12b-it", api_key="local", base_url="http://localhost:11434/v1"
    )
    with pytest.raises(ROSReasonerInvalidPlan) as exc:
        client.select_tool(context_text="go", palette=_empty_palette())
    # It must be a ROSPlanningError so ReasonerCore.tick catches it gracefully.
    assert isinstance(exc.value, ROSPlanningError)
    assert "malformed JSON" in str(exc.value)


def test_non_object_json_arguments_raise_invalid_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    # Valid JSON but a list, not an object → would crash dict() downstream.
    _install_fake_openai(monkeypatch, arguments="[1, 2, 3]")
    client = OpenAICompatibleToolUseClient(
        model_id="gemma-3-12b-it", api_key="local", base_url="http://localhost:11434/v1"
    )
    with pytest.raises(ROSReasonerInvalidPlan) as exc:
        client.select_tool(context_text="go", palette=_empty_palette())
    assert "single JSON object" in str(exc.value)


def test_reflect_on_invalid_plan_carries_error_and_asks_for_valid_call() -> None:
    hint = reflect_on_invalid_plan("malformed JSON arguments (Extra data: line 1 column 30)")
    assert "valid tool call" in hint
    assert "Extra data" in hint  # the verbatim decode error reaches the model


def test_execution_buffer_surfaces_invalid_plan_feedback() -> None:
    # Mirrors the reasoner_node error branch: an invalid plan is appended as an
    # execution event so the next prompt's ## EXECUTION section names it.
    renderer = ContextRenderer()
    seq0 = renderer.seq
    detail = "tool 'emit_prompt' returned malformed JSON arguments"
    renderer.append_execution(
        ExecutionEventRecord(
            rskill_id="(invalid-plan)",
            outcome="failed",
            summary=f"undecodable tool call: {detail}",
            reflection=reflect_on_invalid_plan(detail),
            stamp_ns=1,
        )
    )
    assert renderer.seq == seq0 + 1  # bumped → next heartbeat runs, not suppressed as idle
    out = renderer.render(world_state=None)
    assert "## EXECUTION" in out
    assert "undecodable tool call" in out
    assert "valid tool call" in out
