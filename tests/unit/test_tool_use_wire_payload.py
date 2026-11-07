"""Unit tests for the LLM wire payload: OpenAI shape, wait tool, slug collisions.

Covers the wire-level contracts:

1. The OpenAI ``tools`` conversion must not leak Anthropic's ``input_schema``
   key alongside ``parameters`` (strict endpoints 400 on it; lenient ones
   silently double every tool schema's token cost per call).
2. The ``wait`` no-op tool is offered on both providers' surfaces and decodes
   to ``openral_core.WaitTool``.
3. Slug-colliding rskill ids (``a.b`` vs ``a_b``) get distinct tool names and
   the decoder resolves each back to the right skill.
4. A lifecycle transition may only target a node in the palette.
5. Tool schemas carry no Pydantic docstrings or titles, which held the
   ``franka_panda`` palette at 44K tokens and broke Cosmos3-Edge (#440), and
   every field the LLM fills carries its own description instead. The clients
   render a palette's tools once, not on every tick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, get_args

import pytest
from openral_core import (
    LifecycleTransitionTool,
    ReasonerToolCall,
    RobotDescription,
    RSkillAction,
    RSkillManifest,
    WaitTool,
)
from openral_core.exceptions import ROSReasonerInvalidPlan
from openral_reasoner.palette import RSkillToolEntry, ToolPalette, build_tool_palette
from openral_reasoner.tool_use import (
    AnthropicToolUseClient,
    OpenAICompatibleToolUseClient,
    _decode_tool_payload,
    _llm_schema,
    _skill_id_to_tool_name,
    _tool_palette_to_anthropic_tools,
    _tool_palette_to_openai_tools,
)
from pydantic import BaseModel


def _entry(rskill_id: str) -> RSkillToolEntry:
    return RSkillToolEntry(
        rskill_id=rskill_id,
        description="pick objects on a tabletop",
        actions=(RSkillAction.PICK,),
        objects=("cube",),
        scenes=("tabletop",),
    )


def _palette(*rskill_ids: str) -> ToolPalette:
    return ToolPalette(skills=tuple(_entry(rid) for rid in rskill_ids))


# ── 1. OpenAI payload shape ──────────────────────────────────────────────────


def test_openai_function_objects_carry_no_input_schema_key() -> None:
    palette = _palette("OpenRAL/rskill-smolvla-so100-pick-fp16")
    specs = _tool_palette_to_openai_tools(palette)
    assert specs, "expected a non-empty tool surface"
    for spec in specs:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert isinstance(fn, dict)
        assert set(fn) == {"name", "description", "parameters"}, (
            f"function object for {fn.get('name')!r} leaked extra keys: {sorted(fn)}"
        )
        assert isinstance(fn["parameters"], dict)


def test_openai_and_anthropic_surfaces_expose_the_same_tools() -> None:
    palette = _palette("OpenRAL/rskill-smolvla-so100-pick-fp16")
    anthropic_names = {t["name"] for t in _tool_palette_to_anthropic_tools(palette)}
    openai_names = {
        cast("dict[str, object]", t["function"])["name"]
        for t in _tool_palette_to_openai_tools(palette)
    }
    assert anthropic_names == openai_names


# ── 2. wait tool ─────────────────────────────────────────────────────────────


def test_wait_tool_is_always_offered() -> None:
    names = {t["name"] for t in _tool_palette_to_anthropic_tools(ToolPalette())}
    assert "wait" in names


def test_wait_tool_decodes_to_waittool() -> None:
    call = _decode_tool_payload(
        tool_name="wait",
        arguments={"rationale": "skill in flight, progress nominal"},
        palette=ToolPalette(),
    )
    assert isinstance(call, WaitTool)
    assert call.rationale == "skill in flight, progress nominal"


# ── 3. slug collisions ───────────────────────────────────────────────────────


def test_colliding_slugs_get_distinct_names_and_decode_to_the_right_skill() -> None:
    # "org/a.b" and "org/a_b" both slug to "org__a_b" under the naive mapping.
    palette = _palette("org/a.b", "org/a_b")
    name_map = {
        _skill_id_to_tool_name(entry.rskill_id): entry.rskill_id for entry in palette.skills
    }
    assert len(name_map) == 2
    assert len(set(name_map)) == 2, "colliding ids must yield distinct tool names"
    for name, rskill_id in name_map.items():
        call = _decode_tool_payload(tool_name=name, arguments={}, palette=palette)
        assert call.tool == "execute_rskill"
        assert call.rskill_id == rskill_id


def test_rendered_tools_use_the_collision_free_names() -> None:
    palette = _palette("org/a.b", "org/a_b")
    name_map = {_skill_id_to_tool_name(entry.rskill_id) for entry in palette.skills}
    rendered = {
        t["name"] for t in _tool_palette_to_anthropic_tools(palette) if isinstance(t["name"], str)
    }
    assert name_map.issubset(rendered)


def test_non_colliding_ids_also_get_a_hash_suffix() -> None:
    name = _skill_id_to_tool_name("OpenRAL/rskill-smolvla-so100-pick-fp16")
    assert name.startswith("execute_rskill__OpenRAL__rskill-smolvla-so100-p")
    assert len(name.rsplit("_", 1)[-1]) == 8


# ── 4. Lifecycle targets are bounded by the palette ──────────────────────────


def test_lifecycle_transition_outside_palette_node_ids_is_refused() -> None:
    """A model-chosen lifecycle target not in ``node_ids`` (e.g. the deadman) is refused."""
    palette = ToolPalette(
        execute_rskill_ids=frozenset(), node_ids=frozenset({"/openral_slam_toolbox"})
    )
    ok = _decode_tool_payload(
        tool_name="lifecycle_transition",
        arguments={"node": "/openral_slam_toolbox", "transition": "deactivate"},
        palette=palette,
    )
    assert isinstance(ok, LifecycleTransitionTool)
    with pytest.raises(ROSReasonerInvalidPlan, match="not in the palette"):
        _decode_tool_payload(
            tool_name="lifecycle_transition",
            arguments={"node": "/openral_deadman_watchdog", "transition": "deactivate"},
            palette=palette,
        )


# ── 5. Palette size (#440) ───────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Cosmos3-Edge's window on a Jetson and what else rides in it, from
# docs/reference/cosmos3-edge-reasoner.md. Tool JSON rendered through Edge's own chat
# template (``apply_chat_template(..., tools=...)`` with the nvidia/Cosmos3-Edge
# tokenizer, snapshot 344d602b) costs 3.31 characters of ``json.dumps`` per token on
# the franka palette and 3.28 on the so100 one; 3.27 is the conservative floor. Raw
# ``json.dumps`` alone tokenizes at 3.70, so the pre-#498 "2.7" was simply wrong.
_WINDOW_TOKENS = 32_768
_REPLY_CAP_TOKENS = 512
_SYSTEM_PROMPT_TOKENS = 3_005
_TICK_CONTEXT_TOKENS = 2_000  # the per-tick world-state snapshot, generously
_TOOL_CHARS_PER_TOKEN = 3.27
_TOOL_CHAR_BUDGET = int(
    (_WINDOW_TOKENS - _REPLY_CAP_TOKENS - _SYSTEM_PROMPT_TOKENS - _TICK_CONTEXT_TOKENS)
    * _TOOL_CHARS_PER_TOKEN
)


def test_franka_palette_ships_no_docstrings_and_fits_a_small_model() -> None:
    """The full ``franka_panda`` palette fits Cosmos3-Edge's 32K-token Jetson window.

    Before #440 every per-skill tool carried ``ExecuteRskillTool``'s and its
    ``$defs``' class docstrings plus a ``title`` per node: 168K characters and
    48K Edge tokens. The budget is in characters so the test needs no model
    download. Fitting the window is necessary, not sufficient: Edge still fails
    on 14 franka skills at 25K tokens (#494).
    """
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/franka_panda/robot.yaml"))
    palette = build_tool_palette(
        # Every manifest in the repo: admission (embodiment tags, detector kinds) is
        # build_tool_palette's job, not this test's.
        installed_skills=[
            RSkillManifest.from_yaml(str(p))
            for p in sorted(_REPO_ROOT.glob("rskills/*/rskill.yaml"))
        ],
        robot_capabilities=robot.capabilities,
        detector_available=True,
        scene_query_available=True,
        task_progress_available=True,
        memory_available=True,
    )
    assert len(palette.skills) >= 10, "fixture drift: expected the franka VLA + MoveIt skills"
    assert palette.continuous_detectors and palette.on_demand_detectors, (
        "fixture drift: the any-tagged detectors must reach the locate_in_view description"
    )
    wire = json.dumps(_tool_palette_to_openai_tools(palette))
    # A Pydantic ``title`` is a string value; a field named ``title`` maps to a schema object.
    assert '"title": "' not in wire
    assert "Attributes:" not in wire and "Example:" not in wire
    # "empty = the default" was copied live as the literal string "default", which no
    # camera or locator is named; optional fields say "leave blank" instead.
    assert "empty =" not in wire and "the default" not in wire
    assert len(wire) < _TOOL_CHAR_BUDGET, (
        f"franka palette is {len(wire)} characters ({len(palette.skills)} skills), over the "
        f"{_TOOL_CHAR_BUDGET} that fit a 32K window; offer fewer skills per tick (#494) rather "
        "than raising the budget"
    )


def test_clients_render_tools_once_per_palette() -> None:
    """A tick reuses the rendered tools; a new palette (registry refresh) re-renders."""
    first = _palette("OpenRAL/rskill-smolvla-so100-pick-fp16")
    second = _palette("OpenRAL/rskill-smolvla-so100-place-fp16")
    openai_client = OpenAICompatibleToolUseClient(model_id="local", base_url="http://127.0.0.1:1")
    anthropic_client = AnthropicToolUseClient(model_id="claude-opus-4-8", api_key="unused")
    for client in (openai_client, anthropic_client):
        tools = client._tools(first)
        assert client._tools(first) is tools
        assert client._tools(second) is not tools
    anthropic_tools = anthropic_client._tools(first)
    assert anthropic_tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert sum("cache_control" in t for t in anthropic_tools) == 1


@pytest.mark.parametrize(
    "client_cls", [OpenAICompatibleToolUseClient, AnthropicToolUseClient], ids=lambda c: c.__name__
)
def test_overlapping_render_returns_its_own_palette(
    client_cls: type[OpenAICompatibleToolUseClient] | type[AnthropicToolUseClient],
) -> None:
    """A call whose cache write is overtaken by another palette's still returns its own tools.

    The subclass re-enters ``_tools`` with ``second`` right after ``first`` is
    cached — the interleaving two overlapping ``select_tool`` calls can produce.
    """
    first = _palette("OpenRAL/rskill-smolvla-so100-pick-fp16")
    second = _palette("OpenRAL/rskill-smolvla-so100-place-fp16")

    class Overtaken(client_cls):  # type: ignore[valid-type,misc]  # reason: parametrized base
        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if name == "_rendered" and isinstance(value, tuple) and value[0] is first:
                self._tools(second)

    if client_cls is AnthropicToolUseClient:
        client = Overtaken(model_id="claude-opus-4-8", api_key="unused")
    else:
        client = Overtaken(model_id="local", base_url="http://127.0.0.1:1")
    tools = client._tools(first)
    names = [str(t.get("name") or cast("dict[str, object]", t["function"])["name"]) for t in tools]
    assert any("pick" in n for n in names), names
    assert not any("place" in n for n in names), names


@pytest.mark.parametrize("model", get_args(get_args(ReasonerToolCall)[0]), ids=lambda m: m.__name__)
def test_every_reasoner_tool_field_is_described(model: type[BaseModel]) -> None:
    """Field descriptions are the only field guidance the LLM gets (docstrings are stripped).

    Covers every ``ReasonerToolCall`` variant and the models nested in it
    (``GraspTargetRef``, ``GroundedSubtask``, ...). The ``tool`` discriminator is
    exempt: it is a constant the decoder fills from the tool name.
    """
    schema = _llm_schema(model)
    nodes = [schema, *schema.get("$defs", {}).values()]
    missing = [
        name
        for node in nodes
        for name, prop in node.get("properties", {}).items()
        if name != "tool" and not prop.get("description")
    ]
    assert not missing, f"{model.__name__}: fields without a description: {missing}"
