"""Unit tests for the LLM wire payload: OpenAI shape, wait tool, slug collisions.

Covers three wire-level fixes:

1. The OpenAI ``tools`` conversion must not leak Anthropic's ``input_schema``
   key alongside ``parameters`` (strict endpoints 400 on it; lenient ones
   silently double every tool schema's token cost per call).
2. The ``wait`` no-op tool is offered on both providers' surfaces and decodes
   to :class:`openral_core.WaitTool`.
3. Slug-colliding rskill ids (``a.b`` vs ``a_b``) get distinct tool names and
   the decoder resolves each back to the right skill.
"""

from __future__ import annotations

from openral_core import RSkillAction, WaitTool
from openral_reasoner.palette import RSkillToolEntry, ToolPalette
from openral_reasoner.tool_use import (
    _decode_tool_payload,
    _skill_tool_name_map,
    _tool_palette_to_anthropic_tools,
    _tool_palette_to_openai_tools,
)


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
    openai_names = {t["function"]["name"] for t in _tool_palette_to_openai_tools(palette)}
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
    name_map = _skill_tool_name_map(palette)
    assert len(name_map) == 2
    assert len(set(name_map)) == 2, "colliding ids must yield distinct tool names"
    for name, rskill_id in name_map.items():
        call = _decode_tool_payload(tool_name=name, arguments={}, palette=palette)
        assert call.tool == "execute_rskill"
        assert call.rskill_id == rskill_id


def test_rendered_tools_use_the_collision_free_names() -> None:
    palette = _palette("org/a.b", "org/a_b")
    name_map = _skill_tool_name_map(palette)
    rendered = {
        t["name"] for t in _tool_palette_to_anthropic_tools(palette) if isinstance(t["name"], str)
    }
    assert set(name_map).issubset(rendered)


def test_non_colliding_ids_keep_the_readable_name() -> None:
    palette = _palette("OpenRAL/rskill-smolvla-so100-pick-fp16")
    (name,) = _skill_tool_name_map(palette)
    assert name == "execute_rskill__OpenRAL__rskill-smolvla-so100-pick-fp16"
