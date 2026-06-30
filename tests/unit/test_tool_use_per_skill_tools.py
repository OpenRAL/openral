"""ADR-0022 — :func:`_tool_palette_to_anthropic_tools` emits one tool per skill.

Drives the real :class:`RSkillManifest` loader against the in-tree
``rskills/*/rskill.yaml`` files, builds a real palette, then asserts
on the shape of the LLM-facing tool schema:

1. The fixed scaffold gains one ``execute_rskill__<slug>`` per skill
   when the palette carries ``N`` skills, alongside the always-present
   tools (``reload_gst_pipeline`` / ``lifecycle_transition`` /
   ``emit_prompt`` / ``decompose_mission``).

2. Each per-skill tool's ``description`` includes the manifest's
   description text and the structured action / object / scene tags so
   the LLM can pick on semantics (not slug).

3. The per-skill tool's ``input_schema`` drops ``rskill_id`` from
   ``properties`` and ``required`` — the tool name already identifies
   the skill, so the LLM only needs ``prompt`` / ``deadline_s``.

4. :func:`_decode_tool_payload` round-trips a per-skill tool call:
   given a ``execute_rskill__<slug>`` name, it resolves the canonical
   ``rskill_id`` and the call validates against the
   :class:`~openral_core.ReasonerToolCall` union.

5. Palettes carrying only ``execute_rskill_ids`` (no per-skill metadata
   — synthetic test palettes, the default empty palette) collapse to
   the single-``execute_rskill``-with-enum schema.

Per CLAUDE.md §1.11: no mocks. The palette is the real palette built
from the real on-disk manifests; the Anthropic / OpenAI clients
themselves are not exercised (their SDKs are an external boundary —
covered separately by integration tests with the FakeToolUseClient).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import (
    ExecuteRskillTool,
    RobotCapabilities,
    RSkillManifest,
)
from openral_reasoner.palette import ToolPalette, build_tool_palette
from openral_reasoner.tool_use import (
    _PER_SKILL_TOOL_PREFIX,
    _decode_tool_payload,
    _drop_property,
    _format_skill_tool_description,
    _skill_id_to_tool_name,
    _tool_palette_to_anthropic_tools,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSKILLS_DIR = _REPO_ROOT / "rskills"


def _real_aloha_palette() -> ToolPalette:
    paths = sorted(_RSKILLS_DIR.glob("*/rskill.yaml"))
    manifests = [RSkillManifest.from_yaml(str(p)) for p in paths]
    caps = RobotCapabilities(embodiment_tags=["aloha"])
    return build_tool_palette(installed_skills=manifests, robot_capabilities=caps)


# ── slug + description helpers ─────────────────────────────────────────────────


def test_skill_id_to_tool_name_short_id_round_trip() -> None:
    """Short HF-Hub ids slug 1:1 and stay under the 64-char limit."""
    name = _skill_id_to_tool_name("OpenRAL/rskill-act-aloha")
    assert name == "execute_rskill__OpenRAL__rskill-act-aloha"
    assert len(name) <= 64
    assert name.startswith(_PER_SKILL_TOOL_PREFIX)


def test_skill_id_to_tool_name_long_id_truncates_with_sha1_suffix() -> None:
    """Ids longer than the 64-char tool name budget get a sha1-8 suffix."""
    long_id = "OpenRAL/rskill-pi05-openarm-bimanual-pick-pipe-nf4-variant-x"
    name = _skill_id_to_tool_name(long_id)
    assert len(name) <= 64
    assert name.startswith(_PER_SKILL_TOOL_PREFIX)


def test_skill_id_to_tool_name_collision_resistant() -> None:
    """Two long ids with the same prefix produce distinct tool names (sha1 suffix)."""
    a = "OpenRAL/rskill-pi05-openarm-bimanual-pick-pipe-very-long-variant-a"
    b = "OpenRAL/rskill-pi05-openarm-bimanual-pick-pipe-very-long-variant-b"
    assert _skill_id_to_tool_name(a) != _skill_id_to_tool_name(b)


def test_format_skill_tool_description_includes_structured_tags() -> None:
    """The LLM description carries actions / objects / scenes verbatim."""
    palette = _real_aloha_palette()
    entry = next(e for e in palette.skills if e.rskill_id.endswith("act-aloha-insertion"))
    text = _format_skill_tool_description(entry)
    assert entry.rskill_id in text
    assert entry.description.strip() in text
    assert "insert" in text.lower()
    assert "peg" in text.lower()
    assert "tabletop" in text.lower()


# ── tool schema construction ───────────────────────────────────────────────────


def test_palette_with_skills_emits_one_tool_per_skill_plus_three() -> None:
    """N skills → N execute_rskill__* tools + 3 always-present tools."""
    palette = _real_aloha_palette()
    tools = _tool_palette_to_anthropic_tools(palette)
    skill_tools = [t for t in tools if str(t["name"]).startswith(_PER_SKILL_TOOL_PREFIX)]
    assert len(skill_tools) == len(palette.skills) >= 1
    names = {t["name"] for t in tools}
    assert "reload_gst_pipeline" in names
    assert "lifecycle_transition" in names
    assert "emit_prompt" in names
    # No single execute_rskill fallback when skills is populated:
    assert "execute_rskill" not in names


def test_per_skill_tool_schema_drops_skill_id() -> None:
    """The LLM never sees a ``rskill_id`` field on per-skill tools."""
    palette = _real_aloha_palette()
    tools = _tool_palette_to_anthropic_tools(palette)
    for t in tools:
        if not str(t["name"]).startswith(_PER_SKILL_TOOL_PREFIX):
            continue
        schema = t["input_schema"]
        assert isinstance(schema, dict)
        props = schema.get("properties", {})
        assert "rskill_id" not in props, f"{t['name']}: rskill_id leaked into per-skill schema"
        required = schema.get("required", [])
        assert "rskill_id" not in required, f"{t['name']}: rskill_id still required"


def test_drop_property_helper_idempotent() -> None:
    """Re-running ``_drop_property`` on an already-clean schema is a no-op."""
    schema = ExecuteRskillTool.model_json_schema()
    once = _drop_property(schema, "rskill_id")
    twice = _drop_property(once, "rskill_id")
    assert twice == once
    assert "rskill_id" not in once.get("properties", {})


# ── id-only palette path ──────────────────────────────────────────────────────


def test_id_only_palette_collapses_to_single_execute_skill() -> None:
    """Palettes with only ``execute_rskill_ids`` (no per-skill metadata) emit a single tool."""
    palette = ToolPalette(execute_rskill_ids=frozenset({"openral/id-only"}))
    tools = _tool_palette_to_anthropic_tools(palette)
    names = [t["name"] for t in tools]
    assert "execute_rskill" in names
    assert not any(str(n).startswith(_PER_SKILL_TOOL_PREFIX) for n in names)
    # And the description still embeds the allowed id list:
    execute_tool = next(t for t in tools if t["name"] == "execute_rskill")
    assert "openral/id-only" in str(execute_tool["description"])


def test_empty_palette_omits_execute_skill_entirely() -> None:
    """No skills and no legacy ids → no execute_rskill tool at all (only the fixed scaffold)."""
    palette = ToolPalette()
    tools = _tool_palette_to_anthropic_tools(palette)
    names = {t["name"] for t in tools}
    # The always-present scaffold: three plumbing tools plus the ADR-0073 (#123)
    # decompose_mission ledger editor (a core S2 capability, no resident-resource dep).
    assert names == {
        "reload_gst_pipeline",
        "lifecycle_transition",
        "emit_prompt",
        "decompose_mission",
    }


# ── decoder round-trip ─────────────────────────────────────────────────────────


def test_decode_resolves_per_skill_tool_name_to_canonical_skill_id() -> None:
    """``execute_rskill__<slug>`` decodes to ``ExecuteRskillTool(rskill_id=<id>)``."""
    palette = _real_aloha_palette()
    entry = palette.skills[0]
    tool_name = _skill_id_to_tool_name(entry.rskill_id)
    call = _decode_tool_payload(
        tool_name=tool_name,
        arguments={"prompt": "go", "deadline_s": 0.0},
        palette=palette,
    )
    assert call.tool == "execute_rskill"
    assert call.rskill_id == entry.rskill_id


def test_decode_overrides_llm_supplied_skill_id_with_lookup() -> None:
    """If the LLM also passes a (wrong) rskill_id, the tool-name lookup wins."""
    palette = _real_aloha_palette()
    entry = palette.skills[0]
    tool_name = _skill_id_to_tool_name(entry.rskill_id)
    call = _decode_tool_payload(
        tool_name=tool_name,
        arguments={"rskill_id": "openral/some-other-skill", "prompt": ""},
        palette=palette,
    )
    assert call.rskill_id == entry.rskill_id


def test_decode_unknown_per_skill_tool_raises() -> None:
    """An ``execute_rskill__*`` name with no matching skill in the palette is rejected."""
    palette = _real_aloha_palette()
    from openral_core.exceptions import ROSReasonerInvalidPlan

    with pytest.raises(ROSReasonerInvalidPlan, match="execute_rskill__"):
        _decode_tool_payload(
            tool_name="execute_rskill__no_such_skill",
            arguments={"prompt": ""},
            palette=palette,
        )


# ── ADR-0026 — goal_params_schema surfaced to LLM tool palette ────────────────


def _palette_with_schema_for_skill(skill_id: str, schema: dict) -> ToolPalette:
    """Build a single-skill palette whose entry carries ``goal_params_schema``."""
    from openral_reasoner.palette import RSkillToolEntry

    entry = RSkillToolEntry(
        rskill_id=skill_id,
        description="navigate the base to a pose",
        actions=("navigate",),
        objects=(),
        scenes=("indoor",),
        goal_params_schema=schema,
    )
    return ToolPalette(skills=(entry,))


def test_anthropic_tool_input_schema_substitutes_per_skill_goal_params() -> None:
    """ADR-0026 — when the entry declares ``goal_params_schema``, the per-skill
    tool's ``goal_params_json`` property is replaced with the structured schema.
    """
    nav_schema = {
        "type": "object",
        "properties": {
            "pose": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["x", "y"],
                    },
                },
                "required": ["position"],
            },
        },
        "required": ["pose"],
    }
    palette = _palette_with_schema_for_skill("openral/test-nav", nav_schema)
    tools = _tool_palette_to_anthropic_tools(palette)
    per_skill = next(t for t in tools if t["name"].startswith(_PER_SKILL_TOOL_PREFIX))
    schema = per_skill["input_schema"]
    assert isinstance(schema, dict)
    props = schema["properties"]
    assert isinstance(props, dict)
    # The default ``{"type": "string"}`` for goal_params_json is replaced
    # with the nav-specific structured schema.
    assert props["goal_params_json"] == nav_schema


def test_anthropic_tool_keeps_string_schema_when_no_goal_params_schema_declared() -> None:
    """No declared schema → the LLM sees the default ``string`` surface (back-compat)."""
    from openral_reasoner.palette import RSkillToolEntry

    entry = RSkillToolEntry(
        rskill_id="openral/test-vla",
        description="VLA without structured params",
        actions=("pick_and_place",),
    )
    palette = ToolPalette(skills=(entry,))
    tools = _tool_palette_to_anthropic_tools(palette)
    per_skill = next(t for t in tools if t["name"].startswith(_PER_SKILL_TOOL_PREFIX))
    schema = per_skill["input_schema"]
    assert isinstance(schema, dict)
    props = schema["properties"]
    assert isinstance(props, dict)
    # Default ExecuteRskillTool surface — the field is a freeform string.
    assert props["goal_params_json"]["type"] == "string"


def test_decode_serialises_structured_goal_params_back_to_string() -> None:
    """ADR-0026 — when the LLM emits ``goal_params_json`` as a dict (provider's
    parsed structured output), the decoder JSON-stringifies it back to the
    Pydantic ``str`` field before constructing ``ExecuteRskillTool``.
    """
    palette = _palette_with_schema_for_skill(
        "openral/test-nav",
        {"type": "object", "properties": {"x": {"type": "number"}}},
    )
    entry = palette.skills[0]
    tool_name = _skill_id_to_tool_name(entry.rskill_id)
    call = _decode_tool_payload(
        tool_name=tool_name,
        arguments={
            "prompt": "go",
            # LLM returned a structured object, not a string.
            "goal_params_json": {"pose": {"position": {"x": 1.5, "y": -2.0}}},
            "deadline_s": 0.0,
        },
        palette=palette,
    )
    import json as _json

    # The wire-format str carries the canonical re-serialised JSON.
    parsed = _json.loads(call.goal_params_json)
    assert parsed == {"pose": {"position": {"x": 1.5, "y": -2.0}}}


def test_decode_passes_through_string_goal_params_unchanged() -> None:
    """When the LLM emits ``goal_params_json`` as a string, no re-serialisation."""
    palette = _palette_with_schema_for_skill("openral/test-nav", {"type": "object"})
    entry = palette.skills[0]
    tool_name = _skill_id_to_tool_name(entry.rskill_id)
    raw = '{"pose": {"position": {"x": 3.14}}}'
    call = _decode_tool_payload(
        tool_name=tool_name,
        arguments={"prompt": "", "goal_params_json": raw, "deadline_s": 0.0},
        palette=palette,
    )
    assert call.goal_params_json == raw
