"""Per-tick top-k skill subset for small reasoner models (#440, #494).

Real ``franka_panda`` palette built from the repo's ``rskills/*/rskill.yaml``
manifests, real ``ContextRenderer`` / ``MissionState``, real wire rendering and
decode. The LLM endpoint is the ``FakeToolUseClient`` process-boundary fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from hypothesis import given
from hypothesis import strategies as st
from openral_core import (
    REASONER_MODELS,
    ExecuteRskillTool,
    RobotDescription,
    RSkillManifest,
)
from openral_core.exceptions import ROSConfigError, ROSReasonerInvalidPlan
from openral_reasoner import (
    AnthropicToolUseClient,
    ContextRenderer,
    MissionState,
    OpenAICompatibleToolUseClient,
    PreparedTick,
    PromptRecord,
    ReasonerCore,
    SkillSubset,
    ToolPalette,
    build_tool_palette,
    resolve_max_skill_tools,
    select_skill_subset,
)
from openral_reasoner.palette import RSkillToolEntry
from openral_reasoner.tool_use import (
    REASONER_MAX_SKILL_TOOLS_ENV,
    _decode_tool_payload,
    _skill_id_to_tool_name,
    _tool_palette_to_openai_tools,
)

from tests.integration.fakes.fake_llm import FakeToolUseClient
from tests.unit.conftest import _CaptureProcessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EEF_POSE = "OpenRAL/rskill-moveit-multi-eef_pose"
_LOOK_AT = "OpenRAL/rskill-moveit-multi-look_at"
_MANISKILL = "OpenRAL/rskill-smolvla-franka_panda-maniskill3-bf16"


@pytest.fixture(scope="module")
def franka_palette() -> ToolPalette:
    """The full franka_panda palette with every query tool, as on Thor (#494)."""
    robot = RobotDescription.from_yaml(str(_REPO_ROOT / "robots/franka_panda/robot.yaml"))
    palette = build_tool_palette(
        installed_skills=[
            RSkillManifest.from_yaml(str(p))
            for p in sorted(_REPO_ROOT.glob("rskills/*/rskill.yaml"))
        ],
        robot_capabilities=robot.capabilities,
        node_ids=["/openral/hal/franka_panda"],
        detector_available=True,
        scene_query_available=True,
        task_progress_available=True,
        memory_available=True,
    )
    assert len(palette.skills) >= 10, "fixture drift: expected the franka VLA + MoveIt skills"
    assert {_EEF_POSE, _LOOK_AT, _MANISKILL} <= palette.execute_rskill_ids, "fixture drift"
    return palette


# ── select_skill_subset ──────────────────────────────────────────────────────


def test_subset_caps_skills_and_keeps_every_other_tool(franka_palette: ToolPalette) -> None:
    """Only the skill tools shrink; query tools, lifecycle peers and detectors stay."""
    subset = select_skill_subset(
        franka_palette, query="pick up the black bowl and place it on the plate", k=4
    )
    assert len(subset.offered) == 4
    assert subset.palette.execute_rskill_ids == frozenset(subset.offered)
    assert subset.palette.model_dump(exclude={"skills", "execute_rskill_ids"}) == (
        franka_palette.model_dump(exclude={"skills", "execute_rskill_ids"})
    )
    full = {t["function"]["name"] for t in _tool_palette_to_openai_tools(franka_palette)}  # type: ignore[index]  # reason: OpenAI tool dicts are untyped
    narrowed = {t["function"]["name"] for t in _tool_palette_to_openai_tools(subset.palette)}  # type: ignore[index]  # reason: as above
    skill_tools = {n for n in full if n.startswith("execute_rskill__")}
    assert full - skill_tools == narrowed - skill_tools
    assert len(narrowed & skill_tools) == 4
    # Every skill is scored, best first, so a skill left out is diagnosable.
    assert {i for i, _ in subset.scores} == franka_palette.execute_rskill_ids
    assert [s for _, s in subset.scores] == sorted((s for _, s in subset.scores), reverse=True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("move the end-effector to the pre-grasp pose", _EEF_POSE),
        ("point the wrist camera at the mug", _LOOK_AT),
        ("pick up the cube", _MANISKILL),
    ],
)
def test_subset_ranks_the_matching_skill_first(
    franka_palette: ToolPalette, query: str, expected: str
) -> None:
    """The skill whose objects / verbs / prose match the task is offered."""
    subset = select_skill_subset(franka_palette, query=query, k=4)
    assert subset.scores[0][0] == expected
    assert expected in subset.offered


def test_libero_task_offers_libero_skills(franka_palette: ToolPalette) -> None:
    """A LIBERO pick-and-place task gets pick/place/bowl skills, not the MoveIt ones."""
    subset = select_skill_subset(
        franka_palette, query="pick up the black bowl and place it on the plate", k=4
    )
    for rskill_id in set(subset.offered) - set(subset.support):
        entry = next(e for e in franka_palette.skills if e.rskill_id == rskill_id)
        assert "bowl" in entry.objects
    assert _EEF_POSE not in subset.offered


def test_sticky_skill_is_kept_and_unknown_sticky_ignored(franka_palette: ToolPalette) -> None:
    """The in-flight skill stays offered even when it scores nothing for the task."""
    subset = select_skill_subset(
        franka_palette,
        query="pick up the black bowl and place it on the plate",
        k=4,
        sticky=(_LOOK_AT, "OpenRAL/rskill-dropped-after-failure", None),
    )
    assert _LOOK_AT in subset.offered
    assert subset.sticky == (_LOOK_AT,)
    assert len(subset.offered) == 4


def test_subset_is_deterministic_and_palette_ordered(franka_palette: ToolPalette) -> None:
    """Same input, same subset; skill order follows the full palette (stable tool prefix)."""
    a = select_skill_subset(franka_palette, query="open the top drawer", k=4)
    b = select_skill_subset(franka_palette, query="open the top drawer", k=4)
    assert a == b
    order = [e.rskill_id for e in franka_palette.skills]
    assert list(a.offered) == sorted(a.offered, key=order.index)


def test_small_palette_is_returned_unchanged(franka_palette: ToolPalette) -> None:
    """A palette within the cap is offered whole, as the same object."""
    subset = select_skill_subset(franka_palette, query="anything", k=len(franka_palette.skills))
    assert subset.palette is franka_palette


def test_k_below_one_is_refused(franka_palette: ToolPalette) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        select_skill_subset(franka_palette, query="pick", k=0)


def test_call_to_a_skill_outside_the_subset_is_refused(franka_palette: ToolPalette) -> None:
    """Decode validates against the offered subset, like any off-palette id."""
    subset = select_skill_subset(franka_palette, query="pick up the cube", k=4)
    dropped = next(i for i in sorted(franka_palette.execute_rskill_ids) if i not in subset.offered)
    with pytest.raises(ROSReasonerInvalidPlan, match="no matching skill"):
        _decode_tool_payload(
            tool_name=_skill_id_to_tool_name(dropped),
            arguments={"prompt": "pick up the cube"},
            palette=subset.palette,
        )
    offered = subset.offered[0]
    call = _decode_tool_payload(
        tool_name=_skill_id_to_tool_name(offered),
        arguments={"prompt": "pick up the cube"},
        palette=subset.palette,
    )
    assert isinstance(call, ExecuteRskillTool)
    assert call.rskill_id == offered


def test_four_franka_skills_match_the_thor_measured_prompt_size(
    franka_palette: ToolPalette,
) -> None:
    """The capped franka palette is about the size measured at 19/20 valid on Thor.

    #494: franka + query tools with 4 of its 14 skills was 11,276 Edge tokens
    (19/20); the full palette was 25,191 (0/20). The capped palette must be well
    under half the full one; characters, so no tokenizer download.
    """
    subset = select_skill_subset(franka_palette, query="pick up the cube", k=4)
    full = len(json.dumps(_tool_palette_to_openai_tools(franka_palette)))
    capped = len(json.dumps(_tool_palette_to_openai_tools(subset.palette)))
    assert capped < full / 2


# ── Registry + env ───────────────────────────────────────────────────────────


def test_only_cosmos3_edge_is_capped_in_the_registry() -> None:
    """Large curated models keep the whole palette; Edge gets the measured k=4."""
    assert REASONER_MODELS["cosmos3-edge"].max_skill_tools == 4
    for model_id, entry in REASONER_MODELS.items():
        if model_id != "cosmos3-edge":
            assert entry.max_skill_tools is None, model_id


@pytest.mark.parametrize(
    ("client", "env", "expected"),
    [
        (OpenAICompatibleToolUseClient(model_id="nvidia/Cosmos3-Edge"), None, 4),
        (AnthropicToolUseClient(model_id="claude-opus-4-8", api_key="unused"), None, None),
        (OpenAICompatibleToolUseClient(model_id="nvidia/Cosmos3-Edge"), "0", None),
        (OpenAICompatibleToolUseClient(model_id="nvidia/Cosmos3-Edge"), "8", 8),
        (OpenAICompatibleToolUseClient(model_id="gpt-oss:20b"), "6", 6),  # uncurated: env only
        (OpenAICompatibleToolUseClient(model_id="gpt-oss:20b"), None, None),
        (FakeToolUseClient(responses=[]), None, None),  # an injected client
    ],
    ids=["edge", "claude", "edge-env-0", "edge-env-8", "uncurated-env", "uncurated", "fake"],
)
def test_resolve_max_skill_tools(
    monkeypatch: pytest.MonkeyPatch, client: object, env: str | None, expected: int | None
) -> None:
    """The cap follows the client actually built, not OPENRAL_REASONER_MODEL."""
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "cosmos3-edge")  # must not leak into the cap
    if env is None:
        monkeypatch.delenv(REASONER_MAX_SKILL_TOOLS_ENV, raising=False)
    else:
        monkeypatch.setenv(REASONER_MAX_SKILL_TOOLS_ENV, env)
    assert resolve_max_skill_tools(client) == expected  # type: ignore[arg-type]  # reason: every parametrized client satisfies ToolUseClient


@pytest.mark.parametrize("bad", ["-1", "four"])
def test_bad_max_skill_tools_env_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv(REASONER_MAX_SKILL_TOOLS_ENV, bad)
    with pytest.raises(ROSConfigError, match=REASONER_MAX_SKILL_TOOLS_ENV):
        resolve_max_skill_tools(FakeToolUseClient(responses=[]))


# ── ReasonerCore ─────────────────────────────────────────────────────────────


def _mission_renderer(*tasks: str) -> ContextRenderer:
    renderer = ContextRenderer()
    renderer.set_mission(MissionState(list(tasks)))
    return renderer


def test_core_offers_the_subset_and_reuses_it_across_ticks(franka_palette: ToolPalette) -> None:
    """The LLM sees the capped palette; the same object while the task holds."""
    client = FakeToolUseClient(responses=[])
    core = ReasonerCore(client=client, min_interval_s=0.0, max_skill_tools=4)
    renderer = _mission_renderer("pick up the cube", "point the wrist camera at the mug")
    first = core.prepare_tick(world_state=None, renderer=renderer, palette=franka_palette)
    assert isinstance(first, PreparedTick)
    offered = first.palette
    assert len(offered.skills) == 4
    assert _MANISKILL in offered.execute_rskill_ids
    core.finish_tick(first, error=ROSReasonerInvalidPlan("not run"))
    assert core.offered_palette(franka_palette, renderer) is offered
    # The in-flight skill was offered, so the offered set and object are unchanged.
    renderer.set_inflight_skill(_MANISKILL)
    assert core.offered_palette(franka_palette, renderer) is offered
    # The next task re-ranks; the in-flight skill stays.
    renderer.advance_mission(done=True, verdict="success=0.9")
    nxt = core.offered_palette(franka_palette, renderer)
    assert nxt is not offered
    assert {_LOOK_AT, _MANISKILL} <= nxt.execute_rskill_ids


def test_core_ranks_on_the_pending_prompt_without_a_mission(franka_palette: ToolPalette) -> None:
    core = ReasonerCore(client=FakeToolUseClient(responses=[]), max_skill_tools=2)
    renderer = ContextRenderer()
    renderer.append_prompt(
        PromptRecord(
            text="move the end-effector to the pre-grasp pose", metadata_json="", stamp_ns=0
        )
    )
    assert _EEF_POSE in core.offered_palette(franka_palette, renderer).execute_rskill_ids


def test_core_without_a_cap_offers_the_whole_palette(franka_palette: ToolPalette) -> None:
    """Models without a cap see an unchanged palette (#494 acceptance)."""
    core = ReasonerCore(client=FakeToolUseClient(responses=[]), min_interval_s=0.0)
    renderer = _mission_renderer("pick up the cube")
    prep = core.prepare_tick(world_state=None, renderer=renderer, palette=franka_palette)
    assert isinstance(prep, PreparedTick)
    assert prep.palette is franka_palette
    assert core.max_skill_tools is None


def test_core_tick_dispatches_from_the_subset(franka_palette: ToolPalette) -> None:
    """End to end: a call to an offered skill dispatches; the client saw only k skills."""
    client = FakeToolUseClient(
        responses=[ExecuteRskillTool(rskill_id=_MANISKILL, prompt="pick up the cube")]
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, max_skill_tools=4)
    result = core.tick(
        world_state=None, renderer=_mission_renderer("pick up the cube"), palette=franka_palette
    )
    assert isinstance(result.tool_call, ExecuteRskillTool)
    assert len(client.traces[0].palette.skills) == 4


def test_core_rejects_a_bad_cap() -> None:
    with pytest.raises(ValueError, match="max_skill_tools"):
        ReasonerCore(client=FakeToolUseClient(responses=[]), max_skill_tools=0)


# ── Review follow-ups ────────────────────────────────────────────────────────


def test_look_at_keeps_a_slot_on_an_object_task(franka_palette: ToolPalette) -> None:
    """The system prompt FRAMEs a target with look-at before grasping; the task never says so."""
    subset = select_skill_subset(
        franka_palette, query="pick up the black bowl and place it on the plate", k=4
    )
    assert _LOOK_AT in subset.offered
    assert subset.support == (_LOOK_AT,)
    assert len(subset.offered) == 4
    # The other three are the bowl skills.
    for rskill_id in set(subset.offered) - {_LOOK_AT}:
        entry = next(e for e in franka_palette.skills if e.rskill_id == rskill_id)
        assert "bowl" in entry.objects


def test_support_never_takes_every_slot(franka_palette: ToolPalette) -> None:
    subset = select_skill_subset(franka_palette, query="pick up the cube", k=1)
    assert subset.offered == (_MANISKILL,)
    assert subset.support == ()


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ("start by picking up the cubes", "pick"),
        ("put the boxes on the shelf", "box"),
        ("grab the knives", "knife"),
        ("Nimm die Schüssel", "schüssel"),
    ],
)
def test_inflections_and_non_ascii_match(query: str, match: str) -> None:
    """Plurals, -ing forms and non-ASCII words still match the manifest's base form."""
    from openral_core import RSkillAction

    target = "OpenRAL/rskill-target"
    palette = ToolPalette(
        skills=(
            RSkillToolEntry(
                rskill_id=target,
                description="target",
                actions=(RSkillAction.PICK,) if match == "pick" else (RSkillAction.WAVE,),
                objects=() if match == "pick" else (match,),
            ),
            *(
                RSkillToolEntry(
                    rskill_id=f"OpenRAL/rskill-a{i}",
                    description="other",
                    actions=(RSkillAction.WAVE,),
                )
                for i in range(3)
            ),
        )
    )
    subset = select_skill_subset(palette, query=query, k=1)
    assert subset.offered == (target,)
    assert subset.max_score >= 3.0


def test_id_only_palette_is_capped_too() -> None:
    """The single-``execute_rskill``-tool fallback palette obeys the cap as well."""
    ids = frozenset(f"OpenRAL/rskill-skill{i}" for i in range(10)) | {"OpenRAL/rskill-cube-picker"}
    palette = ToolPalette(execute_rskill_ids=ids)
    subset = select_skill_subset(palette, query="use the cube picker", k=3)
    assert len(subset.palette.execute_rskill_ids) == 3
    assert "OpenRAL/rskill-cube-picker" in subset.offered


def test_operator_reply_mid_mission_changes_the_offer(franka_palette: ToolPalette) -> None:
    """A pending operator prompt is ranked with the active task, at any priority."""
    core = ReasonerCore(client=FakeToolUseClient(responses=[]), max_skill_tools=4)
    renderer = _mission_renderer("pick up the black bowl and place it on the plate")
    before = core.offered_palette(franka_palette, renderer)
    assert _EEF_POSE not in before.execute_rskill_ids
    renderer.append_prompt(
        PromptRecord(
            text="first move the end-effector to the pre-grasp pose",
            metadata_json='{"priority": 100}',
            stamp_ns=0,
        )
    )
    renderer.append_prompt(
        PromptRecord(text="perception: a mug appeared", metadata_json="", stamp_ns=1)
    )
    after = core.offered_palette(franka_palette, renderer)
    assert _EEF_POSE in after.execute_rskill_ids
    assert _LOOK_AT in after.execute_rskill_ids  # the workflow slot survives


def test_every_recompute_is_logged(franka_palette: ToolPalette) -> None:
    """A new task logs its ranking even when the same skills win (replayable trace)."""
    import openral_reasoner.core as core_mod

    proc = _CaptureProcessor()
    structlog.reset_defaults()
    structlog.configure(processors=[proc])
    old_log = core_mod.log
    core_mod.log = structlog.get_logger(core_mod.__name__)
    try:
        core = ReasonerCore(client=FakeToolUseClient(responses=[]), max_skill_tools=4)
        renderer = _mission_renderer(
            "pick up the black bowl and place it on the plate", "place the bowl on the plate"
        )
        first = core.offered_palette(franka_palette, renderer)
        core.offered_palette(franka_palette, renderer)  # cached: no second log
        renderer.advance_mission(done=True, verdict="success=0.9")
        second = core.offered_palette(franka_palette, renderer)
    finally:
        core_mod.log = old_log
        structlog.reset_defaults()
    assert second is first  # same offered set, same object (prefix cache holds)
    events = [kw for name, kw in proc.events if name == "reasoner.skill_subset"]
    assert [e["query"] for e in events] == [
        "pick up the black bowl and place it on the plate",
        "place the bowl on the plate",
    ]
    assert set(events[1]["scores"]) == franka_palette.execute_rskill_ids


@given(
    k=st.integers(min_value=1, max_value=6),
    query=st.text(max_size=60),
    n=st.integers(min_value=0, max_value=8),
)
def test_skill_subset_round_trips_and_respects_k(k: int, query: str, n: int) -> None:
    """Fuzz: any query and cap give at most k offered skills; JSON round-trip and schema hold."""
    from openral_core import RSkillAction

    palette = ToolPalette(
        skills=tuple(
            RSkillToolEntry(
                rskill_id=f"OpenRAL/rskill-s{i}",
                description=f"skill {i}",
                actions=(list(RSkillAction)[i % len(RSkillAction)],),
            )
            for i in range(n)
        )
    )
    subset = select_skill_subset(palette, query=query, k=k)
    assert len(subset.offered) == min(k, n)
    assert subset.palette.execute_rskill_ids == frozenset(subset.offered)
    assert SkillSubset.model_validate_json(subset.model_dump_json()) == subset
    assert SkillSubset.model_json_schema()["title"] == "SkillSubset"
