"""Reasoner clients on models that refuse forced tool use (#522).

The Claude 5 generation answers ``tool_choice: {"type": "any"}`` with HTTP 400
(``tool_choice: type "tool" and "any" are not supported for this model.``), and
OpenAI's own API refuses ``max_tokens`` on its reasoning models. These tests
drive the real ``AnthropicToolUseClient`` / ``OpenAICompatibleToolUseClient``
through the real ``anthropic`` / ``openai`` SDKs against a scripted loopback
HTTP server (``tests/unit/fakes/fake_llm_http.py`` — the network-boundary double
CLAUDE.md §1.11 allows) and check:

1. The curated Claude 5 entries run ``tool_choice="auto"`` with room for thinking.
2. A forced-tool-use refusal downgrades ``required`` to ``auto`` once, logged,
   and the client stays on ``auto`` — but only when the ``auto`` retry
   succeeds, and never for an unrelated 400 that merely mentions ``tool_choice``.
3. Under ``auto`` a prose reply is nudged once, then raised as an invalid plan
   the node feeds back to the model; prompt tokens cover both calls.
4. A refusal or a reply cut off at the token cap is a ``ROSPlanningError``.
5. OpenAI's ``max_completion_tokens`` demand is met at runtime, and the
   ``openai`` named endpoint sends it (and the bare model id) up front.
6. ``OPENRAL_REASONER_TOOL_CHOICE`` applies to the anthropic dialect.
7. The curated skill-tool cap survives the ``openai`` endpoint's bare model id.
"""

from __future__ import annotations

import pytest
from openral_core import REASONER_MODELS, ExecuteRskillTool, RSkillAction, WaitTool
from openral_core.exceptions import ROSPlanningError, ROSReasonerInvalidPlan
from openral_reasoner.palette import RSkillToolEntry, ToolPalette
from openral_reasoner.tool_use import (
    OPENAI_BASE_URL,
    AnthropicToolUseClient,
    OpenAICompatibleToolUseClient,
    _skill_id_to_tool_name,
    build_tool_use_client_from_env,
)
from structlog.testing import capture_logs

from tests.unit.fakes.fake_llm_http import (
    FakeLLMServer,
    anthropic_error,
    anthropic_text,
    anthropic_tool_use,
    openai_error,
    openai_text,
    openai_tool_call,
)

pytest.importorskip("anthropic")
pytest.importorskip("openai")

_SKILL_ID = "OpenRAL/rskill-smolvla-so100-pick-fp16"
_CLAUDE5_REFUSAL = 'tool_choice: type "tool" and "any" are not supported for this model.'
_OPENAI_MAX_TOKENS_REFUSAL = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)
_CLAUDE5 = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")


def _palette() -> ToolPalette:
    return ToolPalette(
        skills=(
            RSkillToolEntry(
                rskill_id=_SKILL_ID,
                description="pick objects on a tabletop",
                actions=(RSkillAction.PICK,),
                objects=("cube",),
                scenes=("tabletop",),
            ),
        )
    )


def _anthropic(server: FakeLLMServer, **kwargs: object) -> AnthropicToolUseClient:
    return AnthropicToolUseClient(
        model_id="claude-opus-5-5",
        api_key="sk-ant-test",
        base_url=server.base_url,
        **kwargs,  # type: ignore[arg-type]  # reason: test forwards optional ctor kwargs
    )


def _openai(server: FakeLLMServer, **kwargs: object) -> OpenAICompatibleToolUseClient:
    return OpenAICompatibleToolUseClient(
        model_id="gpt-5.5",
        api_key="sk-test",
        base_url=server.openai_base_url,
        **kwargs,  # type: ignore[arg-type]  # reason: test forwards optional ctor kwargs
    )


# ── 1. registry ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_key", _CLAUDE5)
def test_claude5_entries_run_auto_with_thinking_headroom(model_key: str) -> None:
    entry = REASONER_MODELS[model_key]
    assert entry.dialect == "anthropic"
    assert entry.served_model_id == model_key
    assert entry.tool_choice == "auto"
    assert entry.max_tokens_default is not None and entry.max_tokens_default >= 8192


@pytest.mark.parametrize("model_key", _CLAUDE5)
def test_curated_claude5_builds_an_auto_client(
    monkeypatch: pytest.MonkeyPatch, model_key: str
) -> None:
    for var in ("OPENRAL_REASONER_ENDPOINT", "OPENRAL_REASONER_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", model_key)
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-ant-test")
    with capture_logs() as logs:
        client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client.model_id == model_key
    assert client._tool_choice == "auto"
    assert client._max_tokens == REASONER_MODELS[model_key].max_tokens_default
    assert not [e for e in logs if e["event"] == "reasoner.model.uncurated"]


# ── 2. runtime downgrade ─────────────────────────────────────────────────────


def test_anthropic_forced_tool_choice_400_downgrades_to_auto() -> None:
    replies = [
        anthropic_error(_CLAUDE5_REFUSAL),
        anthropic_tool_use("wait", {"rationale": "skill in flight"}),
        anthropic_tool_use("wait", {"rationale": "still in flight"}),
    ]
    with FakeLLMServer(replies) as server, capture_logs() as logs:
        client = _anthropic(server)
        first = client.select_tool(context_text="tick 1", palette=_palette())
        second = client.select_tool(context_text="tick 2", palette=_palette())
    assert isinstance(first, WaitTool) and first.rationale == "skill in flight"
    assert isinstance(second, WaitTool)
    sent = [r["tool_choice"]["type"] for r in server.requests]
    assert sent == ["any", "auto", "auto"], "downgrade must stick for later ticks"
    assert all(r["tool_choice"]["disable_parallel_tool_use"] for r in server.requests)
    downgraded = [e for e in logs if e["event"] == "reasoner.tool_choice_downgraded"]
    assert len(downgraded) == 1
    assert client.last_prompt_tokens == 11


def test_anthropic_unrelated_400_is_not_retried() -> None:
    with FakeLLMServer([anthropic_error("messages: field required")]) as server:
        client = _anthropic(server)
        with pytest.raises(ROSPlanningError, match="field required"):
            client.select_tool(context_text="tick", palette=_palette())
    assert len(server.requests) == 1
    assert client._tool_choice == "required"


@pytest.mark.parametrize(
    "message",
    [
        # Validation error about another field that echoes the request body.
        "1 validation error: max_tokens: input_value={'tool_choice': 'required', 'tools': [...]}",
        # A tool_choice sub-field the endpoint does not know.
        "tool_choice.disable_parallel_tool_use: Extra inputs are not permitted",
    ],
)
def test_unrelated_400_mentioning_tool_choice_does_not_downgrade(message: str) -> None:
    with FakeLLMServer([anthropic_error(message)]) as server:
        client = _anthropic(server)
        with pytest.raises(ROSPlanningError):
            client.select_tool(context_text="tick", palette=_palette())
    assert len(server.requests) == 1
    assert client._tool_choice == "required"


def test_failed_auto_retry_keeps_required() -> None:
    """The downgrade sticks only when a call made with ``auto`` succeeds."""
    replies = [anthropic_error(_CLAUDE5_REFUSAL), anthropic_error("overloaded", status=400)]
    with FakeLLMServer(replies) as server, capture_logs() as logs:
        client = _anthropic(server)
        with pytest.raises(ROSPlanningError, match="overloaded"):
            client.select_tool(context_text="tick", palette=_palette())
    assert [r["tool_choice"]["type"] for r in server.requests] == ["any", "auto"]
    assert client._tool_choice == "required"
    assert not [e for e in logs if e["event"] == "reasoner.tool_choice_downgraded"]


def test_openai_forced_tool_choice_400_downgrades_to_auto() -> None:
    replies = [
        openai_error("Thinking mode does not support this tool_choice"),
        openai_tool_call(_skill_id_to_tool_name(_SKILL_ID), {"prompt": "pick the cube"}),
    ]
    with FakeLLMServer(replies) as server, capture_logs() as logs:
        client = _openai(server)
        call = client.select_tool(context_text="tick", palette=_palette())
    assert isinstance(call, ExecuteRskillTool)
    assert call.rskill_id == _SKILL_ID
    assert [r["tool_choice"] for r in server.requests] == ["required", "auto"]
    assert [e for e in logs if e["event"] == "reasoner.tool_choice_downgraded"]


# ── 3. prose reply under auto ────────────────────────────────────────────────


def test_anthropic_prose_reply_is_nudged_then_succeeds() -> None:
    replies = [
        anthropic_text("Let me think about the cube."),
        anthropic_tool_use("wait", {"rationale": "nothing to do"}),
    ]
    with FakeLLMServer(replies) as server:
        client = _anthropic(server, tool_choice="auto")
        call = client.select_tool(context_text="tick", palette=_palette())
    assert isinstance(call, WaitTool) and call.rationale == "nothing to do"
    assert client.last_prompt_tokens == 22, "both billed calls are counted"
    nudge = server.requests[1]["messages"][0]["content"]
    assert nudge[0]["text"] == "tick"
    assert "exactly one tool call" in nudge[-1]["text"]


@pytest.mark.parametrize(
    ("replies", "factory"),
    [
        (
            [anthropic_text("The arm is still moving."), anthropic_text("Still moving.\n\nOK")],
            "anthropic",
        ),
        ([openai_text("The arm is still moving."), openai_text("Still moving.\n\nOK")], "openai"),
    ],
)
def test_second_prose_reply_is_an_invalid_plan(
    replies: list[tuple[int, dict[str, object]]], factory: str
) -> None:
    """Not a ``wait``: a ``wait`` is retry-cap exempt and sends no hint back, so a
    model stuck in prose would stall the mission silently."""
    with FakeLLMServer(replies) as server, capture_logs() as logs:
        client = (
            _anthropic(server, tool_choice="auto")
            if factory == "anthropic"
            else _openai(server, tool_choice="auto")
        )
        with pytest.raises(ROSReasonerInvalidPlan, match=r"Still moving\. OK"):
            client.select_tool(context_text="tick", palette=_palette())
    assert len(server.requests) == 2
    assert client.last_prompt_tokens == 22
    assert [e for e in logs if e["event"] == "reasoner.no_tool_call"]


def test_prose_reply_under_required_stays_an_invalid_plan() -> None:
    """A forced-tool endpoint that ignores the contract is still a typed error."""
    with FakeLLMServer([openai_text("no")]) as server, pytest.raises(ROSReasonerInvalidPlan):
        _openai(server).select_tool(context_text="tick", palette=_palette())
    assert len(server.requests) == 1


# ── 4. refusal / truncation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (anthropic_text("", stop_reason="refusal"), "refused"),
        (anthropic_text("thinking...", stop_reason="max_tokens"), "OPENRAL_REASONER_MAX_TOKENS"),
    ],
)
def test_anthropic_refusal_or_truncation_is_not_a_wait(
    reply: tuple[int, dict[str, object]], match: str
) -> None:
    with FakeLLMServer([reply]) as server, pytest.raises(ROSPlanningError, match=match):
        _anthropic(server, tool_choice="auto").select_tool(context_text="tick", palette=_palette())
    assert len(server.requests) == 1, "no nudge after a refusal or a cut-off"


@pytest.mark.parametrize(
    ("finish_reason", "match"),
    [("length", "OPENRAL_REASONER_MAX_TOKENS"), ("content_filter", "refused")],
)
def test_openai_truncation_or_filter_is_not_a_wait(finish_reason: str, match: str) -> None:
    reply = openai_text("", finish_reason=finish_reason)
    with FakeLLMServer([reply]) as server, pytest.raises(ROSPlanningError, match=match):
        _openai(server, tool_choice="auto", max_tokens=64).select_tool(
            context_text="tick", palette=_palette()
        )
    assert len(server.requests) == 1


# ── 5. OpenAI max_completion_tokens ──────────────────────────────────────────


def test_openai_max_tokens_400_switches_to_max_completion_tokens() -> None:
    replies = [
        openai_error(_OPENAI_MAX_TOKENS_REFUSAL, param="max_tokens"),
        openai_tool_call("wait", {"rationale": "ok"}),
        openai_tool_call("wait", {"rationale": "ok"}),
    ]
    with FakeLLMServer(replies) as server, capture_logs() as logs:
        client = _openai(server, max_tokens=4096)
        client.select_tool(context_text="tick 1", palette=_palette())
        client.select_tool(context_text="tick 2", palette=_palette())
    first, second, third = server.requests
    assert first["max_tokens"] == 4096 and "max_completion_tokens" not in first
    assert second["max_completion_tokens"] == 4096 and "max_tokens" not in second
    assert third["max_completion_tokens"] == 4096
    assert [e for e in logs if e["event"] == "reasoner.max_tokens_param_switched"]


def test_openai_named_endpoint_sends_bare_id_and_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENRAL_REASONER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENRAL_REASONER_TIMEOUT_S", raising=False)
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.5")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openai")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-test")
    client = build_tool_use_client_from_env()
    assert isinstance(client, OpenAICompatibleToolUseClient)
    assert client.model_id == "gpt-5.5"  # not OpenRouter's "openai/gpt-5.5"
    assert client._base_url == OPENAI_BASE_URL
    assert client._max_tokens_param == "max_completion_tokens"
    assert client._tool_choice == "required"


def test_openai_named_endpoint_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from openral_core.exceptions import ROSConfigError

    monkeypatch.delenv("OPENRAL_REASONER_API_KEY", raising=False)
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "gpt-5.6")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "openai")
    with pytest.raises(ROSConfigError, match="OPENRAL_REASONER_API_KEY"):
        build_tool_use_client_from_env()


# ── 6. env override on the anthropic dialect ─────────────────────────────────


@pytest.mark.parametrize(("env", "expected"), [(None, "required"), ("auto", "auto")])
def test_uncurated_anthropic_honours_tool_choice_env(
    monkeypatch: pytest.MonkeyPatch, env: str | None, expected: str
) -> None:
    monkeypatch.delenv("OPENRAL_REASONER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENRAL_REASONER_DIALECT", raising=False)
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-fable-5-1")
    monkeypatch.setenv("OPENRAL_REASONER_ENDPOINT", "anthropic")
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-ant-test")
    if env is None:
        monkeypatch.delenv("OPENRAL_REASONER_TOOL_CHOICE", raising=False)
    else:
        monkeypatch.setenv("OPENRAL_REASONER_TOOL_CHOICE", env)
    with capture_logs() as logs:
        client = build_tool_use_client_from_env()
    assert isinstance(client, AnthropicToolUseClient)
    assert client._tool_choice == expected
    # An uncurated Claude may think before it calls a tool.
    assert client._max_tokens >= 8192
    assert not [e for e in logs if e["event"] == "reasoner.tool_choice_ignored"]


# ── 7. skill-tool cap on a vendor's own endpoint ─────────────────────────────


def test_skill_cap_lookup_matches_a_vendor_stripped_id() -> None:
    """``gpt-5.5`` on the ``openai`` endpoint resolves the same curated entry."""
    from openral_reasoner.tool_use import resolve_max_skill_tools

    cosmos = REASONER_MODELS["cosmos3-edge"]
    assert cosmos.max_skill_tools is not None
    bare = cosmos.served_model_id.split("/", 1)[-1]
    client = OpenAICompatibleToolUseClient(model_id=bare, base_url="http://127.0.0.1:1")
    assert resolve_max_skill_tools(client) == cosmos.max_skill_tools
