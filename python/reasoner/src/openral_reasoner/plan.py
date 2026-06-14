"""Plan and ToolCall — the typed LLM output the reasoner emits.

The LLM does **not** emit BehaviorTree XML directly; it emits a
:class:`Plan` and the reasoner converts the Plan to BT.CPP v4 XML
deterministically (ADR-0005). The BT XML is the executable artifact
the S1 executor consumes; the Plan is the auditable artifact the trace
records.

This module intentionally ships only the Pydantic models. The
``Plan.to_bt_xml`` converter and the registry-driven skill-vocabulary
check are deferred to the first concrete :class:`~openral_reasoner.Reasoner`
implementation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Plan", "ToolCall"]


class ToolCall(BaseModel):
    """One leaf of a :class:`Plan` — a single skill invocation.

    A :class:`ToolCall` is the unit the LLM's tool palette emits. The
    rskill_id must resolve against the local skill registry at
    :meth:`~openral_reasoner.Reasoner.plan` time; an unknown rskill_id
    raises :class:`openral_core.exceptions.ROSReasonerInvalidPlan`
    (per ADR-0005 #3) — there is no silent fallback.

    Attributes:
        rskill_id: The :attr:`RSkillManifest.name` of an installed,
            capable skill. Validated against the local registry by the
            reasoner.
        params: Free-form parameters forwarded to the skill. Concrete
            schemas live on each skill; the reasoner only enforces
            JSON-serialisability here.
        rationale: Optional one-line LLM rationale, recorded on the
            span and the trace for auditability.

    Example:
        >>> tc = ToolCall(rskill_id="pick_cube_so100", params={"color": "red"})
        >>> tc.rskill_id
        'pick_cube_so100'
    """

    model_config = ConfigDict(extra="forbid")

    rskill_id: str = Field(
        ...,
        description="The RSkillManifest.name of the leaf to call. Must be installed and capable.",
        min_length=1,
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serialisable parameters forwarded to the skill's ExecuteSkill goal.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional LLM rationale recorded on the reasoner span for auditability.",
    )


class Plan(BaseModel):
    """The structured LLM output the reasoner emits per planning tick.

    A :class:`Plan` is the input to the deterministic BT XML emitter
    (deferred to the first concrete :class:`~openral_reasoner.Reasoner`
    PR). The LLM produces this object via its provider's
    structured-output mode (OpenAI ``response_format=Plan``, Anthropic
    tools); the reasoner validates it eagerly and converts to BT XML.

    Attributes:
        goal: Natural-language goal the planner is satisfying. Recorded
            on the reasoner span and in the trace.
        tool_calls: Ordered sequence of skill invocations. An empty
            list is rejected by the converter as
            :class:`ROSReasonerInvalidPlan`.
        confidence: LLM-reported confidence in ``[0.0, 1.0]``. Used by
            the failure-anticipation gate (CLAUDE.md §6.3 pattern 2,
            deferred).
        bt_xml: Optional pre-computed BT XML. When ``None`` the
            reasoner generates it from :attr:`tool_calls`; when
            populated (e.g. by a cache or a hand-built test fixture),
            the reasoner forwards it directly. Either way the trace
            records the XML actually executed.

    Example:
        >>> p = Plan(
        ...     goal="pick the red cube",
        ...     tool_calls=[ToolCall(rskill_id="pick_cube_so100")],
        ...     confidence=0.92,
        ... )
        >>> len(p.tool_calls)
        1
    """

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        ...,
        description="Natural-language goal the planner is satisfying.",
        min_length=1,
    )
    tool_calls: list[ToolCall] = Field(
        ...,
        description="Ordered sequence of skill invocations; empty list is invalid.",
        min_length=1,
    )
    confidence: float = Field(
        ...,
        description="LLM-reported confidence in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )
    bt_xml: str | None = Field(
        default=None,
        description=(
            "Optional pre-computed BT XML. When None the reasoner emits it from tool_calls."
        ),
    )
