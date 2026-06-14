"""Reasoner and LLMClient Protocols.

The reasoner sits at CLAUDE.md §6.1 Layer 4 — the slow planning loop
that takes a :class:`~openral_core.WorldState` and a natural-language
goal and emits a :class:`~openral_reasoner.Plan` for the S1 executor.
Per ADR-0005 the LLM emits a typed :class:`Plan` and code emits the
BT XML; the Protocol below is the seam every concrete reasoner satisfies
so the rest of the runtime can compose against a locked signature.

A separate :class:`LLMClient` Protocol abstracts the wire-level provider
(OpenAI, Anthropic) so the reasoner does not import a provider's SDK
directly.

See ``docs/adr/0005-bt-llm-not-langgraph.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openral_core import WorldState

from openral_reasoner.plan import Plan

__all__ = ["LLMClient", "Reasoner"]


@runtime_checkable
class LLMClient(Protocol):
    """Wire-level Protocol for an LLM provider that supports structured output.

    Implementations call into OpenAI's ``response_format=Plan``,
    Anthropic's tools API, or any other provider that can enforce a
    JSON Schema at the wire. The reasoner consumes a fully-validated
    :class:`Plan`; the parsing / validation lives behind this Protocol
    so providers can swap freely.

    Attributes:
        model_id: Provider-specific identifier (``gpt-4o``,
            ``claude-sonnet-4-6``, ...). Recorded on the reasoner span.
    """

    model_id: str

    def complete_structured(self, prompt: str, schema: type[Plan]) -> Plan:
        """Emit a :class:`Plan` for ``prompt``, enforced against ``schema``.

        Args:
            prompt: The reasoner-built prompt — system + tool palette +
                world-state digest + user goal.
            schema: The Pydantic class the provider must validate against.
                Almost always :class:`~openral_reasoner.Plan`; passed as a
                parameter so future variants (e.g., a streaming Plan)
                can reuse the Protocol.

        Returns:
            A validated :class:`Plan` instance. Implementations parse
            and raise on schema failure rather than returning a partial.
        """
        ...


@runtime_checkable
class Reasoner(Protocol):
    """Structural Protocol every S2 reasoner satisfies.

    The reasoner is a planning-rate (5-10 Hz, sometimes lower) component;
    it is **not** on the S1 control hot path. CLAUDE.md §6.2 puts it on
    the slow loop, with the BT executor running independently and
    consuming the XML produced from :meth:`plan`.

    Attributes:
        plan_rate_hz: The expected invocation cadence. Default 5 Hz.
        client: The :class:`LLMClient` backing this reasoner. The
            stub :class:`~openral_reasoner.NullReasoner` carries
            ``None`` here.
    """

    plan_rate_hz: float
    client: LLMClient | None

    def plan(self, world_state: WorldState, goal: str) -> Plan:
        """Produce a :class:`Plan` for ``goal`` given ``world_state``.

        Args:
            world_state: The current snapshot from the
                :class:`~openral_world_state.WorldStateAggregator`.
            goal: The natural-language goal — typically supplied by a
                higher-level orchestrator or the human operator.

        Returns:
            A validated :class:`Plan`. The caller is responsible for
            converting to BT XML and feeding it to the executor.

        Raises:
            ROSReasonerInvalidPlan: When the LLM's emitted Plan
                references skills that are not installed or capable.
            ROSPlanningError: For any other planning-layer failure
                (timeout, provider error, etc.). Per CLAUDE.md §10
                every planning failure is typed.
        """
        ...
