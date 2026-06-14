"""ADR-0018 F4 — deterministic LLM stand-in for the integration tier.

The :class:`FakeToolUseClient` satisfies the
:class:`openral_reasoner.tool_use.ToolUseClient` Protocol exactly the
same way the real Anthropic / OpenAI-compatible clients do; the only
difference is that the tool selection is driven by a static
:class:`Selector` callable (or a queued list of pre-baked
:data:`~openral_core.ReasonerToolCall` instances) rather than by a
remote LLM. Suitable for unit tests of
:class:`openral_reasoner.ReasonerCore` and for end-to-end tests that
need a reproducible tick sequence.

This module lives under ``tests/integration/fakes/`` per CLAUDE.md
§1.11 — production code never imports it.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from collections.abc import Callable

from openral_core import ReasonerToolCall
from openral_core.exceptions import ROSPlanningError, ROSReasonerInvalidPlan
from openral_reasoner.palette import ToolPalette
from openral_reasoner.tool_use import DEFAULT_SYSTEM_PROMPT

__all__ = ["FakeToolUseClient", "SelectorFn", "ToolCallTrace"]

SelectorFn = Callable[[str, ToolPalette], ReasonerToolCall]


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallTrace:
    """One captured ``select_tool`` invocation, for assertion.

    Attributes:
        context_text: The exact text passed to the client.
        palette: The palette the client was offered.
        system_prompt: The system prompt the client was given.
    """

    context_text: str
    palette: ToolPalette
    system_prompt: str


class FakeToolUseClient:
    """In-process deterministic :class:`ToolUseClient` for tests.

    Two configuration styles:

    - **Queued**: pass a list of pre-built :data:`ReasonerToolCall`
      instances via ``responses``. The client pops one per call,
      raises ``IndexError`` when the queue is empty (tests must
      provide as many responses as ticks).
    - **Selector**: pass a callable ``selector(context_text, palette)
      -> ReasonerToolCall``. Useful when the test wants the tool
      choice to depend on the context.

    The two are mutually exclusive — pass exactly one.

    Args:
        model_id: Arbitrary identifier exposed to test code; defaults
            to ``"fake"``.
        responses: Queue of canned responses (consumed FIFO).
        selector: Callable that builds a response from the input.
        raise_on_call: If not ``None``, every ``select_tool`` call
            raises this exception (used to test
            :class:`ROSPlanningError` propagation).
    """

    def __init__(
        self,
        *,
        model_id: str = "fake",
        responses: list[ReasonerToolCall] | None = None,
        selector: SelectorFn | None = None,
        raise_on_call: BaseException | None = None,
    ) -> None:
        """Validate exactly one configuration knob is set."""
        if responses is None and selector is None and raise_on_call is None:
            raise ValueError(
                "FakeToolUseClient: pass exactly one of `responses`, `selector`, "
                "or `raise_on_call`.",
            )
        if sum(int(x is not None) for x in (responses, selector, raise_on_call)) != 1:
            raise ValueError(
                "FakeToolUseClient: `responses`, `selector`, and `raise_on_call` are "
                "mutually exclusive — pass exactly one.",
            )
        self.model_id = model_id
        self._responses: deque[ReasonerToolCall] = deque(responses or [])
        self._selector = selector
        self._raise = raise_on_call
        self._traces: list[ToolCallTrace] = []
        self._lock = threading.Lock()

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Return the next canned tool call (or raise / run selector)."""
        with self._lock:
            self._traces.append(
                ToolCallTrace(
                    context_text=context_text,
                    palette=palette,
                    system_prompt=system_prompt,
                ),
            )
            if self._raise is not None:
                raise self._raise
            if self._selector is not None:
                call = self._selector(context_text, palette)
            else:
                if not self._responses:
                    raise ROSReasonerInvalidPlan(
                        "FakeToolUseClient: no more canned responses — "
                        "queue exhausted (test config error)",
                    )
                call = self._responses.popleft()
        # Validate the canned response against the palette the same way the
        # real client does, so tests catch palette / response mismatches.
        if call.tool == "execute_rskill" and call.rskill_id not in palette.execute_rskill_ids:
            raise ROSPlanningError(
                f"FakeToolUseClient: canned execute_rskill rskill_id={call.rskill_id!r} "
                f"not in palette {sorted(palette.execute_rskill_ids)!r}",
            )
        return call

    @property
    def traces(self) -> tuple[ToolCallTrace, ...]:
        """Snapshot of every ``select_tool`` invocation, oldest first."""
        return tuple(self._traces)

    @property
    def remaining_responses(self) -> int:
        """Number of unconsumed canned responses (queued mode only)."""
        return len(self._responses)
