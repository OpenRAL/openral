"""NullReasoner — no-LLM stub satisfying the Reasoner Protocol.

Useful for plumbing tests, runner integration tests, and any context
where the S2 layer must be present in the trace but a real LLM call is
neither available nor desirable. The :class:`NullReasoner` emits a
:class:`~openral_reasoner.Plan` that calls a single, hard-coded skill;
the BT XML conversion is deferred to the first concrete
:class:`Reasoner` PR.

This stub is **not** a fallback to be silently used in production —
CLAUDE.md §1.4 ("explicit beats implicit") forbids hidden fallbacks. It
is the equivalent of :class:`~openral_runner.safety.NullSafetyClient`
for the planning layer: a stub for tests, replaced by a real
implementation before hardware deployment.
"""

from __future__ import annotations

import structlog
from openral_core import WorldState

from openral_reasoner.plan import Plan, ToolCall
from openral_reasoner.protocol import LLMClient

__all__ = ["NullReasoner"]

log = structlog.get_logger(__name__)


class NullReasoner:
    """A no-LLM :class:`Reasoner` that emits a single hard-coded ToolCall.

    Useful for runner / executor plumbing tests where a real LLM call
    would require API credentials and provider availability. The emitted
    :class:`Plan` calls :attr:`default_skill_id` with empty params and
    confidence 1.0.

    Args:
        default_skill_id: The skill leaf every plan calls. Defaults to
            ``"noop"``; tests typically override with the skill under
            test.
        plan_rate_hz: Recorded for trace fidelity; not enforced by this
            stub (the caller decides when to invoke :meth:`plan`).

    Example:
        >>> from openral_core import JointState, WorldState
        >>> reasoner = NullReasoner(default_skill_id="pick_cube_so100")
        >>> ws = WorldState(
        ...     stamp_ns=0,
        ...     joint_state=JointState(name=["j1"], position=[0.0], stamp_ns=0),
        ... )
        >>> plan = reasoner.plan(ws, goal="pick the red cube")
        >>> plan.tool_calls[0].rskill_id
        'pick_cube_so100'
        >>> plan.confidence
        1.0
    """

    plan_rate_hz: float
    client: LLMClient | None
    default_skill_id: str

    def __init__(
        self,
        default_skill_id: str = "noop",
        *,
        plan_rate_hz: float = 5.0,
    ) -> None:
        """Initialise with a default skill id and an advertised plan rate."""
        self.default_skill_id = default_skill_id
        self.plan_rate_hz = plan_rate_hz
        self.client = None

    def plan(self, world_state: WorldState, goal: str) -> Plan:
        """Emit a one-leaf :class:`Plan` calling :attr:`default_skill_id`.

        ``world_state`` is accepted to satisfy the Protocol but
        not inspected — the stub is intentionally context-free.
        """
        del world_state
        log.debug(
            "reasoner.null_plan",
            goal=goal,
            rskill_id=self.default_skill_id,
        )
        return Plan(
            goal=goal,
            tool_calls=[ToolCall(rskill_id=self.default_skill_id, rationale="null-reasoner")],
            confidence=1.0,
        )
