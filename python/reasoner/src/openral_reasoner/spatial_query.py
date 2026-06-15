"""Spatial-memory query bridge for the S2 reasoner (ADR-0039 Phase 2).

The reasoner emits the read-only :class:`~openral_core.RecallObjectTool` /
:class:`~openral_core.ResolvePlaceTool` variants; this module turns such a tool
call into an ADR-0038 query, runs it against an injected spatial-memory backend,
and renders the result as an LLM-readable text block to feed back into the
reasoning loop (republished as a ``PromptStamped`` by ``reasoner_node`` — the
"result-return via prompt cascade" path of ADR-0018 §F10).

The backend is duck-typed via :class:`SpatialMemoryQuerier` so this Layer-4
module does not import the Layer-2 ``openral_world_state`` package — the concrete
``SpatialMemory`` structurally satisfies the Protocol. These tools are
**read-only** and hold no authority over actuation (CLAUDE.md §3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

from openral_core import (
    ApproachViewpoint,
    RecallObjectQuery,
    RecallObjectResult,
    RecallObjectTool,
    ResolvePlaceQuery,
    ResolvePlaceResult,
    ResolvePlaceTool,
    SceneGraph,
)
from openral_core.exceptions import ROSObjectNotInMemory

SpatialQueryTool = RecallObjectTool | ResolvePlaceTool
"""The read-only ``ReasonerToolCall`` variants this bridge dispatches."""

ApproachRefiner = Callable[
    [ApproachViewpoint, tuple[float, float, float]], ApproachViewpoint | None
]
"""ADR-0044 Phase 4 — optional occupancy-grid approach refinement.

Called per match as ``refiner(viewpoint, target_xyz)``; returns the
grid-validated (possibly snapped) viewpoint, or ``None`` when no reachable
viewpoint exists within the search radius. Duck-typed (like
:class:`SpatialMemoryQuerier`) so this Layer-4 module never imports the
Layer-2 ``openral_world_state`` package — the reasoner node wires
``refine_approach_pose`` over its latched ``/map`` subscription."""


class SpatialMemoryQuerier(Protocol):
    """Read-only spatial-memory query surface (satisfied by ``SpatialMemory``)."""

    def recall_object(self, query: RecallObjectQuery, *, now_ns: int) -> RecallObjectResult:
        """Recall objects matching ``query`` (empty result = not found)."""
        ...

    def resolve_place(
        self, query: ResolvePlaceQuery, *, from_node_id: str | None = None
    ) -> ResolvePlaceResult:
        """Resolve a place/room/agent reference (raises ``ROSObjectNotInMemory`` on miss)."""
        ...

    def to_scene_graph(self) -> SceneGraph:
        """Immutable snapshot of the current graph (for telemetry / dashboard)."""
        ...


def recall_object_tool_to_query(call: RecallObjectTool) -> RecallObjectQuery:
    """Map a :class:`~openral_core.RecallObjectTool` to a :class:`RecallObjectQuery`."""
    return RecallObjectQuery(text=call.query, limit=call.limit)


def resolve_place_tool_to_query(call: ResolvePlaceTool) -> ResolvePlaceQuery:
    """Map a :class:`~openral_core.ResolvePlaceTool` to a :class:`ResolvePlaceQuery`."""
    return ResolvePlaceQuery(reference=call.reference)


def _fmt_xyz(xyz: tuple[float, float, float]) -> str:
    return f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"


def format_recall_object_result(
    query_text: str,
    result: RecallObjectResult,
    *,
    blocked_node_ids: frozenset[str] = frozenset(),
) -> str:
    """Render a :class:`RecallObjectResult` as an LLM-readable text block.

    ``blocked_node_ids`` (ADR-0044 Phase 4) marks matches whose approach
    viewpoint failed occupancy-grid refinement — rendered as an explicit
    "approach blocked" note rather than a pose the robot can't reach (never
    a fabricated viewpoint, CLAUDE.md §1.2).
    """
    if not result.matches:
        return (
            f"spatial_memory: {query_text!r} is not in memory. Check the "
            "scene_objects list in WORLD_STATE first — the live detector may "
            "have it under a different label (e.g. a 'baguette' goal vs a "
            "detected 'bread'); if a listed object is your target, act on it "
            "directly. Otherwise consider an active search of likely locations."
        )
    lines = [f"spatial_memory: {len(result.matches)} match(es) for {query_text!r}:"]
    for m in result.matches:
        parts = [
            f"- {m.label!r} (node {m.node_id}) at map {_fmt_xyz(m.pose.xyz)}",
            f"score={m.score:.2f}",
        ]
        if m.node_id in blocked_node_ids:
            parts.append(
                "approach BLOCKED on the occupancy grid (no free viewpoint with "
                "line-of-sight within the search radius) — consider another match "
                "or a different vantage"
            )
        elif m.approach is not None:
            parts.append(
                f"approach from {_fmt_xyz(m.approach.pose.xyz)} "
                f"(standoff {m.approach.standoff_m:.2f} m, camera {m.approach.camera_frame_id})"
            )
        if m.inside_container_id is not None:
            parts.append(f"INSIDE container {m.inside_container_id!r} — open it before grasping")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def format_resolve_place_result(reference: str, result: ResolvePlaceResult) -> str:
    """Render a :class:`ResolvePlaceResult` as an LLM-readable text block."""
    text = (
        f"spatial_memory: {reference!r} resolves to node {result.node_id} "
        f"at map {_fmt_xyz(result.goal.xyz)}"
    )
    if result.path_node_ids:
        text += ". Path: " + " -> ".join(result.path_node_ids)
    return text


class SpatialQueryOutcome(NamedTuple):
    """Result of a spatial-memory query: the rendered text + whether it matched.

    ``found`` is ``True`` when ``recall_object`` returned ≥1 match (in memory,
    even if every approach is grid-BLOCKED — the object is still known) or
    ``resolve_place`` resolved the reference; ``False`` on a "not in memory"
    miss. The reasoner node uses ``found`` to decide whether to escalate a
    recall miss to a live ``locate_in_view`` (ADR-0043/0056) before handoff.
    """

    text: str
    found: bool


def run_spatial_query_detailed(
    call: SpatialQueryTool,
    querier: SpatialMemoryQuerier,
    *,
    now_ns: int,
    from_node_id: str | None = None,
    refine_approach: ApproachRefiner | None = None,
) -> SpatialQueryOutcome:
    """Execute a read-only spatial-memory tool call; render text + report match.

    Same behaviour as :func:`run_spatial_query` but also reports whether the
    query matched (``SpatialQueryOutcome.found``), so the caller can escalate a
    miss to a live perception check without re-parsing the rendered text.

    Args:
        call: A :class:`~openral_core.RecallObjectTool` or
            :class:`~openral_core.ResolvePlaceTool`.
        querier: The spatial-memory backend (e.g. an ADR-0038 ``SpatialMemory``).
        now_ns: Current time in nanoseconds (recency filtering).
        from_node_id: Optional origin node for ``resolve_place`` path planning.
        refine_approach: ADR-0044 Phase 4 — optional occupancy-grid refiner
            applied to every ``recall_object`` match's approach viewpoint before
            rendering, so the LLM only ever sees grid-valid approach poses.
            ``None`` from the refiner marks the match's approach BLOCKED in the
            rendered text. Absent (no grid) → the geometric viewpoint passes
            through unchanged.

    Returns:
        A :class:`SpatialQueryOutcome`. A miss (no match / unresolved reference)
        is reported as text and ``found=False`` — never a fabricated pose
        (CLAUDE.md §1.2): ``resolve_place`` raises
        :class:`~openral_core.exceptions.ROSObjectNotInMemory` internally, which
        is caught and rendered as a "not in memory" message.
    """
    if isinstance(call, RecallObjectTool):
        result = querier.recall_object(recall_object_tool_to_query(call), now_ns=now_ns)
        found = bool(result.matches)
        blocked: set[str] = set()
        if refine_approach is not None and result.matches:
            refined_matches = []
            for m in result.matches:
                if m.approach is None:
                    refined_matches.append(m)
                    continue
                refined = refine_approach(m.approach, m.pose.xyz)
                if refined is None:
                    blocked.add(m.node_id)
                    refined_matches.append(m.model_copy(update={"approach": None}))
                else:
                    refined_matches.append(m.model_copy(update={"approach": refined}))
            result = result.model_copy(update={"matches": refined_matches})
        text = format_recall_object_result(
            call.query, result, blocked_node_ids=frozenset(blocked)
        )
        return SpatialQueryOutcome(text=text, found=found)
    query = resolve_place_tool_to_query(call)
    try:
        resolved = querier.resolve_place(query, from_node_id=from_node_id)
    except ROSObjectNotInMemory:
        return SpatialQueryOutcome(
            text=(
                f"spatial_memory: {call.reference!r} is not in memory. Check the "
                "scene_objects list in WORLD_STATE first — the live detector may "
                "have it under a different label; if a listed object is your "
                "target, act on it directly. Otherwise consider an active search."
            ),
            found=False,
        )
    return SpatialQueryOutcome(
        text=format_resolve_place_result(call.reference, resolved), found=True
    )


def run_spatial_query(
    call: SpatialQueryTool,
    querier: SpatialMemoryQuerier,
    *,
    now_ns: int,
    from_node_id: str | None = None,
    refine_approach: ApproachRefiner | None = None,
) -> str:
    """Execute a read-only spatial-memory tool call and render the result as text.

    Thin wrapper over :func:`run_spatial_query_detailed` returning only the
    rendered text (the historical signature). See that function for argument
    and return-value semantics.
    """
    return run_spatial_query_detailed(
        call,
        querier,
        now_ns=now_ns,
        from_node_id=from_node_id,
        refine_approach=refine_approach,
    ).text
