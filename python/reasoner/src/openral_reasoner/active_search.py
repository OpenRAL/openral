"""Bounded active object search over the scene graph (ADR-0039 §3, Phase 4).

When a spatial-memory recall misses (``ROSObjectNotInMemory`` / empty
``RecallObjectResult``), a useful robot does not give up — it searches *likely*
places. This module turns an ADR-0038 :class:`~openral_core.SceneGraph` into a
**bounded, ranked frontier** of places to check, so the S2 Reasoner can drive a
search loop that terminates instead of running forever.

Two responsibilities, both pure-Python (``openral_core`` only):

- **Candidate frontier** — :func:`plan_active_search` lists the places/containers
  worth checking, ranked by a generic heuristic (occluding containers first —
  things hide inside them — then standable places). Semantic prioritization
  *among* candidates ("a glass is usually in a cabinet") is the LLM's job via
  its commonsense priors; this supplies the bounded set it chooses from.
- **The bound** — :class:`SearchBudget` caps the frontier (and, via
  :class:`SearchProgress`, the number of attempts). Exhausting the budget is the
  terminal *human-handoff* rung of the ADR-0018 replanning ladder, not an
  unbounded loop.
"""

from __future__ import annotations

from openral_core import Pose6D, SceneGraph, SpatialNodeKind, SpatialRelationKind
from pydantic import BaseModel, ConfigDict, Field


class SearchBudget(BaseModel):
    """Bounds an active object search (ADR-0039 §3).

    Attributes:
        max_candidates: Max places offered in the frontier per search.
        max_attempts: Max places actually visited before human-handoff.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_candidates: int = Field(default=5, ge=1, le=50)
    max_attempts: int = Field(default=5, ge=1, le=50)


class SearchCandidate(BaseModel):
    """One place worth checking for a missing object (ADR-0039 §3).

    Attributes:
        place_node_id: The place/container node to go to.
        goal: Navigation goal pose (map frame) — where to stand to look.
        open_container_id: Set when the candidate is an occluding container the
            planner must open before it can see inside; ``None`` for open places.
        reason: Short human/LLM-readable rationale.
        rank: Heuristic priority in [0, 1] (higher = check first).
    """

    model_config = ConfigDict(extra="forbid")

    place_node_id: str
    goal: Pose6D
    open_container_id: str | None = None
    reason: str
    rank: float = Field(ge=0.0, le=1.0)


def plan_active_search(
    graph: SceneGraph, *, target_text: str, budget: SearchBudget
) -> list[SearchCandidate]:
    """Build a bounded, ranked frontier of places to search for ``target_text``.

    Occluding containers rank first (objects hide inside them), then other
    containers, then standable ``place`` nodes. The list is truncated to
    ``budget.max_candidates``. Returns ``[]`` when the graph has nowhere to
    search — the caller then escalates to human-handoff.

    Args:
        graph: The current scene graph.
        target_text: What is being searched for (carried into each ``reason``).
        budget: The search bound.

    Returns:
        Ranked candidates, at most ``budget.max_candidates``.
    """
    nodes = {n.node_id: n for n in graph.nodes}
    # place a node is reached from: object/container -> at_place -> place.
    at_place = {e.src: e.dst for e in graph.edges if e.kind is SpatialRelationKind.AT_PLACE}

    candidates: list[SearchCandidate] = []
    for node in graph.nodes:
        if node.kind is SpatialNodeKind.OBJECT and node.is_container:
            # A container: go to its standoff place (if any) and open it to look.
            place_id = at_place.get(node.node_id)
            # SceneGraph guarantees edge targets exist; fall back to the node itself.
            goal_node = nodes.get(place_id, node) if place_id is not None else node
            rank = 0.9 if node.occludes_contents else 0.6
            candidates.append(
                SearchCandidate(
                    place_node_id=goal_node.node_id,
                    goal=goal_node.pose,
                    open_container_id=node.node_id,
                    reason=f"open {node.label or node.node_id!r} and look for {target_text!r}",
                    rank=rank,
                )
            )
        elif node.kind is SpatialNodeKind.PLACE:
            candidates.append(
                SearchCandidate(
                    place_node_id=node.node_id,
                    goal=node.pose,
                    reason=f"go to {node.label or node.node_id!r} and look for {target_text!r}",
                    rank=0.3,
                )
            )

    # Stable: rank desc, then node id for determinism.
    candidates.sort(key=lambda c: (-c.rank, c.place_node_id))
    return candidates[: budget.max_candidates]


class SearchProgress:
    """Tracks attempts against a :class:`SearchBudget` (the runaway bound).

    The reasoner increments this on each search step; once attempts reach
    ``budget.max_attempts`` the search is **exhausted** and the caller hands off
    to a human (ADR-0018 ladder terminal rung) instead of looping forever.
    """

    def __init__(self, budget: SearchBudget) -> None:
        """Start a fresh search against ``budget``."""
        self._budget = budget
        self._attempts = 0

    @property
    def attempts(self) -> int:
        """Number of search steps consumed so far."""
        return self._attempts

    @property
    def exhausted(self) -> bool:
        """True once the attempt budget is spent (→ human-handoff)."""
        return self._attempts >= self._budget.max_attempts

    def record_attempt(self) -> bool:
        """Consume one attempt; return ``True`` while budget remains, else ``False``."""
        self._attempts += 1
        return not self.exhausted

    def reset(self) -> None:
        """Reset the counter (e.g. on a fresh operator goal)."""
        self._attempts = 0


def format_search_frontier(candidates: list[SearchCandidate], target_text: str) -> str:
    """Render the bounded frontier as an LLM-readable text block for the cascade."""
    if not candidates:
        return (
            f"active_search: {target_text!r} is not in memory and there are no known "
            "places to search — hand off to a human."
        )
    lines = [
        f"active_search: {target_text!r} not in memory. "
        f"Candidate places to search (bounded to {len(candidates)}, highest priority first):"
    ]
    lines.extend(f"- {c.reason} [place {c.place_node_id}]" for c in candidates)
    return "\n".join(lines)
