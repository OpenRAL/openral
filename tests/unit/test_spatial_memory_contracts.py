"""Contract tests for the ADR-0038 persistent spatial-memory schemas.

Covers the scene-graph typed surface: ``SpatialNode`` / ``SpatialEdge`` /
``SceneGraph`` with their integrity validators, the ``RecallObject*`` and
``ResolvePlace*`` query/result contracts, the ``ApproachViewpoint`` helper
type, and the ``ROSObjectNotInMemory`` exception. Validates against the real
``tests/unit/fixtures/home_scene_graph.json`` fixture (the "bring me a cup of wine"
home), per CLAUDE.md §1.11 — real schemas, real fixture, no mocks.
"""

from __future__ import annotations

from pathlib import Path

from openral_core import (
    ApproachViewpoint,
    Pose6D,
    RecallObjectMatch,
    RecallObjectQuery,
    RecallObjectResult,
    ResolvePlaceQuery,
    ResolvePlaceResult,
    ROSObjectNotInMemory,
    ROSPerceptionStale,
    SceneGraph,
    SpatialEdge,
    SpatialNode,
    SpatialNodeKind,
    SpatialRelationKind,
)
from pydantic import ValidationError

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


def _map_pose() -> Pose6D:
    return Pose6D(xyz=(1.0, 2.0, 0.9), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")


# ── Fixture loads and models the wine scenario ────────────────────────────────


def test_home_fixture_loads_as_scene_graph() -> None:
    """The real home fixture validates as a ``SceneGraph`` and round-trips."""
    graph = SceneGraph.model_validate_json(_FIXTURE.read_text())
    assert graph.schema_version == "0.1"
    assert SceneGraph.model_validate_json(graph.model_dump_json()) == graph


def test_fixture_containment_chain_wine_in_fridge_in_kitchen() -> None:
    """The fixture encodes wine ⊂ fridge ⊂ kitchen — the core wine-task relation."""
    graph = SceneGraph.model_validate_json(_FIXTURE.read_text())
    contains = {(e.src, e.dst) for e in graph.edges if e.kind is SpatialRelationKind.CONTAINS}
    assert ("fridge", "wine_bottle") in contains
    assert ("kitchen", "fridge") in contains
    assert ("cabinet", "wine_glass") in contains

    by_id = {n.node_id: n for n in graph.nodes}
    # The fridge and cabinet occlude their contents until opened.
    assert by_id["fridge"].is_container and by_id["fridge"].occludes_contents
    assert by_id["cabinet"].is_container and by_id["cabinet"].occludes_contents
    # The wine bottle itself is a plain (non-container) object.
    assert not by_id["wine_bottle"].is_container


def test_fixture_requester_is_an_agent_with_a_return_place() -> None:
    """The requester is an ``agent`` node anchored to a living-room place."""
    graph = SceneGraph.model_validate_json(_FIXTURE.read_text())
    by_id = {n.node_id: n for n in graph.nodes}
    assert by_id["requester"].kind is SpatialNodeKind.AGENT
    at_place = {(e.src, e.dst) for e in graph.edges if e.kind is SpatialRelationKind.AT_PLACE}
    assert ("requester", "living_room_sofa") in at_place


def test_fixture_traversable_graph_connects_living_room_to_fridge() -> None:
    """A ``traversable_to`` path exists from the sofa to the fridge."""
    graph = SceneGraph.model_validate_json(_FIXTURE.read_text())
    adj: dict[str, set[str]] = {}
    for e in graph.edges:
        if e.kind is SpatialRelationKind.TRAVERSABLE_TO:
            adj.setdefault(e.src, set()).add(e.dst)
    # BFS sofa → front_of_fridge
    seen, frontier = {"living_room_sofa"}, ["living_room_sofa"]
    while frontier:
        cur = frontier.pop()
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert "front_of_fridge" in seen
    assert "cabinet_shelf" in seen


# ── SceneGraph integrity validators ───────────────────────────────────────────


def _node(node_id: str, kind: SpatialNodeKind = SpatialNodeKind.OBJECT) -> SpatialNode:
    return SpatialNode(
        node_id=node_id, kind=kind, pose=_map_pose(), first_seen_ns=1, last_seen_ns=2
    )


def test_scene_graph_rejects_duplicate_node_ids() -> None:
    """Node ids must be unique within a graph."""
    try:
        SceneGraph(nodes=[_node("a"), _node("a")])
    except ValidationError:
        pass
    else:  # pragma: no cover - the constraint must fire
        raise AssertionError("duplicate node_id was not rejected")


def test_scene_graph_rejects_dangling_edge() -> None:
    """Every edge must reference existing nodes."""
    try:
        SceneGraph(
            nodes=[_node("a")],
            edges=[SpatialEdge(src="a", dst="ghost", kind=SpatialRelationKind.CONTAINS)],
        )
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("dangling edge was not rejected")


# ── SpatialNode invariants ────────────────────────────────────────────────────


def test_node_rejects_last_seen_before_first_seen() -> None:
    """``last_seen_ns`` must be >= ``first_seen_ns``."""
    try:
        SpatialNode(
            node_id="x",
            kind=SpatialNodeKind.OBJECT,
            pose=_map_pose(),
            first_seen_ns=9,
            last_seen_ns=2,
        )
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("inverted timestamps were not rejected")


def test_node_occludes_contents_requires_container() -> None:
    """``occludes_contents`` is only meaningful for a container."""
    try:
        SpatialNode(
            node_id="x",
            kind=SpatialNodeKind.OBJECT,
            pose=_map_pose(),
            is_container=False,
            occludes_contents=True,
            first_seen_ns=1,
            last_seen_ns=2,
        )
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("occludes_contents without is_container was not rejected")


# ── Query / result contracts ──────────────────────────────────────────────────


def test_recall_object_query_requires_a_term() -> None:
    """A recall must name what it is looking for (text or label)."""
    try:
        RecallObjectQuery()
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty RecallObjectQuery was not rejected")


def test_recall_object_result_round_trips_with_approach_viewpoint() -> None:
    """A match carrying a camera-facing approach viewpoint round-trips."""
    match = RecallObjectMatch(
        node_id="wine_bottle",
        label="bottle of wine",
        pose=_map_pose(),
        score=0.91,
        last_seen_ns=1500000000000,
        approach=ApproachViewpoint(
            pose=_map_pose(), standoff_m=0.35, camera_frame_id="gripper_camera"
        ),
        inside_container_id="fridge",
    )
    result = RecallObjectResult(matches=[match])
    assert RecallObjectResult.model_validate_json(result.model_dump_json()) == result
    assert result.matches[0].inside_container_id == "fridge"


def test_resolve_place_result_carries_a_path() -> None:
    """A place resolution returns a goal pose and a traversable path."""
    res = ResolvePlaceResult(
        node_id="front_of_fridge",
        goal=_map_pose(),
        path_node_ids=["living_room_sofa", "kitchen_table", "front_of_fridge"],
    )
    assert ResolvePlaceResult.model_validate_json(res.model_dump_json()) == res
    assert res.path_node_ids[-1] == "front_of_fridge"


def test_resolve_place_query_optional_kind_filter() -> None:
    """A place query may filter by node kind."""
    q = ResolvePlaceQuery(reference="kitchen", kind=SpatialNodeKind.ROOM)
    assert q.kind is SpatialNodeKind.ROOM


# ── Exception lineage ─────────────────────────────────────────────────────────


def test_object_not_in_memory_is_a_perception_stale_error() -> None:
    """``ROSObjectNotInMemory`` is caught by the perception-stale family."""
    assert issubclass(ROSObjectNotInMemory, ROSPerceptionStale)
    try:
        raise ROSObjectNotInMemory("no wine glass in memory")
    except ROSPerceptionStale:
        pass


# ── ADR-0039 read-only query tools decode through the ReasonerToolCall union ──


def test_reasoner_tool_call_decodes_query_variants() -> None:
    """An LLM tool-use payload routes to the right read-only query variant."""
    from openral_core import ReasonerToolCall, RecallObjectTool, ResolvePlaceTool
    from pydantic import TypeAdapter

    adapter: TypeAdapter[ReasonerToolCall] = TypeAdapter(ReasonerToolCall)
    found = adapter.validate_json('{"tool": "recall_object", "query": "the red mug", "limit": 3}')
    assert isinstance(found, RecallObjectTool)
    assert found.query == "the red mug"
    assert found.limit == 3

    place = adapter.validate_json('{"tool": "resolve_place", "reference": "the kitchen"}')
    assert isinstance(place, ResolvePlaceTool)
    assert place.reference == "the kitchen"


def test_query_tools_are_frozen_and_reject_extra_fields() -> None:
    """Read-only query tools are frozen + extra='forbid' (no smuggled fields)."""
    from openral_core import RecallObjectTool
    from pydantic import ValidationError

    try:
        RecallObjectTool.model_validate({"tool": "recall_object", "query": "mug", "rogue": 1})
    except ValidationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("extra field was not rejected")
