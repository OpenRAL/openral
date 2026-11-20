"""Tests for the Phase 2 SpatialMemory builder + query engine.

Exercises instance association, find/resolve queries, the camera-facing
approach-viewpoint geometry, and persistence — including the full
"bring me a cup of wine" scenario loaded from the real
``tests/unit/fixtures/home_scene_graph.json`` fixture (CLAUDE.md §1.11, no mocks).
"""

from __future__ import annotations

import math
from pathlib import Path

from openral_core import (
    DetectedObject,
    ObjectBeliefState,
    Pose6D,
    RecallObjectQuery,
    ResolvePlaceQuery,
    SpatialNodeKind,
)
from openral_core.exceptions import ROSObjectNotInMemory
from openral_world_state import SpatialMemory, compute_approach_viewpoint

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


def _obj(
    label: str,
    xyz: tuple[float, float, float],
    *,
    track_id: int | None = None,
    existence: float | None = None,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=0.9,
        pose=Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        track_id=track_id,
        existence=existence,
    )


# ── Instance association ───────────────────────────────────────────────────────


def test_ingest_creates_one_node_per_object() -> None:
    mem = SpatialMemory()
    touched = mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=10)
    assert len(touched) == 1
    graph = mem.to_scene_graph()
    assert len(graph.nodes) == 1
    assert graph.nodes[0].observation_count == 1


def test_reobservation_by_track_id_updates_not_duplicates() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=10)
    mem.ingest_detected_objects([_obj("mug", (1.2, 0.0, 0.8), track_id=1)], now_ns=20)
    graph = mem.to_scene_graph()
    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.observation_count == 2
    assert node.last_seen_ns == 20
    assert node.pose.xyz[0] == 1.2  # pose updated to the latest observation


def test_association_by_label_and_proximity_without_track_id() -> None:
    mem = SpatialMemory(assoc_distance_m=0.3)
    mem.ingest_detected_objects([_obj("plate", (0.0, 0.0, 0.0))], now_ns=10)
    # Same label within radius → merges.
    mem.ingest_detected_objects([_obj("plate", (0.1, 0.0, 0.0))], now_ns=20)
    # Same label beyond radius → distinct node.
    mem.ingest_detected_objects([_obj("plate", (2.0, 0.0, 0.0))], now_ns=30)
    objs = [n for n in mem.to_scene_graph().nodes if n.kind is SpatialNodeKind.OBJECT]
    assert len(objs) == 2


# ── recall_object ────────────────────────────────────────────────────────────────


def test_recall_object_returns_match_with_approach_viewpoint() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=3)], now_ns=10)
    result = mem.recall_object(RecallObjectQuery(label="mug"), now_ns=20)
    assert len(result.matches) == 1
    m = result.matches[0]
    assert m.label == "mug"
    assert m.approach is not None
    assert m.inside_container_id is None
    assert 0.0 <= m.score <= 1.0


def test_recall_object_recency_filter_drops_stale() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=3)], now_ns=10)
    # last_seen=10, now=1000, max_age=100 → 990 > 100 → filtered out.
    result = mem.recall_object(RecallObjectQuery(label="mug", max_age_ns=100), now_ns=1000)
    assert result.matches == []


def test_recall_object_unknown_returns_empty_not_exception() -> None:
    mem = SpatialMemory()
    result = mem.recall_object(RecallObjectQuery(label="teapot"), now_ns=10)
    assert result.matches == []


# ── resolve_place ────────────────────────────────────────────────────────────────


def test_resolve_place_unknown_raises() -> None:
    mem = SpatialMemory()
    try:
        mem.resolve_place(ResolvePlaceQuery(reference="garage"))
    except ROSObjectNotInMemory:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown reference did not raise ROSObjectNotInMemory")


# ── Approach-viewpoint geometry ─────────────────────────────────────────────────


def test_approach_viewpoint_stands_off_and_faces_object() -> None:
    target = Pose6D(xyz=(1.0, 0.0, 0.8), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")
    approach_from = Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")
    vp = compute_approach_viewpoint(target, standoff_m=0.6, approach_from=approach_from)
    vx, vy, vz = vp.pose.xyz
    # 0.6 m from the target, on the approach_from (−X) side → x = 1.0 − 0.6 = 0.4.
    assert math.isclose(vx, 0.4, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(vz, 0.8, abs_tol=1e-6)
    # Yaw faces +X (toward the target): quaternion ≈ identity.
    assert math.isclose(vp.pose.quat_xyzw[2], 0.0, abs_tol=1e-6)
    assert math.isclose(vp.pose.quat_xyzw[3], 1.0, abs_tol=1e-6)
    assert vp.standoff_m == 0.6


# ── Persistence round-trip ───────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path: Path) -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=10)
    out = tmp_path / "graph.json"
    mem.save(out)
    reloaded = SpatialMemory.load(out)
    assert reloaded.to_scene_graph() == mem.to_scene_graph()


# ── Full wine scenario over the real home fixture ────────────────────────────────


def test_wine_fixture_find_wine_reports_occluding_fridge() -> None:
    mem = SpatialMemory.load(_FIXTURE)
    result = mem.recall_object(RecallObjectQuery(label="bottle of wine"), now_ns=2_000_000_000_000)
    assert result.matches, "wine bottle should be recalled"
    top = result.matches[0]
    assert top.node_id == "wine_bottle"
    # The planner must open the fridge (occluding container) before grasping.
    assert top.inside_container_id == "fridge"
    assert top.approach is not None


def test_wine_fixture_resolve_fridge_gives_path_from_sofa() -> None:
    mem = SpatialMemory.load(_FIXTURE)
    res = mem.resolve_place(ResolvePlaceQuery(reference="fridge"), from_node_id="living_room_sofa")
    # The fridge resolves to its standoff place, reached via the kitchen table.
    assert res.node_id == "front_of_fridge"
    assert res.path_node_ids[0] == "living_room_sofa"
    assert res.path_node_ids[-1] == "front_of_fridge"
    assert "kitchen_table" in res.path_node_ids


def test_wine_fixture_resolve_requester_return_goal() -> None:
    mem = SpatialMemory.load(_FIXTURE)
    res = mem.resolve_place(ResolvePlaceQuery(reference="the person who asked"))
    # "bring it back to me" → the requester agent's standoff place.
    assert res.node_id == "living_room_sofa"


# ── track_id durability across world-state-node sessions ─────────────────────


def test_recycled_track_id_with_different_label_does_not_merge() -> None:
    """A reused track_id (world-state restart) must not merge a cup into a mug.

    ObjectMemory track_ids are per-session monotonic and reset on
    restart, so a durable memory keyed only on track_id would corrupt a node
    when the id is recycled for a different object. The label-guarded fast path
    + label/proximity fallback keeps them distinct.
    """
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=3)], now_ns=1)
    # World-state node restarts; track_id 3 now belongs to a cup far away.
    mem.ingest_detected_objects([_obj("cup", (5.0, 5.0, 0.8), track_id=3)], now_ns=2)

    graph = mem.to_scene_graph()
    objects = {n.label: n for n in graph.nodes if n.kind is SpatialNodeKind.OBJECT}
    assert set(objects) == {"mug", "cup"}
    # The mug node kept its own pose — the cup detection did not overwrite it.
    assert objects["mug"].pose.xyz == (1.0, 0.0, 0.8)
    assert objects["cup"].pose.xyz == (5.0, 5.0, 0.8)


def test_returning_object_with_fresh_track_id_reassociates_by_proximity() -> None:
    """An object seen again under a new track_id merges into its existing node."""
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=1)
    # Same mug, slightly moved, but a fresh session id (1 -> 7).
    touched = mem.ingest_detected_objects([_obj("mug", (1.02, 0.01, 0.8), track_id=7)], now_ns=2)

    objects = [n for n in mem.to_scene_graph().nodes if n.kind is SpatialNodeKind.OBJECT]
    assert len(objects) == 1  # merged, not duplicated
    assert objects[0].observation_count == 2
    assert touched == [objects[0].node_id]


def test_accumulates_objects_across_snapshots() -> None:
    """The reasoner's live-ingest loop: folding successive WorldState snapshots
    accumulates a durable map (stable tracks bump observation_count; new objects
    are added) and recall_object recalls what was seen."""
    mem = SpatialMemory()
    # Snapshot 1 — the gripper camera sees the wine bottle.
    mem.ingest_detected_objects([_obj("wine_bottle", (3.0, 1.0, 0.9), track_id=1)], now_ns=10)
    # Snapshot 2 — bottle still in view (stable track) + a glass comes into view.
    mem.ingest_detected_objects(
        [
            _obj("wine_bottle", (3.0, 1.0, 0.9), track_id=1),
            _obj("wine_glass", (2.0, 0.5, 0.8), track_id=2),
        ],
        now_ns=20,
    )
    # Snapshot 3 — robot looked away; nothing detected. Durable memory persists
    # (unlike the short-horizon ObjectMemory, this layer does not evict).
    mem.ingest_detected_objects([], now_ns=30)

    result = mem.recall_object(RecallObjectQuery(label="wine_bottle"), now_ns=30)
    assert result.matches and result.matches[0].label == "wine_bottle"
    objects = {n.label: n for n in mem.to_scene_graph().nodes if n.kind is SpatialNodeKind.OBJECT}
    assert set(objects) == {"wine_bottle", "wine_glass"}
    assert objects["wine_bottle"].observation_count == 2


# ── Following the world-state object memory's belief (issue #636) ──────────────


def _states(mem: SpatialMemory) -> dict[str, ObjectBeliefState]:
    return {n.label: n.state for n in mem.to_scene_graph().nodes}


def test_node_takes_the_belief_state_of_its_detection() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects(
        [
            _obj("mug", (1.0, 0.0, 0.8), track_id=1, existence=0.9),
            _obj("bowl", (2.0, 0.0, 0.8), track_id=2, existence=0.5),  # restored, unconfirmed
            _obj("fork", (3.0, 0.0, 0.8), track_id=3),  # a producer that keeps no belief
        ],
        now_ns=1,
    )
    assert _states(mem) == {
        "mug": ObjectBeliefState.PRESENT,
        "bowl": ObjectBeliefState.UNCERTAIN,
        "fork": ObjectBeliefState.PRESENT,
    }
    mem.ingest_detected_objects(
        [_obj("bowl", (2.0, 0.0, 0.8), track_id=2, existence=0.8)], now_ns=2
    )
    assert _states(mem)["bowl"] is ObjectBeliefState.PRESENT


def test_object_the_world_state_evicted_is_marked_absent_and_kept() -> None:
    """A node the remembered set drops was observed gone: absent, with its history."""
    mem = SpatialMemory()
    both = [_obj("mug", (1.0, 0.0, 0.8), track_id=1), _obj("bowl", (2.0, 0.0, 0.8), track_id=2)]
    mem.ingest_detected_objects(both, now_ns=10, follow_eviction=True)
    mem.ingest_detected_objects(both[:1], now_ns=20, follow_eviction=True)

    nodes = {n.label: n for n in mem.to_scene_graph().nodes}
    assert nodes["mug"].state is ObjectBeliefState.PRESENT
    assert nodes["bowl"].state is ObjectBeliefState.ABSENT
    assert nodes["bowl"].pose.xyz == (2.0, 0.0, 0.8) and nodes["bowl"].last_seen_ns == 10

    # "Where was the bowl" still answers, flagged; a present match outranks an absent one.
    mem.ingest_detected_objects(
        [both[0], _obj("bowl", (5.0, 0.0, 0.8), track_id=3)], now_ns=30, follow_eviction=True
    )
    matches = mem.recall_object(RecallObjectQuery(label="bowl"), now_ns=30).matches
    assert [(m.pose.xyz[0], m.state) for m in matches] == [
        (5.0, ObjectBeliefState.PRESENT),
        (2.0, ObjectBeliefState.ABSENT),
    ]

    # Seen again where it was: the same node is present once more.
    mem.ingest_detected_objects(
        [_obj("bowl", (2.0, 0.0, 0.8), track_id=9)], now_ns=40, follow_eviction=True
    )
    again = [n for n in mem.to_scene_graph().nodes if n.pose.xyz == (2.0, 0.0, 0.8)]
    assert [n.state for n in again] == [ObjectBeliefState.PRESENT]


def test_resolve_place_never_sends_the_robot_to_where_an_object_was() -> None:
    mem = SpatialMemory()
    old, new = _obj("mug", (1.0, 0.0, 0.8), track_id=1), _obj("mug", (4.0, 0.0, 0.8), track_id=2)
    mem.ingest_detected_objects([old], now_ns=1, follow_eviction=True)
    mem.ingest_detected_objects([new], now_ns=2, follow_eviction=True)  # track 1 evicted
    assert mem.resolve_place(ResolvePlaceQuery(reference="mug")).goal.xyz == (4.0, 0.0, 0.8)
    mem.ingest_detected_objects([], now_ns=3, follow_eviction=True)
    try:
        mem.resolve_place(ResolvePlaceQuery(reference="mug"))
    except ROSObjectNotInMemory:
        pass
    else:  # pragma: no cover
        raise AssertionError("an absent-only reference resolved to a goal")


def test_an_empty_remembered_set_marks_everything_it_carried_absent() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects(
        [_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=1, follow_eviction=True
    )
    mem.ingest_detected_objects([], now_ns=2, follow_eviction=True)
    assert _states(mem) == {"mug": ObjectBeliefState.ABSENT}


def test_without_follow_eviction_a_missing_object_stays_as_it_was() -> None:
    mem = SpatialMemory()
    mem.ingest_detected_objects([_obj("mug", (1.0, 0.0, 0.8), track_id=1)], now_ns=1)
    mem.ingest_detected_objects([], now_ns=2)
    assert _states(mem) == {"mug": ObjectBeliefState.PRESENT}


def test_preloaded_objects_are_never_marked_absent_by_a_snapshot_that_never_had_them() -> None:
    mem = SpatialMemory.load(_FIXTURE)
    before = _states(mem)
    mem.ingest_detected_objects([], now_ns=1, follow_eviction=True)
    assert _states(mem) == before


def test_restored_scene_file_objects_start_uncertain_until_seen() -> None:
    """Issue #636: a restart (or a scene file) is a memory, not an observation."""
    mem = SpatialMemory.load(_FIXTURE, restored_uncertain=True)
    graph = mem.to_scene_graph()
    objects = [n for n in graph.nodes if n.kind is SpatialNodeKind.OBJECT]
    assert objects and all(n.state is ObjectBeliefState.UNCERTAIN for n in objects)
    assert all(
        n.state is ObjectBeliefState.PRESENT
        for n in graph.nodes
        if n.kind is not SpatialNodeKind.OBJECT
    )
    seen = objects[0]
    mem.ingest_detected_objects(
        [_obj(seen.label, seen.pose.xyz, existence=0.9)], now_ns=seen.last_seen_ns + 1
    )
    after = {n.node_id: n.state for n in mem.to_scene_graph().nodes}
    assert after[seen.node_id] is ObjectBeliefState.PRESENT
