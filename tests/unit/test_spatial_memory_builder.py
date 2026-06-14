"""Tests for the ADR-0038 Phase 2 SpatialMemory builder + query engine.

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
    Pose6D,
    RecallObjectQuery,
    ResolvePlaceQuery,
    SpatialNodeKind,
)
from openral_core.exceptions import ROSObjectNotInMemory
from openral_world_state import SpatialMemory, compute_approach_viewpoint

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


def _obj(
    label: str, xyz: tuple[float, float, float], *, track_id: int | None = None
) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=0.9,
        pose=Pose6D(xyz=xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"),
        track_id=track_id,
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


# ── track_id durability across world-state-node sessions (ADR-0038) ───────────


def test_recycled_track_id_with_different_label_does_not_merge() -> None:
    """A reused track_id (world-state restart) must not merge a cup into a mug.

    ADR-0035 ObjectMemory track_ids are per-session monotonic and reset on
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
    are added) and recall_object recalls what was seen (ADR-0038)."""
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
