"""Unit tests for ObjectMemory (freeze-on-match, evidence-weighed existence belief)."""

from __future__ import annotations

import pytest
from openral_core.schemas import (
    DetectedObject,
    ObjectBeliefState,
    Pose6D,
    SceneGraph,
    belief_state,
)
from openral_world_state.object_lift import MissEvidence
from openral_world_state.object_memory import ObjectMemory


def _obj(
    label: str,
    xyz: tuple[float, float, float],
    box: tuple[float, float, float, float, float, float],
    conf: float = 0.9,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        confidence=conf,
        pose=Pose6D(xyz=xyz, quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=box,
    )


_BOX_A = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
_BOX_A_NUDGE = (0.05, 0.0, 0.0, 1.05, 1.0, 1.0)  # high IoU vs A
_BOX_B = (5.0, 5.0, 5.0, 6.0, 6.0, 6.0)  # disjoint from A


def _free(o):
    return MissEvidence.FREE


def _occupied(o):
    return MissEvidence.OCCUPIED


def _none(o):
    return MissEvidence.NONE


def test_new_detection_creates_track_with_id():
    mem = ObjectMemory(iou_threshold=0.3)
    out = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_free)
    assert len(out) == 1
    assert out[0].track_id == 0
    assert out[0].label == "cup"


def test_match_freezes_pose_and_keeps_track_id():
    mem = ObjectMemory(iou_threshold=0.3)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_free)
    out = mem.tick([_obj("cup", (0.9, 0.5, 0.5), _BOX_A_NUDGE)], stamp_ns=2, miss_evidence=_free)
    assert len(out) == 1
    assert out[0].track_id == 0
    assert out[0].pose.xyz == (0.5, 0.5, 0.5)
    assert out[0].bbox_3d == _BOX_A


def test_different_label_same_box_is_a_new_track():
    mem = ObjectMemory(iou_threshold=0.3)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_free)
    # Same box, both labels detected this run: the bowl must NOT be absorbed
    # into the cup track (association is label-specific) -> two tracks; both
    # are matched/created this run so neither is evicted.
    out = mem.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A), _obj("bowl", (0.5, 0.5, 0.5), _BOX_A)],
        stamp_ns=2,
        miss_evidence=_free,
    )
    assert {o.label for o in out} == {"cup", "bowl"}
    assert {o.track_id for o in out} == {0, 1}


def test_one_clear_miss_does_not_evict_but_a_few_do():
    """Issue #636: a once-seen object survives a single miss; it goes after N clear ones."""
    mem = ObjectMemory()  # hit +0.85, clear miss -0.85, absent at -2.0
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_free)
    survived = [len(mem.tick([], stamp_ns=t, miss_evidence=_free)) for t in range(2, 6)]
    assert survived == [1, 1, 1, 0]


def test_unobservable_object_is_kept_indefinitely():
    """Out of view, out of range or occluded is no evidence: nothing changes."""
    mem = ObjectMemory()
    first = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    for t in range(2, 200):
        out = mem.tick([], stamp_ns=t, miss_evidence=_none)
    assert [o.track_id for o in out] == [0]
    assert out[0].existence == first[0].existence


def test_occupied_miss_is_weak_and_can_be_switched_off():
    """An undetected object whose box is still occupied (hidden behind the arm).

    Weak by default — seconds of occlusion cost little — yet not zero, or a false
    detection lifted onto a wall would be remembered forever.
    """
    mem = ObjectMemory()  # hit +0.85, occupied miss -0.1, absent at -2.0
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    kept = [len(mem.tick([], stamp_ns=t, miss_evidence=_occupied)) for t in range(2, 32)]
    assert kept.index(0) == 28  # 0.85 - 29 * 0.1 <= -2.0: 14 s at 2 Hz, not one tick

    off = ObjectMemory(miss_occupied_log_odds=0.0)
    off.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    for t in range(2, 200):
        out = off.tick([], stamp_ns=t, miss_evidence=_occupied)
    assert len(out) == 1


def test_a_huge_negative_belief_does_not_overflow():
    """Eviction pushed out of reach must not crash the memory tick (exp overflow)."""
    mem = ObjectMemory(belief_absent=-1e6, miss_free_log_odds=-900.0)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    out = mem.tick([], stamp_ns=2, miss_evidence=_free)
    assert out[0].existence == 0.0


def test_belief_is_clamped_so_a_long_seen_object_is_still_forgotten():
    mem = ObjectMemory(belief_max=3.5)
    for t in range(1, 100):
        out = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=t, miss_evidence=_free)
    assert out[0].existence == pytest.approx(1.0 / (1.0 + 2.718281828459045**-3.5))
    survived = [len(mem.tick([], stamp_ns=100 + t, miss_evidence=_free)) for t in range(7)]
    assert survived == [1, 1, 1, 1, 1, 1, 0]  # 3.5 - 7 * 0.85 <= -2.0


def test_redetection_restores_belief():
    mem = ObjectMemory()
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_free)
    low = mem.tick([], stamp_ns=2, miss_evidence=_free)
    mem.tick([], stamp_ns=3, miss_evidence=_free)
    assert belief_state(low[0].existence) is ObjectBeliefState.UNCERTAIN
    back = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=4, miss_evidence=_free)
    back = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=5, miss_evidence=_free)
    assert belief_state(back[0].existence) is ObjectBeliefState.PRESENT


def test_several_boxes_on_one_track_in_a_run_are_one_hit():
    """Detector at 5 Hz, memory at 2 Hz: a run's batches are one observation."""
    once, thrice = ObjectMemory(), ObjectMemory()
    a = once.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    thrice.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    b = thrice.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)] * 3, stamp_ns=2, miss_evidence=_none)
    c = once.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=2, miss_evidence=_none)
    assert a[0].existence is not None and c[0].existence is not None
    assert b[0].existence == c[0].existence > a[0].existence


def test_rejects_evidence_weights_of_the_wrong_sign():
    for bad in (
        {"hit_log_odds": 0.0},
        {"miss_free_log_odds": 0.1},
        {"miss_occupied_log_odds": 0.1},
        {"belief_absent": 0.0},
        {"belief_max": 0.0},
    ):
        with pytest.raises(ValueError):
            ObjectMemory(**bad)


def test_restore_starts_uncertain_then_confirms_or_removes():
    """Issue #636 acceptance: restart restores as uncertain, then reconciles."""
    before = ObjectMemory()
    before.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A), _obj("bowl", (5.5, 5.5, 5.5), _BOX_B)],
        stamp_ns=7,
        miss_evidence=_none,
    )
    saved = SceneGraph.model_validate_json(before.to_scene_graph().model_dump_json())
    assert [n.state for n in saved.nodes] == [ObjectBeliefState.PRESENT] * 2

    mem = ObjectMemory()
    assert mem.restore(saved, map_frame="map") == 2
    out = mem.tick([], stamp_ns=8, miss_evidence=_none)
    assert {o.label for o in out} == {"cup", "bowl"}
    assert all(belief_state(o.existence) is ObjectBeliefState.UNCERTAIN for o in out)

    # The cup is seen again (confirmed); the bowl's place is observed clear (removed).
    def evidence(o):
        return MissEvidence.FREE if o.label == "bowl" else MissEvidence.NONE

    for t in range(9, 12):
        out = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=t, miss_evidence=evidence)
    assert [o.label for o in out] == ["cup"]
    assert belief_state(out[0].existence) is ObjectBeliefState.PRESENT


def test_restore_skips_what_it_cannot_reconcile():
    before = ObjectMemory()
    before.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    graph = before.to_scene_graph()
    node = graph.nodes[0]
    variants = [
        node.model_copy(update={"node_id": "no_box", "bbox_3d": None}),
        node.model_copy(update={"node_id": "gone", "state": ObjectBeliefState.ABSENT}),
        node.model_copy(
            update={"node_id": "odom", "pose": node.pose.model_copy(update={"frame_id": "odom"})}
        ),
    ]
    mem = ObjectMemory()
    assert mem.restore(SceneGraph(nodes=variants), map_frame="map") == 0


def test_restore_keeps_the_persisted_track_ids():
    """A reasoner that outlived the restart still holds ``obj_track_<id>``: same object."""
    before = ObjectMemory()
    cup = _obj("cup", (0.5, 0.5, 0.5), _BOX_A).model_copy(update={"track_id": 7})
    bowl = _obj("bowl", (5.5, 5.5, 5.5), _BOX_B).model_copy(update={"track_id": 3})
    before.tick([cup, bowl], stamp_ns=1, miss_evidence=_none)
    graph = before.to_scene_graph()
    scene_file_node = graph.nodes[0].model_copy(
        update={
            "node_id": "wine_bottle",
            "label": "wine",
            "bbox_3d": (9.0, 9.0, 9.0, 9.5, 9.5, 9.5),
        }
    )
    mem = ObjectMemory()
    assert mem.restore(SceneGraph(nodes=[*graph.nodes, scene_file_node]), map_frame="map") == 3
    assert {o.label: o.track_id for o in mem.objects()} == {"cup": 7, "bowl": 3, "wine": 8}
    fresh = mem.tick(
        [_obj("fork", (20.5, 20.5, 20.5), (20.0, 20.0, 20.0, 21.0, 21.0, 21.0))],
        stamp_ns=2,
        miss_evidence=_none,
    )
    assert max(o.track_id or 0 for o in fresh) == 9


def test_new_detection_does_not_adopt_an_id_a_restored_track_holds():
    before = ObjectMemory()
    before.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, miss_evidence=_none)
    mem = ObjectMemory()
    mem.restore(before.to_scene_graph(), map_frame="map")  # restored under id 0
    bowl = _obj("bowl", (5.5, 5.5, 5.5), _BOX_B).model_copy(update={"track_id": 0})
    out = mem.tick([bowl], stamp_ns=2, miss_evidence=_none)
    assert {o.track_id for o in out} == {0, 1}


def test_within_run_duplicates_collapse_to_one_track():
    mem = ObjectMemory(iou_threshold=0.3)
    out = mem.tick(
        [
            _obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.9),
            _obj("cup", (0.55, 0.5, 0.5), _BOX_A_NUDGE, conf=0.7),
        ],
        stamp_ns=1,
        miss_evidence=_free,
    )
    assert len(out) == 1
    assert out[0].track_id == 0


def test_two_disjoint_detections_are_two_tracks():
    mem = ObjectMemory(iou_threshold=0.3)
    out = mem.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A), _obj("cup", (5.5, 5.5, 5.5), _BOX_B)],
        stamp_ns=1,
        miss_evidence=_free,
    )
    assert len(out) == 2
    assert {o.track_id for o in out} == {0, 1}


def test_match_bumps_confidence_upward_only_without_moving():
    mem = ObjectMemory(iou_threshold=0.3)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.6)], stamp_ns=1, miss_evidence=_free)
    # Higher-confidence re-detection (shifted box) -> confidence rises, pose frozen.
    out = mem.tick(
        [_obj("cup", (0.9, 0.5, 0.5), _BOX_A_NUDGE, conf=0.95)], stamp_ns=2, miss_evidence=_free
    )
    assert out[0].confidence == 0.95
    assert out[0].pose.xyz == (0.5, 0.5, 0.5)
    assert out[0].bbox_3d == _BOX_A
    # A subsequent lower-confidence re-detection must NOT lower it.
    out2 = mem.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.4)], stamp_ns=3, miss_evidence=_free
    )
    assert out2[0].confidence == 0.95
