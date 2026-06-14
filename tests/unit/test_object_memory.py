"""Unit tests for ADR-0035 ObjectMemory (freeze-on-match, FOV-guarded evict)."""

from __future__ import annotations

from openral_core.schemas import DetectedObject, Pose6D
from openral_world_state.object_memory import ObjectMemory


def _obj(label, xyz, box, conf=0.9):
    return DetectedObject(
        label=label,
        confidence=conf,
        pose=Pose6D(xyz=xyz, quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=box,
    )


_BOX_A = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
_BOX_A_NUDGE = (0.05, 0.0, 0.0, 1.05, 1.0, 1.0)  # high IoU vs A
_BOX_B = (5.0, 5.0, 5.0, 6.0, 6.0, 6.0)  # disjoint from A


def _always(o):
    return True


def _never(o):
    return False


def test_new_detection_creates_track_with_id():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    out = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    assert len(out) == 1
    assert out[0].track_id == 0
    assert out[0].label == "cup"


def test_match_freezes_pose_and_keeps_track_id():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    out = mem.tick([_obj("cup", (0.9, 0.5, 0.5), _BOX_A_NUDGE)], stamp_ns=2, in_fov=_always)
    assert len(out) == 1
    assert out[0].track_id == 0
    assert out[0].pose.xyz == (0.5, 0.5, 0.5)
    assert out[0].bbox_3d == _BOX_A


def test_different_label_same_box_is_a_new_track():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    # Same box, both labels detected this run: the bowl must NOT be absorbed
    # into the cup track (association is label-specific) -> two tracks; both
    # are matched/created this run so neither is evicted.
    out = mem.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A), _obj("bowl", (0.5, 0.5, 0.5), _BOX_A)],
        stamp_ns=2,
        in_fov=_always,
    )
    assert {o.label for o in out} == {"cup", "bowl"}
    assert {o.track_id for o in out} == {0, 1}


def test_evicts_in_fov_miss_at_max_misses_1():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    out = mem.tick([], stamp_ns=2, in_fov=_always)
    assert out == []


def test_retains_out_of_fov_object():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    out = mem.tick([], stamp_ns=2, in_fov=_never)
    assert len(out) == 1
    assert out[0].track_id == 0


def test_flicker_tolerance_max_misses_2():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=2)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    assert len(mem.tick([], stamp_ns=2, in_fov=_always)) == 1
    assert mem.tick([], stamp_ns=3, in_fov=_always) == []


def test_miss_count_resets_on_redetection():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=2)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=1, in_fov=_always)
    mem.tick([], stamp_ns=2, in_fov=_always)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A)], stamp_ns=3, in_fov=_always)
    assert len(mem.tick([], stamp_ns=4, in_fov=_always)) == 1


def test_within_run_duplicates_collapse_to_one_track():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    out = mem.tick(
        [
            _obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.9),
            _obj("cup", (0.55, 0.5, 0.5), _BOX_A_NUDGE, conf=0.7),
        ],
        stamp_ns=1,
        in_fov=_always,
    )
    assert len(out) == 1
    assert out[0].track_id == 0


def test_two_disjoint_detections_are_two_tracks():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    out = mem.tick(
        [_obj("cup", (0.5, 0.5, 0.5), _BOX_A), _obj("cup", (5.5, 5.5, 5.5), _BOX_B)],
        stamp_ns=1,
        in_fov=_always,
    )
    assert len(out) == 2
    assert {o.track_id for o in out} == {0, 1}


def test_match_bumps_confidence_upward_only_without_moving():
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.6)], stamp_ns=1, in_fov=_always)
    # Higher-confidence re-detection (shifted box) -> confidence rises, pose frozen.
    out = mem.tick(
        [_obj("cup", (0.9, 0.5, 0.5), _BOX_A_NUDGE, conf=0.95)], stamp_ns=2, in_fov=_always
    )
    assert out[0].confidence == 0.95
    assert out[0].pose.xyz == (0.5, 0.5, 0.5)
    assert out[0].bbox_3d == _BOX_A
    # A subsequent lower-confidence re-detection must NOT lower it.
    out2 = mem.tick([_obj("cup", (0.5, 0.5, 0.5), _BOX_A, conf=0.4)], stamp_ns=3, in_fov=_always)
    assert out2[0].confidence == 0.95
