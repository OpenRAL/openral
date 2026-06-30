"""Unit tests for ADR-0076 — camera-space 2D detection identity.

Pins the `DetectionTracker2D` (stable det_ids across frames, new-object minting,
miss eviction) and the id propagation that makes it "all work when octomap is
available": a detection-time `det_id` flows through the 3D lift into
`ObjectMemory`, so a physical object carries one id with or without depth.
"""

from __future__ import annotations

from openral_core import DetectionTracker2D, aabb_iou_2d
from openral_core.schemas import DetectedObject, ObjectDetection2D, Pose6D
from openral_world_state.object_memory import ObjectMemory


def _det(label: str, box: tuple[int, int, int, int], conf: float = 0.9) -> ObjectDetection2D:
    return ObjectDetection2D(label=label, confidence=conf, bbox_xyxy=box)


def test_aabb_iou_2d_identity_and_disjoint() -> None:
    assert aabb_iou_2d((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert aabb_iou_2d((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    assert 0.0 < aabb_iou_2d((0, 0, 10, 10), (5, 0, 15, 10)) < 1.0


def test_same_object_keeps_id_across_frames() -> None:
    """A drifting box (high IoU, same label) keeps its det_id; a new object mints."""
    trk = DetectionTracker2D()
    a = trk.assign([_det("milk", (400, 210, 430, 260)), _det("ketchup", (370, 230, 406, 272))])
    assert [(d.label, d.det_id) for d in a] == [("milk", 0), ("ketchup", 1)]
    b = trk.assign(
        [
            _det("milk", (402, 211, 432, 261)),  # same milk, moved
            _det("ketchup", (372, 231, 408, 273)),
            _det("alphabet soup", (440, 200, 470, 250)),  # new
        ]
    )
    by_label = {d.label: d.det_id for d in b}
    assert by_label == {"milk": 0, "ketchup": 1, "alphabet soup": 2}


def test_duplicate_labels_get_distinct_ids() -> None:
    """Two same-label objects far apart are two tracks (de-dup by id, not label)."""
    trk = DetectionTracker2D()
    out = trk.assign([_det("bowl", (10, 10, 40, 40)), _det("bowl", (300, 300, 340, 340))])
    ids = sorted(d.det_id for d in out)
    assert ids == [0, 1]


def test_miss_eviction_retires_a_vanished_track() -> None:
    """A track unseen for max_misses frames is retired; a later box mints a fresh id."""
    trk = DetectionTracker2D(max_misses=2)
    assert trk.assign([_det("cup", (10, 10, 40, 40))])[0].det_id == 0
    trk.assign([])  # miss 1 (still retained)
    trk.assign([])  # miss 2 → retired
    # same-looking box now → a NEW id (the old track is gone)
    assert trk.assign([_det("cup", (10, 10, 40, 40))])[0].det_id == 1


def test_det_id_propagates_through_lift_into_object_memory() -> None:
    """ADR-0076 §3: a new lifted object ADOPTS its det_id (via DetectedObject.track_id)
    instead of ObjectMemory minting a fresh one — one id across 2D and 3D."""
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    # The lift sets DetectedObject.track_id = ObjectDetection2D.det_id (here 7).
    lifted = DetectedObject(
        label="milk",
        confidence=0.9,
        pose=Pose6D(xyz=(0.5, 0.5, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        track_id=7,
    )
    out = mem.tick([lifted], stamp_ns=1, in_fov=lambda o: True)
    assert out[0].track_id == 7  # adopted, not minted as 0


def test_object_memory_still_mints_for_untracked_legacy_detections() -> None:
    """track_id None (legacy detector / det_id -1) → ObjectMemory mints as before."""
    mem = ObjectMemory(iou_threshold=0.3, max_misses=1)
    legacy = DetectedObject(
        label="cup",
        confidence=0.9,
        pose=Pose6D(xyz=(0.5, 0.5, 0.5), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        track_id=None,
    )
    out = mem.tick([legacy], stamp_ns=1, in_fov=lambda o: True)
    assert out[0].track_id == 0
