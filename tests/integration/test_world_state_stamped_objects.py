"""WorldState <-> WorldStateStamped round-trip of detected_objects (ADR-0035)."""

from __future__ import annotations

import pytest

pytest.importorskip("openral_msgs")

from openral_core.schemas import DetectedObject, JointState, Pose6D, WorldState
from openral_world_state_ros.lifecycle_node import (
    build_world_state_stamped_msg,
    world_state_from_idl,
)


def _ws(objects: list[DetectedObject]) -> WorldState:
    return WorldState(
        stamp_ns=1,
        joint_state=JointState(
            name=["j0"],
            position=[0.0],
            velocity=[0.0],
            effort=[0.0],
            stamp_ns=1,
        ),
        detected_objects=objects,
    )


def _obj() -> DetectedObject:
    return DetectedObject(
        label="cup",
        confidence=0.8,
        pose=Pose6D(xyz=(1.0, 2.0, 3.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        track_id=7,
    )


def test_detected_objects_serialize_into_stamped() -> None:
    msg = build_world_state_stamped_msg(None, _ws([_obj()]))
    assert list(msg.detected_object_labels) == ["cup"]
    assert msg.detected_object_confidences[0] == pytest.approx(0.8, abs=1e-5)
    p = msg.detected_object_positions[0]
    assert (p.x, p.y, p.z) == (1.0, 2.0, 3.0)
    assert list(msg.detected_object_track_ids) == [7]
    assert msg.detected_object_frame == "map"


def test_empty_detected_objects_serialize_empty() -> None:
    msg = build_world_state_stamped_msg(None, _ws([]))
    assert list(msg.detected_object_labels) == []
    assert msg.detected_object_frame == ""


def test_round_trip_back_to_worldstate() -> None:
    back = world_state_from_idl(build_world_state_stamped_msg(None, _ws([_obj()])))
    assert len(back.detected_objects) == 1
    o = back.detected_objects[0]
    assert o.label == "cup"
    assert o.pose.xyz == (1.0, 2.0, 3.0)
    assert o.track_id == 7
    assert o.pose.frame_id == "map"
