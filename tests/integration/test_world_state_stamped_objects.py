"""WorldState <-> WorldStateStamped round-trip of detected_objects."""

from __future__ import annotations

import pytest

pytest.importorskip("openral_msgs")

from openral_core.schemas import (
    AttachedCollisionObject,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    DetectedObject,
    JointState,
    Pose6D,
    SphereShape,
    WorldState,
)
from openral_world_state_ros.lifecycle_node import (
    build_world_state_stamped_msg,
    world_state_from_idl,
)


def _ws(
    objects: list[DetectedObject],
    attached: list[AttachedCollisionObject] | None = None,
) -> WorldState:
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
        attached_objects=attached or [],
    )


def _obj() -> DetectedObject:
    return DetectedObject(
        label="cup",
        confidence=0.8,
        pose=Pose6D(xyz=(1.0, 2.0, 3.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        track_id=7,
    )


def _attached(
    object_id: str,
    shape: BoxShape | CapsuleShape | SphereShape,
) -> AttachedCollisionObject:
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        shape=shape,
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.16),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="panda_link7",
        ),
        mass_kg=0.18,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref=f"robocasa:{object_id}",
        stamp_ns=1,
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


def test_multiple_attached_objects_round_trip() -> None:
    attachments = [
        _attached("baguette_seed1", BoxShape(half_extents_m=(0.12, 0.025, 0.025))),
        _attached("cabinet_handle_tool", SphereShape(radius_m=0.03)),
        _attached("spatula_payload", CapsuleShape(radius_m=0.02, length_m=0.18)),
    ]

    msg = build_world_state_stamped_msg(None, _ws([_obj()], attachments))
    assert [item.object_id for item in msg.attached_objects] == [
        "baguette_seed1",
        "cabinet_handle_tool",
        "spatula_payload",
    ]
    assert msg.attached_objects[0].shape_type == msg.attached_objects[0].SHAPE_BOX
    assert msg.attached_objects[1].shape_type == msg.attached_objects[1].SHAPE_SPHERE
    assert msg.attached_objects[2].shape_type == msg.attached_objects[2].SHAPE_CAPSULE

    decoded = world_state_from_idl(msg)
    assert decoded.attached_objects == attachments
