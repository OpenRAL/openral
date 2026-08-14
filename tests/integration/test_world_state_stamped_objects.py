"""WorldState <-> WorldStateStamped round-trip of detected_objects."""

from __future__ import annotations

import pytest

pytest.importorskip("openral_msgs")
pytest.importorskip("rclpy")

import rclpy
from openral_core.schemas import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    DetectedObject,
    JointState,
    Pose6D,
    RobotDescription,
    SphereShape,
    SupportContactWitness,
    WorldState,
)
from openral_msgs.msg import AttachmentState
from openral_world_state import WorldStateAggregator
from openral_world_state_ros.lifecycle_node import (
    _WorldStateLifecycleNode,
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
    *,
    support_contact: SupportContactWitness | None = None,
) -> AttachedCollisionObject:
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        primitives=[
            AttachedCollisionPrimitive(
                shape=shape,
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            )
        ],
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
        support_contact=support_contact,
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
    baguette = _attached(
        "baguette_seed1",
        BoxShape(half_extents_m=(0.12, 0.025, 0.025)),
    )
    baguette = baguette.model_copy(
        update={
            "primitives": [
                *baguette.primitives,
                AttachedCollisionPrimitive(
                    shape=SphereShape(radius_m=0.02),
                    pose_in_object=Pose6D(
                        xyz=(0.1, 0.0, 0.0),
                        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        frame_id="baguette_seed1",
                    ),
                ),
            ]
        }
    )
    attachments = [
        baguette,
        _attached("cabinet_handle_tool", SphereShape(radius_m=0.03)),
        _attached("spatula_payload", CapsuleShape(radius_m=0.02, length_m=0.18)),
    ]

    msg = build_world_state_stamped_msg(None, _ws([_obj()], attachments))
    assert [item.object_id for item in msg.attached_objects] == [
        "baguette_seed1",
        "cabinet_handle_tool",
        "spatula_payload",
    ]
    assert (
        msg.attached_objects[0].primitives[0].shape_type
        == msg.attached_objects[0].primitives[0].SHAPE_BOX
    )
    assert (
        msg.attached_objects[0].primitives[1].shape_type
        == msg.attached_objects[0].primitives[1].SHAPE_SPHERE
    )
    assert (
        msg.attached_objects[1].primitives[0].shape_type
        == msg.attached_objects[1].primitives[0].SHAPE_SPHERE
    )
    assert (
        msg.attached_objects[2].primitives[0].shape_type
        == msg.attached_objects[2].primitives[0].SHAPE_CAPSULE
    )

    decoded = world_state_from_idl(msg)
    assert decoded.attached_objects == attachments


def test_attachment_state_callback_applies_multiple_objects_atomically() -> None:
    attachments = [
        _attached("baguette_seed1", BoxShape(half_extents_m=(0.12, 0.025, 0.025))),
        _attached("cabinet_handle_tool", SphereShape(radius_m=0.03)),
    ]
    wire_objects = list(build_world_state_stamped_msg(None, _ws([], attachments)).attached_objects)
    state_msg = AttachmentState()
    state_msg.revision = 7
    state_msg.objects = wire_objects
    aggregator = WorldStateAggregator(RobotDescription.from_yaml("robots/panda_mobile/robot.yaml"))

    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    try:
        node._on_attachment_state(state_msg)
        assert aggregator.snapshot().attached_objects == attachments
        assert aggregator.snapshot().attachment_revision == 7
        assert aggregator.snapshot().attachment_stamp_ns > 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_support_contact_witness_round_trips_over_the_idl() -> None:
    # The wire half of ADR-0092 D6. An attachment that attests support contact
    # must arrive at the safety kernel with the attestation intact; one that
    # does not must arrive with support_contact_valid False, because a lost
    # witness would silently re-open the false stop and an invented one would
    # silently license real penetration.
    # The 2026-08-14 acceptance run's attested baguette, verbatim — including
    # the evidence kind the MuJoCo producer actually emits, which is the signed
    # distance probe and not the solver's contact list.
    witness = SupportContactWitness(
        support_id="robocasa:counter_main",
        contact_point_in_object=(0.0, 0.0, -0.025),
        contact_normal_in_object=(0.0, 0.0, 1.0),
        patch_radius_m=0.1476,
        max_penetration_m=0.00118,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_GEOM_DISTANCE,
        evidence_ref="mujoco_geom_distance:counter_main",
        stamp_ns=99,
    )
    attachments = [
        _attached(
            "baguette_seed1",
            BoxShape(half_extents_m=(0.12, 0.025, 0.025)),
            support_contact=witness,
        ),
        _attached("cabinet_handle_tool", SphereShape(radius_m=0.03)),
    ]

    msg = build_world_state_stamped_msg(None, _ws([], attached=attachments))
    assert msg.attached_objects[0].support_contact_valid is True
    assert msg.attached_objects[0].support_contact.support_id == "robocasa:counter_main"
    assert msg.attached_objects[0].support_contact.max_penetration_m == pytest.approx(0.00118)
    assert msg.attached_objects[0].support_contact.contact_normal_in_object.z == pytest.approx(1.0)
    assert msg.attached_objects[0].support_contact.evidence_kind == "sim_geom_distance"
    assert msg.attached_objects[1].support_contact_valid is False

    decoded = world_state_from_idl(msg)
    assert decoded.attached_objects[0].support_contact == witness
    assert decoded.attached_objects[1].support_contact is None
