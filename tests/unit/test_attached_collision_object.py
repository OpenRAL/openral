"""Typed attached-payload contract against the real PandaMobile manifest."""

from __future__ import annotations

import pytest
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    JointState,
    Pose6D,
    RobotDescription,
    WorldState,
)
from openral_world_state import WorldStateAggregator
from pydantic import ValidationError

_ROBOT_YAML = "robots/panda_mobile/robot.yaml"


def _attachment(object_id: str = "baguette_seed1") -> AttachedCollisionObject:
    robot = RobotDescription.from_yaml(_ROBOT_YAML)
    links = {joint.parent_link for joint in robot.joints} | {
        joint.child_link for joint in robot.joints
    }
    assert {"panda_link7", "panda_finger_pair"} <= links
    return AttachedCollisionObject(
        object_id=object_id,
        attach_link="panda_link7",
        touch_links=["panda_finger_pair"],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.12, 0.025, 0.025)),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            ),
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.025, 0.04, 0.02)),
                pose_in_object=Pose6D(
                    xyz=(0.09, 0.0, 0.0),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id=object_id,
                ),
            ),
        ],
        pose_in_link=Pose6D(
            xyz=(0.0, 0.0, 0.16),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="panda_link7",
        ),
        mass_kg=0.18,
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(0.0,) * 9,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref="robocasa:obj_main",
        stamp_ns=1,
    )


def test_attached_collision_object_uses_real_robot_links() -> None:
    attachment = _attachment()

    assert attachment.attach_link == "panda_link7"
    assert attachment.touch_links == ["panda_finger_pair"]
    assert len(attachment.primitives) == 2
    assert attachment.pose_in_link.frame_id == attachment.attach_link


def test_attachment_rejects_pose_in_another_frame() -> None:
    payload = _attachment().model_dump()
    payload["pose_in_link"]["frame_id"] = "base_link"

    with pytest.raises(ValidationError, match="frame_id must equal attach_link"):
        AttachedCollisionObject.model_validate(payload)


def test_attachment_rejects_duplicate_touch_links() -> None:
    payload = _attachment().model_dump()
    payload["touch_links"] = ["panda_finger_pair", "panda_finger_pair"]

    with pytest.raises(ValidationError, match="touch_links must be unique"):
        AttachedCollisionObject.model_validate(payload)


def test_attachment_rejects_primitive_in_another_object_frame() -> None:
    payload = _attachment().model_dump()
    payload["primitives"][0]["pose_in_object"]["frame_id"] = "another_object"

    with pytest.raises(ValidationError, match="frame_id must equal object_id"):
        AttachedCollisionObject.model_validate(payload)


def test_payload_dynamics_require_mass() -> None:
    payload = _attachment().model_dump()
    payload["mass_kg"] = None

    with pytest.raises(ValidationError, match="requires mass_kg"):
        AttachedCollisionObject.model_validate(payload)


def test_attachment_rejects_more_than_eight_primitives() -> None:
    payload = _attachment().model_dump()
    payload["primitives"] = payload["primitives"] * 5

    with pytest.raises(ValidationError, match="at most 8"):
        AttachedCollisionObject.model_validate(payload)


def test_world_state_carries_multiple_attachments() -> None:
    state = WorldState(
        stamp_ns=1,
        joint_state=JointState(name=["panda_joint1"], position=[0.0], stamp_ns=1),
        attached_objects=[_attachment(), _attachment("cabinet_handle_tool")],
    )

    decoded = WorldState.model_validate_json(state.model_dump_json())
    assert [obj.object_id for obj in decoded.attached_objects] == [
        "baguette_seed1",
        "cabinet_handle_tool",
    ]


def test_aggregator_replaces_multiple_attachments_atomically() -> None:
    robot = RobotDescription.from_yaml(_ROBOT_YAML)
    aggregator = WorldStateAggregator(robot)
    aggregator.update_attached_objects([_attachment("cabinet_handle_tool"), _attachment()])

    snapshot = aggregator.snapshot()
    assert [obj.object_id for obj in snapshot.attached_objects] == [
        "baguette_seed1",
        "cabinet_handle_tool",
    ]

    aggregator.update_attached_objects([])
    assert aggregator.snapshot().attached_objects == []


def test_aggregator_rejects_duplicate_attachment_ids() -> None:
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))

    with pytest.raises(ValueError, match="ids must be unique"):
        aggregator.update_attached_objects([_attachment(), _attachment()])


def test_aggregator_preserves_producer_revision_and_timestamp() -> None:
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    attachment = _attachment()
    aggregator.update_attached_objects(
        [attachment],
        revision=4,
        stamp_ns=123_000_000,
    )

    snapshot = aggregator.snapshot()
    assert snapshot.attachment_revision == 4
    assert snapshot.attachment_stamp_ns == 123_000_000

    with pytest.raises(ValueError, match="moved backwards"):
        aggregator.update_attached_objects([], revision=3, stamp_ns=124_000_000)
    assert aggregator.snapshot().attached_objects == [attachment]
