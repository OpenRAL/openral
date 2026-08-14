"""Typed attached-payload contract against the real PandaMobile manifest."""

from __future__ import annotations

import pytest
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    JointState,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    SupportContactWitness,
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


def _witness() -> SupportContactWitness:
    """The counter contact the baguette run actually had at attach.

    Depth is the deepest of the six real MuJoCo counter-payload contacts
    recorded on 2026-08-13 (-0.87..-1.37 mm) — a physical figure, not the
    -15.70 mm the 25 mm occupancy lattice reported for the same contact.
    """
    return SupportContactWitness(
        support_id="robocasa:counter_main",
        contact_point_in_object=(0.0, 0.0, -0.025),
        contact_normal_in_object=(0.0, 0.0, 1.0),
        patch_radius_m=0.13,
        max_penetration_m=0.00137,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_CONTACT,
        evidence_ref="mujoco_body:counter_main",
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


def test_attachment_rejects_more_than_sixteen_primitives() -> None:
    payload = _attachment().model_dump()
    payload["primitives"] = payload["primitives"] * 9

    with pytest.raises(ValidationError, match="at most 16"):
        AttachedCollisionObject.model_validate(payload)


def test_attachment_defaults_to_no_support_attestation() -> None:
    # Fail-closed by construction: a producer that says nothing about support
    # contact — the vision producer, an operator transition — licenses no
    # exemption in the safety kernel at all.
    assert _attachment().support_contact is None


def test_support_witness_rides_on_the_attachment_and_round_trips() -> None:
    attachment = _attachment().model_copy(update={"support_contact": _witness()})

    decoded = AttachedCollisionObject.model_validate_json(attachment.model_dump_json())
    assert decoded.support_contact is not None
    assert decoded.support_contact.support_id == "robocasa:counter_main"
    # The attested depth stays physical. Inflating it to cover the lattice's
    # 15.7 mm reading would license real penetration.
    assert decoded.support_contact.max_penetration_m == pytest.approx(0.00137)


def test_support_witness_rejects_a_non_unit_normal() -> None:
    payload = _witness().model_dump()
    payload["contact_normal_in_object"] = (0.0, 0.0, 4.0)

    with pytest.raises(ValidationError, match="must be a unit vector"):
        SupportContactWitness.model_validate(payload)


def test_support_witness_rejects_an_unbounded_patch_or_negative_depth() -> None:
    payload = _witness().model_dump()
    payload["patch_radius_m"] = 0.0
    with pytest.raises(ValidationError, match="patch_radius_m"):
        SupportContactWitness.model_validate(payload)

    payload = _witness().model_dump()
    payload["max_penetration_m"] = -0.001
    with pytest.raises(ValidationError, match="max_penetration_m"):
        SupportContactWitness.model_validate(payload)


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


# -- Place declaration + its region, at the World State authority (ADR-0097) --
#
# HZ-0097-3 mitigation 2 makes expiry World State's responsibility rather than
# the dispatcher's alone: a dispatcher that dies after issuing a declaration but
# before retracting it must not leave one live. HZ-0097-4 mitigation 4 inherits
# that verbatim for the approach allowance the declaration's region carries — the
# allowance is live only while the declaration is.


def _region(now_ns: int) -> PlaceRegion:
    return PlaceRegion(
        frame_id="base_link",
        pose=Pose6D(
            xyz=(0.62, 0.0, 1.05),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
        ),
        half_extents=(0.18, 0.25, 0.45),
        evidence_ref="mujoco_body_subtree:cab_1_left_group_main",
        stamp_ns=now_ns,
    )


def _place_declaration(now_ns: int, *, timeout_s: float = 60.0, region: bool = True) -> object:
    return PlaceDeclaration(
        target_id="sim:cab_1_left_group_main",
        object_id="baguette_seed1",
        rskill_id="openral/pi05-robocasa",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=timeout_s,
        stamp_ns=now_ns,
        region=_region(now_ns) if region else None,
    )


def test_aggregator_publishes_the_declared_regions_alongside_its_payload() -> None:
    """The region and the payload it is scoped to replace atomically.

    The safety kernel resolves the region's object mask against the attachment
    set in the same snapshot, so a region that could arrive out of step with its
    payload would scope an allowance to the wrong object.
    """
    clock = {"ns": 1_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    attachment = _attachment()
    aggregator.update_attached_objects(
        [attachment],
        revision=1,
        stamp_ns=clock["ns"],
        place_declaration=_place_declaration(clock["ns"]),
    )

    snapshot = aggregator.snapshot()
    assert snapshot.attached_objects == [attachment]
    declaration = snapshot.place_declaration
    assert declaration is not None
    assert declaration.target_id == "sim:cab_1_left_group_main"
    assert declaration.region is not None
    assert declaration.region.frame_id == "base_link"
    assert declaration.region.half_extents == (0.18, 0.25, 0.45)


def test_aggregator_drops_a_declaration_that_is_already_dead_on_arrival() -> None:
    """A retracted or already-expired declaration is stored as no declaration."""
    clock = {"ns": 40_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    retracted = _place_declaration(clock["ns"]).model_copy(update={"active": False})
    aggregator.update_attached_objects(
        [_attachment()], revision=1, stamp_ns=clock["ns"], place_declaration=retracted
    )
    assert aggregator.snapshot().place_declaration is None

    stale = _place_declaration(clock["ns"] - 30_000_000_000, timeout_s=1.0)
    aggregator.update_attached_objects(
        [_attachment()], revision=2, stamp_ns=clock["ns"], place_declaration=stale
    )
    assert aggregator.snapshot().place_declaration is None


def test_the_declaration_expires_on_the_backstop_with_no_retraction() -> None:
    """The crash path: no retraction ever arrives, and the region still dies.

    Nothing re-publishes — the dispatcher is gone — so the only thing that can
    end the allowance is the aggregator re-checking liveness on every snapshot.
    """
    clock = {"ns": 1_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=clock["ns"],
        place_declaration=_place_declaration(clock["ns"], timeout_s=5.0),
    )
    assert aggregator.snapshot().place_declaration is not None

    clock["ns"] += 4_000_000_000
    assert aggregator.snapshot().place_declaration is not None
    clock["ns"] += 2_000_000_000  # past the 5 s backstop
    assert aggregator.snapshot().place_declaration is None


def test_a_declaration_without_a_region_carries_no_allowance() -> None:
    """Dispatch's own publication: it names a target, it measures no geometry.

    The declaration still gates the place witness; the approach allowance needs a
    producer-measured region, and its absence is the pre-amendment behaviour.
    """
    clock = {"ns": 1_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=clock["ns"],
        place_declaration=_place_declaration(clock["ns"], region=False),
    )
    declaration = aggregator.snapshot().place_declaration
    assert declaration is not None
    assert declaration.region is None


def test_detaching_takes_the_declaration_with_it() -> None:
    """No payload, no declaration: the allowance cannot outlive what it was for."""
    clock = {"ns": 1_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=clock["ns"],
        place_declaration=_place_declaration(clock["ns"]),
    )
    assert aggregator.snapshot().place_declaration is not None

    aggregator.update_attached_objects([], revision=2, stamp_ns=clock["ns"])
    snapshot = aggregator.snapshot()
    assert snapshot.attached_objects == []
    assert snapshot.place_declaration is None
