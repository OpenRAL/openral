"""Typed attached-payload contract against the real PandaMobile manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    CapsuleShape,
    JointState,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    SphereShape,
    SupportContactWitness,
    WorldState,
)
from openral_core.exceptions import ROSConfigError
from openral_world_state import WorldStateAggregator
from pydantic import ValidationError

_ROBOT_YAML = "robots/panda_mobile/robot.yaml"


@dataclass(frozen=True)
class _CylinderShape:
    """A fourth primitive kind — what a future ``CollisionShape`` member looks like.

    ``CollisionShape`` is closed over sphere/capsule/box, so this is the only
    way to reach the encoder's unknown-shape branch. Mirrors the stand-in in
    ``tests/unit/test_collision_params.py`` and
    ``packages/openral_slam_bringup/test/test_depth_height_filter.py``.
    """

    shape: str = "cylinder"
    radius_m: float = 0.1
    length_m: float = 0.4


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
#
# These tests run the aggregator on its PRODUCTION default clock (no `clock_fn`,
# i.e. wall `time.time_ns`) and stamp the stream in simulator time, because that
# is the round-7 field configuration and the only one that can see a clock-domain
# mismatch. An earlier revision injected `clock_fn` AND stamped from the same
# injected clock, which made the two domains identical by construction and let a
# wall-vs-sim comparison ship: the aggregator re-checked liveness against
# `time.time_ns` (~1.79e18) while the declaration was stamped in sim time
# (~1.2e9), so every published `WorldStateStamped` carried
# `place_declaration=None` and the approach allowance could never arm.

# 1.24 s of simulator time — the domain the sim producer, the rSkill runner and
# the safety kernel all stamp in under `use_sim_time`.
_SIM_ARM_NS = 1_240_000_000


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


def test_a_sim_stamped_declaration_survives_the_production_wall_clock() -> None:
    """The round-7 regression: sim-stamped stream, production wall-clock aggregator.

    The aggregator is built exactly as ``lifecycle_node`` builds it — no
    ``clock_fn``, so ``time.time_ns`` — while the producer stamps the attachment
    stream and the declaration in simulator time. Liveness must be judged in the
    stream's clock, so the declaration reaches the published snapshot; judging it
    against the aggregator's own wall clock made every declaration ~57 years past
    its backstop and dropped it, which is why the kernel never received the
    region and the approach allowance never armed.
    """
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    attachment = _attachment()
    aggregator.update_attached_objects(
        [attachment],
        revision=1,
        stamp_ns=_SIM_ARM_NS,
        place_declaration=_place_declaration(_SIM_ARM_NS, timeout_s=60.0),
    )

    snapshot = aggregator.snapshot()
    # The two domains really are different here — this is what the old test's
    # injected clock hid. The snapshot's own stamp is wall time; the attachment
    # stream's is simulator time, ~9 orders of magnitude apart.
    assert snapshot.stamp_ns > _SIM_ARM_NS * 1_000_000
    assert snapshot.attachment_stamp_ns == _SIM_ARM_NS
    assert snapshot.attached_objects == [attachment]
    declaration = snapshot.place_declaration
    assert declaration is not None
    assert declaration.target_id == "sim:cab_1_left_group_main"
    assert declaration.region is not None
    assert declaration.region.frame_id == "base_link"
    assert declaration.region.half_extents == (0.18, 0.25, 0.45)


def test_aggregator_publishes_the_declared_regions_alongside_its_payload() -> None:
    """The region and the payload it is scoped to replace atomically.

    The safety kernel resolves the region's object mask against the attachment
    set in the same snapshot, so a region that could arrive out of step with its
    payload would scope an allowance to the wrong object.
    """
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    first = _attachment()
    aggregator.update_attached_objects(
        [first],
        revision=1,
        stamp_ns=_SIM_ARM_NS,
        place_declaration=_place_declaration(_SIM_ARM_NS),
    )
    assert aggregator.snapshot().place_declaration is not None

    # A later revision that carries a payload but no declaration takes the
    # region with it, in the same atomic replacement.
    second = _attachment("spatula_payload")
    aggregator.update_attached_objects(
        [second],
        revision=2,
        stamp_ns=_SIM_ARM_NS + 500_000_000,
    )
    snapshot = aggregator.snapshot()
    assert snapshot.attached_objects == [second]
    assert snapshot.place_declaration is None


def test_aggregator_drops_a_declaration_that_is_already_dead_on_arrival() -> None:
    """A retracted or already-expired declaration is stored as no declaration."""
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    now_ns = _SIM_ARM_NS + 40_000_000_000
    retracted = _place_declaration(now_ns).model_copy(update={"active": False})
    aggregator.update_attached_objects(
        [_attachment()], revision=1, stamp_ns=now_ns, place_declaration=retracted
    )
    assert aggregator.snapshot().place_declaration is None

    stale = _place_declaration(now_ns - 30_000_000_000, timeout_s=1.0)
    aggregator.update_attached_objects(
        [_attachment()], revision=2, stamp_ns=now_ns, place_declaration=stale
    )
    assert aggregator.snapshot().place_declaration is None


def test_the_declaration_expires_on_the_backstop_with_no_retraction() -> None:
    """The crash path: no retraction ever arrives, and the region still dies.

    The dispatcher is gone, so nothing retracts — but the evidence producer
    heartbeats the attachment set, and each heartbeat re-runs the backstop
    against its own stream stamp. Expiry is therefore driven by the stream
    clock advancing past ``timeout_s``, never by the aggregator's wall clock,
    which is in a different domain entirely (HZ-0097-3 mitigation 2).
    """
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    declaration = _place_declaration(_SIM_ARM_NS, timeout_s=5.0)
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=_SIM_ARM_NS,
        place_declaration=declaration,
    )
    assert aggregator.snapshot().place_declaration is not None

    # Heartbeat 4 s of simulator time later — still inside the 5 s backstop.
    aggregator.update_attached_objects(
        [_attachment()],
        revision=2,
        stamp_ns=_SIM_ARM_NS + 4_000_000_000,
        place_declaration=declaration,
    )
    assert aggregator.snapshot().place_declaration is not None

    # Heartbeat past the backstop: dropped, with no retraction and no republish
    # of a fresher declaration.
    aggregator.update_attached_objects(
        [_attachment()],
        revision=3,
        stamp_ns=_SIM_ARM_NS + 6_000_000_000,
        place_declaration=declaration,
    )
    assert aggregator.snapshot().place_declaration is None


def test_an_unstamped_attachment_expires_the_declaration_on_arrival_time() -> None:
    """The one path where the aggregator's own clock IS the stream's clock.

    ``stamp_ns=None`` means "no producer stamp, use arrival time", so ``clock_fn``
    is the stream's only clock reading and the declaration must be stamped from
    the same one. Injecting the clock here tests that fallback for real rather
    than hiding a domain mismatch.
    """
    clock = {"ns": 1_700_000_000_000_000_000}
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml(_ROBOT_YAML), clock_fn=lambda: clock["ns"]
    )
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        place_declaration=_place_declaration(clock["ns"], timeout_s=5.0),
    )
    snapshot = aggregator.snapshot()
    assert snapshot.attachment_stamp_ns == clock["ns"]
    assert snapshot.place_declaration is not None

    clock["ns"] += 6_000_000_000  # past the 5 s backstop
    aggregator.update_attached_objects(
        [_attachment()],
        revision=2,
        place_declaration=_place_declaration(clock["ns"] - 6_000_000_000, timeout_s=5.0),
    )
    assert aggregator.snapshot().place_declaration is None


def test_a_declaration_without_a_region_carries_no_allowance() -> None:
    """Dispatch's own publication: it names a target, it measures no geometry.

    The declaration still gates the place witness; the approach allowance needs a
    producer-measured region, and its absence is the pre-amendment behaviour.
    """
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=_SIM_ARM_NS,
        place_declaration=_place_declaration(_SIM_ARM_NS, region=False),
    )
    declaration = aggregator.snapshot().place_declaration
    assert declaration is not None
    assert declaration.region is None


def test_detaching_takes_the_declaration_with_it() -> None:
    """No payload, no declaration: the allowance cannot outlive what it was for."""
    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT_YAML))
    aggregator.update_attached_objects(
        [_attachment()],
        revision=1,
        stamp_ns=_SIM_ARM_NS,
        place_declaration=_place_declaration(_SIM_ARM_NS),
    )
    assert aggregator.snapshot().place_declaration is not None

    aggregator.update_attached_objects([], revision=2, stamp_ns=_SIM_ARM_NS + 100_000_000)
    snapshot = aggregator.snapshot()
    assert snapshot.attached_objects == []
    assert snapshot.place_declaration is None


# ── AttachedCollisionPrimitive.fill_idl fail-closed contract ──────────────────

_PRIMITIVE_MSG = Path("packages/msgs/msg/AttachedCollisionPrimitive.msg")


def _idl_shape_constants() -> dict[str, int]:
    """The ``SHAPE_*`` constants, read out of the real ``.msg``.

    Deriving them keeps the duck-typed stand-in below honest: if the IDL ever
    renumbers or renames a tag, these tests fail rather than silently agreeing
    with a stale hand-copied number.
    """
    constants: dict[str, int] = {}
    for line in _PRIMITIVE_MSG.read_text().splitlines():
        match = re.match(r"\s*uint8\s+(SHAPE_\w+)\s*=\s*(\d+)", line)
        if match:
            constants[match.group(1)] = int(match.group(2))
    return constants


class _Vec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


class _Pose:
    def __init__(self) -> None:
        self.position = _Vec()
        self.orientation = _Vec()


class _PrimitiveMsg:
    """``openral_msgs/AttachedCollisionPrimitive`` shaped without importing ROS.

    ``fill_idl`` is defined against this duck type on purpose (openral_core must
    stay import-safe without a sourced ROS 2 install), so this is the real
    interface under test, not a stand-in for one. Field defaults mirror the
    IDL's: ``shape_type`` 0 and an empty ``shape_dimensions``.
    """

    def __init__(self) -> None:
        for name, value in _idl_shape_constants().items():
            setattr(self, name, value)
        self.shape_type: int = 0
        self.shape_dimensions: list[float] = []
        self.pose_in_object = _Pose()


def test_idl_reserves_zero_so_an_unset_shape_type_is_not_a_valid_tag() -> None:
    """No ``SHAPE_*`` constant is 0 — the IDL default is an unencodable tag."""
    constants = _idl_shape_constants()
    assert constants == {"SHAPE_SPHERE": 1, "SHAPE_CAPSULE": 2, "SHAPE_BOX": 3}
    assert 0 not in constants.values()


@pytest.mark.parametrize(
    ("shape", "expected_tag", "expected_dims"),
    [
        (SphereShape(radius_m=0.05), "SHAPE_SPHERE", [0.05]),
        (CapsuleShape(radius_m=0.04, length_m=0.3), "SHAPE_CAPSULE", [0.04, 0.3]),
        (BoxShape(half_extents_m=(0.1, 0.2, 0.3)), "SHAPE_BOX", [0.1, 0.2, 0.3]),
    ],
)
def test_fill_idl_encodes_every_union_member(
    shape: object, expected_tag: str, expected_dims: list[float]
) -> None:
    """Each of the three union members lowers to its IDL tag + dimension layout."""
    primitive = AttachedCollisionPrimitive(
        shape=shape,  # type: ignore[arg-type]  # reason: parametrized over the union
        pose_in_object=Pose6D(
            xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="payload"
        ),
    )
    msg = _PrimitiveMsg()
    primitive.fill_idl(msg)
    assert msg.shape_type == _idl_shape_constants()[expected_tag]
    assert msg.shape_dimensions == pytest.approx(expected_dims)


def test_fill_idl_refuses_a_shape_the_idl_cannot_carry() -> None:
    """An unencodable primitive raises instead of publishing an unset shape.

    The branch chain had no ``else``, so the message went out with
    ``shape_type`` at the IDL default 0 and no dimensions. Every consumer
    already refuses tag 0 (the C++ kernel's attached ingest calls
    ``fail_closed()``; the Nav2 payload scan filter raises), so this was a
    diagnosability defect rather than a safety hole — but it pushed a
    producer-side encoding failure across a process boundary, where it
    resurfaced as an E-stop or an unrelated node's ``ValueError`` with the
    shape's name nowhere in the report.
    """
    primitive = AttachedCollisionPrimitive.model_construct(
        shape=_CylinderShape(),  # type: ignore[arg-type]  # reason: future-variant stand-in
        pose_in_object=Pose6D(
            xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="payload"
        ),
    )
    msg = _PrimitiveMsg()

    with pytest.raises(ROSConfigError) as excinfo:
        primitive.fill_idl(msg)

    assert "cylinder" in str(excinfo.value)
    # The refusal must leave the message untouched, not half-populated.
    assert msg.shape_type == 0
    assert msg.shape_dimensions == []
