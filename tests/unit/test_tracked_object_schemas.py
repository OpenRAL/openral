"""Issue #418 object-centric tracking schemas: round-trip, JSON Schema, caps, IDL.

Fixtures come from the real SO-101, DROID Franka/Robotiq and OpenArm manifests and the
PickOrange deploy scene; the IDL round-trip uses the built ``openral_msgs`` when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    DeployRuntime,
    DeployScene,
    FitKind,
    GraspDeclaration,
    ObjectQuery,
    ObjectQueryStatus,
    ObjectRole,
    PlaceDeclaration,
    PlaceRegion,
    Pose6D,
    RobotDescription,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
)
from openral_core.schemas import DOP_AXES, BoxShape, TightCollisionGeometry
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]
_SCENE = _ROOT / "scenes/deploy/isaac_leisaac_pick_orange.yaml"
# (robot, gripper joint, hand link) — each name is asserted against its manifest.
_HANDS = [
    ("so101_follower", "gripper", "jaw"),
    ("droid_franka_robotiq", "finger_joint", "right_distal_phalanx"),
    ("openarm", "left_gripper", "openarm_left_finger_pair"),
]


def _hand(robot: str, joint: str, link: str) -> str:
    desc = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    assert joint in {j.name for j in desc.joints}
    assert link in {c.link_name for c in desc.collision_geometry}
    return desc.base_frame


def _scene_box() -> PlaceRegion:
    declaration = DeployScene.from_yaml(str(_SCENE)).grasp_declaration
    assert declaration is not None and declaration.search_box is not None
    return declaration.search_box


def _round_trip(instance: object) -> None:
    cls = type(instance)
    serialized = instance.model_dump_json()  # type: ignore[attr-defined]
    assert cls.model_validate_json(serialized) == instance  # type: ignore[attr-defined]
    jsonschema.validate(json.loads(serialized), cls.model_json_schema())  # type: ignore[attr-defined]


def _track(
    frame: str,
    *,
    object_id: str = "trk:1",
    status: TrackStatus = TrackStatus.CONFIRMED,
    **kw: object,
) -> TrackedObject:
    box = PlaceRegion(
        frame_id=frame,
        pose=Pose6D(xyz=(0.07, -0.30, 0.05), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame),
        half_extents=(0.03, 0.03, 0.03),
    )
    fields: dict[str, object] = {
        "object_id": object_id,
        "epoch": 0,
        "label": "orange",
        "role": ObjectRole.TARGET,
        "declaration_key": "obj:orange_1@0",
        "status": status,
        "box": box,
        "fit_kind": FitKind.SILHOUETTE_CONE,
        "support_z": 0.02,
        "support_ref": "map",
        "floor_z": 0.035,
        "visible_cells": 6,
        "occupied_cells": 5,
        "last_fit_stamp_ns": 10,
        "last_confirmed_stamp_ns": 12,
    }
    fields.update(kw)
    return TrackedObject.model_validate(fields)


@pytest.mark.parametrize(("robot", "joint", "link"), _HANDS)
def test_bound_track_on_real_manifest_round_trips(robot: str, joint: str, link: str) -> None:
    frame = _hand(robot, joint, link)
    hull = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(0.03, 0.03, 0.03)),
        pose_in_object=Pose6D(xyz=(0.07, -0.3, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id=frame),
    )
    bound = _track(
        frame,
        status=TrackStatus.BOUND,
        tight=(hull,),
        bound_hand=joint,
        attach_link=link,
        pose_in_link=Pose6D(xyz=(0.0, 0.0, 0.02), quat_xyzw=(0, 0, 0, 1), frame_id=link),
    )
    tracked = TrackedObjectSet(frame_id=frame, stamp_ns=12, revision=3, objects=[bound])
    _round_trip(tracked)
    payload = AttachedCollisionObject(
        object_id=bound.object_id,
        attach_link=link,
        touch_links=[link],
        primitives=[
            hull.model_copy(
                update={
                    "pose_in_object": hull.pose_in_object.model_copy(
                        update={"frame_id": bound.object_id}
                    )
                }
            )
        ],
        pose_in_link=bound.pose_in_link,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.TRACKED_OBJECT,
        stamp_ns=12,
        epoch=bound.epoch,
        geometry_revision=1,
    )
    _round_trip(payload)
    with pytest.raises(ValidationError, match="BOUND"):
        _track(frame, status=TrackStatus.BOUND)


def test_scene_declaration_carries_queries() -> None:
    box = _scene_box()
    seed = box.model_copy(update={"geometry": ()})
    grasp = DeployScene.from_yaml(str(_SCENE)).grasp_declaration
    assert grasp is not None
    grasp = grasp.model_copy(
        update={
            "target_queries": (
                ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, seed=seed),
            )
        }
    )
    _round_trip(GraspDeclaration.model_validate(grasp.model_dump()))
    place = PlaceDeclaration(
        target_id="obj:plate",
        timeout_s=60.0,
        stamp_ns=0,
        destination_query=ObjectQuery(label="plate", count="one", role=ObjectRole.DESTINATION),
    )
    _round_trip(place)
    with pytest.raises(ValidationError, match="destination"):
        PlaceDeclaration(
            target_id="obj:plate",
            timeout_s=60.0,
            stamp_ns=0,
            destination_query=ObjectQuery(label="plate", count="one", role=ObjectRole.TARGET),
        )
    with pytest.raises(ValidationError):
        GraspDeclaration.model_validate(
            {**grasp.model_dump(), "target_queries": [grasp.target_queries[0].model_dump()] * 9}
        )
    assert ObjectQueryStatus("exhausted") is ObjectQueryStatus.EXHAUSTED


def test_kernel_caps_are_validators() -> None:
    frame = _scene_box().frame_id
    big = _track(frame).box.model_copy(update={"half_extents": (0.25, 0.01, 0.01)})
    with pytest.raises(ValidationError, match="grasp-region cap"):
        _track(frame, box=big)
    many = [
        AttachedCollisionPrimitive(
            shape=BoxShape(half_extents_m=(0.01, 0.01, 0.01)),
            pose_in_object=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id=frame),
        )
    ] * 5
    with pytest.raises(ValidationError):
        _track(frame, tight=tuple(many))
    dense = many[0].model_copy(
        update={
            "tight_geometry": TightCollisionGeometry(
                dop_lo_m=(-0.01,) * len(DOP_AXES),
                dop_hi_m=(0.01,) * len(DOP_AXES),
                hull_vertices_m=tuple((0.0, 0.0, 0.0001 * i) for i in range(65)),
            )
        }
    )
    with pytest.raises(ValidationError, match="vertices"):
        _track(frame, tight=(dense,))
    with pytest.raises(ValidationError, match="floor_z"):
        _track(frame, floor_z=0.0)
    with pytest.raises(ValidationError):
        TrackedObjectSet(
            frame_id=frame,
            stamp_ns=0,
            revision=0,
            objects=[_track(frame, object_id=f"trk:{i}") for i in range(33)],
        )
    with pytest.raises(ValidationError, match="duplicate"):
        TrackedObjectSet(frame_id=frame, stamp_ns=0, revision=0, objects=[_track(frame)] * 2)
    with pytest.raises(ValidationError, match="not the set"):
        TrackedObjectSet(frame_id="odom", stamp_ns=0, revision=0, objects=[_track(frame)])


def test_deploy_runtime_object_tracking_needs_explicit_wait() -> None:
    assert DeployRuntime().object_tracking == "off"
    with pytest.raises(ValidationError, match="grounding_wait_s"):
        DeployRuntime(object_tracking="shadow")
    _round_trip(
        DeployRuntime(object_tracking="bind", grounding_wait_s=3.0, grounding_required=True)
    )


_frames = st.sampled_from(["base", "panda_link0", "openarm_base"])
_status = st.sampled_from([s for s in TrackStatus if s is not TrackStatus.BOUND])


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    frame=_frames,
    n=st.integers(min_value=0, max_value=TrackedObjectSet.MAX_OBJECTS),
    status=_status,
    fit=st.sampled_from(list(FitKind)),
    epoch=st.integers(min_value=0, max_value=2**31),
    support=st.floats(min_value=-1.0, max_value=1.0),
    lift=st.floats(min_value=0.0, max_value=0.1),
)
def test_fuzz_tracked_object_set(
    frame: str, n: int, status: TrackStatus, fit: FitKind, epoch: int, support: float, lift: float
) -> None:
    objects = [
        _track(
            frame,
            object_id=f"trk:{i}",
            status=status,
            fit_kind=fit,
            epoch=epoch,
            support_z=support,
            floor_z=support + lift,
        )
        for i in range(n)
    ]
    _round_trip(TrackedObjectSet(frame_id=frame, stamp_ns=epoch, revision=epoch, objects=objects))


@settings(max_examples=50, deadline=None)
@given(
    label=st.text(min_size=1, max_size=16),
    count=st.sampled_from(["one", "all"]),
    role=st.sampled_from(list(ObjectRole)),
    hand=st.none() | st.sampled_from([h[1] for h in _HANDS]),
    extra=st.integers(min_value=0, max_value=5),
)
def test_fuzz_object_query(
    label: str, count: str, role: ObjectRole, hand: str | None, extra: int
) -> None:
    query = ObjectQuery.model_validate(
        {
            "label": label,
            "count": count,
            "role": role,
            "hand": hand,
            "min_instances": 1 if count == "one" else 1 + extra,
        }
    )
    _round_trip(query)


def test_idl_round_trip_through_built_messages() -> None:
    msgs = pytest.importorskip("openral_msgs.msg", reason="openral_msgs not built/sourced")
    if not hasattr(msgs, "TrackedObjectSet"):
        pytest.skip("sourced openral_msgs predates issue #418")
    frame = _scene_box().frame_id
    hull = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(0.03, 0.03, 0.03)),
        pose_in_object=Pose6D(xyz=(0.07, -0.3, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id=frame),
    )
    bound = _track(
        frame,
        status=TrackStatus.BOUND,
        tight=(hull,),
        bound_hand="gripper",
        attach_link="jaw",
        pose_in_link=Pose6D(xyz=(0.0, 0.0, 0.02), quat_xyzw=(0, 0, 0, 1), frame_id="jaw"),
        identity_inferred=True,
    )
    tracked = TrackedObjectSet(frame_id=frame, stamp_ns=5, revision=2, objects=[bound])
    msg = msgs.TrackedObjectSet()
    tracked.fill_idl(
        msg, object_factory=msgs.TrackedObject, primitive_factory=msgs.AttachedCollisionPrimitive
    )
    assert TrackedObjectSet.from_idl(msg) == tracked

    query = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, seed=_scene_box())
    grasp = GraspDeclaration(
        target_id="obj:orange_1",
        contact_links=("jaw",),
        timeout_s=120.0,
        stamp_ns=0,
        target_queries=(query,),
    )
    grasp_msg = msgs.GraspDeclaration()
    with pytest.raises(Exception, match="query_factory"):
        grasp.fill_idl(grasp_msg)
    grasp.fill_idl(grasp_msg, query_factory=msgs.ObjectQuery)
    assert GraspDeclaration.from_idl(grasp_msg) == grasp

    place = PlaceDeclaration(
        target_id="obj:plate",
        timeout_s=60.0,
        stamp_ns=0,
        destination_query=ObjectQuery(label="plate", count="one", role=ObjectRole.DESTINATION),
    )
    place_msg = msgs.PlaceDeclaration()
    place.fill_idl(place_msg)
    assert PlaceDeclaration.from_idl(place_msg) == place
