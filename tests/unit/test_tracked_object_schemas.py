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
    JointState,
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
    WorldState,
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

    query = ObjectQuery(
        label="orange", count="all", role=ObjectRole.TARGET, seed=_scene_box(),
        synonyms=("mandarin", "citrus fruit"),
    )  # fmt: skip
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


# --- Every deploy scene, every additive field (issue #418 test matrix A, "Schemas") ---

_DEPLOY_SCENES = sorted((_ROOT / "scenes/deploy").glob("*.yaml"))


def _scene_frame(scene: DeployScene) -> str:
    """The scene robot's manifest base frame, else the scene's own declared frame."""
    if scene.robot_id and (_ROOT / "robots" / scene.robot_id / "robot.yaml").exists():
        return RobotDescription.from_yaml(
            str(_ROOT / "robots" / scene.robot_id / "robot.yaml")
        ).base_frame
    if scene.grasp_declaration is not None and scene.grasp_declaration.search_box is not None:
        return scene.grasp_declaration.search_box.frame_id
    return "base"


@pytest.mark.parametrize("path", _DEPLOY_SCENES, ids=lambda p: p.stem)
@settings(max_examples=6, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    mode=st.sampled_from(["off", "shadow", "bind", "on"]),
    wait=st.floats(min_value=0.01, max_value=60.0),
    required=st.booleans(),
    n_queries=st.integers(min_value=0, max_value=GraspDeclaration.MAX_TARGET_QUERIES),
    destination=st.booleans(),
    epoch=st.integers(min_value=0, max_value=2**31),
    n_tracks=st.integers(min_value=0, max_value=4),
)
def test_fuzz_every_deploy_scene_with_the_additive_fields(
    path: Path,
    mode: str,
    wait: float,
    required: bool,
    n_queries: int,
    destination: bool,
    epoch: int,
    n_tracks: int,
) -> None:
    """Each scene loads as before, then carries every #418 additive field and round-trips."""
    scene = DeployScene.from_yaml(str(path))
    frame = _scene_frame(scene)
    runtime = DeployRuntime.model_validate(
        {
            **(scene.runtime.model_dump() if scene.runtime is not None else {}),
            "object_tracking": mode,
            "grounding_wait_s": wait,
            "grounding_required": required,
        }
    )
    _round_trip(runtime)
    update: dict[str, object] = {"runtime": runtime}
    queries = tuple(
        ObjectQuery(label=f"obj{i}", count="all", role=ObjectRole.TARGET, min_instances=1 + i)
        for i in range(n_queries)
    )
    if scene.grasp_declaration is not None:
        update["grasp_declaration"] = GraspDeclaration.model_validate(
            {**scene.grasp_declaration.model_dump(), "target_queries": queries}
        )
    place = scene.place_declaration or PlaceDeclaration(
        target_id="obj:dest", timeout_s=30.0, stamp_ns=0
    )
    place = PlaceDeclaration.model_validate(
        {
            **place.model_dump(),
            "destination_query": (
                ObjectQuery(label="dest", count="one", role=ObjectRole.DESTINATION)
                if destination
                else None
            ),
        }
    )
    _round_trip(place)
    rebuilt = DeployScene.model_validate({**scene.model_dump(), **update})
    assert DeployScene.model_validate_json(rebuilt.model_dump_json()) == rebuilt
    tracked = TrackedObjectSet(
        frame_id=frame,
        stamp_ns=epoch,
        revision=epoch,
        objects=[_track(frame, object_id=f"trk:{i}", epoch=epoch) for i in range(n_tracks)],
    )
    _round_trip(tracked)
    assert scene.model_dump() == DeployScene.from_yaml(str(path)).model_dump()


@pytest.mark.parametrize(("robot", "joint", "link"), _HANDS)
@settings(max_examples=30, deadline=None)
@given(
    epoch=st.integers(min_value=0, max_value=2**32 - 1),
    revision=st.integers(min_value=0, max_value=2**63 - 1),
    kind=st.sampled_from(list(AttachmentEvidenceKind)),
)
def test_fuzz_attachment_epoch_and_revision_on_real_hands(
    robot: str, joint: str, link: str, epoch: int, revision: int, kind: AttachmentEvidenceKind
) -> None:
    """``epoch``/``geometry_revision`` round-trip (JSON and, when built, the IDL)."""
    _hand(robot, joint, link)
    payload = AttachedCollisionObject(
        object_id="trk:1",
        attach_link=link,
        touch_links=[link],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=(0.02, 0.02, 0.02)),
                pose_in_object=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="trk:1"),
            )
        ],
        pose_in_link=Pose6D(xyz=(0, 0, 0.02), quat_xyzw=(0, 0, 0, 1), frame_id=link),
        confidence=1.0,
        evidence_kind=kind,
        stamp_ns=7,
        epoch=epoch,
        geometry_revision=revision,
    )
    _round_trip(payload)
    msgs = pytest.importorskip("openral_msgs.msg", reason="openral_msgs not built/sourced")
    if not hasattr(msgs.AttachedCollisionObject(), "geometry_revision"):
        pytest.skip("sourced openral_msgs predates issue #418")
    out = msgs.AttachedCollisionObject()
    payload.fill_idl(out, primitive_factory=msgs.AttachedCollisionPrimitive)
    assert (out.epoch, out.geometry_revision) == (epoch, revision)
    assert AttachedCollisionObject.from_idl(out) == payload


def test_world_state_relay_round_trips_with_and_without_a_tracker() -> None:
    """``WorldState.tracked_objects``: absent stays ``None``; present round-trips."""
    frame = _scene_box().frame_id
    joints = JointState(name=["gripper"], position=[0.1], stamp_ns=5)
    bare = WorldState(stamp_ns=5, joint_state=joints)
    assert bare.tracked_objects is None
    _round_trip(bare)
    with_set = bare.model_copy(
        update={
            "tracked_objects": TrackedObjectSet(
                frame_id=frame, stamp_ns=5, revision=1, objects=[_track(frame)]
            )
        }
    )
    _round_trip(WorldState.model_validate(with_set.model_dump()))


def test_a_query_needing_more_instances_than_the_tracker_holds_is_refused() -> None:
    cap = TrackedObjectSet.MAX_OBJECTS
    ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, min_instances=cap)
    with pytest.raises(ValidationError, match="capacity"):
        ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, min_instances=cap + 1)
