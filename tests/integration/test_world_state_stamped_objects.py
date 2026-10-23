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
    GraspDeclaration,
    JointState,
    PlaceDeclaration,
    PlaceRegion,
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
    # The lift's 3D box survives the wire (the reasoner seeds grasp targets from it).
    assert o.bbox_3d == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def test_an_object_without_a_box_round_trips_without_one() -> None:
    unboxed = _obj().model_copy(update={"bbox_3d": None, "label": "plate"})
    back = world_state_from_idl(build_world_state_stamped_msg(None, _ws([_obj(), unboxed])))
    assert [o.bbox_3d for o in back.detected_objects] == [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), None]


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


def test_a_sim_stamped_declaration_survives_the_whole_delivery_seam() -> None:
    # Round-7 clock-domain regression, end to end over the real IDL: AttachmentState stamped
    # in SIMULATOR time (~1.2e9 ns) carrying a sim-stamped declaration -> node callback -> an
    # aggregator on its PRODUCTION clock (wall time.time_ns, as on_configure builds it) ->
    # snapshot -> WorldStateStamped. Kernel only sees the last hop, so a declaration dropped
    # anywhere along it is a region the kernel never receives and an allowance that never arms.
    sim_arm_ns = 1_240_000_000
    declaration = PlaceDeclaration(
        target_id="sim:cab_1_left_group_main",
        object_id="baguette_seed1",
        rskill_id="openral/pi05-robocasa",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=60.0,
        stamp_ns=sim_arm_ns,
        region=PlaceRegion(
            frame_id="base_link",
            pose=Pose6D(
                xyz=(0.62, 0.0, 1.05),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            half_extents=(0.18, 0.25, 0.45),
            evidence_ref="mujoco_body_subtree:cab_1_left_group_main",
            stamp_ns=sim_arm_ns,
        ),
        # The reasoner's optional hint naming the surface: relayed untouched.
        search_box=PlaceRegion(
            frame_id="base_link",
            pose=Pose6D(xyz=(0.62, 0.0, 1.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="base_link"),
            half_extents=(0.3, 0.3, 0.2),
            evidence_ref="reasoner_seed:detection 'cabinet'",
        ),
    )
    attachments = [_attached("baguette_seed1", BoxShape(half_extents_m=(0.12, 0.025, 0.025)))]
    state_msg = AttachmentState()
    state_msg.header.stamp.sec = sim_arm_ns // 1_000_000_000
    state_msg.header.stamp.nanosec = sim_arm_ns % 1_000_000_000
    state_msg.revision = 3
    state_msg.objects = list(
        build_world_state_stamped_msg(None, _ws([], attachments)).attached_objects
    )
    state_msg.place_declaration_valid = True
    declaration.fill_idl(state_msg.place_declaration)
    aggregator = WorldStateAggregator(RobotDescription.from_yaml("robots/panda_mobile/robot.yaml"))

    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    try:
        node._on_attachment_state(state_msg)
        snapshot = aggregator.snapshot()
        assert snapshot.attachment_stamp_ns == sim_arm_ns
        # Wall vs sim: the aggregator's own clock is ~9 orders of magnitude
        # ahead of the stream it is aggregating, and the declaration lives
        # anyway because liveness is judged in the stream's clock.
        assert snapshot.stamp_ns > sim_arm_ns * 1_000_000
        assert snapshot.place_declaration is not None

        msg = build_world_state_stamped_msg(node, snapshot)
        assert msg.place_declaration_valid is True
        assert msg.place_declaration.target_id == "sim:cab_1_left_group_main"
        assert msg.place_declaration.stamp_ns == sim_arm_ns
        assert msg.place_declaration.region_valid is True
        assert msg.place_declaration.region.frame_id == "base_link"
        assert msg.place_declaration.region.half_extents.x == pytest.approx(0.18)
        assert msg.place_declaration.region.half_extents.z == pytest.approx(0.45)
        assert msg.place_declaration.search_box_valid is True
        assert msg.place_declaration.search_box.half_extents.x == pytest.approx(0.3)

        # And the backstop still bites, driven by the stream's clock: a
        # heartbeat 61 s of simulator time later carries the same declaration
        # past its 60 s timeout, and World State drops it with no retraction.
        state_msg.revision = 4
        state_msg.header.stamp.sec = (sim_arm_ns + 61_000_000_000) // 1_000_000_000
        node._on_attachment_state(state_msg)
        assert aggregator.snapshot().place_declaration is None
        expired = build_world_state_stamped_msg(node, aggregator.snapshot())
        assert expired.place_declaration_valid is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_a_grasp_declaration_rides_attachment_state_to_world_state_stamped() -> None:
    # Sibling of the place-declaration seam above, over the real IDL: a sim-stamped
    # AttachmentState carrying a producer-measured grasp region -> node callback ->
    # production-clock aggregator -> WorldStateStamped, then expired by the stream clock.
    sim_arm_ns = 1_240_000_000
    declaration = GraspDeclaration(
        target_id="cell:pick_box",
        contact_links=("openarm_left_finger_pair", "openarm_right_finger_pair"),
        rskill_id="openral/pi05-openarm-pick-place",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=70.0,
        stamp_ns=sim_arm_ns,
        region=PlaceRegion(
            frame_id="openarm_base",
            pose=Pose6D(
                xyz=(0.45, 0.0, 0.12), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="openarm_base"
            ),
            half_extents=(0.06, 0.05, 0.04),
            evidence_ref="zed_sam2_obb:pick_box",
            stamp_ns=sim_arm_ns,
        ),
    )
    state_msg = AttachmentState()
    state_msg.header.stamp.sec = sim_arm_ns // 1_000_000_000
    state_msg.header.stamp.nanosec = sim_arm_ns % 1_000_000_000
    state_msg.revision = 1
    state_msg.grasp_declaration_valid = True
    declaration.fill_idl(state_msg.grasp_declaration)
    aggregator = WorldStateAggregator(RobotDescription.from_yaml("robots/openarm/robot.yaml"))

    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    try:
        node._on_attachment_state(state_msg)
        snapshot = aggregator.snapshot()
        assert snapshot.grasp_declaration == declaration

        msg = build_world_state_stamped_msg(node, snapshot)
        assert msg.grasp_declaration_valid is True
        assert msg.place_declaration_valid is False
        assert list(msg.grasp_declaration.contact_links) == list(declaration.contact_links)
        assert msg.grasp_declaration.region_valid is True
        assert msg.grasp_declaration.region.frame_id == "openarm_base"
        assert msg.grasp_declaration.region.half_extents.x == pytest.approx(0.06)
        # And the in-process decode of the relayed message is the same declaration.
        assert world_state_from_idl(msg).grasp_declaration == declaration

        # Heartbeat 71 s later on the stream clock: past the 70 s backstop, dropped.
        state_msg.revision = 2
        state_msg.header.stamp.sec = (sim_arm_ns + 71_000_000_000) // 1_000_000_000
        node._on_attachment_state(state_msg)
        assert aggregator.snapshot().grasp_declaration is None
        assert (
            build_world_state_stamped_msg(node, aggregator.snapshot()).grasp_declaration_valid
            is False
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_a_grasp_declaration_with_target_queries_relays_them() -> None:
    # Issue #418: dispatch names classes, not boxes. Every relay must encode them; one
    # without a query factory raises rather than drop what the prompt named.
    from openral_core import ObjectQuery, ObjectRole
    from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

    declaration = GraspDeclaration(
        target_id="orange",
        contact_links=("gripper",),
        timeout_s=60.0,
        stamp_ns=5,
        target_queries=(ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),),
    )
    state_msg = AttachmentState()
    state_msg.header.stamp.nanosec = 5
    state_msg.revision = 1
    state_msg.grasp_declaration_valid = True
    declaration.fill_idl(state_msg.grasp_declaration, query_factory=ObjectQueryMsg)
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    )
    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    try:
        node._on_attachment_state(state_msg)
        msg = build_world_state_stamped_msg(node, aggregator.snapshot())
        assert world_state_from_idl(msg).grasp_declaration == declaration
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_tracked_objects_relay_into_world_state_fast_and_go_invalid_when_stale() -> None:
    # Issue #418: /openral/tracked_objects -> node callback -> aggregator -> the fast
    # stream's tracked_objects (valid), and invalid once the tracker stops heartbeating.
    import numpy as np
    from openral_core import TrackedObjectSet
    from openral_msgs.msg import AttachedCollisionPrimitive as RosPrimitive
    from openral_msgs.msg import TrackedObject as RosTrackedObject
    from openral_msgs.msg import TrackedObjectSet as RosTrackedObjectSet

    from tests.unit._object_tracker_fixtures import track

    now = [10_000_000_000]
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml("robots/so101_follower/robot.yaml"), clock_fn=lambda: now[0]
    )
    tracked = TrackedObjectSet(
        frame_id="base",
        stamp_ns=7,
        revision=1_760_000_000_000_000_042,
        objects=[track("trk:1", "base", np.array([0.2, 0.0, 0.03]), (0.03, 0.03, 0.03))],
    )
    wire = RosTrackedObjectSet()
    tracked.fill_idl(wire, object_factory=RosTrackedObject, primitive_factory=RosPrimitive)
    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    try:
        assert (
            build_world_state_stamped_msg(node, aggregator.snapshot()).tracked_objects_valid
            is False
        )
        node._on_tracked_objects(wire)
        msg = build_world_state_stamped_msg(node, aggregator.snapshot())
        assert msg.tracked_objects_valid is True
        assert msg.tracked_objects.revision == tracked.revision
        assert world_state_from_idl(msg).tracked_objects == tracked
        now[0] += 1_400_000_000  # inside the 1.5 s window
        assert build_world_state_stamped_msg(node, aggregator.snapshot()).tracked_objects_valid
        now[0] += 200_000_000  # past it: the kernel sees "no tracker"
        stale = build_world_state_stamped_msg(node, aggregator.snapshot())
        assert stale.tracked_objects_valid is False and not stale.tracked_objects.objects
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_a_configured_node_ages_the_tracked_set_on_sim_time_not_wall_time() -> None:
    # Isaac R23/R26: the tracker heartbeats on ROS time, but the aggregator aged its set on
    # wall time, so on a sim below real time the set went stale between live beats and every
    # tracked object lost its kernel explanation mid-contact. The configured node (the
    # injected aggregator, as compose_so100_runtime and deploy build it) must age it on its
    # own ROS clock: wall time passing with /clock still keeps the set; /clock passing drops it.
    import time

    import numpy as np
    from builtin_interfaces.msg import Time
    from openral_core import TrackedObjectSet
    from openral_msgs.msg import AttachedCollisionPrimitive as RosPrimitive
    from openral_msgs.msg import TrackedObject as RosTrackedObject
    from openral_msgs.msg import TrackedObjectSet as RosTrackedObjectSet
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.lifecycle import TransitionCallbackReturn
    from rosgraph_msgs.msg import Clock

    from tests.unit._object_tracker_fixtures import track

    tracked = TrackedObjectSet(
        frame_id="base",
        stamp_ns=7,
        revision=1,
        objects=[track("trk:1", "base", np.array([0.2, 0.0, 0.03]), (0.03, 0.03, 0.03))],
    )
    wire = RosTrackedObjectSet()
    tracked.fill_idl(wire, object_factory=RosTrackedObject, primitive_factory=RosPrimitive)
    aggregator = WorldStateAggregator(
        RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    )
    rclpy.init()
    node = _WorldStateLifecycleNode(aggregator=aggregator)
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    helper = rclpy.create_node("test_tracked_sim_clock_helper")
    clock_pub = helper.create_publisher(Clock, "/clock", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)

    def publish_clock(sim_ns: int) -> None:
        clock_pub.publish(Clock(clock=Time(sec=sim_ns // 10**9, nanosec=sim_ns % 10**9)))
        deadline = time.monotonic() + 2.0
        while node.get_clock().now().nanoseconds != sim_ns and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert node.get_clock().now().nanoseconds == sim_ns

    def valid() -> bool:
        return bool(
            build_world_state_stamped_msg(node, aggregator.snapshot()).tracked_objects_valid
        )

    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        publish_clock(40_000_000_000)
        node._on_tracked_objects(wire)
        assert valid()
        time.sleep(1.6)  # wall time past the 1.5 s window, sim time standing still
        assert valid()
        publish_clock(41_600_000_000)  # sim time past it: the tracker really stopped
        assert not valid()
    finally:
        executor.remove_node(helper)
        helper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def test_support_contact_witness_round_trips_over_the_idl() -> None:
    # Wire half of ADR-0092 D6: an attachment attesting support contact must arrive at the
    # safety kernel with the attestation intact; one that doesn't must arrive with
    # support_contact_valid False — a lost witness would silently re-open the false stop, an
    # invented one would silently license real penetration.
    # 2026-08-14 acceptance run's attested baguette, verbatim — including the evidence kind
    # the MuJoCo producer actually emits (signed distance probe, not the solver's contact list).
    witness = SupportContactWitness(
        support_id="robocasa:counter_main",
        contact_point_in_object=(0.0, 0.0, -0.025),
        contact_normal_in_object=(0.0, 0.0, 1.0),
        patch_radius_m=0.1476,
        max_penetration_m=0.00118,
        confidence=1.0,
        evidence_kind=AttachmentEvidenceKind.SIM_GEOM_DISTANCE,
        evidence_ref="openral_hal.convex_distance.convex_geom_distance:counter_main",
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
