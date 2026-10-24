# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: an object query the tracker cannot resolve reaches the reasoner's ladder.

Issue #418 Addendum G, end to end: real ``object_tracker_node`` (shadow) → real
``FailureTrigger`` on ``/openral/failure/world_state`` → real ``ReasonerNode`` (real
``ReasonerCore`` behind it). The grasp declaration names a ``banana`` the SO-101
PickOrange ``top`` camera never sees (the ray-cast scene holds one orange), with a short
search budget (``search_start_s`` 0.3 s, ``search_max_s`` 1.5 s, source time).

Asserts, for both escalations:

* the trigger is typed — ``KIND_OBJECT_QUERY``, ``ObjectQueryEvidence`` (label, TARGET
  role, waited source seconds, detection attempts), whose typed failure is
  ``ROSObjectNotInMemory`` — and carries the declaration's ``rskill_id`` and its
  ``traceparent`` (the skill-execution trace it is recorded under);
* both escalations carry typed tracker evidence;
* object-query failures do not publish a human handoff prompt.

The LLM (``FakeToolUseClient``) and a service peer at the detector process boundary are the
doubles. The peer returns the real bounded ``LocateInView.failure_reason`` contract for the
missing-``timm`` failure. The test stops at the reasoner's typed input and ladder decision.
Gated on ``OPENRAL_TEST_ROS_LIVE=1``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENRAL_TEST_ROS_LIVE"),
    reason="live rclpy nodes + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1.",
)

_ROBOT = "robots/so101_follower/robot.yaml"
_OPTICAL = "top_depth_optical"
_SKILL = "openral/test-object-query"


@pytest.fixture
def spans() -> Any:
    """A real OTel SDK provider recording to memory (the exporter is the only double)."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # test-only reset
    trace._TRACER_PROVIDER = None  # test-only reset
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def _wait(predicate: Any, tick: Any, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        tick()
        if predicate():
            return True
        time.sleep(0.15)
    return False


def test_unresolved_then_exhausted_queries_reach_the_reasoner_ladder(spans: Any) -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    pytest.importorskip("pxr.Usd")
    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        CameraTopicKind,
        DeployScene,
        FailureEvidence,
        GraspDeclaration,
        ObjectQuery,
        ObjectQueryEvidence,
        ObjectRole,
        PhysicsBackend,
        RobotDescription,
        SceneSpec,
        USDObjectCatalogRef,
        WaitTool,
        camera_topic,
        load_usd_object_catalog,
    )
    from openral_core.exceptions import ROSObjectNotInMemory
    from openral_core.geometry import rotation_to_quat_wxyz
    from openral_msgs.msg import FailureTrigger, OccupancyVoxels, PromptStamped
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import ObjectQuery as ObjectQueryMsg
    from openral_msgs.srv import LocateInView
    from openral_observability import propagation, rskill_span
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state_ros.object_tracker_node import make_object_tracker_node
    from pydantic import TypeAdapter
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster

    from tests.integration.fakes.fake_llm import FakeToolUseClient
    from tests.unit._tracker_scenes import grid_from_depth, render, so101_top, sphere

    adapter: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)

    rig = so101_top()
    depth, _labels = render([sphere((rig.centre_xy[0], rig.centre_xy[1], 0.024), 0.024)], rig)
    grid = grid_from_depth(depth, rig)
    k = rig.k
    with rskill_span("rskill.execute", rskill_id=_SKILL):
        traceparent = propagation.current_traceparent() or ""
    assert traceparent, "no recording tracer: the trace assertions would be vacuous"

    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    reliable = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1
    )
    sensor = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    helper = rclpy.create_node("object_query_live_helper")
    triggers: list[Any] = []
    prompts: list[Any] = []
    locate_requests: list[str] = []
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/world_state",
        triggers.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        ),
    )
    helper.create_subscription(PromptStamped, "/openral/prompt", prompts.append, 10)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "base", _OPTICAL
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
        float(v) for v in rig.t[:3, 3]
    )
    w, x, y, z = rotation_to_quat_wxyz(rig.t[:3, :3])
    tf.transform.rotation.x, tf.transform.rotation.y = x, y
    tf.transform.rotation.z, tf.transform.rotation.w = z, w
    StaticTransformBroadcaster(helper).sendTransform(tf)
    pub_depth = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.DEPTH_IMAGE), sensor
    )
    pub_info = helper.create_publisher(
        CameraInfo, camera_topic("top", CameraTopicKind.DEPTH_CAMERA_INFO), sensor
    )
    pub_mask = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.ROBOT_MASK), sensor
    )
    pub_grid = helper.create_publisher(OccupancyVoxels, "/openral/world_voxels", reliable)
    pub_decl = helper.create_publisher(GraspDeclarationMsg, "/openral/grasp_declaration", latched)

    def locate_failure(req: Any, resp: Any) -> Any:
        locate_requests.append(req.query)
        resp.found = False
        resp.camera = req.camera
        resp.detector = req.detector
        resp.source_stamp_ns = req.source_stamp_ns
        resp.metadata_json = ""
        resp.failure_reason = "detector_exception:ModuleNotFoundError:No module named 'timm'"
        return resp

    helper.create_service(LocateInView, "/openral/perception/locate_in_view", locate_failure)

    def frame() -> None:
        now = helper.get_clock().now().to_msg()
        info = CameraInfo(width=k.width, height=k.height)
        info.header.frame_id = _OPTICAL
        info.k = [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0]
        pub_info.publish(info)
        img = Image(height=k.height, width=k.width, encoding="32FC1", step=4 * k.width)
        img.header.stamp, img.header.frame_id = now, _OPTICAL
        img.data = depth.astype("<f4").tobytes()
        mask = Image(height=k.height, width=k.width, encoding="mono8", step=k.width)
        mask.header.stamp, mask.header.frame_id = now, _OPTICAL
        mask.data = bytes(k.width * k.height)
        g = OccupancyVoxels()
        g.header.frame_id, g.header.stamp, g.source_stamp = grid.frame_id, now, now
        g.origin.x, g.origin.y, g.origin.z = grid.origin
        g.orientation.w = 1.0
        g.resolution = grid.resolution
        g.size_x, g.size_y, g.size_z = grid.size
        g.occupancy = grid.occupancy.tolist()
        pub_grid.publish(g)
        pub_mask.publish(mask)
        pub_depth.publish(img)

    client = FakeToolUseClient(responses=[WaitTool() for _ in range(64)])
    reasoner = ReasonerNode(
        client=client,
        palette=ToolPalette(execute_rskill_ids=frozenset({_SKILL})),
        tick_hz=0.2,
    )
    reasoner.set_parameters(
        [
            Parameter("robot_yaml", value=_ROBOT),
            Parameter("grasp_target_voxel_m", value=float(grid.resolution)),
        ]
    )
    tracker = make_object_tracker_node()
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "docs/tutorials/deploy/usd-object-catalog-example.usda"
    )
    catalog = load_usd_object_catalog(
        DeployScene(
            scene=SceneSpec(id="object-query-live-catalog", backend=PhysicsBackend.ISAACSIM),
            object_catalog=USDObjectCatalogRef(
                usd=str(catalog_path), prim_path="/Cell/Catalog", expected_revision="tutorial-v1"
            ),
        ),
        scene_path=catalog_path,
    )
    assert catalog is not None
    tracker.set_parameters(
        [
            Parameter("robot_yaml", value=_ROBOT),
            Parameter("mode", value="shadow"),
            Parameter("cameras", value=["top"]),
            Parameter("grounding_wait_s", value=5.0),
            Parameter("cell_m", value=float(grid.resolution)),
            Parameter("search_start_s", value=0.3),
            Parameter("search_max_s", value=1.5),
            Parameter("locate_service", value="/openral/perception/locate_in_view"),
            Parameter("declared_object_catalog_json", value=catalog.model_dump_json()),
        ]
    )
    for node in (helper, reasoner, tracker):
        executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        tracker.trigger_configure()
        tracker.trigger_activate()
        assert tracker.declared_object_catalog is not None
        assert tracker.declared_object_catalog.source_revision == "tutorial-v1"
        decl = GraspDeclaration(
            target_id="obj:banana",
            contact_links=RobotDescription.from_yaml(_ROBOT).grasp_contact_links(),
            timeout_s=120.0,
            stamp_ns=helper.get_clock().now().nanoseconds,
            rskill_id=_SKILL,
            trace_id=traceparent,
            target_queries=(ObjectQuery(label="banana", count="one", role=ObjectRole.TARGET),),
        )
        msg = GraspDeclarationMsg()
        decl.fill_idl(msg, query_factory=ObjectQueryMsg)
        pub_decl.publish(msg)

        def kinds() -> list[str]:
            out: list[str] = []
            for t in triggers:
                ev = adapter.validate_json(t.evidence_json)
                assert isinstance(ev, ObjectQueryEvidence)
                out.append(ev.escalation)
            return out

        assert _wait(lambda: "exhausted" in kinds(), frame), kinds()
        time.sleep(0.5)
        assert not prompts, [p.text for p in prompts]
    finally:
        executor.shutdown()
        tracker.destroy_node()
        reasoner.destroy_node()
        helper.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=5.0)

    by_kind: dict[str, Any] = {}
    for t in triggers:
        decoded = adapter.validate_json(t.evidence_json)
        assert isinstance(decoded, ObjectQueryEvidence)
        by_kind[decoded.escalation] = t
    assert set(by_kind) == {"unresolved", "exhausted"}
    assert locate_requests and all(query == "banana" for query in locate_requests)
    for kind, severity in (
        ("unresolved", FailureTrigger.SEVERITY_WARN),
        ("exhausted", FailureTrigger.SEVERITY_FAIL),
    ):
        t = by_kind[kind]
        assert t.kind == FailureTrigger.KIND_OBJECT_QUERY and t.severity == severity
        assert t.rskill_id == _SKILL and t.trace_id == traceparent, "not on the skill's trace"
        ev = adapter.validate_json(t.evidence_json)
        assert isinstance(ev, ObjectQueryEvidence)
        assert ev.label == "banana" and ev.role is ObjectRole.TARGET
        assert ev.reason == "detector_exception:ModuleNotFoundError:No module named 'timm'"
        assert ev.goal_id == f"grasp:obj:banana@{decl.stamp_ns}"
        assert isinstance(ev.to_exception(), ROSObjectNotInMemory)
    assert by_kind["unresolved"].evidence_json != by_kind["exhausted"].evidence_json
    exhausted = adapter.validate_json(by_kind["exhausted"].evidence_json)
    assert isinstance(exhausted, ObjectQueryEvidence)
    assert exhausted.waited_s >= 1.5, "exhausted before the search budget"
    assert exhausted.attempts >= 1, "the tracker never searched for it"

    # The reasoner's failure buffer records tracker evidence without a prompt.
    failures = [f for f in reasoner._renderer.failures if f.source == "world_state"]
    assert len(failures) >= 2
    assert not [e for e in reasoner._renderer.executions if e.rskill_id == "(object-query)"]
    assert not prompts


def test_unresolved_grounded_query_cancels_and_blocks_at_budget_without_handoff(spans: Any) -> None:
    """The reasoner dispatches a real goal to the real runner; its target is never found.

    The reasoner grounds a ``banana`` target into the action goal, the runner restamps and
    arms that declaration, the real tracker exhausts its query, and the runner ends the goal along
    its cancel path with ``FAILURE_OBJECT_NOT_FOUND`` / ``ROSObjectNotInMemory`` instead
    of running to the 30 s deadline. The reasoner records the typed result without handoff.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    import sys
    from pathlib import Path

    from geometry_msgs.msg import Point, TransformStamped
    from openral_core import (
        CameraTopicKind,
        ExecuteRskillTool,
        FailureEvidence,
        GraspDeclaration,
        GraspTargetRef,
        ObjectQueryEvidence,
        WaitTool,
        camera_topic,
    )
    from openral_core.geometry import rotation_to_quat_wxyz
    from openral_msgs.msg import (
        ActionChunk,
        FailureTrigger,
        OccupancyVoxels,
        PromptStamped,
        WorldStateStamped,
    )
    from openral_msgs.msg import (
        GraspDeclaration as GraspDeclarationMsg,
    )
    from openral_reasoner import MissionState, ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_world_state_ros.object_tracker_node import make_object_tracker_node
    from pydantic import TypeAdapter
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import UInt64
    from tf2_ros import StaticTransformBroadcaster

    from tests.integration.fakes.fake_llm import FakeToolUseClient
    from tests.unit._tracker_scenes import grid_from_depth, render, so101_top, sphere

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "packages/openral_rskill_ros/test")
    )
    from test_place_declaration_lifecycle import (  # type: ignore[import-not-found]  # reason: sibling test module put on sys.path above
        _constant_skill_resolver,
    )

    skill = "openral/test-place-declaration-skill"
    adapter: TypeAdapter[FailureEvidence] = TypeAdapter(FailureEvidence)
    rig = so101_top()
    depth, _labels = render([sphere((rig.centre_xy[0], rig.centre_xy[1], 0.024), 0.024)], rig)
    grid = grid_from_depth(depth, rig)
    k = rig.k
    reliable = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1
    )
    sensor = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=6)
    helper = rclpy.create_node("object_query_goal_helper")
    triggers: list[Any] = []
    prompts: list[Any] = []
    armed_declarations: list[Any] = []
    helper.create_subscription(
        FailureTrigger,
        "/openral/failure/world_state",
        triggers.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=50,
        ),
    )
    helper.create_subscription(PromptStamped, "/openral/prompt", prompts.append, 10)
    helper.create_subscription(
        GraspDeclarationMsg,
        "/openral/grasp_declaration",
        armed_declarations.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )
    chunk_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    applied = helper.create_publisher(UInt64, "/openral/action_applied", chunk_qos)
    helper.create_subscription(
        ActionChunk,
        "/openral/candidate_action",
        lambda m: applied.publish(UInt64(data=int(m.tick_index))),
        chunk_qos,
    )
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "base", _OPTICAL
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
        float(v) for v in rig.t[:3, 3]
    )
    w, x, y, z = rotation_to_quat_wxyz(rig.t[:3, :3])
    tf.transform.rotation.x, tf.transform.rotation.y = x, y
    tf.transform.rotation.z, tf.transform.rotation.w = z, w
    StaticTransformBroadcaster(helper).sendTransform(tf)
    pub_depth = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.DEPTH_IMAGE), sensor
    )
    pub_info = helper.create_publisher(
        CameraInfo, camera_topic("top", CameraTopicKind.DEPTH_CAMERA_INFO), sensor
    )
    pub_mask = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.ROBOT_MASK), sensor
    )
    pub_grid = helper.create_publisher(OccupancyVoxels, "/openral/world_voxels", reliable)

    def frame() -> None:
        now = helper.get_clock().now().to_msg()
        info = CameraInfo(width=k.width, height=k.height)
        info.header.frame_id = _OPTICAL
        info.k = [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0]
        pub_info.publish(info)
        img = Image(height=k.height, width=k.width, encoding="32FC1", step=4 * k.width)
        img.header.stamp, img.header.frame_id = now, _OPTICAL
        img.data = depth.astype("<f4").tobytes()
        mask = Image(height=k.height, width=k.width, encoding="mono8", step=k.width)
        mask.header.stamp, mask.header.frame_id = now, _OPTICAL
        mask.data = bytes(k.width * k.height)
        g = OccupancyVoxels()
        g.header.frame_id, g.header.stamp, g.source_stamp = grid.frame_id, now, now
        g.origin.x, g.origin.y, g.origin.z = grid.origin
        g.orientation.w = 1.0
        g.resolution = grid.resolution
        g.size_x, g.size_y, g.size_z = grid.size
        g.occupancy = grid.occupancy.tolist()
        pub_grid.publish(g)
        pub_mask.publish(mask)
        pub_depth.publish(img)

    runtime = compose_so100_runtime(skill_resolver=_constant_skill_resolver())
    scene_decl = GraspDeclaration(
        target_id="obj:banana", contact_links=("gripper_finger",), timeout_s=70.0, stamp_ns=0
    )
    runtime.skill_runner_node.set_parameters(
        [
            Parameter("grasp_declaration_json", value=scene_decl.model_dump_json()),
            Parameter("joint_state_staleness_limit_s", value=0.5),
            Parameter("object_tracking", value="shadow"),
            Parameter("grounding_wait_s", value=0.2),
        ]
    )
    client = FakeToolUseClient(responses=[WaitTool() for _ in range(64)])
    reasoner = ReasonerNode(
        client=client, palette=ToolPalette(execute_rskill_ids=frozenset({skill})), tick_hz=0.2
    )
    mission = MissionState.from_prompt("pick the banana")
    active = mission.active()
    assert active is not None
    active.attempts = 2  # this dispatch spends the persisted three-attempt object-query budget
    reasoner._renderer.set_mission(mission)
    reasoner.set_parameters(
        [
            Parameter("robot_yaml", value=_ROBOT),
            Parameter("grasp_target_voxel_m", value=float(grid.resolution)),
            Parameter("object_tracking", value="shadow"),
        ]
    )
    tracker = make_object_tracker_node()
    tracker.set_parameters(
        [
            Parameter("robot_yaml", value=_ROBOT),
            Parameter("mode", value="shadow"),
            Parameter("cameras", value=["top"]),
            Parameter("grounding_wait_s", value=5.0),
            Parameter("cell_m", value=float(grid.resolution)),
            Parameter("search_start_s", value=0.3),
            Parameter("search_max_s", value=1.5),
        ]
    )
    nodes = (helper, runtime.world_state_node, runtime.skill_runner_node, reasoner, tracker)
    for node in nodes:
        executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        for node in nodes[1:]:
            node.trigger_configure()
            node.trigger_activate()
        t0 = time.monotonic()
        # A real typed world-state observation grounds the reasoner's target reference.
        # The runner must preserve the action UUID when it restamps this goal-supplied
        # declaration, so the query failure can be joined to the dispatch exactly.
        world = WorldStateStamped()
        world.detected_object_labels = ["banana"]
        world.detected_object_confidences = [0.9]
        world.detected_object_positions = [Point(x=0.35, y=0.0, z=0.1)]
        world.detected_object_track_ids = [17]
        world.detected_object_frame = "base"
        world.detected_object_bbox_valid = [True]
        world.detected_object_bbox_min = [Point(x=0.30, y=-0.05, z=0.05)]
        world.detected_object_bbox_max = [Point(x=0.40, y=0.05, z=0.15)]
        reasoner._world_state_msg = world
        reasoner._dispatch_execute_rskill(
            ExecuteRskillTool(
                rskill_id=skill,
                prompt="pick the banana",
                patience_s=30.0,
                grasp_target=GraspTargetRef(label="banana"),
            ),
            traceparent=None,
        )

        def ended() -> bool:
            return any(e.rskill_id == skill for e in reasoner._renderer.executions)

        assert _wait(ended, frame, 25.0), [e.summary for e in reasoner._renderer.executions]
        elapsed = time.monotonic() - t0
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=5.0)

    [record] = [e for e in reasoner._renderer.executions if e.rskill_id == skill]
    assert record.outcome == "failed"
    assert "object query failed" in (record.reflection or ""), record
    assert not prompts
    active = reasoner._renderer.mission.active()
    assert active is not None and active.attempts == 3
    assert reasoner._object_query_blocked_task_id == active.task_id
    assert elapsed < 20.0, f"the goal ran {elapsed:.1f} s, toward its 30 s deadline"
    object_query_records = [
        e for e in reasoner._renderer.executions if e.rskill_id == "(object-query)"
    ]
    assert object_query_records
    assert "Recovery dispatch budget is 3/3" in (object_query_records[-1].reflection or "")
    query_failures = [
        t
        for t in triggers
        if isinstance(adapter.validate_json(t.evidence_json), ObjectQueryEvidence)
    ]
    assert query_failures and query_failures[0].rskill_id == skill and query_failures[0].trace_id, (
        "the query failure is not on the goal's skill trace"
    )
    active_declarations = [m for m in armed_declarations if m.active]
    assert active_declarations, "the runner did not arm the goal-supplied grasp declaration"
    armed = active_declarations[0]
    assert armed.search_box_valid and armed.target_queries, (
        "the runner used the scene fallback instead of the reasoner's grounded goal declaration"
    )
    invocation_id = armed.invocation_id
    assert len(invocation_id) == 32
    assert all(
        cast(ObjectQueryEvidence, adapter.validate_json(t.evidence_json)).invocation_id
        == invocation_id
        for t in query_failures
    ), "object-query evidence lost the action goal UUID during declaration restamping"
    # The tracker records the typed refusal on its span. The reasoner then cancels the
    # action at the unresolved rung, so the runner span is attributed to the same trace
    # but correctly records cancellation rather than inventing a thrown failure.
    finished = spans.get_finished_spans()
    goal_trace = int(query_failures[0].trace_id.split("-")[1], 16)  # W3C traceparent
    in_goal = [s for s in finished if s.context.trace_id == goal_trace]

    def typed(span: Any) -> bool:
        return any(
            str(e.attributes.get("exception.type")).endswith(".ROSObjectNotInMemory")
            for e in span.events
        )

    goal_spans = [s for s in in_goal if s.name == "rskill.execute"]
    assert goal_spans, "the canceled runner goal is not attributed to the query trace"
    assert not typed(goal_spans[-1]), "a reasoner-requested cancel was misreported as thrown"
    assert any(
        s.name == "world_state.object_query"
        and s.attributes.get("world_state.escalation") in {"unresolved", "exhausted"}
        and typed(s)
        for s in in_goal
    ), "the tracker's escalation span is not in the goal's trace"
