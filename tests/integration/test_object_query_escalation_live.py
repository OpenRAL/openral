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
* ``unresolved`` (WARN) is in the LLM's tick input with the ``find-object`` rung;
* ``exhausted`` (FAIL) takes the human-handoff rung: the question goes out on
  ``/openral/prompt``.

The only double is the LLM (``FakeToolUseClient``): the test stops at the reasoner's typed
input and its ladder decision. Gated on ``OPENRAL_TEST_ROS_LIVE=1``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

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
    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        CameraTopicKind,
        FailureEvidence,
        GraspDeclaration,
        ObjectQuery,
        ObjectQueryEvidence,
        ObjectRole,
        RobotDescription,
        WaitTool,
        camera_topic,
    )
    from openral_core.exceptions import ROSObjectNotInMemory
    from openral_core.geometry import rotation_to_quat_wxyz
    from openral_msgs.msg import FailureTrigger, OccupancyVoxels, PromptStamped
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import ObjectQuery as ObjectQueryMsg
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
    for node in (helper, reasoner, tracker):
        executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        tracker.trigger_configure()
        tracker.trigger_activate()
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
        assert _wait(lambda: bool(prompts), frame, 5.0), "no handoff question was asked"
        # The reasoner's forced tick for `unresolved` ran the real core on the context.
        assert _wait(
            lambda: any("find-object playbook" in t.context_text for t in client.traces),
            frame,
            10.0,
        ), [t.context_text[-400:] for t in client.traces]
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
        assert ev.goal_id == f"grasp:obj:banana@{decl.stamp_ns}"
        assert isinstance(ev.to_exception(), ROSObjectNotInMemory)
    assert by_kind["unresolved"].evidence_json != by_kind["exhausted"].evidence_json
    exhausted = adapter.validate_json(by_kind["exhausted"].evidence_json)
    assert isinstance(exhausted, ObjectQueryEvidence)
    assert exhausted.waited_s >= 1.5, "exhausted before the search budget"
    assert exhausted.attempts >= 1, "the tracker never searched for it"

    # The reasoner's typed input and its ladder decision.
    failures = [f for f in reasoner._renderer.failures if f.source == "world_state"]
    assert len(failures) >= 2
    rungs = [e for e in reasoner._renderer.executions if e.rskill_id == "(object-query)"]
    assert any("find-object playbook" in (e.reflection or "") for e in rungs)
    assert any("exhausted" in (e.reflection or "") for e in rungs)
    assert [p.text for p in prompts] == ["I can't find the banana; where should I look?"]


def test_an_exhausted_query_ends_the_in_flight_goal_with_a_typed_failure(spans: Any) -> None:
    """The reasoner dispatches a real goal to the real runner; its target is never found.

    The runner arms the grasp declaration (scene JSON + ``object_queries_json`` naming a
    ``banana``), the real tracker exhausts the query, and the runner ends the goal along
    its cancel path with ``FAILURE_OBJECT_NOT_FOUND`` / ``ROSObjectNotInMemory`` instead
    of running to the 30 s deadline. The reasoner records that typed result and hands off.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    import sys
    from pathlib import Path

    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        CameraTopicKind,
        ExecuteRskillTool,
        FailureEvidence,
        GraspDeclaration,
        ObjectQueryEvidence,
        WaitTool,
        camera_topic,
    )
    from openral_core.geometry import rotation_to_quat_wxyz
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels
    from openral_reasoner import ToolPalette
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
            Parameter(
                "object_queries_json",
                value='[{"label": "banana", "count": "one", "role": "target"}]',
            ),
            Parameter("grounding_wait_s", value=0.2),
        ]
    )
    client = FakeToolUseClient(responses=[WaitTool() for _ in range(64)])
    reasoner = ReasonerNode(
        client=client, palette=ToolPalette(execute_rskill_ids=frozenset({skill})), tick_hz=0.2
    )
    reasoner.set_parameters(
        [
            Parameter("robot_yaml", value=_ROBOT),
            Parameter("grasp_target_voxel_m", value=float(grid.resolution)),
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
        reasoner._dispatch_execute_rskill(
            ExecuteRskillTool(rskill_id=skill, prompt="pick the banana", patience_s=30.0),
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
    assert record.summary.startswith("ROSObjectNotInMemory: object query exhausted"), record
    assert "FAILURE_OBJECT_NOT_FOUND" in (record.reflection or ""), record
    assert elapsed < 20.0, f"the goal ran {elapsed:.1f} s, toward its 30 s deadline"
    exhausted = [
        t
        for t in triggers
        if isinstance(ev := adapter.validate_json(t.evidence_json), ObjectQueryEvidence)
        and ev.escalation == "exhausted"
    ]
    assert exhausted and exhausted[0].rskill_id == skill and exhausted[0].trace_id, (
        "the escalation is not on the goal's skill trace"
    )
    # Recorded in the same trace: the goal's rskill.execute span and the tracker's
    # world_state.object_query span share a trace id, each with the typed exception.
    finished = spans.get_finished_spans()
    goal_trace = int(exhausted[0].trace_id.split("-")[1], 16)  # W3C traceparent
    in_goal = [s for s in finished if s.context.trace_id == goal_trace]

    def typed(span: Any) -> bool:
        return any(
            str(e.attributes.get("exception.type")).endswith(".ROSObjectNotInMemory")
            for e in span.events
        )

    assert any(s.name == "rskill.execute" and typed(s) for s in in_goal), "not on the goal span"
    assert any(
        s.name == "world_state.object_query"
        and s.attributes.get("world_state.escalation") == "exhausted"
        and typed(s)
        for s in in_goal
    ), "the tracker's escalation span is not in the goal's trace"
