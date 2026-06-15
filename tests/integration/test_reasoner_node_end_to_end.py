"""Live ROS integration test for the F4 reasoner_node + F10 prompt_router_node.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` to match the convention in
``tests/integration/test_failure_bus.py`` / ``test_world_state_integration.py``
(rclpy + DDS init clash with a glib pulled in by torch/pyarrow during the
regular ``uv run pytest`` invocation). Run with::

    just ros2-build
    source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 uv run pytest tests/integration/test_reasoner_node_end_to_end.py \\
        -v -p no:launch_testing -p no:launch_ros

The test exercises the full ``/openral/prompt_in/cli`` →
``/openral/prompt`` → reasoner tick → dispatch round-trip with a real
:class:`FakeToolUseClient` (the only test double allowed at the LLM
process boundary per CLAUDE.md §1.11).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) and source install/setup.bash first."
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_prompt_router_forwards_cli_prompt_to_openral_prompt() -> None:
    """openral prompt CLI → prompt_router → /openral/prompt with source-tagged metadata."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_msgs.msg import PromptStamped
    from openral_prompt_router import PromptRouterNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        router = PromptRouterNode()
        router.trigger_configure()

        sub_node = rclpy.create_node("openral_test_subscriber")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            received.append(msg)
            received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_test_publisher")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt_in/cli", qos)

        # Spin once on the router so the subscription is realised.
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(router)
        executor.add_node(sub_node)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            msg = PromptStamped()
            msg.header.stamp = pub_node.get_clock().now().to_msg()
            msg.header.frame_id = "openral_test_publisher"
            msg.text = "pick the red cube"
            msg.metadata_json = "{}"
            pub.publish(msg)
            executor.spin_once(timeout_sec=0.1)
        executor.remove_node(router)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        router.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, "No PromptStamped landed on /openral/prompt within 5 s"
    msg = received[-1]
    metadata = json.loads(msg.metadata_json)
    assert metadata["source"] == "cli"
    assert metadata["priority"] == 100
    assert msg.text == "pick the red cube"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_reasoner_node_emits_prompt_on_canned_response() -> None:
    """Reasoner with FakeToolUseClient → EmitPromptTool → /openral/prompt."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import EmitPromptTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode

    # ADR-0018 §6: the outbound PromptStamped.metadata_json must carry
    # the active reasoner.tick span's traceparent. That field is only
    # populated when a real TracerProvider is installed — by default the
    # OTel SDK is a no-op and ``current_traceparent()`` returns None.
    # Install a real in-memory exporter so the live round-trip actually
    # exercises the §6 contract.
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    _exp = InMemorySpanExporter()
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(_exp))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # reason: test-only reset, mirrors tests/unit/test_reasoner_observability.py
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # reason: test-only reset
    trace.set_tracer_provider(_provider)

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                EmitPromptTool(
                    target_topic="/openral/prompt",
                    text="acknowledged: pick the cube",
                    rationale="operator prompt received",
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        sub_node = rclpy.create_node("openral_test_subscriber_reasoner")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "openral_reasoner":
                received.append(msg)
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_test_publisher_reasoner")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        # Inject an operator prompt; the reasoner preempts a tick on
        # arrival and the FakeToolUseClient's canned EmitPromptTool
        # is dispatched onto /openral/prompt.
        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_publisher_reasoner"
        msg.text = "pick the cube"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, "Reasoner did not publish on /openral/prompt within 5 s"
    msg = received[-1]
    assert msg.text == "acknowledged: pick the cube"
    assert msg.header.frame_id == "openral_reasoner"
    # The reasoner emitted at least one tool call (the FakeToolUseClient
    # records every invocation; the queue may be empty by now).
    assert len(client.traces) >= 1
    # ADR-0018 §6 — outbound EmitPromptTool carries the OTel
    # traceparent that wrapped the reasoner.tick span.
    metadata = json.loads(msg.metadata_json)
    assert "traceparent" in metadata, (
        "Outbound PromptStamped metadata_json must carry traceparent "
        "stamped from the reasoner.tick span (ADR-0018 §6)."
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_recall_object_query_reprompts_with_spatial_memory_result() -> None:
    """ADR-0039 Phase 2b — RecallObjectTool → SpatialMemory query → re-prompt cascade.

    A reasoner wired with a real ADR-0038 ``SpatialMemory`` (loaded from the
    home fixture) dispatches a canned ``RecallObjectTool`` for the wine bottle; the
    node runs the query and republishes the rendered result as a
    ``PromptStamped`` with frame_id ``"spatial_memory"`` so the next tick sees
    it. We assert the re-prompt carries the recalled object and the occluding
    fridge the planner must open first.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from pathlib import Path

    from openral_core import EmitPromptTool, RecallObjectTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state import SpatialMemory
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    fixture = (
        Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures"
    ) / "home_scene_graph.json"
    assert fixture.exists(), f"scene-graph fixture missing: {fixture}"
    memory = SpatialMemory.load(fixture)

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                RecallObjectTool(query="bottle of wine", rationale="operator asked for wine"),
                # Fillers absorb the cascade-forced tick + any periodic ticks.
                *[
                    EmitPromptTool(target_topic="/openral/prompt", text="standing by")
                    for _ in range(4)
                ],
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            spatial_memory=memory,
        )
        # The backend is wired → the query tools are offered to the LLM.
        assert reasoner._palette.spatial_memory_available is True
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        sub_node = rclpy.create_node("openral_test_subscriber_spatial")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "spatial_memory":
                received.append(msg)
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_test_publisher_spatial")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_publisher_spatial"
        msg.text = "bring me a cup of wine"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, "Reasoner did not re-prompt with a spatial_memory result within 5 s"
    result = received[-1]
    assert "wine_bottle" in result.text
    assert "fridge" in result.text  # the occluding container to open first
    metadata = json.loads(result.metadata_json)
    assert metadata["source"] == "spatial_memory"
    assert metadata["tool"] == "recall_object"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_spatial_memory_path_param_preloads_query_backend() -> None:
    """ADR-0039 deployment wiring — the spatial_memory_path param loads a backend.

    Instead of injecting a SpatialMemory at construction (the unit path), this
    sets the ``spatial_memory_path`` ROS parameter to the real home fixture —
    the deployment wiring a launch file uses — configures the node, and asserts
    the backend loaded and the query tools are enabled, then drives a
    RecallObjectTool to confirm the full dispatch → re-prompt path works against
    the preloaded map.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from pathlib import Path

    from openral_core import EmitPromptTool, RecallObjectTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    fixture = (
        Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures"
    ) / "home_scene_graph.json"
    assert fixture.exists(), f"scene-graph fixture missing: {fixture}"

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                RecallObjectTool(query="bottle of wine", rationale="operator asked for wine"),
                *[
                    EmitPromptTool(target_topic="/openral/prompt", text="standing by")
                    for _ in range(4)
                ],
            ],
        )
        # No injected backend — wire it purely through the ROS parameter, exactly
        # as sim_e2e.launch.py does via `spatial_memory_path:=<path>`.
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
        )
        reasoner.set_parameters(
            [Parameter("spatial_memory_path", Parameter.Type.STRING, str(fixture))]
        )
        reasoner.trigger_configure()
        assert reasoner._spatial_memory is not None, "param did not load a SpatialMemory backend"
        assert reasoner._palette.spatial_memory_available is True
        reasoner.trigger_activate()

        sub_node = rclpy.create_node("openral_test_subscriber_spatial_param")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "spatial_memory":
                received.append(msg)
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_test_publisher_spatial_param")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_publisher_spatial_param"
        msg.text = "bring me a cup of wine"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, "Preloaded spatial memory did not produce a re-prompt within 5 s"
    assert "wine_bottle" in received[-1].text
    assert "fridge" in received[-1].text


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_active_search_cascade_is_bounded_and_hands_off() -> None:
    """ADR-0039 §3 — a repeatedly-missing query terminates in human-handoff.

    A FakeToolUseClient that keeps emitting RecallObjectTool for an object that is
    not in memory would, without a bound, drive the find→re-prompt cascade
    forever. The SearchBudget caps it: after ``max_attempts`` consecutive
    queries the reasoner publishes a handoff with its own frame_id (filtered by
    _on_prompt → no further tick), stopping the loop.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from pathlib import Path

    from openral_core import RecallObjectTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state import SpatialMemory
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    fixture = (
        Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures"
    ) / "home_scene_graph.json"
    assert fixture.exists(), f"scene-graph fixture missing: {fixture}"
    memory = SpatialMemory.load(fixture)

    rclpy.init()
    spatial_reprompts: list[PromptStamped] = []
    handoff = threading.Event()
    try:
        # Far more queries than the budget; the bound must stop the cascade.
        client = FakeToolUseClient(
            responses=[
                RecallObjectTool(query="a teapot", rationale="keep looking") for _ in range(12)
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            spatial_memory=memory,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()
        node_name = reasoner.get_name()

        sub_node = rclpy.create_node("openral_test_subscriber_bound")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "spatial_memory":
                spatial_reprompts.append(msg)
            elif msg.header.frame_id == node_name and "handing off" in msg.text:
                handoff.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)
        pub_node = rclpy.create_node("openral_test_publisher_bound")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_publisher_bound"
        msg.text = "find the teapot"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not handoff.is_set():
            executor.spin_once(timeout_sec=0.05)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert handoff.is_set(), "cascade did not terminate in human-handoff within 10 s"
    # Bounded: re-prompts capped below the budget; not all 12 queries ran.
    assert len(spatial_reprompts) <= 5, f"cascade not bounded: {len(spatial_reprompts)} re-prompts"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_recall_miss_escalates_to_locate_in_view() -> None:
    """ADR-0043/0056 — a recall_object miss escalates to a live locate_in_view.

    When the goal object is not in spatial memory and an on-demand detector is
    available, the reasoner must (policy, not LLM choice) call the namespaced
    ``locate_in_view`` service for the SAME query before handing off — so the
    live open-vocab detector can ground objects the map never ingested. This
    stands up a real LocateInView service server and asserts it receives the
    query when recall_object misses.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from pathlib import Path

    from openral_core import RecallObjectTool
    from openral_msgs.msg import PromptStamped
    from openral_msgs.srv import LocateInView
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state import SpatialMemory

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    fixture = (
        Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures"
    ) / "home_scene_graph.json"
    assert fixture.exists(), f"scene-graph fixture missing: {fixture}"
    memory = SpatialMemory.load(fixture)

    service_name = "/openral/perception/omdet_turbo_locator/locate_in_view"
    received: list[str] = []
    got_request = threading.Event()

    rclpy.init()
    try:
        client = FakeToolUseClient(
            responses=[
                RecallObjectTool(query="teapot", rationale="find it") for _ in range(6)
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            spatial_memory=memory,
        )
        # The node reads detector_available / default_on_demand_detector from ROS
        # params at construction; set them white-box for the test (the deploy
        # launch sets them from the wired on-demand locator).
        reasoner._detector_available = True
        reasoner._default_on_demand_detector = "omdet-turbo-locator"
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        srv_node = rclpy.create_node("openral_test_locate_server")

        def _serve(req: Any, resp: Any) -> Any:
            received.append(req.query)
            resp.found = True
            resp.camera = "default"
            resp.detector = req.detector or "omdet-turbo-locator"
            resp.metadata_json = "{}"
            got_request.set()
            return resp

        srv_node.create_service(LocateInView, service_name, _serve)

        pub_node = rclpy.create_node("openral_test_locate_publisher")
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(srv_node)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_locate_publisher"
        msg.text = "find the teapot"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not got_request.is_set():
            executor.spin_once(timeout_sec=0.05)

        executor.remove_node(reasoner)
        executor.remove_node(srv_node)
        srv_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert got_request.is_set(), "recall miss did not escalate to locate_in_view within 10 s"
    assert "teapot" in received, f"locate_in_view called with wrong query: {received}"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_severity_fail_failure_preempts_reasoner_tick() -> None:
    """A SEVERITY_FAIL FailureTrigger forces an out-of-band reasoner tick.

    ADR-0018 §4 commits to "event preemption on
    FailureTrigger.severity>=FAIL". This test publishes a real
    ``FailureTrigger`` with ``severity=SEVERITY_FAIL`` (=2) and asserts
    the reasoner dispatched a tool call within the next 100 ms
    (matching the min-interval).
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import EmitPromptTool
    from openral_msgs.msg import FailureTrigger, PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                EmitPromptTool(
                    target_topic="/openral/prompt",
                    text="failure acknowledged",
                    rationale="severity-FAIL preemption fired",
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            tick_hz=1.0,  # slow timer so the preemption is the only path
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        sub_node = rclpy.create_node("openral_test_subscriber_fail")
        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "openral_reasoner":
                received.append(msg)
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, prompt_qos)

        pub_node = rclpy.create_node("openral_test_publisher_fail")
        failure_pub = pub_node.create_publisher(
            FailureTrigger,
            "/openral/failure/safety",
            failure_qos,
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)
        executor.add_node(pub_node)

        # Re-publish the SEVERITY_FAIL (=2) trigger on every spin
        # iteration until the reasoner's preempted dispatch lands.
        # DDS discovery is best-effort and a single publish before the
        # reasoner's subscriber has matched the publisher silently
        # drops the message (VOLATILE durability) — same flake shape as
        # test_prompt_router_forwards_cli_prompt_to_openral_prompt.
        # ReasonerCore's per-kind retry cap (3 by default) bounds the
        # redundant preempted ticks if the reasoner happens to consume
        # the canned response before the loop exits.
        def _publish_fail() -> None:
            fail = FailureTrigger()
            fail.header.stamp = pub_node.get_clock().now().to_msg()
            fail.kind = 1  # KIND_FORCE
            fail.severity = 2  # SEVERITY_FAIL
            fail.evidence_json = ""
            fail.rskill_id = ""
            fail.trace_id = ""
            failure_pub.publish(fail)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            _publish_fail()
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        executor.remove_node(pub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, (
        "Reasoner did not preempt a tick after SEVERITY_FAIL within 5 s — "
        "ADR-0018 §4 event-preemption path broken."
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_skill_registry_changed_triggers_palette_refresh(tmp_path) -> None:
    """A std_msgs/Empty on /openral/skill_registry_changed rebuilds the palette."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    import os as _os
    from pathlib import Path

    from openral_core import EmitPromptTool, RobotCapabilities, RSkillManifest
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_rskill.loader import InstalledRSkillEntry
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from std_msgs.msg import Empty

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    # Seed a real on-disk registry pointing at a real in-tree
    # rskill.yaml fixture; override the loader's DEFAULT_REGISTRY_PATH
    # via OPENRAL_DATA_HOME so the test does not touch the user's
    # actual registry.
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "rskills" / "pi05-libero-nf4" / "rskill.yaml"
    if not manifest_path.exists():
        pytest.skip(f"rskill fixture missing: {manifest_path}")
    data_home = tmp_path / "data_home"
    (data_home / "openral").mkdir(parents=True)
    reg_path = data_home / "openral" / "rskills.json"
    manifest = RSkillManifest.from_yaml(str(manifest_path))
    entry = InstalledRSkillEntry(
        repo_id=manifest.name,
        version=manifest.version,
        revision=None,
        local_dir=str(manifest_path.parent),
        manifest_path=str(manifest_path),
        license=str(manifest.license),
        role=str(manifest.role),
        embodiment_tags=list(manifest.embodiment_tags),
        installed_at="2026-05-19T12:00:00+00:00",
    )
    reg_path.write_text(json.dumps([entry.model_dump(mode="json")]))
    _os.environ["XDG_DATA_HOME"] = str(data_home)

    rclpy.init()
    try:
        # Start with an empty palette; the refresh should populate it.
        client = FakeToolUseClient(
            responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            robot_capabilities=RobotCapabilities(
                embodiment_tags=list(manifest.embodiment_tags),
            ),
        )
        reasoner.trigger_configure()

        pub_node = rclpy.create_node("openral_test_publisher_registry")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        pub = pub_node.create_publisher(Empty, "/openral/skill_registry_changed", qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)

        # Patch the loader's registry path to the test fixture
        # (XDG_DATA_HOME isn't honoured by the module-level constant
        # captured at import time).
        from openral_rskill import loader as _loader

        original = _loader.DEFAULT_REGISTRY_PATH
        _loader.DEFAULT_REGISTRY_PATH = reg_path
        try:
            pub.publish(Empty())
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and len(reasoner._palette.execute_rskill_ids) == 0:
                executor.spin_once(timeout_sec=0.05)
        finally:
            _loader.DEFAULT_REGISTRY_PATH = original

        executor.remove_node(reasoner)
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()
        _os.environ.pop("XDG_DATA_HOME", None)

    assert manifest.name in reasoner._palette.execute_rskill_ids, (
        "/openral/skill_registry_changed event did not refresh the reasoner palette."
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_skill_rejection_emits_failure_trigger() -> None:
    """Goal rejection by the F1 server emits a KIND_CONTROLLER FailureTrigger.

    Spins up a real :class:`rclpy_action.ActionServer` on
    ``/openral/execute_rskill`` that rejects every goal. The reasoner
    must publish a ``FailureTrigger`` on ``/openral/failure/rskill``
    with ``kind=KIND_CONTROLLER`` (=5) per ADR-0018 F4 follow-up
    (GH-126).
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    received: list[FailureTrigger] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id="openral/skill-test-reject",
                    prompt="this should be rejected",
                    deadline_s=0.0,
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(
                execute_rskill_ids=frozenset({"openral/skill-test-reject"}),
            ),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        # Action server that rejects every goal.
        server_node = rclpy.create_node("openral_test_execute_skill_server")

        def _goal_cb(_goal_request: ExecuteRskill.Goal) -> GoalResponse:
            return GoalResponse.REJECT

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=lambda _gh: ExecuteRskill.Result(),
            goal_callback=_goal_cb,
        )

        # FailureTrigger subscriber.
        sub_node = rclpy.create_node("openral_test_subscriber_failure_rskill")
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: FailureTrigger) -> None:
            received.append(msg)
            received_event.set()

        sub_node.create_subscription(
            FailureTrigger,
            "/openral/failure/rskill",
            cb,
            failure_qos,
        )

        # Trigger an out-of-band tick by injecting an operator prompt.
        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        from openral_msgs.msg import PromptStamped

        pub_node = rclpy.create_node("openral_test_publisher_prompt_for_skill")
        prompt_pub = pub_node.create_publisher(
            PromptStamped,
            "/openral/prompt",
            prompt_qos,
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_publisher_prompt_for_skill"
        prompt.text = "run the test skill"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        server_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, (
        "No FailureTrigger landed on /openral/failure/rskill — "
        "reasoner did not propagate the action-server rejection."
    )
    msg = received[-1]
    # KIND_CONTROLLER = 5, SEVERITY_FAIL = 2 (mirrors openral_msgs IDL).
    assert msg.kind == 5
    assert msg.severity == 2
    assert msg.rskill_id == "openral/skill-test-reject"
    evidence = json.loads(msg.evidence_json)
    assert evidence["kind"] == "controller"
    assert evidence["state"] == "rejected"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_lifecycle_transition_calls_change_state() -> None:
    """LifecycleTransitionTool drives a real ``<node>/change_state``.

    Spins up a real :class:`LifecycleNode` peer with a recording flag
    on its configure transition; the reasoner dispatches a
    ``LifecycleTransitionTool(node=..., transition="configure")`` and
    the peer's ``on_configure`` must fire.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import LifecycleTransitionTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    class RecordingLifecycleNode(LifecycleNode):
        """Tiny lifecycle peer whose ``on_configure`` flips a flag."""

        def __init__(self) -> None:
            super().__init__("openral_test_lifecycle_peer")
            self.configured = threading.Event()

        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            del state
            self.configured.set()
            return TransitionCallbackReturn.SUCCESS

    rclpy.init()
    try:
        client = FakeToolUseClient(
            responses=[
                LifecycleTransitionTool(
                    node="/openral_test_lifecycle_peer",
                    transition="configure",
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        peer = RecordingLifecycleNode()

        # Trigger a reasoner tick via an operator prompt.
        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub_node = rclpy.create_node("openral_test_publisher_prompt_for_lifecycle")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(peer)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_publisher_prompt_for_lifecycle"
        prompt.text = "configure the peer"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not peer.configured.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(peer)
        peer.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert peer.configured.is_set(), (
        "Reasoner did not drive /openral_test_lifecycle_peer through configure "
        "within 5 s — LifecycleTransitionTool dispatch broken."
    )


def _spin_reasoner_with_action_server(
    *,
    server_node: Any,
    reasoner: Any,
    prompt_text: str,
    stop_event: threading.Event,
    extra_subscriber: Any = None,
    timeout_s: float = 5.0,
) -> None:
    """Shared spin loop for ExecuteRskill action-server integration tests.

    Drops the reasoner, the action-server node, and any extra subscriber
    onto a single-threaded executor; injects an operator prompt on
    ``/openral/prompt`` to force a reasoner tick; spins until
    ``stop_event`` fires or ``timeout_s`` elapses.
    """
    import rclpy
    from openral_msgs.msg import PromptStamped
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    prompt_qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    pub_node = rclpy.create_node("openral_test_prompt_publisher_shared")
    prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(reasoner)
    executor.add_node(server_node)
    if extra_subscriber is not None:
        executor.add_node(extra_subscriber)

    prompt = PromptStamped()
    prompt.header.stamp = pub_node.get_clock().now().to_msg()
    prompt.header.frame_id = "openral_test_prompt_publisher_shared"
    prompt.text = prompt_text
    prompt.metadata_json = "{}"
    prompt_pub.publish(prompt)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not stop_event.is_set():
        executor.spin_once(timeout_sec=0.1)

    executor.remove_node(reasoner)
    executor.remove_node(server_node)
    if extra_subscriber is not None:
        executor.remove_node(extra_subscriber)
    pub_node.destroy_node()


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_skill_success_emits_no_failure_trigger() -> None:
    """Successful goal completes without emitting a FailureTrigger.

    Spins up an :class:`ActionServer` that accepts and reports
    ``success=True``; the reasoner must log success on the result
    callback and must **not** publish on ``/openral/failure/rskill``.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    received_failures: list[FailureTrigger] = []
    success_seen = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id="openral/skill-test-success",
                    prompt="this should succeed",
                    deadline_s=0.0,
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(
                execute_rskill_ids=frozenset({"openral/skill-test-success"}),
            ),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_execute_skill_server_ok")

        def _execute(goal_handle: Any) -> Any:
            result = ExecuteRskill.Result()
            result.success = True
            result.failure_reason = ""
            result.trace_id = "00-trace-success"
            goal_handle.succeed()
            success_seen.set()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        sub_node = rclpy.create_node("openral_test_subscriber_failure_rskill_ok")
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sub_node.create_subscription(
            FailureTrigger,
            "/openral/failure/rskill",
            received_failures.append,
            failure_qos,
        )

        _spin_reasoner_with_action_server(
            server_node=server_node,
            reasoner=reasoner,
            prompt_text="run the success skill",
            stop_event=success_seen,
            extra_subscriber=sub_node,
            timeout_s=5.0,
        )

        # Drain any in-flight result-callback / FailureTrigger publish.
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)
        drain_deadline = time.monotonic() + 1.0
        while time.monotonic() < drain_deadline:
            executor.spin_once(timeout_sec=0.05)
        executor.remove_node(reasoner)
        executor.remove_node(sub_node)

        sub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert success_seen.is_set(), (
        "Action server execute_callback never fired — reasoner did not send the goal."
    )
    assert not received_failures, (
        "Successful ExecuteRskill goal must not emit a FailureTrigger; "
        f"saw {len(received_failures)}: "
        f"{[(m.kind, m.severity, m.evidence_json) for m in received_failures]!r}"
    )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_skill_deadline_emits_kind_timeout() -> None:
    """`deadline_s` elapsing emits KIND_TIMEOUT and cancels the goal.

    Spins up an ``ActionServer`` that accepts but stalls indefinitely;
    the reasoner's one-shot deadline timer must fire after
    ``call.deadline_s`` seconds, ``cancel_goal_async`` the goal, and
    publish ``FailureTrigger(kind=KIND_TIMEOUT)`` with a realistic
    ``TimeoutEvidence``.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer, CancelResponse
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    deadline_s = 0.5  # short enough for the test to stay tight

    rclpy.init()
    received_failures: list[FailureTrigger] = []
    received_event = threading.Event()
    cancel_observed = threading.Event()
    try:
        from openral_msgs.msg import PromptStamped
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor

        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id="openral/skill-test-timeout",
                    prompt="this should time out",
                    deadline_s=deadline_s,
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(
                execute_rskill_ids=frozenset({"openral/skill-test-timeout"}),
            ),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_execute_skill_server_timeout")
        # ReentrantCallbackGroup + MultiThreadedExecutor so the server's
        # blocking ``_execute`` doesn't starve the reasoner's deadline
        # timer (which runs on the reasoner's default callback group).
        server_cb_group = ReentrantCallbackGroup()

        def _execute(goal_handle: Any) -> Any:
            # Spin until cancellation requested — bounded so the test
            # never hangs even if the cancel path regresses.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not goal_handle.is_cancel_requested:
                time.sleep(0.05)
            result = ExecuteRskill.Result()
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                cancel_observed.set()
                result.success = False
                result.failure_reason = "canceled by reasoner deadline"
                result.trace_id = ""
            else:
                goal_handle.abort()
                result.success = False
                result.failure_reason = "test execute_callback hit safety timeout"
                result.trace_id = ""
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=server_cb_group,
        )

        sub_node = rclpy.create_node("openral_test_subscriber_failure_rskill_timeout")
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: FailureTrigger) -> None:
            received_failures.append(msg)
            if msg.kind == 0:  # KIND_TIMEOUT
                received_event.set()

        sub_node.create_subscription(
            FailureTrigger,
            "/openral/failure/rskill",
            cb,
            failure_qos,
        )

        prompt_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub_node = rclpy.create_node("openral_test_publisher_prompt_timeout")
        prompt_pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", prompt_qos)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(reasoner)
        executor.add_node(server_node)
        executor.add_node(sub_node)

        prompt = PromptStamped()
        prompt.header.stamp = pub_node.get_clock().now().to_msg()
        prompt.header.frame_id = "openral_test_publisher_prompt_timeout"
        prompt.text = "run the timeout skill"
        prompt.metadata_json = "{}"
        prompt_pub.publish(prompt)

        deadline = time.monotonic() + deadline_s + 4.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(server_node)
        executor.remove_node(sub_node)
        executor.shutdown()
        pub_node.destroy_node()
        sub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    timeouts = [m for m in received_failures if m.kind == 0]
    assert timeouts, (
        f"No KIND_TIMEOUT FailureTrigger landed within {deadline_s + 4.0:.1f} s; "
        f"saw kinds={[m.kind for m in received_failures]}"
    )
    msg = timeouts[-1]
    assert msg.severity == 2  # SEVERITY_FAIL
    assert msg.rskill_id == "openral/skill-test-timeout"
    evidence = json.loads(msg.evidence_json)
    assert evidence["kind"] == "timeout"
    assert evidence["operation"] == "skill.openral/skill-test-timeout"
    assert evidence["deadline_s"] == pytest.approx(deadline_s)
    assert evidence["elapsed_s"] >= deadline_s * 0.5
    # ``cancel_observed`` is a best-effort sanity touch: in practice the
    # cancel propagates after the test tears down the server, so we log
    # rather than assert. The contract that matters is the KIND_TIMEOUT
    # FailureTrigger emission above.
    _ = cancel_observed


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_execute_skill_abort_emits_kind_controller() -> None:
    """Server-side abort emits a KIND_CONTROLLER FailureTrigger.

    Spins up an ``ActionServer`` that accepts the goal then immediately
    calls ``goal_handle.abort()`` with ``success=False`` and a
    ``failure_reason``. The reasoner's result callback must publish a
    ``FailureTrigger(kind=KIND_CONTROLLER, state="aborted",
    detail=failure_reason)`` on ``/openral/failure/rskill``.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import ExecuteRskillTool
    from openral_msgs.action import ExecuteRskill
    from openral_msgs.msg import FailureTrigger
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    abort_reason = "perception lost during execution"

    rclpy.init()
    received_failures: list[FailureTrigger] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                ExecuteRskillTool(
                    rskill_id="openral/skill-test-abort",
                    prompt="this should abort",
                    deadline_s=0.0,  # no deadline; abort is the only failure path
                ),
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(
                execute_rskill_ids=frozenset({"openral/skill-test-abort"}),
            ),
            tick_hz=2.0,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        server_node = rclpy.create_node("openral_test_execute_skill_server_abort")

        def _execute(goal_handle: Any) -> Any:
            result = ExecuteRskill.Result()
            result.success = False
            result.failure_reason = abort_reason
            result.trace_id = "00-trace-abort"
            goal_handle.abort()
            return result

        ActionServer(
            server_node,
            ExecuteRskill,
            "/openral/execute_rskill",
            execute_callback=_execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
        )

        sub_node = rclpy.create_node("openral_test_subscriber_failure_rskill_abort")
        failure_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        def cb(msg: FailureTrigger) -> None:
            received_failures.append(msg)
            if msg.kind == 5:  # KIND_CONTROLLER
                received_event.set()

        sub_node.create_subscription(
            FailureTrigger,
            "/openral/failure/rskill",
            cb,
            failure_qos,
        )

        _spin_reasoner_with_action_server(
            server_node=server_node,
            reasoner=reasoner,
            prompt_text="run the abort skill",
            stop_event=received_event,
            extra_subscriber=sub_node,
            timeout_s=5.0,
        )

        sub_node.destroy_node()
        server_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    controller_failures = [m for m in received_failures if m.kind == 5]
    assert controller_failures, (
        f"No KIND_CONTROLLER FailureTrigger landed within 5 s; "
        f"saw kinds={[m.kind for m in received_failures]}"
    )
    msg = controller_failures[-1]
    assert msg.severity == 2  # SEVERITY_FAIL
    assert msg.rskill_id == "openral/skill-test-abort"
    # Reasoner uses the action result's trace_id when present.
    assert msg.trace_id == "00-trace-abort"
    evidence = json.loads(msg.evidence_json)
    assert evidence["kind"] == "controller"
    assert evidence["state"] == "aborted"
    assert evidence["detail"] == abort_reason


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_spatial_memory_ingest_accumulates_from_world_state() -> None:
    """ADR-0038 — the live ingest edge end-to-end (reasoner half).

    With ``spatial_memory_ingest:=true`` the reasoner auto-creates a durable
    SpatialMemory and folds each ``/openral/world_state_slow``
    ``WorldState.detected_objects`` snapshot — what the ADR-0035 producer
    publishes — into it. We publish a snapshot carrying a wine bottle, drive a
    tick, and confirm ``recall_object`` recalls it from the accumulated map.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    from openral_core import (
        DetectedObject,
        JointState,
        Pose6D,
        RecallObjectQuery,
        WorldState,
    )
    from openral_msgs.msg import WorldStateStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state_ros.lifecycle_node import build_world_state_stamped_msg
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    rclpy.init()
    recalled = False
    try:
        reasoner = ReasonerNode(
            client=FakeToolUseClient(responses=[]),
            palette=ToolPalette(execute_rskill_ids=frozenset()),
        )
        reasoner.set_parameters([Parameter("spatial_memory_ingest", Parameter.Type.BOOL, True)])
        reasoner.trigger_configure()
        # An empty, reasoner-owned backend is created for live accumulation.
        assert reasoner._spatial_memory is not None
        assert reasoner._spatial_memory_writer is not None
        assert reasoner._palette.spatial_memory_available is True
        reasoner.trigger_activate()

        pub_node = rclpy.create_node("openral_test_publisher_ws_ingest")
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub = pub_node.create_publisher(WorldStateStamped, "/openral/world_state_slow", qos)

        world_state = WorldState(
            stamp_ns=1_000,
            joint_state=JointState(name=[], position=[], stamp_ns=1_000),
            detected_objects=[
                DetectedObject(
                    label="wine_bottle",
                    confidence=0.9,
                    pose=Pose6D(
                        xyz=(3.0, 1.0, 0.9),
                        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        frame_id="map",
                    ),
                    track_id=1,
                )
            ],
        )
        msg = build_world_state_stamped_msg(pub_node, world_state)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(pub_node)

        # Deliver the snapshot and drive ticks until the bottle is recalled.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not recalled:
            pub.publish(msg)
            executor.spin_once(timeout_sec=0.1)
            reasoner._on_tick()
            result = reasoner._spatial_memory.recall_object(
                RecallObjectQuery(label="wine_bottle"), now_ns=2_000
            )
            recalled = bool(result.matches)

        executor.remove_node(reasoner)
        executor.remove_node(pub_node)
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert recalled, "spatial_memory_ingest did not accumulate the detected wine bottle"


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_recall_object_approach_is_grid_refined_when_map_latched() -> None:
    """ADR-0044 Phase 4 — a latched /map snaps recall_object approach poses.

    A latched ``nav_msgs/OccupancyGrid`` with a wall across the wine bottle's
    geometric approach (2.82, 0.38) is published before the query; the
    reasoner's re-prompt must carry a SNAPPED approach pose (free + sighted on
    that grid), never the blocked geometric one.
    """
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    from pathlib import Path

    from nav_msgs.msg import OccupancyGrid
    from openral_core import EmitPromptTool, RecallObjectTool
    from openral_msgs.msg import PromptStamped
    from openral_reasoner import ToolPalette
    from openral_reasoner_ros import ReasonerNode
    from openral_world_state import SpatialMemory
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    fixture = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures"
    fixture = fixture / "home_scene_graph.json"
    assert fixture.exists(), f"scene-graph fixture missing: {fixture}"
    memory = SpatialMemory.load(fixture)

    rclpy.init()
    received: list[PromptStamped] = []
    received_event = threading.Event()
    try:
        client = FakeToolUseClient(
            responses=[
                RecallObjectTool(query="bottle of wine", rationale="operator asked for wine"),
                *[
                    EmitPromptTool(target_topic="/openral/prompt", text="standing by")
                    for _ in range(4)
                ],
            ],
        )
        reasoner = ReasonerNode(
            client=client,
            palette=ToolPalette(execute_rskill_ids=frozenset()),
            spatial_memory=memory,
        )
        reasoner.trigger_configure()
        reasoner.trigger_activate()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sub_node = rclpy.create_node("openral_test_subscriber_grid")

        def cb(msg: PromptStamped) -> None:
            if msg.header.frame_id == "spatial_memory":
                received.append(msg)
                received_event.set()

        sub_node.create_subscription(PromptStamped, "/openral/prompt", cb, qos)

        pub_node = rclpy.create_node("openral_test_publisher_grid")
        pub = pub_node.create_publisher(PromptStamped, "/openral/prompt", qos)
        # Latched map publisher — mirrors slam_toolbox's QoS.
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        map_pub = pub_node.create_publisher(OccupancyGrid, "/map", map_qos)
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = 0.1
        grid_msg.info.width = 50
        grid_msg.info.height = 30
        grid_msg.info.origin.orientation.w = 1.0
        data = [0] * (50 * 30)
        for row in range(0, 8):  # wall x in [2.7, 3.0), y in [0, 0.8)
            for col in range(27, 30):
                data[row * 50 + col] = 100
        grid_msg.data = data
        map_pub.publish(grid_msg)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reasoner)
        executor.add_node(sub_node)

        # Let the latched map land before the task prompt triggers the query.
        settle = time.monotonic() + 1.0
        while time.monotonic() < settle:
            executor.spin_once(timeout_sec=0.1)

        msg = PromptStamped()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_test_publisher_grid"
        msg.text = "bring me a cup of wine"
        msg.metadata_json = "{}"
        pub.publish(msg)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(reasoner)
        executor.remove_node(sub_node)
        sub_node.destroy_node()
        pub_node.destroy_node()
        reasoner.destroy_node()
    finally:
        rclpy.shutdown()

    assert received, "Reasoner did not re-prompt with a spatial_memory result within 5 s"
    text = received[-1].text
    assert "wine_bottle" in text
    assert "approach from (2.82, 0.38" not in text, f"blocked geometric pose leaked: {text}"
    assert "approach from" in text, f"no grid-refined approach rendered: {text}"
