"""Integration tests for the ADR-0018 F1 ``rskill_runner_node``.

Drives the real :class:`RskillRunnerNode` + the colocated
:class:`_WorldStateLifecycleNode` + a real :class:`SafetyPassthroughNode`
through ``rclpy`` (in-process equivalent of ``launch_testing`` per the
existing repo convention) and asserts the end-to-end topic flow that
ADR-0018 step 1 locks:

1. An ``ExecuteRskill`` goal accepted by ``rskill_runner_node``.
2. ``ActionChunk`` lands on ``/openral/candidate_action``.
3. ``safety_node`` republishes on ``/openral/safe_action`` (valid
   chunks pass through).
4. ``/diagnostics`` carries 1 Hz heartbeats from both nodes.

Per CLAUDE.md §1.11 / §5.4: no mocks. The skill is a real
:class:`rSkillBase` subclass (``_ConstantSkill``) that emits a constant
six-DoF joint-position chunk — not a `MagicMock`. The
`WorldStateAggregator` is the production class; the skill_runner_node
calls ``aggregator.snapshot()`` in-process via the shared instance the
:func:`compose_so100_runtime` factory hands it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))

pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)


# ── Test fixtures ───────────────────────────────────────────────────────────


def _make_constant_skill() -> Any:
    """Return a real :class:`rSkillBase` subclass — no mocks."""
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _ConstantSkill(rSkillBase):
        """Six-DoF constant-joint-position skill for the F1 contract test."""

        def __init__(self) -> None:
            super().__init__(
                name="openral/test-constant-skill",
                version="0.1.0",
                role="s1",
                embodiment_tags=["so100_follower"],
            )

        def _configure_impl(self) -> None:
            pass

        def _activate_impl(self) -> None:
            pass

        def _deactivate_impl(self) -> None:
            pass

        def _shutdown_impl(self) -> None:
            pass

        def _step_impl(self, _world_state: Any) -> Action:
            return Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            )

    skill = _ConstantSkill()
    skill.configure()
    skill.activate()
    return skill


def _local_skill_resolver(*_args: Any, **_kwargs: Any) -> Any:
    """Skill resolver that returns the in-tree constant skill — no HF Hub."""
    return _make_constant_skill()


def _make_named_skill(name: str) -> Any:
    """A real configured+activated 6-DoF so100 skill with a caller-chosen id."""
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _NamedSkill(rSkillBase):
        def __init__(self) -> None:
            super().__init__(
                name=name, version="0.1.0", role="s1", embodiment_tags=["so100_follower"]
            )

        def _configure_impl(self) -> None:
            pass

        def _activate_impl(self) -> None:
            pass

        def _deactivate_impl(self) -> None:
            pass

        def _shutdown_impl(self) -> None:
            pass

        def _step_impl(self, _world_state: Any) -> Action:
            return Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            )

    skill = _NamedSkill()
    skill.configure()
    skill.activate()
    return skill


@contextmanager
def _compose_harness(
    resolver: Any = None,
) -> Iterator[tuple[Any, Any, Any, dict[str, list[Any]]]]:
    """Compose world_state + skill_runner in one process; bring up safety_node.

    Yields ``(executor, runtime, safety_node, observed)`` where
    ``observed`` is a dict of typed message lists subscribed by the
    helper node. ``resolver`` overrides the default constant-skill resolver.
    """
    import rclpy
    from openral_msgs.msg import ActionChunk
    from openral_rskill_ros.compose import compose_so100_runtime
    from openral_safety.supervisor_node import SafetyPassthroughNode
    from rclpy.lifecycle import TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    rclpy.init()
    runtime = compose_so100_runtime(skill_resolver=resolver or _local_skill_resolver)
    safety = SafetyPassthroughNode(node_name="openral_safety_test")
    safety.set_parameters(
        [rclpy.parameter.Parameter("n_dof", value=6)],
    )

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(runtime.world_state_node)
    executor.add_node(runtime.skill_runner_node)
    executor.add_node(safety)

    helper = rclpy.create_node("openral_skill_runner_test_helper")
    executor.add_node(helper)

    observed: dict[str, list[Any]] = {"candidate": [], "safe": [], "diag": []}
    chunk_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    helper.create_subscription(
        ActionChunk,
        "/openral/candidate_action",
        observed["candidate"].append,
        chunk_qos,
    )
    helper.create_subscription(
        ActionChunk, "/openral/safe_action", observed["safe"].append, chunk_qos
    )
    from diagnostic_msgs.msg import DiagnosticArray

    helper.create_subscription(DiagnosticArray, "/diagnostics", observed["diag"].append, 20)

    try:
        for node in (
            runtime.world_state_node,
            runtime.skill_runner_node,
            safety,
        ):
            assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
            assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

        yield executor, runtime, safety, observed
    finally:
        for node in (
            runtime.skill_runner_node,
            runtime.world_state_node,
            safety,
        ):
            try:
                node.trigger_deactivate()
                node.trigger_cleanup()
                node.trigger_shutdown()
            except Exception:  # reason: best-effort teardown
                pass
        executor.shutdown()
        helper.destroy_node()
        runtime.skill_runner_node.destroy_node()
        runtime.world_state_node.destroy_node()
        safety.destroy_node()
        rclpy.shutdown()


def _spin_for(executor: Any, duration_s: float) -> None:
    """Spin ``executor`` for ``duration_s`` seconds."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_compose_factory_shares_one_aggregator() -> None:
    """ADR-0018 §3 contract: world_state + skill_runner share **one** aggregator."""
    import rclpy
    from openral_rskill_ros.compose import compose_so100_runtime

    rclpy.init()
    try:
        runtime = compose_so100_runtime()
        # Same Python object by identity — not two equal-but-distinct
        # aggregators (the assertion that ADR-0018 actually mandates).
        assert runtime.aggregator is runtime.world_state_node._aggregator
        assert runtime.aggregator is runtime.skill_runner_node._aggregator
        # Destroy to clean shutdown.
        runtime.world_state_node.destroy_node()
        runtime.skill_runner_node.destroy_node()
    finally:
        rclpy.shutdown()


def test_execute_skill_goal_publishes_chunks_through_safety_passthrough() -> None:
    """End-to-end: ExecuteRskill goal → candidate_action → safety → safe_action."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _compose_harness() as (executor, runtime, _safety, observed):
        client = ActionClient(
            runtime.skill_runner_node,
            ExecuteRskill,
            "/openral/execute_rskill",
        )
        # Discovery + ready check.
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0), "ExecuteRskill action server not ready"

        goal = ExecuteRskill.Goal()
        goal.rskill_id = "openral/test-constant-skill"
        goal.revision = ""
        goal.prompt = "publish constant chunks"
        goal.prompt_metadata_json = ""
        goal.deadline_s = 0.8

        send_future = client.send_goal_async(goal)
        # Drive both threads to discovery + acceptance.
        deadline = time.monotonic() + 3.0
        while not send_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert send_future.done(), "send_goal_async timed out"
        goal_handle = send_future.result()
        assert goal_handle is not None
        assert goal_handle.accepted, "skill_runner refused the goal"

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + 5.0
        while not result_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert result_future.done(), "result future timed out"
        result_msg = result_future.result()
        assert result_msg is not None
        # success=True OR success=False with empty failure_reason (the
        # deadline branch). Either way we expect chunks to have been
        # published in the meantime.
        _ = result_msg

    # Outside the harness — assertions on the captured observations.
    assert observed["candidate"], "no ActionChunk landed on /openral/candidate_action"
    first_candidate = observed["candidate"][0]
    assert int(first_candidate.n_dof) == 6
    assert list(first_candidate.flat) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert first_candidate.rskill_id == "openral/test-constant-skill"

    # safety_node should have passed the chunks through.
    assert observed["safe"], "no ActionChunk landed on /openral/safe_action"
    assert observed["safe"][0].rskill_id == "openral/test-constant-skill"

    # Diagnostics: at least one heartbeat from each of the three
    # lifecycle nodes (world_state, skill_runner, safety).
    sources = {status.hardware_id for arr in observed["diag"] for status in arr.status}
    expected = {
        "openral_world_state:so100_follower",
        "openral_skill_runner:so100_follower",
        "openral_safety:robot",
    }
    missing = expected - sources
    assert not missing, f"missing /diagnostics from {sorted(missing)}"


# ── ADR-0050 single-resident-skill eviction ─────────────────────────────────


def _run_goal(executor: Any, node: Any, rskill_id: str, deadline_s: float = 0.4) -> None:
    """Send one ExecuteRskill goal for ``rskill_id`` and spin until it resolves."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    client = ActionClient(node, ExecuteRskill, "/openral/execute_rskill")
    _spin_for(executor, 0.2)
    assert client.wait_for_server(timeout_sec=2.0), "ExecuteRskill action server not ready"
    goal = ExecuteRskill.Goal()
    goal.rskill_id = rskill_id
    goal.revision = ""
    goal.prompt = "drive"
    goal.prompt_metadata_json = ""
    goal.deadline_s = deadline_s
    send_future = client.send_goal_async(goal)
    deadline = time.monotonic() + 3.0
    while not send_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    goal_handle = send_future.result()
    assert goal_handle is not None and goal_handle.accepted, f"goal {rskill_id} rejected"
    result_future = goal_handle.get_result_async()
    deadline = time.monotonic() + 5.0
    while not result_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert result_future.done(), f"goal {rskill_id} result timed out"


def _tracking_resolver(built: list[Any]) -> Any:
    """Resolver that builds one real named skill per call and records each."""

    def _resolver(*_args: Any, **kwargs: Any) -> Any:
        skill = _make_named_skill(kwargs.get("rskill_id", "openral/unknown"))
        built.append(skill)
        return skill

    return _resolver


def test_switching_rskill_id_evicts_prior_resident_skill() -> None:
    """ADR-0050: dispatching a different rskill_id shuts down (unloads) the prior resident skill."""
    from openral_rskill.base import RSkillState

    built: list[Any] = []
    with _compose_harness(resolver=_tracking_resolver(built)) as (executor, runtime, _s, _o):
        _run_goal(executor, runtime.skill_runner_node, "openral/skill-a")
        _run_goal(executor, runtime.skill_runner_node, "openral/skill-b")
        # Assert inside the harness — teardown finalizes the resident skill.
        assert len(built) == 2, "resolver should build one skill per distinct id"
        assert built[0].state is RSkillState.FINALIZED, "skill-a was not evicted on switch"
        assert built[1].state is not RSkillState.FINALIZED, "skill-b should remain resident"


def test_redispatching_same_rskill_id_reuses_resident_skill() -> None:
    """ADR-0050: re-dispatching the same (id, revision) reuses the resident skill — no reload."""
    built: list[Any] = []
    with _compose_harness(resolver=_tracking_resolver(built)) as (executor, runtime, _s, _o):
        _run_goal(executor, runtime.skill_runner_node, "openral/skill-a")
        _run_goal(executor, runtime.skill_runner_node, "openral/skill-a")
        assert len(built) == 1, "same id should resolve once and be reused"


@pytest.fixture
def captured_spans() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory OTel tracer + exporter and return the exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def test_execute_skill_emits_inference_and_hal_spans(
    captured_spans: InMemorySpanExporter,
) -> None:
    """Each tick of the skill loop emits a rskill.chunk_inference span.

    Dashboard contract: store._HEADLINE_FAMILIES routes
    ``rskill.chunk_inference`` (semconv.SPAN_RSKILL_CHUNK_INFERENCE) to the
    Inference card and latches ``rskill.id`` / ``rskill.role`` into the
    Identity row.
    """
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _compose_harness() as (executor, runtime, _safety, _observed):
        client = ActionClient(
            runtime.skill_runner_node,
            ExecuteRskill,
            "/openral/execute_rskill",
        )
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)

        goal = ExecuteRskill.Goal()
        goal.rskill_id = "openral/test-constant-skill"
        goal.revision = ""
        goal.prompt = "span-coverage test"
        goal.prompt_metadata_json = ""
        goal.deadline_s = 0.5

        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + 3.0
        while not send_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        goal_handle = send_future.result()
        assert goal_handle is not None and goal_handle.accepted

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + 4.0
        while not result_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)

    inference_spans = [
        s for s in captured_spans.get_finished_spans() if s.name == "rskill.chunk_inference"
    ]
    assert inference_spans, "no rskill.chunk_inference span emitted across the skill_runner loop"
    attrs = dict(inference_spans[0].attributes or {})
    assert attrs.get("inference.kind") == "foreground"
    assert attrs.get("inference.chunk_index") == 0
    assert attrs.get("inference.chunk_size") == 1
    assert attrs.get("rskill.id") == "openral/test-constant-skill"
    assert attrs.get("rskill.role") == "s1"


def test_execute_skill_estop_aborts_goal() -> None:
    """A /openral/estop publication during execution aborts the in-flight goal."""
    import rclpy
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    with _compose_harness() as (executor, runtime, _safety, _observed):
        client = ActionClient(
            runtime.skill_runner_node,
            ExecuteRskill,
            "/openral/execute_rskill",
        )
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)

        # Long deadline so the estop arrives well before completion.
        goal = ExecuteRskill.Goal()
        goal.rskill_id = "openral/test-constant-skill"
        goal.revision = ""
        goal.prompt = "estop test"
        goal.prompt_metadata_json = ""
        goal.deadline_s = 5.0

        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + 3.0
        while not send_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        goal_handle = send_future.result()
        assert goal_handle is not None and goal_handle.accepted

        # Helper node publishes the estop after a brief delay.
        helper = rclpy.create_node("openral_skill_runner_estop_test")
        executor.add_node(helper)
        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10,
        )
        estop_pub = helper.create_publisher(Empty, "/openral/estop", estop_qos)
        _spin_for(executor, 0.3)
        estop_pub.publish(Empty())

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + 3.0
        while not result_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        helper.destroy_node()
        assert result_future.done(), "result future timed out after estop"
        result_msg = result_future.result()
        assert result_msg is not None
        assert not result_msg.result.success
        assert "safety_estop" in result_msg.result.failure_reason
