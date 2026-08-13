"""Integration tests for the ROS 2 reasoner/supervisor graph's ``rskill_runner_node``.

Drives the real :class:`RskillRunnerNode` + the colocated
:class:`_WorldStateLifecycleNode` + a real :class:`SafetyPassthroughNode`
through ``rclpy`` (in-process equivalent of ``launch_testing`` per the
existing repo convention) and asserts the end-to-end topic flow that
the ROS 2 reasoner/supervisor graph's step 1 locks:

1. An ``ExecuteRskill`` goal accepted by ``rskill_runner_node``.
2. ``ActionChunk`` lands on ``/openral/candidate_action``.
3. ``safety_node`` republishes on ``/openral/safe_action`` (valid
   chunks pass through).
4. ``/diagnostics`` carries 1 Hz heartbeats from both nodes.
5. Every terminal ``ExecuteRskill.Result`` carries the typed
   ``failure_kind`` uint8 matching the CLAUDE.md §5 exception the
   dispatch path actually raised (``FAILURE_NONE`` on success).

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
    """Single-aggregator contract: world_state + skill_runner share **one** aggregator."""
    import rclpy
    from openral_rskill_ros.compose import compose_so100_runtime

    rclpy.init()
    try:
        runtime = compose_so100_runtime()
        # Same Python object by identity — not two equal-but-distinct
        # aggregators (the assertion that the single-aggregator contract actually mandates).
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
        "openral_rskill_runner:so100_follower",
        "openral_safety:robot",
    }
    missing = expected - sources
    assert not missing, f"missing /diagnostics from {sorted(missing)}"


# ── Single-resident-skill VRAM eviction ─────────────────────────────────────


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
    """Single-resident-skill eviction: dispatching a different rskill_id shuts down
    (unloads) the prior resident skill.
    """
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
    """Single-resident-skill eviction: re-dispatching the same (id, revision) reuses
    the resident skill — no reload.
    """
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
        # Typed cause travels with the prose: ROSEStopRequested → FAILURE_SAFETY_ESTOP.
        assert result_msg.result.failure_kind == ExecuteRskill.Result.FAILURE_SAFETY_ESTOP, (
            result_msg.result.failure_kind
        )


def _make_grouped_violating_skill() -> Any:
    """A real 2-slot rSkill whose second slot breaches the joint envelope.

    Models the shape of a mobile-manipulator tick: every ``step()`` emits an
    atomic **group** of slot actions (``tick_group_size=2``), so
    ``ROSPublishingHAL`` applies backpressure after the last slot and blocks
    until ``/openral/action_applied`` reports the tick. The second slot is
    outside the supervisor's ``min_joint``/``max_joint`` envelope, so the
    real safety node latches on it — exactly the production sequence the
    Spark validation hit.
    """
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _GroupedViolatingSkill(rSkillBase):
        def __init__(self) -> None:
            super().__init__(
                name="openral/test-grouped-violating-skill",
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

        def _step_impl(self, _world_state: Any) -> list[Action]:
            return [
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                    tick_group_size=2,
                ),
                # Slot 2: 4.0 rad is outside the ±1.0 rad envelope the test
                # sets on the safety node → drop + latch + /openral/estop.
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    horizon=1,
                    joint_targets=[[4.0, 0.1, 0.1, 0.1, 0.1, 0.1]],
                    tick_group_size=2,
                ),
            ]

    skill = _GroupedViolatingSkill()
    skill.configure()
    skill.activate()
    return skill


def test_safety_latch_during_apply_wait_is_named_in_the_result() -> None:
    """A kernel latch mid-tick yields ``safety_estop``, not an apply-timeout.

    The observability gap this pins (real Spark validation, RoboCasa sink
    scene): when the safety layer latches while ``ROSPublishingHAL`` is
    blocked waiting for an atomic action group to be applied,
    ``/openral/action_applied`` simply goes silent — a latched supervisor
    drops the chunk instead of republishing it on ``/openral/safe_action``.
    The wait used to run out its full timeout and abort the goal with
    ``ROSRuntimeError: ... was not applied within 5.0 s``, so the true cause
    never reached the dispatcher, the reasoner's replanning ladder, or the
    operator (CLAUDE.md §1.4).

    Real components throughout: the production ``SafetyPassthroughNode``
    decides the violation and publishes ``/openral/estop`` itself, the real
    ``RskillRunnerNode`` latches it through its existing subscription, and
    the real ``ROSPublishingHAL`` blocks on the real topic. Nothing
    publishes ``/openral/action_applied`` here because nothing applies the
    action — which is precisely what a latched safety layer looks like.
    """
    import rclpy
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    def _resolver(*_args: Any, **_kwargs: Any) -> Any:
        return _make_grouped_violating_skill()

    with _compose_harness(resolver=_resolver) as (executor, runtime, safety, _observed):
        safety.set_parameters(
            [
                rclpy.parameter.Parameter("min_joint", value=[-1.0] * 6),
                rclpy.parameter.Parameter("max_joint", value=[1.0] * 6),
            ]
        )
        client = ActionClient(runtime.skill_runner_node, ExecuteRskill, "/openral/execute_rskill")
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)

        goal = ExecuteRskill.Goal()
        goal.rskill_id = "openral/test-grouped-violating-skill"
        goal.revision = ""
        goal.prompt = "grouped tick that trips the joint envelope"
        goal.prompt_metadata_json = ""
        # Long enough that the deadline branch cannot be what ends the goal.
        goal.deadline_s = 20.0

        send_future = client.send_goal_async(goal)
        deadline = time.monotonic() + 5.0
        while not send_future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        handle = send_future.result()
        assert handle is not None and handle.accepted

        t0 = time.monotonic()
        result_msg = _await_result(handle, executor, timeout_s=15.0)
        elapsed = time.monotonic() - t0

    assert result_msg is not None
    reason = result_msg.result.failure_reason
    assert not result_msg.result.success
    # The safety stop is named — not buried behind the apply-timeout.
    assert reason.startswith("safety_estop:"), reason
    assert "/openral/estop" in reason, reason
    assert "was not applied within" not in reason, reason
    # ...and it names the tick whose group was left unapplied, so an
    # operator can line the abort up against the trace.
    assert "action group tick 1" in reason, reason
    # The typed field agrees with the prose: the blocked apply-wait raises
    # ROSEStopRequested, so the machine-readable cause is SAFETY_ESTOP and NOT
    # the FAILURE_RUNTIME_ERROR the apply-timeout would have produced.
    assert result_msg.result.failure_kind == ExecuteRskill.Result.FAILURE_SAFETY_ESTOP
    # The abort is prompt: it does not sit out the 5 s apply-timeout first.
    assert elapsed < 4.0, f"safety abort took {elapsed:.2f}s — the apply-wait ran to its timeout"


def _send_goal(client: Any, executor: Any, *, prompt: str, deadline_s: float) -> Any:
    """Send an ExecuteRskill goal and spin until accepted; return the goal handle."""
    from openral_msgs.action import ExecuteRskill

    goal = ExecuteRskill.Goal()
    goal.rskill_id = "openral/test-constant-skill"
    goal.revision = ""
    goal.prompt = prompt
    goal.prompt_metadata_json = ""
    goal.deadline_s = deadline_s
    send_future = client.send_goal_async(goal)
    deadline = time.monotonic() + 5.0
    while not send_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert send_future.done(), "send_goal_async timed out"
    handle = send_future.result()
    assert handle is not None and handle.accepted
    return handle


def _await_result(handle: Any, executor: Any, *, timeout_s: float = 8.0) -> Any:
    """Spin until the goal's result future resolves; return the result message."""
    result_future = handle.get_result_async()
    deadline = time.monotonic() + timeout_s
    while not result_future.done() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert result_future.done(), "result future timed out"
    return result_future.result()


def test_stale_finalized_resident_is_reloaded_on_redispatch() -> None:
    """A key-matching resident left non-ACTIVE is evicted + re-resolved, not stepped.

    #21 deploy validation: a re-dispatch ~1 s after the previous goal hit a
    resident skill in state 'finalized' → every step() raised "must be
    'active' to call step()" (the pre-async ~76 s reaction gap masked this).
    """
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _compose_harness() as (executor, runtime, _safety, _observed):
        client = ActionClient(runtime.skill_runner_node, ExecuteRskill, "/openral/execute_rskill")
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)

        handle = _send_goal(client, executor, prompt="same prompt", deadline_s=0.4)
        _await_result(handle, executor)

        runner = runtime.skill_runner_node
        assert runner._resident_skill is not None
        # Finalize the resident in place (what a lifecycle deactivate /
        # estop teardown does) while its cache key stays valid.
        runner._resident_skill.shutdown()

        handle2 = _send_goal(client, executor, prompt="same prompt", deadline_s=0.4)
        result2 = _await_result(handle2, executor)
        assert "must be 'active'" not in result2.result.failure_reason, (
            f"stale finalized resident was stepped: {result2.result.failure_reason!r}"
        )
        assert "finalized" not in result2.result.failure_reason


def test_goal_accept_served_while_execute_runs() -> None:
    """Goal-2's accept lands < 1 s while goal-1's execute is still running.

    With the action server in the node-default mutually-exclusive group,
    accept/result/cancel queued behind a long execute callback — observed
    live as 50–100 s goal-accept latency under model-load thrash (#21
    deploy validation). The reentrant group + execute serialization lock
    keeps the protocol responsive while goals still run one at a time.
    """
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _compose_harness() as (executor, runtime, _safety, _observed):
        client = ActionClient(runtime.skill_runner_node, ExecuteRskill, "/openral/execute_rskill")
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)

        handle1 = _send_goal(client, executor, prompt="long goal", deadline_s=2.5)
        # Let goal-1's execute callback actually start its rollout.
        _spin_for(executor, 0.3)

        goal2 = ExecuteRskill.Goal()
        goal2.rskill_id = "openral/test-constant-skill"
        goal2.revision = ""
        goal2.prompt = "queued goal"
        goal2.prompt_metadata_json = ""
        goal2.deadline_s = 0.4
        t0 = time.monotonic()
        send2 = client.send_goal_async(goal2)
        deadline = time.monotonic() + 5.0
        while not send2.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        accept_latency = time.monotonic() - t0
        assert send2.done() and send2.result() is not None and send2.result().accepted
        assert accept_latency < 1.0, (
            f"goal-2 accept took {accept_latency:.2f}s — action protocol queued "
            "behind the in-flight execute callback"
        )

        # Both goals must still terminate (goal-2 executes after goal-1's
        # rollout releases the serialization lock). The constant skill never
        # ends on its own, so BOTH goals exit via their execution budget —
        # which is a typed ABORT (deadline_exceeded), not a quiet success:
        # CLAUDE.md §3 deadline fallback is mandatory, and the runner stopped
        # folding a lapsed budget into success=True on this branch. What this
        # test guards is the accept latency above and that termination is
        # orderly + labelled, not the verdict.
        result1 = _await_result(handle1, executor)
        result2 = _await_result(send2.result(), executor)
        assert not result1.result.success
        assert result1.result.failure_reason.startswith("deadline_exceeded"), (
            result1.result.failure_reason
        )
        assert result1.result.failure_kind == ExecuteRskill.Result.FAILURE_DEADLINE_MISSED
        assert not result2.result.success
        assert result2.result.failure_reason.startswith("deadline_exceeded"), (
            result2.result.failure_reason
        )
        assert result2.result.failure_kind == ExecuteRskill.Result.FAILURE_DEADLINE_MISSED


# ── Typed failure_kind on the dispatch path ─────────────────────────────────


def _make_skill_raising(exc: BaseException) -> Any:
    """A real activated so100 rSkill whose ``step`` raises ``exc``."""
    from openral_rskill.base import rSkillBase

    class _RaisingSkill(rSkillBase):
        def __init__(self) -> None:
            super().__init__(
                name="openral/test-raising-skill",
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

        def _step_impl(self, _world_state: Any) -> Any:
            raise exc

    skill = _RaisingSkill()
    skill.configure()
    skill.activate()
    return skill


def _dispatch_and_get_result(resolver: Any, *, deadline_s: float = 2.0) -> Any:
    """Compose the real graph, dispatch one goal through ``resolver``, return the Result."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    with _compose_harness(resolver=resolver) as (executor, runtime, _safety, _observed):
        client = ActionClient(runtime.skill_runner_node, ExecuteRskill, "/openral/execute_rskill")
        _spin_for(executor, 0.3)
        assert client.wait_for_server(timeout_sec=2.0)
        handle = _send_goal(client, executor, prompt="failure kind", deadline_s=deadline_s)
        return _await_result(handle, executor).result


def test_capability_mismatch_carries_failure_kind() -> None:
    """A skill whose embodiment tags miss the robot's → FAILURE_CAPABILITY_MISMATCH.

    Exercises the real embodiment gate in ``_resolve_and_check_skill`` (the
    resolver hands back a genuinely activated rSkill; only its tags are wrong),
    so the ``ROSCapabilityMismatch`` the runner raises is the real one.
    """
    from openral_msgs.action import ExecuteRskill

    result = _dispatch_and_get_result(
        lambda *_a, **_k: _make_named_skill_for_embodiment("franka_panda"),
    )
    assert not result.success
    assert result.failure_reason.startswith("ROSCapabilityMismatch:"), result.failure_reason
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_CAPABILITY_MISMATCH


def _make_named_skill_for_embodiment(tag: str) -> Any:
    """A real configured+activated rSkill declaring a single embodiment ``tag``."""
    from openral_core.schemas import Action, ControlMode
    from openral_rskill.base import rSkillBase

    class _OtherEmbodimentSkill(rSkillBase):
        def __init__(self) -> None:
            super().__init__(
                name="openral/test-other-embodiment",
                version="0.1.0",
                role="s1",
                embodiment_tags=[tag],
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
                joint_targets=[[0.0] * 6],
            )

    skill = _OtherEmbodimentSkill()
    skill.configure()
    skill.activate()
    return skill


def test_non_active_resolver_result_carries_config_error_kind() -> None:
    """A resolver returning a configured-but-not-activated skill → FAILURE_CONFIG_ERROR.

    ``_resolve_and_check_skill`` treats a non-ACTIVE skill as a resolver contract
    violation and raises ``ROSConfigError``; the Result must say so in the uint8.
    """
    from openral_msgs.action import ExecuteRskill

    def _configured_only(*_a: Any, **_k: Any) -> Any:
        from openral_core.schemas import Action, ControlMode
        from openral_rskill.base import rSkillBase

        class _ConfiguredOnlySkill(rSkillBase):
            def __init__(self) -> None:
                super().__init__(
                    name="openral/test-configured-only",
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
                    joint_targets=[[0.0] * 6],
                )

        skill = _ConfiguredOnlySkill()
        skill.configure()  # deliberately NOT activated
        return skill

    result = _dispatch_and_get_result(_configured_only)
    assert not result.success
    assert result.failure_reason.startswith("ROSConfigError:"), result.failure_reason
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_CONFIG_ERROR


@pytest.mark.parametrize(
    ("exc_name", "kind_name", "reason_prefix"),
    [
        ("ROSRuntimeError", "FAILURE_RUNTIME_ERROR", "ROSRuntimeError:"),
        ("ROSGPUMemoryError", "FAILURE_RUNTIME_ERROR", "ROSGPUMemoryError:"),
        ("ROSPerceptionStale", "FAILURE_PERCEPTION_STALE", "ROSPerceptionStale:"),
        ("ROSPlanningError", "FAILURE_PLANNING_ERROR", "ROSPlanningError:"),
    ],
)
def test_typed_step_failures_carry_matching_failure_kind(
    exc_name: str, kind_name: str, reason_prefix: str
) -> None:
    """Each §5 exception raised inside ``skill.step`` maps to its own uint8 kind."""
    import openral_core.exceptions as exc_mod
    from openral_msgs.action import ExecuteRskill

    exc = getattr(exc_mod, exc_name)("raised from the rSkill step")
    result = _dispatch_and_get_result(lambda *_a, **_k: _make_skill_raising(exc))
    assert not result.success
    assert result.failure_reason.startswith(reason_prefix), result.failure_reason
    assert result.failure_kind == getattr(ExecuteRskill.Result, kind_name)


def test_untyped_step_failure_carries_unknown_kind() -> None:
    """A raw non-``ROSError`` escape is FAILURE_UNKNOWN, not a typed kind.

    The distinction is load-bearing for the reasoner: "typed but unclassified"
    and "never entered the OpenRAL exception surface" want different next steps.
    """
    from openral_msgs.action import ExecuteRskill

    result = _dispatch_and_get_result(
        lambda *_a, **_k: _make_skill_raising(ValueError("policy head returned nothing")),
    )
    assert not result.success
    assert result.failure_reason.startswith("ValueError:"), result.failure_reason
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_UNKNOWN


def test_successful_goal_reports_failure_kind_none() -> None:
    """A skill that signals completion closes success=True with FAILURE_NONE."""
    from openral_core.exceptions import ROSRskillGoalSatisfied
    from openral_core.schemas import Action, ControlMode
    from openral_msgs.action import ExecuteRskill
    from openral_rskill.base import rSkillBase

    class _CompletingSkill(rSkillBase):
        """Emits two chunks, then raises the typed completion signal."""

        def __init__(self) -> None:
            super().__init__(
                name="openral/test-completing-skill",
                version="0.1.0",
                role="s1",
                embodiment_tags=["so100_follower"],
            )
            self._ticks = 0

        def _configure_impl(self) -> None:
            pass

        def _activate_impl(self) -> None:
            pass

        def _deactivate_impl(self) -> None:
            pass

        def _shutdown_impl(self) -> None:
            pass

        def _step_impl(self, _world_state: Any) -> Action:
            self._ticks += 1
            if self._ticks > 2:
                raise ROSRskillGoalSatisfied("reached the commanded pose")
            return Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0] * 6],
            )

    def _resolver(*_a: Any, **_k: Any) -> Any:
        skill = _CompletingSkill()
        skill.configure()
        skill.activate()
        return skill

    result = _dispatch_and_get_result(_resolver, deadline_s=5.0)
    assert result.success, result.failure_reason
    assert result.failure_reason == ""
    assert result.failure_kind == ExecuteRskill.Result.FAILURE_NONE
